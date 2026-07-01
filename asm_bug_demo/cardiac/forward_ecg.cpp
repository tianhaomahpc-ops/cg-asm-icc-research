// forward_ecg.cpp -- coupled cardiac forward-ECG on a CONFORMING unstructured
// tetrahedral P1 FEM mesh (MFEM 4.9 + PETSc 3.24).  Replaces the structured
// 7-point FD "fake-geometry" pipeline (monodomain.c + xsys_precond.c) with a
// real, variationally-consistent FEM coupling of the three cardiac systems on
// the conforming heart-in-torso mesh produced by heart_torso.py:
//
//   Sys1  monodomain Vm on the HEART mesh       (TP06 reaction + IMEX C-N)
//          A1 = (1/dt) M + (1/2) Kdiff,  Kdiff = DiffusionIntegrator(sigma_mono)
//   Sys2  u_e recovery on the HEART (singular, pure Neumann)
//          K_{si+se} u_e = -K_{si} Vm           (ker = span{1})
//   Sys3  torso Laplace on the TORSO mesh; body surface insulated;
//          interface = Dirichlet from u_e  -> real body-surface ECG.
//
// TWO INDEPENDENT MESHES (no ParSubMesh): heart.msh and torso.msh are written
// from the same Gmsh BooleanFragments mesh, so their interface boundary nodes
// have IDENTICAL coordinates.  Each is read + METIS-partitioned independently;
// the ONLY coupling is an explicit, parallel-safe coordinate-matched interface
// transfer of u_e (InterfaceTransfer) -- partition-independent, unlike the MFEM
// ParSubMesh<->SubMesh ParTransferMap it replaces (which was partition-dependent
// and drove a ~20% body-surface-ECG discrepancy in parallel).
//
// Units: mm, ms, mV, mS/mm  (1 S/m == 1 mS/mm, so xsys's S/m values carry over
// numerically unchanged).  chi=140/mm, Cm=0.01 uF/mm^2 => chiCm=1.4.
//
// Build:  make forward_ecg ; make mesh (-> heart_torso.msh + heart.msh + torso.msh)
//   mpirun -n 4 ./forward_ecg -m heart_torso.msh -T 80 -dt 0.02   # EP + forward ECG
//   mpirun -n 4 ./forward_ecg -m heart_torso.msh -xsys            # cross-system study
//   mpirun -n 4 ./forward_ecg -m heart_torso.msh -precond         # ASM vs sASM
//
#include "mfem.hpp"
#include <petsc.h>
#include "tt06.h"
#include "mfem_petsc_util.hpp"
#include "interface_transfer.hpp"
#include "precond_asm.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>

using namespace mfem;
using namespace std;

static const int HEART_ATTR = 1;   // Gmsh Physical Volume("heart",1)
static const int TORSO_ATTR = 2;   // Gmsh Physical Volume("torso",2)
static const int BODY_BDR   = 1;   // Gmsh Physical Surface("body",1)
static const int IFACE_BDR  = 2;   // Gmsh Physical Surface("interface",2)

// diag conductivity tensor as a MatrixConstantCoefficient (fibers || x)
static DenseMatrix DiagSigma(double sL, double sT)
{
    DenseMatrix D(3); D = 0.0;
    D(0,0) = sL; D(1,1) = sT; D(2,2) = sT;
    return D;
}

int main(int argc, char *argv[])
{
    Mpi::Init(argc, argv);
    Hypre::Init();
    const int rank = Mpi::WorldRank();

    // ---- options -----------------------------------------------------------
    const char *mesh_file = "heart_torso.msh";
    double dt = 0.02, Tend = 80.0;
    int    ref_levels = 0;
    bool   do_xsys = false, do_precond = false, do_dump = false;
    OptionsParser opts(argc, argv);
    opts.AddOption(&mesh_file, "-m", "--mesh", "combined mesh (heart.msh/torso.msh"
                   " are read from the same directory).");
    opts.AddOption(&dt, "-dt", "--dt", "Time step (ms).");
    opts.AddOption(&Tend, "-T", "--t-final", "End time (ms).");
    opts.AddOption(&ref_levels, "-refine", "--refine", "Uniform refinements.");
    opts.AddOption(&do_xsys, "-xsys", "--xsys", "-noxsys", "--no-xsys",
                   "Run the cross-system preconditioning study on FEM Sys2.");
    opts.AddOption(&do_precond, "-precond", "--precond", "-noprecond", "--no-precond",
                   "ASM vs sASM iteration-count study on the 3 systems (skips EP).");
    opts.AddOption(&do_dump, "-dump_fields", "--dump-fields", "-nodump", "--no-dump",
                   "Dump node coords + Vm/u_e/torso-phi snapshots for plotting.");
    bool do_prop = false; int prop_maxit = 4; const char *prop_prefix = "prop";
    opts.AddOption(&do_prop, "-propagation", "--propagation", "-noprop", "--no-prop",
                   "Point-source information-propagation probe on the 3 systems (skips EP).");
    opts.AddOption(&prop_maxit, "-prop_maxit", "--prop-maxit",
                   "Cap CG iterations for the propagation probe (the k-th iterate).");
    opts.AddOption(&prop_prefix, "-prop_prefix", "--prop-prefix",
                   "Output filename prefix for the propagation dumps.");
    bool warmstart = false, no_meanremove = false, do_fischer = false;
    opts.AddOption(&warmstart, "-warmstart", "--warmstart", "-cold", "--cold",
                   "Warm-start Sys2 from the previous time-step u_e (initial guess).");
    opts.AddOption(&do_fischer, "-fischer", "--fischer", "-nofischer", "--no-fischer",
                   "Cross-time accel for the Sys2 EP-loop solve: MFEM-CG (||b||-relative "
                   "tol) + Fischer A-orthonormal projection of the previous u_e history "
                   "as the initial guess.  Reports cold/warm/Fischer iteration counts.");
    opts.AddOption(&no_meanremove, "-no_meanremove", "--no-meanremove",
                   "-meanremove", "--meanremove",
                   "Skip the zero-mean projection on the Sys2 RHS (demo: breaks the anchor).");
    opts.Parse();
    if (!opts.Good()) { if (rank==0) opts.PrintUsage(cout); return 1; }
    if (rank==0) opts.PrintOptions(cout);

    MFEMInitializePetsc(&argc, &argv, NULL, NULL);
    PetscOptionsSetValue(NULL, "-options_left", "no");
    // Inner solver PC: block-Jacobi with ICC on each per-process block.
    // PARALLEL-SAFE: PETSc's PCICC/PCILU are single-process only, so a bare
    // PCICC on an MPIAIJ matrix deadlocks on >1 rank; bjacobi gives one SeqAIJ
    // block per process and ICC works on it.  On 1 rank bjacobi has a single
    // block => identical to plain ICC (counts unchanged).  (GAMG was tried for
    // the elliptic torso but added AMG-setup cost without changing the result,
    // because the parallel inconsistency is in the heart->torso transfer, not
    // the solve -- see the parallel-consistency note below.)
    for (const char *pfx : {"sys1_","sys2_","sys3_","mono_","xsys_"})
    {
        std::string p = std::string("-") + pfx;
        PetscOptionsSetValue(NULL, (p+"pc_type").c_str(),     "bjacobi");
        PetscOptionsSetValue(NULL, (p+"sub_pc_type").c_str(), "icc");
    }

    // Scope block: every MFEM/PETSc object (ParMesh, ParSubMesh, PetscParMatrix,
    // solvers, HypreParMatrix) must be destroyed BEFORE MFEMFinalizePetsc() and
    // MPI_Finalize, or their ~Destroy lands on a freed communicator.  (Same
    // pattern as asm_demo.cpp.)
    {
    // ---- physical parameters (mm, ms, mS/mm) ------------------------------
    const double chi = 140.0, Cm = 0.01, chiCm = chi*Cm;     // 1.4
    const double sLm = 0.1334, sTm = 0.0176;                 // monodomain mS/mm
    const double siL = 0.17,  siT = 0.019;                   // intracellular
    const double seL = 0.62,  seT = 0.236;                   // extracellular
    const double so  = 0.22;                                 // torso (isotropic)
    const double Istim = -80.0, tstim = 2.0;                 // mV/ms, ms
    const double stim_box = 1.5;                             // mm cube at slab corner

    // ---- TWO INDEPENDENT meshes (no ParSubMesh) ---------------------------
    // heart.msh and torso.msh are written from the same Gmsh BooleanFragments
    // mesh, so their interface (bdr attr IFACE_BDR) nodes have identical
    // coordinates.  Each is read + METIS-partitioned independently; the only
    // coupling is an explicit coordinate-matched interface transfer (below).
    std::string mf(mesh_file);
    std::string heart_file = "heart.msh", torso_file = "torso.msh";
    { // allow -m <combined>.msh by deriving the split filenames from its dir
      size_t s = mf.find_last_of('/');
      if (s != std::string::npos)
      { heart_file = mf.substr(0,s+1)+"heart.msh"; torso_file = mf.substr(0,s+1)+"torso.msh"; }
    }
    Mesh hser(heart_file.c_str(),1,1), tser(torso_file.c_str(),1,1);
    for (int l=0;l<ref_levels;++l){ hser.UniformRefinement(); tser.UniformRefinement(); }
    ParMesh heart(MPI_COMM_WORLD, hser);  hser.Clear();
    ParMesh torso(MPI_COMM_WORLD, tser);  tser.Clear();

    H1_FECollection fec(1, 3);
    ParFiniteElementSpace fes_h(&heart, &fec);   // heart  (Sys1, Sys2)
    ParFiniteElementSpace fes_t(&torso, &fec);   // torso  (Sys3)
    const HYPRE_BigInt ndof_h = fes_h.GlobalTrueVSize();  // collective: ALL ranks
    const HYPRE_BigInt ndof_t = fes_t.GlobalTrueVSize();
    if (rank==0)
        cout << "[FES] heart dofs=" << ndof_h << "  torso dofs=" << ndof_t << "\n";

    // explicit, parallel-safe heart<->torso interface coupling (coord match)
    InterfaceTransfer iface(fes_h, IFACE_BDR, fes_t, IFACE_BDR);
    if (rank==0)
        cout << "[CONFORM] interface transfer matched " << iface.Matched()
             << " / " << iface.DstTotal() << " torso interface dofs to heart\n";
    MFEM_VERIFY(iface.Matched()==iface.DstTotal() && iface.DstTotal()>0,
                "interface nodes do not coincide -- heart.msh/torso.msh not conforming");

    // ====================================================================
    //  Sys1 -- monodomain on the heart (constant-in-time operators)
    // ====================================================================
    DenseMatrix Dmono = DiagSigma(sLm/chiCm, sTm/chiCm);   // diffusion = sigma/(chiCm)
    MatrixConstantCoefficient sig_mono(Dmono);
    ConstantCoefficient one(1.0);

    ParBilinearForm mform(&fes_h);
    mform.AddDomainIntegrator(new MassIntegrator(one));
    mform.Assemble(); mform.Finalize();
    HypreParMatrix *M = mform.ParallelAssemble();

    ParBilinearForm kform(&fes_h);
    kform.AddDomainIntegrator(new DiffusionIntegrator(sig_mono));
    kform.Assemble(); kform.Finalize();
    HypreParMatrix *Kd = kform.ParallelAssemble();

    HypreParMatrix *A1h = Add(1.0/dt, *M, 0.5, *Kd);   // (1/dt)M + (1/2)Kdiff
    HypreParMatrix *Bh  = Add(1.0/dt, *M, -0.5, *Kd);  // (1/dt)M - (1/2)Kdiff

    PetscParMatrix A1p;
    HypreToPetscAIJ(*A1h, A1p, "Sys1_A1", rank, 1, true);
        PetscPCGSolver cg1(A1p, "sys1_");
    cg1.SetRelTol(1e-10); cg1.SetMaxIter(500); cg1.iterative_mode = false;
    { PC pc; KSPGetPC((KSP)cg1, &pc); PCSetType(pc, PCBJACOBI); }
    // per-DOF TP06 cells (one per LOCAL true dof on the heart)
    const int nloc = fes_h.GetTrueVSize();
    std::vector<TT06> cell(nloc);
    for (int p=0;p<nloc;++p) tt06_init(&cell[p]);

    ParGridFunction Vm_gf(&fes_h);
    Vector Vm(nloc), rhs(nloc), react(nloc), tmp(nloc);
    for (int p=0;p<nloc;++p) Vm(p) = cell[p].V;        // resting potential

    // true-dof physical coordinates (for stimulus + benchmark probes)
    Vector tdof_x(nloc), tdof_y(nloc), tdof_z(nloc);
    {
        // map each true dof to its vertex coordinate (P1: tdof <-> vertex)
        ParGridFunction cx(&fes_h), cy(&fes_h), cz(&fes_h);
        FunctionCoefficient fx([](const Vector&X){return X[0];});
        FunctionCoefficient fy([](const Vector&X){return X[1];});
        FunctionCoefficient fz([](const Vector&X){return X[2];});
        cx.ProjectCoefficient(fx); cy.ProjectCoefficient(fy); cz.ProjectCoefficient(fz);
        cx.GetTrueDofs(tdof_x); cy.GetTrueDofs(tdof_y); cz.GetTrueDofs(tdof_z);
    }
    // heart slab corner (centered mesh): (-10,-3.5,-1.5); stimulus near that corner
    const double cx0=-10.0, cy0=-3.5, cz0=-1.5;

    // Niederer benchmark points P1..P8 in the CENTERED frame (corner offsets)
    struct BP { const char*name; double x,y,z; };
    BP P[8] = {
        {"P1",-10.0,-3.5,-1.5},{"P2",-10.0, 3.5,-1.5},{"P3",10.0,-3.5,-1.5},{"P4",10.0,3.5,-1.5},
        {"P5",-10.0,-3.5, 1.5},{"P6",-10.0, 3.5, 1.5},{"P7",10.0,-3.5, 1.5},{"P8",10.0,3.5, 1.5}};
    // nearest local true-dof to each benchmark point, then global MINLOC owner
    int Pdof[8]; double Pd2[8];
    for (int q=0;q<8;++q){
        double best=1e300; int bi=-1;
        for (int p=0;p<nloc;++p){
            double d=pow(tdof_x(p)-P[q].x,2)+pow(tdof_y(p)-P[q].y,2)+pow(tdof_z(p)-P[q].z,2);
            if (d<best){best=d;bi=p;}
        }
        Pdof[q]=bi; Pd2[q]=best;
    }
    std::vector<double> tact(nloc, -1.0);

    // ---- forward-ECG infrastructure (built once) --------------------------
    // Sys2 operators on the heart: Ki, Kie  (singular)
    DenseMatrix Dsi  = DiagSigma(siL, siT);
    DenseMatrix Dsie = DiagSigma(siL+seL, siT+seT);
    MatrixConstantCoefficient sig_i(Dsi), sig_ie(Dsie);
    ParBilinearForm kif(&fes_h);  kif.AddDomainIntegrator(new DiffusionIntegrator(sig_i));
    kif.Assemble(); kif.Finalize();  HypreParMatrix *Ki = kif.ParallelAssemble();
    ParBilinearForm kief(&fes_h); kief.AddDomainIntegrator(new DiffusionIntegrator(sig_ie));
    kief.Assemble(); kief.Finalize(); HypreParMatrix *Kie = kief.ParallelAssemble();

    PetscParMatrix Kiep;
    HypreToPetscAIJ(*Kie, Kiep, "Sys2_Kie", rank, 1, true);
    if (!no_meanremove)  // (-no_meanremove also skips the nullspace = the TRUE anchor)
        AttachConstNullSpace((Mat)Kiep, MPI_COMM_WORLD);      // singular: ker=const
        PetscPCGSolver cg2(Kiep, "sys2_");
    cg2.SetRelTol(1e-8); cg2.SetMaxIter(2000);
    // iterative_mode=true => Sys2 uses the previous step's u_e as the initial
    // guess (warm start).  Default cold (x0=0): each solve is independent.
    cg2.iterative_mode = warmstart;
    { PC pc; KSPGetPC((KSP)cg2, &pc); PCSetType(pc, PCBJACOBI); }

    // ---- cross-time acceleration path for the Sys2 EP-loop solve (-fischer) --
    // The EP loop re-solves Kie u_e = -Ki Vm(t) every sample step.  When u_e(t)
    // varies smoothly the previous solutions are an excellent initial-guess basis.
    // Two things must be right for the guess to actually pay off:
    //   (1) the stopping test must measure the TRUE (unpreconditioned) residual
    //       relative to ||b|| -- PETSc's default atol is on the *preconditioned*
    //       residual, which masks how good the initial guess is;
    //   (2) the initial guess must be enabled (KSPSetInitialGuessNonzero).
    // cg2 is a PetscPCGSolver, which already handles the singular pure-Neumann
    // operator robustly via MatSetNullSpace (the constant mode is projected out
    // every iteration -- MFEM's CGSolver does not do this and diverges).  So we
    // keep cg2 and just switch its norm type; Fischer generalises warm start from
    // the last solution to an A-orthonormal span of the whole history.  (Scoped to
    // -fischer so the default EP path keeps its original preconditioned-norm test.)
    if (do_fischer) KSPSetNormType((KSP)cg2, KSP_NORM_UNPRECONDITIONED);
    std::vector<Vector> fisch_P, fisch_AP;   // A-orthonormal history + A*history
    Vector ue_prev(nloc); ue_prev = 0.0;     // previous cold u_e (warm-start seed)
    const int FISCH_MAX = 16;
    long fisch_cold=0, fisch_warm=0, fisch_fis=0;   // cumulative iteration tallies
    auto ip2 = [&](const Vector&x,const Vector&y){ return InnerProduct(MPI_COMM_WORLD,x,y); };

    // grid functions for the coupling (interface transfer `iface` built above)
    ParGridFunction ue_h(&fes_h);   ue_h = 0.0;   // heart u_e
    ParGridFunction phi_t(&fes_t);  phi_t = 0.0;  // torso potential (Sys3 sol)

    // Sys3 torso operator (constant): isotropic Laplace, Dirichlet on interface
    ConstantCoefficient sig_o(so);
    ParBilinearForm ktf(&fes_t);  ktf.AddDomainIntegrator(new DiffusionIntegrator(sig_o));
    ktf.Assemble();
    Array<int> ess_iface(torso.bdr_attributes.Max()); ess_iface = 0;
    if (torso.bdr_attributes.Max() >= IFACE_BDR) ess_iface[IFACE_BDR-1] = 1; // interface Dirichlet
    Array<int> ess_tdofs_t;  fes_t.GetEssentialTrueDofs(ess_iface, ess_tdofs_t);

    // Read a field value at the GLOBALLY-nearest dof to a point: each rank
    // offers (dist^2, local value); MINLOC picks the owner, masked SUM returns
    // its value.  Correct on any rank count (exact on 1 rank).
    auto global_at = [&](double d2, double localval)->double{
        struct { double d; int r; } in{d2, rank}, out;
        MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
        double v = (rank==out.r) ? localval : 0.0, g = 0.0;
        MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return g;
    };
    // torso true-dof coordinates (for the body-surface electrode probes)
    Vector txv, tyv, tzv;
    {
        ParGridFunction gx(&fes_t),gy(&fes_t),gz(&fes_t);
        FunctionCoefficient fx([](const Vector&P){return P[0];});
        FunctionCoefficient fy([](const Vector&P){return P[1];});
        FunctionCoefficient fz([](const Vector&P){return P[2];});
        gx.ProjectCoefficient(fx);gy.ProjectCoefficient(fy);gz.ProjectCoefficient(fz);
        gx.GetTrueDofs(txv);gy.GetTrueDofs(tyv);gz.GetTrueDofs(tzv);
    }
    auto torso_probe = [&](double X,double Y,double Z,int &idx)->double{
        double best=1e300;int bi=-1;
        for (int p=0;p<txv.Size();++p){double d=pow(txv(p)-X,2)+pow(tyv(p)-Y,2)+pow(tzv(p)-Z,2);
            if(d<best){best=d;bi=p;}}
        idx=bi; return best;
    };
    int eL=-1, eR=-1;
    const double eL_d2 = torso_probe(-25, 0, 0, eL);   // left body surface
    const double eR_d2 = torso_probe( 25, 0, 0, eR);   // right body surface

    // node coordinates for plotting (dumped once; serial run)
    // Field dump is PER-RANK: each rank writes its own local true-dofs to a
    // rank-suffixed file (..._r<rank>.txt).  The union over ranks is the full
    // field (each true dof is owned by exactly one rank), and plot_results.py
    // concatenates the rank files.  GetTrueDofs is local, so this is safe and
    // gives the COMPLETE field on any rank count (1 rank => just *_r0.txt).
    auto dump_vec = [&](const char*base, int ms, const Vector &v){
        char fn[80]; snprintf(fn,sizeof fn,"%s_%03d_r%d.txt",base,ms,rank);
        FILE*f=fopen(fn,"w"); for(int p=0;p<v.Size();++p) fprintf(f,"%g\n",v(p)); fclose(f); };
    if (do_dump){
        char fn[80];
        snprintf(fn,sizeof fn,"heart_xyz_r%d.txt",rank);
        { FILE*fh=fopen(fn,"w"); for(int p=0;p<nloc;++p)
            fprintf(fh,"%g %g %g\n",tdof_x(p),tdof_y(p),tdof_z(p)); fclose(fh); }
        snprintf(fn,sizeof fn,"torso_xyz_r%d.txt",rank);
        { FILE*ft=fopen(fn,"w"); for(int p=0;p<txv.Size();++p)
            fprintf(ft,"%g %g %g\n",txv(p),tyv(p),tzv(p)); fclose(ft); }
        if (rank==0) cout << "[DUMP] per-rank node coords written (heart/torso)\n";
    }

    // ====================================================================
    //  -precond : ASM vs sASM iteration counts on the three FEM systems.
    //  Subdomains = MPI ranks, so run with mpirun -n>=2 to see the overlap
    //  effect (on 1 rank there is a single subdomain and sASM==ASM).
    // ====================================================================
    if (do_precond)
    {
        // Sys3 stiffness on the torso (interface Dirichlet, non-singular)
        HypreParMatrix Kt3; ktf.FormSystemMatrix(ess_tdofs_t, Kt3);
        PetscParMatrix Kt3p; HypreToPetscAIJ(Kt3, Kt3p, "Sys3_Kt", rank, 1, true);

        struct Sysp { const char *name; Mat A; bool singular; };
        Sysp S3[3] = {
            {"Sys1 monodomain (heart, SPD, mass-dom)", (Mat)A1p,  false},
            {"Sys2 u_e recover (heart, singular)",     (Mat)Kiep, true },
            {"Sys3 torso Laplace (torso, SPD)",        (Mat)Kt3p, false},
        };
        const PetscInt NSUB = 8;   // serial: contiguous matrix blocks; parallel: 1/rank (METIS)
        const int nsub_eff = (Mpi::WorldSize()==1) ? (int)NSUB : Mpi::WorldSize();
        if (rank==0){
            cout << "\n[PRECOND] CG iters to rtol=1e-8, sub_pc=ICC(0), " << nsub_eff
                 << (Mpi::WorldSize()==1 ? " contiguous-block subdomains (serial)\n"
                                         : " METIS geometric subdomains (1/rank)\n");
            cout << "  system                                    O   ASM(BASIC)  sASM\n";
        }
        for (int q=0;q<3;++q){
            Mat A = S3[q].A;
            Vec xstar, b, x; MatCreateVecs(A, &xstar, &b); VecDuplicate(xstar, &x);
            PetscInt rs, re; MatGetOwnershipRange(A, &rs, &re);
            PetscInt N; MatGetSize(A, &N, NULL);
            { PetscScalar *a; VecGetArray(xstar, &a);
              for (PetscInt i=rs;i<re;++i) a[i-rs]=sin(0.7*(i+1))+0.3*cos(0.11*(i+1));
              VecRestoreArray(xstar, &a); }
            if (S3[q].singular){ PetscScalar s; VecSum(xstar,&s); VecShift(xstar,-s/N); }
            MatMult(A, xstar, b);
            if (S3[q].singular){ PetscScalar s; VecSum(b,&s); VecShift(b,-s/N); }
            for (PetscInt O=0;O<=2;++O){
                int ia = CountIters(A, b, x, false, O, 0, 1e-8, NSUB);
                int is = CountIters(A, b, x, true,  O, 0, 1e-8, NSUB);
                if (rank==0)
                    cout << "  " << std::left << std::setw(40) << (O==0?S3[q].name:"")
                         << " " << O << "   " << std::right << std::setw(8) << ia
                         << "   " << std::setw(6) << is << "\n";
            }
            VecDestroy(&xstar); VecDestroy(&b); VecDestroy(&x);
        }
        if (rank==0) cout << "[PRECOND] (negative = DIVERGED)\n";
    }

    // ====================================================================
    //  -propagation : point-source information-propagation probe.
    //  Put a unit impulse at one node and solve each system capped at
    //  prop_maxit CG iterations; the support of the k-th iterate is how far
    //  information has spread in k iterations.  Sys1/Sys2 source at the heart
    //  centre (0,0,0); Sys3 source in the torso interior (15,0,0).  Dumps
    //  per-rank (x y z value) for plot_propagation.py, on the REAL geometry
    //  and REAL Niederer parameters (dt from -dt).
    // ====================================================================
    if (do_prop)
    {
        auto radius=[&](const Vector&v,const Vector&X,const Vector&Y,const Vector&Z,
                        double sx,double sy,double sz)->double{
            double um=0; for(int p=0;p<v.Size();++p) um=std::max(um,std::fabs(v(p)));
            double gum; MPI_Allreduce(&um,&gum,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
            double tol=1e-3*(gum>0?gum:1.0), rad=0;
            for(int p=0;p<v.Size();++p) if(std::fabs(v(p))>tol){
                double d=sqrt(pow(X(p)-sx,2)+pow(Y(p)-sy,2)+pow(Z(p)-sz,2));
                rad=std::max(rad,d);}
            double g; MPI_Allreduce(&rad,&g,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD); return g;
        };
        auto dump=[&](const char*sys,const Vector&v,const Vector&X,const Vector&Y,const Vector&Z){
            char fn[160]; snprintf(fn,sizeof fn,"%s_%s_r%d.txt",prop_prefix,sys,rank);
            FILE*f=fopen(fn,"w"); for(int p=0;p<v.Size();++p)
                fprintf(f,"%g %g %g %g\n",X(p),Y(p),Z(p),v(p)); fclose(f);
        };
        auto pick=[&](const Vector&X,const Vector&Y,const Vector&Z,
                      double sx,double sy,double sz,int&owner)->int{
            double best=1e300;int bi=-1;
            for(int p=0;p<X.Size();++p){double d=pow(X(p)-sx,2)+pow(Y(p)-sy,2)+pow(Z(p)-sz,2);
                if(d<best){best=d;bi=p;}}
            struct{double d;int r;}in{best,rank},out;
            MPI_Allreduce(&in,&out,1,MPI_DOUBLE_INT,MPI_MINLOC,MPI_COMM_WORLD);
            owner=out.r; return bi;
        };
        // Sys1 (heart, (1/dt)M + 1/2 K -- mass-dominated): impulse at centre
        cg1.SetRelTol(1e-30); cg1.SetAbsTol(1e-30); cg1.SetMaxIter(prop_maxit);
        cg1.iterative_mode=false;
        int ow; int j=pick(tdof_x,tdof_y,tdof_z,0,0,0,ow);
        Vector bs(nloc); bs=0.0; if(rank==ow) bs(j)=1.0;
        Vector x1(nloc); x1=0.0; cg1.Mult(bs,x1);
        double r1=radius(x1,tdof_x,tdof_y,tdof_z,0,0,0); dump("sys1",x1,tdof_x,tdof_y,tdof_z);
        // Sys2 (heart, pure-Neumann singular): same impulse, mean-removed
        cg2.SetRelTol(1e-30); cg2.SetAbsTol(1e-30); cg2.SetMaxIter(prop_maxit);
        cg2.iterative_mode=false;
        Vector b2(bs); RemoveGlobalMean(b2,MPI_COMM_WORLD);
        Vector x2(nloc); x2=0.0; cg2.Mult(b2,x2);
        double r2=radius(x2,tdof_x,tdof_y,tdof_z,0,0,0); dump("sys2",x2,tdof_x,tdof_y,tdof_z);
        // Sys3 (torso Laplace, interface grounded): impulse in torso interior
        HypreParMatrix Kt3; ktf.FormSystemMatrix(ess_tdofs_t,Kt3);
        PetscParMatrix Kt3p; HypreToPetscAIJ(Kt3,Kt3p,"Sys3_Kt",rank,1,true);
        PetscPCGSolver cg3(Kt3p,"sys3_");
        cg3.SetRelTol(1e-30); cg3.SetAbsTol(1e-30); cg3.SetMaxIter(prop_maxit);
        cg3.iterative_mode=false;
        { PC pc; KSPGetPC((KSP)cg3,&pc); PCSetType(pc,PCBJACOBI); }
        int nt=txv.Size(); int ow3; int j3=pick(txv,tyv,tzv,15,0,0,ow3);
        Vector b3(nt); b3=0.0; if(rank==ow3) b3(j3)=1.0;
        Vector x3(nt); x3=0.0; cg3.Mult(b3,x3);
        double r3=radius(x3,txv,tyv,tzv,15,0,0); dump("sys3",x3,txv,tyv,tzv);
        if(rank==0) cout<<"[PROP] dt="<<dt<<" k="<<prop_maxit
            <<"  Sys1 r="<<r1<<"  Sys2 r="<<r2<<"  Sys3 r="<<r3<<"\n";
    }

    // Vm snapshots for the -xsys study (stored at ECG sample times)
    std::vector<Vector> Vm_seq;
    FILE *fe = (!do_precond && !do_prop && rank==0) ? fopen("fwd_ecg.txt","w") : nullptr;
    if (fe) fprintf(fe,"# t(ms)  ECG(mV, phi_L-phi_R)  Vm@center(mV)\n");

    // ====================================================================
    //  time loop  (IMEX: explicit TP06 reaction + C-N diffusion)
    // ====================================================================
    const int nsteps = (do_precond||do_prop) ? 0 : (int)(Tend/dt);   // -precond/-propagation skip the EP loop
    const int sample = std::max(1, (int)(1.0/dt));    // ~ every 1 ms
    for (int s=0;s<nsteps;++s)
    {
        const double t = s*dt;
        // explicit reaction at each heart dof
        for (int p=0;p<nloc;++p){
            cell[p].V = Vm(p);
            double Iion = tt06_react(&cell[p], dt);
            double Is = 0.0;
            if (t<tstim &&
                tdof_x(p) < cx0+stim_box && tdof_y(p) < cy0+stim_box && tdof_z(p) < cz0+stim_box)
                Is = Istim;
            react(p) = -(Iion + Is);                  // mV/ms
        }
        // rhs = B Vm + M*react
        Bh->Mult(Vm, rhs);
        M->Mult(react, tmp);
        rhs += tmp;
        cg1.Mult(rhs, Vm);                            // solve Sys1

        // activation detection (V crosses 0 mV)
        for (int p=0;p<nloc;++p) if (tact[p]<0 && Vm(p)>=0.0) tact[p]=t+dt;

        if (s % sample == 0)
        {
            Vm_gf.SetFromTrueDofs(Vm);
            Vm_seq.push_back(Vm);                     // for -xsys

            // ---- forward solve -> body-surface ECG --------------------
            // Sys2: Kie u_e = -Ki Vm on the heart (singular pure-Neumann); then
            // transfer u_e across the conforming interface to the torso as a
            // Dirichlet BC, and solve Sys3 torso Laplace.  The interface
            // transfer is an explicit coordinate match (InterfaceTransfer) ->
            // PARTITION-INDEPENDENT (bit-identical serial vs parallel).
            double ecg = 0.0;
            {
                Vector b2(nloc); Ki->Mult(Vm, b2); b2.Neg();
                if (!no_meanremove) RemoveGlobalMean(b2, MPI_COMM_WORLD);  // zero-mean anchor
                int it2_cold=-1, it2_warm=-1, it2_fis=-1;
                if (do_fischer) {
                    // Fixed accuracy relative to ||b|| (so the guess quality shows in
                    // the iteration count).  The ACTUAL field used downstream is the
                    // clean cold solve (x0=0), so the ECG is bit-identical to the
                    // non-Fischer path; warm/Fischer are measured on throwaway vectors
                    // -- an imperfect accelerated solve can never corrupt the physics
                    // or poison the A-orthonormal history.
                    double bn = std::sqrt(ip2(b2,b2));
                    cg2.SetRelTol(0.0); cg2.SetAbsTol(1e-8*bn);
                    // NOTE: cg2.iterative_mode does NOT propagate to the KSP after
                    // construction (MFEM sets KSPSetInitialGuessNonzero only in the
                    // ctor); we must toggle the initial-guess flag explicitly, else
                    // the guess is silently ignored and warm==cold.
                    // (a) cold (x0=0) -- THIS is the solution we keep
                    KSPSetInitialGuessNonzero((KSP)cg2, PETSC_FALSE);
                    cg2.Mult(b2, ue_h);
                    it2_cold=cg2.GetNumIterations();
                    // (b) warm start: previous cold u_e as the guess (measure only)
                    KSPSetInitialGuessNonzero((KSP)cg2, PETSC_TRUE);
                    Vector xwarm(ue_prev);
                    cg2.Mult(b2, xwarm); it2_warm=cg2.GetNumIterations();
                    // (c) Fischer: x0 = sum_i <p_i,b> p_i (A-orth projection; measure only)
                    Vector xf(nloc); xf=0.0;
                    for (size_t i=0;i<fisch_P.size();++i) xf.Add(ip2(fisch_P[i],b2), fisch_P[i]);
                    cg2.Mult(b2, xf);
                    it2_fis=cg2.GetNumIterations();
                    KSPSetInitialGuessNonzero((KSP)cg2, PETSC_FALSE);
                    // grow the history from the CLEAN cold solution (mean-zero copy)
                    Vector w(ue_h); RemoveGlobalMean(w, MPI_COMM_WORLD);
                    Vector Aw(nloc); Kie->Mult(w, Aw);
                    for (size_t i=0;i<fisch_P.size();++i){
                        double c=ip2(fisch_AP[i],w); w.Add(-c,fisch_P[i]); Aw.Add(-c,fisch_AP[i]); }
                    double nrm=std::sqrt(ip2(w,Aw));
                    if (nrm>1e-12){
                        w*=1.0/nrm; Aw*=1.0/nrm;
                        // SLIDING WINDOW: evict the oldest so the basis tracks the
                        // current regime.  A frozen (append-only) basis fills with
                        // early-QRS modes and is useless during the plateau -- the
                        // recent history is the good predictor for a drifting u_e.
                        if ((int)fisch_P.size()>=FISCH_MAX){
                            fisch_P.erase(fisch_P.begin()); fisch_AP.erase(fisch_AP.begin()); }
                        fisch_P.push_back(w); fisch_AP.push_back(Aw); }
                    ue_prev = ue_h;                   // warm-start seed for next step
                    fisch_cold+=it2_cold; fisch_warm+=it2_warm; fisch_fis+=it2_fis;
                } else {
                    cg2.Mult(b2, ue_h);               // u_e on heart (PETSc path)
                }
                double ue_mean = ue_h.Sum();
                { double g; MPI_Allreduce(&ue_mean,&g,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
                  ue_mean = g / ndof_h; }             // report the solution's mean
                // heart u_e (true dofs) -> torso interface Dirichlet values
                Vector ue_h_tv;  ue_h.GetTrueDofs(ue_h_tv);
                Vector phi_tv(fes_t.GetTrueVSize()); phi_tv = 0.0;
                iface.Transfer(ue_h_tv, phi_tv);       // sets torso interface entries
                phi_t.SetFromTrueDofs(phi_tv);         // lift carries interface BC
                // Sys3: torso Laplace with phi = u_e on interface, Neumann body
                HypreParMatrix Kt; Vector Xt, Bt;
                ParLinearForm zero_lf(&fes_t); zero_lf=0.0; zero_lf.Assemble();
                ktf.FormLinearSystem(ess_tdofs_t, phi_t, zero_lf, Kt, Xt, Bt);
                PetscParMatrix Ktp; HypreToPetscAIJ(Kt, Ktp, "Sys3_Kt", rank, 1, true);
                PetscPCGSolver cg3(Ktp, "sys3_");
                cg3.SetRelTol(1e-8); cg3.SetMaxIter(3000); cg3.iterative_mode=false;
                { PC pc; KSPGetPC((KSP)cg3,&pc); PCSetType(pc,PCBJACOBI); }
                cg3.Mult(Bt, Xt);
                ktf.RecoverFEMSolution(Xt, zero_lf, phi_t);
                if (rank==0) {
                    cout << "[ITERS] t="<<(int)(t+dt+0.5)<<"ms  Sys1(CG+bj-ICC)="
                         <<cg1.GetNumIterations()<<"  Sys2(singular)=";
                    if (do_fischer)
                        cout << it2_fis<<" (cold="<<it2_cold<<" warm="<<it2_warm<<")";
                    else
                        cout << cg2.GetNumIterations();
                    cout <<"  Sys3(torso)="<<cg3.GetNumIterations()
                         <<"  ue_mean="<<std::scientific<<std::setprecision(2)<<ue_mean
                         <<std::defaultfloat<<"\n";
                }
            }
            // ECG = phi(left) - phi(right) at the globally-nearest body dof
            Vector phit_td; phi_t.GetTrueDofs(phit_td);
            // field snapshots for plotting (~every 12 ms), per-rank
            if (do_dump){
                int ms = (int)(t+dt+0.5);
                if (ms>0 && ms%12==0){
                    dump_vec("heart_vm", ms, Vm);
                    { Vector ueh; ue_h.GetTrueDofs(ueh); dump_vec("heart_ue", ms, ueh); }
                    dump_vec("torso_phi", ms, phit_td);
                    if (rank==0) cout << "[DUMP] snapshot t="<<ms<<" ms\n";
                }
            }
            double pl = global_at(eL_d2, (eL>=0)?phit_td(eL):0.0);
            double pr = global_at(eR_d2, (eR>=0)?phit_td(eR):0.0);
            ecg = pl - pr;
            // Vm at heart center (globally-nearest dof to origin)
            int ci=-1; double cbest=1e300;
            for(int p=0;p<nloc;++p){double d=tdof_x(p)*tdof_x(p)+tdof_y(p)*tdof_y(p)+tdof_z(p)*tdof_z(p);
                if(d<cbest){cbest=d;ci=p;}}
            double vc = global_at(cbest, (ci>=0)?Vm(ci):0.0);
            if (fe) fprintf(fe,"%g %g %g\n", t+dt, ecg, vc);
        }
    }
    if (fe) fclose(fe);

    if (do_fischer && rank==0 && fisch_cold>0) {
        cout << "\n[FISCHER] Sys2 EP-loop cross-time acceleration (total CG iters over the run):\n"
             << "  cold (x0=0)          : " << fisch_cold << "\n"
             << "  warm (prev u_e)      : " << fisch_warm
             << "  (-" << (int)(100.0*(fisch_cold-fisch_warm)/fisch_cold) << "%)\n"
             << "  Fischer (history)    : " << fisch_fis
             << "  (-" << (int)(100.0*(fisch_cold-fisch_fis)/fisch_cold) << "%)\n";
    }

    // ---- benchmark activation times + conduction velocity -----------------
    if (!do_precond) {
    if (rank==0) cout << "\nActivation times (V crosses 0 mV):\n";
    double tP1=-1, tP8=-1;
    for (int q=0;q<8;++q){
        // value at the globally-nearest dof to benchmark point q
        double ta = (Pdof[q]>=0)?tact[Pdof[q]]:-1.0;
        double g  = global_at(Pd2[q], ta);
        if (rank==0)
            cout << "  " << P[q].name << " ("<<P[q].x<<","<<P[q].y<<","<<P[q].z<<") : "
                 << g << " ms\n";
        if (q==0) tP1=g; if (q==7) tP8=g;
    }
    if (rank==0 && tP8>tP1 && tP1>=0){
        double diag = sqrt(20.0*20.0+7.0*7.0+3.0*3.0);
        cout << "P1->P8: dt_act="<<(tP8-tP1)<<" ms over "<<diag
             <<" mm => CV~"<<diag/(tP8-tP1)/1000.0*1000.0<<" m/s "
             <<"(Niederer ~0.6-0.7; P8 -> ~43 ms under refinement)\n";
    }
    }   // end if(!do_precond) benchmark block

    // ====================================================================
    //  -xsys : cross-system preconditioning study on the FEM Sys2
    //  (baseline ICC / warm-start / POD deflation), driven by the REAL
    //  Vm(t) sequence recorded above instead of a synthetic sigmoid front.
    //  The Nicolaides shared-coarse / two-level strategy (Task-3 strategies
    //  4-5) needs a heart-submesh-partition coarse space and is the
    //  documented next step on the unstructured FEM operator.
    // ====================================================================
    if (do_xsys && rank==0) cout << "\n[XSYS] cross-system preconditioning on FEM Sys2"
                                 << " (NT="<<Vm_seq.size()<<" real-Vm RHS):\n";
    if (do_xsys)
    {
        const int NT = (int)Vm_seq.size();
        auto build_b = [&](int t, Vector &b){
            Ki->Mult(Vm_seq[t], b); b.Neg(); RemoveGlobalMean(b, MPI_COMM_WORLD);
        };
        // strategy runner: returns total CG iters over the sequence
        auto run = [&](int strat)->long{
            PetscPCGSolver cg(Kiep, "xsys_", /*iter_mode=*/strat>=1); // warm/POD reuse guess
            cg.SetRelTol(1e-8); cg.SetMaxIter(2000);
            cg.iterative_mode = (strat>=1);
            { PC pc; KSPGetPC((KSP)cg,&pc); PCSetType(pc,PCBJACOBI); }
            Vector u(nloc), uprev(nloc), b(nloc); uprev=0.0;
            std::vector<Vector> Phi; int Kpod=8;
            long tot=0;
            for (int t=0;t<NT;++t){
                build_b(t,b);
                if (strat==0) u=0.0;
                else if (strat==1) u=uprev;
                else { // POD Galerkin initial guess onto span(Phi)
                    u=0.0; int np=(int)Phi.size();
                    if (np>0){
                        DenseMatrix G(np,np); Vector c0(np), a(np);
                        Vector KP(nloc);
                        for(int p=0;p<np;++p){ Kie->Mult(Phi[p],KP);
                            c0(p)=Phi[p]*b;
                            for(int j=0;j<np;++j) G(j,p)=Phi[j]*KP; }
                        DenseMatrixInverse Ginv(G); Ginv.Mult(c0,a);
                        for(int p=0;p<np;++p) u.Add(a(p),Phi[p]);
                    }
                }
                int it0;
                cg.Mult(b,u);
                it0 = cg.GetNumIterations();
                tot += it0;
                uprev = u;
                if (strat==2 && (int)Phi.size()<Kpod){      // grow ON-basis
                    Vector v=u;
                    for (auto &ph: Phi){ double d=v*ph; v.Add(-d,ph); }
                    double nn=v.Norml2();
                    if (nn>1e-10){ v*=1.0/nn; Phi.push_back(v); }
                }
            }
            return tot;
        };
        const char* nm[3]={"baseline (ICC, zero IG)","warm start (prev u_e)","POD deflation (history)"};
        long tots[3];
        for (int st=0;st<3;++st){ tots[st]=run(st);
            if (rank==0) cout << "  "<<std::left<<std::setw(26)<<nm[st]
                              <<" total CG iters = "<<tots[st]
                              <<"  (avg "<<(double)tots[st]/std::max(1,NT)<<")\n"; }
        if (rank==0){
            int best=0; for(int s=1;s<3;++s) if(tots[s]<tots[best]) best=s;
            cout << "[XSYS] best (of baseline/warm/POD): "<<nm[best]
                 <<"  ("<<(double)tots[0]/tots[best]<<"x fewer than baseline)\n";
            cout << "[XSYS] NOTE: the Nicolaides shared-coarse / two-level strategy "
                    "(Task-3 strategies 4-5) needs a mesh-partition coarse space on the\n"
                    "       heart submesh; that carries over conceptually and is the "
                    "documented next step on the unstructured FEM operator.\n";
            FILE *fx=fopen("fwd_xsys.txt","w");
            for(int s=0;s<3;++s) fprintf(fx,"%d %ld\n",s,tots[s]); fclose(fx);
        }
    }

    // ---- cleanup (still inside the scope, before MFEMFinalizePetsc) --------
    delete M; delete Kd; delete A1h; delete Bh; delete Ki; delete Kie;
    if (rank==0) cout << "FWD_ECG_DONE\n";
    }   // <- all stack MFEM/PETSc objects destruct here, while MPI/PETSc alive
    MFEMFinalizePetsc();
    return 0;
}
