// forward_ecg.cpp -- coupled cardiac forward-ECG on a CONFORMING unstructured
// tetrahedral P1 FEM mesh (MFEM 4.9 + PETSc 3.24).  Replaces the structured
// 7-point FD "fake-geometry" pipeline (monodomain.c + xsys_precond.c) with a
// real, variationally-consistent FEM coupling of the three cardiac systems on
// the conforming heart-in-torso mesh produced by heart_torso.py:
//
//   Sys1  monodomain Vm on the HEART submesh    (TP06 reaction + IMEX C-N)
//          A1 = (1/dt) M + (1/2) Kdiff,  Kdiff = DiffusionIntegrator(sigma_mono)
//   Sys2  u_e recovery on the HEART (singular, pure Neumann)
//          K_{si+se} u_e = -K_{si} Vm           (ker = span{1})
//   Sys3  torso Laplace on the TORSO region, coupled across the conforming
//          interface; body surface insulated; -> real body-surface ECG.
//
// Two coupling modes:
//   (default) decoupled : solve singular Sys2 on heart, transfer the heart-
//             surface u_e across the conforming interface as a Dirichlet BC
//             into the non-singular torso Sys3.  Keeps the singular Sys2 that
//             the cross-system-preconditioning study (-xsys) targets.
//   -monolithic : one whole-domain elliptic solve on heart u torso with
//             piecewise conductivity (heart si+se, torso so) and source
//             div(si grad Vm) in the heart; potential/flux continuity is
//             natural in P1 on the conforming mesh.  Physically standard
//             forward problem; no interface-transfer error.
//
// Units: mm, ms, mV, mS/mm  (1 S/m == 1 mS/mm, so xsys's S/m values carry over
// numerically unchanged).  chi=140/mm, Cm=0.01 uF/mm^2 => chiCm=1.4.
//
// Build:  make forward_ecg     Run (after `make mesh`):
//   mpirun -n 4 ./forward_ecg -m heart_torso.msh -T 80 -dt 0.02
//   ./forward_ecg -m heart_torso.msh -monolithic          # monolithic forward
//   mpirun -n 4 ./forward_ecg -m heart_torso.msh -xsys     # cross-system study
//
#include "mfem.hpp"
#include <petsc.h>
#include "tt06.h"
#include "mfem_petsc_util.hpp"
#include "sigma_tensor.hpp"
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
    bool   monolithic = false, do_xsys = false, do_precond = false;
    OptionsParser opts(argc, argv);
    opts.AddOption(&mesh_file, "-m", "--mesh", "Gmsh MSH 2.2 conforming mesh.");
    opts.AddOption(&dt, "-dt", "--dt", "Time step (ms).");
    opts.AddOption(&Tend, "-T", "--t-final", "End time (ms).");
    opts.AddOption(&ref_levels, "-refine", "--refine", "Uniform refinements.");
    opts.AddOption(&monolithic, "-monolithic", "--monolithic",
                   "-decoupled", "--decoupled", "Whole-domain vs decoupled coupling.");
    opts.AddOption(&do_xsys, "-xsys", "--xsys", "-noxsys", "--no-xsys",
                   "Run the cross-system preconditioning study on FEM Sys2.");
    opts.AddOption(&do_precond, "-precond", "--precond", "-noprecond", "--no-precond",
                   "ASM vs sASM iteration-count study on the 3 systems (skips EP).");
    opts.Parse();
    if (!opts.Good()) { if (rank==0) opts.PrintUsage(cout); return 1; }
    if (rank==0) opts.PrintOptions(cout);

    MFEMInitializePetsc(&argc, &argv, NULL, NULL);
    PetscOptionsSetValue(NULL, "-options_left", "no");
    // pin each inner solver's PC to ICC via its prefix (survives Customize).
    for (const char *pfx : {"sys1_","sys2_","sys3_","mono_","xsys_"})
    {
        std::string k = std::string("-") + pfx + "pc_type";
        PetscOptionsSetValue(NULL, k.c_str(), "icc");
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

    // ---- mesh + conforming heart/torso submeshes --------------------------
    Mesh serial_mesh(mesh_file, 1, 1);
    MFEM_VERIFY(serial_mesh.Dimension()==3, "expected a 3D mesh");
    for (int l=0;l<ref_levels;++l) serial_mesh.UniformRefinement();
    ParMesh pmesh(MPI_COMM_WORLD, serial_mesh);
    serial_mesh.Clear();
    if (rank==0)
        cout << "[MESH] domain attrs max=" << pmesh.attributes.Max()
             << " bdr attrs max=" << pmesh.bdr_attributes.Max() << "\n";
    MFEM_VERIFY(pmesh.attributes.Max() >= 2,
                "mesh must carry heart(1)/torso(2) domain attributes");

    Array<int> hdom(1); hdom[0] = HEART_ATTR;
    Array<int> tdom(1); tdom[0] = TORSO_ATTR;
    ParSubMesh heart = ParSubMesh::CreateFromDomain(pmesh, hdom);
    ParSubMesh torso = ParSubMesh::CreateFromDomain(pmesh, tdom);

    H1_FECollection fec(1, 3);
    ParFiniteElementSpace fes_h(&heart, &fec);   // heart  (Sys1, Sys2)
    ParFiniteElementSpace fes_t(&torso, &fec);   // torso  (Sys3)
    ParFiniteElementSpace fes_p(&pmesh, &fec);   // parent (monolithic)
    const HYPRE_BigInt ndof_h = fes_h.GlobalTrueVSize();
    const HYPRE_BigInt ndof_t = fes_t.GlobalTrueVSize();
    if (rank==0)
        cout << "[FES] heart dofs=" << ndof_h << "  torso dofs=" << ndof_t
             << "  parent dofs=" << fes_p.GlobalTrueVSize() << "\n";

    // ---- conforming-interface sanity check (do this FIRST) ----------------
    // Partition-independent metric: count interface boundary faces (bdr attr
    // IFACE_BDR).  Boundary elements are uniquely owned, so the global sum is
    // correct on any rank count.  The heart/torso submeshes share exactly this
    // tagged interface (single Gmsh mesh + BooleanFragments => conforming).
    {
        long nfloc = 0;
        for (int be=0; be<pmesh.GetNBE(); ++be)
            if (pmesh.GetBdrAttribute(be)==IFACE_BDR) ++nfloc;
        long nf = 0; MPI_Reduce(&nfloc,&nf,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
        long nfg = 0; MPI_Allreduce(&nfloc,&nfg,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
        if (rank==0)
            cout << "[CONFORM] interface boundary faces (bdr attr "<<IFACE_BDR<<") = "
                 << nf << " (>0 => conforming heart-torso interface present)\n";
        MFEM_VERIFY(nfg > 0, "no interface boundary faces -- mesh missing the "
                             "heart-torso interface; fix heart_torso.py / tags");
        // exact serial cross-check: shared parent vertices must coincide.
        if (Mpi::WorldSize()==1) {
            const Array<int> &hv = heart.GetParentVertexIDMap();
            const Array<int> &tv = torso.GetParentVertexIDMap();
            std::vector<int> hs(hv.begin(),hv.end()), ts(tv.begin(),tv.end());
            std::sort(hs.begin(),hs.end()); std::sort(ts.begin(),ts.end());
            std::vector<int> sh;
            std::set_intersection(hs.begin(),hs.end(),ts.begin(),ts.end(),
                                  std::back_inserter(sh));
            cout << "[CONFORM] (serial) shared parent vertices heart^torso = "
                 << sh.size() << "\n";
            MFEM_VERIFY(sh.size() > 0, "heart/torso share no vertices -- NOT conforming");
        }
    }

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
    { PC pc; KSPGetPC((KSP)cg1, &pc); PCSetType(pc, PCICC); }

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
    AttachConstNullSpace((Mat)Kiep, MPI_COMM_WORLD);          // singular: ker=const
    PetscPCGSolver cg2(Kiep, "sys2_");
    cg2.SetRelTol(1e-8); cg2.SetMaxIter(2000); cg2.iterative_mode = false;
    { PC pc; KSPGetPC((KSP)cg2, &pc); PCSetType(pc, PCICC); }

    // grid functions + transfer maps for the coupling (built once)
    ParGridFunction ue_h(&fes_h);   ue_h = 0.0;   // heart u_e
    ParGridFunction ue_t(&fes_t);   ue_t = 0.0;   // u_e sampled on torso (interface)
    ParGridFunction phi_t(&fes_t);  phi_t = 0.0;  // torso potential (Sys3 sol)
    ParTransferMap heart_to_torso(ue_h, ue_t);    // SubMesh<->SubMesh (shared root)

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
        const PetscInt NSUB = 8;   // ASM subdomains (contiguous blocks of the matrix)
        if (rank==0){
            cout << "\n[PRECOND] CG iterations to rtol=1e-8, sub_pc=ICC(0), "
                 << NSUB << " ASM subdomains (x" << Mpi::WorldSize() << " ranks)\n";
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

    // Vm snapshots for the -xsys study (stored at ECG sample times)
    std::vector<Vector> Vm_seq;

    FILE *fe = (!do_precond && rank==0) ? fopen("fwd_ecg.txt","w") : nullptr;
    if (fe) fprintf(fe,"# t(ms)  ECG(mV, phi_L-phi_R)  Vm@center(mV)\n");

    // ====================================================================
    //  time loop  (IMEX: explicit TP06 reaction + C-N diffusion)
    // ====================================================================
    const int nsteps = do_precond ? 0 : (int)(Tend/dt);   // -precond skips the EP loop
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
            double ecg = 0.0;
            if (!monolithic)
            {
                // Sys2: Kie u_e = -Ki Vm  (singular)
                Vector b2(nloc); Ki->Mult(Vm, b2); b2.Neg();
                RemoveGlobalMean(b2, MPI_COMM_WORLD);
                cg2.Mult(b2, ue_h);                   // u_e on heart
                // transfer heart u_e -> torso (fills shared interface dofs)
                ue_t = 0.0; heart_to_torso.Transfer(ue_h, ue_t);
                // Sys3: torso Laplace with phi = u_e on interface, Neumann body
                phi_t = ue_t;                          // lift carries interface BC
                HypreParMatrix Kt; Vector Xt, Bt;
                ParLinearForm zero_lf(&fes_t); zero_lf=0.0; zero_lf.Assemble();
                ktf.FormLinearSystem(ess_tdofs_t, phi_t, zero_lf, Kt, Xt, Bt);
                PetscParMatrix Ktp; HypreToPetscAIJ(Kt, Ktp, "Sys3_Kt", rank, 1, true);
                PetscPCGSolver cg3(Ktp, "sys3_");
                cg3.SetRelTol(1e-8); cg3.SetMaxIter(1000); cg3.iterative_mode=false;
                { PC pc; KSPGetPC((KSP)cg3,&pc); PCSetType(pc,PCICC); }
                cg3.Mult(Bt, Xt);
                ktf.RecoverFEMSolution(Xt, zero_lf, phi_t);
            }
            else
            {
                // monolithic whole-domain extracellular solve on the parent
                ParGridFunction Vm_par(&fes_p); Vm_par = 0.0;
                ParTransferMap h2p(Vm_gf, Vm_par); h2p.Transfer(Vm_gf, Vm_par);
                SigmaTensor sig_all(HEART_ATTR, TORSO_ATTR, siL+seL, siT+seT, so);
                ParBilinearForm kall(&fes_p);
                kall.AddDomainIntegrator(new DiffusionIntegrator(sig_all));
                kall.Assemble();
                // source: + integral_heart (si grad Vm) . grad v
                MatrixConstantCoefficient si_c(Dsi);
                GradientGridFunctionCoefficient gV(&Vm_par);
                MatrixVectorProductCoefficient q(si_c, gV);
                Array<int> hmark(pmesh.attributes.Max()); hmark=0; hmark[HEART_ATTR-1]=1;
                ParLinearForm src(&fes_p);
                src.AddDomainIntegrator(new DomainLFGradIntegrator(q), hmark);
                src.Assemble();
                // pin ONE global ground dof (remove constant nullspace); only
                // rank 0 contributes a local dof so exactly one point is pinned.
                Array<int> ess_g; if (rank==0) ess_g.Append(0);
                ParGridFunction phi_p(&fes_p); phi_p=0.0;
                HypreParMatrix Ka; Vector Xa, Ba;
                kall.FormLinearSystem(ess_g, phi_p, src, Ka, Xa, Ba);
                PetscParMatrix Kap; HypreToPetscAIJ(Ka, Kap, "Mono_K", rank, 1, true);
                PetscPCGSolver cg0(Kap, "mono_");
                cg0.SetRelTol(1e-8); cg0.SetMaxIter(2000); cg0.iterative_mode=false;
                { PC pc; KSPGetPC((KSP)cg0,&pc); PCSetType(pc,PCICC); }
                cg0.Mult(Ba, Xa);
                kall.RecoverFEMSolution(Xa, src, phi_p);
                // transfer parent phi -> torso for the electrode read
                ParTransferMap p2t(phi_p, phi_t); p2t.Transfer(phi_p, phi_t);
            }
            // ECG = phi(left) - phi(right) at the globally-nearest body dof
            Vector phit_td; phi_t.GetTrueDofs(phit_td);
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
            { PC pc; KSPGetPC((KSP)cg,&pc); PCSetType(pc,PCICC); }
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
