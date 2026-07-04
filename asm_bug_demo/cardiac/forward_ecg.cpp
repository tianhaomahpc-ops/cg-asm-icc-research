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

// ---- shared Nicolaides coarse space (Sys1 -> Sys2 structural transfer) -------
// Two-level ADDITIVE preconditioner  M^{-1} = M_fine^{-1} + R0 A0^{+} R0^T.
// The coarse space R0 has ONE column per MPI subdomain: column k is the indicator
// of rank k's true dofs (a partition of unity, sum_k R0 e_k = 1).  It therefore
// spans the constants = the singular pure-Neumann nullspace of Kie = the slow
// global mode (Task-1 result), and is IDENTICAL to the space Sys1 would build on
// the SAME heart mesh -- so it is a genuine cross-system (Sys1->Sys2) transfer,
// built once and reused.  A0 = R0^T Kie R0 is np x np, singular (A0 1 = 0);
// we invert it on the mean-zero subspace (regularise the nullspace direction).
class TwoLevelNicolaides : public Solver
{
    Solver &fine_;                 // fine level: bjacobi+ICC
    const Operator &A_;            // Kie (HypreParMatrix as Operator)
    MPI_Comm comm_; int np_, rk_;
    DenseMatrix A0inv_;            // (A0 + 11^T/np)^{-1}, replicated
    mutable Vector fz_;
public:
    TwoLevelNicolaides(Solver &fine, const Operator &A, MPI_Comm comm,
                       int np, int rk, int nloc)
        : Solver(nloc), fine_(fine), A_(A), comm_(comm), np_(np), rk_(rk), fz_(nloc)
    {
        // A0(:,k) = R0^T (A R0 e_k):  apply A to the indicator of rank k, then take
        // per-rank sums.  np global matvecs, once.
        DenseMatrix A0(np);
        Vector vk(nloc), Av(nloc), col(np);
        for (int k=0;k<np;++k){
            vk = (rk==k) ? 1.0 : 0.0;          // local part of R0 e_k
            A_.Mult(vk, Av);
            double loc = Av.Sum();             // A0[rk][k] contribution
            MPI_Allgather(&loc,1,MPI_DOUBLE, col.GetData(),1,MPI_DOUBLE, comm_);
            for (int j=0;j<np;++j) A0(j,k)=col(j);
        }
        // regularise the constant nullspace direction so A0 is invertible; for a
        // mean-zero rhs the mean-zero part of the solution equals A0^{+} rhs.
        for (int i=0;i<np;++i) for (int j=0;j<np;++j) A0(i,j) += 1.0/np;
        A0inv_ = A0; A0inv_.Invert();
    }
    void SetOperator(const Operator &) override {}
    void Mult(const Vector &r, Vector &z) const override
    {
        fine_.Mult(r, fz_);                    // fine correction
        // restriction c[k] = sum over rank k's dofs of r  (= rank k's local sum)
        double loc = r.Sum(); Vector c(np_), y(np_);
        MPI_Allgather(&loc,1,MPI_DOUBLE, c.GetData(),1,MPI_DOUBLE, comm_);
        double cm=c.Sum()/np_; for(int k=0;k<np_;++k) c(k)-=cm;   // project out const
        A0inv_.Mult(c, y);
        double ym=y.Sum()/np_; for(int k=0;k<np_;++k) y(k)-=ym;
        z = fz_; z += y(rk_);                  // prolong: add y[rk] to all local dofs
    }
};

// ---- SORAS fine level (strong optimized-Schwarz preconditioner for Sys2) -----
// Ported from soras_par.cpp.  M^{-1} = P^T D (K_loc + alpha M_Gamma)^{-1} D P:
//   * K_loc = kief.SpMat() -- rank-local element-assembled NEUMANN block (natural
//     BC on the inter-rank artificial interface, FREE);
//   * M_Gamma -- real interface mass on the SHARED faces (Robin transmission term,
//     carries interface-interface coupling a lumped diagonal cannot);
//   * D = 1/multiplicity (partition of unity);  P = heart prolongation (T<->L).
// This is the STRONG fine level (overlap + optimized Robin transmission) that
// drops Sys2 far below the block-Jacobi+ICC baseline; compose with the shared
// Nicolaides coarse space (TwoLevelNicolaides) for a scalable two-level solver.
static Mat ToSeqAIJ(SparseMatrix &S) {
    S.Finalize();
    const int n = S.Height(); const int *I = S.GetI(), *J = S.GetJ();
    const double *A = S.GetData();
    std::vector<PetscInt> nnz(n); for (int i=0;i<n;++i) nnz[i]=I[i+1]-I[i];
    Mat M; MatCreateSeqAIJ(PETSC_COMM_SELF,n,n,0,nnz.data(),&M);
    MatSetOption(M, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    std::vector<PetscInt> c; std::vector<PetscScalar> v;
    for (int i=0;i<n;++i){ c.clear(); v.clear();
        for (int k=I[i];k<I[i+1];++k){ c.push_back(J[k]); v.push_back(A[k]); }
        PetscInt r=i,m=(PetscInt)c.size();
        if (m) MatSetValues(M,1,&r,m,c.data(),v.data(),INSERT_VALUES); }
    MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY); MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);
    return M;
}
struct SORASPrec : public Solver {
    const Operator *P;        // T -> L prolongation
    Vector dL;                // partition-of-unity weight per L-dof
    mutable Vector rL, yL;    // work (L-space)
    KSP kloc=nullptr; Vec rloc=nullptr, zloc=nullptr; // local Robin solve (COMM_SELF)
    mutable long inner_iters=0, n_applies=0;  // accumulate local-solve iterations
    SORASPrec(int tsize):Solver(tsize){}
    ~SORASPrec(){ if(kloc) KSPDestroy(&kloc); if(rloc) VecDestroy(&rloc); if(zloc) VecDestroy(&zloc); }
    void SetOperator(const Operator&) override {}
    void Mult(const Vector &r, Vector &z) const override {
        P->Mult(r, rL);
        for (int i=0;i<rL.Size();++i) rL(i) *= dL(i);
        PetscScalar *ra; VecGetArray(rloc,&ra);
        for (int i=0;i<rL.Size();++i) ra[i]=rL(i);
        VecRestoreArray(rloc,&ra);
        KSPSolve(kloc, rloc, zloc);
        { PetscInt ni; KSPGetIterationNumber(kloc,&ni); inner_iters+=ni; n_applies++; }
        const PetscScalar *za; VecGetArrayRead(zloc,&za);
        for (int i=0;i<yL.Size();++i) yL(i)=za[i];
        VecRestoreArrayRead(zloc,&za);
        for (int i=0;i<yL.Size();++i) yL(i) *= dL(i);
        P->MultTranspose(yL, z);
    }
};

// diag conductivity tensor as a MatrixConstantCoefficient (fibers || x)
static DenseMatrix DiagSigma(double sL, double sT)
{
    DenseMatrix D(3); D = 0.0;
    D(0,0) = sL; D(1,1) = sT; D(2,2) = sT;
    return D;
}

// ---- generic SORAS-preconditioned CG iteration count -----------------------
// Build the parallel SORAS preconditioner M^{-1}=P^T D (K_loc+alpha M_Gamma)^{-1} D P
// for ANY system (not just Sys2) and count CG iters to solve Aout x = b.
//   * locform : an ASSEMBLED (+Finalized) ParBilinearForm whose SpMat() is the
//               rank-local ELEMENT-assembled NEUMANN block K_loc on (fes,pmesh);
//   * Aout    : the fully (cross-rank) assembled outer operator actually solved;
//   * ess_ld  : local essential (Dirichlet) vdofs to pin on the Robin block too
//               (matches Aout's elimination; empty for pure-Neumann systems);
//   * pu_mode : 0 = multiplicity PU (dL=1/mult), 1 = coefficient/diagonal PU
//               (dL(j)=K_loc_jj / assembled diag -- anisotropy-aware, differs per
//               subdomain because K_loc is UNASSEMBLED).
//   * loc_mode: local Robin solve.  <0 = near-exact CG+ICC(1e-10) (strong fine
//               level, m x cost); 0 = ONE ICC0 apply ("direct ICC", fixed linear
//               operator, ~sASM cost); K>0 = K Chebyshev steps over ICC0.
//   * bext    : the SHARED outer RHS (so sASM and SORAS solve the identical system).
//   * avg_inner (out): average local-solve iterations m per apply (compute driver);
//   * solve_ms  (out): solve-only wall-clock (ms, avg over NREP warm re-solves).
static int SorasPUIters(ParFiniteElementSpace &fes, ParMesh &pmesh,
                        ParBilinearForm &locform, PetscParMatrix &Aout,
                        double alpha, int pu_mode, int loc_mode, bool singular,
                        const Array<int> &ess_ld, Vec bext,
                        double *avg_inner=nullptr, double *solve_ms=nullptr,
                        int icc_level=0)
{
    const int L = fes.GetVSize();
    const int nloc = fes.GetTrueVSize();
    // interface mass M_Gamma on the SHARED (inter-rank) faces (Robin term)
    SparseMatrix MG(L, L); MassIntegrator mi; IsoparametricTransformation FTr;
    const int nsf = pmesh.GetNSharedFaces();
    for (int sf=0; sf<nsf; ++sf) {
        int lf = pmesh.GetSharedFace(sf);
        const FiniteElement *fe = fes.GetFaceElement(lf); if (!fe) continue;
        pmesh.GetFaceTransformation(lf, &FTr);
        DenseMatrix Me; mi.AssembleElementMatrix(*fe, FTr, Me);
        Array<int> vd; fes.GetFaceVDofs(lf, vd);
        if (vd.Size()==Me.Height()) MG.AddSubMatrix(vd, vd, Me);
    }
    MG.Finalize();
    SparseMatrix Krob(locform.SpMat());   // local Neumann block (copy)
    if (alpha != 0.0) Krob.Add(alpha, MG); // + alpha M_Gamma  (Robin transmission)
    for (int k=0;k<ess_ld.Size();++k) Krob.EliminateRowCol(ess_ld[k]); // local Dirichlet
    // alpha==0 => PURE Neumann local block: singular for a diffusion operator
    // (constant nullspace).  Pin ONE local dof (minimal Dirichlet anchor) so the
    // block is invertible and a cheap ICC0 apply is possible -- the direct
    // "Neumann subdomain" the -neumann study compares against ASM's Dirichlet.
    if (alpha == 0.0 && ess_ld.Size() == 0) Krob.EliminateRowCol(0);
    Mat KrobA = ToSeqAIJ(Krob);

    const Operator *Ph = fes.GetProlongationMatrix();
    SORASPrec sp(nloc);
    sp.P = Ph; sp.dL.SetSize(L);
    if (pu_mode == 0) {                                   // multiplicity PU
        Vector onesL(L); onesL=1.0; Vector multT(nloc); Ph->MultTranspose(onesL,multT);
        Vector multL(L); Ph->Mult(multT, multL);
        for(int i=0;i<L;++i) sp.dL(i)=1.0/multL(i);
    } else {                                              // coefficient/diagonal PU
        Vector dloc(L); locform.SpMat().GetDiag(dloc);
        Vector sumT(nloc); Ph->MultTranspose(dloc, sumT);
        Vector sumL(L);    Ph->Mult(sumT, sumL);
        for(int i=0;i<L;++i) sp.dL(i)= (sumL(i)!=0.0) ? dloc(i)/sumL(i) : 1.0;
    }
    sp.rL.SetSize(L); sp.yL.SetSize(L);
    KSPCreate(PETSC_COMM_SELF, &sp.kloc);
    KSPSetOperators(sp.kloc, KrobA, KrobA);
    if (loc_mode == -2) {                        // DIRECT Cholesky (exact, factor once)
        KSPSetType(sp.kloc, KSPPREONLY);
        { PC pc; KSPGetPC(sp.kloc,&pc); PCSetType(pc,PCCHOLESKY); }
    } else if (loc_mode < 0) {                   // near-exact CG+ICC (memory-flat)
        KSPSetType(sp.kloc, KSPCG);
        KSPSetTolerances(sp.kloc, 1e-10, 1e-14, PETSC_DEFAULT, 500);
        KSPSetNormType(sp.kloc, KSP_NORM_UNPRECONDITIONED);
        { PC pc; KSPGetPC(sp.kloc,&pc); PCSetType(pc,PCICC); }
    } else if (loc_mode == 0) {                  // ONE ICC(icc_level) apply
        KSPSetType(sp.kloc, KSPPREONLY);
        { PC pc; KSPGetPC(sp.kloc,&pc); PCSetType(pc,PCICC); PCFactorSetLevels(pc,icc_level); }
    } else {                                     // K Chebyshev steps over ICC0
        KSPSetType(sp.kloc, KSPCHEBYSHEV);
        KSPSetTolerances(sp.kloc,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,loc_mode);
        KSPSetNormType(sp.kloc, KSP_NORM_NONE);
        KSPChebyshevEstEigSet(sp.kloc, 0.0, 0.1, 0.0, 1.1);
        { PC pc; KSPGetPC(sp.kloc,&pc); PCSetType(pc,PCICC); }
    }
    KSPSetErrorIfNotConverged(sp.kloc, PETSC_FALSE);
    MatCreateVecs(KrobA, &sp.rloc, &sp.zloc);

    if (singular) AttachConstNullSpace((Mat)Aout, MPI_COMM_WORLD);
    PetscPCGSolver cg(Aout, "soraspu_");
    cg.SetMaxIter(2000); cg.SetRelTol(1e-8); cg.iterative_mode=false;
    KSPSetNormType((KSP)cg, KSP_NORM_UNPRECONDITIONED);
    cg.SetPreconditioner(sp);
    // copy the SHARED PETSc RHS bext -> mfem true-dof Vector, solve (EP-path style)
    Vector Bv(nloc), Xv(nloc); Xv=0.0;
    { const PetscScalar *ba; VecGetArrayRead(bext,&ba);
      for(int i=0;i<nloc;++i) Bv(i)=ba[i]; VecRestoreArrayRead(bext,&ba); }
    sp.inner_iters=0; sp.n_applies=0;
    cg.Mult(Bv, Xv);
    int it = cg.GetNumIterations();
    if (avg_inner) *avg_inner = sp.n_applies? (double)sp.inner_iters/sp.n_applies : 0.0;
    if (solve_ms) {                       // solve-only wall-clock (warm re-solves)
        const int NREP=10; MPI_Comm comm=MPI_COMM_WORLD;
        MPI_Barrier(comm); double t0=MPI_Wtime();
        for(int r=0;r<NREP;++r){ Xv=0.0; cg.Mult(Bv,Xv); }
        MPI_Barrier(comm); *solve_ms = 1e3*(MPI_Wtime()-t0)/NREP;
    }

    MatDestroy(&KrobA);
    return it;
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
    bool do_sweep = false, do_weightcmp = false;
    opts.AddOption(&do_sweep, "-sweep", "--sweep", "-nosweep", "--no-sweep",
                   "sASM parameter sweep: ICC level L in {0,1,2} x overlap O in {0,1,2} "
                   "on the 3 systems; report iters + solve-only ms, pick the optimum (skips EP).");
    opts.AddOption(&do_weightcmp, "-weightcmp", "--weightcmp", "-noweightcmp", "--no-weightcmp",
                   "sASM weight comparison: multiplicity vs coefficient/diagonal weight on "
                   "the 3 systems, O=0/1/2, L=0 (skips EP).");
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
    bool do_coarse = false, do_soras = false;
    double soras_alpha = 0.2;
    opts.AddOption(&do_coarse, "-coarse", "--coarse", "-nocoarse", "--no-coarse",
                   "Cross-SYSTEM accel: two-level CG with the shared Nicolaides coarse "
                   "space (one column per MPI subdomain; spans Sys2's singular nullspace "
                   "= slow global mode).  Reports baseline vs two-level Sys2 iterations.");
    opts.AddOption(&do_soras, "-soras", "--soras", "-nosoras", "--no-soras",
                   "Strong FINE level for Sys2: parallel SORAS (optimized Schwarz, "
                   "local Neumann block + Robin transmission on shared faces).  Combine "
                   "with -coarse for a two-level SORAS+coarse solver.  Reports baseline "
                   "(bjacobi+ICC) vs SORAS(+coarse) Sys2 iterations.");
    opts.AddOption(&soras_alpha, "-soras_alpha", "--soras-alpha",
                   "Robin parameter alpha for the SORAS transmission term (optimal ~0.2).");
    int soras_local = -1;   // -1 = near-exact CG+ICC(1e-10); 0 = one ICC0 apply; K>0 = K Chebyshev
    opts.AddOption(&soras_local, "-soras_local", "--soras-local",
                   "SORAS local solve: -1 near-exact CG+ICC (strong, expensive/iter); "
                   "0 one ICC0 apply (cheap); K>0 K-step Chebyshev over ICC0 (memory-flat).");
    int soras_pu = 0;   // partition-of-unity weight: 0=multiplicity, 1=coefficient/diagonal
    opts.AddOption(&soras_pu, "-soras_pu", "--soras-pu",
                   "SORAS PU weight: 0=multiplicity (1/mult), 1=coefficient/diagonal "
                   "(local Neumann-block diagonal / assembled diagonal -- anisotropy-aware).");
    bool do_soraspu = false;
    opts.AddOption(&do_soraspu, "-soraspu", "--soraspu", "-nosoraspu", "--no-soraspu",
                   "SORAS PU-weight study on ALL 3 systems (real cardiac params): "
                   "multiplicity PU vs coefficient/diagonal PU iteration counts, strong "
                   "near-exact local solve (skips EP).");
    bool do_fair = false;
    opts.AddOption(&do_fair, "-fair", "--fair", "-nofair", "--no-fair",
                   "FAIR multi-dim comparison, ICC(0/1/2) local ONLY (no Cholesky -- "
                   "realistic large-scale local solve): sASM(overlap 0/1/2) vs cw-SORAS"
                   "(delta=0, opt alpha).  Reports iters across L x O and the per-apply "
                   "compute/comm/memory characteristics (skips EP).");
    bool do_overlap = false;
    opts.AddOption(&do_overlap, "-overlap", "--overlap", "-nooverlap", "--no-overlap",
                   "OVERLAP study: does overlap help a NEAR-EXACT-local additive Schwarz, "
                   "and does it reach zero-overlap Robin cw-SORAS?  Compares (per system, "
                   "near-exact local, optimal alpha) cw-SORAS(delta=0,Robin) vs sASM at "
                   "overlap 0/1/2 (skips EP).");
    bool do_tuned = false;
    opts.AddOption(&do_tuned, "-tuned", "--tuned", "-notuned", "--no-tuned",
                   "TUNED comparison: cw-SORAS at each system's optimal Robin alpha "
                   "(from -transmit) vs sASM(O1,ICC0) -- iterations, avg inner-solve m, "
                   "solve-only ms, and the derived compute & communication ratios "
                   "(skips EP).");
    bool do_neumann = false;
    opts.AddOption(&do_neumann, "-neumann", "--neumann", "-noneumann", "--no-neumann",
                   "DIRECT Neumann-subdomain vs Dirichlet-subdomain (ASM): local block is "
                   "the UNASSEMBLED Neumann block (alpha=0, one dof pinned to remove the "
                   "constant nullspace) vs ASM's assembled Dirichlet block.  Zero overlap, "
                   "ICC0 and near-exact local, iters + solve-ms (skips EP).");
    bool do_transmiti = false;
    opts.AddOption(&do_transmiti, "-transmiti", "--transmiti", "-notransmiti", "--no-transmiti",
                   "TRANSMISSION sensitivity with INEXACT local solve (one ICC0 apply, "
                   "not near-exact): sweep Robin alpha (0~Neumann .. inf~Dirichlet) on the "
                   "3 systems, report iters AND solve-only ms (skips EP).");
    bool do_transmit = false;
    opts.AddOption(&do_transmit, "-transmit", "--transmit", "-notransmit", "--no-transmit",
                   "TRANSMISSION-SENSITIVITY diagnostic: hold subdomains + near-exact "
                   "local solve + coef PU fixed, sweep ONLY the Robin parameter alpha "
                   "(0=Neumann .. inf=Dirichlet) on the 3 systems.  Answers whether the "
                   "iteration count depends on the interface transmission at all before "
                   "choosing how to treat the boundary (skips EP).");
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

    // ---- accelerated Sys2 solver: strong fine level (SORAS) and/or shared -----
    //      Nicolaides coarse space.  Build a SECOND Sys2 solver whose PC is
    //      { bjacobi | SORAS } [ + coarse ]; the baseline cg2 (bjacobi+ICC) still
    //      produces the KEPT field, so the ECG is unchanged.  Same ||b||-rel test.
    PetscPreconditioner *finePC = nullptr;   // bjacobi fine (when no SORAS)
    SORASPrec           *soras  = nullptr;   // SORAS fine
    Mat                  KrobA  = nullptr;   // SORAS local Robin block (owns memory)
    TwoLevelNicolaides  *twolvl = nullptr;
    Solver              *accPC  = nullptr;   // the composed preconditioner
    PetscPCGSolver      *cg2c   = nullptr;
    long coarse_base=0, coarse_two=0;
    const bool do_acc = do_coarse || do_soras;
    std::string acc_label;
    if (do_acc) {
        KSPSetNormType((KSP)cg2, KSP_NORM_UNPRECONDITIONED);   // fair baseline test
        Solver *fine = nullptr;
        if (do_soras) {
            // interface mass M_Gamma on the heart's SHARED faces (Robin term)
            const int L = fes_h.GetVSize();
            SparseMatrix MG(L, L); MassIntegrator mi; IsoparametricTransformation FTr;
            const int nsf = heart.GetNSharedFaces();
            for (int sf=0; sf<nsf; ++sf) {
                int lf = heart.GetSharedFace(sf);
                const FiniteElement *fe = fes_h.GetFaceElement(lf); if (!fe) continue;
                heart.GetFaceTransformation(lf, &FTr);
                DenseMatrix Me; mi.AssembleElementMatrix(*fe, FTr, Me);
                Array<int> vd; fes_h.GetFaceVDofs(lf, vd);
                if (vd.Size()==Me.Height()) MG.AddSubMatrix(vd, vd, Me);
            }
            MG.Finalize();
            SparseMatrix Krob(kief.SpMat());  // local Neumann block (copy)
            Krob.Add(soras_alpha, MG);        // + alpha M_Gamma  (Robin)
            KrobA = ToSeqAIJ(Krob);
            const Operator *Ph = fes_h.GetProlongationMatrix();
            soras = new SORASPrec(nloc);
            soras->P = Ph; soras->dL.SetSize(L);
            if (soras_pu == 0) {
                // multiplicity PU:  dL = 1/mult   (Sum_ranks dL = 1)
                Vector onesL(L); onesL=1.0; Vector multT(nloc); Ph->MultTranspose(onesL,multT);
                Vector multL(L); Ph->Mult(multT, multL);
                for(int i=0;i<L;++i) soras->dL(i)=1.0/multL(i);
            } else {
                // coefficient/diagonal PU:  dL(j) = K_loc_jj / (assembled diag).
                // Unlike PCASM, K_loc = kief.SpMat() is the UNASSEMBLED Neumann block,
                // so its diagonal at a shared dof is THIS subdomain's own (sigma-
                // weighted) partial contribution -> the weight genuinely differs
                // per subdomain.  P^T sums local diagonals = the assembled diagonal.
                Vector dloc(L); kief.SpMat().GetDiag(dloc);          // local Neumann diag
                Vector sumT(nloc); Ph->MultTranspose(dloc, sumT);    // assembled diag (true)
                Vector sumL(L);    Ph->Mult(sumT, sumL);             // broadcast back to L
                for(int i=0;i<L;++i) soras->dL(i)= (sumL(i)!=0.0) ? dloc(i)/sumL(i) : 1.0;
            }
            soras->rL.SetSize(L); soras->yL.SetSize(L);
            KSPCreate(PETSC_COMM_SELF, &soras->kloc);
            KSPSetOperators(soras->kloc, KrobA, KrobA);
            if (soras_local < 0) {                             // near-exact (strong, dear)
                KSPSetType(soras->kloc, KSPCG);
                KSPSetTolerances(soras->kloc, 1e-10, 1e-14, PETSC_DEFAULT, 500);
                KSPSetNormType(soras->kloc, KSP_NORM_UNPRECONDITIONED);
                { PC pc; KSPGetPC(soras->kloc,&pc); PCSetType(pc,PCICC); }
            } else if (soras_local == 0) {                     // one ICC0 apply (cheap)
                KSPSetType(soras->kloc, KSPPREONLY);
                { PC pc; KSPGetPC(soras->kloc,&pc); PCSetType(pc,PCICC); }
            } else {                                           // K Chebyshev over ICC0
                KSPSetType(soras->kloc, KSPCHEBYSHEV);
                KSPSetTolerances(soras->kloc,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,soras_local);
                KSPSetNormType(soras->kloc, KSP_NORM_NONE);
                KSPChebyshevEstEigSet(soras->kloc, 0.0, 0.1, 0.0, 1.1);
                { PC pc; KSPGetPC(soras->kloc,&pc); PCSetType(pc,PCICC); }
            }
            KSPSetErrorIfNotConverged(soras->kloc, PETSC_FALSE);
            MatCreateVecs(KrobA, &soras->rloc, &soras->zloc);
            fine = soras;
        } else {
            finePC = new PetscPreconditioner(Kiep, "sys2fine_");
            { PC pc=(PC)*finePC; PCSetType(pc, PCBJACOBI); }
            fine = finePC;
        }
        if (do_coarse) {
            twolvl = new TwoLevelNicolaides(*fine, *Kie, MPI_COMM_WORLD,
                                            Mpi::WorldSize(), rank, nloc);
            accPC = twolvl;
        } else { accPC = fine; }
        cg2c = new PetscPCGSolver(Kiep, "sys2c_");
        cg2c->SetMaxIter(2000);
        KSPSetNormType((KSP)*cg2c, KSP_NORM_UNPRECONDITIONED);
        cg2c->SetPreconditioner(*accPC);       // wraps the mfem::Solver as a PCShell
        acc_label = std::string(do_soras?(soras_pu?"SORAS(coefPU)":"SORAS(multPU)"):"bjacobi")
                  + (do_coarse?"+coarse":"");
        if (rank==0) cout << "[ACC] Sys2 accelerated PC = " << acc_label
                          << (do_soras?("  (alpha="+std::to_string(soras_alpha)+")"):"")
                          << (do_coarse?("  nc="+std::to_string(Mpi::WorldSize())):"") << "\n";
    }

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
            cout << "  system                                    O  ASM_it sASM_it |"
                    " ASM_tot sASM_tot | ASM_slv sASM_slv  (ms)\n";
        }
        const int NREP = 20;   // repeat each solve for a stable wall-clock average
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
                // TOTAL wall-clock: full CountIters (KSP+PCASM setup + solve), averaged
                MPI_Barrier(MPI_COMM_WORLD); double wa=MPI_Wtime();
                for (int r=0;r<NREP;++r) CountIters(A, b, x, false, O, 0, 1e-8, NSUB);
                MPI_Barrier(MPI_COMM_WORLD); double ta=1e3*(MPI_Wtime()-wa)/NREP;
                MPI_Barrier(MPI_COMM_WORLD); double ws=MPI_Wtime();
                for (int r=0;r<NREP;++r) CountIters(A, b, x, true,  O, 0, 1e-8, NSUB);
                MPI_Barrier(MPI_COMM_WORLD); double ts=1e3*(MPI_Wtime()-ws)/NREP;
                // SOLVE-ONLY wall-clock: setup once, then time KSPSolve only
                double sa = SolveOnlyMs(A, b, x, false, O, 0, 1e-8, NSUB, NREP);
                double ss = SolveOnlyMs(A, b, x, true,  O, 0, 1e-8, NSUB, NREP);
                if (rank==0)
                    cout << "  " << std::left << std::setw(40) << (O==0?S3[q].name:"")
                         << " " << O << " " << std::right << std::setw(6) << ia
                         << "  " << std::setw(6) << is << "  "
                         << std::fixed << std::setprecision(2)
                         << std::setw(7) << ta << "  " << std::setw(7) << ts << "  "
                         << std::setw(7) << sa << "  " << std::setw(7) << ss
                         << std::defaultfloat << "\n";
            }
            VecDestroy(&xstar); VecDestroy(&b); VecDestroy(&x);
        }
        if (rank==0) cout << "[PRECOND] iters to rtol=1e-8; ms avg over " << NREP
                          << ": tot=KSP+PCASM setup+solve, slv=KSPSolve only (setup excluded)"
                          << "; negative iters = DIVERGED\n";
    }

    // ====================================================================
    //  -sweep : sASM parameter sweep, ICC level L={0,1,2} x overlap O={0,1,2}.
    //  For each (L,O) report iterations and SOLVE-ONLY ms; pick the optimum by
    //  solve-only time (the practical "best").  Subdomains = MPI ranks.
    // ====================================================================
    if (do_sweep)
    {
        HypreParMatrix Kt3; ktf.FormSystemMatrix(ess_tdofs_t, Kt3);
        PetscParMatrix Kt3p; HypreToPetscAIJ(Kt3, Kt3p, "Sys3_Kt", rank, 1, true);
        struct Sysp { const char *name; Mat A; bool singular; };
        Sysp S3[3] = {
            {"Sys1 monodomain (heart, mass-dom)", (Mat)A1p,  false},
            {"Sys2 u_e recover (heart, singular)",(Mat)Kiep, true },
            {"Sys3 torso Laplace (torso)",        (Mat)Kt3p, false},
        };
        const PetscInt NSUB = 8;
        const int nsub_eff = (Mpi::WorldSize()==1) ? (int)NSUB : Mpi::WorldSize();
        const int NREP = 20;
        if (rank==0)
            cout << "\n[SWEEP] sASM  L(ICC level) x O(overlap),  " << nsub_eff
                 << " subdomains,  iters | solve-only ms (avg " << NREP << ")\n";
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
            if (rank==0) cout << "  " << S3[q].name << "\n"
                              << "     L\\O |   O=0        O=1        O=2\n";
            double bestms=1e300; int bestL=-1,bestO=-1,bestit=-1;
            for (int L=0;L<=2;++L){
                if (rank==0) cout << "     L=" << L << " |";
                for (int O=0;O<=2;++O){
                    int it = CountIters(A, b, x, true, O, L, 1e-8, NSUB);
                    double ms = SolveOnlyMs(A, b, x, true, O, L, 1e-8, NSUB, NREP);
                    if (ms<bestms){ bestms=ms; bestL=L; bestO=O; bestit=it; }
                    if (rank==0){ char buf[32]; snprintf(buf,sizeof buf,"%4d/%6.2f",it,ms);
                                  cout << "  " << buf; }
                }
                if (rank==0) cout << "\n";
            }
            if (rank==0) cout << "     --> optimum (min solve ms): L=" << bestL
                              << " O=" << bestO << "  (" << bestit << " it, "
                              << std::fixed << std::setprecision(2) << bestms
                              << " ms)" << std::defaultfloat << "\n";
            VecDestroy(&xstar); VecDestroy(&b); VecDestroy(&x);
        }
        if (rank==0) cout << "[SWEEP] cell = iters/solve-ms; optimum chosen by solve-only ms\n";
    }

    // ====================================================================
    //  -weightcmp : sASM multiplicity weight vs coefficient/diagonal weight.
    //  For the ASSEMBLED PCASM blocks the two are expected to COINCIDE (the
    //  principal-submatrix diagonal A_i_jj == A_jj is subdomain-invariant).
    // ====================================================================
    if (do_weightcmp)
    {
        HypreParMatrix Kt3; ktf.FormSystemMatrix(ess_tdofs_t, Kt3);
        PetscParMatrix Kt3p; HypreToPetscAIJ(Kt3, Kt3p, "Sys3_Kt", rank, 1, true);
        struct Sysp { const char *name; Mat A; bool singular; };
        Sysp S3[3] = {
            {"Sys1 monodomain (heart, mass-dom)", (Mat)A1p,  false},
            {"Sys2 u_e recover (heart, singular, aniso)",(Mat)Kiep, true },
            {"Sys3 torso Laplace (torso)",        (Mat)Kt3p, false},
        };
        const PetscInt NSUB = 8;
        const int nsub_eff = (Mpi::WorldSize()==1) ? (int)NSUB : Mpi::WorldSize();
        if (rank==0)
            cout << "\n[WEIGHTCMP] sASM iters (rtol 1e-8, ICC0), " << nsub_eff
                 << " subdomains:  mult weight vs coef/diag weight\n"
                 << "  system                                    O   mult   coef\n";
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
                int im = CountIters(A, b, x, true, O, 0, 1e-8, NSUB, 0);  // multiplicity
                int ic = CountIters(A, b, x, true, O, 0, 1e-8, NSUB, 1);  // coefficient
                if (rank==0)
                    cout << "  " << std::left << std::setw(40) << (O==0?S3[q].name:"")
                         << " " << O << "  " << std::right << std::setw(5) << im
                         << "  " << std::setw(5) << ic << (im==ic?"   (identical)":"") << "\n";
            }
            VecDestroy(&xstar); VecDestroy(&b); VecDestroy(&x);
        }
        if (rank==0) cout << "[WEIGHTCMP] identical => algebraic diag reweight is a no-op for "
                             "assembled PCASM blocks (A_i_jj=A_jj); anisotropy-aware weighting "
                             "needs UNASSEMBLED Neumann blocks (SORAS).\n";
    }

    // ====================================================================
    //  -soraspu : head-to-head on ALL 3 systems (real cardiac params), at EQUAL
    //  local-solve cost (one ICC0 apply each):  sASM ("just ICC" = assembled
    //  PCASM block + 1/mult weight) vs cw-SORAS ("direct-ICC" = UNASSEMBLED
    //  Neumann block + alpha M_Gamma Robin + coef/diag PU).  Same RHS, same
    //  unpreconditioned ||b||-rel test, same #subdomains -> the iteration delta
    //  is purely the Schwarz STRUCTURE, not the local solver.  The near-exact
    //  cw-SORAS column is the fine-level strength upper bound (m x local cost).
    // ====================================================================
    if (do_soraspu)
    {
        const int nsub_eff = Mpi::WorldSize();
        const PetscInt NSUB = 8;   // serial fallback (parallel: 1 subdomain/rank)
        if (rank==0) {
            cout << "\n[SORASPU] sASM vs cw-SORAS, EQUAL local cost (1 ICC0 apply), rtol 1e-8, "
                 << nsub_eff << " subdomains (=ranks), alpha=" << soras_alpha << "\n";
            if (nsub_eff==1)
                cout << "  NOTE: np=1 => 1 subdomain, no shared faces => SORAS Robin term empty; "
                        "run mpirun -np 4/8 for the real comparison.\n";
            cout << "  system                                     sASM(O0) sASM(O1) | "
                    "cwSORAS-1ICC(mult) (coef) | cwSORAS-exact(coef)\n";
        }

        // Sys1: local Neumann block = (1/dt)M_loc + (1/2)K_mono_loc  (SPD, non-singular)
        ConstantCoefficient inv_dt(1.0/dt);
        DenseMatrix DmonoH = DiagSigma(0.5*sLm/chiCm, 0.5*sTm/chiCm);
        MatrixConstantCoefficient sig_monoH(DmonoH);
        ParBilinearForm a1loc(&fes_h);
        a1loc.AddDomainIntegrator(new MassIntegrator(inv_dt));
        a1loc.AddDomainIntegrator(new DiffusionIntegrator(sig_monoH));
        a1loc.Assemble(); a1loc.Finalize();

        // Sys3: local Neumann torso block = K_o_loc (separate form so it keeps the
        // natural BC; interface Dirichlet is imposed locally via ess_ld3 to match Kt3p)
        ParBilinearForm ktloc(&fes_t);
        ktloc.AddDomainIntegrator(new DiffusionIntegrator(sig_o));
        ktloc.Assemble(); ktloc.Finalize();
        HypreParMatrix Kt3; ktf.FormSystemMatrix(ess_tdofs_t, Kt3);
        PetscParMatrix Kt3p; HypreToPetscAIJ(Kt3, Kt3p, "Sys3_Kt", rank, 1, true);
        // torso interface essential LOCAL vdofs (marker: <0 = essential)
        Array<int> ess_vmark, ess_ld3;
        fes_t.GetEssentialVDofs(ess_iface, ess_vmark);
        for (int i=0;i<ess_vmark.Size();++i) if (ess_vmark[i]<0) ess_ld3.Append(i);

        // sASM iter counter: SAME unpreconditioned ||b||-rel test as cw-SORAS above
        auto sasm_iters = [&](Mat A, Vec b, Vec x, int O)->int{
            KSP ksp; KSPCreate(PetscObjectComm((PetscObject)A), &ksp);
            KSPSetType(ksp, KSPCG); KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED);
            KSPSetOperators(ksp, A, A);
            KSPSetTolerances(ksp, 1e-8, 1e-50, PETSC_DEFAULT, 2000);
            InstallScaledASM(ksp, A, O, 0, NSUB, 0);       // sASM, ICC0, multiplicity weight
            VecSet(x,0.0); KSPSolve(ksp,b,x);
            PetscInt it; KSPGetIterationNumber(ksp,&it);
            KSPConvergedReason r; KSPGetConvergedReason(ksp,&r); if(r<0) it=-it;
            KSPDestroy(&ksp); return (int)it;
        };

        Array<int> none;   // no local Dirichlet for the heart systems
        struct Row { const char *name; ParFiniteElementSpace *fes; ParMesh *pm;
                     ParBilinearForm *loc; PetscParMatrix *A; bool sing; Array<int>*ess; };
        Row R[3] = {
            {"Sys1 monodomain (heart, SPD, mass-dom)",   &fes_h,&heart,&a1loc,&A1p, false,&none},
            {"Sys2 u_e recover (heart, singular, aniso)",&fes_h,&heart,&kief, &Kiep, true, &none},
            {"Sys3 torso Laplace (torso, SPD)",          &fes_t,&torso,&ktloc,&Kt3p,false,&ess_ld3},
        };
        for (int q=0;q<3;++q){
            Mat A = (Mat)*R[q].A;
            if (R[q].sing) AttachConstNullSpace(A, MPI_COMM_WORLD);
            Vec xstar, b, x; MatCreateVecs(A, &xstar, &b); VecDuplicate(xstar, &x);
            PetscInt rs, re; MatGetOwnershipRange(A, &rs, &re);
            PetscInt N; MatGetSize(A, &N, NULL);
            { PetscScalar *a; VecGetArray(xstar, &a);
              for (PetscInt i=rs;i<re;++i) a[i-rs]=sin(0.7*(i+1))+0.3*cos(0.11*(i+1));
              VecRestoreArray(xstar, &a); }
            if (R[q].sing){ PetscScalar s; VecSum(xstar,&s); VecShift(xstar,-s/N); }
            MatMult(A, xstar, b);
            if (R[q].sing){ PetscScalar s; VecSum(b,&s); VecShift(b,-s/N); }

            int sa0 = sasm_iters(A, b, x, 0);
            int sa1 = sasm_iters(A, b, x, 1);
            int cm  = SorasPUIters(*R[q].fes,*R[q].pm,*R[q].loc,*R[q].A,soras_alpha,0,0,R[q].sing,*R[q].ess,b);
            int cc  = SorasPUIters(*R[q].fes,*R[q].pm,*R[q].loc,*R[q].A,soras_alpha,1,0,R[q].sing,*R[q].ess,b);
            int ce  = SorasPUIters(*R[q].fes,*R[q].pm,*R[q].loc,*R[q].A,soras_alpha,1,-1,R[q].sing,*R[q].ess,b);
            if (rank==0){
                cout << "  " << std::left << std::setw(42) << R[q].name << std::right
                     << " " << std::setw(6) << sa0 << "   " << std::setw(6) << sa1 << "   | "
                     << std::setw(12) << cm << "  " << std::setw(6) << cc << "  | "
                     << std::setw(12) << ce << "\n";
            }
            VecDestroy(&xstar); VecDestroy(&b); VecDestroy(&x);
        }
        if (rank==0) cout << "[SORASPU] EQUAL local cost columns are sASM(O0/O1) vs cwSORAS-1ICC: "
                             "the delta is pure Schwarz structure (unassembled Neumann + Robin + "
                             "coef PU).  cwSORAS-exact (near-exact local, m x cost) is the fine-"
                             "level strength ceiling, not a same-cost option.\n";
    }

    // ====================================================================
    //  -transmit : TRANSMISSION-SENSITIVITY diagnostic (do this BEFORE deciding
    //  how to treat the interface).  Everything is held fixed -- same non-
    //  overlapping subdomains, same near-exact local solve (CG+ICC 1e-10), same
    //  coef PU -- and ONLY the Robin parameter alpha is swept:
    //     alpha->0   : Neumann transmission
    //     alpha  ~   : optimized Robin
    //     alpha->inf : alpha*M_Gamma pins the interface => Dirichlet transmission
    //  If iters move a lot across alpha, the interface transmission MATTERS and
    //  it is worth optimizing (Robin).  If iters are flat, the transmission is
    //  irrelevant -- the bottleneck is the global/coarse mode, and no boundary
    //  treatment helps; go straight to a coarse space.
    // ====================================================================
    if (do_transmit)
    {
        const double alphas[] = {1e-3, 1e-2, 0.05, 0.2, 1.0, 10.0, 1e3};
        const int NA = sizeof(alphas)/sizeof(alphas[0]);
        if (rank==0){
            cout << "\n[TRANSMIT] iters vs Robin alpha (near-exact local, coef PU, rtol 1e-8), "
                 << Mpi::WorldSize() << " subdomains.  alpha: 0~Neumann .. inf~Dirichlet\n";
            cout << "  system                                    ";
            for (int a=0;a<NA;++a){ char b[16]; snprintf(b,sizeof b,"a=%g",alphas[a]); cout<<std::setw(9)<<b; }
            cout << "   span(max/min)\n";
        }
        // Sys1 / Sys3 local Neumann forms (same as -soraspu)
        ConstantCoefficient inv_dt(1.0/dt);
        DenseMatrix DmonoH = DiagSigma(0.5*sLm/chiCm, 0.5*sTm/chiCm);
        MatrixConstantCoefficient sig_monoH(DmonoH);
        ParBilinearForm a1loc(&fes_h);
        a1loc.AddDomainIntegrator(new MassIntegrator(inv_dt));
        a1loc.AddDomainIntegrator(new DiffusionIntegrator(sig_monoH));
        a1loc.Assemble(); a1loc.Finalize();
        ParBilinearForm ktloc(&fes_t);
        ktloc.AddDomainIntegrator(new DiffusionIntegrator(sig_o));
        ktloc.Assemble(); ktloc.Finalize();
        HypreParMatrix Kt3; ktf.FormSystemMatrix(ess_tdofs_t, Kt3);
        PetscParMatrix Kt3p; HypreToPetscAIJ(Kt3, Kt3p, "Sys3_Kt", rank, 1, true);
        Array<int> ess_vmark, ess_ld3;
        fes_t.GetEssentialVDofs(ess_iface, ess_vmark);
        for (int i=0;i<ess_vmark.Size();++i) if (ess_vmark[i]<0) ess_ld3.Append(i);
        Array<int> none;
        struct Row { const char *name; ParFiniteElementSpace *fes; ParMesh *pm;
                     ParBilinearForm *loc; PetscParMatrix *A; bool sing; Array<int>*ess; };
        Row R[3] = {
            {"Sys1 monodomain (heart, SPD, mass-dom)",   &fes_h,&heart,&a1loc,&A1p, false,&none},
            {"Sys2 u_e recover (heart, singular, aniso)",&fes_h,&heart,&kief, &Kiep, true, &none},
            {"Sys3 torso Laplace (torso, SPD)",          &fes_t,&torso,&ktloc,&Kt3p,false,&ess_ld3},
        };
        for (int q=0;q<3;++q){
            Mat A = (Mat)*R[q].A;
            if (R[q].sing) AttachConstNullSpace(A, MPI_COMM_WORLD);
            Vec xstar, b, x; MatCreateVecs(A, &xstar, &b); VecDuplicate(xstar, &x);
            PetscInt rs, re; MatGetOwnershipRange(A, &rs, &re);
            PetscInt N; MatGetSize(A, &N, NULL);
            { PetscScalar *a; VecGetArray(xstar, &a);
              for (PetscInt i=rs;i<re;++i) a[i-rs]=sin(0.7*(i+1))+0.3*cos(0.11*(i+1));
              VecRestoreArray(xstar, &a); }
            if (R[q].sing){ PetscScalar s; VecSum(xstar,&s); VecShift(xstar,-s/N); }
            MatMult(A, xstar, b);
            if (R[q].sing){ PetscScalar s; VecSum(b,&s); VecShift(b,-s/N); }
            int mn=1<<30, mx=0;
            if (rank==0) cout << "  " << std::left << std::setw(42) << R[q].name << std::right;
            for (int a=0;a<NA;++a){
                int it = SorasPUIters(*R[q].fes,*R[q].pm,*R[q].loc,*R[q].A,alphas[a],1,-1,R[q].sing,*R[q].ess,b);
                if (it>0){ mn=std::min(mn,it); mx=std::max(mx,it); }
                if (rank==0) cout << std::setw(9) << it;
            }
            if (rank==0){ double span = mn>0 ? (double)mx/mn : 0.0;
                cout << "   " << std::fixed << std::setprecision(2) << span << "x"
                     << std::defaultfloat << "\n"; }
            VecDestroy(&xstar); VecDestroy(&b); VecDestroy(&x);
        }
        if (rank==0) cout << "[TRANSMIT] span = max/min over alpha.  span~1 => iters INSENSITIVE "
                             "to interface transmission (bottleneck is the global/coarse mode, "
                             "boundary treatment won't help -> use a coarse space).  span>>1 with "
                             "an interior minimum => transmission MATTERS -> optimize alpha (Robin).\n";
    }

    // ====================================================================
    //  -transmiti : SAME transmission sweep but with an INEXACT local solve
    //  (one ICC0 apply, the realistic large-scale local solver -- not the
    //  near-exact CG+ICC of -transmit).  Reports iters AND solve-only ms, so
    //  the Neumann(alpha->0) vs Dirichlet(alpha->inf) effect is measured on
    //  BOTH iteration count and wall-clock with a cheap subdomain solver.
    // ====================================================================
    if (do_transmiti)
    {
        const double alphas[] = {1e-3, 1e-2, 0.05, 0.2, 1.0, 10.0, 1e3};
        const int NA = sizeof(alphas)/sizeof(alphas[0]);
        if (rank==0){
            cout << "\n[TRANSMITI] INEXACT local (one ICC0 apply), coef PU, rtol 1e-8, "
                 << Mpi::WorldSize() << " subdomains.  alpha: 0~Neumann .. inf~Dirichlet\n"
                 << "  each cell = iters / solve-ms\n";
            cout << "  system                          ";
            for (int a=0;a<NA;++a){ char b[16]; snprintf(b,sizeof b,"a=%g",alphas[a]); cout<<std::setw(13)<<b; }
            cout << "\n";
        }
        ConstantCoefficient inv_dt(1.0/dt);
        DenseMatrix DmonoH = DiagSigma(0.5*sLm/chiCm, 0.5*sTm/chiCm);
        MatrixConstantCoefficient sig_monoH(DmonoH);
        ParBilinearForm a1loc(&fes_h);
        a1loc.AddDomainIntegrator(new MassIntegrator(inv_dt));
        a1loc.AddDomainIntegrator(new DiffusionIntegrator(sig_monoH));
        a1loc.Assemble(); a1loc.Finalize();
        ParBilinearForm ktloc(&fes_t);
        ktloc.AddDomainIntegrator(new DiffusionIntegrator(sig_o));
        ktloc.Assemble(); ktloc.Finalize();
        HypreParMatrix Kt3; ktf.FormSystemMatrix(ess_tdofs_t, Kt3);
        PetscParMatrix Kt3p; HypreToPetscAIJ(Kt3, Kt3p, "Sys3_Kt", rank, 1, true);
        Array<int> ess_vmark, ess_ld3;
        fes_t.GetEssentialVDofs(ess_iface, ess_vmark);
        for (int i=0;i<ess_vmark.Size();++i) if (ess_vmark[i]<0) ess_ld3.Append(i);
        Array<int> none;
        struct Row { const char *name; ParFiniteElementSpace *fes; ParMesh *pm;
                     ParBilinearForm *loc; PetscParMatrix *A; bool sing; Array<int>*ess; };
        Row R[3] = {
            {"Sys1 monodomain (mass-dom)",   &fes_h,&heart,&a1loc,&A1p, false,&none},
            {"Sys2 u_e (singular, aniso)",   &fes_h,&heart,&kief, &Kiep, true, &none},
            {"Sys3 torso Laplace",           &fes_t,&torso,&ktloc,&Kt3p,false,&ess_ld3},
        };
        for (int q=0;q<3;++q){
            Mat A = (Mat)*R[q].A;
            if (R[q].sing) AttachConstNullSpace(A, MPI_COMM_WORLD);
            Vec xstar, b, x; MatCreateVecs(A, &xstar, &b); VecDuplicate(xstar, &x);
            PetscInt rs, re; MatGetOwnershipRange(A, &rs, &re);
            PetscInt N; MatGetSize(A, &N, NULL);
            { PetscScalar *a; VecGetArray(xstar, &a);
              for (PetscInt i=rs;i<re;++i) a[i-rs]=sin(0.7*(i+1))+0.3*cos(0.11*(i+1));
              VecRestoreArray(xstar, &a); }
            if (R[q].sing){ PetscScalar s; VecSum(xstar,&s); VecShift(xstar,-s/N); }
            MatMult(A, xstar, b);
            if (R[q].sing){ PetscScalar s; VecSum(b,&s); VecShift(b,-s/N); }
            if (rank==0) cout << "  " << std::left << std::setw(30) << R[q].name << std::right;
            for (int a=0;a<NA;++a){
                double ms=0;
                int it = SorasPUIters(*R[q].fes,*R[q].pm,*R[q].loc,*R[q].A,alphas[a],1,0,
                                      R[q].sing,*R[q].ess,b,nullptr,&ms,0);   // loc_mode 0 = ICC0
                if (rank==0){ char c[16]; snprintf(c,sizeof c,"%d/%.1f",it,ms);
                              cout << std::setw(13) << c; }
            }
            if (rank==0) cout << "\n";
            VecDestroy(&xstar); VecDestroy(&b); VecDestroy(&x);
        }
        if (rank==0) cout << "[TRANSMITI] inexact (1 ICC0) local: contrast with -transmit "
                             "(near-exact).  Neumann side = small alpha, Dirichlet side = large "
                             "alpha; both iters and wall-clock shown.\n";
    }

    // ====================================================================
    //  -neumann : DIRECT Neumann-subdomain vs Dirichlet-subdomain (ASM).
    //  Dirichlet subdomain = sASM assembled block R_i A R_i^T (interface pinned
    //  => non-singular, cheap).  Neumann subdomain = UNASSEMBLED Neumann block
    //  (alpha=0, natural BC); it is SINGULAR for pure diffusion (constant null-
    //  space), so we pin ONE local dof (minimal Dirichlet anchor) to invert it
    //  with a cheap ICC0 -- see the analysis: pure Neumann is inconsistent/
    //  singular and really wants a coarse space (Neumann-Neumann/FETI).  Zero
    //  overlap both; ICC0 and near-exact local; iters + solve-ms.
    // ====================================================================
    if (do_neumann)
    {
        const PetscInt NSUB = 8;
        if (rank==0){
            cout << "\n[NEUMANN] Neumann-subdomain (alpha=0, 1 dof pinned) vs Dirichlet-subdomain"
                    " (ASM), zero overlap, rtol 1e-8, " << Mpi::WorldSize() << " subdomains\n";
            cout << "  system                        | Dirichlet(ASM) ICC0  exact | Neumann ICC0  exact"
                    "   (iters/solve-ms)\n";
        }
        ConstantCoefficient inv_dt(1.0/dt);
        DenseMatrix DmonoH = DiagSigma(0.5*sLm/chiCm, 0.5*sTm/chiCm);
        MatrixConstantCoefficient sig_monoH(DmonoH);
        ParBilinearForm a1loc(&fes_h);
        a1loc.AddDomainIntegrator(new MassIntegrator(inv_dt));
        a1loc.AddDomainIntegrator(new DiffusionIntegrator(sig_monoH));
        a1loc.Assemble(); a1loc.Finalize();
        ParBilinearForm ktloc(&fes_t);
        ktloc.AddDomainIntegrator(new DiffusionIntegrator(sig_o));
        ktloc.Assemble(); ktloc.Finalize();
        HypreParMatrix Kt3; ktf.FormSystemMatrix(ess_tdofs_t, Kt3);
        PetscParMatrix Kt3p; HypreToPetscAIJ(Kt3, Kt3p, "Sys3_Kt", rank, 1, true);
        Array<int> ess_vmark, ess_ld3;
        fes_t.GetEssentialVDofs(ess_iface, ess_vmark);
        for (int i=0;i<ess_vmark.Size();++i) if (ess_vmark[i]<0) ess_ld3.Append(i);

        // timed sASM (Dirichlet subdomain, zero overlap); local_exact toggles ICC0 vs near-exact
        auto asm_run = [&](Mat A, Vec b, Vec x, bool exact, double *ms)->int{
            KSP ksp; KSPCreate(PetscObjectComm((PetscObject)A), &ksp);
            KSPSetType(ksp, KSPCG); KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED);
            KSPSetOperators(ksp, A, A);
            KSPSetTolerances(ksp, 1e-8, 1e-50, PETSC_DEFAULT, 2000);
            InstallScaledASM(ksp, A, 0, 0, NSUB, 0, exact);
            VecSet(x,0.0); KSPSetUp(ksp); KSPSolve(ksp,b,x);
            PetscInt it; KSPGetIterationNumber(ksp,&it);
            KSPConvergedReason r; KSPGetConvergedReason(ksp,&r); if(r<0) it=-it;
            const int NREP=10; MPI_Barrier(MPI_COMM_WORLD); double t0=MPI_Wtime();
            for(int rr=0;rr<NREP;++rr){ VecSet(x,0.0); KSPSolve(ksp,b,x); }
            MPI_Barrier(MPI_COMM_WORLD); *ms=1e3*(MPI_Wtime()-t0)/NREP;
            KSPDestroy(&ksp); return (int)it;
        };

        Array<int> none;
        struct Row { const char *name; ParFiniteElementSpace *fes; ParMesh *pm;
                     ParBilinearForm *loc; PetscParMatrix *A; bool sing; Array<int>*ess; };
        Row R[3] = {
            {"Sys1 monodomain (mass-dom)",   &fes_h,&heart,&a1loc,&A1p, false,&none},
            {"Sys2 u_e (singular, aniso)",   &fes_h,&heart,&kief, &Kiep, true, &none},
            {"Sys3 torso Laplace",           &fes_t,&torso,&ktloc,&Kt3p,false,&ess_ld3},
        };
        for (int q=0;q<3;++q){
            Mat A = (Mat)*R[q].A;
            if (R[q].sing) AttachConstNullSpace(A, MPI_COMM_WORLD);
            Vec xstar, b, x; MatCreateVecs(A, &xstar, &b); VecDuplicate(xstar, &x);
            PetscInt rs, re; MatGetOwnershipRange(A, &rs, &re);
            PetscInt N; MatGetSize(A, &N, NULL);
            { PetscScalar *a; VecGetArray(xstar, &a);
              for (PetscInt i=rs;i<re;++i) a[i-rs]=sin(0.7*(i+1))+0.3*cos(0.11*(i+1));
              VecRestoreArray(xstar, &a); }
            if (R[q].sing){ PetscScalar s; VecSum(xstar,&s); VecShift(xstar,-s/N); }
            MatMult(A, xstar, b);
            if (R[q].sing){ PetscScalar s; VecSum(b,&s); VecShift(b,-s/N); }

            double dm0=0, dme=0, nm0=0, nme=0;
            int di0 = asm_run(A, b, x, false, &dm0);   // Dirichlet, ICC0
            int die = asm_run(A, b, x, true,  &dme);   // Dirichlet, near-exact
            int ni0 = SorasPUIters(*R[q].fes,*R[q].pm,*R[q].loc,*R[q].A,0.0,1, 0,
                                   R[q].sing,*R[q].ess,b,nullptr,&nm0);  // Neumann(pin), ICC0
            int nie = SorasPUIters(*R[q].fes,*R[q].pm,*R[q].loc,*R[q].A,0.0,1,-1,
                                   R[q].sing,*R[q].ess,b,nullptr,&nme);  // Neumann(pin), near-exact
            if (rank==0){
                cout << "  " << std::left << std::setw(28) << R[q].name << std::right
                     << " | " << std::fixed << std::setprecision(1)
                     << std::setw(4) << di0 << "/" << std::setw(6) << dm0 << "  "
                     << std::setw(4) << die << "/" << std::setw(6) << dme << " | "
                     << std::setw(4) << ni0 << "/" << std::setw(6) << nm0 << "  "
                     << std::setw(4) << nie << "/" << std::setw(6) << nme
                     << std::defaultfloat << "\n";
            }
            VecDestroy(&xstar); VecDestroy(&b); VecDestroy(&x);
        }
        if (rank==0) cout <<
          "[NEUMANN] Dirichlet block = self-contained, non-singular, cheap ICC0.  Neumann block =\n"
          "  singular (pinned here); it under-constrains the constant mode => its real home is a\n"
          "  coarse space (Neumann-Neumann/FETI), not a stand-alone one-level solve.\n";
    }

    // ====================================================================
    //  -tuned : cw-SORAS at each system's OPTIMAL alpha (from -transmit) vs
    //  sASM(O1,ICC0).  Reports, per system:  outer iters (=> #Allreduce, the
    //  communication proxy), avg inner-solve m (=> local-compute driver),
    //  solve-only ms (measured), and the derived compute/comm ratios.
    //    * communication ~ outer iters (each outer CG iter = 2 Allreduce + halo);
    //    * local compute per outer apply:  sASM = 1 ICC0 (~2 nnz);
    //      cw-SORAS = m x (SpMV+ICC) (~m x 4 nnz);
    //      => total compute ratio ~ (it_soras/it_sasm) x 2m.
    // ====================================================================
    if (do_tuned)
    {
        const double a_opt[3] = {0.001, 0.1, 0.01};   // Sys1, Sys2, Sys3 (from -transmit)
        const PetscInt NSUB = 8;
        if (rank==0){
            cout << "\n[TUNED] cw-SORAS(opt alpha, coef PU) vs sASM(O1,ICC0), rtol 1e-8, "
                 << Mpi::WorldSize() << " subdomains.  cw-SORAS local solve at 3 costs: "
                    "1ICC (=sASM local cost), near-exact CG+ICC (m x), direct Cholesky\n";
            cout << "  system                             a*   | sASM      | cwSORAS-1ICC   "
                    "cwSORAS-exact(m)   cwSORAS-Chol   (each: it / solve-ms)\n";
        }
        ConstantCoefficient inv_dt(1.0/dt);
        DenseMatrix DmonoH = DiagSigma(0.5*sLm/chiCm, 0.5*sTm/chiCm);
        MatrixConstantCoefficient sig_monoH(DmonoH);
        ParBilinearForm a1loc(&fes_h);
        a1loc.AddDomainIntegrator(new MassIntegrator(inv_dt));
        a1loc.AddDomainIntegrator(new DiffusionIntegrator(sig_monoH));
        a1loc.Assemble(); a1loc.Finalize();
        ParBilinearForm ktloc(&fes_t);
        ktloc.AddDomainIntegrator(new DiffusionIntegrator(sig_o));
        ktloc.Assemble(); ktloc.Finalize();
        HypreParMatrix Kt3; ktf.FormSystemMatrix(ess_tdofs_t, Kt3);
        PetscParMatrix Kt3p; HypreToPetscAIJ(Kt3, Kt3p, "Sys3_Kt", rank, 1, true);
        Array<int> ess_vmark, ess_ld3;
        fes_t.GetEssentialVDofs(ess_iface, ess_vmark);
        for (int i=0;i<ess_vmark.Size();++i) if (ess_vmark[i]<0) ess_ld3.Append(i);

        // timed sASM (O1, ICC0, mult weight, same unpreconditioned ||b||-rel test)
        auto sasm_run = [&](Mat A, Vec b, Vec x, double *ms)->int{
            KSP ksp; KSPCreate(PetscObjectComm((PetscObject)A), &ksp);
            KSPSetType(ksp, KSPCG); KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED);
            KSPSetOperators(ksp, A, A);
            KSPSetTolerances(ksp, 1e-8, 1e-50, PETSC_DEFAULT, 2000);
            InstallScaledASM(ksp, A, 1, 0, NSUB, 0);
            VecSet(x,0.0); KSPSetUp(ksp); KSPSolve(ksp,b,x);       // warm
            PetscInt it; KSPGetIterationNumber(ksp,&it);
            KSPConvergedReason r; KSPGetConvergedReason(ksp,&r); if(r<0) it=-it;
            const int NREP=10; MPI_Barrier(MPI_COMM_WORLD); double t0=MPI_Wtime();
            for(int rr=0;rr<NREP;++rr){ VecSet(x,0.0); KSPSolve(ksp,b,x); }
            MPI_Barrier(MPI_COMM_WORLD); *ms=1e3*(MPI_Wtime()-t0)/NREP;
            KSPDestroy(&ksp); return (int)it;
        };

        Array<int> none;
        struct Row { const char *name; ParFiniteElementSpace *fes; ParMesh *pm;
                     ParBilinearForm *loc; PetscParMatrix *A; bool sing; Array<int>*ess; };
        Row R[3] = {
            {"Sys1 monodomain (heart, SPD, mass-dom)",   &fes_h,&heart,&a1loc,&A1p, false,&none},
            {"Sys2 u_e recover (heart, singular, aniso)",&fes_h,&heart,&kief, &Kiep, true, &none},
            {"Sys3 torso Laplace (torso, SPD)",          &fes_t,&torso,&ktloc,&Kt3p,false,&ess_ld3},
        };
        for (int q=0;q<3;++q){
            Mat A = (Mat)*R[q].A;
            if (R[q].sing) AttachConstNullSpace(A, MPI_COMM_WORLD);
            Vec xstar, b, x; MatCreateVecs(A, &xstar, &b); VecDuplicate(xstar, &x);
            PetscInt rs, re; MatGetOwnershipRange(A, &rs, &re);
            PetscInt N; MatGetSize(A, &N, NULL);
            { PetscScalar *a; VecGetArray(xstar, &a);
              for (PetscInt i=rs;i<re;++i) a[i-rs]=sin(0.7*(i+1))+0.3*cos(0.11*(i+1));
              VecRestoreArray(xstar, &a); }
            if (R[q].sing){ PetscScalar s; VecSum(xstar,&s); VecShift(xstar,-s/N); }
            MatMult(A, xstar, b);
            if (R[q].sing){ PetscScalar s; VecSum(b,&s); VecShift(b,-s/N); }

            double sms=0, oms=0, cms=0, hms=0, mo=0, m=0, mh=0;
            int sit = sasm_run(A, b, x, &sms);
            int oit = SorasPUIters(*R[q].fes,*R[q].pm,*R[q].loc,*R[q].A,a_opt[q],1, 0,
                                   R[q].sing,*R[q].ess,b,&mo,&oms);     // ONE ICC0 apply (=sASM cost)
            int cit = SorasPUIters(*R[q].fes,*R[q].pm,*R[q].loc,*R[q].A,a_opt[q],1,-1,
                                   R[q].sing,*R[q].ess,b,&m,&cms);      // near-exact CG+ICC local
            int hit = SorasPUIters(*R[q].fes,*R[q].pm,*R[q].loc,*R[q].A,a_opt[q],1,-2,
                                   R[q].sing,*R[q].ess,b,&mh,&hms);     // direct Cholesky local
            if (rank==0){
                char eb[24]; snprintf(eb,sizeof eb,"%d/%.1f(m%.0f)", cit, cms, m);
                cout << "  " << std::left << std::setw(34) << R[q].name << std::right
                     << " " << std::setw(6) << a_opt[q] << " | "
                     << std::fixed << std::setprecision(1)
                     << std::setw(4) << sit << "/" << std::setw(6) << sms << " | "
                     << std::setw(4) << oit << "/" << std::setw(6) << oms << "   "
                     << std::setw(15) << eb << "   "
                     << std::setw(4) << hit << "/" << std::setw(6) << hms
                     << std::defaultfloat << "\n";
            }
            VecDestroy(&xstar); VecDestroy(&b); VecDestroy(&x);
        }
        if (rank==0) cout << "[TUNED] cwSORAS-1ICC is the EQUAL-local-cost comparison to sASM "
                             "(both one ICC0 apply): if its it/ms don't beat sASM, the SORAS "
                             "structure alone (at equal cost) isn't enough -- the win needs the "
                             "expensive near-exact/Cholesky local.  Comm ~ outer-iter count "
                             "(Allreduce, the >>P<< bottleneck); ms is THIS-node solve-only.\n";
    }

    // ====================================================================
    //  -overlap : does OVERLAP help?  cw-SORAS uses ZERO overlap (the Robin
    //  transmission substitutes for it).  Here, at a NEAR-EXACT local solve
    //  (equal local-accuracy footing), sweep sASM overlap 0/1/2 and compare
    //  to zero-overlap Robin cw-SORAS.  A truly faithful OVERLAPPING-Robin
    //  (Neumann patch + moved Robin interface) needs element reassembly over
    //  ghost layers and is not built; but if overlap barely moves near-exact
    //  sASM, adding it to SORAS won't help either (both attack only the fine
    //  level, not the global mode).
    // ====================================================================
    if (do_overlap)
    {
        const double a_opt[3] = {0.001, 0.1, 0.01};
        const PetscInt NSUB = 8;
        if (rank==0){
            cout << "\n[OVERLAP] near-exact local everywhere, rtol 1e-8, " << Mpi::WorldSize()
                 << " subdomains.  cw-SORAS(delta=0, Robin) vs sASM(overlap 0/1/2)\n";
            cout << "  system                             a*   | cwSORAS(d0)  sASM(O0) sASM(O1) "
                    "sASM(O2)  | overlap gain O0->O2\n";
        }
        ConstantCoefficient inv_dt(1.0/dt);
        DenseMatrix DmonoH = DiagSigma(0.5*sLm/chiCm, 0.5*sTm/chiCm);
        MatrixConstantCoefficient sig_monoH(DmonoH);
        ParBilinearForm a1loc(&fes_h);
        a1loc.AddDomainIntegrator(new MassIntegrator(inv_dt));
        a1loc.AddDomainIntegrator(new DiffusionIntegrator(sig_monoH));
        a1loc.Assemble(); a1loc.Finalize();
        ParBilinearForm ktloc(&fes_t);
        ktloc.AddDomainIntegrator(new DiffusionIntegrator(sig_o));
        ktloc.Assemble(); ktloc.Finalize();
        HypreParMatrix Kt3; ktf.FormSystemMatrix(ess_tdofs_t, Kt3);
        PetscParMatrix Kt3p; HypreToPetscAIJ(Kt3, Kt3p, "Sys3_Kt", rank, 1, true);
        Array<int> ess_vmark, ess_ld3;
        fes_t.GetEssentialVDofs(ess_iface, ess_vmark);
        for (int i=0;i<ess_vmark.Size();++i) if (ess_vmark[i]<0) ess_ld3.Append(i);

        // near-exact sASM at a given overlap (assembled block, Dirichlet transmission)
        auto sasm_exact = [&](Mat A, Vec b, Vec x, int O)->int{
            KSP ksp; KSPCreate(PetscObjectComm((PetscObject)A), &ksp);
            KSPSetType(ksp, KSPCG); KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED);
            KSPSetOperators(ksp, A, A);
            KSPSetTolerances(ksp, 1e-8, 1e-50, PETSC_DEFAULT, 2000);
            InstallScaledASM(ksp, A, O, 0, NSUB, 0, /*local_exact=*/true);
            VecSet(x,0.0); KSPSolve(ksp,b,x);
            PetscInt it; KSPGetIterationNumber(ksp,&it);
            KSPConvergedReason r; KSPGetConvergedReason(ksp,&r); if(r<0) it=-it;
            KSPDestroy(&ksp); return (int)it;
        };

        Array<int> none;
        struct Row { const char *name; ParFiniteElementSpace *fes; ParMesh *pm;
                     ParBilinearForm *loc; PetscParMatrix *A; bool sing; Array<int>*ess; };
        Row R[3] = {
            {"Sys1 monodomain (heart, SPD, mass-dom)",   &fes_h,&heart,&a1loc,&A1p, false,&none},
            {"Sys2 u_e recover (heart, singular, aniso)",&fes_h,&heart,&kief, &Kiep, true, &none},
            {"Sys3 torso Laplace (torso, SPD)",          &fes_t,&torso,&ktloc,&Kt3p,false,&ess_ld3},
        };
        for (int q=0;q<3;++q){
            Mat A = (Mat)*R[q].A;
            if (R[q].sing) AttachConstNullSpace(A, MPI_COMM_WORLD);
            Vec xstar, b, x; MatCreateVecs(A, &xstar, &b); VecDuplicate(xstar, &x);
            PetscInt rs, re; MatGetOwnershipRange(A, &rs, &re);
            PetscInt N; MatGetSize(A, &N, NULL);
            { PetscScalar *a; VecGetArray(xstar, &a);
              for (PetscInt i=rs;i<re;++i) a[i-rs]=sin(0.7*(i+1))+0.3*cos(0.11*(i+1));
              VecRestoreArray(xstar, &a); }
            if (R[q].sing){ PetscScalar s; VecSum(xstar,&s); VecShift(xstar,-s/N); }
            MatMult(A, xstar, b);
            if (R[q].sing){ PetscScalar s; VecSum(b,&s); VecShift(b,-s/N); }

            int cs = SorasPUIters(*R[q].fes,*R[q].pm,*R[q].loc,*R[q].A,a_opt[q],1,-1,
                                  R[q].sing,*R[q].ess,b);           // Robin cw-SORAS, delta=0
            int o0 = sasm_exact(A, b, x, 0);
            int o1 = sasm_exact(A, b, x, 1);
            int o2 = sasm_exact(A, b, x, 2);
            if (rank==0){
                double gain = o0>0? (double)o2/o0 : 0.0;
                cout << "  " << std::left << std::setw(34) << R[q].name << std::right
                     << " " << std::setw(6) << a_opt[q] << " | "
                     << std::setw(9) << cs << "  " << std::setw(7) << o0 << "  "
                     << std::setw(7) << o1 << "  " << std::setw(7) << o2 << "   | "
                     << std::fixed << std::setprecision(2) << gain << "x"
                     << std::defaultfloat << "\n";
            }
            VecDestroy(&xstar); VecDestroy(&b); VecDestroy(&x);
        }
        if (rank==0) cout << "[OVERLAP] near-exact local isolates the coupling mechanism: Robin"
                             "(delta=0) vs Dirichlet+overlap.  If sASM O0->O2 barely drops, overlap "
                             "is a weak lever even at strong local solve => adding it to SORAS won't "
                             "help; if sASM(O2) beats/ties cwSORAS(d0), overlap alone matches Robin "
                             "and combining is redundant.  Neither removes the global-mode floor.\n";
    }

    // ====================================================================
    //  -fair : FAIR multi-dimensional comparison with ICC(0/1/2) local ONLY
    //  (no Cholesky/near-exact -- the realistic large-scale local solve: one
    //  memory-flat ICC(L) apply, a fixed linear op so CG stays valid).
    //  sASM(overlap O, ICC L, mult PU) vs cw-SORAS(delta=0, ICC L, coef PU,
    //  optimal alpha).  Prints the iteration grid L x O, then a per-apply
    //  cost annotation (compute, communication, memory, construction).
    // ====================================================================
    if (do_fair)
    {
        const double a_opt[3] = {0.001, 0.1, 0.01};
        const PetscInt NSUB = 8;
        if (rank==0)
            cout << "\n[FAIR] ICC(L) local only, rtol 1e-8, " << Mpi::WorldSize()
                 << " subdomains.  iters:  sASM(overlap O) | cw-SORAS(delta=0)   [L=ICC level]\n";
        ConstantCoefficient inv_dt(1.0/dt);
        DenseMatrix DmonoH = DiagSigma(0.5*sLm/chiCm, 0.5*sTm/chiCm);
        MatrixConstantCoefficient sig_monoH(DmonoH);
        ParBilinearForm a1loc(&fes_h);
        a1loc.AddDomainIntegrator(new MassIntegrator(inv_dt));
        a1loc.AddDomainIntegrator(new DiffusionIntegrator(sig_monoH));
        a1loc.Assemble(); a1loc.Finalize();
        ParBilinearForm ktloc(&fes_t);
        ktloc.AddDomainIntegrator(new DiffusionIntegrator(sig_o));
        ktloc.Assemble(); ktloc.Finalize();
        HypreParMatrix Kt3; ktf.FormSystemMatrix(ess_tdofs_t, Kt3);
        PetscParMatrix Kt3p; HypreToPetscAIJ(Kt3, Kt3p, "Sys3_Kt", rank, 1, true);
        Array<int> ess_vmark, ess_ld3;
        fes_t.GetEssentialVDofs(ess_iface, ess_vmark);
        for (int i=0;i<ess_vmark.Size();++i) if (ess_vmark[i]<0) ess_ld3.Append(i);

        auto sasm_LO = [&](Mat A, Vec b, Vec x, int O, int L)->int{
            KSP ksp; KSPCreate(PetscObjectComm((PetscObject)A), &ksp);
            KSPSetType(ksp, KSPCG); KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED);
            KSPSetOperators(ksp, A, A);
            KSPSetTolerances(ksp, 1e-8, 1e-50, PETSC_DEFAULT, 2000);
            InstallScaledASM(ksp, A, O, L, NSUB, 0);       // ICC(L) apply, mult PU
            VecSet(x,0.0); KSPSolve(ksp,b,x);
            PetscInt it; KSPGetIterationNumber(ksp,&it);
            KSPConvergedReason r; KSPGetConvergedReason(ksp,&r); if(r<0) it=-it;
            KSPDestroy(&ksp); return (int)it;
        };

        Array<int> none;
        struct Row { const char *name; ParFiniteElementSpace *fes; ParMesh *pm;
                     ParBilinearForm *loc; PetscParMatrix *A; bool sing; Array<int>*ess; };
        Row R[3] = {
            {"Sys1 monodomain",   &fes_h,&heart,&a1loc,&A1p, false,&none},
            {"Sys2 u_e (hard)",   &fes_h,&heart,&kief, &Kiep, true, &none},
            {"Sys3 torso Laplace",&fes_t,&torso,&ktloc,&Kt3p,false,&ess_ld3},
        };
        for (int q=0;q<3;++q){
            Mat A = (Mat)*R[q].A;
            if (R[q].sing) AttachConstNullSpace(A, MPI_COMM_WORLD);
            Vec xstar, b, x; MatCreateVecs(A, &xstar, &b); VecDuplicate(xstar, &x);
            PetscInt rs, re; MatGetOwnershipRange(A, &rs, &re);
            PetscInt N; MatGetSize(A, &N, NULL);
            { PetscScalar *a; VecGetArray(xstar, &a);
              for (PetscInt i=rs;i<re;++i) a[i-rs]=sin(0.7*(i+1))+0.3*cos(0.11*(i+1));
              VecRestoreArray(xstar, &a); }
            if (R[q].sing){ PetscScalar s; VecSum(xstar,&s); VecShift(xstar,-s/N); }
            MatMult(A, xstar, b);
            if (R[q].sing){ PetscScalar s; VecSum(b,&s); VecShift(b,-s/N); }
            if (rank==0) cout << "  " << R[q].name << "  (alpha=" << a_opt[q] << ")\n"
                              << "     L | sASM_O0 sASM_O1 sASM_O2 | cwSORAS_d0\n";
            for (int L=0;L<=2;++L){
                int s0=sasm_LO(A,b,x,0,L), s1=sasm_LO(A,b,x,1,L), s2=sasm_LO(A,b,x,2,L);
                int cs=SorasPUIters(*R[q].fes,*R[q].pm,*R[q].loc,*R[q].A,a_opt[q],1,0,
                                    R[q].sing,*R[q].ess,b,nullptr,nullptr,L);   // cw-SORAS ICC(L)
                if (rank==0) cout << "     " << L << " |  " << std::setw(5) << s0 << "  "
                                  << std::setw(5) << s1 << "  " << std::setw(5) << s2
                                  << "  |  " << std::setw(5) << cs << "\n";
            }
            VecDestroy(&xstar); VecDestroy(&b); VecDestroy(&x);
        }
        if (rank==0) cout <<
          "[FAIR] all local solves = one ICC(L) apply (memory-flat, CG-valid).\n"
          "  per-apply COMPUTE : sASM_O0 ~ cwSORAS_d0 (both 1 ICC on ~same-size block); sASM_O1/O2\n"
          "                      grow the block by 1-2 ghost layers => more work + ICC(L) fill.\n"
          "  COMMUNICATION     : outer Allreduce ~ iter count (the >>P<< bottleneck); halo width\n"
          "                      = 1(cwSORAS_d0, interface only) < O+1(sASM_O overlap) (nearest-nbr).\n"
          "  MEMORY            : ICC(L) fill grows with L and with overlap block size; cwSORAS_d0\n"
          "                      smallest block; sASM_O2 largest.\n"
          "  CONSTRUCTION      : sASM = PCASMSetOverlap (1 line).  cwSORAS_d0 = hand M_Gamma + P +\n"
          "                      coef-PU (moderate).  overlapping cwSORAS = cross-rank Neumann-patch\n"
          "                      assembly + moved Robin + gather/scatter (HARD; not built).\n";
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

    // wall-clock accumulators for the Sys2 solve (baseline vs accelerated)
    double t_base=0.0, t_acc=0.0;
    // Vm snapshots for the -xsys study (stored at ECG sample times)
    std::vector<Vector> Vm_seq;
    FILE *fe = (!do_precond && !do_prop && !do_sweep && !do_weightcmp && !do_soraspu && !do_transmit && !do_tuned && !do_overlap && !do_fair && !do_transmiti && !do_neumann && rank==0) ? fopen("fwd_ecg.txt","w") : nullptr;
    if (fe) fprintf(fe,"# t(ms)  ECG(phi_L-phi_R)  Vm@center  u_e@center  phi_torso@L\n");

    // ====================================================================
    //  time loop  (IMEX: explicit TP06 reaction + C-N diffusion)
    // ====================================================================
    const int nsteps = (do_precond||do_prop||do_sweep||do_weightcmp||do_soraspu||do_transmit||do_tuned||do_overlap||do_fair||do_transmiti||do_neumann) ? 0 : (int)(Tend/dt);   // -precond/-prop/-sweep skip the EP loop
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
                int it2_base=-1, it2_two=-1;
                if (do_acc) {
                    // baseline (bjacobi+ICC) -- keep this solve as the field
                    double bn = std::sqrt(ip2(b2,b2));
                    cg2.SetRelTol(0.0); cg2.SetAbsTol(1e-8*bn);
                    KSPSetInitialGuessNonzero((KSP)cg2, PETSC_FALSE);
                    MPI_Barrier(MPI_COMM_WORLD); double w0=MPI_Wtime();
                    cg2.Mult(b2, ue_h); it2_base=cg2.GetNumIterations();
                    MPI_Barrier(MPI_COMM_WORLD); t_base += MPI_Wtime()-w0;
                    // accelerated (SORAS and/or coarse) -- measure only
                    cg2c->SetRelTol(0.0); cg2c->SetAbsTol(1e-8*bn);
                    KSPSetInitialGuessNonzero((KSP)*cg2c, PETSC_FALSE);
                    Vector xtwo(nloc); xtwo=0.0;
                    MPI_Barrier(MPI_COMM_WORLD); double w1=MPI_Wtime();
                    cg2c->Mult(b2, xtwo); it2_two=cg2c->GetNumIterations();
                    MPI_Barrier(MPI_COMM_WORLD); t_acc += MPI_Wtime()-w1;
                    coarse_base+=it2_base; coarse_two+=it2_two;
                } else if (do_fischer) {
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
                    if (do_acc)
                        cout << it2_two<<" (baseline="<<it2_base<<" "<<acc_label<<")";
                    else if (do_fischer)
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
            // Vm and u_e at heart center (globally-nearest dof to origin)
            int ci=-1; double cbest=1e300;
            for(int p=0;p<nloc;++p){double d=tdof_x(p)*tdof_x(p)+tdof_y(p)*tdof_y(p)+tdof_z(p)*tdof_z(p);
                if(d<cbest){cbest=d;ci=p;}}
            double vc  = global_at(cbest, (ci>=0)?Vm(ci):0.0);
            Vector ueh_c; ue_h.GetTrueDofs(ueh_c);
            double uec = global_at(cbest, (ci>=0)?ueh_c(ci):0.0);   // u_e @ heart center
            // t  ECG(phi_L-phi_R)  Vm@center  u_e@center  phi_torso@left-electrode
            if (fe) fprintf(fe,"%g %g %g %g %g\n", t+dt, ecg, vc, uec, pl);
        }
    }
    if (fe) fclose(fe);

    if (do_acc && rank==0 && coarse_base>0) {
        cout << "\n[ACC] Sys2 EP-loop acceleration (total over the run):\n"
             << std::fixed << std::setprecision(3)
             << "  baseline (bjacobi+ICC)   : iters=" << coarse_base
             << "  time=" << t_base << "s  (" << 1e3*t_base/coarse_base << " ms/iter)\n"
             << "  " << acc_label << std::string(std::max(0,21-(int)acc_label.size()),' ')
             << ": iters=" << coarse_two
             << "  time=" << t_acc << "s  (" << 1e3*t_acc/coarse_two << " ms/iter)\n"
             << std::setprecision(2)
             << "  => iters " << (double)coarse_base/coarse_two << "x fewer,  wall-clock "
             << t_base/t_acc << "x " << (t_acc<t_base?"faster":"SLOWER") << "\n"
             << std::defaultfloat;
    }
    delete cg2c; delete twolvl; delete soras; delete finePC;
    if (KrobA) MatDestroy(&KrobA);

    if (do_fischer && rank==0 && fisch_cold>0) {
        cout << "\n[FISCHER] Sys2 EP-loop cross-time acceleration (total CG iters over the run):\n"
             << "  cold (x0=0)          : " << fisch_cold << "\n"
             << "  warm (prev u_e)      : " << fisch_warm
             << "  (-" << (int)(100.0*(fisch_cold-fisch_warm)/fisch_cold) << "%)\n"
             << "  Fischer (history)    : " << fisch_fis
             << "  (-" << (int)(100.0*(fisch_cold-fisch_fis)/fisch_cold) << "%)\n";
    }

    // ---- benchmark activation times + conduction velocity -----------------
    if (!do_precond && !do_sweep && !do_weightcmp && !do_soraspu && !do_transmit && !do_tuned && !do_overlap && !do_fair && !do_transmiti && !do_neumann) {
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
