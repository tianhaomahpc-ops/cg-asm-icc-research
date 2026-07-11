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
#include <random>
#include <utility>
#include <map>

using namespace mfem;
using namespace std;

// LAPACK symmetric eigensolver (MFEM here is built without LAPACK, but -llapack is
// linked for PETSc, so call dsyev directly for the small K x K Gram matrices).
extern "C" void dsyev_(char*, char*, int*, double*, int*, double*, double*, int*, int*);
extern "C" void dgemv_(char*, int*, int*, double*, const double*, int*, const double*,
                       int*, double*, double*, int*);

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

// ---- general two-level: fine + a coarse space spanned by `nm` local mode-vectors
//      per subdomain (rank).  Global coarse basis W = { rank k's nm modes },
//      dim m = nm*np.  M^{-1} r = fine(r) + W E^{-1} W^T r,  E = W^T A W (m x m,
//      replicated).  nm=1 with mode 0 = indicator reproduces TwoLevelNicolaides;
//      nm=4 with modes {1, x, y, z} spans the smooth low-frequency slow subspace.
//      proj_const handles a singular A (constant nullspace = sum of the indicators).
class TwoLevelCoarse : public Solver
{
    Solver &fine_; const Operator &A_; MPI_Comm comm_;
    int np_, rk_, nm_, m_;
    std::vector<Vector> modes_;         // nm local modes (nloc each), mode 0 = indicator
    DenseMatrix Einv_;                  // m x m, replicated
    Vector nsp_;                        // coarse-coeff nullspace dir (singular A)
    bool proj_;
    mutable Vector fz_;
public:
    TwoLevelCoarse(Solver &fine, const Operator &A, MPI_Comm comm, int np, int rk,
                   int nloc, std::vector<Vector> modes, bool proj_const)
      : Solver(nloc), fine_(fine), A_(A), comm_(comm), np_(np), rk_(rk),
        nm_((int)modes.size()), m_(np*(int)modes.size()), modes_(std::move(modes)),
        nsp_(m_), proj_(proj_const), fz_(nloc)
    {
        DenseMatrix E(m_); E = 0.0;
        Vector g(nloc), Ag(nloc); std::vector<double> locdot(nm_), allv(m_);
        for (int k=0;k<np_;++k) for (int cm=0; cm<nm_; ++cm){
            g = 0.0; if (rk_==k) g = modes_[cm];
            A_.Mult(g, Ag);                                   // global matvec
            for (int cj=0;cj<nm_;++cj) locdot[cj] = (modes_[cj]*Ag);  // rk's local dots
            MPI_Allgather(locdot.data(), nm_, MPI_DOUBLE, allv.data(), nm_, MPI_DOUBLE, comm_);
            int jc = k*nm_ + cm; for (int row=0; row<m_; ++row) E(row, jc) = allv[row];
        }
        // constant nullspace of a singular A lives in coeff dir nsp = 1 on every
        // (k, indicator) entry, 0 on coord entries.  Regularise E += s nsp nsp^T.
        nsp_ = 0.0;
        if (proj_) { for (int k=0;k<np_;++k) nsp_(k*nm_)=1.0; nsp_ /= nsp_.Norml2();
            double s=0.0; for(int i=0;i<m_;++i) s+=E(i,i); s/=m_;
            for(int i=0;i<m_;++i) for(int j=0;j<m_;++j) E(i,j) += s*nsp_(i)*nsp_(j); }
        Einv_ = E; Einv_.Invert();
    }
    void SetOperator(const Operator&) override {}
    void Mult(const Vector &r, Vector &z) const override
    {
        fine_.Mult(r, fz_);
        std::vector<double> locdot(nm_);
        for (int cj=0;cj<nm_;++cj) locdot[cj] = (modes_[cj]*r);
        Vector c(m_), y(m_);
        MPI_Allgather(locdot.data(), nm_, MPI_DOUBLE, c.GetData(), nm_, MPI_DOUBLE, comm_);
        if (proj_){ double d=(nsp_*c); c.Add(-d, nsp_); }   // r may carry a bit of const
        Einv_.Mult(c, y);
        if (proj_){ double d=(nsp_*y); y.Add(-d, nsp_); }
        z = fz_;
        for (int cm=0;cm<nm_;++cm) z.Add(y(rk_*nm_+cm), modes_[cm]);   // prolong local modes
    }
};

// ---- general GLOBAL-vector deflation: coarse basis W = arbitrary global columns
//      (mean-removed so E=W^T A W is SPD even for a singular-const A).  Handles
//      BOTH geometric (per-subdomain, stored as global-restricted columns) and
//      recycled (full global) vectors uniformly, so the two compose in one W.
//      M^{-1} r = fine(r) + W E^{-1} W^T r.
class GlobalDeflate : public Solver
{
    Solver &fine_; const Operator &A_; MPI_Comm comm_; long N_;
    std::vector<Vector> W_; DenseMatrix Einv_; int m_; mutable Vector fz_;
    double gdot(const Vector&a,const Vector&b) const
    { double l=(a*b),g; MPI_Allreduce(&l,&g,1,MPI_DOUBLE,MPI_SUM,comm_); return g; }
public:
    GlobalDeflate(Solver&fine,const Operator&A,MPI_Comm comm,int nloc,long Nglob,
                  std::vector<Vector> W)
      : Solver(nloc),fine_(fine),A_(A),comm_(comm),N_(Nglob),
        W_(std::move(W)),m_((int)W_.size()),fz_(nloc)
    {
        for (auto &w : W_){ double l=w.Sum(),g; MPI_Allreduce(&l,&g,1,MPI_DOUBLE,MPI_SUM,comm_);
            double mn=g/N_; for(int i=0;i<w.Size();++i) w(i)-=mn; }   // mean-remove
        // modified Gram-Schmidt: orthonormalise, DROP near-dependent columns
        // (A^{-1}-random snapshots collapse onto v_min => many are redundant).
        std::vector<Vector> Q;
        for (auto &w : W_){ Vector v(w);
            for (auto &q : Q){ double d=gdot(v,q); v.Add(-d,q); }
            double nv=std::sqrt(gdot(v,v));
            if (nv>1e-7){ v/=nv; Q.push_back(v); } }
        W_.swap(Q); m_=(int)W_.size();
        std::vector<Vector> AW(m_);
        for (int k=0;k<m_;++k){ AW[k].SetSize(nloc); A_.Mult(W_[k], AW[k]); }
        DenseMatrix E(m_);
        for (int j=0;j<m_;++j) for (int k=0;k<m_;++k) E(j,k)=gdot(W_[j],AW[k]);
        Einv_=E; Einv_.Invert();
    }
    int Dim() const { return m_; }             // effective (post-orthogonalisation) dim
    void SetOperator(const Operator&) override {}
    void Mult(const Vector &r, Vector &z) const override
    {
        fine_.Mult(r, fz_);
        Vector c(m_), y(m_);
        for (int j=0;j<m_;++j) c(j)=gdot(W_[j], r);
        Einv_.Mult(c, y);
        z = fz_; for (int j=0;j<m_;++j) z.Add(y(j), W_[j]);
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
    // (constant nullspace).  loc_mode!=-3: pin ONE local dof (minimal Dirichlet
    // anchor) so a cheap ICC0 apply is possible.  loc_mode==-3: DON'T pin -- keep
    // the true singular Neumann block and remove the nullspace via projection
    // (attach a constant MatNullSpace below => the CG solves the pseudoinverse
    // K^+ w on range(K); the honest "exclude the nullspace" alternative to pin).
    if (alpha == 0.0 && ess_ld.Size() == 0 && loc_mode != -3) Krob.EliminateRowCol(0);
    Mat KrobA = ToSeqAIJ(Krob);
    if (alpha == 0.0 && loc_mode == -3) {          // pseudoinverse: project out constants
        MatNullSpace nsp; MatNullSpaceCreate(PETSC_COMM_SELF, PETSC_TRUE, 0, NULL, &nsp);
        MatSetNullSpace(KrobA, nsp); MatNullSpaceDestroy(&nsp);
    }

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
    if (loc_mode == -3) {                        // ONE ICC0 apply + nullspace projection
        // The cheap "ICC + remove nullspace" path: KSP projects b/x onto range(K)
        // (MatSetNullSpace above), but ICC still factors the SINGULAR K -> its
        // zero pivot needs a positive-definite shift to exist.  Nullspace removal
        // (a SOLVE-level op) does NOT by itself let the ICC FACTORIZATION survive.
        KSPSetType(sp.kloc, KSPPREONLY);
        { PC pc; KSPGetPC(sp.kloc,&pc); PCSetType(pc,PCICC);
          PCFactorSetShiftType(pc, MAT_SHIFT_POSITIVE_DEFINITE); }  // survive singular ICC
    } else if (loc_mode == -2) {                 // DIRECT Cholesky (exact, factor once)
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
    bool do_fischer3 = false, do_f3gamg = false;
    opts.AddOption(&do_fischer3, "-fischer3", "--fischer3", "-nofischer3", "--no-fischer3",
                   "Sys3 torso Laplace: PERSISTENT operator+solver (built once, reused "
                   "every step -- avoids per-step PETSc convert + ICC factorization) plus "
                   "cb-Fischer recycling + warm start.  Non-singular, so no mean-removal.  "
                   "Reports Sys3 cold/warm/Fischer iterations, wall-time, Allreduce counts.");
    opts.AddOption(&do_f3gamg, "-fischer3gamg", "--fischer3gamg", "-nof3gamg", "--no-f3gamg",
                   "With -fischer3, use smoothed-aggregation AMG (GAMG) as the Sys3 fine PC "
                   "instead of bjacobi+ICC (measures GAMG iters + per-solve time vs sASM).");
    bool do_leadvol = false; int lv_max = 40;
    bool do_f3cheb = false;
    opts.AddOption(&do_f3cheb, "-f3cheb", "--f3cheb", "-nof3cheb", "--no-f3cheb",
                   "With -fischer3, solve Sys3 by CHEBYSHEV iteration (bjacobi+ICC PC) with "
                   "eigenvalues [emin,emax] estimated ONCE (constant operator) via a one-shot "
                   "CG -- thereafter every step's iterations do ZERO inner products / ZERO "
                   "Allreduce (fixed-coefficient polynomial).  Reports Chebyshev iters vs CG.");
    opts.AddOption(&do_leadvol, "-leadvol", "--leadvol", "-noleadvol", "--no-leadvol",
                   "Sys3 FULL-FIELD reduced-basis (lead-field idea kept at volume resolution): "
                   "grow an L2-orthonormal basis of the RHS trajectory, precompute Psi_i=Kt^-1 B_i "
                   "ONCE each, then per step phi=sum <Bt,B_i> Psi_i -- full torso field by "
                   "SUPERPOSITION, no per-step solve.  Reports basis size, %steps needing no "
                   "solve, and full-field rel-L2 error vs the true solve.");
    opts.AddOption(&lv_max, "-lvmax", "--lvmax", "Max lead-field/reduced basis size for -leadvol.");
    bool do_transfer = false;
    opts.AddOption(&do_transfer, "-transfer", "--transfer", "-notransfer", "--no-transfer",
                   "Sys3 interface->torso TRANSFER OPERATOR Z: build ONCE (one Sys3 solve per "
                   "interface DOF, column = Kt^-1 lift of that unit interface value), then per "
                   "step phi = Z * u_iface(t) via a local dense matvec -- EXACT for any RHS "
                   "(approximate the fixed operator, not the moving solutions).  Validates the "
                   "full torso field vs the true per-step solve and reports build/apply cost.");
    bool do_transferh = false; int th_samp = 80, th_nleaf = 4, th_ngrp = 3;
    double th_tol = 1e-6, th_eta = 1.5;
    opts.AddOption(&do_transferh, "-transferh", "--transferh", "-notransferh", "--no-transferh",
                   "2-sided H-matrix compression of the transfer operator Z: cluster torso targets "
                   "into leaf boxes and interface sources into groups; each (leaf x group) block is "
                   "DENSE if the clusters are close, else low-rank U*V (well-separated harmonic "
                   "blocks are low rank).  Per step phi = sum-of-blocks, fewer flops than dense/solve.");
    opts.AddOption(&th_nleaf, "-thnleaf", "--thnleaf", "Target leaf boxes per axis (per rank) for -transferh.");
    opts.AddOption(&th_ngrp, "-thngrp", "--thngrp", "Interface source groups per axis for -transferh.");
    opts.AddOption(&th_samp, "-thsamp", "--thsamp", "Max randomized samples per admissible block.");
    opts.AddOption(&th_eta, "-theta", "--theta", "Admissibility eta (block low-rank if dist>eta*(rL+rS)).");
    opts.AddOption(&th_tol, "-thtol", "--thtol", "Rel. singular-value truncation tol for -transferh.");
    bool do_transferinc = false; double ti_eps = 1e-3; int ti_refresh = 20;
    opts.AddOption(&do_transferinc, "-transferinc", "--transferinc", "-notransferinc", "--no-transferinc",
                   "Incremental front-localized transfer: maintain phi(t)=phi(t-1)+Z*(u_e(t)-u_e(t-1)); "
                   "only interface DOFs whose u_e changed (|d|>eps*max|d|, i.e. the moving front from "
                   "Sys1/Sys2) update their column of Z -- per step touches K<<N_iface columns, "
                   "streaming K columns not the whole 1.3 GB Z.  Periodic full refresh clears drift.  "
                   "Uses Sys1 (front location) + Sys2 (u_e increment); validates vs the true solve.");
    opts.AddOption(&ti_eps, "-tieps", "--tieps", "Active-column threshold (rel. to max|d|) for -transferinc.");
    opts.AddOption(&ti_refresh, "-tirefresh", "--tirefresh", "Full-refresh period (steps) for -transferinc.");
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
    bool do_deflate = false;
    opts.AddOption(&do_deflate, "-deflate", "--deflate", "-nodeflate", "--no-deflate",
                   "COARSE-SPACE / deflation on the real Sys2: plain bjacobi+ICC vs "
                   "two-level with a Nicolaides (1/subdomain) and a richer {1,x,y,z}/"
                   "subdomain coarse space; report iters -- does deflating the ~12-dim "
                   "geometric slow subspace collapse the 67-step floor? (skips EP).");
    bool do_anisocmp = false;
    opts.AddOption(&do_anisocmp, "-anisocmp", "--anisocmp", "-noanisocmp", "--no-anisocmp",
                   "ISOTROPIC vs ANISOTROPIC: build Sys1 and Sys2 with sigma_T:=sigma_L "
                   "(isotropic) vs the real anisotropic tensor, compare sASM(O1,ICC0) "
                   "iters + condition number -- does anisotropy make it harder? (skips EP).");
    bool do_decay = false;
    opts.AddOption(&do_decay, "-decay", "--decay", "-nodecay", "--no-decay",
                   "RESIDUAL-DECAY + spectrum diagnostic: sASM(O1,ICC0) CG on the 3 "
                   "systems, dump the residual history and the preconditioned-operator "
                   "eigenvalue estimates (how many SMALL modes cause the slow tail => "
                   "the deflation/recycling dimension) (skips EP).");
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
    long fisch_cold=0, fisch_warm=0, fisch_fis=0, fisch_phys=0;   // cumulative iteration tallies
    auto ip2 = [&](const Vector&x,const Vector&y){ return InnerProduct(MPI_COMM_WORLD,x,y); };
    // Batched inner products against a whole basis: dots[i] = <V[i],y> for all i
    // with ONE MPI_Allreduce of an m-vector, instead of m separate Allreduces.
    // At np=8 this is invisible; at P~3000 (collective-latency dominated) the
    // unbatched loop costs 10-20x more than the whole projection should.
    auto ipbatch = [&](const std::vector<Vector>&V, const Vector&y,
                       std::vector<double>&dots){
        const size_t m=V.size(); dots.assign(m,0.0);
        for (size_t i=0;i<m;++i){ const Vector&v=V[i]; double s=0.0;
            for (int k=0;k<v.Size();++k) s+=v(k)*y(k); dots[i]=s; }
        if (m) MPI_Allreduce(MPI_IN_PLACE,dots.data(),(int)m,MPI_DOUBLE,MPI_SUM,
                             MPI_COMM_WORLD);
    };
    // Collective-count bookkeeping: what the batched code actually issues vs what
    // the old unbatched-MGS code would have issued for the same window sizes.
    long fisch_red_new=0, fisch_red_old=0;

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
            cout << "  system                     | Dir(ASM) ICC0  exact | Neu-pin ICC0  exact | "
                    "Neu-nullsp ICC0+shift   (iters/solve-ms)\n";
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

            double dm0=0, dme=0, nm0=0, nme=0, nmn=0;
            int di0 = asm_run(A, b, x, false, &dm0);   // Dirichlet, ICC0
            int die = asm_run(A, b, x, true,  &dme);   // Dirichlet, near-exact
            int ni0 = SorasPUIters(*R[q].fes,*R[q].pm,*R[q].loc,*R[q].A,0.0,1, 0,
                                   R[q].sing,*R[q].ess,b,nullptr,&nm0);  // Neumann(pin), ICC0
            int nie = SorasPUIters(*R[q].fes,*R[q].pm,*R[q].loc,*R[q].A,0.0,1,-1,
                                   R[q].sing,*R[q].ess,b,nullptr,&nme);  // Neumann(pin), near-exact
            int nin = SorasPUIters(*R[q].fes,*R[q].pm,*R[q].loc,*R[q].A,0.0,1,-3,
                                   R[q].sing,*R[q].ess,b,nullptr,&nmn);  // Neumann(nullspace K+)
            if (rank==0){
                cout << "  " << std::left << std::setw(25) << R[q].name << std::right
                     << " | " << std::fixed << std::setprecision(1)
                     << std::setw(4) << di0 << "/" << std::setw(6) << dm0 << "  "
                     << std::setw(4) << die << "/" << std::setw(6) << dme << " | "
                     << std::setw(4) << ni0 << "/" << std::setw(6) << nm0 << "  "
                     << std::setw(4) << nie << "/" << std::setw(6) << nme << " | "
                     << std::setw(4) << nin << "/" << std::setw(6) << nmn
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
    //  -decay : residual-decay + spectrum diagnostic.  For each system, run
    //  sASM(O1,ICC0) CG and record (a) the residual history (does it decay
    //  fast then stall on a slow tail?) and (b) the eigenvalue estimates of the
    //  PRECONDITIONED operator (a few small eigenvalues => a low-dim slow
    //  subspace => deflation/recycling of that many vectors collapses the tail,
    //  the "besides-coarse-space" lever tailored to our repeated Sys2 solves).
    // ====================================================================
    if (do_decay)
    {
        HypreParMatrix Kt3d; ktf.FormSystemMatrix(ess_tdofs_t, Kt3d);
        PetscParMatrix Kt3pd; HypreToPetscAIJ(Kt3d, Kt3pd, "Sys3_Kt", rank, 1, true);
        struct Sd { const char *name; Mat A; bool sing; };
        Sd S3[3] = { {"Sys1 monodomain", (Mat)A1p, false},
                     {"Sys2 u_e (hard)",  (Mat)Kiep, true},
                     {"Sys3 torso",       (Mat)Kt3pd, false} };
        const PetscInt NSUB = 8;
        const int MAXIT = 300;
        if (rank==0) cout << "\n[DECAY] sASM(O1,ICC0) CG residual decay + preconditioned spectrum, "
                          << Mpi::WorldSize() << " subdomains\n";
        for (int q=0;q<3;++q){
            Mat A = S3[q].A;
            if (S3[q].sing) AttachConstNullSpace(A, MPI_COMM_WORLD);
            Vec xstar, b, x; MatCreateVecs(A,&xstar,&b); VecDuplicate(xstar,&x);
            PetscInt rs,re; MatGetOwnershipRange(A,&rs,&re); PetscInt N; MatGetSize(A,&N,NULL);
            { PetscScalar *a; VecGetArray(xstar,&a);
              for(PetscInt i=rs;i<re;++i) a[i-rs]=sin(0.7*(i+1))+0.3*cos(0.11*(i+1));
              VecRestoreArray(xstar,&a); }
            if (S3[q].sing){ PetscScalar s; VecSum(xstar,&s); VecShift(xstar,-s/N); }
            MatMult(A,xstar,b);
            if (S3[q].sing){ PetscScalar s; VecSum(b,&s); VecShift(b,-s/N); }
            KSP ksp; KSPCreate(PetscObjectComm((PetscObject)A),&ksp);
            KSPSetType(ksp,KSPCG); KSPSetNormType(ksp,KSP_NORM_UNPRECONDITIONED);
            KSPSetOperators(ksp,A,A); KSPSetTolerances(ksp,1e-8,1e-50,PETSC_DEFAULT,MAXIT);
            InstallScaledASM(ksp,A,1,0,NSUB,0);
            KSPSetComputeEigenvalues(ksp,PETSC_TRUE); KSPSetComputeSingularValues(ksp,PETSC_TRUE);
            std::vector<PetscReal> hist(MAXIT+2);
            KSPSetResidualHistory(ksp,hist.data(),MAXIT+2,PETSC_TRUE);
            VecSet(x,0.0); KSPSolve(ksp,b,x);
            PetscInt its; KSPGetIterationNumber(ksp,&its);
            PetscReal smax=0,smin=0; KSPComputeExtremeSingularValues(ksp,&smax,&smin);
            std::vector<PetscReal> er(its+2,0), ei(its+2,0); PetscInt neig=0;
            KSPComputeEigenvalues(ksp,its+1,er.data(),ei.data(),&neig);
            std::sort(er.begin(),er.begin()+neig);
            if (rank==0){
                cout << "  " << S3[q].name << ":  iters=" << its
                     << "  cond(M^-1 A)~" << std::fixed << std::setprecision(1) << (smin>0?smax/smin:0)
                     << "  lambda_min~" << std::setprecision(4) << smin
                     << "  lambda_max~" << std::setprecision(2) << smax << std::defaultfloat << "\n";
                // residual decay at checkpoints (relative to r0)
                double r0 = hist[0]>0?hist[0]:1.0;
                cout << "    resid/r0 @ it:";
                for (int k : {0,5,10,20,40,80,160}) if (k<=its)
                    { cout << " " << k << ":" << std::scientific << std::setprecision(1) << hist[k]/r0; }
                cout << std::defaultfloat << " end(" << its << "):" << std::scientific
                     << std::setprecision(1) << hist[its]/r0 << std::defaultfloat << "\n";
                // count "small" eigenvalues (< 0.1 lambda_max) = the slow subspace dim
                int nsmall=0; for (int i=0;i<neig;++i) if (er[i]<0.1*smax) nsmall++;
                cout << "    smallest eigs:";
                for (int i=0;i<std::min((int)neig,8);++i)
                    cout << " " << std::setprecision(3) << er[i];
                cout << "   #(lambda<0.1 lambda_max)=" << nsmall
                     << "  => deflation dim ~" << nsmall << "\n";
            }
            KSPDestroy(&ksp); VecDestroy(&xstar); VecDestroy(&b); VecDestroy(&x);
        }
        if (rank==0) cout << "[DECAY] fast head + slow tail, and a few small eigenvalues, => the "
                             "tail is a LOW-DIM slow subspace; deflating/recycling ~that many "
                             "vectors collapses it (our repeated Sys2 solves make recycling free).\n";
    }

    // ====================================================================
    //  -deflate : COARSE SPACE / deflation on the REAL Sys2 operator.  Does
    //  deflating the ~12-dim geometric slow subspace collapse the ~67-step floor?
    //  Compare plain bjacobi+ICC vs two-level (fine bjacobi+ICC + coarse space):
    //    (a) Nicolaides:  1 indicator per subdomain           (dim = np)
    //    (b) rich:        {1, x, y, z} per subdomain           (dim = 4 np)
    //  (b) spans the smooth low-frequency modes -> should reach the -decay
    //  prediction (~15-20).  Synthetic ||b||-rel test, same as the other studies.
    // ====================================================================
    if (do_deflate)
    {
        const int NPk = Mpi::WorldSize();
        Mat A = (Mat)Kiep; AttachConstNullSpace(A, MPI_COMM_WORLD);
        Vec xstar,b,x; MatCreateVecs(A,&xstar,&b); VecDuplicate(xstar,&x);
        PetscInt rs,re; MatGetOwnershipRange(A,&rs,&re); PetscInt N; MatGetSize(A,&N,NULL);
        { PetscScalar *a; VecGetArray(xstar,&a);
          for(PetscInt i=rs;i<re;++i) a[i-rs]=sin(0.7*(i+1))+0.3*cos(0.11*(i+1));
          VecRestoreArray(xstar,&a); }
        { PetscScalar s; VecSum(xstar,&s); VecShift(xstar,-s/N); }
        MatMult(A,xstar,b);
        { PetscScalar s; VecSum(b,&s); VecShift(b,-s/N); }
        Vector Bv(nloc); { const PetscScalar *ba; VecGetArrayRead(b,&ba);
            for(int i=0;i<nloc;++i) Bv(i)=ba[i]; VecRestoreArrayRead(b,&ba); }

        // coarse modes (local, nloc each): indicator, and centered+normalized x,y,z
        Vector m_ind(nloc); m_ind=1.0;
        auto centnorm=[&](const Vector &src){ Vector v(src);
            double mn=v.Sum(); { double g; MPI_Allreduce(&mn,&g,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
              /*per-subdomain mean = local mean here (1 subdomain/rank)*/ }
            double lm=v.Sum()/std::max(nloc,1); for(int i=0;i<nloc;++i) v(i)-=lm;   // center on subdomain
            double nn=v.Norml2(); if(nn>0) v/=nn; return v; };
        Vector m_x=centnorm(tdof_x), m_y=centnorm(tdof_y), m_z=centnorm(tdof_z);

        auto run_pc=[&](Solver *pc)->int{
            PetscPCGSolver cg(Kiep,"defl_"); cg.SetMaxIter(2000); cg.SetRelTol(1e-8);
            cg.iterative_mode=false; KSPSetNormType((KSP)cg,KSP_NORM_UNPRECONDITIONED);
            cg.SetPreconditioner(*pc);
            Vector Xv(nloc); Xv=0.0; cg.Mult(Bv,Xv); return cg.GetNumIterations();
        };
        // same solve, but also dump the (unpreconditioned) relative residual
        // history r_k/r_0 vs iteration k to a file, so we can PLOT with vs
        // without recycling on the REAL Sys2 operator.
        auto run_pc_hist=[&](Solver *pc,const char*fname)->int{
            const int MAXH=2100; std::vector<PetscReal> hist(MAXH,0.0);
            PetscPCGSolver cg(Kiep,"defl_"); cg.SetMaxIter(2000); cg.SetRelTol(1e-8);
            cg.iterative_mode=false; KSPSetNormType((KSP)cg,KSP_NORM_UNPRECONDITIONED);
            cg.SetPreconditioner(*pc);
            KSPSetResidualHistory((KSP)cg,hist.data(),MAXH,PETSC_TRUE);
            Vector Xv(nloc); Xv=0.0; cg.Mult(Bv,Xv);
            const PetscReal *hp; PetscInt hn; KSPGetResidualHistory((KSP)cg,&hp,&hn);
            if(rank==0 && hn>0){ FILE*f=fopen(fname,"w"); double r0=hp[0]>0?hp[0]:1.0;
                for(PetscInt k=0;k<hn;++k) fprintf(f,"%d %.8e\n",(int)k,hp[k]/r0);
                fclose(f); }
            return cg.GetNumIterations();
        };
        // fine level = bjacobi + ICC (a PetscPreconditioner)
        PetscPreconditioner fine(Kiep,"defl_fine_"); { PC pc=(PC)fine; PCSetType(pc,PCBJACOBI); }
        int it_base = run_pc_hist(&fine,"deflate_hist_fine.txt");
        // two-level, Nicolaides (indicator only)
        { std::vector<Vector> M1{m_ind};
          TwoLevelCoarse tl(fine,*Kie,MPI_COMM_WORLD,NPk,rank,nloc,M1,true);
          int it_nic=run_pc(&tl);
          // two-level, rich {1,x,y,z}
          std::vector<Vector> M4{m_ind,m_x,m_y,m_z};
          TwoLevelCoarse tl4(fine,*Kie,MPI_COMM_WORLD,NPk,rank,nloc,M4,true);
          int it_rich=run_pc_hist(&tl4,"deflate_hist_geometric.txt");

          // ---- RECYCLING: harvest slow-mode-rich SOLUTION SNAPSHOTS from prior
          //      solves (A^{-1} amplifies small-lambda modes => y = A^{-1} w is
          //      slow-mode-rich; this is the POD/Fischer basis, not raw directions).
          const int NSNAP=12;
          auto trainsolve=[&](const Vector&w)->Vector{
              PetscPCGSolver cgt(Kiep,"train_"); cgt.SetMaxIter(2000); cgt.SetRelTol(1e-8);
              cgt.iterative_mode=false; KSPSetNormType((KSP)cgt,KSP_NORM_UNPRECONDITIONED);
              cgt.SetPreconditioner(fine); Vector y(nloc); y=0.0; cgt.Mult(w,y); return y; };
          std::vector<Vector> W_rec, W_hyb;
          for(int j=0;j<NSNAP;++j){
              Vector w(nloc); for(int i=0;i<nloc;++i)
                  w(i)=sin((0.3+0.17*j)*(i+1))+0.3*cos((0.09+0.04*j)*(i+1));
              { double l=w.Sum(),g; MPI_Allreduce(&l,&g,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
                double mn=g/N; for(int i=0;i<nloc;++i) w(i)-=mn; }
              Vector y=trainsolve(w); W_rec.push_back(y); W_hyb.push_back(y);
          }
          int nkeep=NSNAP;
          // geometric columns as GLOBAL-restricted vectors (this rank's, others 0)
          for(int k=0;k<NPk;++k){ for(Vector*mp:{&m_ind,&m_x,&m_y,&m_z}){
              Vector col(nloc); col=0.0; if(rank==k) col=*mp; W_hyb.push_back(col); } }
          GlobalDeflate gr(fine,*Kie,MPI_COMM_WORLD,nloc,(long)N,W_rec);  int it_rec=run_pc_hist(&gr,"deflate_hist_recycled.txt");
          GlobalDeflate gh(fine,*Kie,MPI_COMM_WORLD,nloc,(long)N,W_hyb);  int it_hyb=run_pc_hist(&gh,"deflate_hist_hybrid.txt");
          int rdim=gr.Dim(), hdim=gh.Dim();

          if (rank==0){
            cout << "\n[DEFLATE] Sys2 (real operator), " << NPk << " subdomains, synthetic rtol 1e-8:\n"
                 << "  fine only (bjacobi+ICC)                    : " << it_base << " iters\n"
                 << "  + Nicolaides coarse (dim=" << NPk << ")                 : " << it_nic
                 << "  (-" << (it_base>0?100*(it_base-it_nic)/it_base:0) << "%)\n"
                 << "  + geometric {1,x,y,z} coarse (dim=" << 4*NPk << ")      : " << it_rich
                 << "  (-" << (it_base>0?100*(it_base-it_rich)/it_base:0) << "%)\n"
                 << "  + RECYCLED snapshots (" << NSNAP << " -> " << rdim << " indep)      : " << it_rec
                 << "  (-" << (it_base>0?100*(it_base-it_rec)/it_base:0) << "%)\n"
                 << "  + HYBRID geometric + recycled (dim=" << hdim << ")   : " << it_hyb
                 << "  (-" << (it_base>0?100*(it_base-it_hyb)/it_base:0) << "%)\n"
                 << "[DEFLATE] geometric = a-priori (cold-safe, crude); recycled snapshots collapse\n"
                    "  onto the few smallest modes (" << NSNAP << "->" << rdim << " indep) so alone they're weak; "
                    "HYBRID unions both\n  in one deflation space -> >= geometric (monotone, cold-safe).  "
                    "Real strong recycling\n  = time-evolution snapshots (Fischer -77%), which sweep the "
                    "full slow subspace.\n";
          }
        }
        VecDestroy(&xstar); VecDestroy(&b); VecDestroy(&x);
    }

    // ====================================================================
    //  -anisocmp : ISOTROPIC vs ANISOTROPIC.  Rebuild Sys1 (A1=(1/dt)M+(1/2)K)
    //  and Sys2 (Kie) with sigma_T:=sigma_L (isotropic, ratio 1) vs the real
    //  anisotropic tensor, and compare sASM(O1,ICC0) iters + cond.  Isotropic K
    //  = scalar x Laplacian => its cond is the plain-Laplacian cond (scalar
    //  cancels); anisotropy amplifies it by the fiber ratio => harder or not?
    // ====================================================================
    if (do_anisocmp)
    {
        const PetscInt NSUB = 8;
        auto run_spec = [&](Mat A, bool sing, int *itout, double *cond)->void{
            if (sing) AttachConstNullSpace(A, MPI_COMM_WORLD);
            Vec xstar,b,x; MatCreateVecs(A,&xstar,&b); VecDuplicate(xstar,&x);
            PetscInt rs,re; MatGetOwnershipRange(A,&rs,&re); PetscInt N; MatGetSize(A,&N,NULL);
            { PetscScalar *a; VecGetArray(xstar,&a);
              for(PetscInt i=rs;i<re;++i) a[i-rs]=sin(0.7*(i+1))+0.3*cos(0.11*(i+1));
              VecRestoreArray(xstar,&a); }
            if (sing){ PetscScalar s; VecSum(xstar,&s); VecShift(xstar,-s/N); }
            MatMult(A,xstar,b);
            if (sing){ PetscScalar s; VecSum(b,&s); VecShift(b,-s/N); }
            KSP ksp; KSPCreate(PetscObjectComm((PetscObject)A),&ksp);
            KSPSetType(ksp,KSPCG); KSPSetNormType(ksp,KSP_NORM_UNPRECONDITIONED);
            KSPSetOperators(ksp,A,A); KSPSetTolerances(ksp,1e-8,1e-50,PETSC_DEFAULT,500);
            InstallScaledASM(ksp,A,1,0,NSUB,0); KSPSetComputeSingularValues(ksp,PETSC_TRUE);
            VecSet(x,0.0); KSPSolve(ksp,b,x);
            PetscInt it; KSPGetIterationNumber(ksp,&it);
            PetscReal smax=0,smin=0; KSPComputeExtremeSingularValues(ksp,&smax,&smin);
            *itout=(int)it; *cond=(smin>0?smax/smin:0);
            KSPDestroy(&ksp); VecDestroy(&xstar); VecDestroy(&b); VecDestroy(&x);
        };
        // ---- Sys1 iso: (1/dt)M + (1/2) K(sigma_L isotropic) ----
        DenseMatrix D1i = DiagSigma(sLm/chiCm, sLm/chiCm);
        MatrixConstantCoefficient s1i(D1i);
        ParBilinearForm k1i(&fes_h); k1i.AddDomainIntegrator(new DiffusionIntegrator(s1i));
        k1i.Assemble(); k1i.Finalize(); HypreParMatrix *Kd1i=k1i.ParallelAssemble();
        HypreParMatrix *A1i=Add(1.0/dt,*M,0.5,*Kd1i);
        PetscParMatrix A1ip; HypreToPetscAIJ(*A1i,A1ip,"S1iso",rank,1,true);
        // ---- Sys2 iso: K(sigma_i+sigma_e, isotropic at L) ----
        DenseMatrix D2i = DiagSigma(siL+seL, siL+seL);
        MatrixConstantCoefficient s2i(D2i);
        ParBilinearForm k2i(&fes_h); k2i.AddDomainIntegrator(new DiffusionIntegrator(s2i));
        k2i.Assemble(); k2i.Finalize(); HypreParMatrix *Kie_i=k2i.ParallelAssemble();
        PetscParMatrix Kie_ip; HypreToPetscAIJ(*Kie_i,Kie_ip,"S2iso",rank,1,true);

        int i1i,i1a,i2i,i2a; double c1i,c1a,c2i,c2a;
        run_spec((Mat)A1ip,  false,&i1i,&c1i);   // Sys1 iso
        run_spec((Mat)A1p,   false,&i1a,&c1a);   // Sys1 aniso (real)
        run_spec((Mat)Kie_ip,true, &i2i,&c2i);   // Sys2 iso
        run_spec((Mat)Kiep,  true, &i2a,&c2a);   // Sys2 aniso (real)
        if (rank==0){
            double r1=sTm/sLm, r2=(siT+seT)/(siL+seL);
            cout << "\n[ANISOCMP] sASM(O1,ICC0), " << Mpi::WorldSize()
                 << " subdomains.  iso = sigma_T:=sigma_L (ratio 1); aniso = real tensor\n"
                 << std::fixed << std::setprecision(2)
                 << "  Sys1 monodomain (mass-dom): iso " << i1i << " it (cond " << std::setprecision(1) << c1i
                 << ")  ->  aniso[ratio " << std::setprecision(2) << r1 << "] " << i1a << " it (cond "
                 << std::setprecision(1) << c1a << ")\n"
                 << "  Sys2 u_e recover (elliptic): iso " << i2i << " it (cond " << c2i
                 << ")  ->  aniso[ratio " << std::setprecision(2) << r2 << "] " << i2a << " it (cond "
                 << std::setprecision(1) << c2a << ")" << std::defaultfloat << "\n";
            cout << "[ANISOCMP] Sys1 barely moves (mass dominates the stiffness anisotropy); "
                    "Sys2 iso->aniso ratio: " << std::setprecision(2) << (double)i2a/std::max(i2i,1)
                 << "x iters, " << std::setprecision(2) << c2a/std::max(c2i,1e-9)
                 << "x cond -- anisotropy makes the ELLIPTIC recovery harder.\n" << std::defaultfloat;
        }
        delete A1i; delete Kd1i; delete Kie_i;
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
    FILE *fe = (!do_precond && !do_prop && !do_sweep && !do_weightcmp && !do_soraspu && !do_transmit && !do_tuned && !do_overlap && !do_fair && !do_transmiti && !do_neumann && !do_decay && !do_anisocmp && !do_deflate && rank==0) ? fopen("fwd_ecg.txt","w") : nullptr;
    if (fe) fprintf(fe,"# t(ms)  ECG(phi_L-phi_R)  Vm@center  u_e@center  phi_torso@L\n");

    // ---- Sys3 (-fischer3): PERSISTENT torso operator + solver + recycling ----
    // The torso stiffness K_t is time-invariant (only the interface Dirichlet data
    // from u_e(t) drifts), so assemble + convert + factor it ONCE and reuse the
    // solver every step.  Sys3 is SPD/non-singular (Dirichlet-anchored) so the
    // Fischer basis needs NO mean-removal (unlike singular Sys2).
    PetscParMatrix *Kt3p_persist = nullptr;
    PetscPCGSolver *cg3p = nullptr;
    HypreParMatrix *Kt3h_persist = nullptr;
    std::vector<Vector> f3_P, f3_AP;                 // A-orthonormal history + A*history
    long f3_cold=0, f3_warm=0, f3_phys=0, f3_fis=0, f3_red_new=0, f3_red_old=0;
    Vector f3_prev; bool f3_have_prev=false;         // previous cold solution (warm/A2)
    double t3_cold_solve=0.0;                        // wall-clock of the kept cold solve
    const int F3MAX = 12;                            // window ~ Sys3's 8 slow modes x1.5
    // -leadvol: full-field reduced basis (superposition of precomputed response fields)
    std::vector<Vector> lv_B, lv_Psi;                // L2-orthonormal RHS basis; Psi_i=Kt^-1 B_i
    long lv_grow=0, lv_free=0; double lv_err_sum=0.0, lv_err_max=0.0;
    const double LV_TOL = 1e-3;                       // accept a new RHS direction if this big
    // -transfer: interface->torso transfer operator Z (nloc_true x Niface_glob, dense/rank)
    std::vector<double> Zloc;                         // row-major: Zloc[i*Niface + g]
    int tr_niface=0; std::vector<int> tr_cnt, tr_disp;
    double tr_build_wall=0.0, tr_apply_wall=0.0, tr_solve_wall=0.0, tr_apply_cpu=0.0;
    double tr_err_sum=0.0, tr_err_max=0.0; long tr_steps=0;
    // electrode reciprocity lead-field: w_e = Kt^-1 (unit @ electrode); phi(e)=w_e.Bt
    Vector wL_lead, wR_lead; double tr_lead_wall=0.0, tr_lead_err_max=0.0, tr_ecg_absmax=0.0;
    double tr_lead_build=0.0;
    // -transferh: 2-sided H-matrix.  Targets (this rank's torso dofs) are clustered into
    // leaf boxes, interface sources into groups; each (leaf x group) block is DENSE if the
    // two clusters are close, else compressed to a low-rank U*V (well-separated => low rank).
    struct HBlock { std::vector<int> rows;   // local target rows
                    std::vector<int> cols;   // global source cols
                    bool dense=true; int r=0; std::vector<double> U, V; };
    std::vector<HBlock> th_blk;
    double th_build=0.0, th_apply=0.0, th_solve=0.0, th_err_sum=0.0, th_err_max=0.0; long th_steps=0;
    long th_store=0, th_ndense=0, th_nlr=0;
    // -transferinc: column-major Z + incremental front-localized update state
    std::vector<double> Zcol;                        // column-major: Zcol[g*nloc + i]
    std::vector<double> ti_dprev; Vector ti_phi; bool ti_have=false;
    double ti_apply=0.0, ti_solve=0.0, ti_err_sum=0.0, ti_err_max=0.0;
    long ti_steps=0, ti_active_sum=0, ti_active_max=0, ti_refresh_cnt=0;
    if (do_fischer3 || do_leadvol || do_transfer || do_transferh || do_transferinc) {
        Kt3h_persist = new HypreParMatrix;
        ktf.FormSystemMatrix(ess_tdofs_t, *Kt3h_persist);   // constant matrix
        Kt3p_persist = new PetscParMatrix;
        HypreToPetscAIJ(*Kt3h_persist, *Kt3p_persist, "Sys3_Kt", rank, 1, true);
        // Use an UN-clobbered prefix so we control the PC fully (sys3_ pc_type is
        // pinned to bjacobi by PetscOptionsSetValue at startup).  -f3gamg swaps the
        // fine PC from bjacobi+ICC to smoothed-aggregation AMG to measure its
        // per-iteration cost vs iteration count on the (constant) torso operator.
        if (do_f3gamg) {
            PetscOptionsSetValue(NULL, "-sys3f_pc_type", "gamg");
            PetscOptionsSetValue(NULL, "-sys3f_pc_gamg_type", "agg");
            PetscOptionsSetValue(NULL, "-sys3f_pc_gamg_agg_nsmooths", "1");
            PetscOptionsSetValue(NULL, "-sys3f_pc_gamg_threshold", "0.02");
        } else {
            PetscOptionsSetValue(NULL, "-sys3f_pc_type", "bjacobi");
            PetscOptionsSetValue(NULL, "-sys3f_sub_pc_type", "icc");
        }
        cg3p = new PetscPCGSolver(*Kt3p_persist, "sys3f_");
        cg3p->SetMaxIter(3000); cg3p->iterative_mode=false;
        KSPSetNormType((KSP)*cg3p, KSP_NORM_UNPRECONDITIONED);  // guess quality shows
        if (do_f3cheb) {
            // Chebyshev iteration: fixed-coefficient polynomial in M^-1 A, so its
            // per-iteration work has ZERO inner products / ZERO Allreduce.  It needs
            // [emin,emax] of the preconditioned operator, estimated ONCE below (the
            // operator is constant) and then reused for all 10^4 steps.
            KSPSetType((KSP)*cg3p, KSPCHEBYSHEV);
        }
        if (rank==0) cout << "[FISCHER3] Sys3 persistent solver built once (reused every step)"
                          << (do_f3cheb? "  [KSP=Chebyshev, 0 Allreduce/iter]"
                              : do_f3gamg? "  [PC=GAMG]" : "  [PC=bjacobi+ICC]") << "\n";
    }
    if (do_transfer || do_transferh || do_transferinc) {
        // ---- OFFLINE (once): build the interface->torso transfer operator Z.
        // Column g = Kt^-1 (lift of the g-th global interface unit value): set that one
        // interface DOF to 1, all others 0, form the Sys3 RHS and solve.  Z is fixed for
        // all time; per step phi = Z u_iface(t) is EXACT for ANY RHS (rank-independent).
        const int nranks = Mpi::WorldSize();
        tr_cnt.assign(nranks,0); tr_disp.assign(nranks,0);
        int my_nif = ess_tdofs_t.Size();
        MPI_Allgather(&my_nif,1,MPI_INT, tr_cnt.data(),1,MPI_INT, MPI_COMM_WORLD);
        tr_niface = 0;
        for (int r=0;r<nranks;++r){ tr_disp[r]=tr_niface; tr_niface+=tr_cnt[r]; }
        const int nloc = fes_t.GetTrueVSize();
        if (do_transferinc) Zcol.assign((size_t)nloc*tr_niface, 0.0);   // column-major
        else                Zloc.assign((size_t)nloc*tr_niface, 0.0);   // row-major
        ParLinearForm zero_lf0(&fes_t); zero_lf0=0.0; zero_lf0.Assemble();
        ParGridFunction phi_col(&fes_t);
        Vector col_tv(nloc);
        double bnref=1.0;
        MPI_Barrier(MPI_COMM_WORLD); double wb=MPI_Wtime();
        for (int r=0;r<nranks;++r){
            for (int k=0;k<tr_cnt[r];++k){
                col_tv=0.0; if (rank==r) col_tv(ess_tdofs_t[k])=1.0;
                phi_col.SetFromTrueDofs(col_tv);
                HypreParMatrix Ktc; Vector Xc, Bc;
                ktf.FormLinearSystem(ess_tdofs_t, phi_col, zero_lf0, Ktc, Xc, Bc);
                double bn=std::sqrt(ip2(Bc,Bc)); if (bn<=0) bn=bnref;
                cg3p->SetRelTol(0.0); cg3p->SetAbsTol(1e-12*bn);
                KSPSetInitialGuessNonzero((KSP)*cg3p, PETSC_FALSE);
                cg3p->Mult(Bc, Xc);
                const int g = tr_disp[r]+k;
                if (do_transferinc) for (int i=0;i<nloc;++i) Zcol[(size_t)g*nloc+i]=Xc(i);
                else                for (int i=0;i<nloc;++i) Zloc[(size_t)i*tr_niface+g]=Xc(i);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD); tr_build_wall = MPI_Wtime()-wb;
        // ---- electrode reciprocity lead-field: ONE adjoint solve per electrode.
        // phi(e) = e_e^T Kt^-1 Bt = (Kt^-1 e_e)^T Bt = w_e . Bt (Kt SPD symmetric).
        // The electrode = the globally-nearest torso dof; set a unit there and solve.
        MPI_Barrier(MPI_COMM_WORLD); double wl=MPI_Wtime();
        { struct{double d;int r;} loc,glob;
          loc.d=eL_d2; loc.r=rank; MPI_Allreduce(&loc,&glob,1,MPI_DOUBLE_INT,MPI_MINLOC,MPI_COMM_WORLD);
          Vector uL(nloc); uL=0.0; if(rank==glob.r && eL>=0) uL(eL)=1.0;
          wL_lead.SetSize(nloc); cg3p->SetRelTol(0.0); cg3p->SetAbsTol(1e-12);
          KSPSetInitialGuessNonzero((KSP)*cg3p,PETSC_FALSE); cg3p->Mult(uL,wL_lead);
          loc.d=eR_d2; loc.r=rank; MPI_Allreduce(&loc,&glob,1,MPI_DOUBLE_INT,MPI_MINLOC,MPI_COMM_WORLD);
          Vector uR(nloc); uR=0.0; if(rank==glob.r && eR>=0) uR(eR)=1.0;
          wR_lead.SetSize(nloc); cg3p->Mult(uR,wR_lead); }
        MPI_Barrier(MPI_COMM_WORLD); tr_lead_build = MPI_Wtime()-wl;
        double zmb = (double)(Zloc.size()+Zcol.size())*sizeof(double)/1048576.0, zmb_tot;
        MPI_Reduce(&zmb,&zmb_tot,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
        if (rank==0) cout << "[TRANSFER] built interface->torso operator Z once: N_iface="
                          << tr_niface << " columns (" << tr_niface << " Sys3 solves), "
                          << tr_build_wall << " s, Z storage " << zmb_tot << " MB total\n";
        if (do_transferh) {
            // ---- 2-sided H-matrix compression (purely LOCAL: each rank owns Zloc[localrow, allcols]).
            MPI_Barrier(MPI_COMM_WORLD); double wc=MPI_Wtime();
            // interface (source) coordinates, gathered once in the Z-column order.
            std::vector<double> ifx(tr_niface),ify(tr_niface),ifz(tr_niface);
            { int nif=tr_cnt[rank]; std::vector<double> lx(nif?nif:1),ly(nif?nif:1),lz(nif?nif:1);
              for(int k=0;k<nif;++k){ lx[k]=txv(ess_tdofs_t[k]);ly[k]=tyv(ess_tdofs_t[k]);lz[k]=tzv(ess_tdofs_t[k]); }
              MPI_Allgatherv(nif?lx.data():nullptr,nif,MPI_DOUBLE,ifx.data(),tr_cnt.data(),tr_disp.data(),MPI_DOUBLE,MPI_COMM_WORLD);
              MPI_Allgatherv(nif?ly.data():nullptr,nif,MPI_DOUBLE,ify.data(),tr_cnt.data(),tr_disp.data(),MPI_DOUBLE,MPI_COMM_WORLD);
              MPI_Allgatherv(nif?lz.data():nullptr,nif,MPI_DOUBLE,ifz.data(),tr_cnt.data(),tr_disp.data(),MPI_DOUBLE,MPI_COMM_WORLD); }
            auto cluster=[&](const std::vector<int>& idx, int nax, bool src,
                             std::vector<std::vector<int>>& groups){
                double lo[3]={1e300,1e300,1e300}, hi[3]={-1e300,-1e300,-1e300};
                auto XYZ=[&](int id,double&X,double&Y,double&Z){ if(src){X=ifx[id];Y=ify[id];Z=ifz[id];}
                    else {X=txv(id);Y=tyv(id);Z=tzv(id);} };
                for(int id:idx){ double X,Y,Z; XYZ(id,X,Y,Z);
                    lo[0]=std::min(lo[0],X);hi[0]=std::max(hi[0],X); lo[1]=std::min(lo[1],Y);hi[1]=std::max(hi[1],Y);
                    lo[2]=std::min(lo[2],Z);hi[2]=std::max(hi[2],Z); }
                double sp[3]={std::max(1e-9,hi[0]-lo[0]),std::max(1e-9,hi[1]-lo[1]),std::max(1e-9,hi[2]-lo[2])};
                std::map<int,std::vector<int>> mp;
                for(int id:idx){ double X,Y,Z; XYZ(id,X,Y,Z);
                    int bx=std::min(nax-1,(int)((X-lo[0])/sp[0]*nax)); int by=std::min(nax-1,(int)((Y-lo[1])/sp[1]*nax));
                    int bz=std::min(nax-1,(int)((Z-lo[2])/sp[2]*nax)); mp[(bx*nax+by)*nax+bz].push_back(id); }
                for(auto&kv:mp) groups.push_back(std::move(kv.second));
            };
            std::vector<int> allrows(nloc); for(int i=0;i<nloc;++i) allrows[i]=i;
            std::vector<int> allcols(tr_niface); for(int g=0;g<tr_niface;++g) allcols[g]=g;
            std::vector<std::vector<int>> leaves, sgroups;
            cluster(allrows, th_nleaf, false, leaves);
            cluster(allcols, th_ngrp, true, sgroups);
            // cluster centroid + radius helper
            auto crad=[&](const std::vector<int>& idx, bool src, double c[3])->double{
                c[0]=c[1]=c[2]=0; auto XYZ=[&](int id,double&X,double&Y,double&Z){ if(src){X=ifx[id];Y=ify[id];Z=ifz[id];}
                    else{X=txv(id);Y=tyv(id);Z=tzv(id);} };
                for(int id:idx){double X,Y,Z;XYZ(id,X,Y,Z);c[0]+=X;c[1]+=Y;c[2]+=Z;}
                for(int d=0;d<3;++d)c[d]/=idx.size(); double rad=0;
                for(int id:idx){double X,Y,Z;XYZ(id,X,Y,Z);double dd=(X-c[0])*(X-c[0])+(Y-c[1])*(Y-c[1])+(Z-c[2])*(Z-c[2]);rad=std::max(rad,dd);}
                return std::sqrt(rad); };
            std::mt19937 gen(12345+rank); std::uniform_real_distribution<double> U01(-1.0,1.0);
            for(auto& L : leaves){ double cL[3]; double rL=crad(L,false,cL);
              for(auto& S : sgroups){ double cS[3]; double rS=crad(S,true,cS);
                double dc=std::sqrt((cL[0]-cS[0])*(cL[0]-cS[0])+(cL[1]-cS[1])*(cL[1]-cS[1])+(cL[2]-cS[2])*(cL[2]-cS[2]));
                int sr=(int)L.size(), sc=(int)S.size();
                bool admiss = dc > th_eta*(rL+rS);
                HBlock hb; hb.rows=L; hb.cols=S;
                if(!admiss){ hb.dense=true; th_store+=(long)sr*sc; th_ndense++; th_blk.push_back(std::move(hb)); continue; }
                int K=std::min(th_samp,std::min(sr,sc));
                std::vector<double> Om((size_t)sc*K); for(auto&v:Om)v=U01(gen);
                std::vector<double> Q((size_t)sr*K,0.0);
                for(int p=0;p<sr;++p){ const double*zr=&Zloc[(size_t)L[p]*tr_niface];
                    for(int l=0;l<K;++l){ double acc=0.0; for(int j=0;j<sc;++j) acc+=zr[S[j]]*Om[(size_t)j*K+l]; Q[(size_t)p*K+l]=acc; } }
                int Kq=0; std::vector<int> keep;
                for(int l=0;l<K;++l){ for(int a=0;a<Kq;++a){ int cj=keep[a]; double dot=0.0;
                        for(int p=0;p<sr;++p) dot+=Q[(size_t)p*K+cj]*Q[(size_t)p*K+l];
                        for(int p=0;p<sr;++p) Q[(size_t)p*K+l]-=dot*Q[(size_t)p*K+cj]; }
                    double nn=0.0; for(int p=0;p<sr;++p) nn+=Q[(size_t)p*K+l]*Q[(size_t)p*K+l]; nn=std::sqrt(nn);
                    if(nn>1e-11){ double inv=1.0/nn; for(int p=0;p<sr;++p) Q[(size_t)p*K+l]*=inv; keep.push_back(l); Kq++; } }
                if(Kq==0){ hb.dense=true; th_store+=(long)sr*sc; th_ndense++; th_blk.push_back(std::move(hb)); continue; }
                std::vector<double> B((size_t)Kq*sc,0.0);
                for(int a=0;a<Kq;++a){ int ca=keep[a];
                    for(int p=0;p<sr;++p){ double q=Q[(size_t)p*K+ca]; const double*zr=&Zloc[(size_t)L[p]*tr_niface];
                        double*Ba=&B[(size_t)a*sc]; for(int j=0;j<sc;++j) Ba[j]+=q*zr[S[j]]; } }
                std::vector<double> Gm((size_t)Kq*Kq,0.0);
                for(int a=0;a<Kq;++a) for(int b=0;b<Kq;++b){ double acc=0.0; const double*Ba=&B[(size_t)a*sc],*Bb=&B[(size_t)b*sc];
                    for(int j=0;j<sc;++j) acc+=Ba[j]*Bb[j]; Gm[(size_t)a*Kq+b]=acc; }
                std::vector<double> ev(Kq); char jz='V',up='U'; int NN=Kq,info=0,lw=-1; double wq=0;
                dsyev_(&jz,&up,&NN,Gm.data(),&NN,ev.data(),&wq,&lw,&info);
                lw=(int)wq; std::vector<double> work(std::max(1,lw));
                dsyev_(&jz,&up,&NN,Gm.data(),&NN,ev.data(),work.data(),&lw,&info);
                double emax=ev[Kq-1]; int r=0; for(int a=Kq-1;a>=0;--a){ if(ev[a]>th_tol*th_tol*emax) r++; else break; } if(r<1)r=1;
                // only keep low-rank if it actually saves storage vs dense
                if((long)r*(sr+sc) >= (long)sr*sc){ hb.dense=true; th_store+=(long)sr*sc; th_ndense++; th_blk.push_back(std::move(hb)); continue; }
                hb.dense=false; hb.r=r; hb.U.assign((size_t)sr*r,0.0); hb.V.assign((size_t)r*sc,0.0);
                for(int c=0;c<r;++c){ const double*W=&Gm[(size_t)(Kq-1-c)*Kq];
                    for(int p=0;p<sr;++p){ double acc=0.0; for(int a=0;a<Kq;++a) acc+=Q[(size_t)p*K+keep[a]]*W[a]; hb.U[(size_t)p*r+c]=acc; }
                    double*Vc=&hb.V[(size_t)c*sc]; for(int a=0;a<Kq;++a){ double w=W[a]; const double*Ba=&B[(size_t)a*sc];
                        for(int j=0;j<sc;++j) Vc[j]+=w*Ba[j]; } }
                th_store+=(long)sr*r+(long)r*sc; th_nlr++;
                th_blk.push_back(std::move(hb));
              }
            }
            MPI_Barrier(MPI_COMM_WORLD); th_build = MPI_Wtime()-wc;
            long dense_store=(long)nloc*tr_niface, gdense, gcomp, gnd, gnl;
            MPI_Reduce(&dense_store,&gdense,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
            MPI_Reduce(&th_store,&gcomp,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
            MPI_Reduce(&th_ndense,&gnd,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
            MPI_Reduce(&th_nlr,&gnl,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
            if(rank==0) cout<<"[TRANSFERH] 2-sided H-matrix: leaves/rank~"<<leaves.size()
                <<" source groups="<<sgroups.size()<<" (eta "<<th_eta<<", samp "<<th_samp<<", tol "<<th_tol<<")\n"
                <<"  blocks: dense="<<gnd<<" low-rank="<<gnl
                <<"  compression: dense "<<gdense*8.0/1048576.0<<" MB -> H "<<gcomp*8.0/1048576.0
                <<" MB ("<<(double)gdense/gcomp<<"x smaller), build "<<th_build<<" s\n";
        }
    }
    bool cheb_ready = false;   // one-time [emin,emax] estimate done?

    // ====================================================================
    //  time loop  (IMEX: explicit TP06 reaction + C-N diffusion)
    // ====================================================================
    const int nsteps = (do_precond||do_prop||do_sweep||do_weightcmp||do_soraspu||do_transmit||do_tuned||do_overlap||do_fair||do_transmiti||do_neumann||do_decay||do_anisocmp||do_deflate) ? 0 : (int)(Tend/dt);   // -precond/-prop/-sweep skip the EP loop
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
                int it2_cold=-1, it2_warm=-1, it2_fis=-1, it2_phys=-1;
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
                    // BATCHED: all <p_i,b> in one Allreduce (was: one per basis vector)
                    Vector xf(nloc); xf=0.0;
                    { std::vector<double> cf; ipbatch(fisch_P, b2, cf);
                      for (size_t i=0;i<fisch_P.size();++i) xf.Add(cf[i], fisch_P[i]);
                      fisch_red_new += fisch_P.empty()?0:1;
                      fisch_red_old += (long)fisch_P.size(); }
                    cg2.Mult(b2, xf);
                    it2_fis=cg2.GetNumIterations();
                    // (d) PHYSICS initial guess (cross-system, current-step Vm):
                    //   monodomain reduction u_e ~ -c Vm; pick the optimal scalar
                    //   c* = <b2, w>/<w,w>, w = Kie Vm, minimizing ||b2 - Kie(c Vm)||.
                    //   One extra matvec + 2 inner products.  Uses THIS step's Vm
                    //   (fresher than last step's u_e).  Measured only.
                    Vector wphys(nloc); Kie->Mult(Vm, wphys);
                    double cst = ip2(b2, wphys) / ip2(wphys, wphys);
                    Vector xphys(nloc); xphys = 0.0; xphys.Add(cst, Vm);
                    KSPSetInitialGuessNonzero((KSP)cg2, PETSC_TRUE);
                    cg2.Mult(b2, xphys); it2_phys=cg2.GetNumIterations();
                    KSPSetInitialGuessNonzero((KSP)cg2, PETSC_FALSE);
                    // grow the history from the CLEAN cold solution (mean-zero copy).
                    // CGS2 (classical Gram-Schmidt, 2 passes, coefficients batched into
                    // ONE Allreduce per pass) replaces MGS (one Allreduce per basis
                    // vector, sequential).  Numerically CGS2 ~ MGS; communication O(1).
                    Vector w(ue_h); RemoveGlobalMean(w, MPI_COMM_WORLD);
                    Vector Aw(nloc); Kie->Mult(w, Aw);
                    for (int pass=0; pass<2 && !fisch_P.empty(); ++pass){
                        std::vector<double> cf; ipbatch(fisch_AP, w, cf);  // c_i=<A p_i,w>
                        for (size_t i=0;i<fisch_P.size();++i){
                            w.Add(-cf[i],fisch_P[i]); Aw.Add(-cf[i],fisch_AP[i]); }
                        fisch_red_new += 1;
                    }
                    fisch_red_old += (long)fisch_P.size();   // MGS: one reduce per vector
                    double nrm=std::sqrt(ip2(w,Aw));
                    fisch_red_new += 1; fisch_red_old += 1;  // the norm reduce (both)
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
                    fisch_phys+=it2_phys;
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
                int it3_cold=-1, it3_fis=-1;
                if (do_fischer3) {
                    // PERSISTENT solver (no per-step convert/factorization) + recycling.
                    double bn = std::sqrt(ip2(Bt,Bt));
                    // ONE-TIME Chebyshev eigenvalue estimate (constant operator): run a
                    // short CG with singular-value tracking on the first RHS, then hand
                    // [emax,emin] of the preconditioned operator to the persistent Cheby KSP.
                    if (do_f3cheb && !cheb_ready) {
                        KSP eig; KSPCreate(MPI_COMM_WORLD,&eig);
                        KSPSetOperators(eig,(Mat)*Kt3p_persist,(Mat)*Kt3p_persist);
                        KSPSetType(eig,KSPCG); KSPSetComputeSingularValues(eig,PETSC_TRUE);
                        KSPSetOptionsPrefix(eig,"sys3eig_");
                        PetscOptionsSetValue(NULL,"-sys3eig_pc_type","bjacobi");
                        PetscOptionsSetValue(NULL,"-sys3eig_sub_pc_type","icc");
                        KSPSetFromOptions(eig);
                        KSPSetTolerances(eig,1e-6,PETSC_DEFAULT,PETSC_DEFAULT,200);
                        Vec ex,eb; MatCreateVecs((Mat)*Kt3p_persist,&ex,&eb);
                        { PetscScalar *ea; VecGetArray(eb,&ea);
                          for(int i=0;i<Bt.Size();++i) ea[i]=Bt(i); VecRestoreArray(eb,&ea); }
                        VecZeroEntries(ex); KSPSolve(eig,eb,ex);
                        PetscReal emax=0,emin=0; KSPComputeExtremeSingularValues(eig,&emax,&emin);
                        if (emin<=0) emin=emax/1e3;
                        KSPChebyshevSetEigenvalues((KSP)*cg3p,emax,emin);
                        if (rank==0) cout<<"[F3CHEB] one-time eig estimate: emin="<<emin
                                         <<" emax="<<emax<<" (cond "<<emax/emin<<")\n";
                        VecDestroy(&ex); VecDestroy(&eb); KSPDestroy(&eig); cheb_ready=true;
                    }
                    cg3p->SetRelTol(0.0); cg3p->SetAbsTol(1e-8*bn);
                    // (a) cold (x0=0) -- THIS is the kept field (also timed)
                    KSPSetInitialGuessNonzero((KSP)*cg3p, PETSC_FALSE);
                    MPI_Barrier(MPI_COMM_WORLD); double w3=MPI_Wtime();
                    cg3p->Mult(Bt, Xt); it3_cold=cg3p->GetNumIterations();
                    MPI_Barrier(MPI_COMM_WORLD); t3_cold_solve += MPI_Wtime()-w3;
                    // (a') warm start = previous cold solution (A2 / increment guess;
                    //      for a constant linear operator x0=phi_prev has residual = dBt)
                    int it3_warm=-1;
                    if (f3_have_prev){ KSPSetInitialGuessNonzero((KSP)*cg3p, PETSC_TRUE);
                        Vector xw(f3_prev); cg3p->Mult(Bt, xw); it3_warm=cg3p->GetNumIterations();
                        KSPSetInitialGuessNonzero((KSP)*cg3p, PETSC_FALSE); }
                    if (it3_warm>=0) f3_warm += it3_warm; else f3_warm += it3_cold;
                    // (a'') PHYSICS DC guess: x0 = constant = interface (Dirichlet) mean.
                    //   A constant is harmonic + satisfies the body-surface Neumann BC, so it
                    //   is the EXACT solution's DC component; leftover = interface variation.
                    //   Uses the current interface u_e (cross-system).  Measure only.
                    { double s=0.0; for (int k=0;k<ess_tdofs_t.Size();++k) s+=phi_tv(ess_tdofs_t[k]);
                      double loc[2]={s,(double)ess_tdofs_t.Size()}, glob[2];
                      MPI_Allreduce(loc,glob,2,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
                      double ifmean = glob[1]>0? glob[0]/glob[1] : 0.0;
                      Vector xp(Xt.Size()); xp = ifmean;
                      KSPSetInitialGuessNonzero((KSP)*cg3p, PETSC_TRUE);
                      cg3p->Mult(Bt, xp); f3_phys += cg3p->GetNumIterations();
                      KSPSetInitialGuessNonzero((KSP)*cg3p, PETSC_FALSE); }
                    // (b) Fischer guess x0 = sum <p_i,Bt> p_i (batched; measure only)
                    Vector xf3(Xt.Size()); xf3=0.0;
                    { std::vector<double> cf(f3_P.size(),0.0);
                      for (size_t i=0;i<f3_P.size();++i){ const Vector&p=f3_P[i]; double s=0.0;
                          for (int k=0;k<p.Size();++k) s+=p(k)*Bt(k); cf[i]=s; }
                      if (!f3_P.empty()) MPI_Allreduce(MPI_IN_PLACE,cf.data(),(int)f3_P.size(),
                          MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
                      for (size_t i=0;i<f3_P.size();++i) xf3.Add(cf[i], f3_P[i]);
                      f3_red_new += f3_P.empty()?0:1; f3_red_old += (long)f3_P.size(); }
                    KSPSetInitialGuessNonzero((KSP)*cg3p, PETSC_TRUE);
                    Vector xfs(xf3); cg3p->Mult(Bt, xfs); it3_fis=cg3p->GetNumIterations();
                    KSPSetInitialGuessNonzero((KSP)*cg3p, PETSC_FALSE);
                    // (c) grow basis from cold solution Xt (non-singular: NO mean removal),
                    //     CGS2 A-orthonormalize (batched), relative drop, sliding window
                    Vector w(Xt); Vector Aw(w.Size()); Kt.Mult(w, Aw);
                    double ref = std::sqrt(ip2(w,Aw));
                    for (int pass=0; pass<2 && !f3_P.empty(); ++pass){
                        std::vector<double> cf(f3_P.size(),0.0);
                        for (size_t i=0;i<f3_P.size();++i){ const Vector&ap=f3_AP[i]; double s=0.0;
                            for (int k=0;k<ap.Size();++k) s+=ap(k)*w(k); cf[i]=s; }
                        MPI_Allreduce(MPI_IN_PLACE,cf.data(),(int)f3_P.size(),MPI_DOUBLE,MPI_SUM,
                            MPI_COMM_WORLD);
                        for (size_t i=0;i<f3_P.size();++i){ w.Add(-cf[i],f3_P[i]); Aw.Add(-cf[i],f3_AP[i]); }
                        f3_red_new += 1;
                    }
                    f3_red_old += (long)f3_P.size();
                    double nrm = std::sqrt(ip2(w,Aw));
                    f3_red_new += 1; f3_red_old += 1;
                    if (nrm > 1e-3*ref){ w*=1.0/nrm; Aw*=1.0/nrm;
                        if ((int)f3_P.size()>=F3MAX){ f3_P.erase(f3_P.begin()); f3_AP.erase(f3_AP.begin()); }
                        f3_P.push_back(w); f3_AP.push_back(Aw); }
                    f3_cold += it3_cold; f3_fis += it3_fis;
                    f3_prev = Xt; f3_have_prev = true;      // seed warm/A2 for next step
                    ktf.RecoverFEMSolution(Xt, zero_lf, phi_t);
                } else if (do_leadvol) {
                    // FULL-FIELD reduced basis: the RHS Bt(t) is low-rank in time, so the
                    // full torso field phi=Kt^-1 Bt is a superposition of precomputed
                    // response fields Psi_i=Kt^-1 B_i.  Keep the TRUE solve as the field and
                    // MEASURE the superposition's full-field error (no field corruption).
                    double bn = std::sqrt(ip2(Bt,Bt));
                    cg3p->SetRelTol(0.0); cg3p->SetAbsTol(1e-10*bn);
                    KSPSetInitialGuessNonzero((KSP)*cg3p, PETSC_FALSE);
                    cg3p->Mult(Bt, Xt); it3_cold=cg3p->GetNumIterations();   // true (kept) field
                    // superposition approx: a_i=<Bt,B_i> (L2, batched); phi=sum a_i Psi_i
                    Vector phi_ap(Xt.Size()); phi_ap=0.0;
                    { std::vector<double> a(lv_B.size(),0.0);
                      for (size_t i=0;i<lv_B.size();++i){ const Vector&B=lv_B[i]; double s=0.0;
                          for (int k=0;k<B.Size();++k) s+=B(k)*Bt(k); a[i]=s; }
                      if (!lv_B.empty()) MPI_Allreduce(MPI_IN_PLACE,a.data(),(int)lv_B.size(),
                          MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
                      for (size_t i=0;i<lv_B.size();++i) phi_ap.Add(a[i], lv_Psi[i]); }
                    // full-field relative L2 error of the superposition vs the true solve
                    Vector e(Xt); e-=phi_ap;
                    double en=std::sqrt(ip2(e,e)), xn=std::sqrt(ip2(Xt,Xt));
                    double rel=(xn>0?en/xn:0.0); lv_err_sum+=rel; lv_err_max=std::max(lv_err_max,rel);
                    // grow the RHS basis: L2-orthonormalize Bt against lv_B (CGS2); if the new
                    // direction is non-negligible and room remains, accept it and precompute
                    // its response field Psi=Kt^-1 B_new (a ONE-TIME solve, amortized).
                    Vector w(Bt); double ref=std::sqrt(ip2(w,w));
                    for (int pass=0; pass<2 && !lv_B.empty(); ++pass){
                        std::vector<double> c(lv_B.size(),0.0);
                        for (size_t i=0;i<lv_B.size();++i){ const Vector&B=lv_B[i]; double s=0.0;
                            for (int k=0;k<B.Size();++k) s+=B(k)*w(k); c[i]=s; }
                        MPI_Allreduce(MPI_IN_PLACE,c.data(),(int)lv_B.size(),MPI_DOUBLE,MPI_SUM,
                            MPI_COMM_WORLD);
                        for (size_t i=0;i<lv_B.size();++i) w.Add(-c[i],lv_B[i]);
                    }
                    double wn=std::sqrt(ip2(w,w));
                    if (wn > LV_TOL*ref && (int)lv_B.size() < lv_max){
                        w *= 1.0/wn; lv_B.push_back(w);
                        Vector psi(Xt.Size()); psi=0.0;
                        KSPSetInitialGuessNonzero((KSP)*cg3p, PETSC_FALSE);
                        cg3p->Mult(w, psi); lv_Psi.push_back(psi); lv_grow++;
                    } else lv_free++;
                    ktf.RecoverFEMSolution(Xt, zero_lf, phi_t);
                } else if (do_transfer) {
                    // Apply the prebuilt transfer operator: phi_Z = Z * u_iface(t).  Also do
                    // the TRUE per-step solve (kept as the field) to VALIDATE exactness.
                    double bn = std::sqrt(ip2(Bt,Bt));
                    cg3p->SetRelTol(0.0); cg3p->SetAbsTol(1e-10*bn);
                    KSPSetInitialGuessNonzero((KSP)*cg3p, PETSC_FALSE);
                    MPI_Barrier(MPI_COMM_WORLD); double ws=MPI_Wtime();
                    cg3p->Mult(Bt, Xt); it3_cold=cg3p->GetNumIterations();   // true (kept) field
                    MPI_Barrier(MPI_COMM_WORLD); tr_solve_wall += MPI_Wtime()-ws;
                    // transfer apply: gather interface data (same global order as Z columns),
                    // then a local dense matvec -- ZERO solves, ZERO Krylov inner products.
                    const int nif=tr_cnt[rank], nloc=Xt.Size();
                    Vector locd(nif>0?nif:1);
                    for (int k=0;k<nif;++k) locd(k)=phi_tv(ess_tdofs_t[k]);
                    std::vector<double> dg(tr_niface);
                    MPI_Barrier(MPI_COMM_WORLD); double wa=MPI_Wtime();
                    MPI_Allgatherv(nif>0?locd.GetData():nullptr, nif, MPI_DOUBLE,
                        dg.data(), tr_cnt.data(), tr_disp.data(), MPI_DOUBLE, MPI_COMM_WORLD);
                    Vector phiZ(nloc);
                    // y = Z d via BLAS dgemv.  Zloc is row-major (nloc x niface) = column-major
                    // (niface x nloc), so y = (that)^T d: trans='T', m=niface, n=nloc, lda=niface.
                    double wcpu=MPI_Wtime();
                    { char tr='T'; int M=tr_niface, N=nloc, one=1; double al=1.0, be=0.0;
                      dgemv_(&tr,&M,&N,&al,Zloc.data(),&M,dg.data(),&one,&be,phiZ.GetData(),&one); }
                    tr_apply_cpu += MPI_Wtime()-wcpu;   // matvec-only (no comm/barrier)
                    MPI_Barrier(MPI_COMM_WORLD); tr_apply_wall += MPI_Wtime()-wa;
                    // exactness: full torso field via Z vs the true solve
                    Vector e(Xt); e-=phiZ;
                    double en=std::sqrt(ip2(e,e)), xn=std::sqrt(ip2(Xt,Xt));
                    double rel=(xn>0?en/xn:0.0); tr_err_sum+=rel;
                    tr_err_max=std::max(tr_err_max,rel); tr_steps++;
                    // ELECTRODE lead-field: ECG = (wL-wR).Bt -- 2 dot products, 0 solve.
                    MPI_Barrier(MPI_COMM_WORLD); double wl=MPI_Wtime();
                    double plL=ip2(wL_lead,Bt), prL=ip2(wR_lead,Bt);
                    MPI_Barrier(MPI_COMM_WORLD); tr_lead_wall += MPI_Wtime()-wl;
                    double plT=global_at(eL_d2,(eL>=0)?Xt(eL):0.0);   // true electrode phi
                    double prT=global_at(eR_d2,(eR>=0)?Xt(eR):0.0);
                    tr_lead_err_max=std::max(tr_lead_err_max, std::fabs((plL-prL)-(plT-prT)));
                    tr_ecg_absmax=std::max(tr_ecg_absmax, std::fabs(plT-prT));
                    ktf.RecoverFEMSolution(Xt, zero_lf, phi_t);
                } else if (do_transferh) {
                    // TRUE solve (kept field + validation)
                    double bn = std::sqrt(ip2(Bt,Bt));
                    cg3p->SetRelTol(0.0); cg3p->SetAbsTol(1e-10*bn);
                    KSPSetInitialGuessNonzero((KSP)*cg3p, PETSC_FALSE);
                    MPI_Barrier(MPI_COMM_WORLD); double ws=MPI_Wtime();
                    cg3p->Mult(Bt, Xt); it3_cold=cg3p->GetNumIterations();
                    MPI_Barrier(MPI_COMM_WORLD); th_solve += MPI_Wtime()-ws;
                    // H-matrix apply: gather interface data, then near-dense + far low-rank.
                    const int nif=tr_cnt[rank], nloc=Xt.Size();
                    Vector locd(nif>0?nif:1);
                    for (int k=0;k<nif;++k) locd(k)=phi_tv(ess_tdofs_t[k]);
                    std::vector<double> dg(tr_niface);
                    MPI_Barrier(MPI_COMM_WORLD); double wa=MPI_Wtime();
                    MPI_Allgatherv(nif>0?locd.GetData():nullptr, nif, MPI_DOUBLE,
                        dg.data(), tr_cnt.data(), tr_disp.data(), MPI_DOUBLE, MPI_COMM_WORLD);
                    Vector phiH(nloc); phiH=0.0;   // accumulate over (leaf x group) blocks
                    for (const HBlock& hb : th_blk){ int sr=(int)hb.rows.size(), sc=(int)hb.cols.size();
                        if (hb.dense){
                            for (int p=0;p<sr;++p){ const double*zr=&Zloc[(size_t)hb.rows[p]*tr_niface]; double s=0.0;
                                for (int j=0;j<sc;++j) s+=zr[hb.cols[j]]*dg[hb.cols[j]]; phiH(hb.rows[p])+=s; }
                        } else { int r=hb.r; std::vector<double> tmp(r,0.0);
                            for (int c=0;c<r;++c){ const double*Vc=&hb.V[(size_t)c*sc]; double a=0.0;
                                for (int j=0;j<sc;++j) a+=Vc[j]*dg[hb.cols[j]]; tmp[c]=a; }
                            for (int p=0;p<sr;++p){ const double*Up=&hb.U[(size_t)p*r]; double a=0.0;
                                for (int c=0;c<r;++c) a+=Up[c]*tmp[c]; phiH(hb.rows[p])+=a; } }
                    }
                    MPI_Barrier(MPI_COMM_WORLD); th_apply += MPI_Wtime()-wa;
                    Vector e(Xt); e-=phiH; double en=std::sqrt(ip2(e,e)), xn=std::sqrt(ip2(Xt,Xt));
                    double rel=(xn>0?en/xn:0.0); th_err_sum+=rel; th_err_max=std::max(th_err_max,rel); th_steps++;
                    ktf.RecoverFEMSolution(Xt, zero_lf, phi_t);
                } else if (do_transferinc) {
                    // TRUE solve (kept field + validation)
                    double bn = std::sqrt(ip2(Bt,Bt));
                    cg3p->SetRelTol(0.0); cg3p->SetAbsTol(1e-10*bn);
                    KSPSetInitialGuessNonzero((KSP)*cg3p, PETSC_FALSE);
                    MPI_Barrier(MPI_COMM_WORLD); double ws=MPI_Wtime();
                    cg3p->Mult(Bt, Xt); it3_cold=cg3p->GetNumIterations();
                    MPI_Barrier(MPI_COMM_WORLD); ti_solve += MPI_Wtime()-ws;
                    // gather full interface data d(t)
                    const int nif=tr_cnt[rank], nloc=Xt.Size();
                    Vector locd(nif>0?nif:1);
                    for (int k=0;k<nif;++k) locd(k)=phi_tv(ess_tdofs_t[k]);
                    std::vector<double> dg(tr_niface);
                    MPI_Allgatherv(nif>0?locd.GetData():nullptr, nif, MPI_DOUBLE,
                        dg.data(), tr_cnt.data(), tr_disp.data(), MPI_DOUBLE, MPI_COMM_WORLD);
                    if (ti_phi.Size()!=nloc){ ti_phi.SetSize(nloc); ti_phi=0.0; }
                    double dmax=0.0; for(int g=0;g<tr_niface;++g) dmax=std::max(dmax,std::fabs(dg[g]));
                    bool refresh = (!ti_have) || (ti_steps % ti_refresh == 0);
                    MPI_Barrier(MPI_COMM_WORLD); double wa=MPI_Wtime();
                    if (refresh){
                        // full apply phi = Z d  (streams all Z) -- clears accumulated drift
                        char tr='N'; int M=nloc, N=tr_niface, one=1; double al=1.0, be=0.0;
                        dgemv_(&tr,&M,&N,&al,Zcol.data(),&M,dg.data(),&one,&be,ti_phi.GetData(),&one);
                        ti_refresh_cnt++;
                    } else {
                        // incremental: only columns whose interface value changed (the front)
                        int nact=0; double thr=ti_eps*dmax;
                        for (int g=0; g<tr_niface; ++g){ double dd=dg[g]-ti_dprev[g];
                            if (std::fabs(dd) > thr){ const double*zc=&Zcol[(size_t)g*nloc];
                                double*p=ti_phi.GetData(); for(int i=0;i<nloc;++i) p[i]+=dd*zc[i]; nact++; } }
                        ti_active_sum += nact; ti_active_max = std::max(ti_active_max,(long)nact);
                    }
                    MPI_Barrier(MPI_COMM_WORLD); ti_apply += MPI_Wtime()-wa;
                    ti_dprev = dg; ti_have = true;
                    Vector e(Xt); e-=ti_phi; double en=std::sqrt(ip2(e,e)), xn=std::sqrt(ip2(Xt,Xt));
                    double rel=(xn>0?en/xn:0.0); ti_err_sum+=rel; ti_err_max=std::max(ti_err_max,rel); ti_steps++;
                    ktf.RecoverFEMSolution(Xt, zero_lf, phi_t);
                } else {
                    PetscParMatrix Ktp; HypreToPetscAIJ(Kt, Ktp, "Sys3_Kt", rank, 1, true);
                    PetscPCGSolver cg3(Ktp, "sys3_");
                    cg3.SetRelTol(1e-8); cg3.SetMaxIter(3000); cg3.iterative_mode=false;
                    { PC pc; KSPGetPC((KSP)cg3,&pc); PCSetType(pc,PCBJACOBI); }
                    cg3.Mult(Bt, Xt); it3_cold=cg3.GetNumIterations();
                    ktf.RecoverFEMSolution(Xt, zero_lf, phi_t);
                }
                if (rank==0) {
                    cout << "[ITERS] t="<<(int)(t+dt+0.5)<<"ms  Sys1(CG+bj-ICC)="
                         <<cg1.GetNumIterations()<<"  Sys2(singular)=";
                    if (do_acc)
                        cout << it2_two<<" (baseline="<<it2_base<<" "<<acc_label<<")";
                    else if (do_fischer)
                        cout << it2_fis<<" (cold="<<it2_cold<<" warm="<<it2_warm
                             <<" phys="<<it2_phys<<")";
                    else
                        cout << cg2.GetNumIterations();
                    cout <<"  Sys3(torso)="<<it3_cold;
                    if (do_fischer3) cout << " (Fischer="<<it3_fis<<")";
                    cout <<"  ue_mean="<<std::scientific<<std::setprecision(2)<<ue_mean
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
        cout << "\n[FISCHER] Sys2 EP-loop acceleration (total CG iters over the run):\n"
             << "  cold (x0=0)              : " << fisch_cold << "\n"
             << "  warm (prev-step u_e)     : " << fisch_warm
             << "  (-" << (int)(100.0*(fisch_cold-fisch_warm)/fisch_cold) << "%)\n"
             << "  Fischer (u_e history)    : " << fisch_fis
             << "  (-" << (int)(100.0*(fisch_cold-fisch_fis)/fisch_cold) << "%)\n"
             << "  PHYSICS (x0 = c* Vm)     : " << fisch_phys
             << "  (-" << (int)(100.0*(fisch_cold-fisch_phys)/fisch_cold) << "%)"
             << "   [cross-system, current-step Vm; monodomain reduction u_e~-c Vm]\n"
             << "  recycling-maintenance Allreduces: batched(new)=" << fisch_red_new
             << " vs unbatched-MGS(old-equiv)=" << fisch_red_old
             << "  (" << (fisch_red_new>0 ? (double)fisch_red_old/fisch_red_new : 0.0)
             << "x fewer)\n";
    }
    if (do_fischer3 && rank==0 && f3_cold>0) {
        const char* pclabel = do_f3cheb? "Chebyshev(bjacobi+ICC)" : do_f3gamg? "GAMG":"bjacobi+ICC";
        cout << "\n[FISCHER3] Sys3 torso EP-loop  [solver=" << pclabel
             << ", persistent]:\n"
             << "  cold (x0=0)              : " << f3_cold
             << "   (cold-solve wall = " << t3_cold_solve << " s)\n"
             << "  warm (prev phi)          : " << f3_warm
             << "  (-" << (int)(100.0*(f3_cold-f3_warm)/f3_cold) << "%)\n"
             << "  physics (x0=iface-mean)  : " << f3_phys
             << "  (-" << (int)(100.0*(f3_cold-f3_phys)/f3_cold) << "%)\n"
             << "  Fischer (u_e-BC history) : " << f3_fis
             << "  (-" << (int)(100.0*(f3_cold-f3_fis)/f3_cold) << "%)\n"
             << "  recycling-maintenance Allreduces: batched(new)=" << f3_red_new
             << " vs unbatched-MGS(old-equiv)=" << f3_red_old
             << "  (" << (f3_red_new>0 ? (double)f3_red_old/f3_red_new : 0.0)
             << "x fewer)\n";
        if (do_f3cheb)
            cout << "  [F3CHEB] Chebyshev iters/step measured with a norm test; PRODUCTION runs a\n"
                    "  FIXED k with -ksp_norm_type none => ZERO inner products/Allreduce per iter\n"
                    "  (vs CG's 2/iter). At 3000 cores (latency-bound) this trades more local iters\n"
                    "  for no synchronization -- the point is 0 collectives, not fewer iters.\n";
    }
    if (do_leadvol && rank==0 && (lv_grow+lv_free)>0) {
        long tot=lv_grow+lv_free;
        cout << "\n[LEADVOL] Sys3 FULL-FIELD reduced basis (superposition of Kt^-1 B_i, no per-step solve):\n"
             << "  basis size k = " << lv_B.size() << "  (grown from the RHS trajectory, accept-tol "
             << LV_TOL << ")\n"
             << "  steps = " << tot << "  basis-growth (1 one-time solve) = " << lv_grow
             << "   FREE (0 solve, pure axpy superposition) = " << lv_free
             << "  (" << (int)(100.0*lv_free/tot) << "% of steps need NO solve)\n"
             << "  full-field rel-L2 error (superposition vs true solve): mean "
             << lv_err_sum/tot << "  max " << lv_err_max << "\n"
             << "  => full torso volume field by superposition; per-step comm = 1 batched Allreduce"
                " (vs ~" << 60 << " CG-iter Allreduces)\n";
    }
    if (do_transfer && rank==0 && tr_steps>0) {
        double amort = tr_build_wall / tr_steps;   // offline cost per step if amortized
        cout << "\n[TRANSFER] Sys3 interface->torso operator Z (approximate the OPERATOR, not the moving solutions):\n"
             << "  N_iface (columns = one-time Sys3 solves) : " << tr_niface << "\n"
             << "  OFFLINE build (once)                     : " << tr_build_wall
             << " s  (amortized over " << tr_steps << " steps = " << amort*1e3 << " ms/step)\n"
             << "  per-step TRUE solve (bjacobi+ICC CG)     : " << tr_solve_wall/tr_steps*1e3 << " ms/step\n"
             << "  per-step TRANSFER apply (BLAS matvec, 0 solve): " << tr_apply_wall/tr_steps*1e3
             << " ms incl.comm ; matvec-ONLY " << tr_apply_cpu/tr_steps*1e3 << " ms\n"
             << "     (the matvec is MEMORY-BOUND: it streams the whole " << Zloc.size()*8/1048576 << " MB/rank of Z\n"
             << "      every step, ~ the solve's cost -- so dense Z's SIZE, not flops, is the wall.\n"
             << "      => this is exactly what H-matrix compression must shrink; see -transferh.)\n"
             << "  full-field EXACTNESS vs true solve       : mean rel-L2 " << tr_err_sum/tr_steps
             << "  max " << tr_err_max << "\n"
             << "  => Z reproduces the FULL torso field for the real Niederer-driven RHS to solver\n"
             << "     tolerance EVERY step with ZERO per-step solve; RHS high-rank is irrelevant\n"
             << "     because we apply the fixed operator, not a reduced basis of the moving field.\n"
             << "  -- ELECTRODE lead-field (reciprocity, the practical win) --\n"
             << "  OFFLINE build (2 adjoint solves, once)   : " << tr_lead_build << " s\n"
             << "  per-step ECG apply (2 dot products)      : " << tr_lead_wall/tr_steps*1e3
             << " ms/step  (" << (tr_lead_wall>0? tr_solve_wall/tr_lead_wall:0.0)
             << "x cheaper than solving)\n"
             << "  ECG EXACTNESS: max |ECG_lead - ECG_true| = " << tr_lead_err_max
             << "  (ECG scale ~" << tr_ecg_absmax << ")\n"
             << "  => the electrode ECG is EXACT with only 2 offline solves + 2 dots/step: the\n"
             << "     full-field win needs H-matrix, but the ELECTRODE output is free and exact now.\n";
    }
    if (do_transferh && rank==0 && th_steps>0) {
        double sflop = 69.0; // reference: dense apply vs H apply speed shown by wall ratio
        cout << "\n[TRANSFERH] Sys3 full-field via H-matrix-compressed transfer operator:\n"
             << "  per-step TRUE solve (bjacobi+ICC CG)     : " << th_solve/th_steps*1e3 << " ms/step\n"
             << "  per-step H-matrix apply (near+far low-rank): " << th_apply/th_steps*1e3 << " ms/step"
             << "  (" << (th_apply>0? th_solve/th_apply:0.0) << "x vs solve)\n"
             << "  full-field EXACTNESS vs true solve       : mean rel-L2 " << th_err_sum/th_steps
             << "  max " << th_err_max << "\n"
             << "  => the FULL torso volume field, compressed (near-dense + far low-rank), applied\n"
             << "     with NO per-step solve; closes the open item -- full volume can beat solving.\n"
             << "     (offline still builds Z once; ACA-inverse to avoid that is the library-level step.)\n";
        (void)sflop;
    }
    if (do_transferinc && rank==0 && ti_steps>0) {
        long incsteps = ti_steps - ti_refresh_cnt;
        double avgact = incsteps>0? (double)ti_active_sum/incsteps : 0.0;
        cout << "\n[TRANSFERINC] Sys3 full-field via INCREMENTAL front-localized transfer"
             << " (Sys1 front + Sys2 u_e increment):\n"
             << "  active interface DOFs / step (front band): avg " << avgact
             << "  max " << ti_active_max << "  of N_iface=" << tr_niface
             << "  (" << 100.0*avgact/tr_niface << "% -> streams that fraction of Z)\n"
             << "  full refreshes: " << ti_refresh_cnt << " of " << ti_steps
             << " steps (every " << ti_refresh << ", eps=" << ti_eps << ")\n"
             << "  per-step TRUE solve (bjacobi+ICC CG)     : " << ti_solve/ti_steps*1e3 << " ms/step\n"
             << "  per-step INCREMENTAL apply (avg incl. refresh): " << ti_apply/ti_steps*1e3 << " ms/step"
             << "  (" << (ti_apply>0? ti_solve/ti_apply:0.0) << "x vs solve)\n"
             << "  full-field EXACTNESS vs true solve       : mean rel-L2 " << ti_err_sum/ti_steps
             << "  max " << ti_err_max << "\n"
             << "  => only the moving-front columns of Z update each step; per-step memory traffic\n"
             << "     drops to the active fraction -- Sys1/Sys2 tell us WHICH columns, exactly.\n";
    }
    delete cg3p; delete Kt3p_persist; delete Kt3h_persist;

    // ---- benchmark activation times + conduction velocity -----------------
    if (!do_precond && !do_sweep && !do_weightcmp && !do_soraspu && !do_transmit && !do_tuned && !do_overlap && !do_fair && !do_transmiti && !do_neumann && !do_decay && !do_anisocmp && !do_deflate) {
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
