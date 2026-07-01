// soras_par.cpp -- PARALLEL one-level optimized Schwarz (SORAS-type) with
// 1 subdomain per MPI rank, as a drop-in preconditioner for the heart Sys2.
//
// Parallel realization of soras_demo:
//   * rank-local NEUMANN block  K_loc = a.SpMat()   (local element assembly on
//     the rank's L-dofs; at shared dofs only local element contributions ->
//     natural/Neumann BC on the inter-rank artificial interface).  FREE.
//   * real interface mass  M_Gamma  assembled on the SHARED faces
//     (pmesh.GetSharedFace / GetSharedFaceTransformations), so it carries the
//     interface-interface off-diagonal coupling that a lumped diagonal cannot.
//   * partition of unity  D = 1/multiplicity  (multiplicity from P^T 1).
//   * symmetric additive Schwarz, CG-valid:
//       M^{-1} = P^T D (K_loc + alpha M_Gamma)^{-1} D P
//     P = fes.GetProlongationMatrix() (T->L); P^T does the inter-rank sum
//     (this is the one communication per apply, same as a halo exchange).
//   local block solved with a serial PETSc ICC-CG on COMM_SELF (per rank).
//
//   mpirun -n 4 ./soras_par -mesh heart.msh -aniso -alpha 0.2
//   (sweep alpha; large alpha -> interface over-pinned ~ Dirichlet limit,
//    optimal alpha ~0.2 -> the optimized-transmission gain.)
#include "mfem.hpp"
#include "mfem_petsc_util.hpp"   // HypreToPetscAIJ
#include <petscksp.h>
#include <vector>
#include <cmath>
using namespace mfem;
using namespace std;

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

// SORAS preconditioner: M^{-1} = P^T D B_loc D P.
struct SORASPrec : public Solver {
    const Operator *P;        // T -> L prolongation
    Vector dL;                // partition-of-unity weight per L-dof
    mutable Vector rL, yL;    // work (L-space)
    KSP kloc; Vec rloc, zloc; // local Robin solve (COMM_SELF)
    SORASPrec(int tsize):Solver(tsize){}
    void SetOperator(const Operator&) override {}
    void Mult(const Vector &r, Vector &z) const override {
        P->Mult(r, rL);                 // L = P T
        for (int i=0;i<rL.Size();++i) rL(i) *= dL(i);
        PetscScalar *ra; VecGetArray(rloc,&ra);
        for (int i=0;i<rL.Size();++i) ra[i]=rL(i);
        VecRestoreArray(rloc,&ra);
        KSPSolve(kloc, rloc, zloc);     // (K_loc + alpha M_Gamma)^{-1}
        const PetscScalar *za; VecGetArrayRead(zloc,&za);
        for (int i=0;i<yL.Size();++i) yL(i)=za[i];
        VecRestoreArrayRead(zloc,&za);
        for (int i=0;i<yL.Size();++i) yL(i) *= dL(i);
        P->MultTranspose(yL, z);        // T = P^T L (inter-rank sum)
    }
};

int main(int argc, char *argv[])
{
    Mpi::Init(argc, argv); Hypre::Init();
    const int rank = Mpi::WorldRank(), nranks = Mpi::WorldSize();
    int nx = 16; double alpha = 0.2; const char *meshfile = nullptr;
    bool aniso=false, bjacobi=false, localicc0=false, localLU=false;
    int localcheby=0, sequence=0; double change=0.05;
    for (int i=1;i<argc;++i){ string a=argv[i];
        if (a=="-nx"&&i+1<argc) nx=atoi(argv[++i]);
        else if (a=="-alpha"&&i+1<argc) alpha=atof(argv[++i]);
        else if (a=="-mesh"&&i+1<argc) meshfile=argv[++i];
        else if (a=="-aniso") aniso=true;
        else if (a=="-bjacobi") bjacobi=true;      // baseline: block-Jacobi+ICC
        else if (a=="-localicc0") localicc0=true;
        else if (a=="-localLU") localLU=true;
        else if (a=="-localcheby"&&i+1<argc) localcheby=atoi(argv[++i]); // fixed-degree Cheby/ICC0
        else if (a=="-sequence"&&i+1<argc) sequence=atoi(argv[++i]);
        else if (a=="-change"&&i+1<argc) change=atof(argv[++i]); }
    PetscInitialize(&argc,&argv,NULL,NULL);

    Mesh smesh = meshfile ? Mesh(meshfile,1,1)
                          : Mesh::MakeCartesian3D(nx,nx,nx,Element::TETRAHEDRON);
    ParMesh pmesh(MPI_COMM_WORLD, smesh); smesh.Clear();
    H1_FECollection fec(1,3);
    ParFiniteElementSpace fes(&pmesh, &fec);
    const int T = fes.GetTrueVSize(), L = fes.GetVSize();

    DenseMatrix Sig(3); Sig=0.0;
    Sig(0,0)=aniso?0.79:1.0; Sig(1,1)=aniso?0.255:1.0; Sig(2,2)=aniso?0.255:1.0;
    MatrixConstantCoefficient sig(Sig);
    ParBilinearForm a(&fes);
    a.AddDomainIntegrator(new DiffusionIntegrator(sig));
    a.Assemble(); a.Finalize();
    // keep a copy of the LOCAL Neumann block BEFORE FormSystemMatrix touches it
    SparseMatrix Kloc(a.SpMat());     // local element-assembled Neumann block (L x L)
    Kloc.Finalize();
    // global operator (true-dof), pin true dof 0 on rank 0 (non-singular)
    Array<int> ess_tdof; if (rank==0) ess_tdof.Append(0);
    OperatorPtr A; a.FormSystemMatrix(ess_tdof, A);
    HypreParMatrix *Ah = A.As<HypreParMatrix>();

    // consistent RHS B = A t
    Vector B(T), X(T); X = 0.0;
    { Vector t(T); t.Randomize(1); if (rank==0 && ess_tdof.Size()) t(ess_tdof[0])=0.0;
      Ah->Mult(t, B); }

    // ---- baseline: block-Jacobi + ICC (what forward_ecg uses) --------------
    if (bjacobi) {
        PetscParMatrix Ap; HypreToPetscAIJ(*Ah, Ap, "base", rank, 1, false);
        Vec b,x; MatCreateVecs((Mat)Ap,&b,&x);
        { PetscScalar*ba; VecGetArray(b,&ba); for(int i=0;i<T;++i) ba[i]=B(i);
          VecRestoreArray(b,&ba); VecZeroEntries(x); }
        PetscOptionsSetValue(NULL,"-sub_ksp_type","preonly");
        PetscOptionsSetValue(NULL,"-sub_pc_type","icc");
        KSP ksp; KSPCreate(MPI_COMM_WORLD,&ksp); KSPSetType(ksp,KSPCG);
        KSPSetOperators(ksp,(Mat)Ap,(Mat)Ap);
        KSPSetTolerances(ksp,1e-6,1e-12,PETSC_DEFAULT,2000);
        PC pc; KSPGetPC(ksp,&pc); PCSetType(pc,PCBJACOBI);
        KSPSetFromOptions(ksp); KSPSolve(ksp,b,x);
        PetscInt its; KSPGetIterationNumber(ksp,&its);
        Vec r; VecDuplicate(b,&r); MatMult((Mat)Ap,x,r); VecAYPX(r,-1.0,b);
        PetscReal rn,bn; VecNorm(r,NORM_2,&rn); VecNorm(b,NORM_2,&bn);
        if (rank==0) printf("[BJACOBI-par] ranks=%d T=%d iters=%d true_res=%.3e\n",
                            nranks,(int)fes.GlobalTrueVSize(),(int)its,(double)(rn/bn));
        PetscFinalize(); return 0;
    }

    // interface mass M_Gamma on the shared faces (real, with off-diagonals)
    SparseMatrix MG(L, L);
    MassIntegrator mi;
    const int nsf = pmesh.GetNSharedFaces();
    IsoparametricTransformation FTr;
    for (int sf=0; sf<nsf; ++sf) {
        int lf = pmesh.GetSharedFace(sf);
        const FiniteElement *fe = fes.GetFaceElement(lf);
        if (!fe) continue;
        pmesh.GetFaceTransformation(lf, &FTr);        // face's own 2D->3D map
        DenseMatrix Me; mi.AssembleElementMatrix(*fe, FTr, Me);
        Array<int> vd; fes.GetFaceVDofs(lf, vd);
        if (vd.Size() == Me.Height()) MG.AddSubMatrix(vd, vd, Me);
    }
    MG.Finalize();

    // local Robin block  K_loc + alpha * M_Gamma  (L x L, this rank)
    SparseMatrix Krob(Kloc); Krob.Add(alpha, MG);

    // partition of unity: multiplicity of each L-dof = (P (P^T 1_L))
    const Operator *P = fes.GetProlongationMatrix();
    Vector onesL(L); onesL = 1.0; Vector multT(T); P->MultTranspose(onesL, multT);
    Vector multL(L); P->Mult(multT, multL);
    Vector dL(L); for (int i=0;i<L;++i) dL(i) = 1.0/multL(i);

    // build the SORAS preconditioner
    SORASPrec prec(T);
    prec.P = P; prec.dL = dL; prec.rL.SetSize(L); prec.yL.SetSize(L);
    Mat Kp = ToSeqAIJ(Krob);
    KSPCreate(PETSC_COMM_SELF,&prec.kloc);
    KSPSetOperators(prec.kloc,Kp,Kp);
    if (localicc0) {          // inexact local solve: one ICC0 apply (preonly)
        KSPSetType(prec.kloc,KSPPREONLY);
        PC pc; KSPGetPC(prec.kloc,&pc); PCSetType(pc,PCICC);
    } else if (localcheby>0) {// fixed-degree Chebyshev over ICC0: memory-flat,
        KSPSetType(prec.kloc,KSPCHEBYSHEV);          // parallel SpMV, fixed SPD op
        KSPSetTolerances(prec.kloc,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,localcheby);
        KSPSetNormType(prec.kloc,KSP_NORM_NONE);
        KSPChebyshevEstEigSet(prec.kloc,0.0,0.1,0.0,1.1);
        PC pc; KSPGetPC(prec.kloc,&pc); PCSetType(pc,PCICC);
    } else if (localLU) {     // near-EXACT via direct factor: factor once, cheap apply
        KSPSetType(prec.kloc,KSPPREONLY);
        PC pc; KSPGetPC(prec.kloc,&pc); PCSetType(pc,PCLU);
    } else {                  // near-exact via inner CG+ICC to 1e-10 (default)
        KSPSetType(prec.kloc,KSPCG);
        KSPSetTolerances(prec.kloc,1e-10,1e-14,PETSC_DEFAULT,500);
        KSPSetNormType(prec.kloc,KSP_NORM_UNPRECONDITIONED);
        PC pc; KSPGetPC(prec.kloc,&pc); PCSetType(pc,PCICC);
    }
    KSPSetErrorIfNotConverged(prec.kloc,PETSC_FALSE);
    MatCreateVecs(Kp,&prec.rloc,&prec.zloc);

    // outer parallel CG with the SORAS preconditioner (B, X built above)
    CGSolver cg(MPI_COMM_WORLD);
    cg.SetOperator(*Ah); cg.SetPreconditioner(prec);
    cg.SetRelTol(1e-6); cg.SetAbsTol(1e-12); cg.SetMaxIter(2000);
    cg.SetPrintLevel(0);

    // ---- cross-time acceleration demo: slowly-varying RHS sequence ---------
    // Sys2/Sys3 are re-solved every time step with a slowly-varying RHS (driven
    // by Sys1's Vm(t)).  Compare cold start (x0=0) vs warm start (x0 = previous
    // solution) -- the cheapest cross-time/cross-system acceleration.
    if (sequence>0) {
        auto ip=[&](const Vector&x,const Vector&y){ return InnerProduct(MPI_COMM_WORLD,x,y); };
        Vector xstar(T), b(T), Xcold(T), Xwarm(T), Xf(T);
        xstar.Randomize(1); if (rank==0 && ess_tdof.Size()) xstar(0)=0.0;
        Xwarm = 0.0;
        std::vector<Vector> Pb, APb;   // A-orthonormal history + A*history (Fischer)
        const int MAXB=12;
        // representative trajectory: u_e(t) over a beat lives in a LOW-dim smooth
        // manifold, not a random walk.  xstar(s) = sum_k sin(w_k s + k) v_k,
        // v_k fixed modes -> a smooth NM-dim path.  (change scales the step via w.)
        const int NM=4; std::vector<Vector> modes(NM);
        for (int k=0;k<NM;++k){ modes[k].SetSize(T); modes[k].Randomize(7+k);
            if (rank==0 && ess_tdof.Size()) modes[k](0)=0.0; }
        if (rank==0) printf("[SEQUENCE] low-dim smooth trajectory (NM=%d): cold vs warm vs Fischer\n",NM);
        int tc=0, tw=0, tf=0;
        for (int s=0; s<sequence; ++s) {
            xstar = 0.0;
            for (int k=0;k<NM;++k) xstar.Add(sin(change*10*(k+1)*s + k), modes[k]);
            if (rank==0 && ess_tdof.Size()) xstar(0)=0.0;
            Ah->Mult(xstar, b);
            double bn2 = sqrt(ip(b,b));
            cg.SetRelTol(0.0); cg.SetAbsTol(1e-6*bn2);   // fixed accuracy vs ||b||
            // cold
            Xcold = 0.0; cg.iterative_mode=false; cg.Mult(b, Xcold);
            int cold=cg.GetNumIterations();
            // warm-start (previous solution only)
            cg.iterative_mode=true;  cg.Mult(b, Xwarm);
            int warm=cg.GetNumIterations();
            // Fischer: x0 = sum_i <p_i,b> p_i  (A-orth. projection onto history)
            Xf = 0.0;
            for (size_t i=0;i<Pb.size();++i) Xf.Add(ip(Pb[i],b), Pb[i]);
            cg.iterative_mode=true; cg.Mult(b, Xf);
            int fisch=cg.GetNumIterations();
            // add the new solution to the A-orthonormal history (Gram-Schmidt)
            Vector w(Xf), Aw(T); Ah->Mult(w, Aw);
            for (size_t i=0;i<Pb.size();++i){ double c=ip(APb[i],w); w.Add(-c,Pb[i]); Aw.Add(-c,APb[i]); }
            double nrm=sqrt(ip(w,Aw));
            if (nrm>1e-12 && (int)Pb.size()<MAXB){ w*=1.0/nrm; Aw*=1.0/nrm; Pb.push_back(w); APb.push_back(Aw); }
            tc+=cold; tw+=warm; tf+=fisch;
            if (rank==0) printf("[SEQ] step=%2d  cold=%3d  warm=%3d  Fischer=%3d\n", s, cold, warm, fisch);
        }
        if (rank==0) printf("[SEQ] TOTAL cold=%d  warm=%d (-%.0f%%)  Fischer=%d (-%.0f%%)\n",
                            tc, tw, 100.0*(tc-tw)/tc, tf, 100.0*(tc-tf)/tc);
        PetscFinalize(); return 0;
    }
    cg.Mult(B, X);
    int its = cg.GetNumIterations(); int conv = cg.GetConverged();
    // true residual
    Vector res(T); Ah->Mult(X,res); res -= B;
    double rn = sqrt(InnerProduct(MPI_COMM_WORLD,res,res));
    double bn = sqrt(InnerProduct(MPI_COMM_WORLD,B,B));
    if (rank==0)
        printf("[SORAS-par] ranks=%d  T=%d  alpha=%g  n_shared_faces(r0)=%d  "
               "iters=%d  true_res=%.3e  %s\n",
               nranks, fes.GlobalTrueVSize(), alpha, nsf, its, rn/bn,
               conv?"CONVERGED":"DIVERGED");
    PetscFinalize();
    return 0;
}
