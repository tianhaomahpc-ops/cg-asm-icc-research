// localcost.cpp -- local-solve accuracy vs compute cost on ONE subdomain block.
// Answers: what does ICC(0/1/2) cost (factor fill = apply cost), and for the
// SAME compute budget, how accurate is CG+ICC0?
//
//   cost unit = one ICC0 triangular-solve apply (~ nnz(A) work).
//   ICC(k) preonly apply  : cost ~ nnz(ICC(k) factor)/nnz(ICC0 factor).
//   CG+ICC0, m iterations : cost ~ m * (SpMV + ICC0 apply) ~ m * (1 + 1) units.
//   accuracy = ||b - A x|| / ||b|| after the local solve (x0 = 0).
#include "mfem.hpp"
#include <petscksp.h>
#include <vector>
#include <cstdio>
using namespace mfem;

static Mat ToSeqAIJ(SparseMatrix &S){
    S.Finalize(); int n=S.Height(); const int*I=S.GetI(),*J=S.GetJ(); const double*A=S.GetData();
    std::vector<PetscInt> nnz(n); for(int i=0;i<n;++i) nnz[i]=I[i+1]-I[i];
    Mat M; MatCreateSeqAIJ(PETSC_COMM_SELF,n,n,0,nnz.data(),&M);
    MatSetOption(M,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);
    std::vector<PetscInt> c; std::vector<PetscScalar> v;
    for(int i=0;i<n;++i){c.clear();v.clear();
        for(int k=I[i];k<I[i+1];++k){c.push_back(J[k]);v.push_back(A[k]);}
        PetscInt r=i,m=c.size(); if(m) MatSetValues(M,1,&r,m,c.data(),v.data(),INSERT_VALUES);}
    MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY); return M;
}
// residual ||b - A x||/||b|| after applying a KSP once (x0=0)
static double resid(Mat A, KSP ksp, Vec b){
    Vec x,r; MatCreateVecs(A,&x,&r); KSPSolve(ksp,b,x);
    MatMult(A,x,r); VecAYPX(r,-1.0,b);
    PetscReal rn,bn; VecNorm(r,NORM_2,&rn); VecNorm(b,NORM_2,&bn);
    VecDestroy(&x);VecDestroy(&r); return rn/bn;
}
static long facNnz(KSP ksp){ Mat F; PC pc; KSPGetPC(ksp,&pc);
    PCFactorGetMatrix(pc,&F); MatInfo info; MatGetInfo(F,MAT_LOCAL,&info);
    return (long)info.nz_used; }

int main(int argc,char**argv){
    int nx=16; double alpha=0.15;
    for(int i=1;i<argc;++i){ std::string a=argv[i];
        if(a=="-nx"&&i+1<argc)nx=atoi(argv[++i]); else if(a=="-alpha"&&i+1<argc)alpha=atof(argv[++i]); }
    PetscInitialize(&argc,&argv,NULL,NULL);
    // one representative subdomain: cube Neumann block + Robin (alpha on bdr diag),
    // anisotropic sigma like Sys2; pin dof 0 so it is SPD (stand-in for K+alpha*Mg).
    Mesh mesh=Mesh::MakeCartesian3D(nx,nx,nx,Element::TETRAHEDRON);
    H1_FECollection fec(1,3); FiniteElementSpace fes(&mesh,&fec);
    DenseMatrix Sig(3);Sig=0.0;Sig(0,0)=0.79;Sig(1,1)=0.255;Sig(2,2)=0.255;
    MatrixConstantCoefficient sig(Sig);
    BilinearForm a(&fes); a.AddDomainIntegrator(new DiffusionIntegrator(sig));
    // + boundary mass (the Robin term) on all faces, scaled by alpha
    ConstantCoefficient ac(alpha); a.AddBoundaryIntegrator(new BoundaryMassIntegrator(ac));
    a.Assemble(); Array<int> ess; ess.Append(0); SparseMatrix K; a.FormSystemMatrix(ess,K);
    Mat A=ToSeqAIJ(K); int N; MatGetSize(A,&N,NULL);
    MatInfo ai; MatGetInfo(A,MAT_LOCAL,&ai); long nnzA=(long)ai.nz_used;
    Vec b; MatCreateVecs(A,&b,NULL); VecSetRandom(b,NULL);

    printf("subdomain block N=%d  nnz(A)=%ld  (cost unit = 1 ICC0 apply)\n",N,nnzA);
    printf("%-22s %-12s %-10s %-12s\n","method","factor_nnz","cost~","residual ||r||/||b||");
    // ICC(k) preonly, one apply
    long f0=0;
    for(int lev=0;lev<=2;++lev){
        KSP k; KSPCreate(PETSC_COMM_SELF,&k); KSPSetType(k,KSPPREONLY); KSPSetOperators(k,A,A);
        PC pc; KSPGetPC(k,&pc); PCSetType(pc,PCICC); PCFactorSetLevels(pc,lev);
        KSPSetUp(k); long fn=facNnz(k); if(lev==0)f0=fn;
        double rr=resid(A,k,b);
        printf("ICC(%d) 1 apply        %-12ld %-10.2f %.4e\n",lev,fn,(double)fn/f0,rr);
        KSPDestroy(&k);
    }
    // CG+ICC0, m iterations (cost ~ m*(nnzA + f0)/f0 units)
    for(int m=1;m<=32;m*=2){
        KSP k; KSPCreate(PETSC_COMM_SELF,&k); KSPSetType(k,KSPCG); KSPSetOperators(k,A,A);
        KSPSetTolerances(k,1e-14,1e-16,PETSC_DEFAULT,m); KSPSetNormType(k,KSP_NORM_UNPRECONDITIONED);
        PC pc; KSPGetPC(k,&pc); PCSetType(pc,PCICC); PCFactorSetLevels(pc,0);
        double rr=resid(A,k,b);
        double cost=(double)m*(nnzA+f0)/f0;
        printf("CG+ICC0 %2d iters      %-12ld %-10.2f %.4e\n",m,f0,cost,rr);
        KSPDestroy(&k);
    }
    // the MEANINGFUL metric: CG+ICC(k) iterations to converge (rtol 1e-8),
    // and the total cost = iters * per-iter-cost (SpMV + ICC(k) apply).
    printf("\n-- CG+ICC(k) to rtol 1e-8: iterations & total local cost --\n");
    printf("%-14s %-8s %-14s %-14s\n","precond","iters","per_iter_cost","total_cost~");
    for(int lev=0;lev<=2;++lev){
        KSP k; KSPCreate(PETSC_COMM_SELF,&k); KSPSetType(k,KSPCG); KSPSetOperators(k,A,A);
        KSPSetTolerances(k,1e-8,1e-14,PETSC_DEFAULT,2000); KSPSetNormType(k,KSP_NORM_UNPRECONDITIONED);
        PC pc; KSPGetPC(k,&pc); PCSetType(pc,PCICC); PCFactorSetLevels(pc,lev);
        Vec x; MatCreateVecs(A,&x,NULL); VecZeroEntries(x); KSPSolve(k,b,x);
        PetscInt it; KSPGetIterationNumber(k,&it);
        long fn=facNnz(k);
        double pic=(double)(nnzA+fn)/f0;          // per-iter cost in ICC0-apply units
        printf("CG+ICC(%d)      %-8d %-14.2f %-14.1f\n",lev,(int)it,pic,it*pic);
        VecDestroy(&x); KSPDestroy(&k);
    }
    PetscFinalize(); return 0;
}
