/* twolevel.c -- improving sASM by adding a coarse space (the real lever).
 *
 * Diagnosis (from coloring.c): sASM already makes lambda_max ~ omega = O(1) (the
 * over-count N_hat is removed). What is left large is the CONDITION NUMBER, and it
 * is dominated by a TINY lambda_min ~ C_0^{-2}, which degrades as the number of
 * subdomains grows (the missing global coupling / "1 subdomain per iteration").
 *
 * Fix: a symmetric two-level sASM
 *   M2^{-1} = R0^T A0^{-1} R0  +  D^{-1/2} (sum_i R_i^T A_i^{-1} R_i) D^{-1/2},
 * with a Nicolaides partition-of-unity coarse space: one coarse basis function per
 * subdomain, phi_i(k) = 1/m_k for k in subdomain i (so sum_i phi_i = 1). A0 = R0 A R0^T
 * is the Galerkin coarse operator (dim = #subdomains), solved exactly (Cholesky).
 * The coarse term is symmetric and added -> still CG-compatible, still low-memory.
 *
 * Two experiments:
 *  (A) FIXED 120^2, 6x6, O=2, ICC(0): one-level vs two-level lambda_min/lambda_max/
 *      kappa and CG iterations.
 *  (B) SCALABILITY: fixed subdomain size (n = 20*P, so H/h fixed), P = 4..12, O=2,
 *      ICC(0): CG iterations one-level (grows) vs two-level (flat) -> scalable.
 *
 * lambda's via KSPCG + PCSHELL + KSPComputeExtremeSingularValues. Sequential.
 */
#include <petscksp.h>
#include <math.h>

typedef struct { PetscInt n; PetscInt *idx; IS is; Mat Ai; KSP ksp; Vec ri, yi; } Sub;

static Mat Laplace2D(PetscInt n) {
    Mat A; PetscInt N = n*n; MatCreateSeqAIJ(PETSC_COMM_SELF, N, N, 5, NULL, &A);
    for (PetscInt b = 0; b < n; ++b) for (PetscInt a = 0; a < n; ++a) {
        PetscInt k = b*n + a; PetscScalar four = 4.0, m1 = -1.0;
        MatSetValue(A, k, k, four, INSERT_VALUES);
        if (a > 0)   MatSetValue(A, k, k-1, m1, INSERT_VALUES);
        if (a < n-1) MatSetValue(A, k, k+1, m1, INSERT_VALUES);
        if (b > 0)   MatSetValue(A, k, k-n, m1, INSERT_VALUES);
        if (b < n-1) MatSetValue(A, k, k+n, m1, INSERT_VALUES);
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE); return A;
}

static void BuildSub(Mat A, PetscInt *idx, PetscInt ni, int icc, Sub *s) {
    s->n = ni; s->idx = (PetscInt*)malloc(sizeof(PetscInt)*ni);
    for (PetscInt j = 0; j < ni; ++j) s->idx[j] = idx[j];
    ISCreateGeneral(PETSC_COMM_SELF, ni, s->idx, PETSC_COPY_VALUES, &s->is);
    MatCreateSubMatrix(A, s->is, s->is, MAT_INITIAL_MATRIX, &s->Ai);
    KSPCreate(PETSC_COMM_SELF, &s->ksp); KSPSetType(s->ksp, KSPPREONLY);
    KSPSetOperators(s->ksp, s->Ai, s->Ai);
    PC pc; KSPGetPC(s->ksp, &pc);
    if (icc) { PCSetType(pc, PCICC); PCFactorSetLevels(pc, 0); } else PCSetType(pc, PCCHOLESKY);
    KSPSetUp(s->ksp); MatCreateVecs(s->Ai, &s->ri, &s->yi);
}

static PetscInt *MultArray(Sub *subs, PetscInt nsub, PetscInt N) {
    PetscInt *m = (PetscInt*)calloc(N, sizeof(PetscInt));
    for (PetscInt i = 0; i < nsub; ++i)
        for (PetscInt j = 0; j < subs[i].n; ++j) m[subs[i].idx[j]]++;
    return m;
}

/* ---- two-level context ---- */
typedef struct {
    Sub *subs; PetscInt nsub; Vec dsq, tmp;     /* fine sASM */
    int twolevel; Mat R0t; KSP kspc; Vec cvec, ycvec;  /* coarse */
} Ctx;

static void FineSASM(Ctx *c, Vec r, Vec z) {
    VecCopy(r, c->tmp); VecPointwiseMult(c->tmp, c->tmp, c->dsq);  /* D^{-1/2} r */
    VecZeroEntries(z);
    const PetscScalar *ra; PetscScalar *za;
    VecGetArrayRead(c->tmp, &ra); VecGetArray(z, &za);
    for (PetscInt i = 0; i < c->nsub; ++i) {
        Sub *s = &c->subs[i];
        PetscScalar *rib; VecGetArray(s->ri, &rib);
        for (PetscInt j = 0; j < s->n; ++j) rib[j] = ra[s->idx[j]];
        VecRestoreArray(s->ri, &rib);
        KSPSolve(s->ksp, s->ri, s->yi);
        const PetscScalar *yib; VecGetArrayRead(s->yi, &yib);
        for (PetscInt j = 0; j < s->n; ++j) za[s->idx[j]] += yib[j];
        VecRestoreArrayRead(s->yi, &yib);
    }
    VecRestoreArrayRead(c->tmp, &ra); VecRestoreArray(z, &za);
    VecPointwiseMult(z, z, c->dsq);                                /* D^{-1/2} (...) */
}

static void Apply(Ctx *c, Vec r, Vec z) {
    FineSASM(c, r, z);
    if (c->twolevel) {                       /* z += R0^T A0^{-1} R0 r */
        MatMultTranspose(c->R0t, r, c->cvec);
        KSPSolve(c->kspc, c->cvec, c->ycvec);
        MatMultAdd(c->R0t, c->ycvec, z, z);
    }
}

static PetscErrorCode ShellApply(PC pc, Vec r, Vec z) {
    Ctx *c; PCShellGetContext(pc, &c); Apply(c, r, z); return 0;
}

/* build Nicolaides PoU coarse space: R0t is N x nsub, col i = phi_i, phi_i(k)=1/m_k */
static void BuildCoarse(Mat A, Sub *subs, PetscInt nsub, PetscInt N, PetscInt *mult, Ctx *c) {
    Mat R0t; MatCreateSeqAIJ(PETSC_COMM_SELF, N, nsub, 16, NULL, &R0t);
    for (PetscInt i = 0; i < nsub; ++i)
        for (PetscInt j = 0; j < subs[i].n; ++j) {
            PetscInt k = subs[i].idx[j];
            MatSetValue(R0t, k, i, 1.0/(PetscReal)mult[k], INSERT_VALUES);
        }
    MatAssemblyBegin(R0t, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(R0t, MAT_FINAL_ASSEMBLY);
    c->R0t = R0t;
    Mat A0; MatPtAP(A, R0t, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &A0);   /* A0 = R0 A R0^T */
    KSPCreate(PETSC_COMM_SELF, &c->kspc); KSPSetType(c->kspc, KSPPREONLY);
    KSPSetOperators(c->kspc, A0, A0);
    PC pc; KSPGetPC(c->kspc, &pc); PCSetType(pc, PCCHOLESKY);
    KSPSetUp(c->kspc);
    MatCreateVecs(A0, &c->ycvec, &c->cvec);
    MatDestroy(&A0);
}

static void Spectrum(Mat A, Ctx *c, Vec b, PetscReal *lmax, PetscReal *lmin) {
    Vec x; VecDuplicate(b, &x);
    KSP ksp; KSPCreate(PETSC_COMM_SELF, &ksp); KSPSetType(ksp, KSPCG);
    KSPSetOperators(ksp, A, A);
    PC pc; KSPGetPC(ksp, &pc); PCSetType(pc, PCSHELL);
    PCShellSetContext(pc, c); PCShellSetApply(pc, ShellApply);
    KSPSetComputeSingularValues(ksp, PETSC_TRUE);
    KSPSetTolerances(ksp, 1e-11, 1e-50, PETSC_DEFAULT, 800);
    KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED);
    KSPSetUp(ksp); KSPSolve(ksp, b, x);
    KSPComputeExtremeSingularValues(ksp, lmax, lmin);
    VecDestroy(&x); KSPDestroy(&ksp);
}

static PetscInt PCG(Mat A, Ctx *c, Vec b, Vec x, PetscReal rtol, PetscInt maxit) {
    Vec r, z, p, Ap; VecDuplicate(b,&r); VecDuplicate(b,&z); VecDuplicate(b,&p); VecDuplicate(b,&Ap);
    VecZeroEntries(x); VecCopy(b, r);
    PetscReal bnorm; VecNorm(b, NORM_2, &bnorm);
    Apply(c, r, z); VecCopy(z, p);
    PetscScalar rz; VecDot(r, z, &rz);
    PetscInt it; PetscReal rn;
    for (it = 0; it < maxit; ++it) {
        MatMult(A, p, Ap); PetscScalar pAp; VecDot(p, Ap, &pAp);
        PetscScalar al = rz/pAp; VecAXPY(x, al, p); VecAXPY(r, -al, Ap);
        VecNorm(r, NORM_2, &rn); if (rn/bnorm < rtol) { ++it; break; }
        Apply(c, r, z); PetscScalar rz2; VecDot(r, z, &rz2);
        PetscScalar be = rz2/rz; rz = rz2; VecAYPX(p, be, z);
    }
    VecDestroy(&r); VecDestroy(&z); VecDestroy(&p); VecDestroy(&Ap); return it;
}

/* build P*P overlapped subdomains on an n x n grid, ICC(0); returns nsub, fills arrays */
static Sub *BuildAll(Mat A, PetscInt n, int P, int O, int icc, PetscInt *nsubOut) {
    int NS = P*P; Sub *S = (Sub*)malloc(sizeof(Sub)*NS); int sc = 0;
    for (int q=0;q<P;++q) for (int p=0;p<P;++p) {
        PetscInt axlo=(p*n)/P, axhi=((p+1)*n)/P, bylo=(q*n)/P, byhi=((q+1)*n)/P;
        PetscInt al=axlo-O<0?0:axlo-O, ar=axhi+O>n?n:axhi+O;
        PetscInt bl=bylo-O<0?0:bylo-O, br=byhi+O>n?n:byhi+O;
        PetscInt ni=(ar-al)*(br-bl), *idx=(PetscInt*)malloc(sizeof(PetscInt)*ni), cc=0;
        for(PetscInt bb=bl;bb<br;++bb) for(PetscInt a=al;a<ar;++a) idx[cc++]=bb*n+a;
        BuildSub(A, idx, ni, icc, &S[sc++]); free(idx);
    }
    *nsubOut = NS; return S;
}
static void FreeAll(Sub *S, int n) {
    for (int i=0;i<n;++i){ISDestroy(&S[i].is);MatDestroy(&S[i].Ai);KSPDestroy(&S[i].ksp);
        VecDestroy(&S[i].ri);VecDestroy(&S[i].yi);free(S[i].idx);} free(S);
}
static Ctx MakeCtx(Mat A, Sub *S, int nsub, PetscInt N, int twolevel) {
    Ctx c; c.subs=S; c.nsub=nsub; c.twolevel=twolevel;
    MatCreateVecs(A, &c.tmp, NULL);
    Vec mult; MatCreateVecs(A,&mult,NULL); VecZeroEntries(mult);
    PetscScalar *ma; VecGetArray(mult,&ma);
    for(int i=0;i<nsub;++i) for(PetscInt j=0;j<S[i].n;++j) ma[S[i].idx[j]]+=1.0;
    VecRestoreArray(mult,&ma);
    VecDuplicate(mult,&c.dsq); VecCopy(mult,c.dsq); VecReciprocal(c.dsq); VecSqrtAbs(c.dsq);
    VecDestroy(&mult);
    if (twolevel) { PetscInt *m=MultArray(S,nsub,N); BuildCoarse(A,S,nsub,N,m,&c); free(m); }
    else { c.R0t=NULL; c.kspc=NULL; c.cvec=NULL; c.ycvec=NULL; }
    return c;
}
static void FreeCtx(Ctx *c) {
    VecDestroy(&c->tmp); VecDestroy(&c->dsq);
    if (c->twolevel){ MatDestroy(&c->R0t); KSPDestroy(&c->kspc); VecDestroy(&c->cvec); VecDestroy(&c->ycvec); }
}

int main(int argc, char **argv) {
    PetscInitialize(&argc, &argv, NULL, NULL);

    /* ===== (A) FIXED config: one-level vs two-level spectrum + iters ===== */
    {
        PetscInt n=120, N=n*n; int P=6, O=2;
        Mat A = Laplace2D(n);
        Vec b,x; MatCreateVecs(A,&b,&x); VecZeroEntries(b);
        { PetscScalar *ba; VecGetArray(b,&ba); for(PetscInt bb=0;bb<n;++bb) ba[bb*n+0]=1.0; VecRestoreArray(b,&ba);}
        int nsub; Sub *S = BuildAll(A, n, P, O, 1/*ICC*/, &nsub);
        PetscPrintf(PETSC_COMM_SELF,"=== (A) FIXED 120^2, 6x6, O=2, ICC(0) ===\n");
        PetscPrintf(PETSC_COMM_SELF,"%-18s %10s %10s %10s %8s\n","method","lambda_min","lambda_max","kappa","CG its");
        FILE *fa = fopen("twolevel_fixed.txt","w");
        fprintf(fa,"# level lambda_min lambda_max kappa CG_its\n");
        for (int two=0; two<=1; ++two) {
            Ctx c = MakeCtx(A, S, nsub, N, two);
            PetscReal lmax,lmin; Spectrum(A,&c,b,&lmax,&lmin);
            PetscInt it = PCG(A,&c,b,x,1e-8,2000);
            PetscPrintf(PETSC_COMM_SELF,"%-18s %10.4g %10.4f %10.1f %8d\n",
                two?"two-level sASM":"one-level sASM", lmin, lmax, lmax/lmin, (int)it);
            fprintf(fa,"%d %.6g %.6g %.6g %d\n", two+1, lmin, lmax, lmax/lmin, (int)it);
            FreeCtx(&c);
        }
        fclose(fa);
        FreeAll(S,nsub); VecDestroy(&b); VecDestroy(&x); MatDestroy(&A);
    }

    /* ===== (B) SCALABILITY: fixed subdomain size (n=20P), grow P ===== */
    {
        PetscPrintf(PETSC_COMM_SELF,"\n=== (B) SCALABILITY: subdomain ~20x20 fixed, O=2, ICC(0) ===\n");
        PetscPrintf(PETSC_COMM_SELF,"%4s %8s %10s %14s %14s\n","P","#subdom","N(dof)","one-level its","two-level its");
        FILE *fb = fopen("twolevel_scaling.txt","w");
        fprintf(fb,"# P nsub N onelevel_its twolevel_its\n");
        int Ps[5] = {4,6,8,10,12};
        for (int pi=0; pi<5; ++pi) {
            int P=Ps[pi]; PetscInt n=20*P, N=n*n; int O=2;
            Mat A = Laplace2D(n);
            Vec b,x; MatCreateVecs(A,&b,&x); VecZeroEntries(b);
            { PetscScalar *ba; VecGetArray(b,&ba); for(PetscInt bb=0;bb<n;++bb) ba[bb*n+0]=1.0; VecRestoreArray(b,&ba);}
            int nsub; Sub *S = BuildAll(A, n, P, O, 1, &nsub);
            Ctx c1 = MakeCtx(A,S,nsub,N,0); PetscInt i1 = PCG(A,&c1,b,x,1e-8,5000); FreeCtx(&c1);
            Ctx c2 = MakeCtx(A,S,nsub,N,1); PetscInt i2 = PCG(A,&c2,b,x,1e-8,5000); FreeCtx(&c2);
            PetscPrintf(PETSC_COMM_SELF,"%4d %8d %10d %14d %14d\n",P,nsub,(int)N,(int)i1,(int)i2);
            fprintf(fb,"%d %d %d %d %d\n",P,nsub,(int)N,(int)i1,(int)i2);
            FreeAll(S,nsub); VecDestroy(&b); VecDestroy(&x); MatDestroy(&A);
        }
        fclose(fb);
    }
    PetscPrintf(PETSC_COMM_SELF,"\nTWOLEVEL_DONE\n");
    PetscFinalize(); return 0;
}
