/* coloring.c -- studying the coloring number N_c (the "B" factor).
 *
 * The abstract additive-Schwarz bound is  kappa(M^-1 A) <= C_0^2 * omega * N_c,
 * where:
 *   C_0^2 = stable-decomposition constant  (controls lambda_min)
 *   omega = local stability of the (inexact) subdomain solver  (=1 if exact)
 *   N_c   = coloring number = chromatic number of the subdomain OVERLAP graph
 *           (two subdomains conflict iff they share >=1 DOF); it bounds lambda_max.
 *
 * The project's central claim is that omega and N_c MULTIPLY:
 *   BASIC, exact   :  lambda_max <= N_c            (over-count only)
 *   BASIC, ICC     :  lambda_max <= omega * N_c    (the product -> the anomaly)
 *   sASM,  exact   :  lambda_max <= 1              (partition of unity removes N_c)
 *   sASM,  ICC     :  lambda_max <= omega          (only inexactness remains)
 *
 * This program MEASURES lambda_max / lambda_min of the preconditioned operator
 * M^-1 A for all four combos, plus N_c and the max multiplicity m_max, while
 * sweeping the overlap O. If the claim holds we should see:
 *   lambda_max(BASIC,exact) track N_c upward with O;
 *   lambda_max(BASIC,ICC)   ~ omega * that (higher, rising faster);
 *   lambda_max(sASM,*)      stay flat near omega (1 for exact).
 *
 * lambda_max/min are obtained from a real PETSc KSPCG run with a PCSHELL that
 * wraps the hand-built BASIC/sASM apply, via KSPComputeExtremeSingularValues
 * (the CG-Lanczos Ritz values converge to the extreme eigenvalues).
 *
 * Sequential (run -n 1). Dumps coloring_sweep.txt for plot_coloring.py.
 */
#include <petscksp.h>
#include <math.h>

typedef struct {
    PetscInt  n;
    PetscInt *idx;
    IS        is;
    Mat       Ai;
    KSP       ksp;
    Vec       ri, yi;
} Sub;

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
    if (icc) { PCSetType(pc, PCICC); PCFactorSetLevels(pc, 0); }
    else     { PCSetType(pc, PCCHOLESKY); }
    KSPSetUp(s->ksp); MatCreateVecs(s->Ai, &s->ri, &s->yi);
}

static Vec Multiplicity(Mat A, Sub *subs, PetscInt nsub) {
    Vec m; MatCreateVecs(A, &m, NULL); VecZeroEntries(m);
    PetscScalar *ma; VecGetArray(m, &ma);
    for (PetscInt i = 0; i < nsub; ++i)
        for (PetscInt j = 0; j < subs[i].n; ++j) ma[subs[i].idx[j]] += 1.0;
    VecRestoreArray(m, &ma); return m;
}

static void ApplyPrec(Sub *subs, PetscInt nsub, int useSASM, Vec dsq,
                      Vec r, Vec z, Vec tmp) {
    VecCopy(r, tmp);
    if (useSASM) VecPointwiseMult(tmp, tmp, dsq);
    VecZeroEntries(z);
    const PetscScalar *ra; PetscScalar *za;
    VecGetArrayRead(tmp, &ra); VecGetArray(z, &za);
    for (PetscInt i = 0; i < nsub; ++i) {
        Sub *s = &subs[i];
        PetscScalar *rib; VecGetArray(s->ri, &rib);
        for (PetscInt j = 0; j < s->n; ++j) rib[j] = ra[s->idx[j]];
        VecRestoreArray(s->ri, &rib);
        KSPSolve(s->ksp, s->ri, s->yi);
        const PetscScalar *yib; VecGetArrayRead(s->yi, &yib);
        for (PetscInt j = 0; j < s->n; ++j) za[s->idx[j]] += yib[j];
        VecRestoreArrayRead(s->yi, &yib);
    }
    VecRestoreArrayRead(tmp, &ra); VecRestoreArray(z, &za);
    if (useSASM) VecPointwiseMult(z, z, dsq);
}

/* ---- PCSHELL wrapping ApplyPrec so PETSc KSPCG can estimate eigenvalues ---- */
typedef struct { Sub *subs; PetscInt nsub; int useSASM; Vec dsq, tmp; } ShellCtx;
static PetscErrorCode ShellApply(PC pc, Vec r, Vec z) {
    ShellCtx *c; PCShellGetContext(pc, &c);
    ApplyPrec(c->subs, c->nsub, c->useSASM, c->dsq, r, z, c->tmp);
    return 0;
}

/* lambda_max / lambda_min of M^-1 A via a KSPCG run with singular-value tracking */
static void Spectrum(Mat A, Sub *subs, PetscInt nsub, int useSASM, Vec dsq,
                     Vec b, PetscReal *lmax, PetscReal *lmin) {
    ShellCtx ctx; ctx.subs = subs; ctx.nsub = nsub; ctx.useSASM = useSASM; ctx.dsq = dsq;
    VecDuplicate(b, &ctx.tmp);
    Vec x; VecDuplicate(b, &x);
    KSP ksp; KSPCreate(PETSC_COMM_SELF, &ksp); KSPSetType(ksp, KSPCG);
    KSPSetOperators(ksp, A, A);
    PC pc; KSPGetPC(ksp, &pc); PCSetType(pc, PCSHELL);
    PCShellSetContext(pc, &ctx); PCShellSetApply(pc, ShellApply);
    KSPSetComputeSingularValues(ksp, PETSC_TRUE);
    KSPSetTolerances(ksp, 1e-12, 1e-50, PETSC_DEFAULT, 600);
    KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED);
    KSPSetUp(ksp);
    KSPSolve(ksp, b, x);
    KSPComputeExtremeSingularValues(ksp, lmax, lmin);
    VecDestroy(&x); VecDestroy(&ctx.tmp); KSPDestroy(&ksp);
}

/* m_max = max multiplicity; N_c = greedy chromatic number of the overlap graph */
static void Coloring(Sub *subs, PetscInt nsub, Mat A, PetscInt N,
                     PetscInt *mmax, PetscInt *Nc) {
    /* multiplicity */
    PetscInt *mult = (PetscInt*)calloc(N, sizeof(PetscInt));
    for (PetscInt i = 0; i < nsub; ++i)
        for (PetscInt j = 0; j < subs[i].n; ++j) mult[subs[i].idx[j]]++;
    PetscInt mm = 0; for (PetscInt k = 0; k < N; ++k) if (mult[k] > mm) mm = mult[k];
    *mmax = mm; free(mult);

    /* overlap-graph adjacency: i ~ j iff they share a DOF */
    char *adj = (char*)calloc((size_t)nsub*nsub, 1);
    char *mark = (char*)calloc(N, 1);
    for (PetscInt i = 0; i < nsub; ++i) {
        for (PetscInt j = 0; j < subs[i].n; ++j) mark[subs[i].idx[j]] = 1;
        for (PetscInt jj = i+1; jj < nsub; ++jj) {
            int shared = 0;
            for (PetscInt q = 0; q < subs[jj].n; ++q)
                if (mark[subs[jj].idx[q]]) { shared = 1; break; }
            if (shared) { adj[i*nsub+jj] = 1; adj[jj*nsub+i] = 1; }
        }
        for (PetscInt j = 0; j < subs[i].n; ++j) mark[subs[i].idx[j]] = 0;
    }
    /* greedy coloring in natural order */
    PetscInt *color = (PetscInt*)malloc(nsub*sizeof(PetscInt));
    for (PetscInt i = 0; i < nsub; ++i) color[i] = -1;
    PetscInt nc = 0;
    for (PetscInt i = 0; i < nsub; ++i) {
        char used[256] = {0};
        for (PetscInt jj = 0; jj < nsub; ++jj)
            if (adj[i*nsub+jj] && color[jj] >= 0 && color[jj] < 256) used[color[jj]] = 1;
        PetscInt c = 0; while (c < 256 && used[c]) ++c;
        color[i] = c; if (c+1 > nc) nc = c+1;
    }
    *Nc = nc;
    free(adj); free(mark); free(color);
}

static void FreeSubs(Sub *S, int n) {
    for (int i = 0; i < n; ++i) { ISDestroy(&S[i].is); MatDestroy(&S[i].Ai);
        KSPDestroy(&S[i].ksp); VecDestroy(&S[i].ri); VecDestroy(&S[i].yi); free(S[i].idx); }
}

int main(int argc, char **argv) {
    PetscInitialize(&argc, &argv, NULL, NULL);
    const PetscInt n = 120, P = 6, N = n*n;     /* 6x6 base subdomains of 20x20 */
    Mat A = Laplace2D(n);
    Vec b, x; MatCreateVecs(A, &b, &x);
    VecZeroEntries(b);                            /* left-edge Dirichlet rhs */
    { PetscScalar *ba; VecGetArray(b,&ba);
      for (PetscInt bb=0; bb<n; ++bb) ba[bb*n+0]=1.0; VecRestoreArray(b,&ba); }

    FILE *f = fopen("coloring_sweep.txt","w");
    fprintf(f, "# O m_max N_c  lmax_BE lmin_BE  lmax_SE lmin_SE  lmax_BI lmin_BI  lmax_SI lmin_SI\n");
    PetscPrintf(PETSC_COMM_SELF,
        "  O  m_max  N_c | lmax(B,ex) lmax(B,ICC)  lmax(S,ex) lmax(S,ICC) | k(B,ex) k(B,ICC) k(S,ex) k(S,ICC)\n");

    int Ov[6] = {1, 2, 4, 8, 12, 16};
    for (int oi = 0; oi < 6; ++oi) {
        int O = Ov[oi];
        /* build exact and ICC subdomain sets (identical geometry) */
        int NS = P*P;
        Sub *Sx = (Sub*)malloc(sizeof(Sub)*NS);   /* exact (Cholesky) */
        Sub *Si = (Sub*)malloc(sizeof(Sub)*NS);   /* ICC(0)           */
        int sc = 0;
        for (int q=0;q<P;++q) for (int p=0;p<P;++p) {
            PetscInt axlo=(p*n)/P, axhi=((p+1)*n)/P, bylo=(q*n)/P, byhi=((q+1)*n)/P;
            PetscInt al=axlo-O<0?0:axlo-O, ar=axhi+O>n?n:axhi+O;
            PetscInt bl=bylo-O<0?0:bylo-O, br=byhi+O>n?n:byhi+O;
            PetscInt ni=(ar-al)*(br-bl), *idx=(PetscInt*)malloc(sizeof(PetscInt)*ni), c=0;
            for (PetscInt bb=bl;bb<br;++bb) for (PetscInt a=al;a<ar;++a) idx[c++]=bb*n+a;
            BuildSub(A, idx, ni, 0, &Sx[sc]);
            BuildSub(A, idx, ni, 1, &Si[sc]);
            ++sc; free(idx);
        }
        Vec mult = Multiplicity(A, Sx, NS), dsq; VecDuplicate(mult,&dsq);
        VecCopy(mult,dsq); VecReciprocal(dsq); VecSqrtAbs(dsq);

        PetscInt mmax, Nc; Coloring(Sx, NS, A, N, &mmax, &Nc);

        PetscReal lBE,mBE, lSE,mSE, lBI,mBI, lSI,mSI;
        Spectrum(A, Sx, NS, 0, dsq, b, &lBE, &mBE);   /* BASIC exact */
        Spectrum(A, Sx, NS, 1, dsq, b, &lSE, &mSE);   /* sASM  exact */
        Spectrum(A, Si, NS, 0, dsq, b, &lBI, &mBI);   /* BASIC ICC   */
        Spectrum(A, Si, NS, 1, dsq, b, &lSI, &mSI);   /* sASM  ICC   */

        fprintf(f, "%d %d %d  %.6g %.6g  %.6g %.6g  %.6g %.6g  %.6g %.6g\n",
                O,(int)mmax,(int)Nc, lBE,mBE, lSE,mSE, lBI,mBI, lSI,mSI);
        PetscPrintf(PETSC_COMM_SELF,
            " %2d   %2d    %2d | %8.2f   %8.2f   %8.2f   %8.2f | %6.1f %7.1f %6.2f %6.2f\n",
            O,(int)mmax,(int)Nc, lBE,lBI,lSE,lSI,
            lBE/mBE, lBI/mBI, lSE/mSE, lSI/mSI);

        VecDestroy(&mult); VecDestroy(&dsq);
        FreeSubs(Sx,NS); FreeSubs(Si,NS); free(Sx); free(Si);
    }
    fclose(f);
    VecDestroy(&b); VecDestroy(&x); MatDestroy(&A);
    PetscPrintf(PETSC_COMM_SELF,"COLORING_DONE\n");
    PetscFinalize();
    return 0;
}
