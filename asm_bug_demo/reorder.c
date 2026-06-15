/* reorder.c -- does REORDERING turn the subdomain ICC inexact?
 *
 * Claim under test (user Q1): for the 1D Laplace, ICC(0) in the NATURAL ordering
 * equals the exact Cholesky (a tridiagonal matrix has zero fill, so dropping fill
 * drops nothing). If we PERMUTE the unknowns inside each subdomain, the permuted
 * matrix is no longer banded, exact Cholesky now creates fill, and ICC(0) -- which
 * keeps only the original nonzero pattern -- DROPS that fill and becomes inexact.
 *
 * So: same subdomains, same overlap, same multiplicity weights -- only the ORDER
 * in which each subdomain factorizes its ICC changes. Orderings:
 *    0 natural  (identity)          -> 1D: no fill   -> ICC = exact
 *    1 RCM      (reverse Cuthill-McKee, bandwidth-minimizing) -> 1D: still ~banded
 *    2 random   (Fisher-Yates)      -> 1D: scrambled -> ICC drops fill -> inexact
 *
 * For each ordering we report (i) a subdomain-solve inexactness proxy
 *    eta = avg_i || A_i u_i - v || / ||v||,  u_i = ICC-solve(v),
 * which is ~0 when the subdomain solve is exact and grows as ICC degrades, and
 * (ii) the outer CG iteration counts for BASIC and sASM.
 *
 * Mechanism: Aip = P A_i P^T (MatPermute), ICC(0) on Aip; in the apply we gather
 * with idx[perm[a]] and scatter with idx[perm[a]], i.e. apply  P^T ICC(Aip)^-1 P.
 *
 * Sequential (run -n 1). Dumps ASCII consumed by plot_reorder.py.
 */
#include <petscksp.h>
#include <math.h>

typedef struct {
    PetscInt  n;        /* subdomain size (with overlap)              */
    PetscInt *idx;      /* global indices it owns (physical order)    */
    PetscInt *perm;     /* local reordering: perm[a] = physical-local */
    Mat       Aip;      /* P A_i P^T                                  */
    KSP       ksp;      /* preonly + ICC(0) on Aip                    */
    Vec       ri, yi;   /* work (permuted space)                      */
} Sub;

/* ---- 1D / 2D Laplacians (identical to infoprop.c) ---- */
static Mat Laplace1D(PetscInt N) {
    Mat A; MatCreateSeqAIJ(PETSC_COMM_SELF, N, N, 3, NULL, &A);
    for (PetscInt i = 0; i < N; ++i) {
        PetscScalar two = 2.0, m1 = -1.0;
        MatSetValue(A, i, i, two, INSERT_VALUES);
        if (i > 0)   MatSetValue(A, i, i-1, m1, INSERT_VALUES);
        if (i < N-1) MatSetValue(A, i, i+1, m1, INSERT_VALUES);
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE); return A;
}
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

/* deterministic LCG so the random ordering is reproducible across runs */
static unsigned long g_lcg;
static unsigned lcg(void){ g_lcg = g_lcg*6364136223846793005UL + 1442695040888963407UL;
    return (unsigned)(g_lcg >> 33); }

/* build subdomain with a chosen ICC ordering: 0 natural, 1 RCM, 2 random */
static PetscReal BuildSubOrd(Mat A, PetscInt *idx, PetscInt ni, int ordering,
                             unsigned seed, Sub *s)
{
    s->n = ni;
    s->idx  = (PetscInt*)malloc(sizeof(PetscInt)*ni);
    s->perm = (PetscInt*)malloc(sizeof(PetscInt)*ni);
    for (PetscInt j = 0; j < ni; ++j) s->idx[j] = idx[j];

    IS isidx; ISCreateGeneral(PETSC_COMM_SELF, ni, s->idx, PETSC_COPY_VALUES, &isidx);
    Mat Ai; MatCreateSubMatrix(A, isidx, isidx, MAT_INITIAL_MATRIX, &Ai);

    if (ordering == 0) {                       /* natural = identity */
        for (PetscInt a = 0; a < ni; ++a) s->perm[a] = a;
    } else if (ordering == 1) {                /* RCM */
        IS rp, cp; MatGetOrdering(Ai, MATORDERINGRCM, &rp, &cp);
        const PetscInt *pp; ISGetIndices(rp, &pp);
        for (PetscInt a = 0; a < ni; ++a) s->perm[a] = pp[a];
        ISRestoreIndices(rp, &pp); ISDestroy(&rp); ISDestroy(&cp);
    } else {                                   /* random Fisher-Yates */
        for (PetscInt a = 0; a < ni; ++a) s->perm[a] = a;
        g_lcg = 0x9E3779B97F4A7C15UL ^ ((unsigned long)seed * 2654435761UL + 12345UL);
        for (PetscInt a = ni-1; a > 0; --a) {
            PetscInt j = (PetscInt)(lcg() % (unsigned)(a+1));
            PetscInt t = s->perm[a]; s->perm[a] = s->perm[j]; s->perm[j] = t;
        }
    }

    IS isperm; ISCreateGeneral(PETSC_COMM_SELF, ni, s->perm, PETSC_COPY_VALUES, &isperm);
    MatPermute(Ai, isperm, isperm, &s->Aip);   /* Aip[a][b] = Ai[perm[a]][perm[b]] */
    ISDestroy(&isperm); ISDestroy(&isidx);

    KSPCreate(PETSC_COMM_SELF, &s->ksp);
    KSPSetType(s->ksp, KSPPREONLY);
    KSPSetOperators(s->ksp, s->Aip, s->Aip);
    PC pc; KSPGetPC(s->ksp, &pc);
    PCSetType(pc, PCICC); PCFactorSetLevels(pc, 0);
    KSPSetUp(s->ksp);
    MatCreateVecs(s->Aip, &s->ri, &s->yi);

    /* inexactness proxy: relative residual of one ICC solve on a random rhs */
    Vec v, Av; VecDuplicate(s->ri, &v); VecDuplicate(s->ri, &Av);
    PetscScalar *va; VecGetArray(v, &va);
    g_lcg = 0xD1B54A32D192ED03UL ^ ((unsigned long)seed * 40503UL + 7UL);
    for (PetscInt a = 0; a < ni; ++a) va[a] = ((double)(lcg() & 0xFFFF) / 32768.0) - 1.0;
    VecRestoreArray(v, &va);
    PetscReal vn; VecNorm(v, NORM_2, &vn);
    KSPSolve(s->ksp, v, s->yi);                /* yi = ICC^{-1} v */
    MatMult(s->Aip, s->yi, Av); VecAXPY(Av, -1.0, v);
    PetscReal rn; VecNorm(Av, NORM_2, &rn);
    VecDestroy(&v); VecDestroy(&Av); MatDestroy(&Ai);
    return rn / vn;                            /* ~0 exact, grows as ICC degrades */
}

/* rigorous inexactness: kappa = lambda_max/lambda_min of ICC(0)^{-1} Aip,
 * via a KSPCG run with singular-value tracking. =1 when ICC is exact. */
static PetscReal KappaICC(Mat Aip) {
    KSP k; KSPCreate(PETSC_COMM_SELF, &k); KSPSetType(k, KSPCG);
    KSPSetOperators(k, Aip, Aip);
    PC pc; KSPGetPC(k, &pc); PCSetType(pc, PCICC); PCFactorSetLevels(pc, 0);
    KSPSetComputeSingularValues(k, PETSC_TRUE);
    KSPSetTolerances(k, 1e-12, 1e-50, PETSC_DEFAULT, 400);
    Vec b, x; MatCreateVecs(Aip, &b, &x);
    PetscRandom rng; PetscRandomCreate(PETSC_COMM_SELF, &rng); PetscRandomSetSeed(rng, 7); PetscRandomSeed(rng);
    VecSetRandom(b, rng); PetscRandomDestroy(&rng);
    KSPSolve(k, b, x);
    PetscReal emax = 1, emin = 1; KSPComputeExtremeSingularValues(k, &emax, &emin);
    VecDestroy(&b); VecDestroy(&x); KSPDestroy(&k);
    return (emin > 1e-30) ? emax/emin : 1.0;
}

static Vec Multiplicity(Mat A, Sub *subs, PetscInt nsub) {
    Vec m; MatCreateVecs(A, &m, NULL); VecZeroEntries(m);
    PetscScalar *ma; VecGetArray(m, &ma);
    for (PetscInt i = 0; i < nsub; ++i)
        for (PetscInt j = 0; j < subs[i].n; ++j) ma[subs[i].idx[j]] += 1.0;
    VecRestoreArray(m, &ma); return m;
}

/* z = M^{-1} r, applying P^T ICC(Aip)^{-1} P per subdomain via idx[perm[a]] */
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
        for (PetscInt a = 0; a < s->n; ++a) rib[a] = ra[s->idx[s->perm[a]]];
        VecRestoreArray(s->ri, &rib);
        KSPSolve(s->ksp, s->ri, s->yi);
        const PetscScalar *yib; VecGetArrayRead(s->yi, &yib);
        for (PetscInt a = 0; a < s->n; ++a) za[s->idx[s->perm[a]]] += yib[a];
        VecRestoreArrayRead(s->yi, &yib);
    }
    VecRestoreArrayRead(tmp, &ra); VecRestoreArray(z, &za);
    if (useSASM) VecPointwiseMult(z, z, dsq);
}

static PetscInt PCG(Mat A, Sub *subs, PetscInt nsub, int useSASM, Vec dsq,
                    Vec b, Vec x, PetscReal rtol, PetscInt maxit, PetscReal *hist) {
    Vec r, z, p, Ap, tmp;
    VecDuplicate(b,&r); VecDuplicate(b,&z); VecDuplicate(b,&p);
    VecDuplicate(b,&Ap); VecDuplicate(b,&tmp);
    VecZeroEntries(x); VecCopy(b, r);
    PetscReal bnorm; VecNorm(b, NORM_2, &bnorm);
    ApplyPrec(subs, nsub, useSASM, dsq, r, z, tmp);
    VecCopy(z, p);
    PetscScalar rz; VecDot(r, z, &rz);
    PetscInt it; PetscReal rn; VecNorm(r, NORM_2, &rn); hist[0] = rn/bnorm;
    for (it = 0; it < maxit; ++it) {
        MatMult(A, p, Ap);
        PetscScalar pAp; VecDot(p, Ap, &pAp);
        PetscScalar alpha = rz/pAp;
        VecAXPY(x,  alpha, p);
        VecAXPY(r, -alpha, Ap);
        VecNorm(r, NORM_2, &rn); hist[it+1] = rn/bnorm;
        if (rn/bnorm < rtol) { ++it; break; }
        ApplyPrec(subs, nsub, useSASM, dsq, r, z, tmp);
        PetscScalar rz2; VecDot(r, z, &rz2);
        PetscScalar beta = rz2/rz; rz = rz2;
        VecAYPX(p, beta, z);
    }
    VecDestroy(&r); VecDestroy(&z); VecDestroy(&p); VecDestroy(&Ap); VecDestroy(&tmp);
    return it;
}

static void FreeSubs(Sub *S, int n) {
    for (int i = 0; i < n; ++i) { MatDestroy(&S[i].Aip); KSPDestroy(&S[i].ksp);
        VecDestroy(&S[i].ri); VecDestroy(&S[i].yi); free(S[i].idx); free(S[i].perm); }
}

static const char *ONAME[3] = {"natural", "RCM", "random"};
static const char *OTAG[3]  = {"nat", "rcm", "rnd"};

int main(int argc, char **argv) {
    PetscInitialize(&argc, &argv, NULL, NULL);
    FILE *sum = fopen("reorder_summary.txt", "w");
    fprintf(sum, "# dim ordering eta_inexact kappa_icc iterBASIC iter_sASM\n");

    /* =====================  1D : N=256, 8 subdomains, O=2  ===================== */
    {
        const PetscInt N = 256, NSUB = 8, O = 2;
        Mat A = Laplace1D(N);
        Vec b, x; MatCreateVecs(A, &b, &x);
        VecZeroEntries(b); VecSetValue(b, 0, 1.0, INSERT_VALUES);
        VecAssemblyBegin(b); VecAssemblyEnd(b);
        for (int ord = 0; ord < 3; ++ord) {
            Sub S[8]; PetscReal eta = 0, kap = 0;
            for (PetscInt i = 0; i < NSUB; ++i) {
                PetscInt lo=(i*N)/NSUB, hi=((i+1)*N)/NSUB;
                PetscInt a=lo-O<0?0:lo-O, c=hi+O>N?N:hi+O;
                PetscInt ni=c-a, *idx=(PetscInt*)malloc(sizeof(PetscInt)*ni);
                for (PetscInt j=0;j<ni;++j) idx[j]=a+j;
                eta += BuildSubOrd(A, idx, ni, ord, (unsigned)(100*ord+i+1), &S[i]);
                kap += KappaICC(S[i].Aip);
                free(idx);
            }
            eta /= NSUB; kap /= NSUB;
            Vec mult=Multiplicity(A,S,NSUB), dsq; VecDuplicate(mult,&dsq);
            VecCopy(mult,dsq); VecReciprocal(dsq); VecSqrtAbs(dsq);
            PetscReal hB[4000], hS[4000];
            PetscInt iB=PCG(A,S,NSUB,0,dsq,b,x,1e-8,3999,hB);
            PetscInt iS=PCG(A,S,NSUB,1,dsq,b,x,1e-8,3999,hS);
            char fn[96]; sprintf(fn,"reorder_1d_%s_reshist.txt",OTAG[ord]);
            FILE *f=fopen(fn,"w"); PetscInt mx=iB>iS?iB:iS;
            for(PetscInt k=0;k<=mx;++k) fprintf(f,"%d %g %g\n",(int)k,
                k<=iB?(double)hB[k]:-1.0, k<=iS?(double)hS[k]:-1.0);
            fclose(f);
            PetscPrintf(PETSC_COMM_SELF,
                "[1D %-7s] eta=%.3e  kappa=%.3g  BASIC %d  sASM %d\n",ONAME[ord],(double)eta,(double)kap,(int)iB,(int)iS);
            fprintf(sum,"1D %s %.6e %.6e %d %d\n",OTAG[ord],(double)eta,(double)kap,(int)iB,(int)iS);
            VecDestroy(&mult); VecDestroy(&dsq); FreeSubs(S,NSUB);
        }
        VecDestroy(&b); VecDestroy(&x); MatDestroy(&A);
    }

    /* =====================  2D : n=64, 4x4 subdomains, O=2  ===================== */
    {
        const PetscInt n=64, PX=4, PY=4, O=2, NSUB=PX*PY;
        Mat A = Laplace2D(n);
        Vec b, x; MatCreateVecs(A, &b, &x); VecZeroEntries(b);
        { PetscScalar *ba; VecGetArray(b,&ba);
          for(PetscInt bb=0;bb<n;++bb) ba[bb*n+0]=1.0; VecRestoreArray(b,&ba); }
        for (int ord = 0; ord < 3; ++ord) {
            Sub *S=(Sub*)malloc(sizeof(Sub)*NSUB); PetscReal eta=0, kap=0; int sc=0;
            for (int q=0;q<PY;++q) for (int p=0;p<PX;++p) {
                PetscInt axlo=(p*n)/PX, axhi=((p+1)*n)/PX, bylo=(q*n)/PY, byhi=((q+1)*n)/PY;
                PetscInt al=axlo-O<0?0:axlo-O, ar=axhi+O>n?n:axhi+O;
                PetscInt bl=bylo-O<0?0:bylo-O, br=byhi+O>n?n:byhi+O;
                PetscInt ni=(ar-al)*(br-bl), *idx=(PetscInt*)malloc(sizeof(PetscInt)*ni), c=0;
                for(PetscInt bb=bl;bb<br;++bb) for(PetscInt a=al;a<ar;++a) idx[c++]=bb*n+a;
                eta += BuildSubOrd(A, idx, ni, ord, (unsigned)(1000*ord+sc+1), &S[sc]);
                kap += KappaICC(S[sc].Aip); sc++;
                free(idx);
            }
            eta /= NSUB; kap /= NSUB;
            Vec mult=Multiplicity(A,S,NSUB), dsq; VecDuplicate(mult,&dsq);
            VecCopy(mult,dsq); VecReciprocal(dsq); VecSqrtAbs(dsq);
            PetscReal hB[6000], hS[6000];
            PetscInt iB=PCG(A,S,NSUB,0,dsq,b,x,1e-8,5999,hB);
            PetscInt iS=PCG(A,S,NSUB,1,dsq,b,x,1e-8,5999,hS);
            char fn[96]; sprintf(fn,"reorder_2d_%s_reshist.txt",OTAG[ord]);
            FILE *f=fopen(fn,"w"); PetscInt mx=iB>iS?iB:iS;
            for(PetscInt k=0;k<=mx;++k) fprintf(f,"%d %g %g\n",(int)k,
                k<=iB?(double)hB[k]:-1.0, k<=iS?(double)hS[k]:-1.0);
            fclose(f);
            PetscPrintf(PETSC_COMM_SELF,
                "[2D %-7s] eta=%.3e  kappa=%.3g  BASIC %d  sASM %d\n",ONAME[ord],(double)eta,(double)kap,(int)iB,(int)iS);
            fprintf(sum,"2D %s %.6e %.6e %d %d\n",OTAG[ord],(double)eta,(double)kap,(int)iB,(int)iS);
            VecDestroy(&mult); VecDestroy(&dsq); FreeSubs(S,NSUB); free(S);
        }
        VecDestroy(&b); VecDestroy(&x); MatDestroy(&A);
    }

    fclose(sum);
    PetscPrintf(PETSC_COMM_SELF,"REORDER_DONE\n");
    PetscFinalize();
    return 0;
}
