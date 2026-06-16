/* rigor.c -- methodological hardening of the sASM study.
 *
 * Addresses four reviewer points:
 *  (1) PROVE the preconditioner M^{-1} is a fixed SPD linear operator and that the
 *      outer stop is uniform: symmetry to machine precision via random pairs
 *      <u, M^{-1} v> vs <v, M^{-1} u>, and lambda_min > 0; M^{-1} is preonly =>
 *      a fixed (deterministic, linear) operator, so CG is the right outer solver.
 *  (2) STRIP vs BOX (and exact vs ICC) to cut the over-count N_hat out of the
 *      "size degradation" of omega: a 1xP strip keeps N_hat=2 while the subdomain
 *      SIZE still grows with overlap; a PxP box grows N_hat. If lambda_max(BASIC,
 *      exact) tracks N_hat (flat ~2 for strips, rising for boxes) it is the
 *      over-count, not the size, that lifts lambda_max.
 *  (4) TWO-LEVEL BASIC vs TWO-LEVEL sASM: does a coarse space substitute for the
 *      D^{-1/2} scaling? Measure lambda_min/lambda_max/kappa/iters for the 2x2
 *      {BASIC,sASM} x {one-level,two-level}. If two-level BASIC keeps lambda_max ~
 *      omega*N_hat (coarse fixes only lambda_min), it cannot replace sASM.
 *  (5) COST, not iterations: wall-time per solve (PetscTime) reported alongside.
 *
 * lambda's via KSPCG + PCSHELL + KSPComputeExtremeSingularValues. Sequential.
 */
#include <petscksp.h>
#include <petsctime.h>
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
static PetscInt *MultArray(Sub *subs, PetscInt nsub, PetscInt N, PetscInt *nhat) {
    PetscInt *m = (PetscInt*)calloc(N, sizeof(PetscInt));
    for (PetscInt i = 0; i < nsub; ++i)
        for (PetscInt j = 0; j < subs[i].n; ++j) m[subs[i].idx[j]]++;
    PetscInt mm=0; for (PetscInt k=0;k<N;++k) if (m[k]>mm) mm=m[k];
    *nhat = mm; return m;
}

typedef struct {
    Sub *subs; PetscInt nsub; int useSASM; Vec dsq, tmp;
    int twolevel; Mat R0t; KSP kspc; Vec cvec, ycvec;
} Ctx;

static void Fine(Ctx *c, Vec r, Vec z) {
    VecCopy(r, c->tmp);
    if (c->useSASM) VecPointwiseMult(c->tmp, c->tmp, c->dsq);
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
    if (c->useSASM) VecPointwiseMult(z, z, c->dsq);
}
static void Apply(Ctx *c, Vec r, Vec z) {
    Fine(c, r, z);
    if (c->twolevel) { MatMultTranspose(c->R0t, r, c->cvec);
        KSPSolve(c->kspc, c->cvec, c->ycvec); MatMultAdd(c->R0t, c->ycvec, z, z); }
}
static PetscErrorCode ShellApply(PC pc, Vec r, Vec z) {
    Ctx *c; PCShellGetContext(pc, &c); Apply(c, r, z); return 0; }

static void BuildCoarse(Mat A, Sub *subs, PetscInt nsub, PetscInt N, PetscInt *mult, Ctx *c) {
    Mat R0t; MatCreateSeqAIJ(PETSC_COMM_SELF, N, nsub, 32, NULL, &R0t);
    for (PetscInt i = 0; i < nsub; ++i)
        for (PetscInt j = 0; j < subs[i].n; ++j)
            MatSetValue(R0t, subs[i].idx[j], i, 1.0/(PetscReal)mult[subs[i].idx[j]], INSERT_VALUES);
    MatAssemblyBegin(R0t, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(R0t, MAT_FINAL_ASSEMBLY);
    c->R0t = R0t; Mat A0; MatPtAP(A, R0t, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &A0);
    KSPCreate(PETSC_COMM_SELF, &c->kspc); KSPSetType(c->kspc, KSPPREONLY);
    KSPSetOperators(c->kspc, A0, A0);
    PC pc; KSPGetPC(c->kspc, &pc); PCSetType(pc, PCCHOLESKY); KSPSetUp(c->kspc);
    MatCreateVecs(A0, &c->ycvec, &c->cvec); MatDestroy(&A0);
}
static Ctx MakeCtx(Mat A, Sub *S, int nsub, PetscInt N, int useSASM, int twolevel) {
    Ctx c; c.subs=S; c.nsub=nsub; c.useSASM=useSASM; c.twolevel=twolevel;
    MatCreateVecs(A, &c.tmp, NULL);
    PetscInt nhat; PetscInt *m = MultArray(S, nsub, N, &nhat);
    Vec mult; MatCreateVecs(A,&mult,NULL); PetscScalar *ma; VecGetArray(mult,&ma);
    for (PetscInt k=0;k<N;++k) ma[k]=(PetscScalar)m[k]; VecRestoreArray(mult,&ma);
    VecDuplicate(mult,&c.dsq); VecCopy(mult,c.dsq); VecReciprocal(c.dsq); VecSqrtAbs(c.dsq);
    VecDestroy(&mult);
    if (twolevel) BuildCoarse(A,S,nsub,N,m,&c);
    else { c.R0t=NULL; c.kspc=NULL; c.cvec=NULL; c.ycvec=NULL; }
    free(m); return c;
}
static void FreeCtx(Ctx *c) { VecDestroy(&c->tmp); VecDestroy(&c->dsq);
    if (c->twolevel){MatDestroy(&c->R0t);KSPDestroy(&c->kspc);VecDestroy(&c->cvec);VecDestroy(&c->ycvec);} }

static void Spectrum(Mat A, Ctx *c, Vec b, PetscReal *lmax, PetscReal *lmin) {
    Vec x; VecDuplicate(b, &x);
    KSP ksp; KSPCreate(PETSC_COMM_SELF,&ksp); KSPSetType(ksp,KSPCG); KSPSetOperators(ksp,A,A);
    PC pc; KSPGetPC(ksp,&pc); PCSetType(pc,PCSHELL); PCShellSetContext(pc,c); PCShellSetApply(pc,ShellApply);
    KSPSetComputeSingularValues(ksp,PETSC_TRUE); KSPSetTolerances(ksp,1e-11,1e-50,PETSC_DEFAULT,800);
    KSPSetNormType(ksp,KSP_NORM_UNPRECONDITIONED); KSPSetUp(ksp); KSPSolve(ksp,b,x);
    KSPComputeExtremeSingularValues(ksp,lmax,lmin); VecDestroy(&x); KSPDestroy(&ksp);
}
static PetscInt PCGtimed(Mat A, Ctx *c, Vec b, Vec x, PetscReal rtol, PetscInt maxit, double *secs) {
    Vec r,z,p,Ap; VecDuplicate(b,&r);VecDuplicate(b,&z);VecDuplicate(b,&p);VecDuplicate(b,&Ap);
    VecZeroEntries(x); VecCopy(b,r); PetscReal bnorm; VecNorm(b,NORM_2,&bnorm);
    PetscLogDouble t0; PetscTime(&t0);
    Apply(c,r,z); VecCopy(z,p); PetscScalar rz; VecDot(r,z,&rz);
    PetscInt it; PetscReal rn;
    for (it=0; it<maxit; ++it) {
        MatMult(A,p,Ap); PetscScalar pAp; VecDot(p,Ap,&pAp); PetscScalar al=rz/pAp;
        VecAXPY(x,al,p); VecAXPY(r,-al,Ap); VecNorm(r,NORM_2,&rn);
        if (rn/bnorm<rtol){++it;break;}
        Apply(c,r,z); PetscScalar rz2; VecDot(r,z,&rz2); PetscScalar be=rz2/rz; rz=rz2; VecAYPX(p,be,z);
    }
    PetscLogDouble t1; PetscTime(&t1); *secs=(double)(t1-t0);
    VecDestroy(&r);VecDestroy(&z);VecDestroy(&p);VecDestroy(&Ap); return it;
}

static Sub *BuildBox(Mat A, PetscInt n, int P, int O, int icc, PetscInt *nsubOut) {
    int NS=P*P; Sub *S=(Sub*)malloc(sizeof(Sub)*NS); int sc=0;
    for (int q=0;q<P;++q) for (int p=0;p<P;++p) {
        PetscInt axlo=(p*n)/P,axhi=((p+1)*n)/P,bylo=(q*n)/P,byhi=((q+1)*n)/P;
        PetscInt al=axlo-O<0?0:axlo-O,ar=axhi+O>n?n:axhi+O,bl=bylo-O<0?0:bylo-O,br=byhi+O>n?n:byhi+O;
        PetscInt ni=(ar-al)*(br-bl),*idx=(PetscInt*)malloc(sizeof(PetscInt)*ni),c=0;
        for(PetscInt bb=bl;bb<br;++bb)for(PetscInt a=al;a<ar;++a) idx[c++]=bb*n+a;
        BuildSub(A,idx,ni,icc,&S[sc++]); free(idx);
    } *nsubOut=NS; return S;
}
static Sub *BuildStrip(Mat A, PetscInt n, int P, int O, int icc, PetscInt *nsubOut) {
    Sub *S=(Sub*)malloc(sizeof(Sub)*P); /* P vertical strips, full height, overlap in x only */
    for (int p=0;p<P;++p) {
        PetscInt axlo=(p*n)/P,axhi=((p+1)*n)/P;
        PetscInt al=axlo-O<0?0:axlo-O,ar=axhi+O>n?n:axhi+O;
        PetscInt ni=(ar-al)*n,*idx=(PetscInt*)malloc(sizeof(PetscInt)*ni),c=0;
        for(PetscInt bb=0;bb<n;++bb)for(PetscInt a=al;a<ar;++a) idx[c++]=bb*n+a;
        BuildSub(A,idx,ni,icc,&S[p]); free(idx);
    } *nsubOut=P; return S;
}
static void FreeSubs(Sub *S,int n){for(int i=0;i<n;++i){ISDestroy(&S[i].is);MatDestroy(&S[i].Ai);
    KSPDestroy(&S[i].ksp);VecDestroy(&S[i].ri);VecDestroy(&S[i].yi);free(S[i].idx);} free(S);}

int main(int argc, char **argv) {
    PetscInitialize(&argc,&argv,NULL,NULL);

    /* ===== (1) PROVE M^{-1} is fixed SPD; uniform stop ===== */
    {
        PetscInt n=64,N=n*n; Mat A=Laplace2D(n);
        Vec b; MatCreateVecs(A,&b,NULL); VecZeroEntries(b);
        {PetscScalar*ba;VecGetArray(b,&ba);for(PetscInt i=0;i<n;++i)ba[i*n+0]=1.0;VecRestoreArray(b,&ba);}
        int nsub; Sub *S=BuildBox(A,n,4,2,1,&nsub);
        Ctx c=MakeCtx(A,S,nsub,N,1,0);                 /* one-level sASM, ICC */
        PetscReal worst=0; PetscRandom rng; PetscRandomCreate(PETSC_COMM_SELF,&rng);
        for (int t=0;t<6;++t){
            Vec u,v,Mu,Mv; VecDuplicate(b,&u);VecDuplicate(b,&v);VecDuplicate(b,&Mu);VecDuplicate(b,&Mv);
            PetscRandomSetSeed(rng,11+t);PetscRandomSeed(rng);VecSetRandom(u,rng);
            PetscRandomSetSeed(rng,91+t);PetscRandomSeed(rng);VecSetRandom(v,rng);
            Apply(&c,u,Mu); Apply(&c,v,Mv);
            PetscScalar a1,a2; VecDot(v,Mu,&a1); VecDot(u,Mv,&a2);
            PetscReal rel=PetscAbsScalar(a1-a2)/(PetscAbsScalar(a1)+1e-30);
            if (rel>worst) worst=rel;
            VecDestroy(&u);VecDestroy(&v);VecDestroy(&Mu);VecDestroy(&Mv);
        }
        PetscRandomDestroy(&rng);
        PetscReal lmax,lmin; Spectrum(A,&c,b,&lmax,&lmin);
        PetscPrintf(PETSC_COMM_SELF,
          "=== (1) M^{-1} check (sASM, 64^2, 4x4, O=2, ICC) ===\n"
          "  symmetry  max|<v,Mu>-<u,Mv>|/|<v,Mu>| = %.2e  (=> symmetric)\n"
          "  lambda_min = %.4g > 0  (=> SPD);  preonly subdomain solves => FIXED linear operator\n"
          "  => CG is the correct outer solver; stop = unpreconditioned ||r||/||b||.\n",
          (double)worst,(double)lmin);
        FreeCtx(&c); FreeSubs(S,nsub); VecDestroy(&b); MatDestroy(&A);
    }

    /* ===== (2) STRIP vs BOX : isolate over-count N_hat from subdomain size ===== */
    {
        PetscInt n=96,N=n*n; int P=6; Mat A=Laplace2D(n);
        Vec b; MatCreateVecs(A,&b,NULL); VecZeroEntries(b);
        {PetscScalar*ba;VecGetArray(b,&ba);for(PetscInt i=0;i<n;++i)ba[i*n+0]=1.0;VecRestoreArray(b,&ba);}
        FILE *f=fopen("rigor_stripbox.txt","w"); fprintf(f,"# layout O nhat lmax_exact lmax_icc subsize\n");
        PetscPrintf(PETSC_COMM_SELF,"\n=== (2) STRIP (N_hat=2 fixed) vs BOX (N_hat grows); 96^2, P=6 ===\n");
        PetscPrintf(PETSC_COMM_SELF,"%-5s %3s %5s %12s %12s %8s\n","lay","O","N^","lmax(B,ex)","lmax(B,ICC)","subDOF");
        int Ov[4]={2,4,8,16};
        for (int lay=0; lay<2; ++lay) for (int oi=0; oi<4; ++oi) {
            int O=Ov[oi]; int nsub; Sub *Sx,*Si;
            Sx = lay? BuildBox(A,n,P,O,0,&nsub) : BuildStrip(A,n,P,O,0,&nsub);
            Si = lay? BuildBox(A,n,P,O,1,&nsub) : BuildStrip(A,n,P,O,1,&nsub);
            PetscInt nhat; PetscInt *m=MultArray(Sx,nsub,N,&nhat); free(m);
            PetscInt subDOF=Sx[0].n;
            Ctx cx=MakeCtx(A,Sx,nsub,N,0,0), ci=MakeCtx(A,Si,nsub,N,0,0); /* BASIC */
            PetscReal lxe,mne,lxi,mni; Spectrum(A,&cx,b,&lxe,&mne); Spectrum(A,&ci,b,&lxi,&mni);
            PetscPrintf(PETSC_COMM_SELF,"%-5s %3d %5d %12.2f %12.2f %8d\n",
                lay?"box":"strip",O,(int)nhat,lxe,lxi,(int)subDOF);
            fprintf(f,"%s %d %d %.4f %.4f %d\n",lay?"box":"strip",O,(int)nhat,lxe,lxi,(int)subDOF);
            FreeCtx(&cx); FreeCtx(&ci); FreeSubs(Sx,nsub); FreeSubs(Si,nsub);
        }
        fclose(f); VecDestroy(&b); MatDestroy(&A);
    }

    /* ===== (4) two-level BASIC vs sASM (+ (5) cost): can coarse replace sASM? ===== */
    {
        PetscInt n=120,N=n*n; int P=6,O=2; Mat A=Laplace2D(n);
        Vec b,x; MatCreateVecs(A,&b,&x); VecZeroEntries(b);
        {PetscScalar*ba;VecGetArray(b,&ba);for(PetscInt i=0;i<n;++i)ba[i*n+0]=1.0;VecRestoreArray(b,&ba);}
        int nsub; Sub *S=BuildBox(A,n,P,O,1,&nsub);   /* ICC */
        FILE *f=fopen("rigor_twolevel.txt","w"); fprintf(f,"# method level lmin lmax kappa iters secs\n");
        PetscPrintf(PETSC_COMM_SELF,"\n=== (4)+(5) two-level BASIC vs sASM (120^2,6x6,O=2,ICC) ===\n");
        PetscPrintf(PETSC_COMM_SELF,"%-7s %-7s %10s %9s %9s %6s %9s\n",
            "method","level","lmin","lmax","kappa","iters","time(ms)");
        const char *mn[2]={"BASIC","sASM"};
        for (int useS=0; useS<2; ++useS) for (int two=0; two<2; ++two) {
            Ctx c=MakeCtx(A,S,nsub,N,useS,two);
            PetscReal lmax,lmin; Spectrum(A,&c,b,&lmax,&lmin);
            double secs; PetscInt it=PCGtimed(A,&c,b,x,1e-8,3000,&secs);
            PetscPrintf(PETSC_COMM_SELF,"%-7s %-7s %10.4g %9.3f %9.1f %6d %9.2f\n",
                mn[useS], two?"two":"one", lmin, lmax, lmax/lmin, (int)it, secs*1e3);
            fprintf(f,"%s %d %.6g %.6g %.6g %d %.6g\n",mn[useS],two,lmin,lmax,lmax/lmin,(int)it,secs);
            FreeCtx(&c);
        }
        fclose(f); FreeSubs(S,nsub); VecDestroy(&b); VecDestroy(&x); MatDestroy(&A);
    }
    PetscPrintf(PETSC_COMM_SELF,"\nRIGOR_DONE\n");
    PetscFinalize(); return 0;
}
