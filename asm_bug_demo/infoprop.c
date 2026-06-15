/* infoprop.c -- information-propagation visualization of BASIC vs sASM.
 *
 * 1D and 2D Laplace, 8 (1D) / 4x4 (2D) subdomains, explicit BASIC and sASM
 * preconditioner apply with EXACT (Cholesky) and ICC(0) subdomain solves.
 * Everything numeric is PETSc (Mat/Vec/KSP/PC); overlap, restriction and the
 * D^{-1/2} weights are controlled by hand so we can dump the overlap "seams".
 *
 * Dumps ASCII files consumed by plot_infoprop.py.  Sequential (run -n 1).
 */
#include <petscksp.h>
#include <math.h>

typedef struct {
    PetscInt  n;        /* size of this subdomain (with overlap) */
    PetscInt *idx;      /* global indices it owns (with overlap) */
    IS        is;
    Mat       Ai;
    KSP       ksp;
    Vec       ri, yi;   /* work: restricted rhs / local solution */
} Sub;

/* ---- assemble 1D P1/FD Laplacian, N interior unknowns, Dirichlet ends ---- */
static Mat Laplace1D(PetscInt N)
{
    Mat A;
    MatCreateSeqAIJ(PETSC_COMM_SELF, N, N, 3, NULL, &A);
    for (PetscInt i = 0; i < N; ++i) {
        PetscScalar two = 2.0, m1 = -1.0;
        MatSetValue(A, i, i, two, INSERT_VALUES);
        if (i > 0)     MatSetValue(A, i, i-1, m1, INSERT_VALUES);
        if (i < N-1)   MatSetValue(A, i, i+1, m1, INSERT_VALUES);
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE);
    return A;
}

/* ---- assemble 2D 5-point Laplacian on n x n interior grid, Dirichlet ---- */
static Mat Laplace2D(PetscInt n)
{
    Mat A; PetscInt N = n*n;
    MatCreateSeqAIJ(PETSC_COMM_SELF, N, N, 5, NULL, &A);
    for (PetscInt b = 0; b < n; ++b)
      for (PetscInt a = 0; a < n; ++a) {
        PetscInt k = b*n + a; PetscScalar four = 4.0, m1 = -1.0;
        MatSetValue(A, k, k, four, INSERT_VALUES);
        if (a > 0)   MatSetValue(A, k, k-1, m1, INSERT_VALUES);
        if (a < n-1) MatSetValue(A, k, k+1, m1, INSERT_VALUES);
        if (b > 0)   MatSetValue(A, k, k-n, m1, INSERT_VALUES);
        if (b < n-1) MatSetValue(A, k, k+n, m1, INSERT_VALUES);
      }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE);
    return A;
}

/* build a subdomain from a list of global indices + exact/icc solver */
static void BuildSub(Mat A, PetscInt *idx, PetscInt ni, int icc, Sub *s)
{
    s->n = ni;
    s->idx = (PetscInt*)malloc(sizeof(PetscInt)*ni);
    for (PetscInt j = 0; j < ni; ++j) s->idx[j] = idx[j];
    ISCreateGeneral(PETSC_COMM_SELF, ni, s->idx, PETSC_COPY_VALUES, &s->is);
    MatCreateSubMatrix(A, s->is, s->is, MAT_INITIAL_MATRIX, &s->Ai);
    KSPCreate(PETSC_COMM_SELF, &s->ksp);
    KSPSetType(s->ksp, KSPPREONLY);
    KSPSetOperators(s->ksp, s->Ai, s->Ai);
    PC pc; KSPGetPC(s->ksp, &pc);
    if (icc) { PCSetType(pc, PCICC); PCFactorSetLevels(pc, 0); }
    else     { PCSetType(pc, PCCHOLESKY); }
    KSPSetUp(s->ksp);
    MatCreateVecs(s->Ai, &s->ri, &s->yi);
}

/* multiplicity vector m[k] = #subdomains containing k */
static Vec Multiplicity(Mat A, Sub *subs, PetscInt nsub)
{
    Vec m; MatCreateVecs(A, &m, NULL); VecZeroEntries(m);
    PetscScalar *ma; VecGetArray(m, &ma);
    for (PetscInt i = 0; i < nsub; ++i)
        for (PetscInt j = 0; j < subs[i].n; ++j) ma[subs[i].idx[j]] += 1.0;
    VecRestoreArray(m, &ma);
    return m;
}

/* z = M^{-1} r .  useSASM: z = D^{-1/2} (sum_i R_i^T A_i^{-1} R_i) D^{-1/2} r */
static void ApplyPrec(Sub *subs, PetscInt nsub, int useSASM, Vec dsq,
                      Vec r, Vec z, Vec tmp)
{
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

/* preconditioned CG; returns #iters, fills relres history (caller-sized) */
static void DumpField(Vec r, PetscInt n, const char *fn)
{
    FILE *f = fopen(fn, "w"); const PetscScalar *ra; VecGetArrayRead(r, &ra);
    for (PetscInt b = 0; b < n; ++b) { for (PetscInt a = 0; a < n; ++a)
        fprintf(f, "%g ", PetscRealPart(ra[b*n+a])); fprintf(f, "\n"); }
    VecRestoreArrayRead(r, &ra); fclose(f);
}

static PetscInt PCG(Mat A, Sub *subs, PetscInt nsub, int useSASM, Vec dsq,
                    Vec b, Vec x, PetscReal rtol, PetscInt maxit,
                    PetscReal *hist,
                    PetscInt n2dump, const char *dumptag, PetscInt *diter, PetscInt ndi)
{
    Vec r, z, p, Ap, tmp;
    VecDuplicate(b, &r); VecDuplicate(b, &z); VecDuplicate(b, &p);
    VecDuplicate(b, &Ap); VecDuplicate(b, &tmp);
    VecZeroEntries(x); VecCopy(b, r);                 /* r = b - A0 */
    PetscReal bnorm; VecNorm(b, NORM_2, &bnorm);
    ApplyPrec(subs, nsub, useSASM, dsq, r, z, tmp);
    VecCopy(z, p);
    PetscScalar rz; VecDot(r, z, &rz);
    PetscInt it = 0; PetscReal rn; VecNorm(r, NORM_2, &rn); hist[0] = rn/bnorm;
    for (it = 0; it < maxit; ++it) {
        MatMult(A, p, Ap);
        PetscScalar pAp; VecDot(p, Ap, &pAp);
        PetscScalar alpha = rz/pAp;
        VecAXPY(x,  alpha, p);
        VecAXPY(r, -alpha, Ap);
        VecNorm(r, NORM_2, &rn); hist[it+1] = rn/bnorm;
        if (n2dump > 0)
            for (PetscInt j = 0; j < ndi; ++j)
                if (it+1 == diter[j]) {
                    char fn[160]; sprintf(fn,"ip2d_pcg_%s_k%d.txt",dumptag,(int)diter[j]);
                    DumpField(r, n2dump, fn);
                }
        if (rn/bnorm < rtol) { ++it; break; }
        ApplyPrec(subs, nsub, useSASM, dsq, r, z, tmp);
        PetscScalar rz2; VecDot(r, z, &rz2);
        PetscScalar beta = rz2/rz; rz = rz2;
        VecAYPX(p, beta, z);                          /* p = z + beta p */
    }
    VecDestroy(&r); VecDestroy(&z); VecDestroy(&p); VecDestroy(&Ap); VecDestroy(&tmp);
    return it;
}

int main(int argc, char **argv)
{
    PetscInitialize(&argc, &argv, NULL, NULL);

    /* =====================  1D  ===================== */
    const PetscInt N1 = 256, NSUB1 = 8;
    Mat A1 = Laplace1D(N1);

    /* RHS for the "information front": left Dirichlet = 1 -> u* ~ 1 - x */
    Vec b1, xstar; MatCreateVecs(A1, &b1, &xstar);
    VecZeroEntries(b1); VecSetValue(b1, 0, 1.0, INSERT_VALUES);
    VecAssemblyBegin(b1); VecAssemblyEnd(b1);
    { KSP kd; KSPCreate(PETSC_COMM_SELF,&kd); KSPSetType(kd,KSPPREONLY);
      KSPSetOperators(kd,A1,A1); PC pc; KSPGetPC(kd,&pc); PCSetType(pc,PCCHOLESKY);
      KSPSolve(kd,b1,xstar); KSPDestroy(&kd); }

    for (int O = 2; O <= 4; O += 2) {
      for (int icc = 0; icc <= 1; ++icc) {
        /* build 8 overlapped subdomains */
        Sub subs[NSUB1];
        for (PetscInt i = 0; i < NSUB1; ++i) {
            PetscInt lo = (i*N1)/NSUB1, hi = ((i+1)*N1)/NSUB1;
            PetscInt a = lo-O < 0 ? 0 : lo-O, c = hi+O > N1 ? N1 : hi+O;
            PetscInt ni = c-a, *idx = (PetscInt*)malloc(sizeof(PetscInt)*ni);
            for (PetscInt j = 0; j < ni; ++j) idx[j] = a+j;
            BuildSub(A1, idx, ni, icc, &subs[i]); free(idx);
        }
        Vec mult = Multiplicity(A1, subs, NSUB1);
        Vec dsq; VecDuplicate(mult,&dsq); VecCopy(mult,dsq);
        VecReciprocal(dsq); VecSqrtAbs(dsq);          /* 1/sqrt(m) */

        Vec r, z, tmp, x; VecDuplicate(b1,&r); VecDuplicate(b1,&z);
        VecDuplicate(b1,&tmp); VecDuplicate(b1,&x);

        /* (a) single apply to a localized bump (only for exact, O) */
        if (icc == 0) {
            VecZeroEntries(r); PetscScalar *ra; VecGetArray(r,&ra);
            for (PetscInt k=0;k<N1;++k){double xx=(double)(k+1)/(N1+1);
                ra[k]=exp(-((xx-0.5)*(xx-0.5))/(2*0.04*0.04));}
            VecRestoreArray(r,&ra);
            Vec zB,zS; VecDuplicate(r,&zB); VecDuplicate(r,&zS);
            ApplyPrec(subs,NSUB1,0,dsq,r,zB,tmp);
            ApplyPrec(subs,NSUB1,1,dsq,r,zS,tmp);
            char fn[128]; sprintf(fn,"ip1d_apply_O%d.txt",O);
            FILE *f=fopen(fn,"w");
            const PetscScalar *ma,*rb,*zb,*zs; VecGetArrayRead(mult,&ma);
            VecGetArrayRead(r,&rb);VecGetArrayRead(zB,&zb);VecGetArrayRead(zS,&zs);
            for(PetscInt k=0;k<N1;++k) fprintf(f,"%g %g %g %g %g\n",
                (double)(k+1)/(N1+1),PetscRealPart(ma[k]),PetscRealPart(rb[k]),
                PetscRealPart(zb[k]),PetscRealPart(zs[k]));
            VecRestoreArrayRead(mult,&ma);VecRestoreArrayRead(r,&rb);
            VecRestoreArrayRead(zB,&zb);VecRestoreArrayRead(zS,&zs);
            fclose(f); VecDestroy(&zB); VecDestroy(&zS);
        }

        /* (b) stationary iteration front: x_{k+1}=x_k + M^{-1}(b-Ax_k) */
        const char *tag = icc ? "icc" : "exact";
        for (int useSASM = 0; useSASM <= 1; ++useSASM) {
            VecZeroEntries(x);
            char fn[128]; sprintf(fn,"ip1d_front_O%d_%s_%s.txt",O,tag,useSASM?"sASM":"BASIC");
            FILE *f=fopen(fn,"w");
            int Ks[6]={0,1,2,4,8,16}; int kidx=0;
            for (int k=0;k<=16;++k){
                if (k==Ks[kidx]){
                    const PetscScalar *xa; VecGetArrayRead(x,&xa);
                    fprintf(f,"# k=%d\n",k);
                    for(PetscInt q=0;q<N1;++q) fprintf(f,"%g %g\n",
                        (double)(q+1)/(N1+1),PetscRealPart(xa[q]));
                    VecRestoreArrayRead(x,&xa); fprintf(f,"\n"); ++kidx;
                }
                MatMult(A1,x,r); VecAYPX(r,-1.0,b1);   /* r=b-Ax */
                ApplyPrec(subs,NSUB1,useSASM,dsq,r,z,tmp);
                VecAXPY(x,1.0,z);
            }
            fclose(f);
        }

        /* (c) residual history (PCG) */
        if (O == 2) {
            PetscReal hB[2000],hS[2000];
            PetscInt iB=PCG(A1,subs,NSUB1,0,dsq,b1,x,1e-8,1999,hB,0,NULL,NULL,0);
            PetscInt iS=PCG(A1,subs,NSUB1,1,dsq,b1,x,1e-8,1999,hS,0,NULL,NULL,0);
            char fn[128]; sprintf(fn,"ip1d_reshist_%s.txt",tag);
            FILE *f=fopen(fn,"w");
            PetscInt mx=iB>iS?iB:iS;
            for(PetscInt k=0;k<=mx;++k) fprintf(f,"%d %g %g\n",(int)k,
                k<=iB?(double)hB[k]:-1.0, k<=iS?(double)hS[k]:-1.0);
            fclose(f);
            PetscPrintf(PETSC_COMM_SELF,"[1D %s] BASIC %d iters, sASM %d iters\n",tag,(int)iB,(int)iS);
        }

        VecDestroy(&r);VecDestroy(&z);VecDestroy(&tmp);VecDestroy(&x);
        VecDestroy(&mult);VecDestroy(&dsq);
        for(PetscInt i=0;i<NSUB1;++i){ISDestroy(&subs[i].is);MatDestroy(&subs[i].Ai);
            KSPDestroy(&subs[i].ksp);VecDestroy(&subs[i].ri);VecDestroy(&subs[i].yi);
            free(subs[i].idx);}
      }
    }
    VecDestroy(&b1);VecDestroy(&xstar);MatDestroy(&A1);

    /* =====================  2D  ===================== */
    const PetscInt n2 = 64, PX = 4, PY = 4, O2 = 2, N2 = n2*n2;
    Mat A2 = Laplace2D(n2);
    Vec b2; MatCreateVecs(A2,&b2,NULL);
    VecZeroEntries(b2);                       /* left edge Dirichlet = 1 */
    { PetscScalar *ba; VecGetArray(b2,&ba);
      for(PetscInt b=0;b<n2;++b) ba[b*n2+0]=1.0; VecRestoreArray(b2,&ba); }

    int NSUB2 = PX*PY;
    Sub *subs2 = (Sub*)malloc(sizeof(Sub)*NSUB2);
    int sc=0;
    for(int q=0;q<PY;++q) for(int p=0;p<PX;++p){
        PetscInt axlo=(p*n2)/PX, axhi=((p+1)*n2)/PX;
        PetscInt bylo=(q*n2)/PY, byhi=((q+1)*n2)/PY;
        PetscInt al=axlo-O2<0?0:axlo-O2, ar=axhi+O2>n2?n2:axhi+O2;
        PetscInt bl=bylo-O2<0?0:bylo-O2, br=byhi+O2>n2?n2:byhi+O2;
        PetscInt ni=(ar-al)*(br-bl), *idx=(PetscInt*)malloc(sizeof(PetscInt)*ni),c=0;
        for(PetscInt b=bl;b<br;++b) for(PetscInt a=al;a<ar;++a) idx[c++]=b*n2+a;
        BuildSub(A2,idx,ni,0,&subs2[sc++]); free(idx);
    }
    Vec mult2=Multiplicity(A2,subs2,NSUB2);
    Vec dsq2; VecDuplicate(mult2,&dsq2);VecCopy(mult2,dsq2);
    VecReciprocal(dsq2);VecSqrtAbs(dsq2);
    Vec r2,z2,tmp2,x2; VecDuplicate(b2,&r2);VecDuplicate(b2,&z2);
    VecDuplicate(b2,&tmp2);VecDuplicate(b2,&x2);

    for(int useSASM=0;useSASM<=1;++useSASM){
        VecZeroEntries(x2);
        int Ks[5]={1,2,3,5,8}; int kidx=0;
        for(int k=1;k<=8;++k){
            MatMult(A2,x2,r2); VecAYPX(r2,-1.0,b2);     /* r=b-Ax */
            if(k==Ks[kidx]){
                char fn[128]; sprintf(fn,"ip2d_res_k%d_%s.txt",k,useSASM?"sASM":"BASIC");
                FILE *f=fopen(fn,"w");
                const PetscScalar *ra; VecGetArrayRead(r2,&ra);
                for(PetscInt b=0;b<n2;++b){for(PetscInt a=0;a<n2;++a)
                    fprintf(f,"%g ",PetscRealPart(ra[b*n2+a])); fprintf(f,"\n");}
                VecRestoreArrayRead(r2,&ra); fclose(f); ++kidx;
            }
            ApplyPrec(subs2,NSUB2,useSASM,dsq2,r2,z2,tmp2);
            VecAXPY(x2,1.0,z2);
        }
    }
    /* ---- 2D residual history: the anomaly lives here, because ICC(0) of the
     *      5-point 2D Laplacian is GENUINELY incomplete (unlike 1D, where ICC=exact).
     *      Compare BASIC vs sASM with exact and with ICC(0) subdomain solves. ---- */
    for (int icc = 0; icc <= 1; ++icc) {
        Sub *S; int owns = 0;
        if (icc == 0) { S = subs2; }
        else {
            owns = 1; S = (Sub*)malloc(sizeof(Sub)*NSUB2); int s2=0;
            for (int q=0;q<PY;++q) for (int p=0;p<PX;++p) {
                PetscInt axlo=(p*n2)/PX, axhi=((p+1)*n2)/PX;
                PetscInt bylo=(q*n2)/PY, byhi=((q+1)*n2)/PY;
                PetscInt al=axlo-O2<0?0:axlo-O2, ar=axhi+O2>n2?n2:axhi+O2;
                PetscInt bl=bylo-O2<0?0:bylo-O2, br=byhi+O2>n2?n2:byhi+O2;
                PetscInt ni=(ar-al)*(br-bl), *idx=(PetscInt*)malloc(sizeof(PetscInt)*ni),c=0;
                for(PetscInt b=bl;b<br;++b) for(PetscInt a=al;a<ar;++a) idx[c++]=b*n2+a;
                BuildSub(A2,idx,ni,1,&S[s2++]); free(idx);
            }
        }
        const char *t2 = icc ? "icc" : "exact";
        PetscInt dit[4] = {5,15,30,60}; PetscInt nd = icc ? 4 : 0;
        char tagB[40], tagS[40]; sprintf(tagB,"BASIC_%s",t2); sprintf(tagS,"sASM_%s",t2);
        PetscReal hB[5000], hS[5000];
        PetscInt iB = PCG(A2, S, NSUB2, 0, dsq2, b2, x2, 1e-8, 4999, hB, icc?n2:0, tagB, dit, nd);
        PetscInt iS = PCG(A2, S, NSUB2, 1, dsq2, b2, x2, 1e-8, 4999, hS, icc?n2:0, tagS, dit, nd);
        char fn[128]; sprintf(fn,"ip2d_reshist_%s.txt",t2);
        FILE *f=fopen(fn,"w"); PetscInt mx=iB>iS?iB:iS;
        for(PetscInt k=0;k<=mx;++k) fprintf(f,"%d %g %g\n",(int)k,
            k<=iB?(double)hB[k]:-1.0, k<=iS?(double)hS[k]:-1.0);
        fclose(f);
        PetscPrintf(PETSC_COMM_SELF,"[2D %s] BASIC %d iters, sASM %d iters\n",t2,(int)iB,(int)iS);
        if (owns) { for(int i=0;i<NSUB2;++i){ISDestroy(&S[i].is);MatDestroy(&S[i].Ai);
            KSPDestroy(&S[i].ksp);VecDestroy(&S[i].ri);VecDestroy(&S[i].yi);free(S[i].idx);}
            free(S); }
    }

    /* dump 2D multiplicity for reference */
    { FILE *f=fopen("ip2d_mult.txt","w"); const PetscScalar *ma;
      VecGetArrayRead(mult2,&ma);
      for(PetscInt b=0;b<n2;++b){for(PetscInt a=0;a<n2;++a)
          fprintf(f,"%g ",PetscRealPart(ma[b*n2+a])); fprintf(f,"\n");}
      VecRestoreArrayRead(mult2,&ma); fclose(f); }

    PetscPrintf(PETSC_COMM_SELF,"[2D] dumped residual fields (4x4 subdomains, O=%d)\n",(int)O2);
    VecDestroy(&r2);VecDestroy(&z2);VecDestroy(&tmp2);VecDestroy(&x2);
    VecDestroy(&mult2);VecDestroy(&dsq2);VecDestroy(&b2);MatDestroy(&A2);
    for(int i=0;i<NSUB2;++i){ISDestroy(&subs2[i].is);MatDestroy(&subs2[i].Ai);
        KSPDestroy(&subs2[i].ksp);VecDestroy(&subs2[i].ri);VecDestroy(&subs2[i].yi);
        free(subs2[i].idx);}
    free(subs2);

    PetscPrintf(PETSC_COMM_SELF,"INFOPROP_DONE\n");
    PetscFinalize();
    return 0;
}
