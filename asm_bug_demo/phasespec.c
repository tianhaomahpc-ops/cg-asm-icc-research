/* phasespec.c -- reviewer point (3): locate WHERE the mechanism acts.
 *
 * Part A  FULL SPECTRUM: the CG Ritz values (KSPComputeEigenvalues) approximate the
 *   whole spectrum of M^{-1}A. For {BASIC,sASM} x {exact,ICC(0)} we dump them to show
 *   the over-count lives at the HIGH end (lambda_max outliers, removed by sASM) while
 *   the global/propagation mode lives at the LOW end (a small-lambda cluster, common
 *   to all one-level methods).
 *
 * Part B  PHASE DIAGRAM (overlap x inner accuracy): sweep overlap O against ICC fill
 *   level L (L=0 most inexact .. higher L -> omega->1; L<0 = exact Cholesky). For
 *   BASIC and sASM dump kappa(M^{-1}A). The anomaly (kappa up with O) lives only at
 *   low L (large omega); at high L overlap helps -- the A x B interaction made a map.
 *
 * Sequential. Dumps phasespec_eigs_*.txt and phasespec_phase.txt.
 */
#include <petscksp.h>
#include <math.h>

typedef struct { PetscInt n; PetscInt *idx; IS is; Mat Ai; KSP ksp; Vec ri, yi; } Sub;

static Mat Laplace2D(PetscInt n) {
    Mat A; PetscInt N=n*n; MatCreateSeqAIJ(PETSC_COMM_SELF,N,N,5,NULL,&A);
    for (PetscInt b=0;b<n;++b) for (PetscInt a=0;a<n;++a){
        PetscInt k=b*n+a; PetscScalar four=4.0,m1=-1.0;
        MatSetValue(A,k,k,four,INSERT_VALUES);
        if(a>0)MatSetValue(A,k,k-1,m1,INSERT_VALUES); if(a<n-1)MatSetValue(A,k,k+1,m1,INSERT_VALUES);
        if(b>0)MatSetValue(A,k,k-n,m1,INSERT_VALUES); if(b<n-1)MatSetValue(A,k,k+n,m1,INSERT_VALUES);
    }
    MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
    MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE); return A;
}
/* level<0 => exact Cholesky; level>=0 => ICC(level) */
static void BuildSub(Mat A, PetscInt *idx, PetscInt ni, int level, Sub *s) {
    s->n=ni; s->idx=(PetscInt*)malloc(sizeof(PetscInt)*ni);
    for(PetscInt j=0;j<ni;++j) s->idx[j]=idx[j];
    ISCreateGeneral(PETSC_COMM_SELF,ni,s->idx,PETSC_COPY_VALUES,&s->is);
    MatCreateSubMatrix(A,s->is,s->is,MAT_INITIAL_MATRIX,&s->Ai);
    KSPCreate(PETSC_COMM_SELF,&s->ksp); KSPSetType(s->ksp,KSPPREONLY);
    KSPSetOperators(s->ksp,s->Ai,s->Ai);
    PC pc; KSPGetPC(s->ksp,&pc);
    if(level<0){PCSetType(pc,PCCHOLESKY);} else {PCSetType(pc,PCICC);PCFactorSetLevels(pc,level);}
    KSPSetUp(s->ksp); MatCreateVecs(s->Ai,&s->ri,&s->yi);
}
typedef struct { Sub *subs; PetscInt nsub; int useSASM; Vec dsq,tmp; } Ctx;
static void Apply(Ctx *c, Vec r, Vec z){
    VecCopy(r,c->tmp); if(c->useSASM)VecPointwiseMult(c->tmp,c->tmp,c->dsq);
    VecZeroEntries(z); const PetscScalar*ra; PetscScalar*za;
    VecGetArrayRead(c->tmp,&ra);VecGetArray(z,&za);
    for(PetscInt i=0;i<c->nsub;++i){ Sub*s=&c->subs[i]; PetscScalar*rib; VecGetArray(s->ri,&rib);
        for(PetscInt j=0;j<s->n;++j) rib[j]=ra[s->idx[j]]; VecRestoreArray(s->ri,&rib);
        KSPSolve(s->ksp,s->ri,s->yi); const PetscScalar*yib; VecGetArrayRead(s->yi,&yib);
        for(PetscInt j=0;j<s->n;++j) za[s->idx[j]]+=yib[j]; VecRestoreArrayRead(s->yi,&yib); }
    VecRestoreArrayRead(c->tmp,&ra);VecRestoreArray(z,&za);
    if(c->useSASM)VecPointwiseMult(z,z,c->dsq);
}
static PetscErrorCode ShellApply(PC pc,Vec r,Vec z){Ctx*c;PCShellGetContext(pc,&c);Apply(c,r,z);return 0;}
static Ctx MakeCtx(Mat A,Sub*S,int nsub,int useSASM){
    Ctx c; c.subs=S;c.nsub=nsub;c.useSASM=useSASM; MatCreateVecs(A,&c.tmp,NULL);
    Vec mult; MatCreateVecs(A,&mult,NULL);VecZeroEntries(mult); PetscScalar*ma;VecGetArray(mult,&ma);
    for(int i=0;i<nsub;++i)for(PetscInt j=0;j<S[i].n;++j) ma[S[i].idx[j]]+=1.0; VecRestoreArray(mult,&ma);
    VecDuplicate(mult,&c.dsq);VecCopy(mult,c.dsq);VecReciprocal(c.dsq);VecSqrtAbs(c.dsq);VecDestroy(&mult);
    return c;
}
static Sub *BuildBox(Mat A,PetscInt n,int P,int O,int level,int*ns){
    int NS=P*P; Sub*S=(Sub*)malloc(sizeof(Sub)*NS); int sc=0;
    for(int q=0;q<P;++q)for(int p=0;p<P;++p){
        PetscInt axlo=(p*n)/P,axhi=((p+1)*n)/P,bylo=(q*n)/P,byhi=((q+1)*n)/P;
        PetscInt al=axlo-O<0?0:axlo-O,ar=axhi+O>n?n:axhi+O,bl=bylo-O<0?0:bylo-O,br=byhi+O>n?n:byhi+O;
        PetscInt ni=(ar-al)*(br-bl),*idx=(PetscInt*)malloc(sizeof(PetscInt)*ni),c=0;
        for(PetscInt bb=bl;bb<br;++bb)for(PetscInt a=al;a<ar;++a) idx[c++]=bb*n+a;
        BuildSub(A,idx,ni,level,&S[sc++]); free(idx);
    } *ns=NS; return S;
}
static void FreeSubs(Sub*S,int n){for(int i=0;i<n;++i){ISDestroy(&S[i].is);MatDestroy(&S[i].Ai);
    KSPDestroy(&S[i].ksp);VecDestroy(&S[i].ri);VecDestroy(&S[i].yi);free(S[i].idx);}free(S);}
static void FreeCtx(Ctx*c){VecDestroy(&c->tmp);VecDestroy(&c->dsq);}

static KSP MakeKSP(Mat A, Ctx *c, int wantEig){
    KSP ksp; KSPCreate(PETSC_COMM_SELF,&ksp); KSPSetType(ksp,KSPCG); KSPSetOperators(ksp,A,A);
    PC pc; KSPGetPC(ksp,&pc); PCSetType(pc,PCSHELL); PCShellSetContext(pc,c); PCShellSetApply(pc,ShellApply);
    if(wantEig) KSPSetComputeEigenvalues(ksp,PETSC_TRUE); else KSPSetComputeSingularValues(ksp,PETSC_TRUE);
    KSPSetTolerances(ksp,1e-11,1e-50,PETSC_DEFAULT,1200); KSPSetNormType(ksp,KSP_NORM_UNPRECONDITIONED);
    KSPSetUp(ksp); return ksp;
}

int main(int argc,char**argv){
    PetscInitialize(&argc,&argv,NULL,NULL);

    /* ===== Part A: full spectrum (Ritz values) ===== */
    {
        PetscInt n=64,N=n*n; int P=4,O=2; Mat A=Laplace2D(n);
        Vec b,x; MatCreateVecs(A,&b,&x); VecZeroEntries(b);
        {PetscScalar*ba;VecGetArray(b,&ba);for(PetscInt i=0;i<n;++i)ba[i*n+0]=1.0;VecRestoreArray(b,&ba);}
        const char *tag[4]={"BASIC_exact","sASM_exact","BASIC_icc","sASM_icc"};
        int useS[4]={0,1,0,1}, lev[4]={-1,-1,0,0};
        PetscPrintf(PETSC_COMM_SELF,"=== (3A) full spectrum (Ritz), 64^2 4x4 O=2 ===\n");
        PetscPrintf(PETSC_COMM_SELF,"%-12s %6s %10s %10s\n","combo","#Ritz","lmin","lmax");
        for(int t=0;t<4;++t){
            int nsub; Sub*S=BuildBox(A,n,P,O,lev[t],&nsub); Ctx c=MakeCtx(A,S,nsub,useS[t]);
            KSP ksp=MakeKSP(A,&c,1); KSPSolve(ksp,b,x);
            PetscInt nz; KSPGetIterationNumber(ksp,&nz);
            PetscInt neig; PetscReal *er=(PetscReal*)malloc(sizeof(PetscReal)*(nz+2)),
                                     *ei=(PetscReal*)malloc(sizeof(PetscReal)*(nz+2));
            KSPComputeEigenvalues(ksp,nz+2,er,ei,&neig);
            /* sort ascending (simple) */
            for(int a=0;a<neig;++a)for(int bb=a+1;bb<neig;++bb) if(er[bb]<er[a]){PetscReal tt=er[a];er[a]=er[bb];er[bb]=tt;}
            char fn[80]; sprintf(fn,"phasespec_eigs_%s.txt",tag[t]); FILE*f=fopen(fn,"w");
            for(int a=0;a<neig;++a) fprintf(f,"%g\n",(double)er[a]); fclose(f);
            PetscPrintf(PETSC_COMM_SELF,"%-12s %6d %10.4g %10.4g\n",tag[t],(int)neig,
                (double)er[0],(double)er[neig-1]);
            free(er);free(ei); KSPDestroy(&ksp); FreeCtx(&c); FreeSubs(S,nsub);
        }
        VecDestroy(&b);VecDestroy(&x);MatDestroy(&A);
    }

    /* ===== Part B: phase diagram (overlap x ICC level) ===== */
    {
        PetscInt n=96,N=n*n; int P=6; Mat A=Laplace2D(n);
        Vec b,x; MatCreateVecs(A,&b,&x); VecZeroEntries(b);
        {PetscScalar*ba;VecGetArray(b,&ba);for(PetscInt i=0;i<n;++i)ba[i*n+0]=1.0;VecRestoreArray(b,&ba);}
        int Ov[5]={1,2,4,6,8}, Lv[4]={0,1,2,-1}; /* L=-1 => exact */
        FILE*f=fopen("phasespec_phase.txt","w"); fprintf(f,"# useSASM level O kappa\n");
        PetscPrintf(PETSC_COMM_SELF,"\n=== (3B) phase diagram kappa(O x ICC-level), 96^2 6x6 ===\n");
        for(int useS=0;useS<2;++useS){
          PetscPrintf(PETSC_COMM_SELF,"-- %s --  (rows=ICC level L0/L1/L2/exact, cols O=1/2/4/6/8)\n",useS?"sASM":"BASIC");
          for(int li=0;li<4;++li){
            PetscPrintf(PETSC_COMM_SELF,"  L=%2d: ",Lv[li]);
            for(int oi=0;oi<5;++oi){
                int nsub; Sub*S=BuildBox(A,n,P,Ov[oi],Lv[li],&nsub); Ctx c=MakeCtx(A,S,nsub,useS);
                KSP ksp=MakeKSP(A,&c,0); KSPSolve(ksp,b,x);
                PetscReal smax,smin; KSPComputeExtremeSingularValues(ksp,&smax,&smin);
                PetscReal kap=smax/smin;
                PetscPrintf(PETSC_COMM_SELF,"%9.1f",(double)kap);
                fprintf(f,"%d %d %d %.6g\n",useS,Lv[li],Ov[oi],(double)kap);
                KSPDestroy(&ksp); FreeCtx(&c); FreeSubs(S,nsub);
            }
            PetscPrintf(PETSC_COMM_SELF,"\n");
          }
        }
        fclose(f); VecDestroy(&b);VecDestroy(&x);MatDestroy(&A);
    }
    PetscPrintf(PETSC_COMM_SELF,"\nPHASESPEC_DONE\n");
    PetscFinalize(); return 0;
}
