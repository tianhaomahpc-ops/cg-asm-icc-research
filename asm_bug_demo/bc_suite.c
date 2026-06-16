/* bc_suite.c -- interpretability across BOUNDARY CONDITIONS, on the unit geometry.
 *
 * PROBLEM (stated first, per request):
 *   Solve a Poisson/Laplace problem  -Lap u = f  on the unit hypercube [0,1]^d
 *   (d = 1,2,3), discretized by the standard (2d+1)-point finite-difference
 *   Laplacian with n nodes per axis, with one of three boundary-condition sets:
 *     (D) all-Dirichlet : u = 0 on every face   (SPD, the existing 1D/2D minimal case)
 *     (N) all-Neumann   : du/dn = 0 on every face (SINGULAR, null space = constants)
 *     (M) mixed         : 1 face Dirichlet + (2d-1) faces Neumann (the Sys3-type case)
 *   Preconditioner: P x P (x P) box overlapping additive Schwarz, BASIC vs sASM,
 *   with exact (Cholesky) or inexact (ICC(0)) subdomain solves; optional Nicolaides
 *   two-level coarse space. We measure the same interpretability quantities as the
 *   Dirichlet study (N_hat, kappa, lambda_max/min of M^-1 A, iterations, two-level
 *   lambda_min) so the three BC sets can be compared ITEM BY ITEM.
 *
 * Key facts the comparison should expose:
 *   - N_hat (over-count) is GEOMETRIC -> identical across BC.
 *   - The BC moves lambda_min: Dirichlet lambda_min>0; Neumann has a ZERO mode
 *     (the constant) = the global/propagation mode; mixed is in between.
 *   - The Nicolaides PoU coarse space spans the constant -> it is the *exact* fix
 *     for the Neumann zero mode (two-level especially decisive there).
 *
 * Singular (all-Neumann) handled with a constant MatNullSpace (consistent mean-zero
 * RHS, nullspace removed in CG and in the coarse solve). Sequential. Dumps bc_suite.txt.
 */
#include <petscksp.h>
#include <math.h>

typedef struct { PetscInt n; PetscInt *idx; IS is; Mat Ai; KSP ksp; Vec ri, yi; } Sub;

/* ---- parametrized Laplacian on [0,1]^d, n nodes/axis, per-face BC (0=Dir,1=Neu) ---- */
static Mat make_laplace(int dim, PetscInt n, const int *bc, int *singular) {
    PetscInt N = 1; for (int a=0;a<dim;++a) N*=n;
    Mat A; MatCreateSeqAIJ(PETSC_COMM_SELF, N, N, 2*dim+1, NULL, &A);
    PetscInt stride[3]={1,1,1}; for(int a=1;a<dim;++a) stride[a]=stride[a-1]*n;
    int allNeu=1; for(int f=0;f<2*dim;++f) if(bc[f]==0) allNeu=0;
    for (PetscInt k=0;k<N;++k) {
        PetscInt c[3]={0,0,0}, t=k; for(int a=0;a<dim;++a){c[a]=t%n; t/=n;}
        PetscScalar diag=0.0;
        for (int a=0;a<dim;++a) {
            /* lower neighbor (-a) */
            if (c[a]>0){ MatSetValue(A,k,k-stride[a],-1.0,INSERT_VALUES); diag+=1.0; }
            else if (bc[2*a]==0) diag+=1.0;                 /* Dirichlet ghost adds to diag */
            /* upper neighbor (+a) */
            if (c[a]<n-1){ MatSetValue(A,k,k+stride[a],-1.0,INSERT_VALUES); diag+=1.0; }
            else if (bc[2*a+1]==0) diag+=1.0;
        }
        MatSetValue(A,k,k,diag,INSERT_VALUES);
    }
    MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
    MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);
    *singular=allNeu;
    if (allNeu){ MatNullSpace ns; MatNullSpaceCreate(PETSC_COMM_SELF,PETSC_TRUE,0,NULL,&ns);
        MatSetNullSpace(A,ns); MatNullSpaceDestroy(&ns); }
    return A;
}

static void BuildSub(Mat A,PetscInt*idx,PetscInt ni,int icc,Sub*s){
    s->n=ni;s->idx=(PetscInt*)malloc(sizeof(PetscInt)*ni);for(PetscInt j=0;j<ni;++j)s->idx[j]=idx[j];
    ISCreateGeneral(PETSC_COMM_SELF,ni,s->idx,PETSC_COPY_VALUES,&s->is);
    MatCreateSubMatrix(A,s->is,s->is,MAT_INITIAL_MATRIX,&s->Ai);
    /* subdomain blocks are SPD even for global-Neumann (they keep a Dirichlet cut) */
    KSPCreate(PETSC_COMM_SELF,&s->ksp);KSPSetType(s->ksp,KSPPREONLY);KSPSetOperators(s->ksp,s->Ai,s->Ai);
    PC pc;KSPGetPC(s->ksp,&pc);if(icc){PCSetType(pc,PCICC);PCFactorSetLevels(pc,0);}else PCSetType(pc,PCCHOLESKY);
    KSPSetUp(s->ksp);MatCreateVecs(s->Ai,&s->ri,&s->yi);
}
static PetscInt *MultArr(Sub*S,int ns,PetscInt N,PetscInt*nhat){
    PetscInt*m=(PetscInt*)calloc(N,sizeof(PetscInt));
    for(int i=0;i<ns;++i)for(PetscInt j=0;j<S[i].n;++j)m[S[i].idx[j]]++;
    PetscInt mm=0;for(PetscInt k=0;k<N;++k)if(m[k]>mm)mm=m[k];*nhat=mm;return m;
}
typedef struct { Sub*subs; PetscInt nsub; int useSASM; Vec dsq,tmp;
                 int twolevel; Mat R0t; KSP kspc; Vec cvec,ycvec; } Ctx;
static void Fine(Ctx*c,Vec r,Vec z){
    VecCopy(r,c->tmp); if(c->useSASM)VecPointwiseMult(c->tmp,c->tmp,c->dsq); VecZeroEntries(z);
    const PetscScalar*ra;PetscScalar*za;VecGetArrayRead(c->tmp,&ra);VecGetArray(z,&za);
    for(PetscInt i=0;i<c->nsub;++i){Sub*s=&c->subs[i];PetscScalar*rib;VecGetArray(s->ri,&rib);
        for(PetscInt j=0;j<s->n;++j)rib[j]=ra[s->idx[j]];VecRestoreArray(s->ri,&rib);
        KSPSolve(s->ksp,s->ri,s->yi);const PetscScalar*yib;VecGetArrayRead(s->yi,&yib);
        for(PetscInt j=0;j<s->n;++j)za[s->idx[j]]+=yib[j];VecRestoreArrayRead(s->yi,&yib);}
    VecRestoreArrayRead(c->tmp,&ra);VecRestoreArray(z,&za); if(c->useSASM)VecPointwiseMult(z,z,c->dsq);
}
static void Apply(Ctx*c,Vec r,Vec z){ Fine(c,r,z);
    if(c->twolevel){MatMultTranspose(c->R0t,r,c->cvec);KSPSolve(c->kspc,c->cvec,c->ycvec);MatMultAdd(c->R0t,c->ycvec,z,z);} }
static PetscErrorCode ShellApply(PC pc,Vec r,Vec z){Ctx*c;PCShellGetContext(pc,&c);Apply(c,r,z);return 0;}

static void BuildCoarse(Mat A,Sub*S,int ns,PetscInt N,PetscInt*mult,int singular,Ctx*c){
    Mat R0t;MatCreateSeqAIJ(PETSC_COMM_SELF,N,ns,64,NULL,&R0t);
    for(int i=0;i<ns;++i)for(PetscInt j=0;j<S[i].n;++j)
        MatSetValue(R0t,S[i].idx[j],i,1.0/(PetscReal)mult[S[i].idx[j]],INSERT_VALUES);
    MatAssemblyBegin(R0t,MAT_FINAL_ASSEMBLY);MatAssemblyEnd(R0t,MAT_FINAL_ASSEMBLY);
    c->R0t=R0t; Mat A0;MatPtAP(A,R0t,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&A0);
    KSPCreate(PETSC_COMM_SELF,&c->kspc);
    if(singular){ /* A0 inherits the constant null space (sum of PoU = 1) */
        MatNullSpace ns0;MatNullSpaceCreate(PETSC_COMM_SELF,PETSC_TRUE,0,NULL,&ns0);MatSetNullSpace(A0,ns0);MatNullSpaceDestroy(&ns0);
        KSPSetType(c->kspc,KSPCG); PC pc;KSPGetPC(c->kspc,&pc);PCSetType(pc,PCJACOBI);
        KSPSetTolerances(c->kspc,1e-10,1e-50,PETSC_DEFAULT,200);
    } else { KSPSetType(c->kspc,KSPPREONLY); PC pc;KSPGetPC(c->kspc,&pc);PCSetType(pc,PCCHOLESKY); }
    KSPSetOperators(c->kspc,A0,A0);KSPSetUp(c->kspc);
    MatCreateVecs(A0,&c->ycvec,&c->cvec);MatDestroy(&A0);
}
static Ctx MakeCtx(Mat A,Sub*S,int ns,PetscInt N,int useSASM,int twolevel,int singular){
    Ctx c;c.subs=S;c.nsub=ns;c.useSASM=useSASM;c.twolevel=twolevel;MatCreateVecs(A,&c.tmp,NULL);
    PetscInt nhat;PetscInt*m=MultArr(S,ns,N,&nhat);
    Vec mult;MatCreateVecs(A,&mult,NULL);PetscScalar*ma;VecGetArray(mult,&ma);
    for(PetscInt k=0;k<N;++k)ma[k]=(PetscScalar)m[k];VecRestoreArray(mult,&ma);
    VecDuplicate(mult,&c.dsq);VecCopy(mult,c.dsq);VecReciprocal(c.dsq);VecSqrtAbs(c.dsq);VecDestroy(&mult);
    if(twolevel)BuildCoarse(A,S,ns,N,m,singular,&c); else {c.R0t=NULL;c.kspc=NULL;c.cvec=NULL;c.ycvec=NULL;}
    free(m);return c;
}
static void FreeCtx(Ctx*c){VecDestroy(&c->tmp);VecDestroy(&c->dsq);
    if(c->twolevel){MatDestroy(&c->R0t);KSPDestroy(&c->kspc);VecDestroy(&c->cvec);VecDestroy(&c->ycvec);}}

static void Spectrum(Mat A,Ctx*c,Vec b,int singular,PetscReal*lmax,PetscReal*lmin){
    Vec x;VecDuplicate(b,&x);
    KSP ksp;KSPCreate(PETSC_COMM_SELF,&ksp);KSPSetType(ksp,KSPCG);KSPSetOperators(ksp,A,A);
    PC pc;KSPGetPC(ksp,&pc);PCSetType(pc,PCSHELL);PCShellSetContext(pc,c);PCShellSetApply(pc,ShellApply);
    KSPSetComputeSingularValues(ksp,PETSC_TRUE);KSPSetTolerances(ksp,1e-10,1e-50,PETSC_DEFAULT,800);
    KSPSetNormType(ksp,KSP_NORM_UNPRECONDITIONED);KSPSetUp(ksp);KSPSolve(ksp,b,x);
    KSPComputeExtremeSingularValues(ksp,lmax,lmin);VecDestroy(&x);KSPDestroy(&ksp);
}
static PetscInt PCGit(Mat A,Ctx*c,Vec b,Vec x,PetscReal rtol,PetscInt maxit){
    Vec r,z,p,Ap;VecDuplicate(b,&r);VecDuplicate(b,&z);VecDuplicate(b,&p);VecDuplicate(b,&Ap);
    VecZeroEntries(x);VecCopy(b,r);PetscReal bnorm;VecNorm(b,NORM_2,&bnorm);
    Apply(c,r,z);VecCopy(z,p);PetscScalar rz;VecDot(r,z,&rz);PetscInt it;PetscReal rn;
    for(it=0;it<maxit;++it){MatMult(A,p,Ap);PetscScalar pAp;VecDot(p,Ap,&pAp);PetscScalar al=rz/pAp;
        VecAXPY(x,al,p);VecAXPY(r,-al,Ap);VecNorm(r,NORM_2,&rn);if(rn/bnorm<rtol){++it;break;}
        Apply(c,r,z);PetscScalar rz2;VecDot(r,z,&rz2);PetscScalar be=rz2/rz;rz=rz2;VecAYPX(p,be,z);}
    VecDestroy(&r);VecDestroy(&z);VecDestroy(&p);VecDestroy(&Ap);return it;
}
static Sub*BuildBoxes(Mat A,int dim,PetscInt n,int P,int O,int icc,int*nsOut){
    PetscInt stride[3]={1,1,1};for(int a=1;a<dim;++a)stride[a]=stride[a-1]*n;
    int NS=1;for(int a=0;a<dim;++a)NS*=P;
    Sub*S=(Sub*)malloc(sizeof(Sub)*NS);int sc=0;
    PetscInt pc[3];
    for(int s=0;s<NS;++s){int t=s;for(int a=0;a<dim;++a){pc[a]=t%P;t/=P;}
        PetscInt lo[3],hi[3];for(int a=0;a<dim;++a){PetscInt l=(pc[a]*n)/P-O,h=((pc[a]+1)*n)/P+O;
            lo[a]=l<0?0:l;hi[a]=h>n?n:h;}
        PetscInt ni=1;for(int a=0;a<dim;++a)ni*=(hi[a]-lo[a]);
        PetscInt*idx=(PetscInt*)malloc(sizeof(PetscInt)*ni),cc=0;
        PetscInt cc3[3];
        if(dim==1)for(PetscInt i=lo[0];i<hi[0];++i)idx[cc++]=i;
        if(dim==2)for(PetscInt j=lo[1];j<hi[1];++j)for(PetscInt i=lo[0];i<hi[0];++i)idx[cc++]=j*stride[1]+i;
        if(dim==3)for(PetscInt kk=lo[2];kk<hi[2];++kk)for(PetscInt j=lo[1];j<hi[1];++j)for(PetscInt i=lo[0];i<hi[0];++i)
            idx[cc++]=kk*stride[2]+j*stride[1]+i;
        (void)cc3;
        BuildSub(A,idx,ni,icc,&S[sc++]);free(idx);
    }
    *nsOut=NS;return S;
}
static void FreeSubs(Sub*S,int n){for(int i=0;i<n;++i){ISDestroy(&S[i].is);MatDestroy(&S[i].Ai);KSPDestroy(&S[i].ksp);
    VecDestroy(&S[i].ri);VecDestroy(&S[i].yi);free(S[i].idx);}free(S);}

/* mean-zero left-ish source so the RHS is consistent for the singular case */
static void RHS(Vec b,int dim,PetscInt n,int singular){
    VecZeroEntries(b);PetscScalar*ba;VecGetArray(b,&ba);PetscInt N;VecGetSize(b,&N);
    PetscInt stride[3]={1,1,1};for(int a=1;a<dim;++a)stride[a]=stride[a-1]*n;
    /* source on the x=0 slab */
    for(PetscInt k=0;k<N;++k){PetscInt t=k,c0=t%n; ba[k]=(c0==0)?1.0:0.0;}
    VecRestoreArray(b,&ba);
    if(singular){PetscScalar mean;VecSum(b,&mean);mean/=N;VecShift(b,-mean);}  /* project out constant */
}

int main(int argc,char**argv){
    PetscInitialize(&argc,&argv,NULL,NULL);
    int dims[3]={1,2,3}; PetscInt ns_[3]={256,64,24}; int Ps[3]={8,4,4}; int O=2;
    const char*bcname[3]={"Dirichlet","Neumann","Mixed1D5N"};
    FILE*f=fopen("bc_suite.txt","w");
    fprintf(f,"# dim bc Nhat lmaxBE lminBE lmaxSE lminSE lmaxBI lminBI lmaxSI lminSI itBI itSI it2lvlSI\n");
    PetscPrintf(PETSC_COMM_SELF,"%-3s %-10s %4s | %9s %9s %9s %9s | %6s %6s %6s\n",
        "dim","BC","Nhat","lmax(B,ex)","kap(B,ex)","lmax(S,ic)","kap(S,ic)","it(B,ic)","it(S,ic)","it2lvl");
    for(int di=0;di<3;++di){ int dim=dims[di]; PetscInt n=ns_[di]; int P=Ps[di];
      for(int bci=0;bci<3;++bci){
        int bc[6]; for(int ff=0;ff<2*dim;++ff)bc[ff]=(bci==0)?0:1;
        if(bci==2) bc[0]=0;                    /* mixed: x=0 face Dirichlet, rest Neumann */
        int sing; Mat A=make_laplace(dim,n,bc,&sing);
        PetscInt N;MatGetSize(A,&N,NULL);
        Vec b,x;MatCreateVecs(A,&b,&x);RHS(b,dim,n,sing);
        int NS; Sub*Sx=BuildBoxes(A,dim,n,P,O,0,&NS); Sub*Si=BuildBoxes(A,dim,n,P,O,1,&NS);
        PetscInt nhat;{PetscInt*m=MultArr(Sx,NS,N,&nhat);free(m);}
        PetscReal lBE,mBE,lSE,mSE,lBI,mBI,lSI,mSI;
        Ctx cBE=MakeCtx(A,Sx,NS,N,0,0,sing);Spectrum(A,&cBE,b,sing,&lBE,&mBE);FreeCtx(&cBE);
        Ctx cSE=MakeCtx(A,Sx,NS,N,1,0,sing);Spectrum(A,&cSE,b,sing,&lSE,&mSE);FreeCtx(&cSE);
        Ctx cBI=MakeCtx(A,Si,NS,N,0,0,sing);Spectrum(A,&cBI,b,sing,&lBI,&mBI);
        PetscInt itBI=PCGit(A,&cBI,b,x,1e-8,4000);FreeCtx(&cBI);
        Ctx cSI=MakeCtx(A,Si,NS,N,1,0,sing);Spectrum(A,&cSI,b,sing,&lSI,&mSI);
        PetscInt itSI=PCGit(A,&cSI,b,x,1e-8,4000);FreeCtx(&cSI);
        Ctx c2=MakeCtx(A,Si,NS,N,1,1,sing); PetscReal l2,m2; Spectrum(A,&c2,b,sing,&l2,&m2);
        PetscInt it2=PCGit(A,&c2,b,x,1e-8,4000);FreeCtx(&c2);
        fprintf(f,"%d %s %d  %.5g %.5g %.5g %.5g  %.5g %.5g %.5g %.5g  %d %d %d  %.5g %.5g\n",
            dim,bcname[bci],(int)nhat,lBE,mBE,lSE,mSE,lBI,mBI,lSI,mSI,(int)itBI,(int)itSI,(int)it2,l2,m2);
        PetscPrintf(PETSC_COMM_SELF,"%-3d %-10s %4d | %9.2f %9.1f %9.3f %9.1f | %6d %6d %6d\n",
            dim,bcname[bci],(int)nhat,lBE,(mBE>0?lBE/mBE:-1),lSI,(mSI>0?lSI/mSI:-1),(int)itBI,(int)itSI,(int)it2);
        FreeSubs(Sx,NS);FreeSubs(Si,NS);VecDestroy(&b);VecDestroy(&x);MatDestroy(&A);
      }
    }
    fclose(f);PetscPrintf(PETSC_COMM_SELF,"BC_SUITE_DONE\n");PetscFinalize();return 0;
}
