/* zfield_2d.c -- how does BASIC's correction z = M_BASIC^{-1} r change when the
 * subdomain solver goes from EXACT (Cholesky) to INEXACT (ICC(0))?  In 1D ICC=exact
 * so z is identical; here in 2D (5-point Laplace, ICC drops fill) they differ.
 *
 * One BASIC apply, z = sum_i R_i^T (solve_i) R_i r, with a domain-wide residual r=1:
 *   z_exact : exact Cholesky subdomain solves
 *   z_icc   : ICC(0) subdomain solves
 *   z_icc - z_exact : the inexactness error in the correction (the dropped fill E_i,
 *                     DOUBLE-counted by BASIC in the overlap seams).
 * 2D Laplace 64^2, 4x4 subdomains, O=2. Dumps zfield_{exact,icc,diff,mult}.txt (64x64).
 */
#include <petscksp.h>
#include <math.h>

typedef struct { PetscInt n; PetscInt *idx; IS is; Mat Ai; KSP ksp; Vec ri, yi; } Sub;

static Mat Laplace2D(PetscInt n){
    Mat A; PetscInt N=n*n; MatCreateSeqAIJ(PETSC_COMM_SELF,N,N,5,NULL,&A);
    for(PetscInt b=0;b<n;++b)for(PetscInt a=0;a<n;++a){PetscInt k=b*n+a;PetscScalar four=4.0,m1=-1.0;
        MatSetValue(A,k,k,four,INSERT_VALUES);
        if(a>0)MatSetValue(A,k,k-1,m1,INSERT_VALUES);if(a<n-1)MatSetValue(A,k,k+1,m1,INSERT_VALUES);
        if(b>0)MatSetValue(A,k,k-n,m1,INSERT_VALUES);if(b<n-1)MatSetValue(A,k,k+n,m1,INSERT_VALUES);}
    MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
    MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);return A;
}
static void BuildSub(Mat A,PetscInt*idx,PetscInt ni,int icc,Sub*s){
    s->n=ni;s->idx=(PetscInt*)malloc(sizeof(PetscInt)*ni);for(PetscInt j=0;j<ni;++j)s->idx[j]=idx[j];
    ISCreateGeneral(PETSC_COMM_SELF,ni,s->idx,PETSC_COPY_VALUES,&s->is);
    MatCreateSubMatrix(A,s->is,s->is,MAT_INITIAL_MATRIX,&s->Ai);
    KSPCreate(PETSC_COMM_SELF,&s->ksp);KSPSetType(s->ksp,KSPPREONLY);KSPSetOperators(s->ksp,s->Ai,s->Ai);
    PC pc;KSPGetPC(s->ksp,&pc);if(icc){PCSetType(pc,PCICC);PCFactorSetLevels(pc,0);}else PCSetType(pc,PCCHOLESKY);
    KSPSetUp(s->ksp);MatCreateVecs(s->Ai,&s->ri,&s->yi);
}
static void ApplyBASIC(Sub*subs,PetscInt nsub,Vec r,Vec z){
    VecZeroEntries(z);const PetscScalar*ra;PetscScalar*za;VecGetArrayRead(r,&ra);VecGetArray(z,&za);
    for(PetscInt i=0;i<nsub;++i){Sub*s=&subs[i];PetscScalar*rib;VecGetArray(s->ri,&rib);
        for(PetscInt j=0;j<s->n;++j)rib[j]=ra[s->idx[j]];VecRestoreArray(s->ri,&rib);
        KSPSolve(s->ksp,s->ri,s->yi);const PetscScalar*yib;VecGetArrayRead(s->yi,&yib);
        for(PetscInt j=0;j<s->n;++j)za[s->idx[j]]+=yib[j];VecRestoreArrayRead(s->yi,&yib);}
    VecRestoreArrayRead(r,&ra);VecRestoreArray(z,&za);
}
static Sub *Build(Mat A,PetscInt n,int P,int O,int icc,int*ns){
    int NS=P*P;Sub*S=(Sub*)malloc(sizeof(Sub)*NS);int sc=0;
    for(int q=0;q<P;++q)for(int p=0;p<P;++p){
        PetscInt axlo=(p*n)/P,axhi=((p+1)*n)/P,bylo=(q*n)/P,byhi=((q+1)*n)/P;
        PetscInt al=axlo-O<0?0:axlo-O,ar=axhi+O>n?n:axhi+O,bl=bylo-O<0?0:bylo-O,br=byhi+O>n?n:byhi+O;
        PetscInt ni=(ar-al)*(br-bl),*idx=(PetscInt*)malloc(sizeof(PetscInt)*ni),c=0;
        for(PetscInt bb=bl;bb<br;++bb)for(PetscInt a=al;a<ar;++a)idx[c++]=bb*n+a;
        BuildSub(A,idx,ni,icc,&S[sc++]);free(idx);} *ns=NS;return S;
}
static void Dump(Vec v,PetscInt n,const char*fn){FILE*f=fopen(fn,"w");const PetscScalar*a;VecGetArrayRead(v,&a);
    for(PetscInt b=0;b<n;++b){for(PetscInt c=0;c<n;++c)fprintf(f,"%g ",PetscRealPart(a[b*n+c]));fprintf(f,"\n");}
    VecRestoreArrayRead(v,&a);fclose(f);}

int main(int argc,char**argv){
    PetscInitialize(&argc,&argv,NULL,NULL);
    const PetscInt n=64,N=n*n;int P=4,O=2;
    Mat A=Laplace2D(n);
    Vec r,ze,zi,diff;MatCreateVecs(A,&r,NULL);VecDuplicate(r,&ze);VecDuplicate(r,&zi);VecDuplicate(r,&diff);
    VecSet(r,1.0);                                   /* domain-wide residual */
    int ns; Sub*Se=Build(A,n,P,O,0,&ns);  ApplyBASIC(Se,ns,r,ze);   /* exact */
    Sub*Si=Build(A,n,P,O,1,&ns);          ApplyBASIC(Si,ns,r,zi);   /* ICC   */
    VecWAXPY(diff,-1.0,ze,zi);                       /* diff = z_icc - z_exact */
    Vec mult;VecDuplicate(r,&mult);VecZeroEntries(mult);PetscScalar*ma;VecGetArray(mult,&ma);
    for(int i=0;i<ns;++i)for(PetscInt j=0;j<Se[i].n;++j)ma[Se[i].idx[j]]+=1.0;VecRestoreArray(mult,&ma);
    Dump(ze,n,"zfield_exact.txt");Dump(zi,n,"zfield_icc.txt");Dump(diff,n,"zfield_diff.txt");Dump(mult,n,"zfield_mult.txt");
    PetscReal ne,ni_,nd;VecNorm(ze,NORM_2,&ne);VecNorm(zi,NORM_2,&ni_);VecNorm(diff,NORM_2,&nd);
    /* seam vs interior share of the inexactness error */
    const PetscScalar*dd,*mm;VecGetArrayRead(diff,&dd);VecGetArrayRead(mult,&mm);
    PetscReal seam=0,inte=0;for(PetscInt k=0;k<N;++k){PetscReal v=PetscRealPart(dd[k]);v*=v;
        if(PetscRealPart(mm[k])>1.5)seam+=v;else inte+=v;}
    VecRestoreArrayRead(diff,&dd);VecRestoreArrayRead(mult,&mm);
    PetscPrintf(PETSC_COMM_SELF,
        "2D 64^2, 4x4, O=2, BASIC:\n"
        "[smooth r=1]  ||z_exact||=%.4g  ||z_icc||=%.4g  ||z_icc-z_exact||=%.1f%% of ||z_exact||\n"
        "              inexactness error: seam=%.1f%% interior=%.1f%%\n",
        (double)ne,(double)ni_,100.0*nd/ne,100.0*seam/(seam+inte),100.0*inte/(seam+inte));

    /* contrast: a ROUGH (random) input -- ICC is a much better approximation there */
    { Vec rr,zer,zir,dr; VecDuplicate(r,&rr);VecDuplicate(r,&zer);VecDuplicate(r,&zir);VecDuplicate(r,&dr);
      PetscRandom rng;PetscRandomCreate(PETSC_COMM_SELF,&rng);PetscRandomSetSeed(rng,7);PetscRandomSeed(rng);
      VecSetRandom(rr,rng);PetscRandomDestroy(&rng);
      ApplyBASIC(Se,ns,rr,zer);ApplyBASIC(Si,ns,rr,zir);VecWAXPY(dr,-1.0,zer,zir);
      PetscReal a,d;VecNorm(zer,NORM_2,&a);VecNorm(dr,NORM_2,&d);
      PetscPrintf(PETSC_COMM_SELF,"[rough random] ||z_exact||=%.4g  ||z_icc-z_exact||=%.1f%% of ||z_exact||\n",
                  (double)a,100.0*d/a);
      VecDestroy(&rr);VecDestroy(&zer);VecDestroy(&zir);VecDestroy(&dr); }
    PetscPrintf(PETSC_COMM_SELF,"ZFIELD_DONE\n");
    PetscFinalize();return 0;
}
