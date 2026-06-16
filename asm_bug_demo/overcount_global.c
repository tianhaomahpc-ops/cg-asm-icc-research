/* overcount_global.c -- the over-count at EVERY seam, with a domain-wide residual.
 *
 * Fig 1 fed a localized bump, so only the central seam reacted. Here we feed a
 * residual that is nonzero over the WHOLE domain (constant r=1), so z = M^{-1} r is
 * nonzero everywhere and BASIC's over-count shows at all 7 seams. We dump x, m_k,
 * r, z_BASIC, z_sASM and also the difference z_BASIC - z_sASM (the pure over-count,
 * nonzero only in the overlap). 1D Laplace, 256 DOF, 8 subdomains, exact Cholesky.
 */
#include <petscksp.h>
#include <math.h>

typedef struct { PetscInt n; PetscInt *idx; IS is; Mat Ai; KSP ksp; Vec ri, yi; } Sub;

static Mat Laplace1D(PetscInt N){
    Mat A; MatCreateSeqAIJ(PETSC_COMM_SELF,N,N,3,NULL,&A);
    for(PetscInt i=0;i<N;++i){PetscScalar two=2.0,m1=-1.0;MatSetValue(A,i,i,two,INSERT_VALUES);
        if(i>0)MatSetValue(A,i,i-1,m1,INSERT_VALUES); if(i<N-1)MatSetValue(A,i,i+1,m1,INSERT_VALUES);}
    MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
    MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);return A;
}
static void BuildSub(Mat A,PetscInt*idx,PetscInt ni,Sub*s){
    s->n=ni;s->idx=(PetscInt*)malloc(sizeof(PetscInt)*ni);for(PetscInt j=0;j<ni;++j)s->idx[j]=idx[j];
    ISCreateGeneral(PETSC_COMM_SELF,ni,s->idx,PETSC_COPY_VALUES,&s->is);
    MatCreateSubMatrix(A,s->is,s->is,MAT_INITIAL_MATRIX,&s->Ai);
    KSPCreate(PETSC_COMM_SELF,&s->ksp);KSPSetType(s->ksp,KSPPREONLY);KSPSetOperators(s->ksp,s->Ai,s->Ai);
    PC pc;KSPGetPC(s->ksp,&pc);PCSetType(pc,PCCHOLESKY);KSPSetUp(s->ksp);MatCreateVecs(s->Ai,&s->ri,&s->yi);
}
static void ApplyPrec(Sub*subs,PetscInt nsub,int useSASM,Vec dsq,Vec r,Vec z,Vec tmp){
    VecCopy(r,tmp);if(useSASM)VecPointwiseMult(tmp,tmp,dsq);VecZeroEntries(z);
    const PetscScalar*ra;PetscScalar*za;VecGetArrayRead(tmp,&ra);VecGetArray(z,&za);
    for(PetscInt i=0;i<nsub;++i){Sub*s=&subs[i];PetscScalar*rib;VecGetArray(s->ri,&rib);
        for(PetscInt j=0;j<s->n;++j)rib[j]=ra[s->idx[j]];VecRestoreArray(s->ri,&rib);
        KSPSolve(s->ksp,s->ri,s->yi);const PetscScalar*yib;VecGetArrayRead(s->yi,&yib);
        for(PetscInt j=0;j<s->n;++j)za[s->idx[j]]+=yib[j];VecRestoreArrayRead(s->yi,&yib);}
    VecRestoreArrayRead(tmp,&ra);VecRestoreArray(z,&za);if(useSASM)VecPointwiseMult(z,z,dsq);
}

int main(int argc,char**argv){
    PetscInitialize(&argc,&argv,NULL,NULL);
    const PetscInt N=256,NSUB=8;
    Mat A=Laplace1D(N);
    for(int O=2;O<=4;O+=2){
        Sub subs[8];
        for(PetscInt i=0;i<NSUB;++i){
            PetscInt lo=(i*N)/NSUB,hi=((i+1)*N)/NSUB; PetscInt a=lo-O<0?0:lo-O,c=hi+O>N?N:hi+O;
            PetscInt ni=c-a,*idx=(PetscInt*)malloc(sizeof(PetscInt)*ni);
            for(PetscInt j=0;j<ni;++j)idx[j]=a+j; BuildSub(A,idx,ni,&subs[i]);free(idx);
        }
        Vec mult;MatCreateVecs(A,&mult,NULL);VecZeroEntries(mult);PetscScalar*ma;VecGetArray(mult,&ma);
        for(PetscInt i=0;i<NSUB;++i)for(PetscInt j=0;j<subs[i].n;++j)ma[subs[i].idx[j]]+=1.0;VecRestoreArray(mult,&ma);
        Vec dsq;VecDuplicate(mult,&dsq);VecCopy(mult,dsq);VecReciprocal(dsq);VecSqrtAbs(dsq);
        Vec r,zB,zS,tmp;VecDuplicate(mult,&r);VecDuplicate(mult,&zB);VecDuplicate(mult,&zS);VecDuplicate(mult,&tmp);
        VecSet(r,1.0);                                  /* domain-wide residual r = 1 */
        ApplyPrec(subs,NSUB,0,dsq,r,zB,tmp);
        ApplyPrec(subs,NSUB,1,dsq,r,zS,tmp);
        char fn[64];sprintf(fn,"overcount_global_O%d.txt",O);FILE*f=fopen(fn,"w");
        const PetscScalar*mk,*rb,*zb,*zs;VecGetArrayRead(mult,&mk);VecGetArrayRead(r,&rb);
        VecGetArrayRead(zB,&zb);VecGetArrayRead(zS,&zs);
        for(PetscInt k=0;k<N;++k)fprintf(f,"%g %g %g %g %g %g\n",(double)(k+1)/(N+1),PetscRealPart(mk[k]),
            PetscRealPart(rb[k]),PetscRealPart(zb[k]),PetscRealPart(zs[k]),
            PetscRealPart(zb[k])-PetscRealPart(zs[k]));
        VecRestoreArrayRead(mult,&mk);VecRestoreArrayRead(r,&rb);VecRestoreArrayRead(zB,&zb);VecRestoreArrayRead(zS,&zs);
        fclose(f);
        PetscPrintf(PETSC_COMM_SELF,"[O=%d] wrote overcount_global_O%d.txt (r=const over whole domain)\n",O,O);
        VecDestroy(&mult);VecDestroy(&dsq);VecDestroy(&r);VecDestroy(&zB);VecDestroy(&zS);VecDestroy(&tmp);
        for(PetscInt i=0;i<NSUB;++i){ISDestroy(&subs[i].is);MatDestroy(&subs[i].Ai);KSPDestroy(&subs[i].ksp);
            VecDestroy(&subs[i].ri);VecDestroy(&subs[i].yi);free(subs[i].idx);}
    }
    MatDestroy(&A);PetscPrintf(PETSC_COMM_SELF,"OVERCOUNT_GLOBAL_DONE\n");PetscFinalize();return 0;
}
