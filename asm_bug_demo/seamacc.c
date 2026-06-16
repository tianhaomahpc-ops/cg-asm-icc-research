/* seamacc.c -- exact vs inexact BOUNDARY ERROR ACCUMULATION at the seams.
 *
 * The question: with an EXACT subdomain solve the over-count is harmless, but with
 * an INEXACT (ICC) solve the over-counted ERROR piles up at the subdomain seams.
 * Fig 3 only shows the global residual-norm gap; here we localize it: at every CG
 * iteration we measure the SEAM-RESIDUAL FRACTION
 *      f_seam(k) = || r_k restricted to overlap (m>1) ||^2 / || r_k ||^2,
 * i.e. how much of the remaining error lives in the over-counted seam region, for
 *   BASIC-exact   (over-count present, but solve exact -> no accumulation)
 *   BASIC-ICC     (over-count x inexact -> accumulates at seams)
 *   sASM-ICC      (over-count removed -> no accumulation)
 * Also dumps the 2D residual fields at matched iterations for the heatmaps.
 *
 * 2D Laplace 64^2, 4x4, O=2. Sequential. Dumps seamacc_<tag>.txt + seamacc_field_*.txt.
 */
#include <petscksp.h>
#include <math.h>

typedef struct { PetscInt n; PetscInt *idx; IS is; Mat Ai; KSP ksp; Vec ri, yi; } Sub;
static Vec g_seammask;   /* 1 on overlap seam (m>1), else 0 */

static Mat Laplace2D(PetscInt n) {
    Mat A; PetscInt N=n*n; MatCreateSeqAIJ(PETSC_COMM_SELF,N,N,5,NULL,&A);
    for(PetscInt b=0;b<n;++b)for(PetscInt a=0;a<n;++a){
        PetscInt k=b*n+a; PetscScalar four=4.0,m1=-1.0; MatSetValue(A,k,k,four,INSERT_VALUES);
        if(a>0)MatSetValue(A,k,k-1,m1,INSERT_VALUES); if(a<n-1)MatSetValue(A,k,k+1,m1,INSERT_VALUES);
        if(b>0)MatSetValue(A,k,k-n,m1,INSERT_VALUES); if(b<n-1)MatSetValue(A,k,k+n,m1,INSERT_VALUES);
    }
    MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
    MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE); return A;
}
static void BuildSub(Mat A,PetscInt*idx,PetscInt ni,int icc,Sub*s){
    s->n=ni; s->idx=(PetscInt*)malloc(sizeof(PetscInt)*ni);
    for(PetscInt j=0;j<ni;++j)s->idx[j]=idx[j];
    ISCreateGeneral(PETSC_COMM_SELF,ni,s->idx,PETSC_COPY_VALUES,&s->is);
    MatCreateSubMatrix(A,s->is,s->is,MAT_INITIAL_MATRIX,&s->Ai);
    KSPCreate(PETSC_COMM_SELF,&s->ksp);KSPSetType(s->ksp,KSPPREONLY);KSPSetOperators(s->ksp,s->Ai,s->Ai);
    PC pc;KSPGetPC(s->ksp,&pc); if(icc){PCSetType(pc,PCICC);PCFactorSetLevels(pc,0);}else PCSetType(pc,PCCHOLESKY);
    KSPSetUp(s->ksp);MatCreateVecs(s->Ai,&s->ri,&s->yi);
}
static Vec Multiplicity(Mat A,Sub*subs,PetscInt nsub){
    Vec m;MatCreateVecs(A,&m,NULL);VecZeroEntries(m);PetscScalar*ma;VecGetArray(m,&ma);
    for(PetscInt i=0;i<nsub;++i)for(PetscInt j=0;j<subs[i].n;++j)ma[subs[i].idx[j]]+=1.0;
    VecRestoreArray(m,&ma);return m;
}
static void ApplyPrec(Sub*subs,PetscInt nsub,int useSASM,Vec dsq,Vec r,Vec z,Vec tmp){
    VecCopy(r,tmp); if(useSASM)VecPointwiseMult(tmp,tmp,dsq); VecZeroEntries(z);
    const PetscScalar*ra;PetscScalar*za;VecGetArrayRead(tmp,&ra);VecGetArray(z,&za);
    for(PetscInt i=0;i<nsub;++i){Sub*s=&subs[i];PetscScalar*rib;VecGetArray(s->ri,&rib);
        for(PetscInt j=0;j<s->n;++j)rib[j]=ra[s->idx[j]];VecRestoreArray(s->ri,&rib);
        KSPSolve(s->ksp,s->ri,s->yi);const PetscScalar*yib;VecGetArrayRead(s->yi,&yib);
        for(PetscInt j=0;j<s->n;++j)za[s->idx[j]]+=yib[j];VecRestoreArrayRead(s->yi,&yib);}
    VecRestoreArrayRead(tmp,&ra);VecRestoreArray(z,&za); if(useSASM)VecPointwiseMult(z,z,dsq);
}
static void DumpField(Vec r,PetscInt n,const char*fn){
    FILE*f=fopen(fn,"w");const PetscScalar*ra;VecGetArrayRead(r,&ra);
    for(PetscInt b=0;b<n;++b){for(PetscInt a=0;a<n;++a)fprintf(f,"%g ",PetscRealPart(ra[b*n+a]));fprintf(f,"\n");}
    VecRestoreArrayRead(r,&ra);fclose(f);
}
/* PCG logging relres + seam fraction each iter; dumps fields at dit[] */
static PetscInt PCG(Mat A,Sub*subs,PetscInt nsub,int useSASM,Vec dsq,
                    Vec b,Vec x,PetscReal rtol,PetscInt maxit,const char*tag,PetscInt n,
                    PetscInt*dit,int nd){
    Vec r,z,p,Ap,tmp;VecDuplicate(b,&r);VecDuplicate(b,&z);VecDuplicate(b,&p);VecDuplicate(b,&Ap);VecDuplicate(b,&tmp);
    VecZeroEntries(x);VecCopy(b,r);PetscReal bnorm;VecNorm(b,NORM_2,&bnorm);
    ApplyPrec(subs,nsub,useSASM,dsq,r,z,tmp);VecCopy(z,p);PetscScalar rz;VecDot(r,z,&rz);
    char fn[96];sprintf(fn,"seamacc_%s.txt",tag);FILE*lg=fopen(fn,"w");fprintf(lg,"# iter relres seamfrac\n");
    PetscInt it;PetscReal rn;
    const PetscScalar *mk; VecGetArrayRead(g_seammask,&mk);
    for(it=0;it<=maxit;++it){
        VecNorm(r,NORM_2,&rn);
        const PetscScalar*ra;VecGetArrayRead(r,&ra);
        PetscReal s=0,t=0; for(PetscInt k=0;k<n*n;++k){PetscReal v=PetscRealPart(ra[k]);t+=v*v; if(PetscRealPart(mk[k])>0.5)s+=v*v;}
        VecRestoreArrayRead(r,&ra);
        fprintf(lg,"%d %.6e %.6f\n",(int)it,(double)(rn/bnorm),(double)(t>0?s/t:0));
        for(int q=0;q<nd;++q) if(it==dit[q]){char ff[96];sprintf(ff,"seamacc_field_%s_k%d.txt",tag,(int)dit[q]);DumpField(r,n,ff);}
        if(rn/bnorm<rtol){++it;break;}
        MatMult(A,p,Ap);PetscScalar pAp;VecDot(p,Ap,&pAp);PetscScalar al=rz/pAp;
        VecAXPY(x,al,p);VecAXPY(r,-al,Ap);
        ApplyPrec(subs,nsub,useSASM,dsq,r,z,tmp);PetscScalar rz2;VecDot(r,z,&rz2);PetscScalar be=rz2/rz;rz=rz2;VecAYPX(p,be,z);
    }
    VecRestoreArrayRead(g_seammask,&mk); fclose(lg);
    VecDestroy(&r);VecDestroy(&z);VecDestroy(&p);VecDestroy(&Ap);VecDestroy(&tmp); return it;
}

int main(int argc,char**argv){
    PetscInitialize(&argc,&argv,NULL,NULL);
    const PetscInt n=64,P=4,O=2,N=n*n;
    Mat A=Laplace2D(n);
    Vec b;MatCreateVecs(A,&b,NULL);VecZeroEntries(b);
    {PetscScalar*ba;VecGetArray(b,&ba);for(PetscInt bb=0;bb<n;++bb)ba[bb*n+0]=1.0;VecRestoreArray(b,&ba);}

    /* three solver setups */
    int combos=3; int useS[3]={0,0,1}, icc[3]={0,1,1};
    const char*tag[3]={"BASICexact","BASICicc","sASMicc"};
    PetscInt dit[3]={5,15,30};

    /* shared subdomains per icc flavour */
    for(int c=0;c<combos;++c){
        int NS=P*P; Sub*S=(Sub*)malloc(sizeof(Sub)*NS); int sc=0;
        for(int q=0;q<P;++q)for(int p=0;p<P;++p){
            PetscInt axlo=(p*n)/P,axhi=((p+1)*n)/P,bylo=(q*n)/P,byhi=((q+1)*n)/P;
            PetscInt al=axlo-O<0?0:axlo-O,ar=axhi+O>n?n:axhi+O,bl=bylo-O<0?0:bylo-O,br=byhi+O>n?n:byhi+O;
            PetscInt ni=(ar-al)*(br-bl),*idx=(PetscInt*)malloc(sizeof(PetscInt)*ni),cc=0;
            for(PetscInt bb=bl;bb<br;++bb)for(PetscInt a=al;a<ar;++a)idx[cc++]=bb*n+a;
            BuildSub(A,idx,ni,icc[c],&S[sc++]);free(idx);
        }
        Vec mult=Multiplicity(A,S,NS);
        Vec dsq;VecDuplicate(mult,&dsq);VecCopy(mult,dsq);VecReciprocal(dsq);VecSqrtAbs(dsq);
        /* seam mask: 1 where mult>1 */
        VecDuplicate(mult,&g_seammask);{const PetscScalar*ma;PetscScalar*sm;VecGetArrayRead(mult,&ma);VecGetArray(g_seammask,&sm);
            for(PetscInt k=0;k<N;++k)sm[k]=(PetscRealPart(ma[k])>1.5)?1.0:0.0;VecRestoreArrayRead(mult,&ma);VecRestoreArray(g_seammask,&sm);}
        if(c==0) DumpField(mult,n,"seamacc_mult.txt");
        Vec x;VecDuplicate(b,&x);
        PetscInt iters=PCG(A,S,NS,useS[c],dsq,b,x,1e-8,4000,tag[c],n,dit,3);
        PetscPrintf(PETSC_COMM_SELF,"%-12s converged in %d iters\n",tag[c],(int)iters);
        VecDestroy(&x);VecDestroy(&mult);VecDestroy(&dsq);VecDestroy(&g_seammask);
        for(int i=0;i<NS;++i){ISDestroy(&S[i].is);MatDestroy(&S[i].Ai);KSPDestroy(&S[i].ksp);VecDestroy(&S[i].ri);VecDestroy(&S[i].yi);free(S[i].idx);}free(S);
    }
    VecDestroy(&b);MatDestroy(&A);
    PetscPrintf(PETSC_COMM_SELF,"SEAMACC_DONE\n");
    PetscFinalize();return 0;
}
