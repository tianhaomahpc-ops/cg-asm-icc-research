/* xsys_precond.c -- Task 3: use the earlier system to precondition the later one.
 *
 * PROBLEM: in the cardiac pipeline the monodomain Vm (Sys 1) drives, at EVERY time step,
 * the elliptic "recover extracellular potential" solve
 *     Sys 2:   K_{sigma_i+sigma_e} u_e = - K_{sigma_i} Vm(t)        (pure Neumann, singular).
 * As the depolarization wave sweeps the slab, the RHS -- and hence u_e(t) -- varies
 * SMOOTHLY in time, so the sequence of solutions lives near a low-dimensional subspace.
 * Task 3 implements and compares every cross-system-preconditioning idea we discussed,
 * and finds the best:
 *   (0) baseline      : CG + ICC(0), zero initial guess (no transfer).
 *   (1) warm start    : CG + ICC(0), initial guess = previous u_e (history reuse).
 *   (2) POD deflation : Galerkin projection onto a POD basis built from the first K
 *                       solutions (solution-history / Fischer-POD transfer), then CG.
 *   (3) shared coarse : two-level CG with the Nicolaides coarse space built ONCE on the
 *                       shared heart mesh (geometric/structure transfer Sys1 -> Sys2).
 * Driver: a synthetic propagating depolarization front Vm(t,x)=Vrest+dV*sigmoid((x-c t)/w)
 * on the 20x7x3 mm slab (h=0.5), giving a smoothly-moving, consistent (mean-zero) RHS.
 * Metric: CG iterations per solve (and total) to ||r||/||b||<1e-8. Sequential.
 */
#include <petscksp.h>
#include <math.h>

typedef struct { PetscInt n; PetscInt *idx; IS is; Mat Ai; KSP ksp; Vec ri, yi; } Sub;

/* anisotropic Neumann sigma-Laplacian (stiffness) on the structured slab; singular */
static Mat stiffness(PetscInt nx,PetscInt ny,PetscInt nz,double h,double sx,double syz){
    PetscInt N=nx*ny*nz; Mat K; MatCreateSeqAIJ(PETSC_COMM_SELF,N,N,7,NULL,&K);
    double cx=sx*h, cy=syz*h, cz=syz*h;          /* 3D conductances = sigma*h */
    #define ID(i,j,k) ((k)*nx*ny+(j)*nx+(i))
    for(PetscInt k=0;k<nz;++k)for(PetscInt j=0;j<ny;++j)for(PetscInt i=0;i<nx;++i){
        PetscInt r=ID(i,j,k); double d=0;
        #define NB(ii,jj,kk,cc){MatSetValue(K,r,ID(ii,jj,kk),-(cc),INSERT_VALUES);d+=(cc);}
        if(i>0)NB(i-1,j,k,cx); if(i<nx-1)NB(i+1,j,k,cx);
        if(j>0)NB(i,j-1,k,cy); if(j<ny-1)NB(i,j+1,k,cy);
        if(k>0)NB(i,j,k-1,cz); if(k<nz-1)NB(i,j,k+1,cz);
        MatSetValue(K,r,r,d,INSERT_VALUES);
    }
    MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY);MatAssemblyEnd(K,MAT_FINAL_ASSEMBLY);
    MatSetOption(K,MAT_SYMMETRIC,PETSC_TRUE);
    MatNullSpace ns;MatNullSpaceCreate(PETSC_COMM_SELF,PETSC_TRUE,0,NULL,&ns);MatSetNullSpace(K,ns);MatNullSpaceDestroy(&ns);
    return K;
}

/* ---- additive Schwarz BASIC/sASM apply + optional Nicolaides coarse (shared with Sys1) ---- */
typedef struct { Sub*subs; PetscInt nsub; Vec dsq,tmp; int two; Mat R0t; KSP kspc; Vec cv,yc; } ASM;
static void asm_apply(ASM*a,Vec r,Vec z){
    VecCopy(r,a->tmp);VecPointwiseMult(a->tmp,a->tmp,a->dsq);VecZeroEntries(z);
    const PetscScalar*ra;PetscScalar*za;VecGetArrayRead(a->tmp,&ra);VecGetArray(z,&za);
    for(PetscInt i=0;i<a->nsub;++i){Sub*s=&a->subs[i];PetscScalar*rib;VecGetArray(s->ri,&rib);
        for(PetscInt j=0;j<s->n;++j)rib[j]=ra[s->idx[j]];VecRestoreArray(s->ri,&rib);
        KSPSolve(s->ksp,s->ri,s->yi);const PetscScalar*yib;VecGetArrayRead(s->yi,&yib);
        for(PetscInt j=0;j<s->n;++j)za[s->idx[j]]+=yib[j];VecRestoreArrayRead(s->yi,&yib);}
    VecRestoreArrayRead(a->tmp,&ra);VecRestoreArray(z,&za);VecPointwiseMult(z,z,a->dsq);
    if(a->two){MatMultTranspose(a->R0t,r,a->cv);KSPSolve(a->kspc,a->cv,a->yc);MatMultAdd(a->R0t,a->yc,z,z);}
}
static PetscErrorCode asm_pc(PC pc,Vec r,Vec z){ASM*a;PCShellGetContext(pc,&a);asm_apply(a,r,z);return 0;}

int main(int argc,char**argv){
    PetscInitialize(&argc,&argv,NULL,NULL);
    double h=0.5,Lx=20,Ly=7,Lz=3;
    PetscInt nx=(PetscInt)(Lx/h)+1,ny=(PetscInt)(Ly/h)+1,nz=(PetscInt)(Lz/h)+1,N=nx*ny*nz;
    double siL=0.17,siT=0.019,seL=0.62,seT=0.236;       /* S/m intra/extra */
    Mat Ki =stiffness(nx,ny,nz,h,siL,siT);              /* sigma_i */
    Mat Kie=stiffness(nx,ny,nz,h,siL+seL,siT+seT);      /* sigma_i+sigma_e = Sys 2 operator */
    PetscPrintf(PETSC_COMM_SELF,"Task 3 cross-system precond: slab %dx%dx%d=%d, recover Sys2=K_{si+se} (singular)\n",
        (int)nx,(int)ny,(int)nz,(int)N);

    /* sequence of consistent RHS from a moving depolarization front */
    int NT=40; double cv=0.12; /* mm per solve: realistic sampling of a 0.6 m/s front */
    Vec Vm,b,u,uprev; MatCreateVecs(Kie,&Vm,&b);VecDuplicate(Vm,&u);VecDuplicate(Vm,&uprev);
    Vec *sol=(Vec*)malloc(sizeof(Vec)*NT); for(int t=0;t<NT;++t)VecDuplicate(Vm,&sol[t]);

    /* build subdomains (4x2x1 boxes) for ASM, shared across systems */
    int Px=4,Py=2,Pz=1,O=2,NS=Px*Py*Pz; Sub*S=(Sub*)malloc(sizeof(Sub)*NS); int sc=0;
    for(int pz=0;pz<Pz;++pz)for(int py=0;py<Py;++py)for(int px=0;px<Px;++px){
        PetscInt al=(px*nx)/Px-O,ar=((px+1)*nx)/Px+O,bl=(py*ny)/Py-O,br=((py+1)*ny)/Py+O,cl=(pz*nz)/Pz-O,cr=((pz+1)*nz)/Pz+O;
        if(al<0)al=0;if(ar>nx)ar=nx;if(bl<0)bl=0;if(br>ny)br=ny;if(cl<0)cl=0;if(cr>nz)cr=nz;
        PetscInt ni=(ar-al)*(br-bl)*(cr-cl),*idx=(PetscInt*)malloc(sizeof(PetscInt)*ni),q=0;
        for(PetscInt k=cl;k<cr;++k)for(PetscInt j=bl;j<br;++j)for(PetscInt i=al;i<ar;++i)idx[q++]=k*nx*ny+j*nx+i;
        Sub*s=&S[sc++]; s->n=ni;s->idx=(PetscInt*)malloc(sizeof(PetscInt)*ni);for(PetscInt j=0;j<ni;++j)s->idx[j]=idx[j];
        ISCreateGeneral(PETSC_COMM_SELF,ni,s->idx,PETSC_COPY_VALUES,&s->is);
        MatCreateSubMatrix(Kie,s->is,s->is,MAT_INITIAL_MATRIX,&s->Ai);
        KSPCreate(PETSC_COMM_SELF,&s->ksp);KSPSetType(s->ksp,KSPPREONLY);KSPSetOperators(s->ksp,s->Ai,s->Ai);
        PC pc;KSPGetPC(s->ksp,&pc);PCSetType(pc,PCICC);PCFactorSetLevels(pc,0);KSPSetUp(s->ksp);
        MatCreateVecs(s->Ai,&s->ri,&s->yi); free(idx);
    }
    ASM A; A.subs=S;A.nsub=NS;A.two=0; MatCreateVecs(Kie,&A.tmp,NULL);
    { Vec m;MatCreateVecs(Kie,&m,NULL);VecZeroEntries(m);PetscScalar*ma;VecGetArray(m,&ma);
      for(int i=0;i<NS;++i)for(PetscInt j=0;j<S[i].n;++j)ma[S[i].idx[j]]+=1.0;VecRestoreArray(m,&ma);
      VecDuplicate(m,&A.dsq);VecCopy(m,A.dsq);VecReciprocal(A.dsq);VecSqrtAbs(A.dsq);
      /* Nicolaides coarse (shared structure) */
      Mat R0t;MatCreateSeqAIJ(PETSC_COMM_SELF,N,NS,32,NULL,&R0t);
      VecGetArray(m,&ma); for(int i=0;i<NS;++i)for(PetscInt j=0;j<S[i].n;++j)
          MatSetValue(R0t,S[i].idx[j],i,1.0/PetscRealPart(ma[S[i].idx[j]]),INSERT_VALUES);
      VecRestoreArray(m,&ma);MatAssemblyBegin(R0t,MAT_FINAL_ASSEMBLY);MatAssemblyEnd(R0t,MAT_FINAL_ASSEMBLY);
      A.R0t=R0t; Mat A0;MatPtAP(Kie,R0t,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&A0);
      MatNullSpace ns0;MatNullSpaceCreate(PETSC_COMM_SELF,PETSC_TRUE,0,NULL,&ns0);MatSetNullSpace(A0,ns0);MatNullSpaceDestroy(&ns0);
      KSPCreate(PETSC_COMM_SELF,&A.kspc);KSPSetType(A.kspc,KSPCG);KSPSetOperators(A.kspc,A0,A0);
      PC pc;KSPGetPC(A.kspc,&pc);PCSetType(pc,PCJACOBI);KSPSetTolerances(A.kspc,1e-10,1e-50,PETSC_DEFAULT,100);KSPSetUp(A.kspc);
      MatCreateVecs(A0,&A.yc,&A.cv);MatDestroy(&A0);VecDestroy(&m);
    }

    /* helper: build RHS b = -Ki*Vm(t), Vm a moving sigmoid front */
    #define BUILDRHS(tt){ PetscScalar*vp;VecGetArray(Vm,&vp); double front=2.0+cv*(tt); \
        for(PetscInt k=0;k<nz;++k)for(PetscInt j=0;j<ny;++j)for(PetscInt i=0;i<nx;++i){ \
            double x=i*h; vp[k*nx*ny+j*nx+i]=-85.0+110.0/(1.0+exp((x-front)/0.6)); } \
        VecRestoreArray(Vm,&vp); MatMult(Ki,Vm,b); VecScale(b,-1.0); \
        PetscScalar mean;VecSum(b,&mean);VecShift(b,-PetscRealPart(mean)/N); }

    /* run a strategy over the sequence, return total iters */
    int totals[5]={0,0,0,0,0};
    const char*name[5]={"baseline (zero IG)","warm start (prev u)","POD deflation (hist)","shared coarse (2-level)","warm + coarse (both)"};
    for(int strat=0;strat<5;++strat){
        A.two=(strat>=3)?1:0;
        KSP ksp;KSPCreate(PETSC_COMM_SELF,&ksp);KSPSetType(ksp,KSPCG);KSPSetOperators(ksp,Kie,Kie);
        PC pc;KSPGetPC(ksp,&pc);
        if(strat>=3){PCSetType(pc,PCSHELL);PCShellSetContext(pc,&A);PCShellSetApply(pc,asm_pc);}
        else {PCSetType(pc,PCICC);PCFactorSetLevels(pc,0);}
        KSPSetTolerances(ksp,1e-8,1e-50,PETSC_DEFAULT,2000);
        if(strat==1||strat==2||strat==4)KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);
        VecZeroEntries(uprev);
        /* POD basis storage */
        int Kpod=8; Vec*Phi=(Vec*)malloc(sizeof(Vec)*Kpod); for(int q=0;q<Kpod;++q)VecDuplicate(Vm,&Phi[q]);
        int npod=0;
        int tot=0;
        for(int t=0;t<NT;++t){
            BUILDRHS(t);
            if(strat==0||strat==3) VecZeroEntries(u);
            else if(strat==1||strat==4) VecCopy(uprev,u);
            else if(strat==2){ /* POD: Galerkin initial guess onto span(Phi[0..npod]) */
                VecZeroEntries(u);
                if(npod>0){ /* u0 = Phi (Phi^T K Phi)^-1 Phi^T b ; small dense solve */
                    double G[64*64],c0[64]; Vec KP; VecDuplicate(Vm,&KP);
                    for(int p=0;p<npod;++p){ MatMult(Kie,Phi[p],KP); PetscScalar bp;VecDot(Phi[p],b,&bp);c0[p]=PetscRealPart(bp);
                        for(int qq=0;qq<npod;++qq){PetscScalar g;VecDot(Phi[qq],KP,&g);G[qq*npod+p]=PetscRealPart(g);} }
                    /* solve G a = c0 (Gauss elim, npod<=8) */
                    double a[64]; for(int p=0;p<npod;++p)a[p]=c0[p];
                    for(int p=0;p<npod;++p){ double piv=G[p*npod+p]; if(fabs(piv)<1e-14)piv=1e-14;
                        for(int qq=p+1;qq<npod;++qq){double f=G[qq*npod+p]/piv; for(int rr=p;rr<npod;++rr)G[qq*npod+rr]-=f*G[p*npod+rr]; a[qq]-=f*a[p];}}
                    for(int p=npod-1;p>=0;--p){ for(int qq=p+1;qq<npod;++qq)a[p]-=G[p*npod+qq]*a[qq]; a[p]/=G[p*npod+p];}
                    for(int p=0;p<npod;++p)VecAXPY(u,a[p],Phi[p]);
                    VecDestroy(&KP);
                }
            }
            KSPSolve(ksp,b,u);
            PetscInt it;KSPGetIterationNumber(ksp,&it);tot+=it;
            VecCopy(u,uprev); if(strat==0)VecCopy(u,sol[t]);
            if(strat==2 && npod<Kpod){ VecCopy(sol[t],Phi[npod]); /* orthonormalize vs previous */
                for(int p=0;p<npod;++p){PetscScalar d;VecDot(Phi[npod],Phi[p],&d);VecAXPY(Phi[npod],-d,Phi[p]);}
                PetscReal nn;VecNorm(Phi[npod],NORM_2,&nn); if(nn>1e-10){VecScale(Phi[npod],1.0/nn);npod++;} }
        }
        totals[strat]=tot;
        PetscPrintf(PETSC_COMM_SELF,"  %-26s total CG iters over %d solves = %4d  (avg %.1f)\n",name[strat],NT,tot,(double)tot/NT);
        for(int q=0;q<Kpod;++q)VecDestroy(&Phi[q]); free(Phi); KSPDestroy(&ksp);
    }
    int best=0; for(int s=1;s<5;++s)if(totals[s]<totals[best])best=s;
    PetscPrintf(PETSC_COMM_SELF,"\nBEST cross-system strategy: %s (%d iters; baseline %d => %.1fx fewer)\n",
        name[best],totals[best],totals[0],(double)totals[0]/totals[best]);
    /* dump for plotting */
    FILE*f=fopen("xsys_totals.txt","w");for(int s=0;s<5;++s)fprintf(f,"%d %d\n",s,totals[s]);fclose(f);
    PetscPrintf(PETSC_COMM_SELF,"XSYS_DONE\n");
    PetscFinalize();return 0;
}
