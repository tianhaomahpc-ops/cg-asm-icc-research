/* monodomain.c -- Niederer (2011) monodomain benchmark on the 20x7x3 mm slab.
 *
 * PROBLEM: chi(Cm dV/dt + Iion) = div(sigma grad V) + Istim on Omega=[0,20]x[0,7]x[0,3] mm,
 * zero-flux (Neumann) tissue boundary. TP06 epicardial ionic model. Niederer params:
 * chi=140/mm, Cm=0.01 uF/mm^2, monodomain sigma_L=0.1334, sigma_T=0.0176 mS/mm (fiber || x).
 * Stimulus: 1.5x1.5x1.5 mm cube at the (0,0,0) corner, 2 ms.
 * DISCRETIZATION (cardiac.pdf): P1 (lumped-mass) FEM on a structured grid; Crank-Nicolson
 * diffusion + explicit reaction (IMEX) => Sys1 = M/dt + 1/2 D ; Rush-Larsen gating + FE conc.
 * OUTPUT: activation time (V crosses 0 mV) at the 9 benchmark points P1..P9 + the diagonal;
 *         compare the far-corner P8 time and the longitudinal conduction velocity.
 *
 * Linear solve A1 V = rhs by CG (this matrix IS the study's "Sys 1"). Sequential.
 */
#include <petscksp.h>
#include "../cardiac/tt06.h"

int main(int argc,char**argv){
    PetscInitialize(&argc,&argv,NULL,NULL);
    PetscReal h=0.5;                                  /* mm; pass -h for finer */
    PetscOptionsGetReal(NULL,NULL,"-h",&h,NULL);
    PetscReal dt=0.02, Tend=80.0;                     /* ms */
    PetscOptionsGetReal(NULL,NULL,"-dt",&dt,NULL);
    PetscOptionsGetReal(NULL,NULL,"-T",&Tend,NULL);
    PetscReal Lx=20.0,Ly=7.0,Lz=3.0;
    PetscInt nx=(PetscInt)(Lx/h)+1, ny=(PetscInt)(Ly/h)+1, nz=(PetscInt)(Lz/h)+1;
    PetscInt N=nx*ny*nz;
    PetscReal chi=140.0, Cm=0.01, chiCm=chi*Cm;
    PetscReal sL=0.1334, sT=0.0176;                    /* mS/mm monodomain */
    PetscReal DL=sL/chiCm, DT=sT/chiCm;                /* mm^2/ms diffusion */
    PetscReal Istim=-80.0, tstim=2.0;                  /* mV/ms, ms */
    PetscPrintf(PETSC_COMM_SELF,"Niederer slab: %gx%gx%g mm, h=%g => %dx%dx%d=%d nodes, dt=%g\n",
        Lx,Ly,Lz,h,(int)nx,(int)ny,(int)nz,(int)N,dt);
    PetscPrintf(PETSC_COMM_SELF,"DL=%.4f DT=%.4f mm^2/ms (chiCm=%.2f)\n",DL,DT,chiCm);

    #define IDX(i,j,k) ((k)*nx*ny+(j)*nx+(i))
    /* control-volume lumped mass */
    PetscReal *Ml=(PetscReal*)malloc(sizeof(PetscReal)*N);
    for(PetscInt k=0;k<nz;++k)for(PetscInt j=0;j<ny;++j)for(PetscInt i=0;i<nx;++i){
        PetscReal vx=(i==0||i==nx-1)?0.5:1.0, vy=(j==0||j==ny-1)?0.5:1.0, vz=(k==0||k==nz-1)?0.5:1.0;
        Ml[IDX(i,j,k)]=vx*vy*vz*h*h*h; }
    /* A1 = M/dt + 1/2 D ; B = M/dt - 1/2 D  (D = sigma/(chiCm)-weighted Neumann Laplacian) */
    Mat A1,B; MatCreateSeqAIJ(PETSC_COMM_SELF,N,N,7,NULL,&A1);
    MatCreateSeqAIJ(PETSC_COMM_SELF,N,N,7,NULL,&B);
    PetscReal cL=DL*h, cT=DT*h;                        /* off-diag conductance (3D: D*h) */
    for(PetscInt k=0;k<nz;++k)for(PetscInt j=0;j<ny;++j)for(PetscInt i=0;i<nx;++i){
        PetscInt r=IDX(i,j,k); PetscReal Dd=0.0, md=Ml[r]/dt;
        #define NB(ii,jj,kk,cc) { PetscInt cidx=IDX(ii,jj,kk); \
            MatSetValue(A1,r,cidx,-0.5*(cc),INSERT_VALUES); MatSetValue(B,r,cidx,0.5*(cc),INSERT_VALUES); Dd+=(cc); }
        if(i>0)    NB(i-1,j,k,cL); if(i<nx-1) NB(i+1,j,k,cL);
        if(j>0)    NB(i,j-1,k,cT); if(j<ny-1) NB(i,j+1,k,cT);
        if(k>0)    NB(i,j,k-1,cT); if(k<nz-1) NB(i,j,k+1,cT);
        MatSetValue(A1,r,r, md+0.5*Dd, INSERT_VALUES);
        MatSetValue(B, r,r, md-0.5*Dd, INSERT_VALUES);
    }
    MatAssemblyBegin(A1,MAT_FINAL_ASSEMBLY);MatAssemblyEnd(A1,MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);
    MatSetOption(A1,MAT_SYMMETRIC,PETSC_TRUE);

    KSP ksp;KSPCreate(PETSC_COMM_SELF,&ksp);KSPSetType(ksp,KSPCG);KSPSetOperators(ksp,A1,A1);
    PC pc;KSPGetPC(ksp,&pc);PCSetType(pc,PCICC);KSPSetTolerances(ksp,1e-8,1e-50,PETSC_DEFAULT,500);KSPSetFromOptions(ksp);

    TT06 *cell=(TT06*)malloc(sizeof(TT06)*N); for(PetscInt p=0;p<N;++p) tt06_init(&cell[p]);
    Vec V,rhs,Iv; MatCreateVecs(A1,&V,&rhs); VecDuplicate(V,&Iv);
    { PetscScalar*v;VecGetArray(V,&v);for(PetscInt p=0;p<N;++p)v[p]=cell[p].V;VecRestoreArray(V,&v); }
    PetscReal *tact=(PetscReal*)malloc(sizeof(PetscReal)*N); for(PetscInt p=0;p<N;++p)tact[p]=-1;

    /* pseudo-ECG electrode 20 mm from the slab; precompute 1/r weights */
    PetscReal ex=Lx/2, ey=Ly+20.0, ez=Lz/2;
    PetscReal *invr=(PetscReal*)malloc(sizeof(PetscReal)*N);
    for(PetscInt k=0;k<nz;++k)for(PetscInt j=0;j<ny;++j)for(PetscInt i=0;i<nx;++i){
        PetscReal dx=i*h-ex,dy=j*h-ey,dz=k*h-ez,r=sqrt(dx*dx+dy*dy+dz*dz)+1e-9; invr[IDX(i,j,k)]=1.0/r; }
    FILE*fe=fopen("mono_ecg.txt","w"); fprintf(fe,"# t(ms) pseudoECG(mV) Vm@center(mV)\n");

    PetscInt nsteps=(PetscInt)(Tend/dt);
    for(PetscInt s=0;s<nsteps;++s){
        PetscReal t=s*dt;
        const PetscScalar*vr;VecGetArrayRead(V,&vr);PetscScalar*iv;VecGetArray(Iv,&iv);
        for(PetscInt k=0;k<nz;++k)for(PetscInt j=0;j<ny;++j)for(PetscInt i=0;i<nx;++i){
            PetscInt p=IDX(i,j,k); cell[p].V=PetscRealPart(vr[p]);
            double Iion=tt06_react(&cell[p],dt);
            double Is=0.0;
            if(t<tstim && i*h<1.5 && j*h<1.5 && k*h<1.5) Is=Istim;
            iv[p]=-(Iion+Is);                          /* reaction+stim source (mV/ms) */
        }
        VecRestoreArrayRead(V,&vr);VecRestoreArray(Iv,&iv);
        /* rhs = B V + M*(reaction)  ;  reaction already = -(Iion+Istim) so add M.*Iv */
        MatMult(B,V,rhs);
        { const PetscScalar*ivp;PetscScalar*rp;VecGetArrayRead(Iv,&ivp);VecGetArray(rhs,&rp);
          for(PetscInt p=0;p<N;++p)rp[p]+=Ml[p]*PetscRealPart(ivp[p]);
          VecRestoreArrayRead(Iv,&ivp);VecRestoreArray(rhs,&rp); }
        KSPSolve(ksp,rhs,V);
        /* activation detection + pseudo-ECG: phi_e = sum_node [div(sigma grad Vm)]_node /r */
        const PetscScalar*vn;VecGetArrayRead(V,&vn);
        for(PetscInt p=0;p<N;++p) if(tact[p]<0 && PetscRealPart(vn[p])>=0.0) tact[p]=t+dt;
        if(s%((PetscInt)(1.0/dt))==0){                 /* sample every 1 ms */
            PetscReal ecg=0.0;
            for(PetscInt k=0;k<nz;++k)for(PetscInt j=0;j<ny;++j)for(PetscInt i=0;i<nx;++i){
                PetscInt p=IDX(i,j,k); PetscReal Vp=PetscRealPart(vn[p]), src=0.0;
                if(i>0)src+=cL*(PetscRealPart(vn[IDX(i-1,j,k)])-Vp); if(i<nx-1)src+=cL*(PetscRealPart(vn[IDX(i+1,j,k)])-Vp);
                if(j>0)src+=cT*(PetscRealPart(vn[IDX(i,j-1,k)])-Vp); if(j<ny-1)src+=cT*(PetscRealPart(vn[IDX(i,j+1,k)])-Vp);
                if(k>0)src+=cT*(PetscRealPart(vn[IDX(i,j,k-1)])-Vp); if(k<nz-1)src+=cT*(PetscRealPart(vn[IDX(i,j,k+1)])-Vp);
                ecg+=src*invr[p];
            }
            fprintf(fe,"%g %g %g\n",(double)(t+dt),(double)ecg,(double)PetscRealPart(vn[IDX(nx/2,ny/2,nz/2)]));
        }
        VecRestoreArrayRead(V,&vn);
    }
    fclose(fe);
    /* benchmark points (mm): P1=(0,0,0) corner stim ... P8=(20,7,3) far corner; centre P9 */
    struct{const char*name;PetscReal x,y,z;} P[9]={
        {"P1",0,0,0},{"P2",0,7,0},{"P3",20,0,0},{"P4",20,7,0},
        {"P5",0,0,3},{"P6",0,7,3},{"P7",20,0,3},{"P8",20,7,3}};
    PetscPrintf(PETSC_COMM_SELF,"\nActivation times (V crosses 0 mV):\n");
    PetscReal tP1=-1,tP8=-1;
    for(int q=0;q<8;++q){PetscInt i=(PetscInt)(P[q].x/h),jj=(PetscInt)(P[q].y/h),kk=(PetscInt)(P[q].z/h);
        PetscReal ta=tact[IDX(i,jj,kk)];
        PetscPrintf(PETSC_COMM_SELF,"  %s (%4.1f,%4.1f,%4.1f) : %.2f ms\n",P[q].name,P[q].x,P[q].y,P[q].z,(double)ta);
        if(q==0)tP1=ta; if(q==7)tP8=ta; }
    PetscReal diag=sqrt(Lx*Lx+Ly*Ly+Lz*Lz);
    PetscPrintf(PETSC_COMM_SELF,"\nP1->P8 corner-to-corner: dt_act=%.2f ms over %.2f mm diagonal => CV~%.3f m/s\n",
        (double)(tP8-tP1),(double)diag,(double)(diag/(tP8-tP1)/1000.0*1000.0));
    /* longitudinal CV along x at mid (y=3.5,z=1.5): activation at x=2 and x=18 */
    PetscInt jm=ny/2,km=nz/2; PetscReal tA=tact[IDX((PetscInt)(2.0/h),jm,km)], tBx=tact[IDX((PetscInt)(18.0/h),jm,km)];
    PetscPrintf(PETSC_COMM_SELF,"longitudinal CV (x=2->18 mm): %.3f m/s  (Niederer ~0.6-0.7 m/s)\n",
        (double)(16.0/(tBx-tA)));
    /* dump activation field (mid z-slice) */
    FILE*f=fopen("mono_tact.txt","w");
    for(PetscInt j=0;j<ny;++j){for(PetscInt i=0;i<nx;++i)fprintf(f,"%g ",tact[IDX(i,j,km)]);fprintf(f,"\n");}
    fclose(f);
    PetscPrintf(PETSC_COMM_SELF,"MONO_DONE\n");
    free(Ml);free(cell);free(tact);free(invr);VecDestroy(&V);VecDestroy(&rhs);VecDestroy(&Iv);
    MatDestroy(&A1);MatDestroy(&B);KSPDestroy(&ksp);PetscFinalize();return 0;
}
