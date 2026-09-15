/* asm_stride_test.c -- how often should the extracellular / torso solves run?
 *
 * Three configurations the forward-ECG pipeline can be run in:
 *
 *   1  Sys2 (u_e recovery) and Sys3 (torso) solved EVERY monodomain step
 *      (dt = 0.01 ms), sASM + CG, cold start.
 *   2  Sys2 / Sys3 solved once every 1 ms (stride = 100 steps), sASM + CG,
 *      cold start.
 *   3  same as 2, plus the true sliding-window (Fischer) initial guess.
 *
 * and, for the comparison to mean anything, the fourth cell of the table:
 *
 *   4  stride = 1 WITH the sliding window -- "solve every step but almost for
 *      free" is the direct competitor of "solve 100x less often".
 *
 * The three systems are the same ones as in fischer_sys13_test.c (identical
 * grid, conductivities, Vm(t) driver), so the numbers are comparable with the
 * earlier window study.  Two things are new:
 *
 *   sASM   symmetric Additive Schwarz, the preconditioner the production code
 *          actually uses.  Overlapping boxes, IC(0) local solves, R_i^T (not
 *          restricted/RAS) so the operator stays symmetric and CG is valid.
 *          Selectable against Jacobi to show how much of the earlier result
 *          was an artefact of the weaker preconditioner.
 *   dt     an explicit millisecond time step.  fischer_sys13_test.c stepped at
 *          1 ms/step; the production run steps at 0.01 ms, which is exactly the
 *          variable under study here.
 *
 * Build: cc -O2 -o asm_stride_test asm_stride_test.c -lm
 * Run:   ./asm_stride_test [n] [steps] [dt_ms] [stride] [win] [regime]
 *                          [nsub/axis] [overlap] [pc: 0=jacobi 1=sASM]
 *                          [probe file]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static int N, ND;
#define IDX(i,j,k) ((((i)*N)+(j))*N+(k))

static const double SIE[3] = {1.00, 0.35, 0.35};   /* sigma_i + sigma_e (Sys2) */
static const double SI [3] = {0.30, 0.03, 0.03};   /* sigma_i           (Sys2) */
static const double SM [3] = {0.45, 0.16, 0.16};   /* monodomain sigma  (Sys1) */
static const double ST [3] = {1.00, 1.00, 1.00};   /* torso, isotropic  (Sys3) */
static double C1   = 0.5;     /* chi*Cm/dt for Sys1, tuned to the real run's
                               * Sys1 = 15 iterations/step at dt = 0.01 ms */
static double GCUT = 1e-8;

static char   *isD;
static double *dg1, *dg2, *dg3;

static void applyK(const double *sg, const double *x, double *y, int dirich)
{
    for (int i=0;i<N;i++) for (int j=0;j<N;j++) for (int k=0;k<N;k++){
        int p = IDX(i,j,k);
        if (dirich && isD[p]){ y[p]=0.0; continue; }
        double xp = x[p], s = 0.0;
        int nb[6]; double sg6[6]; int nn=0;
        if (i>0)  { nb[nn]=p-N*N; sg6[nn++]=sg[0]; }
        if (i<N-1){ nb[nn]=p+N*N; sg6[nn++]=sg[0]; }
        if (j>0)  { nb[nn]=p-N;   sg6[nn++]=sg[1]; }
        if (j<N-1){ nb[nn]=p+N;   sg6[nn++]=sg[1]; }
        if (k>0)  { nb[nn]=p-1;   sg6[nn++]=sg[2]; }
        if (k<N-1){ nb[nn]=p+1;   sg6[nn++]=sg[2]; }
        for (int q=0;q<nn;q++){
            double xq = (dirich && isD[nb[q]]) ? 0.0 : x[nb[q]];
            s += sg6[q]*(xp - xq);
        }
        y[p] = s;
    }
}
static void applyA(int sys, const double *x, double *y)
{
    if (sys==1){ applyK(SM,x,y,0); for (int p=0;p<ND;p++) y[p] = C1*x[p] + 0.5*y[p]; }
    else if (sys==2) applyK(SIE,x,y,0);
    else             applyK(ST ,x,y,1);
}
static double *diagof(int sys){ return sys==1? dg1 : (sys==2? dg2 : dg3); }

/* the off-diagonal entry A(p,q) for the neighbour q of p along axis ax */
static double a_off(int sys,int ax,int p,int q)
{
    if (sys==1) return -0.5*SM[ax];
    if (sys==2) return -SIE[ax];
    return (isD[p]||isD[q]) ? 0.0 : -ST[ax];
}

static void build_diags(void)
{
    dg1=malloc(ND*sizeof(double)); dg2=malloc(ND*sizeof(double)); dg3=malloc(ND*sizeof(double));
    for (int i=0;i<N;i++) for (int j=0;j<N;j++) for (int k=0;k<N;k++){
        int p=IDX(i,j,k); double dm=0,de=0,dt=0;
        int has[6]={i>0,i<N-1,j>0,j<N-1,k>0,k<N-1};
        int ax [6]={0,0,1,1,2,2};
        for (int q=0;q<6;q++) if (has[q]){ dm += SM[ax[q]]; de += SIE[ax[q]]; dt += ST[ax[q]]; }
        dg1[p] = C1 + 0.5*dm;
        dg2[p] = de;
        dg3[p] = isD[p] ? 1.0 : dt;
    }
}

/* ------------------------------------------------------------- vector ops */
static double wall(void)
{ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
  return ts.tv_sec + 1e-9*ts.tv_nsec; }
static double dot(const double*a,const double*b){ double s=0; for(int i=0;i<ND;i++) s+=a[i]*b[i]; return s; }
static double nrm2(const double*a){ return sqrt(dot(a,a)); }
static void   axpy(double a,const double*x,double*y){ for(int i=0;i<ND;i++) y[i]+=a*x[i]; }
static void   scal(double a,double*x){ for(int i=0;i<ND;i++) x[i]*=a; }
static void   copyv(const double*x,double*y){ memcpy(y,x,ND*sizeof(double)); }
static void   zerov(double*x){ memset(x,0,ND*sizeof(double)); }
static void   rmmean(double*x){ double s=0; for(int i=0;i<ND;i++) s+=x[i]; s/=ND; for(int i=0;i<ND;i++) x[i]-=s; }
static void   zeroD(double*x){ for(int p=0;p<ND;p++) if (isD[p]) x[p]=0.0; }

/* =============================================== symmetric Additive Schwarz
 * z = sum_i R_i^T (L_i L_i^T)^{-1} R_i r
 *
 * R_i restricts to an overlapping box, L_i is the IC(0) factor of the LOCAL
 * matrix R_i A R_i^T (full diagonal of A kept, couplings that leave the box
 * dropped -- so the local matrix is strictly diagonally dominant on the box
 * boundary and SPD even for the singular Sys2).  R_i^T and R_i are transposes
 * of each other, so the preconditioner is symmetric and CG stays valid; the
 * restricted variant (RAS) is cheaper in communication but NOT symmetric.
 *
 * The local factor keeps the box's own 7-point structure, so the IC(0) sweep
 * is three back-neighbours per node and the apply is two triangular sweeps.  */
typedef struct {
    int i0,j0,k0, la,lb,lc, nl;
    double *d, *ea, *eb, *ec;    /* IC(0): diag of L, and L(q,q-stride_{a,b,c}) */
} Sub;
static Sub   *SUB[4] = {NULL,NULL,NULL,NULL};
static int    NS = 0, NSX=1, NSY=1, NSZ=1, OVL = 0;
static double *ybuf = NULL;   static int YBUF_N = 0;
static int    PCTYPE = 1;            /* 0 = Jacobi, 1 = sASM, 2 = ASM BASIC */
static long   IC_SHIFTS = 0;
/* sASM = D^{-1/2} M_BASIC^{-1} D^{-1/2}, D = overlap multiplicity.  This is the
 * scheme-3 PCSHELL of asm_demo.cpp: self-adjoint, so CG stays valid, and it
 * forces the over-counting factor N_c in
 *     kappa(M^-1 A) <~ C(1+H/delta) * (w_max/w_min) * N_c
 * down to 1.  Without it (PCTYPE=2, "ASM BASIC") a dof in the overlap is summed
 * once per owning subdomain, which is why BASIC gets WORSE as overlap grows. */
static double *wmul = NULL;          /* D^{-1/2} */

static void asm_build(int sys,int ovl)
{
    const double *dg = diagof(sys);
    const int nsub = NSX*NSY*NSZ;
    if (!SUB[sys]) SUB[sys] = calloc((size_t)nsub,sizeof(Sub));
    int s = 0, maxnl = 0;
    int *mult = (sys==1)? calloc(ND,sizeof(int)) : NULL;
    for (int bi=0;bi<NSX;bi++) for (int bj=0;bj<NSY;bj++) for (int bk=0;bk<NSZ;bk++,s++){
        Sub *S = &SUB[sys][s];
        int i0 = bi*N/NSX - ovl,     i1 = (bi+1)*N/NSX + ovl;
        int j0 = bj*N/NSY - ovl,     j1 = (bj+1)*N/NSY + ovl;
        int k0 = bk*N/NSZ - ovl,     k1 = (bk+1)*N/NSZ + ovl;
        if (i0<0) i0=0; if (i1>N) i1=N;
        if (j0<0) j0=0; if (j1>N) j1=N;
        if (k0<0) k0=0; if (k1>N) k1=N;
        S->i0=i0; S->j0=j0; S->k0=k0;
        S->la=i1-i0; S->lb=j1-j0; S->lc=k1-k0;
        S->nl = S->la*S->lb*S->lc;
        if (S->nl > maxnl) maxnl = S->nl;
        S->d =malloc(S->nl*sizeof(double)); S->ea=malloc(S->nl*sizeof(double));
        S->eb=malloc(S->nl*sizeof(double)); S->ec=malloc(S->nl*sizeof(double));
        const int sa = S->lb*S->lc, sb = S->lc, sc = 1;
        for (int a=0;a<S->la;a++) for (int b=0;b<S->lb;b++) for (int c=0;c<S->lc;c++){
            int q = (a*S->lb + b)*S->lc + c;
            int p = IDX(i0+a, j0+b, k0+c);
            double t = dg[p], xa=0, xb=0, xc=0;
            if (a>0){ xa = a_off(sys,0,p,p-N*N)/S->d[q-sa]; t -= xa*xa; }
            if (b>0){ xb = a_off(sys,1,p,p-N  )/S->d[q-sb]; t -= xb*xb; }
            if (c>0){ xc = a_off(sys,2,p,p-1  )/S->d[q-sc]; t -= xc*xc; }
            if (t <= 1e-14*dg[p]){ t = dg[p]; IC_SHIFTS++; }   /* diagonal shift */
            S->d[q]=sqrt(t); S->ea[q]=xa; S->eb[q]=xb; S->ec[q]=xc;
            if (mult) mult[p]++;
        }
    }
    NS = nsub;
    if (maxnl > YBUF_N){ free(ybuf); ybuf = malloc((size_t)maxnl*sizeof(double)); YBUF_N = maxnl; }
    if (mult){                                   /* geometry only: built once */
        wmul = malloc(ND*sizeof(double));
        for (int p=0;p<ND;p++) wmul[p] = 1.0/sqrt((double)(mult[p]?mult[p]:1));
        free(mult);
    }
}

static double *rscal = NULL;
static void asm_apply(int sys,const double*r,double*z)
{
    if (PCTYPE==1){                               /* r <- D^{-1/2} r */
        if (!rscal) rscal = malloc(ND*sizeof(double));
        for (int i=0;i<ND;i++) rscal[i] = r[i]*wmul[i];
        r = rscal;
    }
    zerov(z);
    for (int s=0;s<NS;s++){
        const Sub *S = &SUB[sys][s];
        const int sa = S->lb*S->lc, sb = S->lc, sc = 1;
        double *y = ybuf;
        /* gather + forward solve  L y = R r */
        for (int a=0;a<S->la;a++) for (int b=0;b<S->lb;b++){
            int base = (a*S->lb + b)*S->lc;
            int g    = IDX(S->i0+a, S->j0+b, S->k0);
            for (int c=0;c<S->lc;c++){
                int q = base + c;
                double t = r[g+c];
                if (a>0) t -= S->ea[q]*y[q-sa];
                if (b>0) t -= S->eb[q]*y[q-sb];
                if (c>0) t -= S->ec[q]*y[q-sc];
                y[q] = t/S->d[q];
            }
        }
        /* backward solve  L^T y = y */
        for (int a=S->la-1;a>=0;a--) for (int b=S->lb-1;b>=0;b--){
            int base = (a*S->lb + b)*S->lc;
            for (int c=S->lc-1;c>=0;c--){
                int q = base + c;
                double t = y[q];
                if (a<S->la-1) t -= S->ea[q+sa]*y[q+sa];
                if (b<S->lb-1) t -= S->eb[q+sb]*y[q+sb];
                if (c<S->lc-1) t -= S->ec[q+sc]*y[q+sc];
                y[q] = t/S->d[q];
            }
        }
        /* scatter-add  z += R^T y  (additive, no partition of unity) */
        for (int a=0;a<S->la;a++) for (int b=0;b<S->lb;b++){
            int base = (a*S->lb + b)*S->lc;
            int g    = IDX(S->i0+a, S->j0+b, S->k0);
            for (int c=0;c<S->lc;c++) z[g+c] += y[base+c];
        }
    }
    if (PCTYPE==1) for (int i=0;i<ND;i++) z[i] *= wmul[i];   /* z <- D^{-1/2} z */
}

static void precond(int sys,const double*r,double*z)
{
    if (PCTYPE) asm_apply(sys,r,z);
    else { const double *dg=diagof(sys); for (int i=0;i<ND;i++) z[i]=r[i]/dg[i]; }
    if (sys==2) rmmean(z);
    if (sys==3) zeroD(z);
}

/* CG with the unpreconditioned residual test done BEFORE the first iteration,
 * so a good enough guess costs 0 iterations -- and with rtol = 0, atol = e||b||
 * so the test does not move with the initial guess. */
static int cg(int sys, const double *b, double *x, double atol, int maxit)
{
    static double *r=NULL,*z=NULL,*p=NULL,*Ap=NULL;
    if (!r){ r=malloc(ND*sizeof(double)); z=malloc(ND*sizeof(double));
             p=malloc(ND*sizeof(double)); Ap=malloc(ND*sizeof(double)); }
    applyA(sys,x,Ap);
    for (int i=0;i<ND;i++) r[i]=b[i]-Ap[i];
    if (sys==3) zeroD(r);
    if (nrm2(r) <= atol) return 0;
    precond(sys,r,z);
    copyv(z,p);
    double rz = dot(r,z);
    int it;
    for (it=1; it<=maxit; ++it){
        applyA(sys,p,Ap);
        double pAp = dot(p,Ap);
        if (pAp <= 0) break;
        double a = rz/pAp;
        axpy(a,p,x); axpy(-a,Ap,r);
        if (nrm2(r) <= atol) break;
        precond(sys,r,z);
        double rz2 = dot(r,z), beta = rz2/rz; rz = rz2;
        for (int i=0;i<ND;i++) p[i]=z[i]+beta*p[i];
    }
    return it;
}

/* -------------------------------- small dense symmetric truncated pseudo-solve */
#define MMAX 96
static double Gbuf[MMAX][MMAX], Vbuf[MMAX][MMAX], Abuf[MMAX][MMAX];
static void sym_eig(double A[MMAX][MMAX], double V[MMAX][MMAX], double *ev, int m)
{
    for (int i=0;i<m;i++) for (int j=0;j<m;j++) V[i][j]=(i==j);
    for (int sweep=0; sweep<60; ++sweep){
        double off=0;
        for (int i=0;i<m;i++) for (int j=i+1;j<m;j++) off += A[i][j]*A[i][j];
        if (off < 1e-30) break;
        for (int p=0;p<m;p++) for (int q=p+1;q<m;q++){
            if (fabs(A[p][q]) < 1e-300) continue;
            double th = 0.5*(A[q][q]-A[p][p])/A[p][q];
            double t  = (th>=0? 1.0:-1.0)/(fabs(th)+sqrt(th*th+1.0));
            double c  = 1.0/sqrt(t*t+1.0), s = t*c;
            for (int k=0;k<m;k++){ double akp=A[k][p],akq=A[k][q];
                A[k][p]=c*akp-s*akq; A[k][q]=s*akp+c*akq; }
            for (int k=0;k<m;k++){ double apk=A[p][k],aqk=A[q][k];
                A[p][k]=c*apk-s*aqk; A[q][k]=s*apk+c*aqk; }
            for (int k=0;k<m;k++){ double vkp=V[k][p],vkq=V[k][q];
                V[k][p]=c*vkp-s*vkq; V[k][q]=s*vkp+c*vkq; }
        }
    }
    for (int i=0;i<m;i++) ev[i]=A[i][i];
}
static void gram_solve(double G[MMAX][MMAX],const double*f,double*c,int m,double cut)
{
    double ev[MMAX];
    for (int i=0;i<m;i++) for (int j=0;j<m;j++) Abuf[i][j]=G[i][j];
    sym_eig(Abuf,Vbuf,ev,m);
    double lmax=0; for (int i=0;i<m;i++) if (ev[i]>lmax) lmax=ev[i];
    for (int i=0;i<m;i++) c[i]=0.0;
    for (int e=0;e<m;e++){
        if (ev[e] <= cut*lmax || ev[e] <= 0) continue;
        double vf=0; for (int i=0;i<m;i++) vf += Vbuf[i][e]*f[i];
        double s = vf/ev[e];
        for (int i=0;i<m;i++) c[i] += s*Vbuf[i][e];
    }
}

/* --------------------------------------------- true sliding-window recycler */
typedef struct {
    int sys, mmax, m;
    double *P, *AP, *G;
    long accepted;
} Rec;
static void rec_init(Rec*R,int sys,int mmax)
{
    R->sys=sys; R->mmax=mmax; R->m=0; R->accepted=0;
    R->P =calloc((size_t)mmax*ND,sizeof(double));
    R->AP=calloc((size_t)mmax*ND,sizeof(double));
    R->G =calloc((size_t)mmax*mmax,sizeof(double));
}
static void rec_guess(const Rec*R,const double*b,double*x0)
{
    zerov(x0); if (!R->m) return;
    double f[MMAX], c[MMAX];
    for (int i=0;i<R->m;i++){
        f[i]=dot(R->P+(size_t)i*ND,b);
        for (int j=0;j<R->m;j++) Gbuf[i][j]=R->G[(size_t)i*R->mmax+j];
    }
    gram_solve(Gbuf,f,c,R->m,GCUT);
    for (int i=0;i<R->m;i++) axpy(c[i],R->P+(size_t)i*ND,x0);
}
static void rec_update(Rec*R,const double*u,double*w,double*Aw)
{
    copyv(u,w);
    if (R->sys==2) rmmean(w);
    if (R->sys==3) zeroD(w);
    applyA(R->sys,w,Aw);
    int m = R->m;
    double gnew[MMAX+1];
    for (int i=0;i<m;i++) gnew[i]=dot(R->P+(size_t)i*ND,Aw);
    gnew[m]=dot(w,Aw);
    if (m>=R->mmax){                                   /* drop the oldest SNAPSHOT */
        for (int i=0;i<m-1;i++){
            copyv(R->P +(size_t)(i+1)*ND, R->P +(size_t)i*ND);
            copyv(R->AP+(size_t)(i+1)*ND, R->AP+(size_t)i*ND);
        }
        m--;
        for (int i=0;i<=m;i++) gnew[i]=gnew[i+1];
        for (int i=1;i<=m;i++) for (int j=1;j<=m;j++)
            R->G[(size_t)(i-1)*R->mmax+(j-1)] = R->G[(size_t)i*R->mmax+j];
    }
    for (int i=0;i<m;i++){ R->G[(size_t)i*R->mmax+m]=gnew[i];
                           R->G[(size_t)m*R->mmax+i]=gnew[i]; }
    R->G[(size_t)m*R->mmax+m]=gnew[m];
    copyv(w, R->P+(size_t)m*ND); copyv(Aw,R->AP+(size_t)m*ND);
    R->m=m+1; R->accepted++;
}

/* ------------------------------------------------------------ Vm(x,t) model */
static double *tact, *apd, *mode;
static int REGIME;
static const double BEAT = 350.0;
static double ap_shape(double tau,double A)
{
    const double VR=-83.0;
    if (tau < 0)    return VR;
    if (tau < 1.0)  return VR + 113.0*tau;
    if (tau < 5.0)  return 30.0 - 10.0*(tau-1.0)/4.0;
    if (tau < A)    return 20.0 - 15.0*(tau-5.0)/(A-5.0);
    if (tau < A+60.0){ double s=(tau-A)/60.0; return 5.0 + (VR-5.0)*0.5*(1.0-cos(M_PI*s)); }
    return VR;
}
static void vm_field(double t,double *vm)
{
    if (REGIME==2){
        const double f[4]={0.091,0.127,0.053,0.113}, ph[4]={0.0,1.7,3.1,0.6};
        double a[4]; for (int q=0;q<4;q++) a[q]=sin(2*M_PI*f[q]*t+ph[q]);
        for (int p=0;p<ND;p++){ double v=0;
            for (int q=0;q<4;q++) v += a[q]*mode[(size_t)q*ND+p];
            vm[p]=v; }
        return;
    }
    double tb = fmod(t,BEAT);
    for (int p=0;p<ND;p++) vm[p] = ap_shape(tb-tact[p], apd[p]);
}

/* ===================================================================== main */
/* The five initial guesses.  V_SPACED is the point of this testbed: a window of
 * m CONSECUTIVE solutions at dt = 0.01 ms spans only 0.16 ms, over which the
 * trajectory is very nearly a straight line -- the snapshots are almost
 * linearly dependent and the Gram is near-singular, so the window buys little
 * more than a warm start.  Spacing the SAME m snapshots `wstride` solves apart
 * makes them span wstride*m steps of real trajectory instead.  V_SPACED_LAST
 * adds the freshest solution back as an extra column, so the guess is optimal
 * over span{spaced history} + span{previous solution} -- the best of both. */
enum { V_COLD=0, V_WARM=1, V_FISCH=2, V_SPACED=3, V_SPACED_LAST=4, NV=5 };
static const char *VN[NV] = {"cold(不做回收)","warm(上一次解)","滑窗 m(连续)",
                             "稀疏窗 m(隔 w 次)","稀疏窗 + 最新解"};

/* A-optimal guess over span{R} + span{p}, where Ap is already known.
 * Augments the maintained Gram with one cross row instead of rebuilding it. */
static void rec_guess_plus(const Rec*R,const double*b,const double*p,
                           const double*Ap,double*x0)
{
    int m = R? R->m : 0;
    if (m+1 > MMAX){ rec_guess(R,b,x0); return; }
    double f[MMAX], c[MMAX];
    for (int i=0;i<m;i++){
        f[i]=dot(R->P+(size_t)i*ND,b);
        for (int j=0;j<m;j++) Gbuf[i][j]=R->G[(size_t)i*R->mmax+j];
        double g = dot(R->P+(size_t)i*ND, Ap);      /* cross term <u^i, A p> */
        Gbuf[i][m]=g; Gbuf[m][i]=g;
    }
    Gbuf[m][m]=dot(p,Ap); f[m]=dot(p,b);
    gram_solve(Gbuf,f,c,m+1,GCUT);
    zerov(x0);
    for (int i=0;i<m;i++) axpy(c[i],R->P+(size_t)i*ND,x0);
    axpy(c[m],p,x0);
}

int main(int argc,char**argv)
{
    N          = (argc>1)? atoi(argv[1]) : 24;
    int nT     = (argc>2)? atoi(argv[2]) : 6000;
    double DT  = (argc>3)? atof(argv[3]) : 0.01;     /* ms per monodomain step */
    int stride = (argc>4)? atoi(argv[4]) : 1;        /* Sys2/Sys3 every `stride` */
    int win    = (argc>5)? atoi(argv[5]) : 16;
    REGIME     = (argc>6)? atoi(argv[6]) : 0;
    const char *sub = (argc>7)? argv[7] : "1x2x2";   /* subdomain grid, e.g. 1x2x2 */
    OVL        = (argc>8)? atoi(argv[8]) : 1;
    PCTYPE     = (argc>9)? atoi(argv[9]) : 1;         /* 0 Jacobi 1 sASM 2 ASM BASIC */
    double TOL = (argc>10)? pow(10.0,-atof(argv[10])) : 1e-8;
    int wstride= (argc>11)? atoi(argv[11]) : 1;   /* window snapshot spacing */
    if (argc>12) GCUT = pow(10.0,-atof(argv[12]));
    const char *probef = (argc>13)? argv[13] : NULL;
    if (wstride < 1) wstride = 1;
    if (sscanf(sub,"%dx%dx%d",&NSX,&NSY,&NSZ)!=3){ NSX=NSY=NSZ=atoi(sub); }
    ND = N*N*N;
    const int W4 = (win*4 <= MMAX)? win*4 : MMAX;

    isD = calloc(ND,1);
    for (int j=0;j<N;j++) for (int k=0;k<N;k++) isD[IDX(0,j,k)] = 1;
    build_diags();

    tact=malloc(ND*sizeof(double)); apd=malloc(ND*sizeof(double));
    for (int i=0;i<N;i++) for (int j=0;j<N;j++) for (int k=0;k<N;k++){
        double dx=i/2.2,dy=j,dz=k;
        tact[IDX(i,j,k)] = 0.5*sqrt(dx*dx+dy*dy+dz*dz);
        apd [IDX(i,j,k)] = 170.0 + 90.0*((double)i/(N-1));
    }
    if (REGIME==2){
        mode=malloc((size_t)4*ND*sizeof(double));
        for (int i=0;i<N;i++) for (int j=0;j<N;j++) for (int k=0;k<N;k++){
            double X=(double)i/(N-1),Y=(double)j/(N-1),Z=(double)k/(N-1); int p=IDX(i,j,k);
            mode[0*ND+p]=60.0*exp(-8.0*((X-0.3)*(X-0.3)+(Y-0.4)*(Y-0.4)+(Z-0.5)*(Z-0.5)));
            mode[1*ND+p]=40.0*(X-0.5);
            mode[2*ND+p]=30.0*sin(M_PI*Y)*cos(2*M_PI*Z);
            mode[3*ND+p]=25.0*sin(2*M_PI*X)*sin(M_PI*Z);
        }
    }

    double t_pc = wall();
    if (PCTYPE) for (int sy=1;sy<=3;sy++) asm_build(sy,OVL);
    t_pc = wall()-t_pc;

    /* ---- calibration mode (steps <= 0): one cold Sys2 solve, the same problem
     * as recoverue_demo in REPORT_Sys2_zh.md (nx=24, 4 ranks, overlap 1,
     * ICC(0)) -- so this testbed's preconditioner can be checked against the
     * published 68 (ASM BASIC) / 52 (sASM) before anything is concluded. */
    if (nT <= 0){
        double *vmc=malloc(ND*sizeof(double)), *bc=malloc(ND*sizeof(double));
        double *xc=malloc(ND*sizeof(double));
        vm_field(6.0,vmc);                        /* mid-activation, like the demo */
        applyK(SI,vmc,bc,0); scal(-1.0,bc); rmmean(bc);
        double nb=nrm2(bc);
        for (int pc=0;pc<3;pc++){
            const char *nm[3]={"Jacobi","sASM(D^-1/2 sandwich)","ASM BASIC"};
            PCTYPE=pc; zerov(xc);
            int it = cg(2,bc,xc,TOL*nb,4000);
            printf("# CALIB Sys2  n=%d nd=%d sub=%dx%dx%d ovl=%d tol=%.0e  %-24s %4d iters\n",
                   N,ND,NSX,NSY,NSZ,OVL,TOL,nm[pc],it);
        }
        return 0;
    }

    double *vm=malloc(ND*sizeof(double)), *b=malloc(ND*sizeof(double));
    double *x0=malloc(ND*sizeof(double)), *u=malloc(ND*sizeof(double));
    double *w =malloc(ND*sizeof(double)), *Aw=malloc(ND*sizeof(double));
    double *prev[4], *ue=malloc(ND*sizeof(double));
    for (int q=1;q<=3;q++){ prev[q]=malloc(ND*sizeof(double)); zerov(prev[q]); }
    zerov(ue);
    Rec Wm[4], Wsp[4];
    for (int sy=1;sy<=3;sy++){ rec_init(&Wm[sy],sy,win); rec_init(&Wsp[sy],sy,win); }
    double *prevA[4]; for (int q=1;q<=3;q++){ prevA[q]=malloc(ND*sizeof(double)); zerov(prevA[q]); }
    (void)W4;

    long tot[4][NV], zer[4][NV], nsolve[4]; double tim[4][NV];
    for (int sy=1;sy<=3;sy++){ nsolve[sy]=0;
        for (int v=0;v<NV;v++){ tot[sy][v]=0; zer[sy][v]=0; tim[sy][v]=0; } }

    /* torso "electrodes": 8 nodes on the far face i = N-1 */
    int NE=8, epr[8];
    for (int e=0;e<NE;e++){
        int jj = (N-1)*((e%4)+1)/5, kk = (N-1)*((e/4)+1)/3;
        epr[e] = IDX(N-1,jj,kk);
    }
    FILE *pf = probef? fopen(probef,"w") : NULL;
    if (pf){ fprintf(pf,"step,t_ms"); for (int e=0;e<NE;e++) fprintf(pf,",e%d",e); fprintf(pf,"\n"); }

    printf("# asm_stride_test  n=%d nd=%d steps=%d dt=%g ms (T=%g ms) stride=%d "
           "(Sys2/Sys3 every %g ms) win=%d wstride=%d (span %g ms) gcut=%.0e regime=%d "
           "pc=%s nsub=%d ovl=%d "
           "setup=%.3f s\n",
           N,ND,nT,DT,nT*DT,stride,stride*DT,win,wstride,
           (double)win*wstride*stride*DT,GCUT,REGIME,
           PCTYPE==1?"sASM+IC(0)":(PCTYPE==2?"ASM-BASIC+IC(0)":"Jacobi"),
           NSX*NSY*NSZ,OVL,t_pc);
    printf("step,t_ms");
    for (int sy=1;sy<=3;sy++) for (int v=0;v<NV;v++) printf(",s%d_v%d",sy,v);
    printf("\n");

    for (int st=0; st<nT; ++st){
        double t = st*DT;
        vm_field(t,vm);
        printf("%d,%.4f",st,t);

        for (int sy=1;sy<=3;sy++){
            int active = (sy==1) || (st % stride == 0);
            if (!active){ for (int v=0;v<NV;v++) printf(",-"); continue; }

            if (sy==1)      applyA(1,vm,b);                       /* exact sol = Vm */
            else if (sy==2){ applyK(SI,vm,b,0); scal(-1.0,b); rmmean(b); }
            else { zerov(b);                                       /* lift u_e|_Gamma */
                for (int j=0;j<N;j++) for (int kk=0;kk<N;kk++)
                    b[IDX(1,j,kk)] += ST[0]*ue[IDX(0,j,kk)];
                zeroD(b); }
            double nb=nrm2(b);
            if (nb < 1e-9){ for (int v=0;v<NV;v++) printf(",0"); continue; }
            double atol=TOL*nb;
            nsolve[sy]++;

            for (int v=0; v<NV; ++v){
                double t0 = wall();
                if      (v==V_COLD)   zerov(x0);
                else if (v==V_WARM)   copyv(prev[sy],x0);
                else if (v==V_FISCH)  rec_guess(&Wm[sy],b,x0);
                else if (v==V_SPACED) rec_guess(&Wsp[sy],b,x0);
                else                  rec_guess_plus(&Wsp[sy],b,prev[sy],prevA[sy],x0);
                if (sy==3) zeroD(x0);
                int it = cg(sy,b,x0,atol,4000);
                tim[sy][v]+=wall()-t0;
                tot[sy][v]+=it; if(!it) zer[sy][v]++;
                printf(",%d",it);
                if (v==V_COLD) copyv(x0,u);      /* the cold solve is the kept field */
            }
            /* window maintenance, charged to the variants that use it */
            double t0=wall(); rec_update(&Wm[sy],u,w,Aw); tim[sy][V_FISCH]+=wall()-t0;
            if (nsolve[sy] % wstride == 0){
                t0=wall(); rec_update(&Wsp[sy],u,w,Aw);
                double dtu=wall()-t0; tim[sy][V_SPACED]+=dtu; tim[sy][V_SPACED_LAST]+=dtu;
            }
            /* A*prev is needed by V_SPACED_LAST; charge that one matvec to it */
            t0=wall(); copyv(u,w);
            if (sy==2) rmmean(w); if (sy==3) zeroD(w);
            applyA(sy,w,prevA[sy]); tim[sy][V_SPACED_LAST]+=wall()-t0;
            copyv(u,prev[sy]);
            if (sy==2) copyv(u,ue);
            if (sy==3 && pf){
                fprintf(pf,"%d,%.4f",st,t);
                for (int e=0;e<NE;e++) fprintf(pf,",%.10e",u[epr[e]]);
                fprintf(pf,"\n");
            }
        }
        printf("\n");
    }
    if (pf) fclose(pf);

    printf("#\n# ===== stride=%d (Sys2/Sys3 every %g ms), pc=%s, tol=%.0e, %d steps of "
           "%g ms = %g ms =====\n",
           stride,stride*DT,
           PCTYPE==1?"sASM+IC(0)":(PCTYPE==2?"ASM-BASIC+IC(0)":"Jacobi"),
           TOL,nT,DT,nT*DT);
    if (IC_SHIFTS) printf("# (IC(0) diagonal shift used on %ld local rows)\n",IC_SHIFTS);
    long g_it[NV]={0,0,0,0}; double g_tm[NV]={0,0,0,0};
    for (int sy=1;sy<=3;sy++){
        printf("# --- Sys%d : %ld solves ---\n",sy,nsolve[sy]);
        for (int v=0;v<NV;v++){
            printf("#   %-16s %8ld iters  %7.2f /solve  (%+6.1f%%)  %8.3f s (%+6.1f%%)"
                   "  0-iter %ld/%ld\n",
                   VN[v],tot[sy][v],(double)tot[sy][v]/(nsolve[sy]?nsolve[sy]:1),
                   tot[sy][0]? -100.0*(1.0-(double)tot[sy][v]/tot[sy][0]) : 0.0,
                   tim[sy][v], tim[sy][0]? -100.0*(1.0-tim[sy][v]/tim[sy][0]) : 0.0,
                   zer[sy][v],nsolve[sy]);
            g_it[v]+=tot[sy][v]; g_tm[v]+=tim[sy][v];
        }
    }
    printf("# --- pipeline total (Sys1+Sys2+Sys3) ---\n");
    for (int v=0;v<NV;v++)
        printf("#   %-16s %8ld iters (%+6.1f%%)   %8.3f s (%+6.1f%%)\n",
               VN[v],g_it[v],-100.0*(1.0-(double)g_it[v]/g_it[0]),
               g_tm[v],-100.0*(1.0-g_tm[v]/g_tm[0]));
    printf("# --- Sys2+Sys3 only (the part the stride changes) ---\n");
    for (int v=0;v<NV;v++){
        long it = tot[2][v]+tot[3][v]; double tm = tim[2][v]+tim[3][v];
        long it0= tot[2][0]+tot[3][0]; double tm0= tim[2][0]+tim[3][0];
        printf("#   %-16s %8ld iters (%+6.1f%%)   %8.3f s (%+6.1f%%)\n",
               VN[v],it,-100.0*(1.0-(double)it/it0),tm,-100.0*(1.0-tm/tm0));
    }
    return 0;
}
