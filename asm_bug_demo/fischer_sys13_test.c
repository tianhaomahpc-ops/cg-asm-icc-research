/* fischer_sys13_test.c -- does the sliding-window fix carry over to Sys1 and Sys3?
 *
 * fischer_vm_test.c settled Sys2 (singular u_e recovery).  forward_ecg.cpp uses
 * the SAME recycler shape for Sys3 (`-fischer3`, f3_P/f3_AP, window 12, and the
 * same f3_P.erase(f3_P.begin()) at :2705), and no recycling at all for Sys1.
 * The open questions:
 *
 *   A  Sys3's reported ceiling ("initial guesses cap at ~10%": warm -9%,
 *      Fischer -11%) was measured with the broken eviction.  What is the ceiling
 *      once the window really is the last m solutions?
 *   B  Sys3's interface data is high rank (the -leadvol run spanned ~54 dims in
 *      80 steps).  Is the window SIZE the real knob there?  -> m = 16 vs 64.
 *   C  Sys1 is mass-dominated (~15 iterations/step).  Is there anything to win,
 *      given that maintaining a window costs ~1 matvec + 2 reductions/step?
 *
 * All three systems live on the same n^3 grid and are driven by the same
 * prescribed Vm(t), so the comparison is apples to apples:
 *
 *   Sys1  A1 = c*I + 0.5*K_sigma          SPD, mass-dominated (c = chi*Cm/dt).
 *         RHS manufactured so the exact solution is Vm(t) itself.
 *   Sys2  A2 = K_{si+se}                  singular pure-Neumann, b = -K_si Vm.
 *   Sys3  A3 = K_iso restricted to the interior, Dirichlet on the face i=0
 *         ("the interface"), Neumann elsewhere => non-singular, ill-conditioned.
 *         RHS = the lifting of g = trace of the Sys2 solution on that face,
 *         i.e. exactly the one-way heart -> torso coupling.
 *
 * Per system, five initial guesses: cold / warm / current-implementation
 * recycler (A-orthonormal basis, evict the oldest VECTOR) / true sliding window
 * (raw snapshots + Gram solve) at the same m / true sliding window at 4m.
 * Every recycler grows from the same clean cold solution, as in forward_ecg.cpp.
 * Also reports the numerical rank of each system's solution trajectory, which
 * is what decides whether a window of size m can ever cover it.
 *
 * Build: cc -O2 -o fischer_sys13_test fischer_sys13_test.c -lm
 * Run:   ./fischer_sys13_test [n] [steps] [window] [regime] [-log10 gram cut]
 *                             [slice-dump file] [sweep?] > out.csv
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
static double C1 = 0.5;    /* chi*Cm/dt for Sys1: tuned so cold ~ 15 iterations,
                            * matching the real run's Sys1(CG+bj-ICC)=15/step */
/* Truncation cutoff for the snapshot Gram.  The snapshots are solutions computed
 * to 1e-8||b||, so Gram eigenvalues below ~1e-8*lambda_max carry no information;
 * a cutoff far below that (1e-12) lets the pseudo-inverse amplify solve noise,
 * which is what makes a LARGER window look worse.  Swept by argv[5]. */
static double GCUT = 1e-8;

static char   *isD;                                /* Dirichlet face mask (Sys3) */
static double *dg1, *dg2, *dg3;                    /* Jacobi diagonals           */

/* generic anisotropic 7-point graph Laplacian; skips Dirichlet neighbours when
 * dirich!=0 (their value is carried by the right-hand side instead) */
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

static void build_diags(void)
{
    dg1=malloc(ND*sizeof(double)); dg2=malloc(ND*sizeof(double)); dg3=malloc(ND*sizeof(double));
    for (int i=0;i<N;i++) for (int j=0;j<N;j++) for (int k=0;k<N;k++){
        int p=IDX(i,j,k); double dm=0,de=0,dt=0;
        int has[6]={i>0,i<N-1,j>0,j<N-1,k>0,k<N-1};
        int nbp[6]={p-N*N,p+N*N,p-N,p+N,p-1,p+1};
        int ax [6]={0,0,1,1,2,2};
        (void)nbp;
        for (int q=0;q<6;q++) if (has[q]){
            dm += SM[ax[q]]; de += SIE[ax[q]];
            dt += ST[ax[q]];      /* Dirichlet neighbours still sit on the diagonal;
                                   * their value is carried by the right-hand side */
        }
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

/* CG, Jacobi PC, unpreconditioned residual test done BEFORE the first
 * iteration (a good enough guess costs 0 iterations).  sys==2 is singular, so
 * the constant mode is projected out; sys==3 lives on the interior only. */
static void coarse_geo_apply(int sys,const double*r,double*z);   /* defined below */
static int   USE_GEO = 0;                 /* add the geometric coarse correction */
static void (*SNAP_APPLY)(const double*,double*) = NULL;  /* snapshot coarse corr. */

static int cg(int sys, const double *b, double *x, double atol, int maxit)
{
    static double *r=NULL,*z=NULL,*p=NULL,*Ap=NULL;
    if (!r){ r=malloc(ND*sizeof(double)); z=malloc(ND*sizeof(double));
             p=malloc(ND*sizeof(double)); Ap=malloc(ND*sizeof(double)); }
    const double *dg = diagof(sys);
    applyA(sys,x,Ap);
    for (int i=0;i<ND;i++) r[i]=b[i]-Ap[i];
    if (sys==3) zeroD(r);
    if (nrm2(r) <= atol) return 0;
    for (int i=0;i<ND;i++) z[i]=r[i]/dg[i];
    if (USE_GEO)    coarse_geo_apply(sys,r,z);
    if (SNAP_APPLY) SNAP_APPLY(r,z);
    if (sys==2) rmmean(z);
    if (sys==3) zeroD(z);
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
        for (int i=0;i<ND;i++) z[i]=r[i]/dg[i];
        if (USE_GEO)    coarse_geo_apply(sys,r,z);
        if (SNAP_APPLY) SNAP_APPLY(r,z);
        if (sys==2) rmmean(z);
        if (sys==3) zeroD(z);
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

/* ------------------------------------------- two-level additive coarse space
 * z = D^-1 r + Z E^+ (Z^T r),  E = Z^T A Z  -- the same additive two-level form
 * as TwoLevelNicolaides in forward_ecg.cpp.  Two kinds of Z:
 *   GEO   subdomain indicator functions (Nicolaides).  LOCAL SUPPORT, so the
 *         apply is a segmented sum + a scatter: O(N), independent of k.
 *   SNAP  the recycled window's raw snapshots.  GLOBAL dense vectors, so the
 *         apply costs k dots + k axpys = O(kN) EVERY iteration -- the cost
 *         asymmetry that iteration counts alone hide.
 * E for GEO is built once per system (k matvecs); E for SNAP is the window's
 * Gram, already maintained incrementally. */
static int   NSUB = 0;                 /* number of subdomains = k for GEO */
static int  *sublab = NULL;            /* subdomain label per dof */
static double Egeo[4][MMAX][MMAX];
static double Epinv[4][MMAX][MMAX];        /* E^+ factored ONCE, not per iteration */

/* Pi = G^+ (truncated symmetric pseudo-inverse), so that an apply is a k x k
 * matvec.  Doing the eigen-decomposition inside the apply, as a first draft did,
 * costs O(k^3) per CG iteration and is not what a real code does. */
static void pinv_of(double G[MMAX][MMAX], double Pi[MMAX][MMAX], int m, double cut)
{
    static double A[MMAX][MMAX], V[MMAX][MMAX]; double ev[MMAX];
    for (int i=0;i<m;i++) for (int j=0;j<m;j++){ A[i][j]=G[i][j]; Pi[i][j]=0.0; }
    sym_eig(A,V,ev,m);
    double lmax=0; for (int i=0;i<m;i++) if (ev[i]>lmax) lmax=ev[i];
    for (int e=0;e<m;e++){
        if (ev[e] <= cut*lmax || ev[e] <= 0) continue;
        double inv=1.0/ev[e];
        for (int i=0;i<m;i++) for (int j=0;j<m;j++) Pi[i][j] += inv*V[i][e]*V[j][e];
    }
}

static void coarse_geo_build(int sys, double *z, double *az)
{
    static double *AZ = NULL;
    if (!AZ) AZ = malloc((size_t)MMAX*ND*sizeof(double));
    for (int j=0;j<NSUB;j++){
        for (int p=0;p<ND;p++) z[p] = (sublab[p]==j)? 1.0 : 0.0;
        if (sys==3) zeroD(z);
        applyA(sys,z,az);
        copyv(az, AZ+(size_t)j*ND);
    }
    for (int i=0;i<NSUB;i++) for (int j=i;j<NSUB;j++){
        double e=0;
        for (int p=0;p<ND;p++)
            if (sublab[p]==i && !(sys==3 && isD[p])) e += AZ[(size_t)j*ND+p];
        Egeo[sys][i][j]=e; Egeo[sys][j][i]=e;
    }
    pinv_of(Egeo[sys], Epinv[sys], NSUB, 1e-12);
}
static void coarse_geo_apply(int sys,const double*r,double*z)
{
    double f[MMAX], c[MMAX];
    for (int i=0;i<NSUB;i++) f[i]=0.0;
    for (int p=0;p<ND;p++) f[sublab[p]] += r[p];               /* Z^T r : O(N)  */
    for (int i=0;i<NSUB;i++){ double t=0;                      /* c = E^+ f     */
        for (int j=0;j<NSUB;j++) t += Epinv[sys][i][j]*f[j]; c[i]=t; }
    for (int p=0;p<ND;p++) z[p] += c[sublab[p]];               /* Z c   : O(N)  */
    if (sys==3) zeroD(z);
}

/* ------------------------------------------------------------- recyclers ---
 * ORTHO = what forward_ecg.cpp does today (A-orthonormal basis, relative
 *         acceptance threshold as in the Sys3 block, evict the oldest VECTOR)
 * RAW   = the true sliding window (last m solutions, A-optimal coefficients) */
enum { ORTHO=0, RAW=1 };
typedef struct {
    int sys, kind, mmax, m;
    double *P, *AP;      /* ORTHO: A-orthonormal basis; RAW: raw snapshots */
    double *G;           /* RAW: m x m Gram, maintained incrementally as in the
                          * patch (one new row per step instead of m^2 dots) */
    long accepted, rejected, evicted;
} Rec;

static void rec_init(Rec*R,int sys,int kind,int mmax)
{
    R->sys=sys; R->kind=kind; R->mmax=mmax; R->m=0;
    R->P =calloc((size_t)mmax*ND,sizeof(double));
    R->AP=calloc((size_t)mmax*ND,sizeof(double));
    R->G =calloc((size_t)mmax*mmax,sizeof(double));
    R->accepted=R->rejected=R->evicted=0;
}
static double last_cancel = 1.0;      /* sum_k |c_k| ||u^k|| / ||x0||, RAW only */
static void rec_guess(const Rec*R,const double*b,double*x0)
{
    last_cancel = 1.0;
    zerov(x0); if (!R->m) return;
    if (R->kind==ORTHO){
        for (int i=0;i<R->m;i++) axpy(dot(R->P+(size_t)i*ND,b), R->P+(size_t)i*ND, x0);
    } else {
        double f[MMAX], c[MMAX];
        for (int i=0;i<R->m;i++){
            f[i]=dot(R->P+(size_t)i*ND,b);
            for (int j=0;j<R->m;j++) Gbuf[i][j]=R->G[(size_t)i*R->mmax+j];
        }
        gram_solve(Gbuf,f,c,R->m,GCUT);
        double terms=0;
        for (int i=0;i<R->m;i++){
            axpy(c[i],R->P+(size_t)i*ND,x0);
            terms += fabs(c[i])*nrm2(R->P+(size_t)i*ND);
        }
        double nx=nrm2(x0); last_cancel = nx>0? terms/nx : 1.0;
    }
}
static void rec_drop_oldest(Rec*R)
{
    for (int i=0;i<R->m-1;i++){
        copyv(R->P +(size_t)(i+1)*ND, R->P +(size_t)i*ND);
        copyv(R->AP+(size_t)(i+1)*ND, R->AP+(size_t)i*ND);
    }
    R->m--; R->evicted++;
}
static void rec_update(Rec*R,const double*u,double*w,double*Aw,double*tmp)
{
    copyv(u,w);
    if (R->sys==2) rmmean(w);
    if (R->sys==3) zeroD(w);
    applyA(R->sys,w,Aw);
    if (R->kind==RAW){
        int m = R->m;
        double gnew[MMAX+1];
        for (int i=0;i<m;i++) gnew[i]=dot(R->P+(size_t)i*ND,Aw);
        gnew[m]=dot(w,Aw);
        if (m>=R->mmax){                       /* drop the oldest SNAPSHOT */
            rec_drop_oldest(R);
            for (int i=0;i<m;i++) gnew[i]=gnew[i+1];
            for (int i=1;i<m;i++) for (int j=1;j<m;j++)
                R->G[(size_t)(i-1)*R->mmax+(j-1)] = R->G[(size_t)i*R->mmax+j];
            m--;
        }
        for (int i=0;i<m;i++){ R->G[(size_t)i*R->mmax+m]=gnew[i];
                               R->G[(size_t)m*R->mmax+i]=gnew[i]; }
        R->G[(size_t)m*R->mmax+m]=gnew[m];
        copyv(w, R->P+(size_t)m*ND); copyv(Aw,R->AP+(size_t)m*ND);
        R->m=m+1; R->accepted++;
        return;
    }
    double q0 = dot(w,Aw); double ref = q0>0?sqrt(q0):0.0;     /* ||u||_A */
    for (int pass=0; pass<2 && R->m>0; ++pass)
        for (int i=0;i<R->m;i++){
            double c = dot(R->AP+(size_t)i*ND,w);
            axpy(-c, R->P +(size_t)i*ND, w);
            axpy(-c, R->AP+(size_t)i*ND, Aw);
        }
    double q = dot(w,Aw), nrm = q>0?sqrt(q):0.0;
    if (nrm <= 1e-3*ref){ R->rejected++; return; }             /* as in :2704 */
    scal(1.0/nrm,w); scal(1.0/nrm,Aw);
    if (R->m >= R->mmax) rec_drop_oldest(R);                   /* as in :2705 */
    copyv(w, R->P+(size_t)R->m*ND); copyv(Aw,R->AP+(size_t)R->m*ND);
    R->m++; R->accepted++;
    (void)tmp;
}

/* snapshot coarse correction: z += Z E^+ (Z^T r), Z = the window's snapshots,
 * E = its Gram (already maintained).  Cost: k dots + k axpys per iteration. */
static const Rec *SNAP_REC = NULL;
static double Spinv[MMAX][MMAX]; static int Spinv_m = 0;
static void snap_prepare(const Rec*R)   /* once per step, NOT once per iteration */
{
    static double G[MMAX][MMAX];
    SNAP_REC = R; Spinv_m = R? R->m : 0;
    if (!Spinv_m) return;
    for (int i=0;i<R->m;i++) for (int j=0;j<R->m;j++) G[i][j]=R->G[(size_t)i*R->mmax+j];
    pinv_of(G, Spinv, R->m, GCUT);
}
static void coarse_snap_apply(const double*r,double*z)
{
    const Rec*R = SNAP_REC;
    if (!R || !Spinv_m) return;
    double f[MMAX], c[MMAX];
    for (int i=0;i<R->m;i++) f[i]=dot(R->P+(size_t)i*ND, r);    /* k dots  : O(kN) */
    for (int i=0;i<R->m;i++){ double t=0;
        for (int j=0;j<R->m;j++) t += Spinv[i][j]*f[j]; c[i]=t; }
    for (int i=0;i<R->m;i++) axpy(c[i], R->P+(size_t)i*ND, z);  /* k axpys : O(kN) */
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

/* numerical rank of a set of snapshots (Gram eigenvalues, relative cutoffs) */
static void traj_rank(const double *S,int k,const char*label)
{
    if (k<2) return;
    for (int i=0;i<k;i++) for (int j=0;j<k;j++){
        double s=0; const double*a=S+(size_t)i*ND,*b=S+(size_t)j*ND;
        for (int q=0;q<ND;q++) s+=a[q]*b[q];
        Abuf[i][j]=s;
    }
    double ev[MMAX]; sym_eig(Abuf,Vbuf,ev,k);
    double lmax=0; for (int i=0;i<k;i++) if (ev[i]>lmax) lmax=ev[i];
    int r4=0,r6=0,r8=0;
    for (int i=0;i<k;i++){
        if (ev[i] > 1e-4 *lmax) r4++;
        if (ev[i] > 1e-6 *lmax) r6++;
        if (ev[i] > 1e-8 *lmax) r8++;
    }
    printf("# rank %-22s over %d snapshots: %d (1e-4)  %d (1e-6)  %d (1e-8)\n",
           label,k,r4,r6,r8);
    /* normalised Gram spectrum, descending -- the plot script reads this line */
    { double *ev2=malloc(k*sizeof(double));
      for (int i=0;i<k;i++) ev2[i]=ev[i];
      for (int i=0;i<k;i++) for (int j=i+1;j<k;j++)
          if (ev2[j]>ev2[i]){ double t=ev2[i]; ev2[i]=ev2[j]; ev2[j]=t; }
      printf("# spec %s :",label);
      for (int i=0;i<k;i++) printf(" %.4e", lmax>0? ev2[i]/lmax : 0.0);
      printf("\n"); free(ev2); }
}

/* ===================================================================== main */
int main(int argc,char**argv)
{
    N        = (argc>1)? atoi(argv[1]) : 24;
    int nT   = (argc>2)? atoi(argv[2]) : 350;
    int win  = (argc>3)? atoi(argv[3]) : 16;
    REGIME   = (argc>4)? atoi(argv[4]) : 0;
    if (argc>5) GCUT = pow(10.0, -atof(argv[5]));   /* argv[5] = -log10(cutoff) */
    ND = N*N*N;
    const int WBIG = (win*4 <= MMAX)? win*4 : MMAX;

    isD = calloc(ND,1);
    for (int j=0;j<N;j++) for (int k=0;k<N;k++) isD[IDX(0,j,k)] = 1;   /* face i=0 = interface */
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

    /* ---------------- deflation comparison (argv[7]==2): how much room is left
     * once the window has taken the easy steps?  Six variants per system:
     *   cold / window / geo-deflation / geo+window / snap+window / geo+snap+window
     * "geo" = Nicolaides subdomain indicators (local support, O(N) apply);
     * "snap" = the window's own snapshots as the coarse space (O(kN) apply). */
    if (argc>7 && atoi(argv[7])==2){
        const int BEST[4] = {0,4,4,8};          /* per-system best m from the sweep */
        const int ns = 4; NSUB = ns*ns*ns;      /* 64 subdomains */
        sublab = malloc(ND*sizeof(int));
        for (int i=0;i<N;i++) for (int j=0;j<N;j++) for (int k=0;k<N;k++){
            int bi=i*ns/N, bj=j*ns/N, bk=k*ns/N;
            sublab[IDX(i,j,k)] = (bi*ns+bj)*ns+bk;
        }
        double *vm=malloc(ND*sizeof(double)), *b=malloc(ND*sizeof(double));
        double *u=malloc(ND*sizeof(double)), *x0=malloc(ND*sizeof(double));
        double *w=malloc(ND*sizeof(double)), *Aw=malloc(ND*sizeof(double));
        double *tmp=malloc(ND*sizeof(double));
        double *pv[4]; for (int q=1;q<=3;q++){ pv[q]=malloc(ND*sizeof(double)); zerov(pv[q]); }
        Rec W[4];
        double t_setup[4];
        for (int sy=1;sy<=3;sy++){
            rec_init(&W[sy],sy,RAW,BEST[sy]);
            double t0=wall(); coarse_geo_build(sy,w,Aw); t_setup[sy]=wall()-t0;
        }
        enum { NV = 6 };
        const char *vn[NV] = {"cold(不做)","窗口 only","几何粗空间 only",
                              "几何粗空间+窗口","快照粗空间+窗口","几何+快照+窗口"};
        long tot[4][NV]; double tim[4][NV]; long zer[4][NV];
        for (int sy=1;sy<=3;sy++) for (int v=0;v<NV;v++){ tot[sy][v]=0; tim[sy][v]=0; zer[sy][v]=0; }

        printf("# deflation comparison: n=%d nd=%d steps=%d regime=%d nsub=%d "
               "(m: Sys1 %d Sys2 %d Sys3 %d)\n",N,ND,nT,REGIME,NSUB,BEST[1],BEST[2],BEST[3]);
        printf("step,t");
        for (int sy=1;sy<=3;sy++) for (int v=0;v<NV;v++) printf(",s%d_v%d",sy,v);
        printf("\n");

        for (int st=0; st<nT; ++st){
            vm_field((double)st,vm);
            printf("%d,%d",st,st);
            for (int sy=1;sy<=3;sy++){
                if (sy==1){ applyA(1,vm,b); }
                else if (sy==2){ applyK(SI,vm,b,0); scal(-1.0,b); rmmean(b); }
                else { zerov(b);
                    for (int j=0;j<N;j++) for (int kk=0;kk<N;kk++)
                        b[IDX(1,j,kk)] += ST[0]*pv[2][IDX(0,j,kk)];
                    zeroD(b); }
                double nb=nrm2(b);
                if (nb < 1e-9){ for (int v=0;v<NV;v++) printf(",0"); continue; }
                double atol=1e-8*nb;
                int kk2; double t0;

                for (int v=0; v<NV; ++v){
                    USE_GEO = (v==2||v==3||v==5);
                    SNAP_APPLY = (v==4||v==5)? coarse_snap_apply : NULL;
                    if (SNAP_APPLY) snap_prepare(&W[sy]);
                    t0 = wall();
                    if (v==0||v==2) zerov(x0); else rec_guess(&W[sy],b,x0);
                    kk2 = cg(sy,b,x0,atol,4000);
                    tim[sy][v]+=wall()-t0;
                    tot[sy][v]+=kk2; if(!kk2) zer[sy][v]++;
                    printf(",%d",kk2);
                    if (v==0) copyv(x0,u);          /* the cold solve is the kept field */
                }
                USE_GEO=0; SNAP_APPLY=NULL;
                t0=wall(); rec_update(&W[sy],u,w,Aw,tmp);
                for (int v=1;v<NV;v++) if (v!=2) tim[sy][v]+=(wall()-t0)/(NV-2);
                copyv(u,pv[sy]);
            }
            printf("\n");
        }
        printf("#\n# ===== deflation comparison (regime %d) =====\n",REGIME);
        printf("# 几何粗空间一次性 setup: Sys1 %.3f s  Sys2 %.3f s  Sys3 %.3f s "
               "(%d 次 matvec,被所有步摊销)\n",t_setup[1],t_setup[2],t_setup[3],NSUB);
        for (int sy=1;sy<=3;sy++){
            printf("# --- Sys%d (m=%d) ---\n",sy,BEST[sy]);
            for (int v=0;v<NV;v++)
                printf("#   %-22s %8ld iters (%+6.1f%%)  %8.3f s (%+6.1f%%)  0-iter %ld\n",
                       vn[v],tot[sy][v],-100.0*(1.0-(double)tot[sy][v]/tot[sy][0]),
                       tim[sy][v],-100.0*(1.0-tim[sy][v]/tim[sy][0]),zer[sy][v]);
        }
        return 0;
    }

    /* ---------------- window-size sweep (argv[7]): what is the right m for
     * each system?  Same true sliding window everywhere, only m changes. */
    if (argc>7 && atoi(argv[7])){
        const int MS[] = {4,8,12,16,24,32,48,64};
        const int nM = (int)(sizeof(MS)/sizeof(MS[0]));
        double *vm=malloc(ND*sizeof(double)), *b=malloc(ND*sizeof(double));
        double *u=malloc(ND*sizeof(double)), *x0=malloc(ND*sizeof(double));
        double *w=malloc(ND*sizeof(double)), *Aw=malloc(ND*sizeof(double));
        double *tmp=malloc(ND*sizeof(double));
        double *pv[4]; for (int q=1;q<=3;q++){ pv[q]=malloc(ND*sizeof(double)); zerov(pv[q]); }
        Rec SW[4][8], CUR[4];            /* CUR = what forward_ecg.cpp does today */
        for (int sy=1;sy<=3;sy++){
            for (int q=0;q<nM;q++) rec_init(&SW[sy][q],sy,RAW,MS[q]);
            rec_init(&CUR[sy],sy,ORTHO, sy==3? 12 : 16);   /* F3MAX=12, FISCH_MAX=16 */
        }
        /* slots: 0..nM-1 = the windows, nM = cold, nM+1 = warm, nM+2 = current impl */
        enum { NSLOT = 8+3 };
        long tot[4][NSLOT], zer[4][NSLOT]; double tim[4][NSLOT];
        for (int sy=1;sy<=3;sy++) for (int q=0;q<NSLOT;q++){ tot[sy][q]=0; zer[sy][q]=0; tim[sy][q]=0; }
        double *pw=malloc(ND*sizeof(double));      /* warm-start seed per system */
        /* index nM = cold, nM+1 = warm */

        printf("# window sweep: n=%d nd=%d steps=%d regime=%d gcut=%.0e\n",N,ND,nT,REGIME,GCUT);
        printf("step,t");
        for (int sy=1;sy<=3;sy++){ printf(",s%d_cold",sy);
            for (int q=0;q<nM;q++) printf(",s%d_m%d",sy,MS[q]); }
        printf("\n");

        for (int st=0; st<nT; ++st){
            vm_field((double)st,vm);
            printf("%d,%d",st,st);
            for (int sy=1;sy<=3;sy++){
                if (sy==1){ applyA(1,vm,b); }
                else if (sy==2){ applyK(SI,vm,b,0); scal(-1.0,b); rmmean(b); }
                else { zerov(b);
                    for (int j=0;j<N;j++) for (int kk=0;kk<N;kk++)
                        b[IDX(1,j,kk)] += ST[0]*pv[2][IDX(0,j,kk)];
                    zeroD(b); }
                double nb=nrm2(b);
                if (nb < 1e-9){ printf(",0"); for (int q=0;q<nM;q++) printf(",0"); continue; }
                double atol=1e-8*nb;
                double t0=wall();
                zerov(u); int kc=cg(sy,b,u,atol,4000);
                tim[sy][nM]+=wall()-t0;
                tot[sy][nM]+=kc; if(!kc) zer[sy][nM]++;
                printf(",%d",kc);
                t0=wall();
                copyv(pv[sy],pw); int kw=cg(sy,b,pw,atol,4000);
                tim[sy][nM+1]+=wall()-t0;
                tot[sy][nM+1]+=kw; if(!kw) zer[sy][nM+1]++;
                t0=wall();
                rec_guess(&CUR[sy],b,x0);
                int kcur=cg(sy,b,x0,atol,4000);
                tim[sy][nM+2]+=wall()-t0;
                tot[sy][nM+2]+=kcur; if(!kcur) zer[sy][nM+2]++;
                for (int q=0;q<nM;q++){
                    /* time the WHOLE recycler: guess + solve + window update */
                    t0=wall();
                    rec_guess(&SW[sy][q],b,x0);
                    int kk2=cg(sy,b,x0,atol,4000);
                    tim[sy][q]+=wall()-t0;
                    tot[sy][q]+=kk2; if(!kk2) zer[sy][q]++;
                    printf(",%d",kk2);
                }
                for (int q=0;q<nM;q++){
                    t0=wall(); rec_update(&SW[sy][q],u,w,Aw,tmp); tim[sy][q]+=wall()-t0;
                }
                t0=wall(); rec_update(&CUR[sy],u,w,Aw,tmp); tim[sy][nM+2]+=wall()-t0;
                copyv(u,pv[sy]);
            }
            printf("\n");
        }
        printf("#\n# ===== window sweep summary (regime %d) =====\n",REGIME);
        printf("# times are SERIAL wall clock of the local work only (no MPI):\n"
               "#   window column = guess + solve + window update, i.e. everything\n");
        for (int sy=1;sy<=3;sy++){
            printf("# Sys%d  cold %8ld iters  %8.3f s\n",sy,tot[sy][nM],tim[sy][nM]);
            printf("#    current %8ld iters  %8.3f s  (%+.1f%% iters, %+.1f%% time)  [m=%d, evict oldest vector]\n",
                   tot[sy][nM+2],tim[sy][nM+2],
                   -100.0*(1.0-(double)tot[sy][nM+2]/tot[sy][nM]),
                   -100.0*(1.0-tim[sy][nM+2]/tim[sy][nM]), CUR[sy].mmax);
            printf("#       warm %8ld iters  %8.3f s  (%+.1f%% iters, %+.1f%% time)\n",
                   tot[sy][nM+1],tim[sy][nM+1],
                   -100.0*(1.0-(double)tot[sy][nM+1]/tot[sy][nM]),
                   -100.0*(1.0-tim[sy][nM+1]/tim[sy][nM]));
            for (int q=0;q<nM;q++)
                printf("#   m=%-3d %8ld iters  %8.3f s  (%+.1f%% iters, %+.1f%% time)"
                       "  0-iter %ld\n",MS[q],tot[sy][q],tim[sy][q],
                       -100.0*(1.0-(double)tot[sy][q]/tot[sy][nM]),
                       -100.0*(1.0-tim[sy][q]/tim[sy][nM]),zer[sy][q]);
        }
        return 0;
    }

    double *vm=malloc(ND*sizeof(double)), *b=malloc(ND*sizeof(double));
    double *u =malloc(ND*sizeof(double)), *x0=malloc(ND*sizeof(double));
    double *w =malloc(ND*sizeof(double)), *Aw=malloc(ND*sizeof(double));
    double *tmp=malloc(ND*sizeof(double));
    double *prev[4]; for (int s=1;s<=3;s++){ prev[s]=malloc(ND*sizeof(double)); zerov(prev[s]); }

    Rec R[4][3];                       /* [sys][0=current impl,1=SW m,2=SW 4m] */
    for (int s=1;s<=3;s++){
        rec_init(&R[s][0],s,ORTHO,win);
        rec_init(&R[s][1],s,RAW  ,win);
        rec_init(&R[s][2],s,RAW  ,WBIG);
    }
    long it[4][5]; long zr[4][5]; double eA[4][3], canc[4][3];
    for (int s=1;s<=3;s++){ for (int v=0;v<5;v++){ it[s][v]=0; zr[s][v]=0; }
                            for (int v=0;v<3;v++){ eA[s][v]=0; canc[s][v]=0; } }

    /* subsampled solution snapshots, for the trajectory-rank diagnostic */
    const int SUB = (nT/80 > 0)? nT/80 : 1;
    int nsnap=0; double *snap1=malloc((size_t)MMAX*ND*sizeof(double));
    double *snap2=malloc((size_t)MMAX*ND*sizeof(double));
    double *snap3=malloc((size_t)MMAX*ND*sizeof(double));

    /* optional: mid-plane slices of all three solutions, for the field figure */
    const char *dumpfile = (argc>6)? argv[6] : NULL;
    FILE *df = dumpfile? fopen(dumpfile,"w") : NULL;
    const int dsteps[] = {8,15,25,40,60,90,150,250};
    const int ndsteps = (int)(sizeof(dsteps)/sizeof(dsteps[0]));
    if (df) fprintf(df,"# n=%d regime=%d\n",N,REGIME);

    printf("# n=%d nd=%d steps=%d window=%d big=%d regime=%d C1=%.2f gcut=%.0e\n",
           N,ND,nT,win,WBIG,REGIME,C1,GCUT);
    printf("step,t,"
           "s1_cold,s1_warm,s1_cur,s1_sw,s1_swb,"
           "s2_cold,s2_warm,s2_cur,s2_sw,s2_swb,"
           "s3_cold,s3_warm,s3_cur,s3_sw,s3_swb\n");

    int nsteps=0;
    for (int st=0; st<nT; ++st){
        double t=st;
        vm_field(t,vm);
        int k[4][5];

        for (int s=1;s<=3;s++){
            /* ---- right-hand side of system s ---- */
            if (s==1){                       /* manufactured: exact solution = Vm(t) */
                applyA(1,vm,b);
            } else if (s==2){
                applyK(SI,vm,b,0); scal(-1.0,b); rmmean(b);
            } else {                         /* lifting of g = trace of u_e on i=0 */
                zerov(b);
                for (int j=0;j<N;j++) for (int kk=0;kk<N;kk++){
                    int pd=IDX(0,j,kk), pi=IDX(1,j,kk);
                    b[pi] += ST[0]*prev[2][pd];      /* prev[2] holds this step's u_e */
                }
                zeroD(b);
            }
            double nb=nrm2(b);
            if (nb < 1e-9){ for (int v=0;v<5;v++) k[s][v]=0; continue; }
            double atol=1e-8*nb;

            zerov(u); k[s][0]=cg(s,b,u,atol,4000);              /* cold = kept */
            copyv(prev[s],x0); k[s][1]=cg(s,b,x0,atol,4000);    /* warm        */
            for (int v=0;v<3;v++){
                rec_guess(&R[s][v],b,x0);
                if (v>=1){                       /* A-norm error of the guess */
                    for (int q=0;q<ND;q++) w[q]=u[q]-x0[q];
                    applyA(s,w,Aw); double qq=dot(w,Aw);
                    eA[s][v]  += qq>0? sqrt(qq):0.0;
                    canc[s][v]+= last_cancel;
                }
                k[s][2+v]=cg(s,b,x0,atol,4000);
            }
            for (int v=0;v<5;v++){ it[s][v]+=k[s][v]; if(!k[s][v]) zr[s][v]++; }
            for (int v=0;v<3;v++) rec_update(&R[s][v],u,w,Aw,tmp);
            copyv(u,prev[s]);
            /* Sys2's solution is Sys3's boundary data, so it must be solved first */
        }
        nsteps++;

        if (st % SUB == 0 && nsnap < MMAX){
            copyv(prev[1],snap1+(size_t)nsnap*ND);
            copyv(prev[2],snap2+(size_t)nsnap*ND);
            copyv(prev[3],snap3+(size_t)nsnap*ND);
            nsnap++;
        }
        if (df){
            int hit=0; for (int q=0;q<ndsteps;q++) if (dsteps[q]==st) hit=1;
            if (hit){
                const int kk=N/2;
                for (int sy=1;sy<=3;sy++){
                    fprintf(df,"SLICE %d %d\n",sy,st);
                    for (int i=0;i<N;i++){
                        for (int j=0;j<N;j++) fprintf(df,"%.6e ",prev[sy][IDX(i,j,kk)]);
                        fprintf(df,"\n");
                    }
                }
            }
        }
        printf("%d,%.0f,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",st,t,
               k[1][0],k[1][1],k[1][2],k[1][3],k[1][4],
               k[2][0],k[2][1],k[2][2],k[2][3],k[2][4],
               k[3][0],k[3][1],k[3][2],k[3][3],k[3][4]);
    }

    const char *vn[5]={"cold","warm","current impl (evict oldest vector)",
                       "true sliding window","true sliding window x4"};
    printf("#\n# ===== summary (%d steps, window %d / %d, regime %d) =====\n",
           nsteps,win,WBIG,REGIME);
    for (int s=1;s<=3;s++){
        printf("# --- Sys%d ---\n",s);
        for (int v=0;v<5;v++)
            printf("#   %-38s %8ld  %7.1f%%  0-iter %ld\n",vn[v],it[s][v],
                   100.0*(1.0-(double)it[s][v]/it[s][0]), zr[s][v]);
        printf("#   basis churn (current impl): acc=%ld rej=%ld evict=%ld\n",
               R[s][0].accepted,R[s][0].rejected,R[s][0].evicted);
        printf("#   mean ||x*-x0||_A : SW %.4e  SWx4 %.4e   |  cancellation "
               "sum|c_k|.||u^k||/||x0|| : SW %.1f  SWx4 %.1f\n",
               eA[s][1]/nsteps, eA[s][2]/nsteps, canc[s][1]/nsteps, canc[s][2]/nsteps);
    }
    traj_rank(snap1,nsnap,"Sys1 solution Vm");
    traj_rank(snap2,nsnap,"Sys2 solution u_e");
    traj_rank(snap3,nsnap,"Sys3 solution u_T");
    if (df) fclose(df);
    return 0;
}
