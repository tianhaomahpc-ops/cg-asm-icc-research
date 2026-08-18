/* fischer_vm_test.c -- controlled testbed for Sys2-style time recycling.
 *
 * WHY: forward_ecg.cpp (branch cardiac-sim-fakegeo) measures cold / warm /
 * Fischer / physics initial guesses for the singular Sys2 solve
 *     K_{si+se} u_e = -K_{si} Vm(t),      ker(K_{si+se}) = span{1}.
 * This file answers -- without MFEM/PETSc, in a system with the same structure
 * (singular pure-Neumann, unequal anisotropy, travelling front, multi-beat):
 *
 *   Q1  Does using the CURRENT-step Vm to choose the projection COEFFICIENTS
 *       beat the A-orthogonal projection?  Theory says no: on a fixed span the
 *       A-projection minimises ||x*-x0||_A.   -> SW-vmfit vs SW-Aopt, compared
 *       on the SAME window, both in iterations and in ||x*-x0||_A.
 *   Q2  Is the ABSOLUTE acceptance threshold (nrm>1e-12, forward_ecg.cpp:2597)
 *       polluting the basis, and does a RELATIVE threshold fix it?
 *   Q3  Window maintenance: is "evict the oldest ORTHONORMAL vector"
 *       (forward_ecg.cpp:2604) equivalent to a sliding window over the last m
 *       SOLUTIONS?  -> F-evict-oldest vs restart-when-full vs the true sliding
 *       window (Gram solve over raw snapshots) vs phase-diverse eviction.
 *   Q4  Is the Vm-side least-squares residual eta a usable ZERO-MATVEC
 *       predictor of "this guess lands inside tol" (0 CG iterations)?
 *   Q5  Does the current-step increment dVm = Vm^n - Vm^{n-1} buy anything?
 *       -> warm+delta (x0 = u^{n-1} + SGS(b^n-b^{n-1})) and SW+enrich.
 *
 * MODEL PROBLEM
 *   n^3 grid, 7-point anisotropic graph Laplacian, natural (Neumann) BCs =>
 *   symmetric positive SEMI-definite, ker = constants, exactly like Sys 2.
 *   The two conductivity tensors have DIFFERENT anisotropy ratios (unequal
 *   anisotropy), so u_e is not proportional to Vm -- that is what kills the
 *   "physics" guess in the real runs.
 *   Vm(x,t): TP06-shaped AP travelling from a corner, with a spatial APD
 *   gradient so repolarisation sweeps too (no perfectly static plateau).
 *   Beat period 350 ms, sampled every 1 ms like the real ECG loop.
 *
 * The KEPT solution is always the cold solve, exactly as in forward_ecg.cpp:
 * every recycler grows from that clean field, so no variant pollutes another.
 *
 * Build: cc -O2 -o fischer_vm_test fischer_vm_test.c -lm
 * Run:   ./fischer_vm_test [n] [beats] [window] > fischer_vm_test.csv
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ------------------------------------------------------------------ grid */
static int N, ND;
#define IDX(i,j,k) ((((i)*N)+(j))*N+(k))

/* fibre along x; the two tensors are NOT proportional (unequal anisotropy) */
static const double SIE[3] = {1.00, 0.35, 0.35};   /* sigma_i + sigma_e */
static const double SI [3] = {0.30, 0.03, 0.03};   /* sigma_i           */

static double *diagA;            /* diagonal of K_SIE (Jacobi PC / SGS) */

static void applyK(const double *sg, const double *x, double *y)
{
    for (int i=0;i<N;i++) for (int j=0;j<N;j++) for (int k=0;k<N;k++){
        int p = IDX(i,j,k); double xp = x[p], s = 0.0;
        if (i>0)   s += sg[0]*(xp - x[p-N*N]);
        if (i<N-1) s += sg[0]*(xp - x[p+N*N]);
        if (j>0)   s += sg[1]*(xp - x[p-N]);
        if (j<N-1) s += sg[1]*(xp - x[p+N]);
        if (k>0)   s += sg[2]*(xp - x[p-1]);
        if (k<N-1) s += sg[2]*(xp - x[p+1]);
        y[p] = s;
    }
}

static void build_diag(void)
{
    diagA = malloc(ND*sizeof(double));
    for (int i=0;i<N;i++) for (int j=0;j<N;j++) for (int k=0;k<N;k++){
        double d = 0.0;
        if (i>0)   d += SIE[0];
        if (i<N-1) d += SIE[0];
        if (j>0)   d += SIE[1];
        if (j<N-1) d += SIE[1];
        if (k>0)   d += SIE[2];
        if (k<N-1) d += SIE[2];
        diagA[IDX(i,j,k)] = d;
    }
}

/* one symmetric Gauss-Seidel sweep for K_SIE x = b (the cheap local smoother
 * standing in for one additive-Schwarz application) */
static void sgs_sweep(const double *b, double *x)
{
    for (int i=0;i<N;i++) for (int j=0;j<N;j++) for (int k=0;k<N;k++){
        int p=IDX(i,j,k); double s=b[p];
        if (i>0)   s += SIE[0]*x[p-N*N];
        if (i<N-1) s += SIE[0]*x[p+N*N];
        if (j>0)   s += SIE[1]*x[p-N];
        if (j<N-1) s += SIE[1]*x[p+N];
        if (k>0)   s += SIE[2]*x[p-1];
        if (k<N-1) s += SIE[2]*x[p+1];
        x[p] = s/diagA[p];
    }
    for (int i=N-1;i>=0;i--) for (int j=N-1;j>=0;j--) for (int k=N-1;k>=0;k--){
        int p=IDX(i,j,k); double s=b[p];
        if (i>0)   s += SIE[0]*x[p-N*N];
        if (i<N-1) s += SIE[0]*x[p+N*N];
        if (j>0)   s += SIE[1]*x[p-N];
        if (j<N-1) s += SIE[1]*x[p+N];
        if (k>0)   s += SIE[2]*x[p-1];
        if (k<N-1) s += SIE[2]*x[p+1];
        x[p] = s/diagA[p];
    }
}

/* ------------------------------------------------------------- vector ops */
static double dot(const double*a,const double*b){ double s=0; for(int i=0;i<ND;i++) s+=a[i]*b[i]; return s; }
static double nrm2(const double*a){ return sqrt(dot(a,a)); }
static void   axpy(double a,const double*x,double*y){ for(int i=0;i<ND;i++) y[i]+=a*x[i]; }
static void   scal(double a,double*x){ for(int i=0;i<ND;i++) x[i]*=a; }
static void   copyv(const double*x,double*y){ memcpy(y,x,ND*sizeof(double)); }
static void   zerov(double*x){ memset(x,0,ND*sizeof(double)); }
static void   rmmean(double*x){ double s=0; for(int i=0;i<ND;i++) s+=x[i]; s/=ND; for(int i=0;i<ND;i++) x[i]-=s; }
static double anorm(const double*x,double*t){ applyK(SIE,x,t); double q=dot(x,t); return q>0?sqrt(q):0.0; }

/* ------------------------------------------------- CG (singular, Jacobi PC)
 * Stops on the UNPRECONDITIONED residual, tested BEFORE the first iteration,
 * so a guess already inside tol costs 0 iterations -- what forward_ecg.cpp
 * arranges with KSP_NORM_UNPRECONDITIONED (:765).  The constant nullspace is
 * projected out of the preconditioned direction (PETSc MatSetNullSpace). */
static int cg(const double *b, double *x, double atol, int maxit)
{
    static double *r=NULL,*z=NULL,*p=NULL,*Ap=NULL;
    if (!r){ r=malloc(ND*sizeof(double)); z=malloc(ND*sizeof(double));
             p=malloc(ND*sizeof(double)); Ap=malloc(ND*sizeof(double)); }
    applyK(SIE,x,Ap);
    for (int i=0;i<ND;i++) r[i]=b[i]-Ap[i];
    if (nrm2(r) <= atol) return 0;
    for (int i=0;i<ND;i++) z[i]=r[i]/diagA[i];
    rmmean(z); copyv(z,p);
    double rz = dot(r,z);
    int it;
    for (it=1; it<=maxit; ++it){
        applyK(SIE,p,Ap);
        double pAp = dot(p,Ap);
        if (pAp <= 0) break;
        double a = rz/pAp;
        axpy(a,p,x); axpy(-a,Ap,r);
        if (nrm2(r) <= atol) break;
        for (int i=0;i<ND;i++) z[i]=r[i]/diagA[i];
        rmmean(z);
        double rz2 = dot(r,z), beta = rz2/rz; rz = rz2;
        for (int i=0;i<ND;i++) p[i]=z[i]+beta*p[i];
    }
    return it;
}

/* ---------------------------------------- small dense symmetric pseudo-solve
 * Jacobi eigenvalue method; solves G c = f dropping eigenvalues below
 * cut*lambda_max (snapshot Gram matrices are numerically singular). */
#define MMAX 96
static void sym_pinv_solve(double G[MMAX][MMAX],const double*f,double*c,int m,double cut)
{
    static double V[MMAX][MMAX], A[MMAX][MMAX];
    for (int i=0;i<m;i++) for (int j=0;j<m;j++){ A[i][j]=G[i][j]; V[i][j]=(i==j); }
    for (int sweep=0; sweep<60; ++sweep){
        double off=0;
        for (int i=0;i<m;i++) for (int j=i+1;j<m;j++) off += A[i][j]*A[i][j];
        if (off < 1e-30) break;
        for (int p=0;p<m;p++) for (int q=p+1;q<m;q++){
            if (fabs(A[p][q]) < 1e-300) continue;
            double th = 0.5*(A[q][q]-A[p][p])/A[p][q];
            double t  = (th>=0? 1.0:-1.0)/(fabs(th)+sqrt(th*th+1.0));
            double cs = 1.0/sqrt(t*t+1.0), sn = t*cs;
            for (int k=0;k<m;k++){ double akp=A[k][p],akq=A[k][q];
                A[k][p]=cs*akp-sn*akq; A[k][q]=sn*akp+cs*akq; }
            for (int k=0;k<m;k++){ double apk=A[p][k],aqk=A[q][k];
                A[p][k]=cs*apk-sn*aqk; A[q][k]=sn*apk+cs*aqk; }
            for (int k=0;k<m;k++){ double vkp=V[k][p],vkq=V[k][q];
                V[k][p]=cs*vkp-sn*vkq; V[k][q]=sn*vkp+cs*vkq; }
        }
    }
    double lmax=0; for (int i=0;i<m;i++) if (A[i][i]>lmax) lmax=A[i][i];
    for (int i=0;i<m;i++) c[i]=0.0;
    for (int e=0;e<m;e++){
        double lam=A[e][e];
        if (lam <= cut*lmax || lam<=0) continue;
        double vf=0; for (int i=0;i<m;i++) vf += V[i][e]*f[i];
        double s = vf/lam;
        for (int i=0;i<m;i++) c[i] += s*V[i][e];
    }
}

/* --------------------------------------------------------- Fischer recycler
 * P: A-orthonormal history, AP = K_SIE*P, KEY = the Vm snapshot that produced
 * each vector (only used by the phase-diverse eviction rule). */
enum { EV_OLDEST=0, EV_RESTART=1, EV_DIVERSE=2 };
typedef struct {
    const char *name;
    int     mmax, m, relthresh, evict;
    double  tau;
    double *P, *AP, *KEY;
    long    accepted, rejected, evicted, restarts;
} Fisch;

static void fisch_init(Fisch*F,const char*nm,int mmax,int rel,double tau,int ev)
{
    F->name=nm; F->mmax=mmax; F->m=0; F->relthresh=rel; F->tau=tau; F->evict=ev;
    F->P  = calloc((size_t)mmax*ND,sizeof(double));
    F->AP = calloc((size_t)mmax*ND,sizeof(double));
    F->KEY= calloc((size_t)mmax*ND,sizeof(double));
    F->accepted=F->rejected=F->evicted=F->restarts=0;
}
/* x0 = sum_i <p_i,b> p_i : the A-orthogonal projection of the exact solution */
static void fisch_guess(const Fisch*F,const double*b,double*x0)
{
    zerov(x0);
    for (int i=0;i<F->m;i++) axpy(dot(F->P+(size_t)i*ND,b), F->P+(size_t)i*ND, x0);
}
static void fisch_drop(Fisch*F,int idx)
{
    for (int i=idx;i<F->m-1;i++){
        copyv(F->P  +(size_t)(i+1)*ND, F->P  +(size_t)i*ND);
        copyv(F->AP +(size_t)(i+1)*ND, F->AP +(size_t)i*ND);
        copyv(F->KEY+(size_t)(i+1)*ND, F->KEY+(size_t)i*ND);
    }
    F->m--; F->evicted++;
}
static double corr(const double*a,const double*b)
{
    double na=nrm2(a), nb=nrm2(b);
    return (na<1e-300||nb<1e-300) ? 0.0 : dot(a,b)/(na*nb);
}
/* grow the basis from the CLEAN cold solution u (mean removed inside) */
static void fisch_update(Fisch*F,const double*u,const double*vmkey,
                         double*w,double*Aw,double*tmp)
{
    copyv(u,w); rmmean(w);
    double ref = anorm(w,tmp);                 /* ||u_e||_A: the scale */
    applyK(SIE,w,Aw);
    for (int pass=0; pass<2 && F->m>0; ++pass) /* CGS2, as in forward_ecg.cpp */
        for (int i=0;i<F->m;i++){
            double c = dot(F->AP+(size_t)i*ND,w);
            axpy(-c, F->P +(size_t)i*ND, w);
            axpy(-c, F->AP+(size_t)i*ND, Aw);
        }
    double q = dot(w,Aw), nrm = q>0?sqrt(q):0.0;
    double thresh = F->relthresh ? F->tau*ref : 1e-12;
    if (nrm <= thresh){ F->rejected++; return; }
    scal(1.0/nrm,w); scal(1.0/nrm,Aw);
    if (F->m >= F->mmax){
        if (F->evict==EV_RESTART){ F->m=0; F->restarts++; }
        else if (F->evict==EV_DIVERSE){
            /* drop the most redundant phase: the vector whose Vm key is most
             * similar to some other kept key (keeps the window phase-diverse) */
            int victim=0; double worst=-2;
            for (int i=0;i<F->m;i++){
                double best=-2;
                for (int j=0;j<F->m;j++) if (j!=i){
                    double c=fabs(corr(F->KEY+(size_t)i*ND,F->KEY+(size_t)j*ND));
                    if (c>best) best=c;
                }
                if (best>worst){ worst=best; victim=i; }
            }
            fisch_drop(F,victim);
        } else fisch_drop(F,0);            /* forward_ecg.cpp: erase(begin()) */
    }
    copyv(w,  F->P  +(size_t)F->m*ND);
    copyv(Aw, F->AP +(size_t)F->m*ND);
    copyv(vmkey, F->KEY+(size_t)F->m*ND);
    F->m++; F->accepted++;
}
/* relative A-error of projecting v onto the basis: 0 means "v is in the span".
 * Applied to u_e^{n-1} it asks: does this window really contain the last m
 * solutions?  A true sliding window answers 0; dropping the oldest ORTHONORMAL
 * vector does not. */
static double fisch_spanerr(const Fisch*F,const double*v,double*x,double*t)
{
    zerov(x);
    for (int i=0;i<F->m;i++){
        double c=0; for(int q=0;q<ND;q++) c+=F->AP[(size_t)i*ND+q]*v[q];
        axpy(c,F->P+(size_t)i*ND,x);
    }
    double nv=anorm(v,t);
    for (int i=0;i<ND;i++) x[i]=v[i]-x[i];
    double ne=anorm(x,t);
    return nv>0? ne/nv : 0.0;
}
static double fisch_orthloss(const Fisch*F)
{
    double worst=0;
    for (int i=0;i<F->m;i++) for (int j=0;j<F->m;j++) if (i!=j){
        double c=fabs(dot(F->P+(size_t)i*ND,F->AP+(size_t)j*ND));
        if (c>worst) worst=c;
    }
    return worst;
}

/* ------------------------------------------------------- raw-window recycler
 * The TRUE sliding window: keep the last m raw pairs (Vm^k, u_e^k) and pick
 * coefficients on that span.  Two rules, same span:
 *   Aopt  : G c = f, G_kl=<u^k,A u^l>, f_k=<u^k,b>   (A-optimal == Fischer)
 *   vmfit : H c = g, H_kl=<V^k,V^l>,   g_k=<V^k,Vm>  ("use the latest Vm")
 * vmfit also yields eta = ||Vm - sum c_k V^k|| / ||Vm||, a zero-matvec
 * predictor of whether the history covers the current step. */
typedef struct { int mmax,m,evict; double *U,*AU,*V; } Raw;

static void raw_init(Raw*R,int mmax,int evict)
{
    R->mmax=mmax; R->m=0; R->evict=evict;
    R->U =calloc((size_t)mmax*ND,sizeof(double));
    R->AU=calloc((size_t)mmax*ND,sizeof(double));
    R->V =calloc((size_t)mmax*ND,sizeof(double));
}
static void raw_drop(Raw*R,int idx)
{
    for (int i=idx;i<R->m-1;i++){
        copyv(R->U +(size_t)(i+1)*ND, R->U +(size_t)i*ND);
        copyv(R->AU+(size_t)(i+1)*ND, R->AU+(size_t)i*ND);
        copyv(R->V +(size_t)(i+1)*ND, R->V +(size_t)i*ND);
    }
    R->m--;
}
static void raw_push(Raw*R,const double*u,const double*v)
{
    if (R->m==R->mmax && R->evict==EV_DIVERSE){
        /* keep the window phase-diverse: drop the snapshot whose Vm is most
         * redundant with another kept snapshot, not the oldest one */
        int victim=0; double worst=-2;
        for (int i=0;i<R->m;i++){
            double best=-2;
            for (int j=0;j<R->m;j++) if (j!=i){
                double c=fabs(corr(R->V+(size_t)i*ND,R->V+(size_t)j*ND));
                if (c>best) best=c;
            }
            if (best>worst){ worst=best; victim=i; }
        }
        raw_drop(R,victim);
    }
    if (R->m==R->mmax){
        for (int i=0;i<R->m-1;i++){
            copyv(R->U +(size_t)(i+1)*ND, R->U +(size_t)i*ND);
            copyv(R->AU+(size_t)(i+1)*ND, R->AU+(size_t)i*ND);
            copyv(R->V +(size_t)(i+1)*ND, R->V +(size_t)i*ND);
        }
        R->m--;
    }
    copyv(u,R->U+(size_t)R->m*ND); rmmean(R->U+(size_t)R->m*ND);
    applyK(SIE,R->U+(size_t)R->m*ND,R->AU+(size_t)R->m*ND);
    copyv(v,R->V+(size_t)R->m*ND); rmmean(R->V+(size_t)R->m*ND);
    R->m++;
}
static void raw_guess_aopt(const Raw*R,const double*b,double*x0)
{
    static double G[MMAX][MMAX]; double f[MMAX],c[MMAX];
    zerov(x0); if (!R->m) return;
    for (int i=0;i<R->m;i++){
        f[i]=dot(R->U+(size_t)i*ND,b);
        for (int j=0;j<R->m;j++) G[i][j]=dot(R->U+(size_t)i*ND,R->AU+(size_t)j*ND);
    }
    sym_pinv_solve(G,f,c,R->m,1e-12);
    for (int i=0;i<R->m;i++) axpy(c[i],R->U+(size_t)i*ND,x0);
}
static double raw_guess_vmfit(const Raw*R,const double*vm,double*x0)
{
    static double H[MMAX][MMAX]; double g[MMAX],c[MMAX];
    static double *vc=NULL; if(!vc) vc=malloc(ND*sizeof(double));
    zerov(x0); if (!R->m) return 1.0;
    copyv(vm,vc); rmmean(vc);
    for (int i=0;i<R->m;i++){
        g[i]=dot(R->V+(size_t)i*ND,vc);
        for (int j=0;j<R->m;j++) H[i][j]=dot(R->V+(size_t)i*ND,R->V+(size_t)j*ND);
    }
    sym_pinv_solve(H,g,c,R->m,1e-12);
    double nv=nrm2(vc);
    for (int i=0;i<R->m;i++){
        axpy( c[i],R->U+(size_t)i*ND,x0);
        axpy(-c[i],R->V+(size_t)i*ND,vc);      /* vc becomes the fit residual */
    }
    return nv>0 ? nrm2(vc)/nv : 0.0;
}

/* ------------------------------------------------------------ Vm(x,t) model */
static double *tact, *apd;
static double *mode;          /* regime 2: 4 fixed spatial modes, ND each */
static int    REGIME = 0;
static const double BEAT = 350.0;

static double ap_shape(double tau,double A)      /* mV; tau = ms since activation */
{
    const double VR=-83.0;
    if (tau < 0)    return VR;
    if (tau < 1.0)  return VR + 113.0*tau;                  /* upstroke -> +30 */
    if (tau < 5.0)  return 30.0 - 10.0*(tau-1.0)/4.0;       /* notch    -> +20 */
    if (tau < A)    return 20.0 - 15.0*(tau-5.0)/(A-5.0);   /* plateau  -> +5  */
    if (tau < A+60.0){ double s=(tau-A)/60.0;               /* repolarisation  */
                       return 5.0 + (VR-5.0)*0.5*(1.0-cos(M_PI*s)); }
    return VR;
}
static void vm_field(double t,double *vm)
{
    if (REGIME==2){
        /* Low-dimensional but FAST: 4 fixed spatial modes with rapidly varying
         * amplitudes.  Consecutive samples are far apart (warm start is weak)
         * yet the whole trajectory lives in a 4-D subspace (a window nails it).
         * This is the regime forward_ecg.cpp reports for Sys2: warm -4%,
         * Fischer -69%. */
        const double f[4]={0.091,0.127,0.053,0.113}, ph[4]={0.0,1.7,3.1,0.6};
        double a[4];
        for (int q=0;q<4;q++) a[q]=sin(2*M_PI*f[q]*t+ph[q]);
        for (int p=0;p<ND;p++){
            double v=0;
            for (int q=0;q<4;q++) v += a[q]*mode[(size_t)q*ND+p];
            vm[p]=v;
        }
        return;
    }
    double tb = fmod(t,BEAT);
    for (int p=0;p<ND;p++) vm[p] = ap_shape(tb-tact[p], apd[p]);
}

/* ===================================================================== main */
int main(int argc,char**argv)
{
    N        = (argc>1)? atoi(argv[1]) : 24;
    int beats= (argc>2)? atoi(argv[2]) : 3;
    int win  = (argc>3)? atoi(argv[3]) : 16;
    int regime=(argc>4)? atoi(argv[4]) : 0;   /* 0 travelling+dispersed, 1 quasi-static,
                                                 2 low-dim but fast (warm weak) */
    int nsw  = (argc>5)? atoi(argv[5]) : 3;   /* SGS sweeps for the dVm increment */
    ND = N*N*N;
    REGIME = regime;
    build_diag();

    /* activation map (CV 2 units/ms, faster along fibres) and a spatial APD
     * gradient, so repolarisation sweeps as well -- no perfectly static plateau */
    tact=malloc(ND*sizeof(double)); apd=malloc(ND*sizeof(double));
    for (int i=0;i<N;i++) for (int j=0;j<N;j++) for (int k=0;k<N;k++){
        double dx=i/2.2,dy=j,dz=k;
        /* regime 0: fast front (CV 2/ms) + APD gradient -> repolarisation sweeps
         * regime 1: slow front (CV 0.5/ms) + uniform APD -> near-static plateau,
         *           the regime where forward_ecg.cpp sees 55/80 zero-iter steps */
        double cv = regime? 2.0 : 0.5;
        tact[IDX(i,j,k)] = cv*sqrt(dx*dx+dy*dy+dz*dz);
        apd [IDX(i,j,k)] = regime? 220.0 : 170.0 + 90.0*((double)i/(N-1));
    }

    if (regime==2){                       /* four fixed smooth spatial modes */
        mode=malloc((size_t)4*ND*sizeof(double));
        for (int i=0;i<N;i++) for (int j=0;j<N;j++) for (int k=0;k<N;k++){
            double X=(double)i/(N-1), Y=(double)j/(N-1), Z=(double)k/(N-1);
            int p=IDX(i,j,k);
            mode[0*ND+p]=60.0*exp(-8.0*((X-0.3)*(X-0.3)+(Y-0.4)*(Y-0.4)+(Z-0.5)*(Z-0.5)));
            mode[1*ND+p]=40.0*(X-0.5);
            mode[2*ND+p]=30.0*sin(M_PI*Y)*cos(2*M_PI*Z);
            mode[3*ND+p]=25.0*sin(2*M_PI*X)*sin(M_PI*Z);
        }
    }
    double *vm=malloc(ND*sizeof(double)), *b=malloc(ND*sizeof(double));
    double *bprev=malloc(ND*sizeof(double)), *ue=malloc(ND*sizeof(double));
    double *ueprev=malloc(ND*sizeof(double)), *x0=malloc(ND*sizeof(double));
    double *w=malloc(ND*sizeof(double)), *Aw=malloc(ND*sizeof(double));
    double *tmp=malloc(ND*sizeof(double)), *tmp2=malloc(ND*sizeof(double));
    double *d=malloc(ND*sizeof(double));
    zerov(ueprev); zerov(bprev);

    Fisch Fold, Frel, Fres, Fdiv;
    fisch_init(&Fold,"F-evict-oldest(abs)" ,win,0,0.0 ,EV_OLDEST);
    fisch_init(&Frel,"F-evict-oldest(rel)" ,win,1,1e-3,EV_OLDEST);
    fisch_init(&Fres,"F-restart-when-full" ,win,1,1e-3,EV_RESTART);
    fisch_init(&Fdiv,"F-evict-diverse"     ,win,1,1e-3,EV_DIVERSE);
    Raw R,Rdiv,Rbig;
    raw_init(&R   ,win  ,EV_OLDEST);   /* the true sliding window            */
    raw_init(&Rdiv,win  ,EV_DIVERSE);  /* phase-diverse retention            */
    raw_init(&Rbig,win*3,EV_OLDEST);   /* control: just 3x more memory       */

    long it_cold=0,it_warm=0,it_old=0,it_rel=0,it_res=0,it_div=0,
         it_vmf=0,it_aop=0,it_phys=0,it_wd=0,it_wd2=0,it_enr=0,it_swd=0,it_swb=0;
    long z_cold=0,z_warm=0,z_old=0,z_rel=0,z_res=0,z_div=0,z_vmf=0,z_aop=0,
         z_wd=0,z_wd2=0,z_enr=0,z_swd=0,z_swb=0;
    double sum_span_old=0,sum_span_rel=0;
    long nsteps=0,nskip=0;
    double sum_eA_vmf=0,sum_eA_aop=0,sum_eA_old=0,sum_eA_rel=0;
    long tp=0,fp=0,tn=0,fn=0; const double ETA_GATE=2e-2;

    printf("# n=%d nd=%d beats=%d window=%d regime=%d sweeps=%d\n",N,ND,beats,win,regime,nsw);
    printf("step,t,nb,cold,warm,f_old,f_rel,f_res,f_div,vmfit,aopt,phys,warmdelta,"
           "warmdelta12,enrich,sw_div,sw_big,eta,eA_vmfit,eA_aopt,eA_old,eA_rel,"
           "span_old,span_rel,m_old,m_rel,orth_old,orth_rel\n");

    int nT=(int)(beats*BEAT);
    for (int s=0;s<nT;s++){
        double t=s;                                   /* 1 ms sampling */
        vm_field(t,vm);
        applyK(SI,vm,b); scal(-1.0,b); rmmean(b);
        double nb=nrm2(b);
        if (nb < 1e-9){ nskip++; copyv(b,bprev); continue; }   /* diastole */
        double atol=1e-8*nb;
        nsteps++;

        /* kept solution: cold solve (this is the field, as in the real code) */
        zerov(ue); int k_cold=cg(b,ue,atol,4000); it_cold+=k_cold; if(!k_cold) z_cold++;

        copyv(ueprev,x0); int k_warm=cg(b,x0,atol,4000); it_warm+=k_warm; if(!k_warm) z_warm++;

        /* --- Fischer variants (Q2, Q3) --- */
        fisch_guess(&Fold,b,x0);
        for (int i=0;i<ND;i++) tmp[i]=ue[i]-x0[i];
        double eA_old=anorm(tmp,tmp2); sum_eA_old+=eA_old;
        int k_old=cg(b,x0,atol,4000); it_old+=k_old; if(!k_old) z_old++;

        fisch_guess(&Frel,b,x0);
        for (int i=0;i<ND;i++) tmp[i]=ue[i]-x0[i];
        double eA_rel=anorm(tmp,tmp2); sum_eA_rel+=eA_rel;
        int k_rel=cg(b,x0,atol,4000); it_rel+=k_rel; if(!k_rel) z_rel++;

        fisch_guess(&Fres,b,x0); int k_res=cg(b,x0,atol,4000); it_res+=k_res; if(!k_res) z_res++;
        fisch_guess(&Fdiv,b,x0); int k_div=cg(b,x0,atol,4000); it_div+=k_div; if(!k_div) z_div++;

        /* --- same span, two coefficient rules (Q1) --- */
        double eta=raw_guess_vmfit(&R,vm,x0);
        for (int i=0;i<ND;i++) tmp[i]=ue[i]-x0[i];
        double eA_vmf=anorm(tmp,tmp2); sum_eA_vmf+=eA_vmf;
        int k_vmf=cg(b,x0,atol,4000); it_vmf+=k_vmf; if(!k_vmf) z_vmf++;

        raw_guess_aopt(&R,b,x0);
        for (int i=0;i<ND;i++) tmp[i]=ue[i]-x0[i];
        double eA_aop=anorm(tmp,tmp2); sum_eA_aop+=eA_aop;
        int k_aop=cg(b,x0,atol,4000); it_aop+=k_aop; if(!k_aop) z_aop++;

        /* --- Q3b  same A-optimal rule, different retention policies --- */
        raw_guess_aopt(&Rdiv,b,x0); int k_swd=cg(b,x0,atol,4000);
        it_swd+=k_swd; if(!k_swd) z_swd++;
        raw_guess_aopt(&Rbig,b,x0); int k_swb=cg(b,x0,atol,4000);
        it_swb+=k_swb; if(!k_swb) z_swb++;

        /* --- physics guess x0 = c* Vm (cross-system, current-step Vm) --- */
        applyK(SIE,vm,tmp);
        double cst=dot(b,tmp)/dot(tmp,tmp);
        zerov(x0); axpy(cst,vm,x0); rmmean(x0);
        int k_phys=cg(b,x0,atol,4000); it_phys+=k_phys;

        /* --- Q5a  warm + smoothed increment:  u^{n-1} + SGS_3(b^n - b^{n-1})
         * b^n-b^{n-1} = -Ki (Vm^n - Vm^{n-1}) is free; 3 SGS sweeps ~ 3 matvecs */
        for (int i=0;i<ND;i++) tmp[i]=b[i]-bprev[i];
        zerov(d); for (int q=0;q<nsw;q++) sgs_sweep(tmp,d);
        rmmean(d);
        copyv(ueprev,x0); axpy(1.0,d,x0);
        int k_wd=cg(b,x0,atol,4000); it_wd+=k_wd; if(!k_wd) z_wd++;

        /* same idea with 12 sweeps: a 4x more expensive increment prediction */
        zerov(tmp2); for (int q=0;q<12;q++) sgs_sweep(tmp,tmp2);
        rmmean(tmp2);
        copyv(ueprev,x0); axpy(1.0,tmp2,x0);
        int k_wd2=cg(b,x0,atol,4000); it_wd2+=k_wd2; if(!k_wd2) z_wd2++;

        /* --- Q5b  sliding-window guess + the same increment direction, with an
         * A-optimal 1-D line search along it (cost: 3 sweeps + 2 matvecs) --- */
        raw_guess_aopt(&R,b,x0);
        applyK(SIE,x0,tmp); for (int i=0;i<ND;i++) tmp[i]=b[i]-tmp[i];  /* r */
        applyK(SIE,d,Aw);
        double dAd=dot(d,Aw);
        if (dAd>0) axpy(dot(d,tmp)/dAd, d, x0);
        int k_enr=cg(b,x0,atol,4000); it_enr+=k_enr; if(!k_enr) z_enr++;

        /* --- does the window still contain the previous solution? --- */
        double span_old = nsteps>1? fisch_spanerr(&Fold,ueprev,x0,tmp) : 0.0;
        double span_rel = nsteps>1? fisch_spanerr(&Frel,ueprev,x0,tmp) : 0.0;
        sum_span_old+=span_old; sum_span_rel+=span_rel;

        /* --- Q4  eta as a 0-iteration predictor, judged against SW-Aopt --- */
        if      (eta< ETA_GATE && k_aop==0) tp++;
        else if (eta< ETA_GATE && k_aop >0) fp++;
        else if (eta>=ETA_GATE && k_aop >0) tn++;
        else                                fn++;

        printf("%d,%.0f,%.6e,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,"
               "%.4e,%.6e,%.6e,%.6e,%.6e,%.4e,%.4e,%d,%d,%.2e,%.2e\n",
               s,t,nb,k_cold,k_warm,k_old,k_rel,k_res,k_div,k_vmf,k_aop,k_phys,
               k_wd,k_wd2,k_enr,k_swd,k_swb,eta,eA_vmf,eA_aop,eA_old,eA_rel,span_old,span_rel,
               Fold.m,Frel.m,fisch_orthloss(&Fold),fisch_orthloss(&Frel));

        /* every recycler grows from the SAME clean cold solution */
        fisch_update(&Fold,ue,vm,w,Aw,tmp);
        fisch_update(&Frel,ue,vm,w,Aw,tmp);
        fisch_update(&Fres,ue,vm,w,Aw,tmp);
        fisch_update(&Fdiv,ue,vm,w,Aw,tmp);
        raw_push(&R,ue,vm); raw_push(&Rdiv,ue,vm); raw_push(&Rbig,ue,vm);
        copyv(ue,ueprev); copyv(b,bprev);
    }

    printf("#\n# ===== summary (%ld solved steps, %ld skipped in diastole) =====\n",
           nsteps,nskip);
    printf("# %-22s %8s %9s %8s\n","guess","iters","vs cold","0-iter");
    #define ROW(nm,v,z) printf("# %-22s %8ld %8.1f%% %8ld\n",nm,(long)(v),\
        100.0*(1.0-(double)(v)/it_cold),(long)(z))
    ROW("cold",it_cold,z_cold);
    ROW("warm (prev u_e)",it_warm,z_warm);
    ROW("F-evict-oldest(abs)",it_old,z_old);
    ROW("F-evict-oldest(rel)",it_rel,z_rel);
    ROW("F-restart-when-full",it_res,z_res);
    ROW("F-evict-diverse",it_div,z_div);
    ROW("SW-vmfit (Vm coeffs)",it_vmf,z_vmf);
    ROW("SW-Aopt (A-optimal)",it_aop,z_aop);
    ROW("physics c*Vm",it_phys,0L);
    ROW("warm + SGS3(dVm)",it_wd,z_wd);
    ROW("warm + SGS12(dVm)",it_wd2,z_wd2);
    ROW("SW-Aopt + dVm enrich",it_enr,z_enr);
    ROW("SW-Aopt phase-diverse",it_swd,z_swd);
    ROW("SW-Aopt 3x window",it_swb,z_swb);
    printf("# mean ||x*-x0||_A : SW-vmfit %.4e  SW-Aopt %.4e  (vmfit/Aopt %.3f)\n",
           sum_eA_vmf/nsteps,sum_eA_aop/nsteps,
           sum_eA_aop>0? sum_eA_vmf/sum_eA_aop : 0.0);
    printf("# mean ||x*-x0||_A : F-old   %.4e  F-rel   %.4e  SW-Aopt %.4e\n",
           sum_eA_old/nsteps,sum_eA_rel/nsteps,sum_eA_aop/nsteps);
    printf("# churn: F-abs acc=%ld rej=%ld evict=%ld | F-rel acc=%ld rej=%ld evict=%ld"
           " | F-restart restarts=%ld\n",
           Fold.accepted,Fold.rejected,Fold.evicted,
           Frel.accepted,Frel.rejected,Frel.evicted,Fres.restarts);
    printf("# mean ||u^{n-1}-P u^{n-1}||_A/||u^{n-1}||_A (0 = window really holds"
           " the last m solutions): F-old %.3f  F-rel %.3f\n",
           sum_span_old/nsteps,sum_span_rel/nsteps);
    printf("# final A-orthogonality loss: F-abs %.3e  F-rel %.3e\n",
           fisch_orthloss(&Fold),fisch_orthloss(&Frel));
    printf("# eta<%.0e predicts 0-iter (vs SW-Aopt): TP=%ld FP=%ld TN=%ld FN=%ld"
           "  precision=%.3f recall=%.3f\n",ETA_GATE,tp,fp,tn,fn,
           (tp+fp)?(double)tp/(tp+fp):0.0,(tp+fn)?(double)tp/(tp+fn):0.0);
    return 0;
}
