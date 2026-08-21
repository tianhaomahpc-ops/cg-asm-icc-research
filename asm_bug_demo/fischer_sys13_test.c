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
 * Run:   ./fischer_sys13_test [n] [steps] [window] [regime] > out.csv
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

/* ------------------------------------------------------------- recyclers ---
 * ORTHO = what forward_ecg.cpp does today (A-orthonormal basis, relative
 *         acceptance threshold as in the Sys3 block, evict the oldest VECTOR)
 * RAW   = the true sliding window (last m solutions, A-optimal coefficients) */
enum { ORTHO=0, RAW=1 };
typedef struct {
    int sys, kind, mmax, m;
    double *P, *AP;      /* ORTHO: A-orthonormal basis; RAW: raw snapshots */
    long accepted, rejected, evicted;
} Rec;

static void rec_init(Rec*R,int sys,int kind,int mmax)
{
    R->sys=sys; R->kind=kind; R->mmax=mmax; R->m=0;
    R->P =calloc((size_t)mmax*ND,sizeof(double));
    R->AP=calloc((size_t)mmax*ND,sizeof(double));
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
            for (int j=i;j<R->m;j++){                       /* Gram is symmetric */
                double g=dot(R->P+(size_t)i*ND,R->AP+(size_t)j*ND);
                Gbuf[i][j]=g; Gbuf[j][i]=g;
            }
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
        if (R->m>=R->mmax) rec_drop_oldest(R);
        copyv(w, R->P+(size_t)R->m*ND); copyv(Aw,R->AP+(size_t)R->m*ND);
        R->m++; R->accepted++;
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
    int nsnap=0; double *snap2=malloc((size_t)MMAX*ND*sizeof(double));
    double *snap3=malloc((size_t)MMAX*ND*sizeof(double));

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
            copyv(prev[2],snap2+(size_t)nsnap*ND);
            copyv(prev[3],snap3+(size_t)nsnap*ND);
            nsnap++;
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
    traj_rank(snap2,nsnap,"Sys2 solution u_e");
    traj_rank(snap3,nsnap,"Sys3 solution u_T");
    return 0;
}
