/* fischer_gram_selftest.cpp -- unit test for the bookkeeping introduced by
 * asm_bug_demo/fischer_sliding_window.patch (the Sys2 recycler in
 * forward_ecg.cpp).  That patch is the one piece that cannot be exercised here,
 * because building forward_ecg.cpp needs MFEM + PETSc; what CAN be checked is
 * that the incremental Gram matrix stays correct across eviction and that the
 * resulting guess really is the A-orthogonal projection onto the window.
 *
 * The code under test below is copied VERBATIM from the patch, with mfem::Vector
 * / HypreParMatrix / MPI replaced by minimal local stand-ins and LAPACK's dsyev_
 * replaced by a Jacobi eigensolver that honours the same contract (ascending
 * eigenvalues, column-major eigenvectors, workspace query).
 *
 * Checks, every step:
 *   (1) the incrementally maintained Gram equals the one recomputed from scratch;
 *   (2) the residual of the guess is orthogonal to every stored snapshot -- the
 *       defining property of the A-orthogonal projection onto span{u^k}
 *       (checked independently of gram_solve, so the test is not circular).
 *
 * Build: c++ -O2 -o fischer_gram_selftest fischer_gram_selftest.cpp && ./fischer_gram_selftest
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

/* ----------------------------------------------------------- stand-ins ---- */
struct Vector {
    std::vector<double> d;
    Vector(){} explicit Vector(int n):d(n,0.0){}
    double  operator()(int i) const { return d[i]; }
    double& operator()(int i)       { return d[i]; }
    int  Size() const { return (int)d.size(); }
    void Add(double a,const Vector&x){ for(size_t i=0;i<d.size();++i) d[i]+=a*x.d[i]; }
    Vector& operator=(double v){ std::fill(d.begin(),d.end(),v); return *this; }
};
static int NDOF;
static std::vector<double> Amat;                 /* dense SPD "Kie" */
struct Op { void Mult(const Vector&x,Vector&y) const {
    for (int i=0;i<NDOF;i++){ double s=0; for(int j=0;j<NDOF;j++) s+=Amat[i*NDOF+j]*x(j); y(i)=s; } } };
static Op KieObj; static Op*Kie=&KieObj;

#define MPI_IN_PLACE      nullptr
#define MPI_DOUBLE        0
#define MPI_SUM           0
#define MPI_COMM_WORLD    0
static void MPI_Allreduce(void*,double*,int,int,int,int){}   /* serial: no-op */

/* dsyev_ stand-in with the LAPACK contract (jobz='V', uplo='U'): ascending
 * eigenvalues in w, eigenvector e in a[e*n .. e*n+n-1] (column-major). */
extern "C" void dsyev_(char*,char*,int*n,double*a,int*,double*w,double*work,int*lwork,int*info)
{
    if (*lwork < 0){ work[0]=(double)(4*(*n)); *info=0; return; }
    const int m=*n;
    std::vector<double> A(a,a+m*m), V(m*m,0.0);
    for (int i=0;i<m;i++) V[i*m+i]=1.0;
    for (int sweep=0;sweep<80;sweep++){
        double off=0; for(int i=0;i<m;i++) for(int j=i+1;j<m;j++) off+=A[i*m+j]*A[i*m+j];
        if (off<1e-30) break;
        for (int p=0;p<m;p++) for (int q=p+1;q<m;q++){
            if (std::fabs(A[p*m+q])<1e-300) continue;
            double th=0.5*(A[q*m+q]-A[p*m+p])/A[p*m+q];
            double t=(th>=0?1.0:-1.0)/(std::fabs(th)+std::sqrt(th*th+1.0));
            double c=1.0/std::sqrt(t*t+1.0), s=t*c;
            for (int k=0;k<m;k++){ double akp=A[k*m+p],akq=A[k*m+q];
                A[k*m+p]=c*akp-s*akq; A[k*m+q]=s*akp+c*akq; }
            for (int k=0;k<m;k++){ double apk=A[p*m+k],aqk=A[q*m+k];
                A[p*m+k]=c*apk-s*aqk; A[q*m+k]=s*apk+c*aqk; }
            for (int k=0;k<m;k++){ double vkp=V[k*m+p],vkq=V[k*m+q];
                V[k*m+p]=c*vkp-s*vkq; V[k*m+q]=s*vkp+c*vkq; }
        }
    }
    std::vector<int> ord(m); for (int i=0;i<m;i++) ord[i]=i;
    std::sort(ord.begin(),ord.end(),[&](int x,int y){ return A[x*m+x]<A[y*m+y]; });
    for (int e=0;e<m;e++){ w[e]=A[ord[e]*m+ord[e]];
        for (int i=0;i<m;i++) a[e*m+i]=V[i*m+ord[e]]; }
    *info=0;
}

/* ============ code under test: copied verbatim from the patch ============== */
static std::vector<Vector> fisch_U, fisch_AU;
static std::vector<double> fisch_G;
static const int FISCH_MAX = 8;
static long fisch_red_new=0, fisch_red_old=0;

static void ipbatch(const std::vector<Vector>&V,const Vector&y,std::vector<double>&dots)
{
    const size_t m=V.size(); dots.assign(m,0.0);
    for (size_t i=0;i<m;++i){ const Vector&v=V[i]; double s=0.0;
        for (int k=0;k<v.Size();++k) s+=v(k)*y(k); dots[i]=s; }
    if (m) MPI_Allreduce(MPI_IN_PLACE,dots.data(),(int)m,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
}
static void gram_solve(const std::vector<double>&G,const std::vector<double>&f,
                       std::vector<double>&c)
{
    const int m=(int)f.size(); c.assign(m,0.0); if (!m) return;
    std::vector<double> Q(G), ev(m);
    char jz='V', up='U'; int NN=m, info=0, lw=-1; double wq=0;
    dsyev_(&jz,&up,&NN,Q.data(),&NN,ev.data(),&wq,&lw,&info);
    lw=(int)wq; std::vector<double> work(std::max(1,lw));
    dsyev_(&jz,&up,&NN,Q.data(),&NN,ev.data(),work.data(),&lw,&info);
    if (info) return;
    const double emax = ev[m-1];
    for (int e=0;e<m;++e){
        if (ev[e] <= 1e-12*emax || ev[e] <= 0) continue;
        const double *v = &Q[(size_t)e*m];
        double vf=0; for (int i=0;i<m;++i) vf += v[i]*f[i];
        const double sc = vf/ev[e];
        for (int i=0;i<m;++i) c[i] += sc*v[i];
    }
}
static void window_update(const Vector&w_in)          /* the patch's update block */
{
    Vector w(w_in); Vector Aw(NDOF); Kie->Mult(w,Aw);
    {   int m = (int)fisch_U.size();
        std::vector<double> gnew(m+1, 0.0);
        for (int i=0;i<m;++i){ const Vector&v=fisch_U[i]; double s=0.0;
            for (int k=0;k<v.Size();++k) s += v(k)*Aw(k); gnew[i]=s; }
        { double s=0.0; for (int k=0;k<w.Size();++k) s += w(k)*Aw(k); gnew[m]=s; }
        MPI_Allreduce(MPI_IN_PLACE,gnew.data(),m+1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        fisch_red_new += 1; fisch_red_old += (long)m + 1;
        if (m >= FISCH_MAX){
            fisch_U.erase(fisch_U.begin());
            fisch_AU.erase(fisch_AU.begin());
            gnew.erase(gnew.begin());
            std::vector<double> Gs((size_t)(m-1)*(m-1));
            for (int i=1;i<m;++i) for (int j=1;j<m;++j)
                Gs[(size_t)(i-1)*(m-1)+(j-1)] = fisch_G[(size_t)i*m+j];
            fisch_G.swap(Gs); m--;
        }
        std::vector<double> Gn((size_t)(m+1)*(m+1));
        for (int i=0;i<m;++i) for (int j=0;j<m;++j)
            Gn[(size_t)i*(m+1)+j] = fisch_G[(size_t)i*m+j];
        for (int i=0;i<m;++i){
            Gn[(size_t)i*(m+1)+m] = gnew[i];
            Gn[(size_t)m*(m+1)+i] = gnew[i]; }
        Gn[(size_t)m*(m+1)+m] = gnew[m];
        fisch_G.swap(Gn);
        fisch_U.push_back(w); fisch_AU.push_back(Aw);
    }
}
/* ========================================================================== */

int main()
{
    NDOF = 40;
    /* SPD operator: A = B^T B + I with a fixed pseudo-random B */
    unsigned seed=12345;
    auto rnd=[&](){ seed=seed*1103515245u+12345u; return ((seed>>16)&0x7fff)/16384.0-1.0; };
    std::vector<double> B(NDOF*NDOF); for (auto&v:B) v=rnd();
    Amat.assign(NDOF*NDOF,0.0);
    for (int i=0;i<NDOF;i++) for (int j=0;j<NDOF;j++){
        double s=0; for (int k=0;k<NDOF;k++) s+=B[k*NDOF+i]*B[k*NDOF+j];
        Amat[i*NDOF+j]=s+(i==j?1.0:0.0);
    }

    double worst_gram=0, worst_orth=0;
    for (int t=0;t<40;t++){
        /* a smoothly drifting "solution" on a low-dimensional manifold */
        Vector u(NDOF);
        for (int i=0;i<NDOF;i++)
            u(i) = std::sin(0.3*i+0.05*t) + 0.4*std::cos(0.11*i-0.02*t)
                 + 0.02*std::sin(1.7*i+0.9*t);
        Vector Au(NDOF); Kie->Mult(u,Au);          /* b = A u  => exact solution u */

        /* --- the guess, exactly as the patch computes it --- */
        std::vector<double> f,c; ipbatch(fisch_U,Au,f);
        gram_solve(fisch_G,f,c);
        Vector xf(NDOF); xf=0.0;
        for (size_t i=0;i<fisch_U.size();++i) xf.Add(c[i],fisch_U[i]);

        /* check (2): r = b - A x0 must be orthogonal to every stored snapshot */
        Vector Axf(NDOF); Kie->Mult(xf,Axf);
        double nb=0; for (int i=0;i<NDOF;i++) nb+=Au(i)*Au(i); nb=std::sqrt(nb);
        for (size_t k=0;k<fisch_U.size();++k){
            double s=0,nu=0;
            for (int i=0;i<NDOF;i++){ s+=fisch_U[k](i)*(Au(i)-Axf(i)); nu+=fisch_U[k](i)*fisch_U[k](i); }
            double rel=std::fabs(s)/(std::sqrt(nu)*nb+1e-300);
            worst_orth=std::max(worst_orth,rel);
        }

        window_update(u);

        /* check (1): incremental Gram == Gram recomputed from scratch */
        const int m=(int)fisch_U.size();
        for (int i=0;i<m;i++) for (int j=0;j<m;j++){
            double s=0; for (int q=0;q<NDOF;q++) s+=fisch_U[i](q)*fisch_AU[j](q);
            double e=std::fabs(s-fisch_G[(size_t)i*m+j])/(std::fabs(s)+1e-300);
            worst_gram=std::max(worst_gram,e);
        }
    }
    printf("window size            : %d (max %d)\n",(int)fisch_U.size(),FISCH_MAX);
    printf("max rel Gram error     : %.3e   (incremental vs from scratch)\n",worst_gram);
    printf("max rel <u^k, b-A x0>  : %.3e   (A-orthogonal projection property)\n",worst_orth);
    printf("Allreduces: new=%ld  old-equivalent=%ld\n",fisch_red_new,fisch_red_old);
    bool ok = worst_gram<1e-10 && worst_orth<1e-8;
    printf("%s\n", ok? "SELFTEST PASS":"SELFTEST FAIL");
    return ok?0:1;
}
