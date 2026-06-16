/* frontprop.c -- QUANTIFYING information propagation in one-level Schwarz.
 *
 * Elliptic (Laplace) solutions are globally coupled: the value anywhere depends
 * on ALL the boundary data (A is sparse but A^{-1} is dense). One-level Schwarz,
 * however, only solves local subdomain problems each iteration, so boundary
 * information advances by ~one subdomain (plus the overlap) per iteration. That
 * latency is exactly the slow global mode = the small lambda_min that makes
 * one-level CG need O(#subdomains) iterations -- and it is what a coarse space
 * (two-level) removes.
 *
 * Setup: 1D Laplace, N interior nodes, P subdomains, left node Dirichlet source
 * (b_0=1) so the exact solution is the ramp u*(x)=1-x. We run the STATIONARY
 * one-level BASIC iteration  x_{k+1} = x_k + M^{-1}(b - A x_k)  (exact Cholesky
 * subdomain solves -> a clean contraction) and, each iteration, record the
 * "propagation front": the furthest node the signal has reached, in units of
 * subdomains. We sweep the overlap O to show the front advances faster (fewer
 * iterations to cross) with more overlap.
 *
 * Dumps frontprop_front.txt (O k front_in_subdomains) and frontprop_profile.txt
 * (the fill-in profiles at O=ProfO). Sequential.
 */
#include <petscksp.h>
#include <math.h>

typedef struct { PetscInt n; PetscInt *idx; IS is; Mat Ai; KSP ksp; Vec ri, yi; } Sub;

static Mat Laplace1D(PetscInt N) {
    Mat A; MatCreateSeqAIJ(PETSC_COMM_SELF, N, N, 3, NULL, &A);
    for (PetscInt i = 0; i < N; ++i) {
        PetscScalar two = 2.0, m1 = -1.0;
        MatSetValue(A, i, i, two, INSERT_VALUES);
        if (i > 0)   MatSetValue(A, i, i-1, m1, INSERT_VALUES);
        if (i < N-1) MatSetValue(A, i, i+1, m1, INSERT_VALUES);
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE); return A;
}
static void BuildSub(Mat A, PetscInt *idx, PetscInt ni, Sub *s) {
    s->n = ni; s->idx = (PetscInt*)malloc(sizeof(PetscInt)*ni);
    for (PetscInt j = 0; j < ni; ++j) s->idx[j] = idx[j];
    ISCreateGeneral(PETSC_COMM_SELF, ni, s->idx, PETSC_COPY_VALUES, &s->is);
    MatCreateSubMatrix(A, s->is, s->is, MAT_INITIAL_MATRIX, &s->Ai);
    KSPCreate(PETSC_COMM_SELF, &s->ksp); KSPSetType(s->ksp, KSPPREONLY);
    KSPSetOperators(s->ksp, s->Ai, s->Ai);
    PC pc; KSPGetPC(s->ksp, &pc); PCSetType(pc, PCCHOLESKY);
    KSPSetUp(s->ksp); MatCreateVecs(s->Ai, &s->ri, &s->yi);
}
/* z = M_BASIC^{-1} r = sum_i R_i^T A_i^{-1} R_i r */
static void ApplyBASIC(Sub *subs, PetscInt nsub, Vec r, Vec z) {
    VecZeroEntries(z);
    const PetscScalar *ra; PetscScalar *za;
    VecGetArrayRead(r, &ra); VecGetArray(z, &za);
    for (PetscInt i = 0; i < nsub; ++i) {
        Sub *s = &subs[i];
        PetscScalar *rib; VecGetArray(s->ri, &rib);
        for (PetscInt j = 0; j < s->n; ++j) rib[j] = ra[s->idx[j]];
        VecRestoreArray(s->ri, &rib);
        KSPSolve(s->ksp, s->ri, s->yi);
        const PetscScalar *yib; VecGetArrayRead(s->yi, &yib);
        for (PetscInt j = 0; j < s->n; ++j) za[s->idx[j]] += yib[j];
        VecRestoreArrayRead(s->yi, &yib);
    }
    VecRestoreArrayRead(r, &ra); VecRestoreArray(z, &za);
}

int main(int argc, char **argv) {
    PetscInitialize(&argc, &argv, NULL, NULL);
    const PetscInt N = 384, P = 12;       /* stride = 32 nodes per subdomain */
    const PetscReal stride = (PetscReal)N / P;
    const int MAXK = 40;
    Mat A = Laplace1D(N);
    Vec b, ustar; MatCreateVecs(A, &b, &ustar);
    VecZeroEntries(b); VecSetValue(b, 0, 1.0, INSERT_VALUES);
    VecAssemblyBegin(b); VecAssemblyEnd(b);
    { KSP kd; KSPCreate(PETSC_COMM_SELF,&kd); KSPSetType(kd,KSPPREONLY);
      KSPSetOperators(kd,A,A); PC pc; KSPGetPC(kd,&pc); PCSetType(pc,PCCHOLESKY);
      KSPSolve(kd,b,ustar); KSPDestroy(&kd); }          /* u* = ramp 1 -> 0 */

    FILE *ff = fopen("frontprop_front.txt","w");
    fprintf(ff, "# O k front_in_subdomains\n");
    int Ovs[3] = {2, 6, 12}; int ProfO = 6;
    PetscPrintf(PETSC_COMM_SELF,"1D Laplace N=%d, %d subdomains (stride %.0f), exact BASIC stationary\n",
                (int)N,(int)P,(double)stride);
    PetscPrintf(PETSC_COMM_SELF,"%4s %20s\n","O","iters for front->end");

    for (int oi = 0; oi < 3; ++oi) {
        int O = Ovs[oi];
        Sub subs[12];
        for (PetscInt i = 0; i < P; ++i) {
            PetscInt lo=(i*N)/P, hi=((i+1)*N)/P;
            PetscInt a=lo-O<0?0:lo-O, c=hi+O>N?N:hi+O;
            PetscInt ni=c-a, *idx=(PetscInt*)malloc(sizeof(PetscInt)*ni);
            for (PetscInt j=0;j<ni;++j) idx[j]=a+j;
            BuildSub(A, idx, ni, &subs[i]); free(idx);
        }
        Vec x, r, z; VecDuplicate(b,&x); VecDuplicate(b,&r); VecDuplicate(b,&z);
        VecZeroEntries(x);
        const PetscScalar *us; VecGetArrayRead(ustar,&us);
        int kreach = -1;
        FILE *fp = (O==ProfO) ? fopen("frontprop_profile.txt","w") : NULL;
        int profKs[6] = {0,1,2,3,6,10}; int pidx = 0;
        for (int k = 0; k <= MAXK; ++k) {
            const PetscScalar *xa; VecGetArrayRead(x,&xa);
            /* front = furthest node whose value has reached half of the exact ramp */
            PetscInt front = 0;
            for (PetscInt i = 0; i < N; ++i)
                if (PetscRealPart(xa[i]) >= 0.5*PetscRealPart(us[i])) front = i;
            PetscReal fsub = (PetscReal)front / stride;
            fprintf(ff, "%d %d %.4f\n", O, k, (double)fsub);
            if (kreach < 0 && fsub >= P-1.0) kreach = k;
            if (fp && pidx < 6 && k == profKs[pidx]) {
                fprintf(fp, "# k=%d\n", k);
                for (PetscInt i=0;i<N;++i) fprintf(fp, "%g %g\n",
                    (double)(i+1)/(N+1), PetscRealPart(xa[i]));
                fprintf(fp, "\n"); ++pidx;
            }
            VecRestoreArrayRead(x,&xa);
            MatMult(A, x, r); VecAYPX(r, -1.0, b);       /* r = b - A x */
            ApplyBASIC(subs, P, r, z);
            VecAXPY(x, 1.0, z);                          /* x += M^{-1} r */
        }
        VecRestoreArrayRead(ustar,&us);
        if (fp) fclose(fp);
        PetscPrintf(PETSC_COMM_SELF,"%4d %20d\n", O, kreach);
        VecDestroy(&x); VecDestroy(&r); VecDestroy(&z);
        for (PetscInt i=0;i<P;++i){ISDestroy(&subs[i].is);MatDestroy(&subs[i].Ai);
            KSPDestroy(&subs[i].ksp);VecDestroy(&subs[i].ri);VecDestroy(&subs[i].yi);free(subs[i].idx);}
    }
    fclose(ff);
    VecDestroy(&b); VecDestroy(&ustar); MatDestroy(&A);
    PetscPrintf(PETSC_COMM_SELF,"FRONTPROP_DONE\n");
    PetscFinalize(); return 0;
}
