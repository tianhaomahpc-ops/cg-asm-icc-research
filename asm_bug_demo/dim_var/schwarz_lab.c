/* schwarz_lab.c  --  PETSc dimensional + variable-coefficient additive-Schwarz lab.
 *
 * The PETSc counterpart of dim_var/asm_spectral.py.  Same problem, same box
 * subdomains, same BASIC / sASM variants, but using REAL PETSc PCASM + PCICC /
 * PCCHOLESKY local solves and PETSc's own spectral estimator.  Built to run on
 * both Homebrew PETSc 3.24 (this Mac) and the spack PETSc 3.24 on the HPC box,
 * so results cross-validate the NumPy reference and scale up unchanged.
 *
 * Operator:  -div( a(x) grad u ) = 1   on [0,1]^d,
 *   u = 0 on the x0 = 0 face (Dirichlet, symmetric elimination),
 *   zero-flux Neumann on the other faces.
 *   Conservative vertex-centred FD, harmonic face averaging => a==1 gives the
 *   standard (2d+1)-point Laplacian (identical to the prior pure_petsc_demo.c).
 *
 * Decomposition: S^d regular box subdomains given EXPLICITLY to PCASM via
 *   PCASMSetLocalSubdomains (so the count is independent of MPI ranks; run -n 1
 *   for the controlled study), grown by PCASMSetOverlap(O).
 *
 * Knobs (CLI):
 *   -dim {1,2,3}      spatial dimension                  (default 2)
 *   -nx  M            grid points per axis               (default 33)
 *   -S   S            subdomains per axis (total S^dim)   (default 4)
 *   -overlap O        ASM overlap in grid points         (default 1)
 *   -coef NAME        const|smooth|layers|layers_unaligned|checker  (default const)
 *   -contrast RHO     coefficient contrast for jump fields(default 1)
 *   -scheme {0,3}     0 = CG + ASM_BASIC, 3 = CG + sASM PCSHELL   (default 0)
 *   -exact            use exact local solve (Cholesky) instead of ICC
 *   -icc_levels L     ICC fill level when not -exact      (default 0)
 *   -measure_omega    also measure omega = max_i lam_max(M_i^{-1} A_i)
 *
 * Reports one line:
 *   [RESULT] dim S nx coef rho scheme exact iccL overlap | iter reason
 *            lam_min lam_max kappa Nhat omega
 */
#include <petscksp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---------------- coefficient field a(x) at node ----------------------- */
static double coef_at(const char *name, int dim, int M, double rho,
                      const int *mi)
{
    /* mi = multi-index (length dim), coords x_k = mi[k]/(M-1) in [0,1] */
    double x[3] = {0, 0, 0};
    for (int k = 0; k < dim; ++k) x[k] = (double)mi[k] / (double)(M - 1);

    if (!strcmp(name, "const"))  return 1.0;
    if (!strcmp(name, "smooth")) {
        double s = 1.0;
        for (int k = 0; k < dim; ++k) s *= sin(2.0 * M_PI * x[k]);
        return 1.0 + 0.5 * s;                       /* in [0.5,1.5] */
    }
    if (!strcmp(name, "layers") || !strcmp(name, "layers_unaligned")) {
        int n_layers = 4;
        double shift = strcmp(name, "layers") ? 0.5 / n_layers : 0.0;
        int idx = (int)floor((x[0] - shift) * n_layers);
        return (idx % 2 == 0) ? 1.0 : rho;
    }
    if (!strcmp(name, "checker")) {
        int idx = 0;
        for (int k = 0; k < dim; ++k) idx += (int)floor(x[k] * 4);
        return (idx % 2 == 0) ? 1.0 : rho;
    }
    return 1.0;
}

/* lexicographic global index, x0 fastest */
static PetscInt lin_index(int dim, int M, const int *mi)
{
    PetscInt lin = 0, stride = 1;
    for (int k = 0; k < dim; ++k) { lin += (PetscInt)mi[k] * stride; stride *= M; }
    return lin;
}
static void decode(int dim, int M, PetscInt lin, int *mi)
{
    for (int k = 0; k < dim; ++k) { mi[k] = (int)(lin % M); lin /= M; }
}

/* ---------------- assemble A (SeqAIJ) and RHS b ------------------------ */
static PetscErrorCode assemble(int dim, int M, const char *coef, double rho,
                               Mat *Aout, Vec *bout)
{
    PetscInt Npts = 1;
    for (int k = 0; k < dim; ++k) Npts *= M;
    double h = 1.0 / (M - 1), hinv2 = 1.0 / (h * h);

    Mat A;
    PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
    PetscCall(MatSetType(A, MATAIJ));
    PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, Npts, Npts));
    PetscCall(MatMPIAIJSetPreallocation(A, 2 * dim + 1, NULL, 2 * dim + 1, NULL));
    PetscCall(MatSeqAIJSetPreallocation(A, 2 * dim + 1, NULL));
    PetscCall(MatSetUp(A));

    PetscInt rstart, rend;
    PetscCall(MatGetOwnershipRange(A, &rstart, &rend));

    /* symmetric Neumann operator: per node, add couplings to in-domain nbrs */
    for (PetscInt lin = rstart; lin < rend; ++lin) {
        int mi[3]; decode(dim, M, lin, mi);
        double ai = coef_at(coef, dim, M, rho, mi);
        double diag = 0.0;
        for (int k = 0; k < dim; ++k) {
            for (int step = -1; step <= 1; step += 2) {
                int mj[3]; for (int t = 0; t < dim; ++t) mj[t] = mi[t];
                mj[k] += step;
                if (mj[k] < 0 || mj[k] >= M) continue;       /* Neumann: drop */
                double aj = coef_at(coef, dim, M, rho, mj);
                double aface = 2.0 * ai * aj / (ai + aj);     /* harmonic mean */
                double w = aface * hinv2;
                PetscInt col = lin_index(dim, M, mj);
                PetscScalar v = -w;
                PetscCall(MatSetValues(A, 1, &lin, 1, &col, &v, ADD_VALUES));
                diag += w;
            }
        }
        PetscScalar dv = diag;
        PetscCall(MatSetValues(A, 1, &lin, 1, &lin, &dv, ADD_VALUES));
    }
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

    /* RHS and Dirichlet dof list (x0 == 0, i.e. mi[0]==0) */
    Vec b;
    PetscCall(MatCreateVecs(A, &b, NULL));
    PetscCall(VecSet(b, 1.0));
    PetscInt *dir = (PetscInt*)malloc(sizeof(PetscInt) * Npts);
    PetscInt ndir = 0;
    for (PetscInt lin = 0; lin < Npts; ++lin) {
        int mi[3]; decode(dim, M, lin, mi);
        if (mi[0] == 0) dir[ndir++] = lin;
    }
    /* symmetric Dirichlet elimination: zero rows AND cols, diag=1, u=0 */
    PetscCall(VecSetValue(b, 0, 0.0, INSERT_VALUES));            /* placeholder */
    for (PetscInt i = 0; i < ndir; ++i)
        PetscCall(VecSetValue(b, dir[i], 0.0, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(b)); PetscCall(VecAssemblyEnd(b));
    PetscCall(MatZeroRowsColumns(A, ndir, dir, 1.0, NULL, NULL));
    free(dir);

    PetscCall(MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE));
    PetscCall(MatSetOption(A, MAT_SPD, PETSC_TRUE));
    *Aout = A; *bout = b;
    return PETSC_SUCCESS;
}

/* ---------------- S^d box subdomain index sets ------------------------- */
static PetscErrorCode make_subdomains(int dim, int M, int S, PetscInt *n_sub_out,
                                      IS **is_out)
{
    int n_sub = 1; for (int k = 0; k < dim; ++k) n_sub *= S;
    /* per-axis base ranges [lo,hi) from linspace(0,M,S+1) */
    int edge[64];
    for (int s = 0; s <= S; ++s) edge[s] = (int)((long)s * M / S);

    IS *is = (IS*)malloc(sizeof(IS) * n_sub);
    int box[3];
    for (int b = 0; b < n_sub; ++b) {
        int bb = b;
        for (int k = 0; k < dim; ++k) { box[k] = bb % S; bb /= S; }
        /* count + collect node indices in this box */
        PetscInt cap = 1;
        for (int k = 0; k < dim; ++k) cap *= (edge[box[k] + 1] - edge[box[k]]);
        PetscInt *idx = (PetscInt*)malloc(sizeof(PetscInt) * cap);
        PetscInt cnt = 0;
        int mi[3];
        /* odometer over the box's per-axis ranges */
        int lo[3], hi[3];
        for (int k = 0; k < dim; ++k) { lo[k] = edge[box[k]]; hi[k] = edge[box[k] + 1]; }
        for (int k = 0; k < dim; ++k) mi[k] = lo[k];
        int done = 0;
        while (!done) {
            idx[cnt++] = lin_index(dim, M, mi);
            int k = 0;
            while (k < dim) {
                mi[k]++;
                if (mi[k] < hi[k]) break;
                mi[k] = lo[k]; k++;
            }
            if (k == dim) done = 1;
        }
        PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, cnt, idx, PETSC_COPY_VALUES, &is[b]));
        free(idx);
    }
    *n_sub_out = n_sub; *is_out = is;
    return PETSC_SUCCESS;
}

/* ---------------- sASM PCSHELL (scheme 3) ------------------------------ */
typedef struct { PC inner; Vec invsqrt, tmp; } SASMCtx;
static PetscErrorCode sASMApply(PC pc, Vec r, Vec z) {
    SASMCtx *c; PetscCall(PCShellGetContext(pc, (void**)&c));
    PetscCall(VecPointwiseMult(c->tmp, c->invsqrt, r));
    PetscCall(PCApply(c->inner, c->tmp, z));
    PetscCall(VecPointwiseMult(z, c->invsqrt, z));
    return PETSC_SUCCESS;
}
static PetscErrorCode sASMDestroy(PC pc) {
    SASMCtx *c; PetscCall(PCShellGetContext(pc, (void**)&c));
    if (c) { PetscCall(PCDestroy(&c->inner)); PetscCall(VecDestroy(&c->invsqrt));
             PetscCall(VecDestroy(&c->tmp)); free(c); }
    return PETSC_SUCCESS;
}

/* compute Nhat from the overlapping subdomains of an already-setup PCASM */
static PetscErrorCode compute_nhat(PC asmpc, Mat A, PetscReal *nhat) {
    PetscInt n_local; IS *is_over;
    PetscCall(PCASMGetLocalSubdomains(asmpc, &n_local, &is_over, NULL));
    Vec mult; PetscCall(MatCreateVecs(A, &mult, NULL));
    PetscCall(VecSet(mult, 0.0));
    for (PetscInt i = 0; i < n_local; ++i) {
        const PetscInt *idx; PetscInt n;
        PetscCall(ISGetLocalSize(is_over[i], &n));
        PetscCall(ISGetIndices(is_over[i], &idx));
        for (PetscInt j = 0; j < n; ++j)
            PetscCall(VecSetValue(mult, idx[j], 1.0, ADD_VALUES));
        PetscCall(ISRestoreIndices(is_over[i], &idx));
    }
    PetscCall(VecAssemblyBegin(mult)); PetscCall(VecAssemblyEnd(mult));
    PetscCall(VecMax(mult, NULL, nhat));
    PetscCall(VecDestroy(&mult));
    return PETSC_SUCCESS;
}

int main(int argc, char **argv)
{
    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));

    int dim = 2, M = 33, S = 4, overlap = 1, scheme = 0, iccL = 0;
    char coef[64] = "const";
    double rho = 1.0;
    PetscBool exact = PETSC_FALSE, measure_omega = PETSC_FALSE, set;
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-dim", &dim, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-nx", &M, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-S", &S, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-overlap", &overlap, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-scheme", &scheme, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-icc_levels", &iccL, NULL));
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-contrast", &rho, NULL));
    PetscCall(PetscOptionsGetString(NULL, NULL, "-coef", coef, sizeof(coef), NULL));
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-exact", &exact, &set));
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-measure_omega", &measure_omega, &set));

    Mat A; Vec b, x;
    PetscCall(assemble(dim, M, coef, rho, &A, &b));
    PetscCall(VecDuplicate(b, &x)); PetscCall(VecSet(x, 0.0));

    PetscInt n_sub; IS *is_sub;
    PetscCall(make_subdomains(dim, M, S, &n_sub, &is_sub));

    KSP ksp; PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSetType(ksp, KSPCG));
    PetscCall(KSPSetNormType(ksp, KSP_NORM_PRECONDITIONED));
    PetscCall(KSPSetTolerances(ksp, 1e-8, 1e-50, PETSC_DEFAULT, 2000));
    PetscCall(KSPSetComputeSingularValues(ksp, PETSC_TRUE));

    /* build the inner BASIC PCASM (shared by scheme 0 and 3) */
    PC innerpc; PetscCall(PCCreate(PETSC_COMM_WORLD, &innerpc));
    PetscCall(PCSetType(innerpc, PCASM));
    PetscCall(PCASMSetType(innerpc, PC_ASM_BASIC));
    PetscCall(PCASMSetLocalSubdomains(innerpc, n_sub, is_sub, NULL));
    PetscCall(PCASMSetOverlap(innerpc, overlap));
    PetscCall(PCSetOperators(innerpc, A, A));
    PetscCall(PCSetUp(innerpc));
    {   /* sub-PC = preonly + (ICC(L) or Cholesky) */
        KSP *subksp; PetscInt nl, first;
        PetscCall(PCASMGetSubKSP(innerpc, &nl, &first, &subksp));
        for (PetscInt i = 0; i < nl; ++i) {
            PC sub; PetscCall(KSPSetType(subksp[i], KSPPREONLY));
            PetscCall(KSPGetPC(subksp[i], &sub));
            if (exact) { PetscCall(PCSetType(sub, PCCHOLESKY)); }
            else { PetscCall(PCSetType(sub, PCICC)); PetscCall(PCFactorSetLevels(sub, iccL)); }
        }
        PetscCall(PCSetUpOnBlocks(innerpc));
    }

    PetscReal nhat = 1.0;
    PetscCall(compute_nhat(innerpc, A, &nhat));

    if (scheme == 0) {
        PetscCall(KSPSetPC(ksp, innerpc));
    } else { /* scheme 3: PCSHELL sASM with D^{-1/2} multiplicity scaling */
        IS *is_over; PetscInt nl;
        PetscCall(PCASMGetLocalSubdomains(innerpc, &nl, &is_over, NULL));
        Vec mult; PetscCall(MatCreateVecs(A, &mult, NULL)); PetscCall(VecSet(mult, 0.0));
        for (PetscInt i = 0; i < nl; ++i) {
            const PetscInt *idx; PetscInt n;
            PetscCall(ISGetLocalSize(is_over[i], &n));
            PetscCall(ISGetIndices(is_over[i], &idx));
            for (PetscInt j = 0; j < n; ++j) PetscCall(VecSetValue(mult, idx[j], 1.0, ADD_VALUES));
            PetscCall(ISRestoreIndices(is_over[i], &idx));
        }
        PetscCall(VecAssemblyBegin(mult)); PetscCall(VecAssemblyEnd(mult));
        Vec invsqrt; PetscCall(VecDuplicate(mult, &invsqrt));
        PetscCall(VecCopy(mult, invsqrt));
        PetscCall(VecReciprocal(invsqrt)); PetscCall(VecSqrtAbs(invsqrt));
        PetscCall(VecDestroy(&mult));
        SASMCtx *c = (SASMCtx*)malloc(sizeof(SASMCtx));
        c->inner = innerpc; c->invsqrt = invsqrt;
        PetscCall(MatCreateVecs(A, &c->tmp, NULL));
        PC outer; PetscCall(KSPGetPC(ksp, &outer));
        PetscCall(PCSetType(outer, PCSHELL));
        PetscCall(PCShellSetContext(outer, c));
        PetscCall(PCShellSetApply(outer, sASMApply));
        PetscCall(PCShellSetDestroy(outer, sASMDestroy));
        PetscCall(PCShellSetName(outer, "sASM"));
    }

    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(KSPSolve(ksp, b, x));

    PetscInt iters; KSPConvergedReason reason; PetscReal smax, smin;
    PetscCall(KSPGetIterationNumber(ksp, &iters));
    PetscCall(KSPGetConvergedReason(ksp, &reason));
    PetscCall(KSPComputeExtremeSingularValues(ksp, &smax, &smin));

    /* omega = max_i lam_max(M_i^{-1} A_i) via subdomain blocks */
    double omega = 1.0;
    if (measure_omega && !exact) {
        IS *is_over; PetscInt nl;
        PetscCall(PCASMGetLocalSubdomains(innerpc, &nl, &is_over, NULL));
        Mat *subA;
        PetscCall(MatCreateSubMatrices(A, nl, is_over, is_over, MAT_INITIAL_MATRIX, &subA));
        for (PetscInt i = 0; i < nl; ++i) {
            KSP lk; PetscCall(KSPCreate(PETSC_COMM_SELF, &lk));
            PetscCall(KSPSetOperators(lk, subA[i], subA[i]));
            PetscCall(KSPSetType(lk, KSPCG));
            PetscCall(KSPSetComputeSingularValues(lk, PETSC_TRUE));
            PetscCall(KSPSetTolerances(lk, 1e-10, 1e-50, PETSC_DEFAULT, 500));
            PC lpc; PetscCall(KSPGetPC(lk, &lpc));
            PetscCall(PCSetType(lpc, PCICC)); PetscCall(PCFactorSetLevels(lpc, iccL));
            Vec lb, lx; PetscCall(MatCreateVecs(subA[i], &lx, &lb));
            PetscCall(VecSet(lb, 1.0)); PetscCall(VecSet(lx, 0.0));
            PetscCall(KSPSolve(lk, lb, lx));
            PetscReal lmax, lmin;
            PetscCall(KSPComputeExtremeSingularValues(lk, &lmax, &lmin));
            if (lmax > omega) omega = lmax;
            PetscCall(VecDestroy(&lb)); PetscCall(VecDestroy(&lx)); PetscCall(KSPDestroy(&lk));
        }
        PetscCall(MatDestroySubMatrices(nl, &subA));
    }

    PetscInt N; PetscCall(MatGetSize(A, &N, NULL));
    PetscReal kappa = (smin > 0) ? smax / smin : -1.0;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "[RESULT] dim=%d S=%d nx=%d coef=%s rho=%g scheme=%d exact=%d iccL=%d overlap=%d "
        "| N=%d iter=%d reason=%d lam_min=%.4e lam_max=%.4e kappa=%.4e Nhat=%.0f omega=%.4f\n",
        dim, S, M, coef, rho, scheme, (int)exact, iccL, overlap,
        (int)N, (int)iters, (int)reason, (double)smin, (double)smax,
        (double)kappa, (double)nhat, omega));

    PetscCall(KSPDestroy(&ksp));
    if (scheme == 0) { /* innerpc owned by ksp; do not double free */ }
    PetscCall(MatDestroy(&A)); PetscCall(VecDestroy(&b)); PetscCall(VecDestroy(&x));
    for (PetscInt i = 0; i < n_sub; ++i) PetscCall(ISDestroy(&is_sub[i]));
    free(is_sub);
    PetscCall(PetscFinalize());
    return 0;
}
