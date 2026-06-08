/* pure_petsc_load.c
 *
 * Pure-PETSc KSP driver that LOADS the MFEM-assembled matrix and RHS
 * from binary files dumped by asm_demo.cpp.  Because the file already
 * contains the MFEM METIS partition (rows are distributed exactly the
 * same way MFEM placed them), running this binary with the SAME number
 * of ranks makes "matrix + partition" bit-identical to MFEM's run, so
 * any iter difference between the two implementations now collapses to
 * "do their PETSc KSP options agree?", which we control end-to-end.
 *
 * Usage:
 *   mpirun -n N ./pure_petsc_load -nx 24 -scheme 0 \
 *     -ksp_type cg -ksp_norm_type preconditioned ...
 *
 * Requires that  mfem_A_nxNN_nN.petscbin  and
 *                mfem_b_nxNN_nN.petscbin
 * already exist (run asm_demo with the same nx, ranks first).
 */
#include <petscksp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- diagonal-weighted ASM PCSHELL (mirrors asm_demo.cpp) -----------
 *   apply(r) = [post? W .] M_BASIC^{-1} [pre? W .] r,   W = diag(w)
 *   weight_mode 0: W=D^-1/2, pre=post=1  (sASM, sym)        [scheme 3/4]
 *   weight_mode 1: W=D^-1,   pre=post=1  (sym, over-norm)   [scheme 5]
 *   weight_mode 2: W=D^-1,   pre=0,post=1 (non-symmetric)   [scheme 6]
 */
typedef struct {
    PC  inner_pc;
    Vec w;          /* D^-1/2 or D^-1 */
    Vec tmp;
    int pre;
    int post;
} SASMCtx;

static PetscErrorCode sASMApply(PC pc, Vec r, Vec z)
{
    SASMCtx *ctx = NULL;
    PetscCall(PCShellGetContext(pc, (void**)&ctx));
    if (ctx->pre) PetscCall(VecPointwiseMult(ctx->tmp, ctx->w, r));
    else          PetscCall(VecCopy(r, ctx->tmp));
    PetscCall(PCApply(ctx->inner_pc, ctx->tmp, z));
    if (ctx->post) PetscCall(VecPointwiseMult(z, ctx->w, z));
    return PETSC_SUCCESS;
}
static PetscErrorCode sASMDestroy(PC pc)
{
    SASMCtx *ctx = NULL;
    PetscCall(PCShellGetContext(pc, (void**)&ctx));
    if (ctx) {
        PetscCall(PCDestroy(&ctx->inner_pc));
        PetscCall(VecDestroy(&ctx->w));
        PetscCall(VecDestroy(&ctx->tmp));
        free(ctx);
    }
    PetscCall(PCShellSetContext(pc, NULL));
    return PETSC_SUCCESS;
}
static PetscErrorCode InstallScaledASM(KSP ksp, Mat A,
                                       PetscInt overlap, PetscInt icc_levels,
                                       PetscInt cheby_deg, int weight_mode)
{
    // cheby_deg == 0 : block solve = preonly + ICC(icc_levels)   [scheme 3]
    // cheby_deg >= 1 : block solve = cheby_deg Chebyshev steps over ICC,
    //                  frozen eigenvalue bounds -> fixed SPD linear op  [scheme 4]
    PC      inner = NULL;
    PetscMPIInt rank;
    PetscCall(MPI_Comm_rank(PetscObjectComm((PetscObject)A), &rank));
    PetscCall(PCCreate(PetscObjectComm((PetscObject)A), &inner));
    PetscCall(PCSetType(inner, PCASM));
    PetscCall(PCASMSetType(inner, PC_ASM_BASIC));
    PetscCall(PCASMSetOverlap(inner, overlap));
    PetscCall(PCSetOperators(inner, A, A));
    PetscCall(PCSetUp(inner));
    {
        KSP     *subksp = NULL;
        PetscInt n_local = 0, first = 0;
        PetscCall(PCASMGetSubKSP(inner, &n_local, &first, &subksp));
        for (PetscInt i = 0; i < n_local; ++i) {
            PC sub = NULL;
            if (cheby_deg <= 0) {
                PetscCall(KSPSetType(subksp[i], KSPPREONLY));
                PetscCall(KSPGetPC(subksp[i], &sub));
                PetscCall(PCSetType(sub, PCICC));
                PetscCall(PCFactorSetLevels(sub, icc_levels));
            } else {
                PetscCall(KSPSetType(subksp[i], KSPCHEBYSHEV));
                PetscCall(KSPSetTolerances(subksp[i], PETSC_DEFAULT,
                          PETSC_DEFAULT, PETSC_DEFAULT, cheby_deg));
                PetscCall(KSPSetNormType(subksp[i], KSP_NORM_NONE));
                PetscCall(KSPSetInitialGuessNonzero(subksp[i], PETSC_FALSE));
                PetscCall(KSPChebyshevEstEigSet(subksp[i], 0.0, 0.1, 0.0, 1.1));
                PetscCall(KSPGetPC(subksp[i], &sub));
                PetscCall(PCSetType(sub, PCICC));
                PetscCall(PCFactorSetLevels(sub, icc_levels));
            }
        }
        PetscCall(PCSetUpOnBlocks(inner));
    }
    Vec mult = NULL, invsqrt = NULL;
    PetscCall(MatCreateVecs(A, &mult, NULL));
    PetscCall(VecSet(mult, 0.0));
    {
        PetscInt n_local = 0;
        IS *is_full = NULL, *is_local_only = NULL;
        PetscCall(PCASMGetLocalSubdomains(inner, &n_local, &is_full, &is_local_only));
        for (PetscInt i = 0; i < n_local; ++i) {
            const PetscInt *idx = NULL;
            PetscInt        n   = 0;
            PetscCall(ISGetLocalSize(is_full[i], &n));
            PetscCall(ISGetIndices(is_full[i], &idx));
            PetscScalar *ones = (PetscScalar*)malloc(sizeof(PetscScalar) * (n>0?n:1));
            for (PetscInt j = 0; j < n; ++j) ones[j] = 1.0;
            PetscCall(VecSetValues(mult, n, idx, ones, ADD_VALUES));
            free(ones);
            PetscCall(ISRestoreIndices(is_full[i], &idx));
        }
        PetscCall(VecAssemblyBegin(mult));
        PetscCall(VecAssemblyEnd(mult));
    }
    PetscCall(VecDuplicate(mult, &invsqrt));
    PetscCall(VecCopy(mult, invsqrt));
    PetscCall(VecReciprocal(invsqrt));                       /* 1/m[k] (D^-1) */
    if (weight_mode == 0) PetscCall(VecSqrtAbs(invsqrt));    /* -> 1/sqrt(m[k]) for sASM */
    {
        PetscReal mn, mx;
        PetscCall(VecMin(mult, NULL, &mn));
        PetscCall(VecMax(mult, NULL, &mx));
        if (rank == 0) {
            printf("[sASM] multiplicity range = [%d, %d]\n", (int)mn, (int)mx);
        }
    }
    PetscCall(VecDestroy(&mult));
    PC outer = NULL;
    PetscCall(KSPGetPC(ksp, &outer));
    PetscCall(PCSetType(outer, PCSHELL));
    SASMCtx *ctx = (SASMCtx*)malloc(sizeof(SASMCtx));
    ctx->inner_pc = inner;
    ctx->w        = invsqrt;                       /* D^-1/2 (mode 0) or D^-1 (mode 1/2) */
    ctx->pre      = (weight_mode == 2) ? 0 : 1;    /* mode 2 = left/post only */
    ctx->post     = 1;
    PetscCall(MatCreateVecs(A, &ctx->tmp, NULL));
    PetscCall(PCShellSetContext(outer, ctx));
    PetscCall(PCShellSetApply  (outer, sASMApply));
    PetscCall(PCShellSetDestroy(outer, sASMDestroy));
    PetscCall(PCShellSetName   (outer, "weighted_ASM"));
    PetscCall(PCSetUp(outer));
    if (rank == 0) {
        printf("[sASM] PCSHELL installed (overlap=%d, icc_levels=%d)\n",
               (int)overlap, (int)icc_levels);
    }
    return PETSC_SUCCESS;
}

static void InjectSchemeOptions(int scheme)
{
    if (scheme == 1) {
        PetscOptionsSetValue(NULL, "-ksp_type",    "gmres");
        PetscOptionsSetValue(NULL, "-pc_asm_type", "restrict");
    } else if (scheme == 2) {
        PetscOptionsSetValue(NULL, "-ksp_type",    "bcgs");
        PetscOptionsSetValue(NULL, "-pc_asm_type", "restrict");
    }
}

int main(int argc, char **argv)
{
    PetscMPIInt rank, size;
    int  scheme = 0, nx = 24;

    /* Strip our knobs from argv. */
    {
        int out = 1;
        for (int i = 1; i < argc; ++i) {
            const char *a = argv[i];
            if (i + 1 < argc && (!strcmp(a, "-scheme") || !strcmp(a, "--scheme"))) {
                scheme = atoi(argv[++i]); continue;
            }
            if (i + 1 < argc && (!strcmp(a, "-nx") || !strcmp(a, "--nx"))) {
                nx = atoi(argv[++i]); continue;
            }
            argv[out++] = argv[i];
        }
        argc = out;
    }

    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
    PetscOptionsSetValue(NULL, "-options_left", "no");
    InjectSchemeOptions(scheme);

    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

    char mfn[256], bfn[256];
    snprintf(mfn, sizeof(mfn), "mfem_A_nx%d_n%d.petscbin", nx, (int)size);
    snprintf(bfn, sizeof(bfn), "mfem_b_nx%d_n%d.petscbin", nx, (int)size);
    if (rank == 0) {
        const char *sn[] = {"CG+ASM_BASIC", "GMRES+ASM_RESTRICT (RAS)",
                            "BCGS+ASM_RESTRICT (RAS)", "CG+sASM (D^-1/2 .. D^-1/2)",
                            "CG+sASM+Chebyshev",
                            "D^-1 BASIC D^-1 (sym, over-norm)",
                            "D^-1 BASIC (non-symmetric)"};
        printf("================================================\n");
        printf("  pure_petsc_load  ranks=%d nx=%d scheme=%d:%s\n",
               (int)size, nx, scheme,
               (scheme>=0 && scheme<=6) ? sn[scheme] : "?");
        printf("  matrix file = %s\n  rhs    file = %s\n", mfn, bfn);
        printf("================================================\n");
    }

    /* Read per-rank layout file so we can PRESET the local row count
     * before MatLoad and thereby preserve MFEM's METIS distribution.
     * Without this, PETSc's MatLoad defaults to PETSC_DECIDE (i.e.
     * contiguous equal chunks), silently re-partitioning the rows. */
    char lfn[256];
    snprintf(lfn, sizeof(lfn), "mfem_layout_nx%d_n%d.bin", nx, (int)size);
    PetscInt my_mloc = PETSC_DECIDE, my_nloc = PETSC_DECIDE;
    if (rank == 0) {
        FILE *f = fopen(lfn, "rb");
        if (!f) {
            fprintf(stderr, "ERROR: cannot open %s. Run asm_demo first.\n", lfn);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        int nr;
        fread(&nr, sizeof(int), 1, f);
        if (nr != (int)size) {
            fprintf(stderr, "ERROR: layout file is for %d ranks, not %d\n",
                    nr, (int)size);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        PetscInt *mlocs = (PetscInt*)malloc(sizeof(PetscInt) * size);
        PetscInt *nlocs = (PetscInt*)malloc(sizeof(PetscInt) * size);
        fread(mlocs, sizeof(PetscInt), size, f);
        fread(nlocs, sizeof(PetscInt), size, f);
        fclose(f);
        printf("[LAYOUT] read per-rank rows = [");
        for (int r = 0; r < size; ++r) {
            printf("%lld%s", (long long)mlocs[r], r+1<size ? "," : "");
        }
        printf("]\n");
        /* Scatter local sizes back to each rank. */
        MPI_Scatter(mlocs, 1, MPIU_INT, &my_mloc, 1, MPIU_INT, 0,
                    PETSC_COMM_WORLD);
        MPI_Scatter(nlocs, 1, MPIU_INT, &my_nloc, 1, MPIU_INT, 0,
                    PETSC_COMM_WORLD);
        free(mlocs); free(nlocs);
    } else {
        MPI_Scatter(NULL, 1, MPIU_INT, &my_mloc, 1, MPIU_INT, 0,
                    PETSC_COMM_WORLD);
        MPI_Scatter(NULL, 1, MPIU_INT, &my_nloc, 1, MPIU_INT, 0,
                    PETSC_COMM_WORLD);
    }

    /* Load matrix and RHS, preserving MFEM's per-rank local size. */
    Mat A;
    Vec b, x;
    PetscViewer vw;
    PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
    PetscCall(MatSetType(A, MATAIJ));
    PetscCall(MatSetSizes(A, my_mloc, my_nloc, PETSC_DETERMINE, PETSC_DETERMINE));
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, mfn, FILE_MODE_READ, &vw));
    PetscCall(MatLoad(A, vw));
    PetscCall(PetscViewerDestroy(&vw));
    PetscCall(VecCreate(PETSC_COMM_WORLD, &b));
    PetscCall(VecSetType(b, VECMPI));
    PetscCall(VecSetSizes(b, my_mloc, PETSC_DETERMINE));
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, bfn, FILE_MODE_READ, &vw));
    PetscCall(VecLoad(b, vw));
    PetscCall(PetscViewerDestroy(&vw));
    PetscCall(VecDuplicate(b, &x));
    PetscCall(VecSet(x, 0.0));

    /* Honour the same algebraic flags pure_petsc_fem sets.  The matrix
     * really IS symmetric SPD here (we are loading the same A that the
     * MFEM run built and verified). */
    PetscCall(MatSetOption(A, MAT_SYMMETRIC,        PETSC_TRUE));
    PetscCall(MatSetOption(A, MAT_SYMMETRY_ETERNAL, PETSC_TRUE));
    PetscCall(MatSetOption(A, MAT_SPD,              PETSC_TRUE));

    {
        PetscInt m, n;
        MatInfo  info;
        PetscCall(MatGetSize(A, &m, &n));
        PetscCall(MatGetInfo(A, MAT_GLOBAL_SUM, &info));
        if (rank == 0) {
            printf("[MAT] loaded size=%d x %d, nz_used=%d (%.2f nnz/row)\n",
                   (int)m, (int)n, (int)info.nz_used,
                   (double)info.nz_used / (double)m);
        }
        /* Operator fingerprint with the SAME deterministic vector used
         * in asm_demo / pure_petsc_fem.  All three must match. */
        Vec xv, yv;
        PetscCall(MatCreateVecs(A, &xv, &yv));
        PetscInt mloc, mlo;
        PetscCall(VecGetLocalSize(xv, &mloc));
        PetscCall(VecGetOwnershipRange(xv, &mlo, NULL));
        PetscScalar *arr;
        PetscCall(VecGetArray(xv, &arr));
        for (PetscInt p = 0; p < mloc; ++p) {
            PetscInt g = mlo + p;
            arr[p] = cos((double)g * 0.012345) + sin((double)g * 0.054321);
        }
        PetscCall(VecRestoreArray(xv, &arr));
        PetscCall(MatMult(A, xv, yv));
        PetscReal nxn, nyn;
        PetscCall(VecNorm(xv, NORM_2, &nxn));
        PetscCall(VecNorm(yv, NORM_2, &nyn));
        if (rank == 0) {
            printf("[FINGERPRINT] ||x||=%.6e ||Ax||=%.6e\n",
                   (double)nxn, (double)nyn);
        }
        PetscCall(VecDestroy(&xv));
        PetscCall(VecDestroy(&yv));
    }

    KSP ksp;
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSetType(ksp, KSPCG));
    PetscCall(KSPSetTolerances(ksp, 1e-6, 1e-12, PETSC_DEFAULT, 1000));
    PetscCall(KSPSetFromOptions(ksp));

    if (scheme >= 3 && scheme <= 6) {
        PetscInt overlap = 0, icc_lev = 0;
        PetscOptionsGetInt(NULL, NULL, "-pc_asm_overlap",       &overlap, NULL);
        PetscOptionsGetInt(NULL, NULL, "-sub_pc_factor_levels", &icc_lev, NULL);
        PetscInt cheby_deg = 0;
        if (scheme == 4) {
            cheby_deg = 2;
            for (int i = 1; i < argc; ++i)
                if (!strcmp(argv[i], "-localcheby") && i + 1 < argc)
                    cheby_deg = atoi(argv[i+1]);
        }
        int weight_mode = (scheme == 5) ? 1 : (scheme == 6) ? 2 : 0;
        PetscCall(InstallScaledASM(ksp, A, overlap, icc_lev, cheby_deg, weight_mode));
    }

    double t0 = MPI_Wtime();
    PetscCall(KSPSolve(ksp, b, x));
    double t1 = MPI_Wtime();
    PetscInt iters = 0;
    KSPConvergedReason reason;
    PetscReal rnorm = 0;
    PetscCall(KSPGetIterationNumber(ksp, &iters));
    PetscCall(KSPGetConvergedReason(ksp, &reason));
    PetscCall(KSPGetResidualNorm(ksp, &rnorm));
    if (rank == 0) {
        printf("[RESULT] scheme=%d iters=%d reason=%d pnorm=%.3e time=%.3f s\n",
               scheme, (int)iters, (int)reason, (double)rnorm, t1 - t0);
    }

    PetscCall(KSPDestroy(&ksp));
    PetscCall(MatDestroy(&A));
    PetscCall(VecDestroy(&b));
    PetscCall(VecDestroy(&x));
    PetscCall(PetscFinalize());
    return 0;
}
