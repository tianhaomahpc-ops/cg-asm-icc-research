/* pure_petsc_demo.c
 *
 * Pure-PETSc reproducer (no MFEM, no MFEM bilinear forms, no Hypre).
 * We discretise -Delta u = 1 on the unit cube [0,1]^3 with
 *   u = 0   on x = 0                       (Dirichlet face)
 *   du/dn = 0   on the other five faces    (Neumann faces)
 * using a 7-point central finite-difference stencil on a uniform
 * (N+1)x(N+1)x(N+1) grid via DMDA.
 *
 * Neumann is enforced by simply DROPPING the missing-neighbour
 * contribution in the stencil (zero-flux  =>  u_outside = u_inside
 * cancels with the opposite-side term).  Dirichlet is enforced by
 * REPLACING the row at the boundary nodes with the identity row.
 * The resulting matrix is SPD.
 *
 * Everything (KSP type, PC type, ASM type, overlap, ICC levels,
 * sub-PC ordering / shift, etc.) is controlled via PETSc CLI options
 * exactly as in asm_demo.cpp, so the sweep script can drive both.
 *
 * Build:    see Makefile target  pure_petsc_demo
 * Run:      see sweep_pure.sh
 *
 * The whole point of this file is to verify, end-to-end,
 *   "MFEM is NOT in the loop and the iter-up-with-overlap behaviour
 *    appears nonetheless"
 * i.e. it is pure PCASM_BASIC + ICC behaviour, nothing to do with
 * how MFEM hands the matrix to PETSc.
 */
#include <petscksp.h>
#include <petscdmda.h>
#include <stdio.h>
#include <stdlib.h>

static int g_nx = 24;   /* elements per side; grid points = nx+1 */

/* Inject scheme overrides.  Mirrors asm_demo's InjectSchemeOptions. */
static PetscErrorCode InjectSchemeOptions(int scheme)
{
    if (scheme == 1) {
        PetscOptionsSetValue(NULL, "-ksp_type",    "gmres");
        PetscOptionsSetValue(NULL, "-pc_asm_type", "restrict");
    } else if (scheme == 2) {
        PetscOptionsSetValue(NULL, "-ksp_type",    "bcgs");
        PetscOptionsSetValue(NULL, "-pc_asm_type", "restrict");
    }
    /* scheme 0 leaves CG+BASIC; scheme 3 = PCSHELL sASM, handled below */
    return PETSC_SUCCESS;
}

/* --- sASM PCSHELL (scheme 3) -------------------------------------- */
typedef struct {
    PC  inner_pc;        /* PCASM BASIC + sub_pc=ICC                    */
    Vec inv_sqrt_mult;   /* D^{-1/2} where D = diag(multiplicity)        */
    Vec tmp;
} SASMCtx;

static PetscErrorCode sASMApply(PC pc, Vec r, Vec z)
{
    SASMCtx *ctx = NULL;
    PetscCall(PCShellGetContext(pc, (void**)&ctx));
    PetscCall(VecPointwiseMult(ctx->tmp, ctx->inv_sqrt_mult, r));
    PetscCall(PCApply(ctx->inner_pc, ctx->tmp, z));
    PetscCall(VecPointwiseMult(z, ctx->inv_sqrt_mult, z));
    return PETSC_SUCCESS;
}

static PetscErrorCode sASMDestroy(PC pc)
{
    SASMCtx *ctx = NULL;
    PetscCall(PCShellGetContext(pc, (void**)&ctx));
    if (ctx) {
        PetscCall(PCDestroy(&ctx->inner_pc));
        PetscCall(VecDestroy(&ctx->inv_sqrt_mult));
        PetscCall(VecDestroy(&ctx->tmp));
        free(ctx);
    }
    PetscCall(PCShellSetContext(pc, NULL));
    return PETSC_SUCCESS;
}

static PetscErrorCode InstallScaledASM(KSP ksp, Mat A,
                                       PetscInt overlap, PetscInt icc_levels)
{
    PC      inner = NULL;
    PetscMPIInt rank;

    PetscCall(MPI_Comm_rank(PetscObjectComm((PetscObject)A), &rank));

    PetscCall(PCCreate(PetscObjectComm((PetscObject)A), &inner));
    PetscCall(PCSetType(inner, PCASM));
    PetscCall(PCASMSetType(inner, PC_ASM_BASIC));
    PetscCall(PCASMSetOverlap(inner, overlap));
    PetscCall(PCSetOperators(inner, A, A));
    PetscCall(PCSetUp(inner));

    /* Configure sub-PCs to preonly + ICC(icc_levels) */
    {
        KSP     *subksp = NULL;
        PetscInt n_local = 0, first = 0, i;
        PetscCall(PCASMGetSubKSP(inner, &n_local, &first, &subksp));
        for (i = 0; i < n_local; ++i) {
            PC sub = NULL;
            PetscCall(KSPSetType(subksp[i], KSPPREONLY));
            PetscCall(KSPGetPC(subksp[i], &sub));
            PetscCall(PCSetType(sub, PCICC));
            PetscCall(PCFactorSetLevels(sub, icc_levels));
        }
        PetscCall(PCSetUpOnBlocks(inner));
    }

    /* Compute multiplicity m[k] = #subdomains containing k */
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
            PetscScalar *ones = (PetscScalar*)malloc(sizeof(PetscScalar) * n);
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
    PetscCall(VecReciprocal(invsqrt));
    PetscCall(VecSqrtAbs(invsqrt));
    {
        PetscReal mn, mx;
        PetscCall(VecMin(mult, NULL, &mn));
        PetscCall(VecMax(mult, NULL, &mx));
        if (rank == 0) {
            printf("[sASM] multiplicity range = [%d, %d]\n", (int)mn, (int)mx);
        }
    }
    PetscCall(VecDestroy(&mult));

    /* Wire PCSHELL on the outer KSP */
    PC outer = NULL;
    PetscCall(KSPGetPC(ksp, &outer));
    PetscCall(PCSetType(outer, PCSHELL));

    SASMCtx *ctx = (SASMCtx*)malloc(sizeof(SASMCtx));
    ctx->inner_pc      = inner;
    ctx->inv_sqrt_mult = invsqrt;
    PetscCall(MatCreateVecs(A, &ctx->tmp, NULL));

    PetscCall(PCShellSetContext(outer, ctx));
    PetscCall(PCShellSetApply  (outer, sASMApply));
    PetscCall(PCShellSetDestroy(outer, sASMDestroy));
    PetscCall(PCShellSetName   (outer, "sASM_pure_petsc"));
    PetscCall(PCSetUp(outer));
    if (rank == 0) {
        printf("[sASM] PCSHELL installed (overlap=%d, icc_levels=%d)\n",
               (int)overlap, (int)icc_levels);
    }
    return PETSC_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* Build the 7-point Laplacian on the (N+1)^3 DMDA grid with the BCs   */
/* described at the top.  Returns assembled SPD Mat A and RHS Vec b.   */
/* ------------------------------------------------------------------ */
static PetscErrorCode AssembleLaplace(DM da, Mat A, Vec b)
{
    PetscInt   xs, ys, zs, xm, ym, zm;
    PetscInt   N = g_nx;                          /* elements / side  */
    PetscInt   M = N + 1;                         /* grid points     */
    PetscReal  h = 1.0 / (PetscReal)N;
    PetscReal  hinv2 = 1.0 / (h * h);

    PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm));

    /* RHS = 1 everywhere except the Dirichlet face (i=0), where RHS = 0
     * so that the identity row gives u=0 directly.  */
    PetscScalar ***bb = NULL;
    PetscCall(DMDAVecGetArray(da, b, &bb));

    /* Loop over local rows */
    for (PetscInt k = zs; k < zs + zm; ++k) {
      for (PetscInt j = ys; j < ys + ym; ++j) {
        for (PetscInt i = xs; i < xs + xm; ++i) {
            MatStencil row = {.i = i, .j = j, .k = k};

            if (i == 0) {
                /* Dirichlet at x=0:  u_node = 0
                 * Row becomes:  1 * u_node = 0
                 * (RHS at this node is zeroed below.)
                 */
                PetscScalar v   = 1.0;
                MatStencil  col = row;
                PetscCall(MatSetValuesStencil(A, 1, &row, 1, &col, &v, INSERT_VALUES));
                bb[k][j][i] = 0.0;
                continue;
            }

            /* Otherwise: interior node OR Neumann boundary node.
             * Build a row with up to 6 off-diagonal entries.  For
             * each missing neighbour (out of bounds), we just SKIP
             * it.  That implements zero-flux Neumann naturally: the
             * (-1)*u_out term and the +1*u_in term cancel, so the
             * row gets one fewer "+1" on the diagonal.  Compare the
             * exact "ghost-cell reflection" Neumann discretisation:
             *
             *     u_outside = u_inside  =>  -u_out + u_in = 0
             *
             * which is what we get by simply not adding the missing
             * (-1) and the matching +1 to the diagonal.            */
            MatStencil  col[7];
            PetscScalar v  [7];
            PetscInt    nc = 0;
            PetscScalar diag = 0.0;

            #define ADDN(di, dj, dk)                          \
                do {                                          \
                    PetscInt ni = i + (di);                   \
                    PetscInt nj = j + (dj);                   \
                    PetscInt nk = k + (dk);                   \
                    if (ni >= 0 && ni < M &&                  \
                        nj >= 0 && nj < M &&                  \
                        nk >= 0 && nk < M) {                  \
                        col[nc].i = ni; col[nc].j = nj;       \
                        col[nc].k = nk;                       \
                        v  [nc]   = -hinv2;                   \
                        ++nc;                                 \
                        diag     += hinv2;                    \
                    }                                         \
                } while (0)

            ADDN(-1, 0, 0);
            ADDN(+1, 0, 0);
            ADDN(0, -1, 0);
            ADDN(0, +1, 0);
            ADDN(0, 0, -1);
            ADDN(0, 0, +1);
            #undef ADDN

            col[nc] = row;
            v  [nc] = diag;
            ++nc;

            PetscCall(MatSetValuesStencil(A, 1, &row, nc, col, v, INSERT_VALUES));
            bb[k][j][i] = 1.0;
        }
      }
    }
    PetscCall(DMDAVecRestoreArray(da, b, &bb));
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd  (A, MAT_FINAL_ASSEMBLY));

    /* The matrix is SPD (the Dirichlet identity row is trivially SPD).
     * Flag it; the AIJ flag triggers PETSc's SBAIJ fast path on ICC.   */
    PetscCall(MatSetOption(A, MAT_SYMMETRIC,        PETSC_TRUE));
    PetscCall(MatSetOption(A, MAT_SYMMETRY_ETERNAL, PETSC_TRUE));
    PetscCall(MatSetOption(A, MAT_SPD,              PETSC_TRUE));

    return PETSC_SUCCESS;
}

int main(int argc, char **argv)
{
    PetscMPIInt rank, size;
    int         scheme = 0;
    int         nx     = 24;
    /* Extract our own knobs (-scheme, -nx) BEFORE PetscInitialize sees
     * argv, the same way asm_demo does it.  PETSc's options system
     * with -options_left no will silently ignore them after this.    */
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
    g_nx = nx;

    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
    PetscOptionsSetValue(NULL, "-options_left", "no");
    InjectSchemeOptions(scheme);

    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

    if (rank == 0) {
        const char *sn[] = {"CG+ASM_BASIC", "GMRES+ASM_RESTRICT (RAS)",
                            "BCGS+ASM_RESTRICT (RAS)", "CG+sASM (PCSHELL)"};
        printf("================================================\n");
        printf("  pure_petsc_demo  ranks=%d  nx=%d  scheme=%d:%s\n",
               size, nx, scheme,
               (scheme>=0 && scheme<=3) ? sn[scheme] : "?");
        printf("================================================\n");
    }

    /* Build DMDA: (nx+1)^3 grid, 1 DOF per node, star stencil width 1 */
    DM da;
    PetscCall(DMDACreate3d(PETSC_COMM_WORLD,
                           DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                           DMDA_STENCIL_STAR,
                           nx + 1, nx + 1, nx + 1,
                           PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
                           1, 1, NULL, NULL, NULL, &da));
    PetscCall(DMSetUp(da));

    Mat A;
    Vec b, x;
    PetscCall(DMCreateMatrix(da, &A));
    PetscCall(DMCreateGlobalVector(da, &b));
    PetscCall(VecDuplicate(b, &x));
    PetscCall(VecSet(x, 0.0));

    PetscCall(AssembleLaplace(da, A, b));

    if (rank == 0) {
        PetscInt m, n;
        MatInfo  info;
        PetscCall(MatGetSize(A, &m, &n));
        PetscCall(MatGetInfo(A, MAT_GLOBAL_SUM, &info));
        printf("[MAT] size=%d x %d, nz_used=%d\n",
               (int)m, (int)n, (int)info.nz_used);
    } else {
        MatInfo  info;
        PetscCall(MatGetInfo(A, MAT_GLOBAL_SUM, &info));   /* collective */
        (void)info;
    }

    KSP ksp;
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSetType(ksp, KSPCG));                    /* default; CLI overrides */
    PetscCall(KSPSetTolerances(ksp, 1e-6, 1e-12, PETSC_DEFAULT, 1000));
    PetscCall(KSPSetFromOptions(ksp));

    if (scheme == 3) {
        PetscInt overlap = 0, icc_lev = 0;
        PetscOptionsGetInt(NULL, NULL, "-pc_asm_overlap",       &overlap, NULL);
        PetscOptionsGetInt(NULL, NULL, "-sub_pc_factor_levels", &icc_lev, NULL);
        PetscCall(InstallScaledASM(ksp, A, overlap, icc_lev));
    }

    double t0 = MPI_Wtime();
    PetscCall(KSPSolve(ksp, b, x));
    double t1 = MPI_Wtime();

    PetscInt           iters = 0;
    KSPConvergedReason reason;
    PetscReal          rnorm = 0.0;
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
    PetscCall(DMDestroy(&da));
    PetscCall(PetscFinalize());
    return 0;
}
