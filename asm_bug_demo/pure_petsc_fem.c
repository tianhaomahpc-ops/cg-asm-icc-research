/* pure_petsc_fem.c   --   pure-PETSc P1 LINEAR tet FEM
 *
 * Solves -div(grad u) = 1 on the unit cube [0,1]^3 with:
 *   - Dirichlet u = 0  on the x = 0 face
 *   - homogeneous Neumann on the other 5 faces
 * using EXACTLY THE SAME MESH AND THE SAME ELEMENT TYPE as the MFEM
 * side (asm_demo.cpp built with Element::TETRAHEDRON):
 *
 *   * Same (nx+1)^3 vertex grid on [0,1]^3
 *   * Each hex [ex,ex+1] x [ey,ey+1] x [ez,ez+1] is decomposed into
 *     the SAME 6 tetrahedra used by MFEM's Mesh::AddHexAsTets, all
 *     sharing the (vertex 0)-(vertex 6) main diagonal:
 *
 *         hex_to_tet = { {0,1,2,6}, {0,5,1,6}, {0,4,5,6},
 *                        {0,2,3,6}, {0,3,7,6}, {0,7,4,6} }
 *
 *     with the same local hex vertex order MFEM uses
 *     (mesh.cpp:1942-1949):
 *         0:(0,0,0)  1:(1,0,0)  2:(1,1,0)  3:(0,1,0)
 *         4:(0,0,1)  5:(1,0,1)  6:(1,1,1)  7:(0,1,1)
 *
 * NO MFEM, NO Hypre.  Local 4x4 tet stiffness is computed from the
 * standard linear-element formula  K^{loc}_{ij} = V (grad phi_i,
 * grad phi_j) with the (constant) gradients obtained via the
 * Jacobian B^{-T} g_i,  V = |det B| / 6.
 *
 * NOTE on partition: MFEM's ParMesh uses METIS, while we use DMDA's
 * cuboid decomposition; the multiplicity profile (and hence the
 * absolute iter count) therefore differs slightly between the two
 * sides, but the SAME mesh + SAME FE + SAME BC means the matrix
 * sparsity and assembled entry VALUES are identical (modulo round-
 * off).  The qualitative scheme-vs-scheme trend is the strict
 * answer; absolute numbers differ only because of partition shape.
 */
#include <petscksp.h>
#include <petscdmda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ----- MFEM-matched local data ----------------------------------- */

/* Local vertex offsets inside one hex, MFEM order (mesh.cpp:1942-) */
static const int HEX_OFF[8][3] = {
    {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0},
    {0,0,1}, {1,0,1}, {1,1,1}, {0,1,1}
};

/* MFEM Mesh::AddHexAsTets table (mesh.cpp:986-991) */
static const int HEX_TO_TET[6][4] = {
    {0,1,2,6}, {0,5,1,6}, {0,4,5,6},
    {0,2,3,6}, {0,3,7,6}, {0,7,4,6}
};

/* ----- sASM PCSHELL (scheme 3) -- same as before ----------------- */
typedef struct {
    PC  inner_pc;
    Vec inv_sqrt_mult;
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
    {
        KSP     *subksp = NULL;
        PetscInt n_local = 0, first = 0;
        PetscCall(PCASMGetSubKSP(inner, &n_local, &first, &subksp));
        for (PetscInt i = 0; i < n_local; ++i) {
            PC sub = NULL;
            PetscCall(KSPSetType(subksp[i], KSPPREONLY));
            PetscCall(KSPGetPC(subksp[i], &sub));
            PetscCall(PCSetType(sub, PCICC));
            PetscCall(PCFactorSetLevels(sub, icc_levels));
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
    PetscCall(PCShellSetName   (outer, "sASM_pure_tet_p1"));
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

/* ------------------------------------------------------------------ */
/* TetStiffness: 4-vertex tet at v[4][3] -> 4x4 local stiffness K.     */
/* Uses constant gradients ∇phi_i = B^{-T} g_i, with V = |det B|/6.    */
/* g_0 = (-1,-1,-1), g_1 = (1,0,0), g_2 = (0,1,0), g_3 = (0,0,1).      */
/* ------------------------------------------------------------------ */
static void TetStiffness(const double v[4][3], double K[4][4])
{
    /* B[d][k] = (v[k+1] - v[0])_d                                    */
    double B[3][3];
    for (int d = 0; d < 3; ++d) {
        B[d][0] = v[1][d] - v[0][d];
        B[d][1] = v[2][d] - v[0][d];
        B[d][2] = v[3][d] - v[0][d];
    }
    double detB = B[0][0]*(B[1][1]*B[2][2] - B[1][2]*B[2][1])
                - B[0][1]*(B[1][0]*B[2][2] - B[1][2]*B[2][0])
                + B[0][2]*(B[1][0]*B[2][1] - B[1][1]*B[2][0]);
    double V = fabs(detB) / 6.0;
    double inv_det = 1.0 / detB;
    /* adj(B)^T / det(B) = B^{-1}.  We need B^{-T}.                   */
    double Binv[3][3];
    Binv[0][0] = (B[1][1]*B[2][2] - B[1][2]*B[2][1]) * inv_det;
    Binv[0][1] = (B[0][2]*B[2][1] - B[0][1]*B[2][2]) * inv_det;
    Binv[0][2] = (B[0][1]*B[1][2] - B[0][2]*B[1][1]) * inv_det;
    Binv[1][0] = (B[1][2]*B[2][0] - B[1][0]*B[2][2]) * inv_det;
    Binv[1][1] = (B[0][0]*B[2][2] - B[0][2]*B[2][0]) * inv_det;
    Binv[1][2] = (B[0][2]*B[1][0] - B[0][0]*B[1][2]) * inv_det;
    Binv[2][0] = (B[1][0]*B[2][1] - B[1][1]*B[2][0]) * inv_det;
    Binv[2][1] = (B[0][1]*B[2][0] - B[0][0]*B[2][1]) * inv_det;
    Binv[2][2] = (B[0][0]*B[1][1] - B[0][1]*B[1][0]) * inv_det;
    /* grad[i][d] = sum_k Binv[k][d] * g_i[k]                         */
    double grad[4][3];
    for (int d = 0; d < 3; ++d) {
        grad[0][d] = -(Binv[0][d] + Binv[1][d] + Binv[2][d]);
        grad[1][d] = Binv[0][d];
        grad[2][d] = Binv[1][d];
        grad[3][d] = Binv[2][d];
    }
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        K[i][j] = V * (grad[i][0]*grad[j][0]
                     + grad[i][1]*grad[j][1]
                     + grad[i][2]*grad[j][2]);
      }
    }
}

/* ------------------------------------------------------------------ */
/* Assemble global stiffness K (P1 tets) + load b (f=1), then enforce */
/* Dirichlet u=0 on the x=0 face via MatZeroRowsColumnsStencil.        */
/* ------------------------------------------------------------------ */
static PetscErrorCode AssembleFEM_Tet(DM da, PetscInt nx, Mat A, Vec b)
{
    PetscInt    M = nx + 1;
    PetscReal   h = 1.0 / (PetscReal)nx;
    PetscInt    xs, ys, zs, xm, ym, zm;
    PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm));

    PetscInt exs = xs, eys = ys, ezs = zs;
    PetscInt exm = (xs + xm == M) ? (xm - 1) : xm;
    PetscInt eym = (ys + ym == M) ? (ym - 1) : ym;
    PetscInt ezm = (zs + zm == M) ? (zm - 1) : zm;
    if (exm < 0) exm = 0;
    if (eym < 0) eym = 0;
    if (ezm < 0) ezm = 0;

    /* RHS volume per vertex per tet for f=1:
     *   tet vol = h^3/6, split equally among 4 vertices = h^3/24      */
    const double rhs_per_tet_vertex = (h * h * h) / 24.0;

    /* We accumulate RHS into a LOCAL vector (with ghost vertices),
     * then DMLocalToGlobal( ADD_VALUES ) deposits ghost contributions
     * onto their owners.  Matrix uses MatSetValuesStencil which has
     * built-in routing.                                                */
    Vec b_local;
    PetscCall(DMCreateLocalVector(da, &b_local));
    PetscCall(VecSet(b_local, 0.0));
    PetscScalar ***b_arr;
    PetscCall(DMDAVecGetArray(da, b_local, &b_arr));

    for (PetscInt ez = ezs; ez < ezs + ezm; ++ez) {
      for (PetscInt ey = eys; ey < eys + eym; ++ey) {
        for (PetscInt ex = exs; ex < exs + exm; ++ex) {
            /* Compute the 8 hex-vertex positions and their stencils */
            double vhex[8][3];
            MatStencil shex[8];
            for (int l = 0; l < 8; ++l) {
                vhex[l][0] = (ex + HEX_OFF[l][0]) * h;
                vhex[l][1] = (ey + HEX_OFF[l][1]) * h;
                vhex[l][2] = (ez + HEX_OFF[l][2]) * h;
                shex[l].i  = ex + HEX_OFF[l][0];
                shex[l].j  = ey + HEX_OFF[l][1];
                shex[l].k  = ez + HEX_OFF[l][2];
            }
            for (int t = 0; t < 6; ++t) {
                double      vtet[4][3];
                MatStencil  stet[4];
                int         hex_locs[4];
                for (int l = 0; l < 4; ++l) {
                    int hex_local = HEX_TO_TET[t][l];
                    hex_locs[l]   = hex_local;
                    vtet[l][0]    = vhex[hex_local][0];
                    vtet[l][1]    = vhex[hex_local][1];
                    vtet[l][2]    = vhex[hex_local][2];
                    stet[l]       = shex[hex_local];
                }
                double      K[4][4];
                TetStiffness(vtet, K);

                PetscScalar Kflat[16];
                for (int i = 0; i < 4; ++i)
                  for (int j = 0; j < 4; ++j)
                    Kflat[i*4 + j] = K[i][j];

                PetscCall(MatSetValuesStencil(A, 4, stet, 4, stet,
                                              Kflat, ADD_VALUES));

                /* RHS: write into LOCAL vector, ghosts OK */
                for (int l = 0; l < 4; ++l) {
                    PetscInt vi = stet[l].i;
                    PetscInt vj = stet[l].j;
                    PetscInt vk = stet[l].k;
                    b_arr[vk][vj][vi] += rhs_per_tet_vertex;
                }
            }
        }
      }
    }

    PetscCall(DMDAVecRestoreArray(da, b_local, &b_arr));

    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd  (A, MAT_FINAL_ASSEMBLY));

    /* Scatter ghost contributions into the global b vector */
    PetscCall(DMLocalToGlobalBegin(da, b_local, ADD_VALUES, b));
    PetscCall(DMLocalToGlobalEnd  (da, b_local, ADD_VALUES, b));
    PetscCall(VecDestroy(&b_local));

    PetscCall(MatSetOption(A, MAT_SYMMETRIC,        PETSC_TRUE));
    PetscCall(MatSetOption(A, MAT_SYMMETRY_ETERNAL, PETSC_TRUE));

    /* ----- Apply Dirichlet u = 0 on x = 0 face --------------------- */
    PetscInt    n_bc = 0;
    MatStencil *bc_st = NULL;
    if (xs == 0) {
        n_bc = ym * zm;
        bc_st = (MatStencil*)malloc(sizeof(MatStencil) * (n_bc > 0 ? n_bc : 1));
        PetscInt idx = 0;
        for (PetscInt k = zs; k < zs + zm; ++k) {
          for (PetscInt j = ys; j < ys + ym; ++j) {
              bc_st[idx].i = 0;
              bc_st[idx].j = j;
              bc_st[idx].k = k;
              ++idx;
          }
        }
    }
    Vec u_bc;
    PetscCall(VecDuplicate(b, &u_bc));
    PetscCall(VecSet(u_bc, 0.0));
    PetscCall(MatZeroRowsColumnsStencil(A, n_bc, bc_st, 1.0, u_bc, b));
    PetscCall(VecDestroy(&u_bc));
    if (bc_st) free(bc_st);

    PetscCall(MatSetOption(A, MAT_SPD, PETSC_TRUE));
    PetscCall(MatFilter(A, 1e-14, PETSC_TRUE, PETSC_TRUE));
    return PETSC_SUCCESS;
}

int main(int argc, char **argv)
{
    PetscMPIInt rank, size;
    int         scheme = 0;
    int         nx     = 24;

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
    if (rank == 0) {
        const char *sn[] = {"CG+ASM_BASIC", "GMRES+ASM_RESTRICT (RAS)",
                            "BCGS+ASM_RESTRICT (RAS)", "CG+sASM (PCSHELL)"};
        printf("================================================\n");
        printf("  pure_petsc_fem  P1 TET  ranks=%d nx=%d scheme=%d:%s\n",
               size, nx, scheme,
               (scheme>=0 && scheme<=3) ? sn[scheme] : "?");
        printf("================================================\n");
    }

    /* DMDA over vertex grid (nx+1)^3 with BOX stencil width 1 to
     * accommodate the tet-mesh's worst-case 27-vertex neighbourhood. */
    DM da;
    PetscCall(DMDACreate3d(PETSC_COMM_WORLD,
                           DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                           DMDA_STENCIL_BOX,
                           nx + 1, nx + 1, nx + 1,
                           PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
                           1, 1, NULL, NULL, NULL, &da));
    PetscCall(DMSetUp(da));

    Mat A;
    Vec b, x;
    PetscCall(DMCreateMatrix(da, &A));
    PetscCall(DMCreateGlobalVector(da, &b));
    PetscCall(VecDuplicate(b, &x));
    PetscCall(VecSet(b, 0.0));
    PetscCall(VecSet(x, 0.0));

    PetscCall(AssembleFEM_Tet(da, nx, A, b));

    {
        PetscInt m, n;
        MatInfo  info;
        PetscCall(MatGetSize(A, &m, &n));
        PetscCall(MatGetInfo(A, MAT_GLOBAL_SUM, &info));
        if (rank == 0) {
            printf("[MAT] size=%d x %d, nz_used=%d (%.2f nnz/row)\n",
                   (int)m, (int)n, (int)info.nz_used,
                   (double)info.nz_used / (double)m);
        }
        /* Operator fingerprint, same deterministic vector as MFEM side  */
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
        PetscReal nx_, ny_;
        PetscCall(VecNorm(xv, NORM_2, &nx_));
        PetscCall(VecNorm(yv, NORM_2, &ny_));
        if (rank == 0) {
            printf("[FINGERPRINT] ||x||=%.6e ||Ax||=%.6e\n", (double)nx_, (double)ny_);
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
