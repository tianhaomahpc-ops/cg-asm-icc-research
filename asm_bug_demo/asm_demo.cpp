// asm_demo.cpp
//
// Minimal reproducer: cube with 1 Dirichlet face (x=0, u=0) and
// 5 Neumann faces, P1 H1 discretisation, PETSc CG + ASM + sub_pc=icc.
//
// Mirrors the femheart.cpp non-POD path:
//   - HypreParMatrix from MFEM ParBilinearForm::FormLinearSystem
//   - Hand-rolled "ConvertHypreToPetscAIJSafe" that inserts every stored
//     entry (including stored zeros from BC col elimination) one at a time
//   - PetscPCGSolver(comm, prefix, iter_mode=true)
//   - No MatSetOption(MAT_SYMMETRIC/MAT_SPD/...) anywhere
//   - No nullspace handling here because Dirichlet pins the system
//
// What changes with -fix_level N:
//   N = 0  : pure baseline, all suspected issues intact (default)
//   N = 1  : also pass MAT_IGNORE_ZERO_ENTRIES + skip 0.0 in the loop
//            => stored zeros from BC elimination no longer pollute the
//               PETSc sparsity pattern that PCASM walks for overlap.
//   N = 2  : also flag MAT_SYMMETRIC / MAT_SPD / MAT_SYMMETRY_ETERNAL
//            => lets PETSc honour the SBAIJ-ICC fast path on submatrices.
//   N = 3  : also force a positive-definite shift and RCM ordering on
//            the sub-PC ICC factorisation (via PETSc options injected
//            from C++ so the comparison is hermetic).
//
// Everything else (KSP type, ASM type, overlap, ICC fill) is left to
// PETSc command-line options so we can sweep without recompiling.
//
// Run examples (all on 4 ranks; nx defaults to 24 -> ~15.6k DOFs):
//   mpirun -n 4 ./asm_demo -fix_level 0 \
//       -ksp_type cg -ksp_norm_type preconditioned \
//       -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 500 \
//       -ksp_converged_reason -ksp_view \
//       -pc_type asm -pc_asm_type basic -pc_asm_overlap 0 \
//       -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels 0

#include "mfem.hpp"
#include <petsc.h>

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>

using namespace mfem;

// ---------------------------------------------------------------------------
// "Safe" Hypre -> PETSc AIJ converter that mirrors femheart.cpp::ConvertHypre
// ToPetscAIJSafe.  `fix_level` controls how many of the suspected issues we
// actually fix.  At fix_level=0 the code path is bit-for-bit equivalent to
// the original (one MatSetValues per entry, no symmetry hint, stored zeros
// preserved).
// ---------------------------------------------------------------------------
static void HypreToPetscAIJ(HypreParMatrix &hypre_mat,
                            PetscParMatrix &petsc_mat,
                            const char     *name,
                            int             my_rank,
                            int             fix_level,
                            bool            assume_spd)
{
    MPI_Comm comm = hypre_mat.GetComm();

    const HYPRE_BigInt row_start_big = hypre_mat.GetRowStarts()[0];
    const HYPRE_BigInt row_end_big   = hypre_mat.GetRowStarts()[1];
    const HYPRE_BigInt col_start_big = hypre_mat.GetColStarts()[0];
    const HYPRE_BigInt col_end_big   = hypre_mat.GetColStarts()[1];

    const PetscInt local_rows  = static_cast<PetscInt>(row_end_big - row_start_big);
    const PetscInt local_cols  = static_cast<PetscInt>(col_end_big - col_start_big);
    const PetscInt global_rows = static_cast<PetscInt>(hypre_mat.GetGlobalNumRows());
    const PetscInt global_cols = static_cast<PetscInt>(hypre_mat.GetGlobalNumCols());
    const PetscInt col_start   = static_cast<PetscInt>(col_start_big);
    const PetscInt col_end     = static_cast<PetscInt>(col_end_big);

    // ----- merge diag + offd into a single SparseMatrix with GLOBAL cols
    SparseMatrix merged;
    hypre_mat.MergeDiagAndOffd(merged);
    MFEM_VERIFY(merged.Height() == local_rows,
                "Unexpected local row count in Hypre to PETSc conversion.");

    const int    *I    = merged.HostReadI();
    const int    *J    = merged.HostReadJ();
    const real_t *data = merged.HostReadData();

    // ----- preallocation counts.  Counting stored zeros here mirrors the
    //       baseline; with fix_level>=1 we want to skip them so the alloc
    //       matches what we will actually insert.  We use a tiny absolute
    //       cut-off (rather than "== 0.0") so MFEM-quadrature round-off
    //       zeros are also dropped, matching the analytical zeros from
    //       hand-coded element stiffness in pure_petsc_fem.c.
    const real_t zero_tol = 1e-12;
    std::vector<PetscInt> d_nnz(local_rows, 0);
    std::vector<PetscInt> o_nnz(local_rows, 0);
    for (PetscInt i = 0; i < local_rows; ++i)
    {
        for (int p = I[i]; p < I[i + 1]; ++p)
        {
            if (fix_level >= 1 && std::abs(data[p]) < zero_tol) { continue; }
            const PetscInt col = static_cast<PetscInt>(J[p]);
            if (col_start <= col && col < col_end) { ++d_nnz[i]; }
            else                                    { ++o_nnz[i]; }
        }
    }

    // ----- create the PETSc AIJ matrix with exact preallocation
    Mat mat = nullptr;
    PetscErrorCode ierr = MatCreateAIJ(
        comm, local_rows, local_cols, global_rows, global_cols,
        0, local_rows ? d_nnz.data() : nullptr,
        0, local_rows ? o_nnz.data() : nullptr, &mat);
    MFEM_VERIFY(ierr == 0, "MatCreateAIJ failed.");

    if (name)
    {
        ierr = PetscObjectSetName((PetscObject) mat, name);
        MFEM_VERIFY(ierr == 0, "PetscObjectSetName failed.");
    }

    // ----- FIX 1: tell PETSc to silently drop true zeros even if a future
    //              MatSetValues happens to pass one (belt + suspenders to
    //              the explicit `if (data[p]==0.0) continue;` below).
    if (fix_level >= 1)
    {
        ierr = MatSetOption(mat, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);
        MFEM_VERIFY(ierr == 0, "MAT_IGNORE_ZERO_ENTRIES failed.");
    }

    // ----- insertion loop.  At fix_level==0 we keep the original
    //       "single-element MatSetValues per entry" pattern, which is what
    //       femheart.cpp uses.
    for (PetscInt i = 0; i < local_rows; ++i)
    {
        const PetscInt row = static_cast<PetscInt>(row_start_big) + i;
        for (int p = I[i]; p < I[i + 1]; ++p)
        {
            if (fix_level >= 1 && std::abs(data[p]) < zero_tol) { continue; }
            const PetscInt    col   = static_cast<PetscInt>(J[p]);
            const PetscScalar value = static_cast<PetscScalar>(data[p]);
            ierr = MatSetValues(mat, 1, &row, 1, &col, &value, INSERT_VALUES);
            MFEM_VERIFY(ierr == 0, "MatSetValues failed.");
        }
    }

    ierr = MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY);
    MFEM_VERIFY(ierr == 0, "MatAssemblyBegin failed.");
    ierr = MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY);
    MFEM_VERIFY(ierr == 0, "MatAssemblyEnd failed.");

    // ----- FIX 2: flag the matrix as symmetric (and SPD when we know).
    //              This is the "metadata-only" fix.  PETSc will then take
    //              the SBAIJ-ICC fast path on submatrices, and it will
    //              propagate the symmetric flag to MatCreateSubMatrices.
    if (fix_level >= 2)
    {
        ierr = MatSetOption(mat, MAT_SYMMETRIC,        PETSC_TRUE);
        ierr = MatSetOption(mat, MAT_SYMMETRY_ETERNAL, PETSC_TRUE);
        if (assume_spd)
        {
            ierr = MatSetOption(mat, MAT_SPD,          PETSC_TRUE);
            // 3.21+ would prefer MAT_SPD_ETERNAL; ignore if missing.
        }
        MFEM_VERIFY(ierr == 0, "MAT_SYMMETRIC/MAT_SPD failed.");
    }

    petsc_mat.SetMat(mat);
    ierr = MatDestroy(&mat);
    MFEM_VERIFY(ierr == 0, "MatDestroy failed.");

    if (my_rank == 0)
    {
        std::cout << "[CONVERT] " << name
                  << " global=" << global_rows << "x" << global_cols
                  << " fix_level=" << fix_level << "\n";
    }
}

// ---------------------------------------------------------------------------
// Inject sub-PC quality options for fix_level>=3 BEFORE KSPSetFromOptions
// runs.  We do it from C++ rather than the runtime options file so the
// baseline vs fix comparison cannot accidentally pick them up.
// ---------------------------------------------------------------------------
static void InjectSubPCOptions(int fix_level)
{
    if (fix_level < 3) { return; }
    PetscOptionsSetValue(NULL, "-sub_pc_factor_shift_type",
                                "positive_definite");
    PetscOptionsSetValue(NULL, "-sub_pc_factor_mat_ordering_type", "rcm");
}

// ---------------------------------------------------------------------------
// Solver schemes (controlled by -scheme N):
//   0 : CG    + PCASM BASIC                                  (default; baseline)
//   1 : GMRES + PCASM RESTRICT (== RAS, non-symmetric)
//   2 : BCGS  + PCASM RESTRICT (== RAS, non-symmetric)
//   3 : CG    + PCSHELL wrapping PCASM BASIC, with the D^{-1/2}(.)D^{-1/2}
//       multiplicity scaling that turns BASIC into "scaled additive
//       Schwarz" (sASM).  Symmetric, no over-counting.
//
// For schemes 1 and 2 we just push KSP and ASM-type via PetscOptionsSetValue
// so KSPSetFromOptions picks them up (overriding MFEM's KSPCG default and
// PETSc's default PC_ASM_RESTRICT).  For scheme 3 we have to bypass the
// outer PC and install a PCSHELL after MFEM's Customize() has run.
// ---------------------------------------------------------------------------
static void InjectSchemeOptions(int scheme)
{
    if (scheme == 1)
    {
        PetscOptionsSetValue(NULL, "-ksp_type",     "gmres");
        PetscOptionsSetValue(NULL, "-pc_asm_type",  "restrict");
    }
    else if (scheme == 2)
    {
        PetscOptionsSetValue(NULL, "-ksp_type",     "bcgs");
        PetscOptionsSetValue(NULL, "-pc_asm_type",  "restrict");
    }
    // scheme == 3 is installed later via raw PETSc API
}

// ---------------------------------------------------------------------------
// PCSHELL context: diagonal-weighted additive Schwarz.
//   apply(r) = [post? W .] M_BASIC^{-1} [pre? W .] r,   W = diag(w)
// weight_mode (set in InstallScaledASM) selects W and which sides:
//   0  W = D^{-1/2}, pre=post=1  ->  D^{-1/2} M_BASIC^{-1} D^{-1/2}  (sASM, sym)  [scheme 3/4]
//   1  W = D^{-1},   pre=post=1  ->  D^{-1}   M_BASIC^{-1} D^{-1}    (sym, over-norm) [scheme 5]
//   2  W = D^{-1},   pre=0,post=1->  D^{-1}   M_BASIC^{-1}          (non-symmetric)  [scheme 6]
// ---------------------------------------------------------------------------
struct SASMCtx
{
    PC   inner_pc;   // child PCASM_BASIC with overlap and ICC/Chebyshev sub-PCs
    Vec  w;          // diagonal weight (D^{-1/2} or D^{-1})
    Vec  tmp;        // scratch
    int  pre;        // apply W before  M_BASIC^{-1}
    int  post;       // apply W after   M_BASIC^{-1}
};

extern "C" PetscErrorCode SASMApply(PC pc, Vec r, Vec z)
{
    SASMCtx *ctx = nullptr;
    PetscCall(PCShellGetContext(pc, (void**)&ctx));
    if (ctx->pre) { PetscCall(VecPointwiseMult(ctx->tmp, ctx->w, r)); }  // tmp = W r
    else          { PetscCall(VecCopy(r, ctx->tmp)); }                   // tmp = r
    PetscCall(PCApply(ctx->inner_pc, ctx->tmp, z));                      // z = M_BASIC^{-1} tmp
    if (ctx->post) { PetscCall(VecPointwiseMult(z, ctx->w, z)); }        // z = W z (aliased ok)
    return PETSC_SUCCESS;
}

extern "C" PetscErrorCode SASMDestroy(PC pc)
{
    SASMCtx *ctx = nullptr;
    PetscCall(PCShellGetContext(pc, (void**)&ctx));
    if (ctx)
    {
        PetscCall(PCDestroy(&ctx->inner_pc));
        PetscCall(VecDestroy(&ctx->w));
        PetscCall(VecDestroy(&ctx->tmp));
        delete ctx;
    }
    PetscCall(PCShellSetContext(pc, nullptr));
    return PETSC_SUCCESS;
}

// Build an inner PCASM_BASIC + ICC, compute its overlap-aware multiplicity
// vector, then install a PCSHELL on the outer KSP that applies
// D^{-1/2} * M_BASIC^{-1} * D^{-1/2}.
static void InstallScaledASM(KSP ksp, Mat A,
                             PetscInt overlap, PetscInt icc_levels,
                             int my_rank,
                             int cheby_deg = 0,
                             int weight_mode = 0)   // 0:D^-1/2 both, 1:D^-1 both, 2:D^-1 left only
{
    // cheby_deg == 0 : block solve = preonly + ICC(icc_levels)   [scheme 3]
    // cheby_deg >= 1 : block solve = cheby_deg steps of Chebyshev,
    //                  preconditioned by ICC(icc_levels)         [scheme 4]
    //   A fixed-degree Chebyshev iteration with frozen eigenvalue bounds
    //   and a symmetric (ICC) smoother is a fixed SPD linear operator, so
    //   the outer CG remains valid.  This makes the *inexact* local solve
    //   closer to the exact block inverse without the O(n_i^3) cost (or the
    //   extra fill / memory) of higher-level ICC -- attacking the second
    //   half of the "overcounting x inexactness" disease cheaply.
    // 1. Build the inner PCASM, fully self-contained (no prefix).
    if (my_rank == 0) { std::cout << "\n  [sASM step 1] create inner pc... " << std::flush; }
    PC inner_pc = nullptr;
    PCCreate(PetscObjectComm((PetscObject)A), &inner_pc);
    PCSetType(inner_pc, PCASM);
    PCASMSetType(inner_pc, PC_ASM_BASIC);
    PCASMSetOverlap(inner_pc, overlap);
    PCSetOperators(inner_pc, A, A);
    if (my_rank == 0) { std::cout << "PCSetUp... " << std::flush; }
    PCSetUp(inner_pc);
    if (my_rank == 0) { std::cout << "ok\n  [sASM step 2] sub-KSP cfg... " << std::flush; }

    // 2. Configure the sub-KSPs.
    {
        KSP     *sub_ksps = nullptr;
        PetscInt n_local  = 0, first = 0;
        PCASMGetSubKSP(inner_pc, &n_local, &first, &sub_ksps);
        for (PetscInt i = 0; i < n_local; ++i)
        {
            PC sub_pc = nullptr;
            if (cheby_deg <= 0)
            {
                KSPSetType(sub_ksps[i], KSPPREONLY);
                KSPGetPC(sub_ksps[i], &sub_pc);
                PCSetType(sub_pc, PCICC);
                PCFactorSetLevels(sub_pc, icc_levels);
            }
            else
            {
                // Fixed-degree Chebyshev over ICC, eigenvalues estimated
                // once at setup then frozen (fixed linear operator).
                KSPSetType(sub_ksps[i], KSPCHEBYSHEV);
                KSPSetTolerances(sub_ksps[i], PETSC_DEFAULT, PETSC_DEFAULT,
                                 PETSC_DEFAULT, cheby_deg);
                KSPSetNormType(sub_ksps[i], KSP_NORM_NONE);
                KSPSetInitialGuessNonzero(sub_ksps[i], PETSC_FALSE);
                KSPChebyshevEstEigSet(sub_ksps[i], 0.0, 0.1, 0.0, 1.1);
                KSPGetPC(sub_ksps[i], &sub_pc);
                PCSetType(sub_pc, PCICC);
                PCFactorSetLevels(sub_pc, icc_levels);
            }
        }
        PCSetUpOnBlocks(inner_pc);
    }
    if (my_rank == 0) { std::cout << "ok\n  [sASM step 3] multiplicity... " << std::flush; }

    // 3. Compute the multiplicity vector m[k] = #subdomains that contain k.
    //    PCASMGetLocalSubdomains returns the IS lists.  is[i] is the WITH-
    //    overlap version (so the boundary DOFs of neighbour partitions are
    //    in there); is_local[i] is just the rank's own DOFs.  We sum the
    //    "1" indicator across is[i] to get m globally.
    int dbg_rank;
    MPI_Comm_rank(PetscObjectComm((PetscObject)A), &dbg_rank);
    Vec mult = nullptr;
    if (dbg_rank == 0) { std::cout << "MatCreateVecs..." << std::flush; }
    MatCreateVecs(A, &mult, NULL);
    if (dbg_rank == 0) { std::cout << "ok VecSet..." << std::flush; }
    VecSet(mult, 0.0);
    if (dbg_rank == 0) { std::cout << "ok GetIS..." << std::flush; }
    {
        PetscInt n_local = 0;
        IS *is_with_overlap = nullptr, *is_local_only = nullptr;
        PCASMGetLocalSubdomains(inner_pc, &n_local,
                                &is_with_overlap, &is_local_only);
        // Every rank prints its own n_local + IS size; helpful even if
        // the global stdout interleaves.
        for (int r = 0; r < 1; ++r)   // just rank 0 first
        {
            if (dbg_rank == r)
            {
                PetscInt isize = 0;
                if (n_local > 0)
                {
                    ISGetLocalSize(is_with_overlap[0], &isize);
                }
                std::cout << "ok n_local=" << n_local
                          << " is_size[0]=" << isize << " loop..." << std::flush;
            }
        }
        for (PetscInt i = 0; i < n_local; ++i)
        {
            const PetscInt *idx = nullptr;
            PetscInt        n   = 0;
            ISGetLocalSize(is_with_overlap[i], &n);
            ISGetIndices(is_with_overlap[i], &idx);
            std::vector<PetscScalar> ones(n, 1.0);
            VecSetValues(mult, n, idx, ones.data(), ADD_VALUES);
            ISRestoreIndices(is_with_overlap[i], &idx);
        }
        if (dbg_rank == 0) { std::cout << "ok AsmBegin..." << std::flush; }
        VecAssemblyBegin(mult);
        if (dbg_rank == 0) { std::cout << "ok AsmEnd..." << std::flush; }
        VecAssemblyEnd(mult);
        if (dbg_rank == 0) { std::cout << "ok\n  " << std::flush; }
    }

    // 4. Build the diagonal weight w:
    //      weight_mode 0 -> w = 1/sqrt(m[k])   (D^{-1/2})
    //      weight_mode 1 -> w = 1/m[k]         (D^{-1}, applied both sides)
    //      weight_mode 2 -> w = 1/m[k]         (D^{-1}, applied left only)
    if (my_rank == 0) { std::cout << "[step4] dup... " << std::flush; }
    Vec inv_sqrt = nullptr;
    VecDuplicate(mult, &inv_sqrt);
    VecCopy(mult, inv_sqrt);
    VecReciprocal(inv_sqrt);                 // 1/m[k]
    if (weight_mode == 0) { VecSqrtAbs(inv_sqrt); }   // -> 1/sqrt(m[k]) for sASM
    if (my_rank == 0) { std::cout << "ok\n  [sASM step 4b] range... " << std::flush; }

    {
        PetscReal mmin, mmax;
        VecMin(mult, NULL, &mmin);
        VecMax(mult, NULL, &mmax);
        if (my_rank == 0)
        {
            std::cout << "ok [" << (int)mmin << "," << (int)mmax << "]\n  " << std::flush;
        }
    }
    VecDestroy(&mult);
    if (my_rank == 0) { std::cout << "ok\n  [sASM step 5] wire PCSHELL... " << std::flush; }

    // 5. Now wire the outer KSP's PC to a PCSHELL that does the wrap.
    //    The shell owns inner_pc, inv_sqrt and tmp.
    if (my_rank == 0) { std::cout << "[step5] GetPC... " << std::flush; }
    PC pc_outer = nullptr;
    KSPGetPC(ksp, &pc_outer);
    if (my_rank == 0) { std::cout << "SetType SHELL... " << std::flush; }
    PCSetType(pc_outer, PCSHELL);
    if (my_rank == 0) { std::cout << "alloc ctx... " << std::flush; }

    SASMCtx *ctx = new SASMCtx;
    ctx->inner_pc = inner_pc;
    ctx->w        = inv_sqrt;            // D^{-1/2} (mode 0) or D^{-1} (mode 1/2)
    ctx->pre      = (weight_mode == 2) ? 0 : 1;   // left+right except mode 2 (left/post only)
    ctx->post     = 1;
    MatCreateVecs(A, &ctx->tmp, NULL);

    PCShellSetContext(pc_outer, ctx);
    PCShellSetApply  (pc_outer, SASMApply);
    PCShellSetDestroy(pc_outer, SASMDestroy);
    PCShellSetName   (pc_outer, "weighted_ASM");
    if (my_rank == 0) { std::cout << "ok\n  [step6] PCSetUp(outer)... " << std::flush; }

    // 6. PCSetUp on the shell triggers any internal init; for our shell
    //    there is no SetUp callback, so this is essentially a no-op.
    PCSetUp(pc_outer);
    if (my_rank == 0) { std::cout << "ok\n" << std::flush; }

    if (my_rank == 0)
    {
        std::cout << "[sASM] PCSHELL installed (overlap=" << overlap
                  << ", icc_levels=" << icc_levels << ")\n";
    }
}

// Pull out our own knobs (-fix_level, -nx, -scheme) from argv BEFORE we
// hand it to either MFEM's OptionsParser or PETSc, so neither one yells.
// Mutates argc/argv in place.
static void ExtractOwnOptions(int &argc, char **argv,
                              int &fix_level, int &nx, int &scheme, bool &warmup,
                              bool &pure_neumann,   // -pure_neumann: drop Dirichlet face -> Sys2
                              bool &all_dirichlet)  // -all_dirichlet: all 6 faces Dirichlet
{
    int out = 1;                       // keep argv[0]
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if ((a == "-fix_level" || a == "--fix-level") && i + 1 < argc)
        {
            fix_level = std::atoi(argv[++i]);  continue;
        }
        if ((a == "-nx" || a == "--nx") && i + 1 < argc)
        {
            nx = std::atoi(argv[++i]);  continue;
        }
        if ((a == "-scheme" || a == "--scheme") && i + 1 < argc)
        {
            scheme = std::atoi(argv[++i]);  continue;
        }
        if (a == "-warmup")            // no value: do one untimed solve first
        {
            warmup = true;  continue;  // so the timed Mult is solve-only (PC pre-set-up)
        }
        if (a == "-pure_neumann")      // no value: all-Neumann singular (Sys2)
        {
            pure_neumann = true;  continue;
        }
        if (a == "-all_dirichlet")     // no value: u=0 on all 6 faces
        {
            all_dirichlet = true;  continue;
        }
        argv[out++] = argv[i];         // keep everything else
    }
    argc = out;
}

int main(int argc, char *argv[])
{
    Mpi::Init(argc, argv);
    int my_rank   = Mpi::WorldRank();
    int num_ranks = Mpi::WorldSize();

    // -------- our own knobs first, so neither MFEM nor PETSc sees them
    int fix_level = 0;   // 0=baseline, 1=zeros, 2=+symmetric, 3=+shift/RCM
    int nx        = 24;
    int scheme    = 0;   // 0=CG+BASIC, 1=GMRES+RAS, 2=BCGS+RAS, 3=CG+sASM
    bool warmup   = false;   // -warmup: one untimed solve so time= is solve-only
    bool pure_neumann = false;   // -pure_neumann: drop Dirichlet -> singular Sys2
    bool all_dirichlet = false;  // -all_dirichlet: u=0 on all 6 faces
    ExtractOwnOptions(argc, argv, fix_level, nx, scheme, warmup, pure_neumann,
                      all_dirichlet);
    if (pure_neumann && all_dirichlet)
    {
        if (my_rank == 0)
            std::cerr << "error: -pure_neumann and -all_dirichlet are mutually exclusive\n";
        return 1;
    }

    // Boot PETSc through MFEM (no rc file; everything via CLI).
    MFEMInitializePetsc(&argc, &argv, NULL, NULL);

    // Silence PETSc's "options-left" abort.
    PetscOptionsSetValue(NULL, "-options_left", "no");

    InjectSubPCOptions(fix_level);
    InjectSchemeOptions(scheme);   // pushes -ksp_type / -pc_asm_type for 1,2

    if (my_rank == 0)
    {
        const char *scheme_name[] = {
            "0:CG+ASM_BASIC",
            "1:GMRES+ASM_RESTRICT (RAS)",
            "2:BCGS +ASM_RESTRICT (RAS)",
            "3:CG+sASM (D^-1/2 BASIC D^-1/2)",
            "4:CG+sASM + Chebyshev block solve",
            "5:CG + D^-1 BASIC D^-1 (sym, over-norm)",
            "6:D^-1 BASIC (non-symmetric, one-sided)" };
        std::cout << "================================================\n"
                  << "  asm_demo  (ranks=" << num_ranks
                  << ", fix_level=" << fix_level
                  << ", scheme=" << (scheme>=0 && scheme<=6 ? scheme_name[scheme] : "?")
                  << ", nx=" << nx << ")\n"
                  << "================================================\n";
    }

    // All PETSc-holding objects must be destroyed BEFORE
    // MFEMFinalizePetsc(), otherwise their ~Destroy calls land on a
    // PETSc that has already finalized and abort the program.  Hence
    // the explicit scope block below.
    {
    // -------- 1) build a serial cube mesh, then partition ---------------
    // P1 LINEAR FE on a tetrahedral mesh.  MFEM's MakeCartesian3D with
    // TETRAHEDRON splits each hex into 6 tets sharing the (0,6) main
    // diagonal -- see mfem-3.3.2 mesh.cpp:986 (hex_to_tet table).
    //
    // Boundary attributes from Make3D() (mesh.cpp:1964-..):
    //   1: z=0   2: y=0   3: x=nx   4: y=ny   5: x=0   6: z=nz
    // We Dirichlet the x=0 face (attr 5), Neumann elsewhere.
    Mesh serial_mesh = Mesh::MakeCartesian3D(
        nx, nx, nx, Element::TETRAHEDRON, 1.0, 1.0, 1.0);
    const int dirichlet_attr = 5;

    if (my_rank == 0) {
        std::cout << "[MESH] serial NumVert=" << serial_mesh.GetNV()
                  << " NumEl=" << serial_mesh.GetNE()
                  << " NumBdrEl=" << serial_mesh.GetNBE() << "\n";
    }

    // Use MFEM's default METIS partitioning of the element graph.  We will
    // dump this partition to a file so the PETSc demo can apply the
    // identical METIS-derived element-to-rank mapping.
    int *metis_part = serial_mesh.GeneratePartitioning(num_ranks, 1);
    Array<int> elem_part(metis_part, serial_mesh.GetNE());
    if (my_rank == 0) {
        std::string fn = std::string("metis_part_nx") + std::to_string(nx)
                       + "_n" + std::to_string(num_ranks) + ".bin";
        std::ofstream ofs(fn, std::ios::binary);
        int hdr[3] = { serial_mesh.GetNE(), nx, num_ranks };
        ofs.write(reinterpret_cast<char*>(hdr), sizeof(hdr));
        ofs.write(reinterpret_cast<char*>(metis_part),
                  sizeof(int) * serial_mesh.GetNE());
        ofs.close();
        std::cout << "[PART] wrote METIS element partition to " << fn
                  << " (" << serial_mesh.GetNE() << " elements)\n";
    }

    ParMesh pmesh(MPI_COMM_WORLD, serial_mesh, elem_part.GetData());
    serial_mesh.Clear();

    // -------- 2) P1 H1 finite element space -----------------------------
    H1_FECollection fec(/*order=*/1, /*dim=*/3);
    ParFiniteElementSpace fes(&pmesh, &fec);

    const HYPRE_BigInt n_dofs = fes.GlobalTrueVSize();
    if (my_rank == 0)
    {
        std::cout << "[FES] global true dofs = " << n_dofs << "\n";
    }

    // -------- 3) essential (Dirichlet) DOF list -------------------------
    Array<int> ess_bdr(pmesh.bdr_attributes.Max());
    ess_bdr = 0;
    if (all_dirichlet)
        ess_bdr = 1;                       // u=0 on all 6 faces (fully pinned)
    else if (!pure_neumann)
        ess_bdr[dirichlet_attr - 1] = 1;   // mark only the x=0 face (Sys3)
    // pure_neumann (Sys2): leave ess_bdr all-zero -> all 6 faces Neumann -> singular K
    Array<int> ess_tdof_list;
    fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    if (all_dirichlet && my_rank == 0)
        std::cout << "[ALL_DIRICHLET] u=0 on all 6 faces (fully pinned, non-singular)\n";

    // -------- 4) bilinear & linear forms --------------------------------
    // Source term:
    //   source_type = 0  ->  f(x,y,z) = 1            [analytically 1D]
    //   source_type = 1  ->  f(x,y,z) = sin(pi y) sin(pi z)
    //                        [genuinely 3D solution; useful for plots]
    int source_type = 0;
    double dt = -1.0;   // <=0 : pure stiffness K (elliptic, default).
                        // >0  : Crank-Nicolson monodomain matrix
                        //       A = (1/dt) M + (1/2) K  (mass-dominated as
                        //       dt -> 0, mirrors the parabolic diffusion step)
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-source_type" && i + 1 < argc) {
            source_type = std::atoi(argv[i+1]);
        }
        if (std::string(argv[i]) == "-dt" && i + 1 < argc) {
            dt = std::atof(argv[i+1]);
        }
    }

    ConstantCoefficient sigma(1.0);
    ConstantCoefficient one_rhs(1.0);
    FunctionCoefficient yz_rhs([](const Vector &x) {
        return std::sin(M_PI * x[1]) * std::sin(M_PI * x[2]);
    });

    // Build the system matrix.
    //   dt <= 0 : A = K            (pure Laplacian, elliptic — Sys2/Sys3 shape)
    //   dt  > 0 : A = (1/dt) M + (1/2) K   (Crank-Nicolson monodomain step;
    //             theta = 1/2; coefficients on M, K fold chi*C_m, sigma = 1)
    ParBilinearForm a(&fes);
    const bool monodomain = (dt > 0.0);
    if (monodomain)
    {
        ConstantCoefficient mass_coef(1.0 / dt);   // (1/dt) on the mass term
        ConstantCoefficient diff_coef(0.5);        // theta = 1/2 on stiffness
        a.AddDomainIntegrator(new MassIntegrator(mass_coef));
        a.AddDomainIntegrator(new DiffusionIntegrator(diff_coef));
    }
    else
    {
        a.AddDomainIntegrator(new DiffusionIntegrator(sigma));
    }
    a.Assemble();
    if (my_rank == 0)
    {
        if (monodomain)
            std::cout << "[MATRIX] Crank-Nicolson monodomain  A = (1/dt) M + (1/2) K"
                      << ", dt=" << dt << "  (mass-dominated, well-conditioned)\n";
        else
            std::cout << "[MATRIX] pure stiffness  A = K  (elliptic, ill-conditioned)\n";
    }

    ParLinearForm b(&fes);
    if (source_type == 0) {
        b.AddDomainIntegrator(new DomainLFIntegrator(one_rhs));
    } else {
        b.AddDomainIntegrator(new DomainLFIntegrator(yz_rhs));
    }
    b.Assemble();
    if (my_rank == 0) {
        std::cout << "[SOURCE] source_type=" << source_type
                  << " (" << (source_type == 0 ? "f=1, 1D-symmetric"
                                              : "f=sin(pi y)sin(pi z), genuinely 3D")
                  << ")\n";
    }

    // Dirichlet value u = 0 on the x=0 face.
    ParGridFunction u_gf(&fes);
    u_gf = 0.0;

    HypreParMatrix A_hypre;
    Vector X, B;
    a.FormLinearSystem(ess_tdof_list, u_gf, b, A_hypre, X, B);

    // pure_neumann (Sys2): A is singular with ker = span{1}.  Make the RHS
    // compatible (orthogonal to the constant null space) by subtracting its
    // global mean, so sum(B) = 0.  The null space itself is attached to the
    // PETSc operator below via MatSetNullSpace.
    if (pure_neumann)
    {
        double loc[2] = {0.0, (double)B.Size()}, glob[2] = {0.0, 0.0};
        for (int i = 0; i < B.Size(); ++i) loc[0] += B(i);
        MPI_Allreduce(loc, glob, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        const double mean = glob[0] / glob[1];
        for (int i = 0; i < B.Size(); ++i) B(i) -= mean;
        if (my_rank == 0)
            std::cout << "[PURE_NEUMANN] singular K, ker=span{1}; RHS mean removed "
                      << "(global N=" << (long)glob[1] << ")\n";
    }

    if (my_rank == 0)
    {
        std::cout << "[ASSEMBLY] hypre matrix global rows = "
                  << A_hypre.GetGlobalNumRows()
                  << ", local rows on rank 0 = "
                  << (A_hypre.GetRowStarts()[1] - A_hypre.GetRowStarts()[0])
                  << "\n";
    }

    // -------- 5) hand-roll Hypre -> PETSc AIJ (femheart pattern) --------
    PetscParMatrix A_petsc;
    HypreToPetscAIJ(A_hypre, A_petsc, "demo_A",
                    my_rank, fix_level, /*assume_spd=*/true);

    // pure_neumann (Sys2): attach the constant null space so the outer Krylov
    // method projects it out of the residual/iterates (and removes it from the
    // RHS at KSPSolve).  Subdomain blocks remain non-singular (Dirichlet cuts),
    // so ICC/Chebyshev sub-solves are unaffected.
    if (pure_neumann)
    {
        MatNullSpace nsp = NULL;
        MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &nsp);
        MatSetNullSpace(static_cast<Mat>(A_petsc), nsp);
        MatNullSpaceDestroy(&nsp);
    }

    // Dump the assembled, METIS-partitioned PETSc matrix to binary so the
    // pure-PETSc demo can MatLoad it and run the IDENTICAL KSP problem
    // (same matrix values, same row distribution, same partition).  The
    // RHS is also dumped so the test is truly bit-identical.
    {
        std::string mfn = std::string("mfem_A_nx") + std::to_string(nx)
                        + "_n" + std::to_string(num_ranks) + ".petscbin";
        std::string bfn = std::string("mfem_b_nx") + std::to_string(nx)
                        + "_n" + std::to_string(num_ranks) + ".petscbin";
        PetscViewer vw;
        PetscViewerBinaryOpen(MPI_COMM_WORLD, mfn.c_str(),
                              FILE_MODE_WRITE, &vw);
        MatView(A_petsc, vw);
        PetscViewerDestroy(&vw);
        PetscViewerBinaryOpen(MPI_COMM_WORLD, bfn.c_str(),
                              FILE_MODE_WRITE, &vw);
        // B is an mfem::Vector with the local true-DOF length matching
        // A_petsc's row partition.  Wrap it as a PETSc Vec for VecView.
        Vec B_raw = nullptr;
        VecCreateMPIWithArray(MPI_COMM_WORLD, 1, B.Size(), PETSC_DECIDE,
                              B.HostRead(), &B_raw);
        VecView(B_raw, vw);
        VecDestroy(&B_raw);
        PetscViewerDestroy(&vw);

        // Dump per-rank local row count so that pure_petsc_load can call
        // MatSetSizes(A, mloc, nloc, ...) BEFORE MatLoad and thereby
        // preserve MFEM's METIS row distribution.  Without this MatLoad
        // would default to PETSC_DECIDE (contiguous equal chunks), which
        // would silently re-partition and erase our METIS layout.
        {
            PetscInt mloc, nloc;
            MatGetLocalSize(A_petsc, &mloc, &nloc);
            std::vector<PetscInt> mlocs(num_ranks, 0), nlocs(num_ranks, 0);
            MPI_Allgather(&mloc, 1, MPIU_INT, mlocs.data(), 1, MPIU_INT,
                          MPI_COMM_WORLD);
            MPI_Allgather(&nloc, 1, MPIU_INT, nlocs.data(), 1, MPIU_INT,
                          MPI_COMM_WORLD);
            if (my_rank == 0) {
                std::string lfn = std::string("mfem_layout_nx")
                                + std::to_string(nx) + "_n"
                                + std::to_string(num_ranks) + ".bin";
                std::ofstream ofs(lfn, std::ios::binary);
                int hdr[2] = { num_ranks, (int)mlocs[0] };  // header
                hdr[1] = num_ranks;
                ofs.write(reinterpret_cast<char*>(&hdr[0]), sizeof(int));
                ofs.write(reinterpret_cast<char*>(mlocs.data()),
                          sizeof(PetscInt) * num_ranks);
                ofs.write(reinterpret_cast<char*>(nlocs.data()),
                          sizeof(PetscInt) * num_ranks);
                ofs.close();
                std::cout << "[LAYOUT] per-rank rows = [";
                for (int r = 0; r < num_ranks; ++r) {
                    std::cout << (long long)mlocs[r]
                              << (r+1<num_ranks ? "," : "");
                }
                std::cout << "]\n";
            }
        }
        if (my_rank == 0) {
            std::cout << "[DUMP] wrote " << mfn << " and " << bfn << "\n";
        }
    }

    // MatGetSize + MatGetInfo(..., MAT_GLOBAL_SUM, ...) are COLLECTIVE,
    // so every rank must call them.  Print only on rank 0.
    {
        PetscInt m, n;
        MatGetSize(A_petsc, &m, &n);
        MatInfo info;
        MatGetInfo(A_petsc, MAT_GLOBAL_SUM, &info);
        if (my_rank == 0)
        {
            std::cout << "[PETSC MAT] size = " << m << "x" << n
                      << ", nz_used = "      << (PetscInt)info.nz_used
                      << ", nz_allocated = " << (PetscInt)info.nz_allocated
                      << ", nz_unneeded = "  << (PetscInt)info.nz_unneeded
                      << "\n";
        }
        // Operator fingerprint: compute  Ax  with x = a deterministic
        // vector (cos waves), then print  ||x|| and ||Ax||  -- if two
        // implementations agree on the operator, these match to ~1e-12.
        {
            Mat M = A_petsc;
            Vec xv, yv;
            MatCreateVecs(M, &xv, &yv);
            PetscInt mloc, mlo;
            VecGetLocalSize(xv, &mloc);
            VecGetOwnershipRange(xv, &mlo, NULL);
            PetscScalar *arr;
            VecGetArray(xv, &arr);
            for (PetscInt p = 0; p < mloc; ++p) {
                PetscInt g = mlo + p;
                arr[p] = std::cos(g * 0.012345) + std::sin(g * 0.054321);
            }
            VecRestoreArray(xv, &arr);
            MatMult(M, xv, yv);
            PetscReal nx_, ny_;
            VecNorm(xv, NORM_2, &nx_);
            VecNorm(yv, NORM_2, &ny_);
            if (my_rank == 0) {
                std::cout << "[FINGERPRINT] ||x||=" << std::scientific << nx_
                          << " ||Ax||=" << ny_ << "\n";
            }
            VecDestroy(&xv); VecDestroy(&yv);
        }
        // Per-row nnz histogram on rank 0 only (works for square SeqAIJ
        // when num_ranks==1; that is exactly what we use here for the
        // structural-diff inspection).
        if (num_ranks == 1)
        {
            Mat M = A_petsc;
            int hist[40] = {0};
            int max_row_nnz = 0;
            for (PetscInt row = 0; row < m; ++row) {
                PetscInt ncols = 0;
                const PetscInt *cols = nullptr;
                const PetscScalar *vals = nullptr;
                MatGetRow(M, row, &ncols, &cols, &vals);
                // count entries with |val| > 0
                int n_nonzero = 0;
                for (PetscInt p = 0; p < ncols; ++p) {
                    if (vals[p] != 0.0) ++n_nonzero;
                }
                MatRestoreRow(M, row, &ncols, &cols, &vals);
                if (n_nonzero >= 40) n_nonzero = 39;
                ++hist[n_nonzero];
                if (n_nonzero > max_row_nnz) max_row_nnz = n_nonzero;
            }
            if (my_rank == 0) {
                std::cout << "[HIST] per-row nonzero counts (after filtering exact 0): ";
                for (int k = 0; k <= max_row_nnz; ++k) {
                    if (hist[k] > 0) std::cout << k << ":" << hist[k] << " ";
                }
                std::cout << "(max=" << max_row_nnz << ")\n";
            }
        }
    }

    // -------- 6) PetscPCGSolver (mirror femheart) -----------------------
    // 3-arg constructor: (comm, prefix, iter_mode).  No prefix here, so
    // command-line options are read with no prefix:  -ksp_*, -pc_*, -sub_*.
    PetscPCGSolver pcg(MPI_COMM_WORLD, /*prefix=*/std::string(),
                       /*iter_mode=*/true);
    pcg.SetOperator(A_petsc);
    pcg.SetMaxIter(1000);
    pcg.SetRelTol(1e-6);
    pcg.SetAbsTol(1e-12);
    pcg.SetPrintLevel(0);

    // For scheme==3 we have to (a) force KSPSetFromOptions to run NOW so
    // the user's -ksp_* options land on the KSP, then (b) overwrite the
    // outer PC with our PCSHELL.  Done in this order, the PCSHELL is the
    // final PC seen by KSPSolve and our installation is not clobbered.
    if (scheme >= 3 && scheme <= 6)
    {
        // scheme 3 = sASM (D^-1/2 .. D^-1/2) + preonly/ICC block solve
        // scheme 4 = sASM + Chebyshev(deg) block solve (deg from -localcheby, def 2)
        // scheme 5 = D^-1 BASIC D^-1 (symmetric, over-normalized) + ICC
        // scheme 6 = D^-1 BASIC      (non-symmetric, one-sided)    + ICC
        pcg.Customize(true);                       // runs KSPSetFromOptions
        KSP   ksp_raw = static_cast<KSP>(pcg);
        Mat   A_raw   = nullptr;
        KSPGetOperators(ksp_raw, &A_raw, NULL);

        PetscInt overlap = 0, icc_lev = 0;
        PetscOptionsGetInt(NULL, NULL, "-pc_asm_overlap",         &overlap, NULL);
        PetscOptionsGetInt(NULL, NULL, "-sub_pc_factor_levels",   &icc_lev, NULL);

        int cheby_deg = 0;
        if (scheme == 4)
        {
            cheby_deg = 2;
            for (int i = 1; i < argc; ++i)
                if (std::string(argv[i]) == "-localcheby" && i + 1 < argc)
                    cheby_deg = std::atoi(argv[i+1]);
        }
        int weight_mode = (scheme == 5) ? 1 : (scheme == 6) ? 2 : 0;
        InstallScaledASM(ksp_raw, A_raw, overlap, icc_lev, my_rank, cheby_deg, weight_mode);
    }

    if (warmup)
    {
        // One untimed solve: forces all lazy PCSetUp (ICC factorisation, overlap)
        // to happen now, so the timed Mult below measures solve-only cost.
        // CG zeros the initial guess each Mult, so the timed solve does the same
        // work as a cold solve -- fair, setup-excluded ("amortised") timing.
        Vector Xtmp(X);
        pcg.Mult(B, Xtmp);
    }
    double t0 = MPI_Wtime();
    pcg.Mult(B, X);
    double t1 = MPI_Wtime();

    int    iters = pcg.GetNumIterations();
    double rnorm = pcg.GetFinalNorm();

    if (my_rank == 0)
    {
        std::cout << "[RESULT] fix_level=" << fix_level
                  << "  iters=" << iters
                  << "  final_pnorm=" << std::scientific << std::setprecision(3)
                  << rnorm
                  << "  time=" << std::fixed << std::setprecision(3)
                  << (t1 - t0) << " s\n";
    }

    // [SOLN] block -- statistics of the actual computed solution so we
    // can check (a) whether the KSP truly converged for this (O,L)
    // (b) whether the solution matches the analytical u(x,y,z)=x-x^2/2.
    {
        // Wrap mfem::Vector X (true dofs) and B as PETSc Vecs.
        Vec Xpet = nullptr, Bpet = nullptr;
        VecCreateMPIWithArray(MPI_COMM_WORLD, 1, X.Size(), PETSC_DECIDE,
                              X.HostRead(), &Xpet);
        VecCreateMPIWithArray(MPI_COMM_WORLD, 1, B.Size(), PETSC_DECIDE,
                              B.HostRead(), &Bpet);
        // raw KSP convergence reason
        KSPConvergedReason reason;
        KSPGetConvergedReason(static_cast<KSP>(pcg), &reason);
        // true residual ||b - A x||_2 (NOT the preconditioned norm)
        Vec resid;
        VecDuplicate(Xpet, &resid);
        MatMult(A_petsc, Xpet, resid);   // resid = A x
        VecAYPX(resid, -1.0, Bpet);      // resid = B - A x
        PetscReal r_norm, b_norm;
        VecNorm(resid, NORM_2, &r_norm);
        VecNorm(Bpet,  NORM_2, &b_norm);
        VecDestroy(&resid);
        // solution statistics
        PetscReal x_l2, x_min, x_max;
        PetscScalar x_sum;
        VecNorm(Xpet, NORM_2, &x_l2);
        VecMin (Xpet, NULL, &x_min);
        VecMax (Xpet, NULL, &x_max);
        VecSum (Xpet, &x_sum);
        PetscInt nDof; VecGetSize(Xpet, &nDof);
        double x_mean = (double)(PetscRealPart(x_sum) / (double)nDof);
        // analytical comparison: u_ref(x,y,z) = x - x^2/2
        ParGridFunction u_ref_gf(&fes);
        for (int i = 0; i < fes.GetVSize(); ++i) {
            const real_t *crd = pmesh.GetVertex(i);
            u_ref_gf(i) = crd[0] - 0.5 * crd[0] * crd[0];
        }
        Vector U_ref(fes.GetTrueVSize());
        u_ref_gf.GetTrueDofs(U_ref);
        Vec uref;
        VecCreateMPIWithArray(MPI_COMM_WORLD, 1, U_ref.Size(), PETSC_DECIDE,
                              U_ref.HostRead(), &uref);
        PetscReal uref_l2; VecNorm(uref, NORM_2, &uref_l2);
        Vec err; VecDuplicate(Xpet, &err);
        VecWAXPY(err, -1.0, uref, Xpet);   // err = Xpet - uref
        PetscReal err_l2; VecNorm(err, NORM_2, &err_l2);
        VecDestroy(&err); VecDestroy(&uref);
        VecDestroy(&Xpet); VecDestroy(&Bpet);

        const char *rstr =
            (reason > 0) ? (reason == KSP_CONVERGED_RTOL ? "CONVERGED_RTOL" :
                            reason == KSP_CONVERGED_ATOL ? "CONVERGED_ATOL" :
                            "CONVERGED_OTHER")
                         : (reason == KSP_DIVERGED_ITS   ? "DIVERGED_ITS"  :
                            reason == KSP_DIVERGED_DTOL  ? "DIVERGED_DTOL" :
                            "DIVERGED_OTHER");
        if (my_rank == 0)
        {
            std::cout << "[SOLN] reason=" << rstr
                      << "  iters=" << iters
                      << "  ||r||/||b||=" << std::scientific << std::setprecision(3)
                      << (double)(r_norm / b_norm)
                      << "  ||u||="   << std::scientific << std::setprecision(6) << (double)x_l2
                      << "  min="     << (double)x_min
                      << "  max="     << (double)x_max
                      << "  mean="    << x_mean
                      << "  ||u-u*||/||u*||=" << std::scientific << std::setprecision(3)
                      << (double)(err_l2 / uref_l2)
                      << "\n";
        }
    }

    a.RecoverFEMSolution(X, b, u_gf);

    // Probe the solution along three orthogonal lines.  Each rank only
    // owns part of the mesh after METIS partitioning, so the probe must
    // be COLLECTIVE: each rank finds its own nearest vertex to the query
    // point, then MPI_Allreduce(MINLOC) picks the global owner, which
    // broadcasts the value.
    {
        auto probe = [&](double xq, double yq, double zq) -> double {
            double local_best_d2 = 1e300;
            int    local_best_idx = -1;
            for (int i = 0; i < pmesh.GetNV(); ++i) {
                const real_t *c = pmesh.GetVertex(i);
                double d2 = (c[0]-xq)*(c[0]-xq)
                          + (c[1]-yq)*(c[1]-yq)
                          + (c[2]-zq)*(c[2]-zq);
                if (d2 < local_best_d2) { local_best_d2 = d2; local_best_idx = i; }
            }
            struct { double v; int r; } in, out;
            in.v = local_best_d2;
            in.r = my_rank;
            MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
            double val = (out.r == my_rank && local_best_idx >= 0)
                         ? u_gf(local_best_idx) : 0.0;
            MPI_Bcast(&val, 1, MPI_DOUBLE, out.r, MPI_COMM_WORLD);
            return val;
        };
        const int NS = 9;
        if (my_rank == 0) {
            std::cout << "[PROBE] source_type=" << source_type << "\n"
                      << "  Line A  (vary x, y=z=0.5):\n";
        }
        for (int i = 0; i <= NS; ++i) {
            double xq = i / (double)NS;
            double v  = probe(xq, 0.5, 0.5);
            if (my_rank == 0) {
                std::cout << "    u(" << std::fixed << std::setprecision(3) << xq
                          << ",0.500,0.500) = " << std::setprecision(6) << v << "\n";
            }
        }
        if (my_rank == 0) {
            std::cout << "  Line B  (vary y, x=0.5, z=0.5):  "
                         "[if 1D-degenerate every value equals 0.375]\n";
        }
        for (int i = 0; i <= NS; ++i) {
            double yq = i / (double)NS;
            double v  = probe(0.5, yq, 0.5);
            if (my_rank == 0) {
                std::cout << "    u(0.500," << std::fixed << std::setprecision(3) << yq
                          << ",0.500) = " << std::setprecision(6) << v << "\n";
            }
        }
        if (my_rank == 0) {
            std::cout << "  Line C  (vary z, x=0.5, y=0.5):  [same expectation]\n";
        }
        for (int i = 0; i <= NS; ++i) {
            double zq = i / (double)NS;
            double v  = probe(0.5, 0.5, zq);
            if (my_rank == 0) {
                std::cout << "    u(0.500,0.500," << std::fixed << std::setprecision(3) << zq
                          << ") = " << std::setprecision(6) << v << "\n";
            }
        }
    }

    // ParaView output so the user can actually SEE the 3D solution.
    // Writes a paraview/<tag>/ directory; open paraview/<tag>.pvd.
    {
        std::string tag = std::string("nx") + std::to_string(nx)
                        + "_n"   + std::to_string(num_ranks)
                        + "_src" + std::to_string(source_type);
        ParaViewDataCollection pv("u", &pmesh);
        pv.SetPrefixPath("paraview_" + tag);
        pv.RegisterField("u", &u_gf);
        pv.SetLevelsOfDetail(1);
        pv.SetCycle(0);
        pv.SetTime(0.0);
        pv.Save();
        if (my_rank == 0) {
            std::cout << "[VIZ] wrote paraview_" << tag
                      << "/  (open the .pvd file in ParaView)\n";
        }
    }
    } // close PETSc-object scope: pcg, A_petsc, A_hypre etc. destroy here

    MFEMFinalizePetsc();
    return 0;
}
