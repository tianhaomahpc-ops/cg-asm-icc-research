/* recoverue_demo.cpp
 *
 * "Recover extracellular potential" model problem with ALL-NEUMANN boundary
 * conditions on the unit cube, P1 tetrahedral FE.  This mirrors the
 * pseudo-bidomain `u_e` recovery solve in cardioid's
 *   hack/femheart.cpp  (Sys2, recoverue_).
 *
 * PDE (with sigma_i = sigma_e = sigma = 1 for simplicity):
 *
 *     -div(sigma * grad u) = -div(sigma_i * grad V_m)   on [0,1]^3
 *      du/dn               = 0                          on all 6 faces
 *
 * Discrete system:   A u = A V_m   (same stiffness on both sides, since
 * sigma == sigma_i here).  A is SINGULAR with null space spanned by the
 * constant vector.
 *
 * Reference: pick V_m(x,y,z) = cos(pi x) cos(pi y) cos(pi z).  It satisfies
 * the Neumann BC exactly and has zero mean.  Then the analytical solution
 * is u = V_m + const.
 *
 * Compares two single-level preconditioners (CG outer):
 *   scheme 0 : PCASM BASIC          (unscaled, "iter up with overlap")
 *   scheme 3 : PCSHELL = D^{-1/2} * PCASM_BASIC * D^{-1/2}  (sASM)
 *
 * Build: see Makefile target  recoverue_demo
 */

#include "mfem.hpp"
#include <petsc.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>

using namespace mfem;

// ---------------------------------------------------------------------------
// Filtered Hypre -> PETSc AIJ conversion.  Same body as asm_demo.cpp; uses
// |val| < 1e-12 to drop both BC stored zeros and quadrature round-off
// zeros, so the matrix entries are bit-identical to a clean independent
// assembly.
// ---------------------------------------------------------------------------
static void HypreToPetscAIJ(HypreParMatrix &hypre_mat,
                            PetscParMatrix &petsc_mat,
                            const char *name, int my_rank)
{
    MPI_Comm comm = hypre_mat.GetComm();
    const HYPRE_BigInt rs = hypre_mat.GetRowStarts()[0];
    const HYPRE_BigInt re = hypre_mat.GetRowStarts()[1];
    const HYPRE_BigInt cs = hypre_mat.GetColStarts()[0];
    const HYPRE_BigInt ce = hypre_mat.GetColStarts()[1];
    const PetscInt m = (PetscInt)(re - rs);
    const PetscInt n = (PetscInt)(ce - cs);
    const PetscInt M = (PetscInt)hypre_mat.GetGlobalNumRows();
    const PetscInt N = (PetscInt)hypre_mat.GetGlobalNumCols();
    const PetscInt c0 = (PetscInt)cs, c1 = (PetscInt)ce;
    SparseMatrix merged; hypre_mat.MergeDiagAndOffd(merged);
    const int *I = merged.HostReadI();
    const int *J = merged.HostReadJ();
    const real_t *data = merged.HostReadData();
    const real_t tol = 1e-12;

    std::vector<PetscInt> d_nnz(m, 0), o_nnz(m, 0);
    for (PetscInt i = 0; i < m; ++i)
      for (int p = I[i]; p < I[i+1]; ++p) {
        if (std::abs(data[p]) < tol) continue;
        if (c0 <= (PetscInt)J[p] && (PetscInt)J[p] < c1) ++d_nnz[i];
        else                                              ++o_nnz[i];
      }

    Mat mat = nullptr;
    PetscErrorCode ierr = MatCreateAIJ(comm, m, n, M, N,
        0, m ? d_nnz.data() : nullptr, 0, m ? o_nnz.data() : nullptr, &mat);
    MFEM_VERIFY(ierr == 0, "MatCreateAIJ failed");
    MatSetOption(mat, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);
    MatSetOption(mat, MAT_SYMMETRIC,           PETSC_TRUE);
    MatSetOption(mat, MAT_SYMMETRY_ETERNAL,    PETSC_TRUE);

    for (PetscInt i = 0; i < m; ++i) {
        const PetscInt row = (PetscInt)rs + i;
        for (int p = I[i]; p < I[i+1]; ++p) {
            if (std::abs(data[p]) < tol) continue;
            const PetscInt    col = (PetscInt)J[p];
            const PetscScalar val = (PetscScalar)data[p];
            MatSetValues(mat, 1, &row, 1, &col, &val, INSERT_VALUES);
        }
    }
    MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (mat, MAT_FINAL_ASSEMBLY);
    petsc_mat.SetMat(mat);
    MatDestroy(&mat);
    if (my_rank == 0) {
        std::cout << "[CONVERT] " << name
                  << "  global " << M << " x " << N << "\n";
    }
}

// ---------------------------------------------------------------------------
// PCSHELL = scaled additive Schwarz   D^{-1/2} * PCASM_BASIC * D^{-1/2}
// Same machinery as asm_demo.cpp (scheme 3).
// ---------------------------------------------------------------------------
struct SASMCtx { PC inner; Vec invsqrt; Vec tmp; };

extern "C" PetscErrorCode SASMApply(PC pc, Vec r, Vec z) {
    SASMCtx *ctx = nullptr;
    PetscCall(PCShellGetContext(pc, (void**)&ctx));
    PetscCall(VecPointwiseMult(ctx->tmp, ctx->invsqrt, r));
    PetscCall(PCApply(ctx->inner, ctx->tmp, z));
    PetscCall(VecPointwiseMult(z, ctx->invsqrt, z));
    return PETSC_SUCCESS;
}
extern "C" PetscErrorCode SASMDestroy(PC pc) {
    SASMCtx *ctx = nullptr;
    PetscCall(PCShellGetContext(pc, (void**)&ctx));
    if (ctx) {
        PCDestroy(&ctx->inner);
        VecDestroy(&ctx->invsqrt);
        VecDestroy(&ctx->tmp);
        delete ctx;
    }
    PCShellSetContext(pc, nullptr);
    return PETSC_SUCCESS;
}

static void InstallScaledASM(KSP ksp, Mat A,
                             PetscInt overlap, PetscInt icc_levels,
                             int my_rank)
{
    PC inner = nullptr;
    PCCreate(PetscObjectComm((PetscObject)A), &inner);
    PCSetType(inner, PCASM);
    PCASMSetType(inner, PC_ASM_BASIC);
    PCASMSetOverlap(inner, overlap);
    PCSetOperators(inner, A, A);
    PCSetUp(inner);
    {
        KSP     *subksp = nullptr;
        PetscInt n_loc = 0, first = 0;
        PCASMGetSubKSP(inner, &n_loc, &first, &subksp);
        for (PetscInt i = 0; i < n_loc; ++i) {
            KSPSetType(subksp[i], KSPPREONLY);
            PC sub = nullptr;
            KSPGetPC(subksp[i], &sub);
            PCSetType(sub, PCICC);
            PCFactorSetLevels(sub, icc_levels);
        }
        PCSetUpOnBlocks(inner);
    }
    // multiplicity vector
    Vec mult, invsqrt;
    MatCreateVecs(A, &mult, NULL);
    VecSet(mult, 0.0);
    {
        PetscInt n_loc = 0;
        IS *is_full = nullptr, *is_loc = nullptr;
        PCASMGetLocalSubdomains(inner, &n_loc, &is_full, &is_loc);
        for (PetscInt i = 0; i < n_loc; ++i) {
            const PetscInt *idx = nullptr;
            PetscInt n = 0;
            ISGetLocalSize(is_full[i], &n);
            ISGetIndices(is_full[i], &idx);
            std::vector<PetscScalar> ones(n > 0 ? n : 1, 1.0);
            VecSetValues(mult, n, idx, ones.data(), ADD_VALUES);
            ISRestoreIndices(is_full[i], &idx);
        }
        VecAssemblyBegin(mult);
        VecAssemblyEnd  (mult);
    }
    VecDuplicate(mult, &invsqrt);
    VecCopy(mult, invsqrt);
    VecReciprocal(invsqrt);
    VecSqrtAbs(invsqrt);
    {
        PetscReal mn, mx;
        VecMin(mult, NULL, &mn);
        VecMax(mult, NULL, &mx);
        if (my_rank == 0) {
            std::cout << "[sASM] multiplicity range = ["
                      << (int)mn << ", " << (int)mx << "]\n";
        }
    }
    VecDestroy(&mult);

    PC outer = nullptr;
    KSPGetPC(ksp, &outer);
    PCSetType(outer, PCSHELL);
    SASMCtx *ctx = new SASMCtx;
    ctx->inner  = inner;
    ctx->invsqrt = invsqrt;
    MatCreateVecs(A, &ctx->tmp, NULL);
    PCShellSetContext(outer, ctx);
    PCShellSetApply  (outer, SASMApply);
    PCShellSetDestroy(outer, SASMDestroy);
    PCShellSetName   (outer, "sASM_recoverue");
    PCSetUp(outer);
    if (my_rank == 0) {
        std::cout << "[sASM] PCSHELL installed  overlap=" << overlap
                  << "  icc=" << icc_levels << "\n";
    }
}

// ---------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    Mpi::Init(argc, argv);
    int my_rank   = Mpi::WorldRank();
    int num_ranks = Mpi::WorldSize();

    // Knobs (parsed by hand BEFORE PETSc init).
    int nx      = 24;
    int scheme  = 0;     // 0 = ASM, 3 = sASM
    int overlap = 1;
    int icc_lev = 0;
    {
        int out = 1;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "-nx"      && i+1 < argc) { nx      = std::atoi(argv[++i]); continue; }
            if (a == "-scheme"  && i+1 < argc) { scheme  = std::atoi(argv[++i]); continue; }
            if (a == "-overlap" && i+1 < argc) { overlap = std::atoi(argv[++i]); continue; }
            if (a == "-icc"     && i+1 < argc) { icc_lev = std::atoi(argv[++i]); continue; }
            argv[out++] = argv[i];
        }
        argc = out;
    }

    MFEMInitializePetsc(&argc, &argv, NULL, NULL);
    PetscOptionsSetValue(NULL, "-options_left", "no");

    if (my_rank == 0) {
        std::cout << "================================================\n"
                  << "  recoverue_demo  nx=" << nx
                  << "  ranks=" << num_ranks
                  << "  scheme=" << scheme
                  << " (" << (scheme == 0 ? "CG+ASM" : "CG+sASM") << ")"
                  << "  overlap=" << overlap
                  << "  icc=" << icc_lev << "\n"
                  << "  all-Neumann singular Laplacian   "
                  << "u = cos(pi x) cos(pi y) cos(pi z)\n"
                  << "================================================\n";
    }

    {  // PETSc-object lifetime scope (must end BEFORE MFEMFinalizePetsc)

        // ----- 1. Mesh: cube hex split into 6 tets (MFEM hex_to_tet)
        Mesh serial = Mesh::MakeCartesian3D(
            nx, nx, nx, Element::TETRAHEDRON, 1.0, 1.0, 1.0);
        ParMesh pmesh(MPI_COMM_WORLD, serial);
        serial.Clear();

        // ----- 2. P1 FE space
        H1_FECollection fec(/*order=*/1, /*dim=*/3);
        ParFiniteElementSpace fes(&pmesh, &fec);
        const HYPRE_BigInt ndof = fes.GlobalTrueVSize();
        if (my_rank == 0) {
            std::cout << "[FES] global true DOFs = " << ndof << "\n";
        }

        // ----- 3. All-Neumann: no essential DOFs
        Array<int> ess_tdof_list;     // intentionally left empty

        // ----- 4. Bilinear form for the unit Laplacian (sigma = 1)
        ConstantCoefficient sigma(1.0);
        ParBilinearForm a(&fes);
        a.AddDomainIntegrator(new DiffusionIntegrator(sigma));
        a.Assemble();

        // ----- 5. Reference V_m = cos(pi x) cos(pi y) cos(pi z)
        ParGridFunction Vm_gf(&fes);
        FunctionCoefficient Vm_coef([](const Vector &x) {
            return std::cos(M_PI * x[0]) *
                   std::cos(M_PI * x[1]) *
                   std::cos(M_PI * x[2]);
        });
        Vm_gf.ProjectCoefficient(Vm_coef);

        // ----- 6. Assemble HypreParMatrix.  Use FormLinearSystem with an
        //         empty essential-list and zero RHS just to extract A.
        ParGridFunction u_gf(&fes);
        u_gf = 0.0;
        ParLinearForm dummy_lf(&fes);
        dummy_lf = 0.0;
        HypreParMatrix A_hypre;
        Vector U_tmp, B_tmp;
        a.FormLinearSystem(ess_tdof_list, u_gf, dummy_lf,
                           A_hypre, U_tmp, B_tmp);

        // ----- 7. Build RHS  B = A * V_m   (the "recoverue" RHS pattern)
        Vector Vm_true(fes.GetTrueVSize());
        Vm_gf.GetTrueDofs(Vm_true);
        Vector B(Vm_true.Size());
        A_hypre.Mult(Vm_true, B);

        Vector U(Vm_true.Size());
        U = 0.0;

        // ----- 8. Convert to PETSc AIJ
        PetscParMatrix A_petsc;
        HypreToPetscAIJ(A_hypre, A_petsc, "recoverue_A", my_rank);

        // ----- 9. Attach the constant null space to A  (this is the
        //         critical bit for the singular all-Neumann system)
        MatNullSpace nsp = nullptr;
        MatNullSpaceCreate(MPI_COMM_WORLD, PETSC_TRUE,
                           0, NULL, &nsp);
        MatSetNullSpace         ((Mat)A_petsc, nsp);
        MatSetTransposeNullSpace((Mat)A_petsc, nsp);

        // Project RHS so it lives in range(A) (KSP does this internally
        // anyway, but doing it here keeps the [SOLN] residual clean).
        {
            Vec B_petsc = nullptr;
            VecCreateMPIWithArray(MPI_COMM_WORLD, 1, B.Size(),
                                  PETSC_DECIDE, B.HostRead(), &B_petsc);
            MatNullSpaceRemove(nsp, B_petsc);
            VecDestroy(&B_petsc);
        }

        // ----- 10. Set up CG + (ASM or sASM)
        PetscPCGSolver pcg(MPI_COMM_WORLD, std::string(), /*iter_mode=*/true);
        pcg.SetOperator(A_petsc);
        pcg.SetMaxIter(2000);
        pcg.SetRelTol(1e-6);
        pcg.SetAbsTol(1e-12);
        pcg.SetPrintLevel(0);

        // For BOTH schemes we need the same outer-CG options to land first,
        // then for scheme 3 we replace the outer PC with PCSHELL.
        // Inject the common ASM options for scheme 0 (these go through
        // KSPSetFromOptions inside pcg.Customize / pcg.Mult).
        if (scheme == 0) {
            PetscOptionsSetValue(NULL, "-pc_type",                "asm");
            PetscOptionsSetValue(NULL, "-pc_asm_type",            "basic");
            std::string ov = std::to_string(overlap);
            PetscOptionsSetValue(NULL, "-pc_asm_overlap",         ov.c_str());
            PetscOptionsSetValue(NULL, "-sub_ksp_type",           "preonly");
            PetscOptionsSetValue(NULL, "-sub_pc_type",            "icc");
            std::string il = std::to_string(icc_lev);
            PetscOptionsSetValue(NULL, "-sub_pc_factor_levels",   il.c_str());
        } else {
            // scheme 3: still set asm options so InstallScaledASM's
            // internal inner_pc could be configured by user CLI, but the
            // actual sub-PC is set via direct API in InstallScaledASM.
            // Empty body on purpose.
        }
        // Always use the preconditioned norm as the convergence measure.
        PetscOptionsSetValue(NULL, "-ksp_norm_type", "preconditioned");
        PetscOptionsSetValue(NULL, "-ksp_initial_guess_nonzero", "false");

        if (scheme == 3) {
            pcg.Customize(true);   // KSPSetFromOptions runs now
            KSP ksp_raw = static_cast<KSP>(pcg);
            Mat A_raw = nullptr;
            KSPGetOperators(ksp_raw, &A_raw, NULL);
            InstallScaledASM(ksp_raw, A_raw, overlap, icc_lev, my_rank);
        }

        // ----- 11. Solve, time it
        double t0 = MPI_Wtime();
        pcg.Mult(B, U);
        double t1 = MPI_Wtime();
        int    iters = pcg.GetNumIterations();
        double pnorm = pcg.GetFinalNorm();

        // ----- 12. Diagnostics: mean-gauge, error vs Vm, true residual
        // mean(U) over global DOFs
        auto sum_global = [&](const Vector &v) {
            double s = 0.0;
            for (int i = 0; i < v.Size(); ++i) s += v(i);
            double g = 0.0;
            MPI_Allreduce(&s, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            return g;
        };
        const double meanU  = sum_global(U)       / (double)ndof;
        const double meanVm = sum_global(Vm_true) / (double)ndof;

        // err = (U - meanU) - (Vm - meanVm)
        Vector err(U.Size()), Vm0(U.Size()), U0(U.Size());
        for (int i = 0; i < U.Size(); ++i) {
            U0(i)  = U(i)        - meanU;
            Vm0(i) = Vm_true(i)  - meanVm;
            err(i) = U0(i) - Vm0(i);
        }
        auto nrm = [&](const Vector &v) {
            double s = 0;
            for (int i = 0; i < v.Size(); ++i) s += v(i)*v(i);
            double g = 0;
            MPI_Allreduce(&s, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            return std::sqrt(g);
        };
        double err_l2 = nrm(err);
        double Vm0_l2 = nrm(Vm0);
        double U0_l2  = nrm(U0);

        // True residual b - A u
        Vector Au(U.Size()), resid(U.Size());
        A_hypre.Mult(U, Au);
        for (int i = 0; i < U.Size(); ++i) resid(i) = B(i) - Au(i);
        double r_l2 = nrm(resid);
        double b_l2 = nrm(B);

        if (my_rank == 0) {
            std::cout << "[RESULT]"
                      << "  scheme=" << scheme
                      << "  iters="  << iters
                      << "  time="   << std::fixed << std::setprecision(3) << (t1-t0) << " s"
                      << "  pnorm="  << std::scientific << std::setprecision(3) << pnorm
                      << "\n";
            std::cout << "[SOLN]"
                      << "  meanU="          << std::scientific << std::setprecision(3) << meanU
                      << "  ||r||/||b||="    << (r_l2 / b_l2)
                      << "  ||u_0||="        << std::setprecision(6) << U0_l2
                      << "  ||u_0 - Vm_0||/||Vm_0||=" << std::setprecision(3)
                      << (err_l2 / Vm0_l2)
                      << "\n";
        }

        MatNullSpaceDestroy(&nsp);
    } // close PETSc object scope

    MFEMFinalizePetsc();
    return 0;
}
