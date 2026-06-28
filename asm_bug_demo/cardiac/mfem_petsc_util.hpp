// mfem_petsc_util.hpp -- shared MFEM 4.9 -> PETSc 3.24 glue for the cardiac
// forward-ECG driver (forward_ecg.cpp).  The HypreToPetscAIJ converter is the
// same "femheart pattern" used in asm_demo.cpp (one MatSetValues per stored
// entry, exact preallocation), factored out so the cardiac systems plug into
// the SAME PCASM-overlap / cross-system-preconditioning study the repo is
// about.  Plus two helpers for the singular pure-Neumann Sys2 (u_e recovery).
#ifndef MFEM_PETSC_UTIL_HPP
#define MFEM_PETSC_UTIL_HPP

#include "mfem.hpp"
#include <petsc.h>
#include <vector>
#include <cmath>
#include <iostream>

using namespace mfem;

// ---------------------------------------------------------------------------
// Hypre -> PETSc AIJ converter (mirrors asm_demo.cpp::HypreToPetscAIJ).
// fix_level>=1 drops stored zeros so the PCASM overlap walk sees the true
// sparsity; fix_level>=2 flags MAT_SYMMETRIC/MAT_SPD for the SBAIJ-ICC path.
// For the cardiac SPD systems we always pass fix_level>=1, assume_spd=true.
// ---------------------------------------------------------------------------
static inline void HypreToPetscAIJ(HypreParMatrix &hypre_mat,
                                   PetscParMatrix &petsc_mat,
                                   const char *name,
                                   int my_rank,
                                   int fix_level,
                                   bool assume_spd)
{
    MPI_Comm comm = hypre_mat.GetComm();
    const HYPRE_BigInt row_start_big = hypre_mat.GetRowStarts()[0];
    const HYPRE_BigInt row_end_big   = hypre_mat.GetRowStarts()[1];
    const HYPRE_BigInt col_start_big = hypre_mat.GetColStarts()[0];
    const HYPRE_BigInt col_end_big   = hypre_mat.GetColStarts()[1];

    const PetscInt local_rows  = (PetscInt)(row_end_big - row_start_big);
    const PetscInt local_cols  = (PetscInt)(col_end_big - col_start_big);
    const PetscInt global_rows = (PetscInt)hypre_mat.GetGlobalNumRows();
    const PetscInt global_cols = (PetscInt)hypre_mat.GetGlobalNumCols();
    const PetscInt col_start   = (PetscInt)col_start_big;
    const PetscInt col_end     = (PetscInt)col_end_big;

    SparseMatrix merged;
    hypre_mat.MergeDiagAndOffd(merged);
    MFEM_VERIFY(merged.Height() == local_rows, "row count mismatch in conversion");

    const int    *I    = merged.HostReadI();
    const int    *J    = merged.HostReadJ();
    const real_t *data = merged.HostReadData();

    const real_t zero_tol = 1e-12;
    std::vector<PetscInt> d_nnz(local_rows, 0), o_nnz(local_rows, 0);
    for (PetscInt i = 0; i < local_rows; ++i)
        for (int p = I[i]; p < I[i+1]; ++p)
        {
            if (fix_level >= 1 && std::abs(data[p]) < zero_tol) continue;
            const PetscInt col = (PetscInt)J[p];
            (col_start <= col && col < col_end) ? ++d_nnz[i] : ++o_nnz[i];
        }

    Mat mat = nullptr;
    PetscErrorCode ierr = MatCreateAIJ(
        comm, local_rows, local_cols, global_rows, global_cols,
        0, local_rows ? d_nnz.data() : nullptr,
        0, local_rows ? o_nnz.data() : nullptr, &mat);
    MFEM_VERIFY(ierr == 0, "MatCreateAIJ failed");
    if (name) PetscObjectSetName((PetscObject)mat, name);
    if (fix_level >= 1) MatSetOption(mat, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);

    for (PetscInt i = 0; i < local_rows; ++i)
    {
        const PetscInt row = (PetscInt)row_start_big + i;
        for (int p = I[i]; p < I[i+1]; ++p)
        {
            if (fix_level >= 1 && std::abs(data[p]) < zero_tol) continue;
            const PetscInt    col   = (PetscInt)J[p];
            const PetscScalar value = (PetscScalar)data[p];
            ierr = MatSetValues(mat, 1, &row, 1, &col, &value, INSERT_VALUES);
            MFEM_VERIFY(ierr == 0, "MatSetValues failed");
        }
    }
    MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY);

    if (fix_level >= 2)
    {
        MatSetOption(mat, MAT_SYMMETRIC, PETSC_TRUE);
        MatSetOption(mat, MAT_SYMMETRY_ETERNAL, PETSC_TRUE);
        if (assume_spd) MatSetOption(mat, MAT_SPD, PETSC_TRUE);
    }

    petsc_mat.SetMat(mat);
    MatDestroy(&mat);
    if (my_rank == 0)
        std::cout << "[CONVERT] " << (name ? name : "")
                  << " global=" << global_rows << "x" << global_cols
                  << " fix_level=" << fix_level << "\n";
}

// ---------------------------------------------------------------------------
// Singular pure-Neumann helpers (Sys2: K_{sigma_i+sigma_e} u_e = -K_i Vm).
// ---------------------------------------------------------------------------

// Make a RHS compatible with ker(K)=span{1}: subtract the GLOBAL mean so
// sum(B)=0 (identical to asm_demo.cpp's pure_neumann branch).
static inline void RemoveGlobalMean(Vector &B, MPI_Comm comm)
{
    double loc[2] = {0.0, (double)B.Size()}, glob[2] = {0.0, 0.0};
    for (int i = 0; i < B.Size(); ++i) loc[0] += B(i);
    MPI_Allreduce(loc, glob, 2, MPI_DOUBLE, MPI_SUM, comm);
    const double mean = glob[0] / glob[1];
    for (int i = 0; i < B.Size(); ++i) B(i) -= mean;
}

// Attach the constant null space to a PETSc operator so the outer Krylov
// projects it out (and removes it from the RHS at KSPSolve).
static inline void AttachConstNullSpace(Mat A, MPI_Comm comm)
{
    MatNullSpace nsp = nullptr;
    MatNullSpaceCreate(comm, PETSC_TRUE, 0, nullptr, &nsp);
    MatSetNullSpace(A, nsp);
    MatNullSpaceDestroy(&nsp);
}

#endif // MFEM_PETSC_UTIL_HPP
