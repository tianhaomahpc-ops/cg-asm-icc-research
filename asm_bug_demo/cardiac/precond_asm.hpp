// precond_asm.hpp -- the two single-level preconditioners the repo compares,
// applied to the cardiac FEM systems:
//
//   ASM   : CG + PCASM(PC_ASM_BASIC, overlap O) + sub_pc=ICC(L)   [SetupASM]
//           classic additive Schwarz; overlap nodes are OVER-COUNTED, so on an
//           elliptic/stiffness-dominated system the CG iteration count GROWS
//           with overlap (the anomaly this repo isolates).
//   sASM  : CG + PCSHELL[ D^{-1/2} M_BASIC^{-1} D^{-1/2} ] + ICC(L)  [InstallScaledASM]
//           scaled additive Schwarz; D=diag(multiplicity) removes the
//           over-count.  Symmetric => CG-valid; overlap helps (or is neutral).
//
// Both are extracted faithfully from asm_demo.cpp (weight_mode 0 only).
#ifndef PRECOND_ASM_HPP
#define PRECOND_ASM_HPP

#include <petscksp.h>
#include <vector>

// ---- sASM PCSHELL: apply z = W . M_BASIC^{-1} . (W r),  W = D^{-1/2} --------
struct SASMCtx { PC inner_pc; Vec w; Vec tmp; };

extern "C" inline PetscErrorCode SASMApply(PC pc, Vec r, Vec z)
{
    SASMCtx *ctx = nullptr;
    PetscCall(PCShellGetContext(pc, (void**)&ctx));
    PetscCall(VecPointwiseMult(ctx->tmp, ctx->w, r));   // tmp = W r
    PetscCall(PCApply(ctx->inner_pc, ctx->tmp, z));     // z = M_BASIC^{-1} tmp
    PetscCall(VecPointwiseMult(z, ctx->w, z));          // z = W z
    return PETSC_SUCCESS;
}
extern "C" inline PetscErrorCode SASMDestroy(PC pc)
{
    SASMCtx *ctx = nullptr;
    PetscCall(PCShellGetContext(pc, (void**)&ctx));
    if (ctx){ PCDestroy(&ctx->inner_pc); VecDestroy(&ctx->w); VecDestroy(&ctx->tmp); delete ctx; }
    PetscCall(PCShellSetContext(pc, nullptr));
    return PETSC_SUCCESS;
}

// configure an already-set-up PCASM's local blocks as preonly + ICC(L)
static inline void SetSubICC(PC asm_pc, PetscInt icc_levels)
{
    KSP *subs = nullptr; PetscInt nl = 0, first = 0;
    PCASMGetSubKSP(asm_pc, &nl, &first, &subs);
    for (PetscInt i = 0; i < nl; ++i)
    {
        KSPSetType(subs[i], KSPPREONLY);
        PC sp = nullptr; KSPGetPC(subs[i], &sp);
        PCSetType(sp, PCICC); PCFactorSetLevels(sp, icc_levels);
    }
}

// Split A's local row range into `nsub` contiguous (non-overlapping) blocks
// and hand them to PCASM; PCASMSetOverlap then grows the overlap layers.
// This gives multiple subdomains per rank, so the overlap/over-counting
// effect shows even sequentially (PETSc's own PCASMCreateSubdomains style).
static inline void SetBlocks(PC asm_pc, Mat A, PetscInt nsub, PetscInt overlap)
{
    PetscInt rs, re; MatGetOwnershipRange(A, &rs, &re);
    const PetscInt n = re - rs;
    std::vector<IS> is(nsub);
    for (PetscInt b = 0; b < nsub; ++b)
    {
        PetscInt lo = (b*n)/nsub, hi = ((b+1)*n)/nsub;
        ISCreateStride(PETSC_COMM_SELF, hi-lo, rs+lo, 1, &is[b]);
    }
    PCASMSetLocalSubdomains(asm_pc, nsub, is.data(), NULL);
    PCASMSetOverlap(asm_pc, overlap);
    for (PetscInt b = 0; b < nsub; ++b) ISDestroy(&is[b]); // PCASM keeps a ref
}

// ---- plain ASM (BASIC) + ICC on the outer KSP's PC -------------------------
static inline void SetupASM(KSP ksp, Mat A, PetscInt overlap, PetscInt icc_levels,
                            PetscInt nsub)
{
    PC pc = nullptr; KSPGetPC(ksp, &pc);
    PCSetType(pc, PCASM);
    PCASMSetType(pc, PC_ASM_BASIC);
    PCSetOperators(pc, A, A);
    SetBlocks(pc, A, nsub, overlap);
    PCSetUp(pc);
    SetSubICC(pc, icc_levels);
}

// ---- sASM: inner PCASM(BASIC)+ICC wrapped in D^{-1/2}(.)D^{-1/2} PCSHELL ----
static inline void InstallScaledASM(KSP ksp, Mat A, PetscInt overlap, PetscInt icc_levels,
                                    PetscInt nsub)
{
    MPI_Comm comm = PetscObjectComm((PetscObject)A);
    PC inner = nullptr;
    PCCreate(comm, &inner);
    PCSetType(inner, PCASM);
    PCASMSetType(inner, PC_ASM_BASIC);
    PCSetOperators(inner, A, A);
    SetBlocks(inner, A, nsub, overlap);
    PCSetUp(inner);
    SetSubICC(inner, icc_levels);

    // multiplicity m[k] = #subdomains containing dof k (with overlap)
    Vec mult = nullptr; MatCreateVecs(A, &mult, NULL); VecSet(mult, 0.0);
    PetscInt n_sub = 0; IS *is_ovl = nullptr, *is_loc = nullptr;
    PCASMGetLocalSubdomains(inner, &n_sub, &is_ovl, &is_loc);
    for (PetscInt i = 0; i < n_sub; ++i)
    {
        const PetscInt *idx = nullptr; PetscInt n = 0;
        ISGetLocalSize(is_ovl[i], &n);
        ISGetIndices(is_ovl[i], &idx);
        std::vector<PetscScalar> ones(n, 1.0);
        VecSetValues(mult, n, idx, ones.data(), ADD_VALUES);
        ISRestoreIndices(is_ovl[i], &idx);
    }
    VecAssemblyBegin(mult); VecAssemblyEnd(mult);

    Vec w = nullptr; VecDuplicate(mult, &w); VecCopy(mult, w);
    VecReciprocal(w); VecSqrtAbs(w);            // w = 1/sqrt(m[k]) = D^{-1/2}
    VecDestroy(&mult);

    PC outer = nullptr; KSPGetPC(ksp, &outer);
    PCSetType(outer, PCSHELL);
    SASMCtx *ctx = new SASMCtx; ctx->inner_pc = inner; ctx->w = w;
    MatCreateVecs(A, &ctx->tmp, NULL);
    PCShellSetContext(outer, ctx);
    PCShellSetApply(outer, SASMApply);
    PCShellSetDestroy(outer, SASMDestroy);
    PCShellSetName(outer, "sASM");
    PCSetUp(outer);
}

// ---- solve A x = b with CG + (ASM | sASM); return iteration count -----------
// `sasm`=false -> ASM(BASIC); true -> sASM.  A may carry a const MatNullSpace
// (singular Sys2): CG then projects it out.
static inline int CountIters(Mat A, Vec b, Vec x, bool sasm,
                             PetscInt overlap, PetscInt icc_levels, double rtol,
                             PetscInt nsub)
{
    KSP ksp = nullptr;
    KSPCreate(PetscObjectComm((PetscObject)A), &ksp);
    KSPSetType(ksp, KSPCG);
    KSPSetNormType(ksp, KSP_NORM_PRECONDITIONED);
    KSPSetOperators(ksp, A, A);
    KSPSetTolerances(ksp, rtol, 1e-50, PETSC_DEFAULT, 2000);
    if (sasm) InstallScaledASM(ksp, A, overlap, icc_levels, nsub);
    else      SetupASM       (ksp, A, overlap, icc_levels, nsub);
    VecSet(x, 0.0);
    KSPSolve(ksp, b, x);
    PetscInt its = -1; KSPGetIterationNumber(ksp, &its);
    KSPConvergedReason rsn; KSPGetConvergedReason(ksp, &rsn);
    if (rsn < 0) its = -its;            // negative => DIVERGED (flag it)
    KSPDestroy(&ksp);
    return (int)its;
}

#endif // PRECOND_ASM_HPP
