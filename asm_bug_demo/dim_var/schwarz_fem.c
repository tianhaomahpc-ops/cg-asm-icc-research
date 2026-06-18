/* schwarz_fem.c -- ASM vs sASM overlap study on a LOADED unstructured P1 FEM matrix.
 *
 * Companion to fem_build.py.  Loads the PETSc-binary stiffness K and load b and a
 * node->subdomain membership IS (geometric box OR METIS), builds the base
 * subdomains, then runs CG + PCASM (BASIC or sASM PCSHELL) + ICC(L) local solves
 * with an overlap O, reporting iterations, SETUP and SOLVE wall time, condition
 * number (extreme singular values), Nhat (max multiplicity), omega
 * (max_i lambda_max(M_i^{-1}A_i)), and -- optionally -- the CG Ritz spectrum to a file.
 *
 * Knobs:
 *   -K file -b file -part file   (PETSc binary inputs from fem_build.py)
 *   -scheme {0 BASIC, 3 sASM}   -overlap O   -icc_levels L   -exact
 *   -measure_omega              -eig_out FILE   (dump Ritz values)
 *   -tag STR                    (label echoed in [RESULT])
 */
#include <petscksp.h>
#include <stdlib.h>
#include <string.h>

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

/* build base (non-overlapping) subdomain ISs by grouping nodes with equal part id */
static PetscErrorCode subdomains_from_membership(IS membership, PetscInt *nsub_out,
                                                 IS **is_out)
{
    const PetscInt *m; PetscInt N;
    PetscCall(ISGetLocalSize(membership, &N));
    PetscCall(ISGetIndices(membership, &m));
    PetscInt nparts = 0;
    for (PetscInt i = 0; i < N; ++i) if (m[i] + 1 > nparts) nparts = m[i] + 1;
    PetscInt *cnt = (PetscInt*)calloc(nparts, sizeof(PetscInt));
    for (PetscInt i = 0; i < N; ++i) cnt[m[i]]++;
    PetscInt **buf = (PetscInt**)malloc(nparts * sizeof(PetscInt*));
    PetscInt *pos = (PetscInt*)calloc(nparts, sizeof(PetscInt));
    for (PetscInt p = 0; p < nparts; ++p) buf[p] = (PetscInt*)malloc(cnt[p] * sizeof(PetscInt));
    for (PetscInt i = 0; i < N; ++i) { PetscInt p = m[i]; buf[p][pos[p]++] = i; }
    IS *is = (IS*)malloc(nparts * sizeof(IS));
    for (PetscInt p = 0; p < nparts; ++p)
        PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, cnt[p], buf[p], PETSC_COPY_VALUES, &is[p]));
    PetscCall(ISRestoreIndices(membership, &m));
    for (PetscInt p = 0; p < nparts; ++p) free(buf[p]);
    free(buf); free(pos); free(cnt);
    *nsub_out = nparts; *is_out = is;
    return PETSC_SUCCESS;
}

static PetscErrorCode load_obj(const char *fname, Mat *A, Vec *v, IS *is)
{
    PetscViewer vw;
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, fname, FILE_MODE_READ, &vw));
    if (A) { PetscCall(MatCreate(PETSC_COMM_WORLD, A)); PetscCall(MatSetType(*A, MATAIJ));
             PetscCall(MatLoad(*A, vw)); }
    if (v) { PetscCall(VecCreate(PETSC_COMM_WORLD, v)); PetscCall(VecLoad(*v, vw)); }
    if (is){ PetscCall(ISCreate(PETSC_COMM_WORLD, is)); PetscCall(ISLoad(*is, vw)); }
    PetscCall(PetscViewerDestroy(&vw));
    return PETSC_SUCCESS;
}

int main(int argc, char **argv)
{
    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
    char Kf[512] = "", bf[512] = "", pf[512] = "", tag[128] = "fem", eigf[512] = "";
    int scheme = 0, overlap = 1, iccL = 0;
    PetscBool exact = PETSC_FALSE, measure_omega = PETSC_FALSE, set, do_eig;
    PetscCall(PetscOptionsGetString(NULL, NULL, "-K", Kf, sizeof(Kf), NULL));
    PetscCall(PetscOptionsGetString(NULL, NULL, "-b", bf, sizeof(bf), NULL));
    PetscCall(PetscOptionsGetString(NULL, NULL, "-part", pf, sizeof(pf), NULL));
    PetscCall(PetscOptionsGetString(NULL, NULL, "-tag", tag, sizeof(tag), NULL));
    PetscCall(PetscOptionsGetString(NULL, NULL, "-eig_out", eigf, sizeof(eigf), &do_eig));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-scheme", &scheme, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-overlap", &overlap, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-icc_levels", &iccL, NULL));
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-exact", &exact, &set));
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-measure_omega", &measure_omega, &set));

    Mat A; Vec b, x; IS membership;
    PetscCall(load_obj(Kf, &A, NULL, NULL));
    PetscCall(load_obj(bf, NULL, &b, NULL));
    PetscCall(load_obj(pf, NULL, NULL, &membership));
    PetscCall(VecDuplicate(b, &x)); PetscCall(VecSet(x, 0.0));
    PetscCall(MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE));
    PetscCall(MatSetOption(A, MAT_SPD, PETSC_TRUE));

    PetscInt n_sub; IS *is_sub;
    PetscCall(subdomains_from_membership(membership, &n_sub, &is_sub));

    KSP ksp; PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSetType(ksp, KSPCG));
    PetscCall(KSPSetNormType(ksp, KSP_NORM_PRECONDITIONED));
    PetscCall(KSPSetTolerances(ksp, 1e-8, 1e-50, PETSC_DEFAULT, 3000));
    PetscCall(KSPSetComputeSingularValues(ksp, PETSC_TRUE));
    if (do_eig) PetscCall(KSPSetComputeEigenvalues(ksp, PETSC_TRUE));

    /* ---- setup (timed) ---- */
    PetscLogDouble ts0, ts1, tsolve0, tsolve1;
    PetscCall(PetscTime(&ts0));
    PC innerpc; PetscCall(PCCreate(PETSC_COMM_WORLD, &innerpc));
    PetscCall(PCSetType(innerpc, PCASM));
    PetscCall(PCASMSetType(innerpc, PC_ASM_BASIC));
    PetscCall(PCASMSetLocalSubdomains(innerpc, n_sub, is_sub, NULL));
    PetscCall(PCASMSetOverlap(innerpc, overlap));
    PetscCall(PCSetOperators(innerpc, A, A));
    PetscCall(PCSetUp(innerpc));
    {
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
    /* Nhat from overlapping subdomains */
    PetscReal nhat = 1.0;
    {
        IS *isov; PetscInt nl;
        PetscCall(PCASMGetLocalSubdomains(innerpc, &nl, &isov, NULL));
        Vec mult; PetscCall(MatCreateVecs(A, &mult, NULL)); PetscCall(VecSet(mult, 0.0));
        for (PetscInt i = 0; i < nl; ++i) {
            const PetscInt *idx; PetscInt n;
            PetscCall(ISGetLocalSize(isov[i], &n)); PetscCall(ISGetIndices(isov[i], &idx));
            for (PetscInt j = 0; j < n; ++j) PetscCall(VecSetValue(mult, idx[j], 1.0, ADD_VALUES));
            PetscCall(ISRestoreIndices(isov[i], &idx));
        }
        PetscCall(VecAssemblyBegin(mult)); PetscCall(VecAssemblyEnd(mult));
        PetscCall(VecMax(mult, NULL, &nhat));
        if (scheme == 3) {                          /* build D^{-1/2} */
            Vec invsqrt; PetscCall(VecDuplicate(mult, &invsqrt));
            PetscCall(VecCopy(mult, invsqrt));
            PetscCall(VecReciprocal(invsqrt)); PetscCall(VecSqrtAbs(invsqrt));
            SASMCtx *c = (SASMCtx*)malloc(sizeof(SASMCtx));
            c->inner = innerpc; c->invsqrt = invsqrt;
            PetscCall(MatCreateVecs(A, &c->tmp, NULL));
            PC outer; PetscCall(KSPGetPC(ksp, &outer));
            PetscCall(PCSetType(outer, PCSHELL));
            PetscCall(PCShellSetContext(outer, c));
            PetscCall(PCShellSetApply(outer, sASMApply));
            PetscCall(PCShellSetDestroy(outer, sASMDestroy));
        }
        PetscCall(VecDestroy(&mult));
    }
    if (scheme == 0) PetscCall(KSPSetPC(ksp, innerpc));
    PetscCall(KSPSetUp(ksp));
    PetscCall(PetscTime(&ts1));

    /* ---- solve (timed) ---- */
    PetscCall(PetscTime(&tsolve0));
    PetscCall(KSPSolve(ksp, b, x));
    PetscCall(PetscTime(&tsolve1));

    PetscInt iters, N; KSPConvergedReason reason; PetscReal smax, smin;
    PetscCall(KSPGetIterationNumber(ksp, &iters));
    PetscCall(KSPGetConvergedReason(ksp, &reason));
    PetscCall(KSPComputeExtremeSingularValues(ksp, &smax, &smin));
    PetscCall(MatGetSize(A, &N, NULL));

    if (do_eig) {                                   /* dump CG Ritz spectrum */
        PetscReal *re = (PetscReal*)malloc(sizeof(PetscReal) * (iters + 2));
        PetscReal *im = (PetscReal*)malloc(sizeof(PetscReal) * (iters + 2));
        PetscInt neig;
        PetscCall(KSPComputeEigenvalues(ksp, iters + 2, re, im, &neig));
        FILE *fp = fopen(eigf, "w");
        if (fp) { for (PetscInt i = 0; i < neig; ++i) fprintf(fp, "%.8e\n", (double)re[i]); fclose(fp); }
        free(re); free(im);
    }

    double omega = 1.0;
    if (measure_omega && !exact) {
        IS *isov; PetscInt nl;
        PetscCall(PCASMGetLocalSubdomains(innerpc, &nl, &isov, NULL));
        Mat *subA;
        PetscCall(MatCreateSubMatrices(A, nl, isov, isov, MAT_INITIAL_MATRIX, &subA));
        for (PetscInt i = 0; i < nl; ++i) {
            KSP lk; PetscCall(KSPCreate(PETSC_COMM_SELF, &lk));
            PetscCall(KSPSetOperators(lk, subA[i], subA[i]));
            PetscCall(KSPSetType(lk, KSPCG));
            PetscCall(KSPSetComputeSingularValues(lk, PETSC_TRUE));
            PetscCall(KSPSetTolerances(lk, 1e-10, 1e-50, PETSC_DEFAULT, 500));
            PC lpc; PetscCall(KSPGetPC(lk, &lpc));
            PetscCall(PCSetType(lpc, PCICC)); PetscCall(PCFactorSetLevels(lpc, iccL));
            PetscCall(PCFactorSetShiftType(lpc, MAT_SHIFT_POSITIVE_DEFINITE));
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

    PetscReal kappa = (smin > 0) ? smax / smin : -1.0;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "[RESULT] tag=%s scheme=%d exact=%d iccL=%d overlap=%d nsub=%d | N=%d iter=%d reason=%d "
        "lam_min=%.4e lam_max=%.4e kappa=%.4e Nhat=%.0f omega=%.4f t_setup=%.4f t_solve=%.4f\n",
        tag, scheme, (int)exact, iccL, overlap, (int)n_sub, (int)N, (int)iters, (int)reason,
        (double)smin, (double)smax, (double)kappa, (double)nhat, omega,
        (double)(ts1 - ts0), (double)(tsolve1 - tsolve0)));

    PetscCall(KSPDestroy(&ksp));
    PetscCall(MatDestroy(&A)); PetscCall(VecDestroy(&b)); PetscCall(VecDestroy(&x));
    PetscCall(ISDestroy(&membership));
    for (PetscInt i = 0; i < n_sub; ++i) PetscCall(ISDestroy(&is_sub[i]));
    free(is_sub);
    PetscCall(PetscFinalize());
    return 0;
}
