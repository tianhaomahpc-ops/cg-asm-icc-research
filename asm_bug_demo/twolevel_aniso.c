/* twolevel_aniso.c -- the conductivity-tensor axis for the coarse space.
 *
 * twolevel.c showed that a Nicolaides coarse space (phi_i(k) = 1/m_k) turns
 * one-level sASM into a scalable method ON AN ISOTROPIC LAPLACIAN.  This file
 * asks the question that matters for cardiac tissue: does that still hold when
 * sigma is a TENSOR?
 *
 * Answer (measured, see REPORT_aniso_pu_zh.md): NO.  The 1/m_k coarse basis is
 * built from the mesh, not from the operator, and under contrast it stops
 * lifting lambda_min.  Two operator-aware replacements are provided, both of
 * which REUSE the sigma-harmonic partition of unity chi_i that -puharm already
 * builds for the fine level, so the coarse space costs nothing extra:
 *
 *   -coarse mult     phi_i(k) = 1/m_k                     (twolevel.c, baseline)
 *   -coarse harm     phi_i    = chi_i                     (reuse, 1 vec/subdomain)
 *   -coarse enrich   phi_i    = chi_i * {1, s, t}         (reuse, 3 vec/subdomain)
 *                    with (s,t) the fibre-aligned local coordinates
 *
 * Operator: P1 FEM on a structured triangulated square for -div(sigma grad u),
 * sigma = R(theta) diag(sigma_l, sigma_t) R(theta)^T, det-normalised so that the
 * CONTRAST r = sigma_l/sigma_t is the only knob.  Full Dirichlet boundary.
 * (P1 FEM rather than a 9-point FD stencil: guaranteed SPD for ANY SPD tensor
 * and any fibre angle, which a cross-derivative FD stencil is not.)
 *
 * Options:  -n 120  -P 6  -O 2  -aniso r  -fiber_deg theta  -coarse mult|harm|enrich
 *           -scale  (run the scalability sweep instead of the fixed config)
 *
 * Build:  mpicc -O3 -std=c11 -I$PETSC_DIR/include twolevel_aniso.c \
 *               -o twolevel_aniso -L$PETSC_DIR/lib -lpetsc
 * Run:    ./twolevel_aniso -aniso 100 -fiber_deg 45 -coarse harm
 */
#include <petscksp.h>
#include <math.h>
#include <string.h>

#ifndef M_PI                 /* -std=c11 hides it; the Makefile builds with c11 */
#define M_PI 3.14159265358979323846
#endif

typedef struct { PetscInt n, *idx, *depth; IS is; Mat Ai; KSP ksp; Vec ri, yi; } Sub;
typedef struct {
    Sub *subs; PetscInt nsub; Vec wflat, tmp;            /* fine level */
    Vec *wsub;                                            /* per-subdomain weights */
    int twolevel; Mat R0t; KSP kspc; Vec cvec, ycvec;     /* coarse level */
} Ctx;

/* ---------------------------------------------------------------- operator */
/* P1 FEM, structured right triangles on [0,1]^2, u = 0 on the whole boundary.
 * Unknowns are the (n-1)^2 interior vertices, numbered (j-1)*(n-1) + (i-1). */
static Mat Diffusion2D(PetscInt n, PetscReal sxx, PetscReal sxy, PetscReal syy,
                       PetscInt *Nout)
{
    PetscInt m = n - 1, N = m * m;              /* interior vertices */
    Mat A;
    MatCreateSeqAIJ(PETSC_COMM_SELF, N, N, 9, NULL, &A);
    const PetscReal h = 1.0 / (PetscReal)n, area = 0.5 * h * h;
    /* two triangles per cell: (0,0)(1,0)(1,1) and (0,0)(1,1)(0,1), local
     * vertex offsets in (di,dj) and the barycentric gradients (gx,gy). */
    const int off[2][3][2] = {{{0,0},{1,0},{1,1}}, {{0,0},{1,1},{0,1}}};
    for (PetscInt cj = 0; cj < n; ++cj)
    for (PetscInt ci = 0; ci < n; ++ci)
    for (int t = 0; t < 2; ++t) {
        PetscReal X[3], Y[3];  PetscInt vi[3], vj[3];
        for (int a = 0; a < 3; ++a) {
            vi[a] = ci + off[t][a][0]; vj[a] = cj + off[t][a][1];
            X[a] = vi[a] * h;          Y[a] = vj[a] * h;
        }
        PetscReal det = (X[1]-X[0])*(Y[2]-Y[0]) - (X[2]-X[0])*(Y[1]-Y[0]);
        PetscReal bg[3], cg[3];
        bg[0] = (Y[1]-Y[2])/det; bg[1] = (Y[2]-Y[0])/det; bg[2] = (Y[0]-Y[1])/det;
        cg[0] = (X[2]-X[1])/det; cg[1] = (X[0]-X[2])/det; cg[2] = (X[1]-X[0])/det;
        for (int a = 0; a < 3; ++a) {
            if (vi[a] == 0 || vi[a] == n || vj[a] == 0 || vj[a] == n) continue;
            PetscInt ra = (vj[a]-1)*m + (vi[a]-1);
            for (int b = 0; b < 3; ++b) {
                if (vi[b] == 0 || vi[b] == n || vj[b] == 0 || vj[b] == n) continue;
                PetscInt rb = (vj[b]-1)*m + (vi[b]-1);
                PetscScalar v = area * (sxx*bg[a]*bg[b] + sxy*(bg[a]*cg[b] + cg[a]*bg[b])
                                        + syy*cg[a]*cg[b]);
                MatSetValue(A, ra, rb, v, ADD_VALUES);
            }
        }
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE);
    *Nout = N; return A;
}

/* ------------------------------------------------------------- subdomains */
static void BuildSub(Mat A, PetscInt *idx, PetscInt *dep, PetscInt ni, Sub *s) {
    s->n = ni;
    s->idx   = (PetscInt*)malloc(sizeof(PetscInt)*ni);
    s->depth = (PetscInt*)malloc(sizeof(PetscInt)*ni);
    memcpy(s->idx, idx, sizeof(PetscInt)*ni);
    memcpy(s->depth, dep, sizeof(PetscInt)*ni);
    ISCreateGeneral(PETSC_COMM_SELF, ni, s->idx, PETSC_COPY_VALUES, &s->is);
    MatCreateSubMatrix(A, s->is, s->is, MAT_INITIAL_MATRIX, &s->Ai);
    KSPCreate(PETSC_COMM_SELF, &s->ksp); KSPSetType(s->ksp, KSPPREONLY);
    KSPSetOperators(s->ksp, s->Ai, s->Ai);
    PC pc; KSPGetPC(s->ksp, &pc); PCSetType(pc, PCICC); PCFactorSetLevels(pc, 0);
    PCFactorSetShiftType(pc, MAT_SHIFT_POSITIVE_DEFINITE);
    KSPSetUp(s->ksp); MatCreateVecs(s->Ai, &s->ri, &s->yi);
}

/* P*P box subdomains on the m x m interior grid, grown by O layers; depth =
 * how many layers out from the owned core (0 = core), exactly as PETSc's
 * MatIncreaseOverlap would produce it for this box decomposition. */
static Sub *BuildAll(Mat A, PetscInt m, int P, int O, PetscInt *nsubOut) {
    int NS = P*P; Sub *S = (Sub*)malloc(sizeof(Sub)*NS); int sc = 0;
    for (int q = 0; q < P; ++q) for (int p = 0; p < P; ++p) {
        PetscInt alo=(p*m)/P, ahi=((p+1)*m)/P, blo=(q*m)/P, bhi=((q+1)*m)/P;
        PetscInt al=alo-O<0?0:alo-O, ar=ahi+O>m?m:ahi+O;
        PetscInt bl=blo-O<0?0:blo-O, br=bhi+O>m?m:bhi+O;
        PetscInt ni=(ar-al)*(br-bl), cc=0;
        PetscInt *idx=(PetscInt*)malloc(sizeof(PetscInt)*ni);
        PetscInt *dep=(PetscInt*)malloc(sizeof(PetscInt)*ni);
        for (PetscInt b=bl;b<br;++b) for (PetscInt a=al;a<ar;++a) {
            PetscInt da = (a<alo)?(alo-a):((a>=ahi)?(a-ahi+1):0);
            PetscInt db = (b<blo)?(blo-b):((b>=bhi)?(b-bhi+1):0);
            dep[cc] = da > db ? da : db;            /* Chebyshev graph distance */
            idx[cc++] = b*m + a;
        }
        BuildSub(A, idx, dep, ni, &S[sc++]); free(idx); free(dep);
    }
    *nsubOut = NS; return S;
}
static void FreeAll(Sub *S, int n) {
    for (int i=0;i<n;++i){ ISDestroy(&S[i].is); MatDestroy(&S[i].Ai); KSPDestroy(&S[i].ksp);
        VecDestroy(&S[i].ri); VecDestroy(&S[i].yi); free(S[i].idx); free(S[i].depth); }
    free(S);
}

/* ------------------------------- the sigma-harmonic partition of unity ---- */
/* chi_i: 1 on the owned core, sigma-harmonic through the overlap band, 0 at the
 * artificial boundary (implicit -- it lies outside the subdomain).  This is the
 * minimiser of int sigma |grad chi|^2 for those data; the graph-distance ramps
 * used everywhere else are the sigma = I special case. */
static void HarmonicPU(Sub *s, PetscReal *w) {
    PetscInt nb = 0, nc = 0;
    for (PetscInt j = 0; j < s->n; ++j) { if (s->depth[j]) nb++; else nc++; }
    for (PetscInt j = 0; j < s->n; ++j) w[j] = (s->depth[j] == 0) ? 1.0 : 0.0;
    if (!nb || !nc) return;
    PetscInt *bi=(PetscInt*)malloc(sizeof(PetscInt)*nb), *ci=(PetscInt*)malloc(sizeof(PetscInt)*nc);
    PetscInt pb=0, pc2=0;
    for (PetscInt j = 0; j < s->n; ++j) { if (s->depth[j]) bi[pb++] = j; else ci[pc2++] = j; }
    IS isb, isc; Mat Abb, Abc;
    ISCreateGeneral(PETSC_COMM_SELF, nb, bi, PETSC_COPY_VALUES, &isb);
    ISCreateGeneral(PETSC_COMM_SELF, nc, ci, PETSC_COPY_VALUES, &isc);
    MatCreateSubMatrix(s->Ai, isb, isb, MAT_INITIAL_MATRIX, &Abb);
    MatCreateSubMatrix(s->Ai, isb, isc, MAT_INITIAL_MATRIX, &Abc);
    Vec ones, rhs, chi;
    MatCreateVecs(Abc, &ones, &rhs); VecSet(ones, 1.0);
    MatMult(Abc, ones, rhs); VecScale(rhs, -1.0); VecDuplicate(rhs, &chi);
    KSP kh; PC pch;
    KSPCreate(PETSC_COMM_SELF, &kh); KSPSetType(kh, KSPCG);
    KSPSetOperators(kh, Abb, Abb);
    KSPSetTolerances(kh, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, 1000);
    KSPGetPC(kh, &pch); PCSetType(pch, PCICC); PCFactorSetLevels(pch, 0);
    PCFactorSetShiftType(pch, MAT_SHIFT_POSITIVE_DEFINITE);
    KSPSolve(kh, rhs, chi);
    const PetscScalar *ca; VecGetArrayRead(chi, &ca);
    for (PetscInt j = 0; j < nb; ++j) {
        PetscReal v = PetscRealPart(ca[j]);
        w[bi[j]] = v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
    }
    VecRestoreArrayRead(chi, &ca);
    KSPDestroy(&kh); VecDestroy(&ones); VecDestroy(&rhs); VecDestroy(&chi);
    MatDestroy(&Abb); MatDestroy(&Abc); ISDestroy(&isb); ISDestroy(&isc);
    free(bi); free(ci);
}

/* Build per-subdomain weights w_i with the exact global renormalisation
 * sum_i w_i(k)^power = 1.  power = 2 for the symmetric sqrt-PU fine level,
 * power = 1 for a genuine partition of unity (coarse basis). */
static void BuildPU(Sub *S, PetscInt nsub, PetscInt N, int harmonic, int power,
                    PetscReal **wout) {
    PetscReal *acc = (PetscReal*)calloc(N, sizeof(PetscReal));
    for (PetscInt i = 0; i < nsub; ++i) {
        wout[i] = (PetscReal*)malloc(sizeof(PetscReal)*S[i].n);
        if (harmonic) HarmonicPU(&S[i], wout[i]);
        else for (PetscInt j = 0; j < S[i].n; ++j) wout[i][j] = 1.0;
        for (PetscInt j = 0; j < S[i].n; ++j)
            acc[S[i].idx[j]] += (power == 2) ? wout[i][j]*wout[i][j] : wout[i][j];
    }
    for (PetscInt k = 0; k < N; ++k) if (acc[k] <= 0.0) acc[k] = 1.0;
    for (PetscInt i = 0; i < nsub; ++i)
        for (PetscInt j = 0; j < S[i].n; ++j) {
            PetscReal d = acc[S[i].idx[j]];
            wout[i][j] /= (power == 2) ? PetscSqrtReal(d) : d;
        }
    free(acc);
}

/* ------------------------------------------------- weighted ASM + coarse -- */
/* M^-1 = R0^T A0^-1 R0 + sum_i R_i^T W_i Atil_i^-1 W_i R_i ,  sum_i w_i^2 = 1.
 * With w_i(k) = 1/sqrt(m_k) this is exactly twolevel.c's fine level. */
static void Apply(Ctx *c, Vec r, Vec z) {
    VecZeroEntries(z);
    const PetscScalar *ra; PetscScalar *za;
    VecGetArrayRead(r, &ra); VecGetArray(z, &za);
    for (PetscInt i = 0; i < c->nsub; ++i) {
        Sub *s = &c->subs[i];
        const PetscScalar *wa; VecGetArrayRead(c->wsub[i], &wa);
        PetscScalar *rib; VecGetArray(s->ri, &rib);
        for (PetscInt j = 0; j < s->n; ++j) rib[j] = ra[s->idx[j]] * wa[j];
        VecRestoreArray(s->ri, &rib);
        KSPSolve(s->ksp, s->ri, s->yi);
        const PetscScalar *yib; VecGetArrayRead(s->yi, &yib);
        for (PetscInt j = 0; j < s->n; ++j) za[s->idx[j]] += yib[j] * wa[j];
        VecRestoreArrayRead(s->yi, &yib);
        VecRestoreArrayRead(c->wsub[i], &wa);
    }
    VecRestoreArrayRead(r, &ra); VecRestoreArray(z, &za);
    if (c->twolevel) {
        MatMultTranspose(c->R0t, r, c->cvec);
        KSPSolve(c->kspc, c->cvec, c->ycvec);
        MatMultAdd(c->R0t, c->ycvec, z, z);
    }
}
static PetscErrorCode ShellApply(PC pc, Vec r, Vec z) {
    Ctx *c; PCShellGetContext(pc, (void**)&c); Apply(c, r, z); return 0;
}

/* coarse space: nvec basis functions per subdomain built from the SAME
 * partition of unity that weights the fine level (mode 2 adds the two
 * fibre-aligned linear modes, MsFEM style). */
static void BuildCoarse(Mat A, Sub *S, PetscInt nsub, PetscInt N, PetscInt m,
                        int harmonic, int nvec, PetscReal fdeg, Ctx *c) {
    PetscReal **phi = (PetscReal**)malloc(sizeof(PetscReal*)*nsub);
    BuildPU(S, nsub, N, harmonic, 1, phi);          /* sum_i phi_i = 1 */
    PetscInt ncol = nsub * nvec;
    /* a DOF in the overlap can belong to several subdomains (up to ~9 in 2D for
     * a generous overlap), each contributing nvec columns -- preallocate for
     * that and let PETSc grow rather than error if a corner needs more. */
    Mat R0t; MatCreateSeqAIJ(PETSC_COMM_SELF, N, ncol, 9*nvec, NULL, &R0t);
    MatSetOption(R0t, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    const PetscReal th = fdeg * M_PI / 180.0, cs = PetscCosReal(th), sn = PetscSinReal(th);
    for (PetscInt i = 0; i < nsub; ++i) {
        PetscReal sm = 0, tm = 0, smax = 1e-12, tmax = 1e-12;
        for (PetscInt j = 0; j < S[i].n; ++j) {
            PetscReal x = (PetscReal)(S[i].idx[j] % m), y = (PetscReal)(S[i].idx[j] / m);
            sm += x*cs + y*sn;  tm += -x*sn + y*cs;
        }
        sm /= (PetscReal)S[i].n; tm /= (PetscReal)S[i].n;
        for (PetscInt j = 0; j < S[i].n; ++j) {
            PetscReal x = (PetscReal)(S[i].idx[j] % m), y = (PetscReal)(S[i].idx[j] / m);
            PetscReal sv = PetscAbsReal(x*cs + y*sn - sm), tv = PetscAbsReal(-x*sn + y*cs - tm);
            if (sv > smax) smax = sv;
            if (tv > tmax) tmax = tv;
        }
        for (PetscInt j = 0; j < S[i].n; ++j) {
            if (phi[i][j] <= 1e-14) continue;
            PetscInt k = S[i].idx[j];
            PetscReal x = (PetscReal)(k % m), y = (PetscReal)(k / m);
            PetscReal md[3] = { 1.0, (x*cs + y*sn - sm)/smax, (-x*sn + y*cs - tm)/tmax };
            for (int v = 0; v < nvec; ++v)
                MatSetValue(R0t, k, i*nvec + v, phi[i][j]*md[v], INSERT_VALUES);
        }
    }
    MatAssemblyBegin(R0t, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(R0t, MAT_FINAL_ASSEMBLY);
    for (PetscInt i = 0; i < nsub; ++i) free(phi[i]);
    free(phi);
    c->R0t = R0t;
    Mat A0; MatPtAP(A, R0t, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &A0);
    KSPCreate(PETSC_COMM_SELF, &c->kspc); KSPSetType(c->kspc, KSPPREONLY);
    KSPSetOperators(c->kspc, A0, A0);
    PC pc; KSPGetPC(c->kspc, &pc); PCSetType(pc, PCCHOLESKY);
    PCFactorSetShiftType(pc, MAT_SHIFT_POSITIVE_DEFINITE);
    KSPSetUp(c->kspc);
    MatCreateVecs(A0, &c->ycvec, &c->cvec);
    MatDestroy(&A0);
}

static Ctx MakeCtx(Mat A, Sub *S, PetscInt nsub, PetscInt N, PetscInt m,
                   int fine_harm, int coarse_mode, PetscReal fdeg) {
    Ctx c; c.subs = S; c.nsub = nsub;
    MatCreateVecs(A, &c.tmp, NULL);
    PetscReal **w = (PetscReal**)malloc(sizeof(PetscReal*)*nsub);
    BuildPU(S, nsub, N, fine_harm, 2, w);           /* sum_i w_i^2 = 1 */
    c.wsub = (Vec*)malloc(sizeof(Vec)*nsub);
    for (PetscInt i = 0; i < nsub; ++i) {
        VecCreateSeq(PETSC_COMM_SELF, S[i].n, &c.wsub[i]);
        PetscScalar *a; VecGetArray(c.wsub[i], &a);
        for (PetscInt j = 0; j < S[i].n; ++j) a[j] = w[i][j];
        VecRestoreArray(c.wsub[i], &a);
        free(w[i]);
    }
    free(w);
    c.wflat = NULL;
    c.twolevel = (coarse_mode > 0);
    c.R0t = NULL; c.kspc = NULL; c.cvec = NULL; c.ycvec = NULL;
    if (coarse_mode == 1) BuildCoarse(A, S, nsub, N, m, 0,          1, fdeg, &c);
    if (coarse_mode == 2) BuildCoarse(A, S, nsub, N, m, 1,          1, fdeg, &c);
    if (coarse_mode == 3) BuildCoarse(A, S, nsub, N, m, 1,          3, fdeg, &c);
    return c;
}
static void FreeCtx(Ctx *c) {
    VecDestroy(&c->tmp);
    for (PetscInt i = 0; i < c->nsub; ++i) VecDestroy(&c->wsub[i]);
    free(c->wsub);
    if (c->twolevel) { MatDestroy(&c->R0t); KSPDestroy(&c->kspc);
                       VecDestroy(&c->cvec); VecDestroy(&c->ycvec); }
}

static PetscInt PCG(Mat A, Ctx *c, Vec b, Vec x, PetscReal rtol, PetscInt maxit,
                    PetscReal *lmin, PetscReal *lmax) {
    KSP ksp; PC pc; Vec r;
    VecDuplicate(b, &r);
    KSPCreate(PETSC_COMM_SELF, &ksp); KSPSetType(ksp, KSPCG);
    KSPSetOperators(ksp, A, A);
    KSPGetPC(ksp, &pc); PCSetType(pc, PCSHELL);
    PCShellSetContext(pc, c); PCShellSetApply(pc, ShellApply);
    KSPSetComputeSingularValues(ksp, PETSC_TRUE);
    KSPSetNormType(ksp, KSP_NORM_PRECONDITIONED);
    KSPSetTolerances(ksp, rtol, 1e-50, PETSC_DEFAULT, maxit);
    KSPSetUp(ksp); KSPSolve(ksp, b, x);
    PetscInt it; KSPGetIterationNumber(ksp, &it);
    KSPComputeExtremeSingularValues(ksp, lmax, lmin);
    VecDestroy(&r); KSPDestroy(&ksp);
    return it;
}

int main(int argc, char **argv) {
    PetscInitialize(&argc, &argv, NULL, NULL);
    PetscInt n = 120, P = 6, O = 2;
    PetscReal aniso = 1.0, fdeg = 45.0;
    PetscBool scale = PETSC_FALSE, flg;
    char cname[32] = "all", fname[32] = "harm";
    PetscOptionsGetInt (NULL, NULL, "-n",      &n,     NULL);
    PetscOptionsGetInt (NULL, NULL, "-P",      &P,     NULL);
    PetscOptionsGetInt (NULL, NULL, "-O",      &O,     NULL);
    PetscOptionsGetReal(NULL, NULL, "-aniso",  &aniso, NULL);
    PetscOptionsGetReal(NULL, NULL, "-fiber_deg", &fdeg, NULL);
    PetscOptionsGetBool(NULL, NULL, "-scale",  &scale, NULL);
    PetscOptionsGetString(NULL, NULL, "-coarse", cname, sizeof(cname), &flg);
    PetscOptionsGetString(NULL, NULL, "-fine",   fname, sizeof(fname), &flg);
    const int fine_harm = (strcmp(fname, "mult") != 0);   /* -fine mult -> 1/sqrt(m_k) */

    /* det-normalised tensor: the CONTRAST is the knob, not the overall scale */
    const PetscReal sl = PetscSqrtReal(aniso), st = 1.0/PetscSqrtReal(aniso);
    const PetscReal th = fdeg*M_PI/180.0, cs = PetscCosReal(th), sn = PetscSinReal(th);
    const PetscReal sxx = st + (sl-st)*cs*cs, syy = st + (sl-st)*sn*sn,
                    sxy =      (sl-st)*cs*sn;

    const char *lbl[4] = {"one-level (no coarse)", "two-level mult 1/m_k",
                          "two-level harm  (reuse)", "two-level enrich(reuse,3)"};
    int modes[4] = {0, 1, 2, 3}, nmode = 4;
    if (strcmp(cname, "mult")   == 0) { modes[0]=1; nmode=1; }
    if (strcmp(cname, "harm")   == 0) { modes[0]=2; nmode=1; }
    if (strcmp(cname, "enrich") == 0) { modes[0]=3; nmode=1; }

    PetscPrintf(PETSC_COMM_SELF,
        "=== twolevel_aniso: contrast r=%g  fibre=%g deg  (sigma_l=%.4g sigma_t=%.4g)\n"
        "    fine level = %s sqrt-PU;  O=%d;  ICC(0) sub-solve\n",
        (double)aniso, (double)fdeg, (double)sl, (double)st,
        fine_harm ? "sigma-harmonic" : "multiplicity 1/sqrt(m_k)", (int)O);

    PetscInt Ps[6] = {2,3,4,5,6,0}, npt = scale ? 5 : 1;
    for (PetscInt pi = 0; pi < npt; ++pi) {
        PetscInt Pl = scale ? Ps[pi] : P;
        PetscInt nl = scale ? 20*Pl : n, N, m = nl - 1;
        Mat A = Diffusion2D(nl, sxx, sxy, syy, &N);
        Vec b, x; MatCreateVecs(A, &b, &x); VecSet(b, 1.0);
        PetscInt nsub; Sub *S = BuildAll(A, m, (int)Pl, (int)O, &nsub);
        PetscPrintf(PETSC_COMM_SELF, "\n  P=%d  #subdomains=%d  N=%d\n",
                    (int)Pl, (int)nsub, (int)N);
        PetscPrintf(PETSC_COMM_SELF, "  %-26s %8s %12s %10s %10s\n",
                    "method", "CG its", "lambda_min", "lambda_max", "kappa");
        for (int mi = 0; mi < nmode; ++mi) {
            PetscReal lo, hi;
            Ctx c = MakeCtx(A, S, nsub, N, m, fine_harm, modes[mi], fdeg);
            PetscInt it = PCG(A, &c, b, x, 1e-6, 4000, &lo, &hi);
            PetscPrintf(PETSC_COMM_SELF, "  %-26s %8d %12.4e %10.4f %10.1f\n",
                        lbl[modes[mi]], (int)it, (double)lo, (double)hi, (double)(hi/lo));
            FreeCtx(&c);
        }
        FreeAll(S, nsub); VecDestroy(&b); VecDestroy(&x); MatDestroy(&A);
    }
    PetscPrintf(PETSC_COMM_SELF, "\nTWOLEVEL_ANISO_DONE\n");
    PetscFinalize(); return 0;
}
