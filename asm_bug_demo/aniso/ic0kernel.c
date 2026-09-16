/* ic0kernel.c -- IC(0) factorisation + triangular solves, faithful to PETSc's
 * `-sub_pc_type icc -sub_pc_factor_levels 0`: the factor L has EXACTLY the
 * sparsity of tril(A) (zero fill), and a Manteuffel-style diagonal shift is
 * retried on a non-positive pivot (PETSc's MAT_SHIFT_POSITIVE_DEFINITE).
 *
 * Storage: lower-triangular CSR including the diagonal, column indices sorted
 * ascending within each row.  L uses the same (Lp, Lj) pattern as A.
 *
 * Build:  gcc -O3 -fPIC -shared ic0kernel.c -o libic0.so -lm
 */
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* dot of the parts of rows i and j with column index < jcut.
 * Both rows are sorted ascending.  O(nnz_i + nnz_j). */
static double sparse_dot_lt(const int *Lj, const double *Lv,
                            int ib, int ie, int jb, int je, int jcut)
{
    double s = 0.0;
    int p = ib, q = jb;
    while (p < ie && q < je) {
        int cp = Lj[p], cq = Lj[q];
        if (cp >= jcut || cq >= jcut) break;
        if      (cp < cq) ++p;
        else if (cp > cq) ++q;
        else { s += Lv[p] * Lv[q]; ++p; ++q; }
    }
    return s;
}

/* One IC(0) pass at a given diagonal shift.
 * Returns 0 on success, 1 on a non-positive pivot. */
static int ic0_pass(int n, const int *Ap, const int *Aj, const double *Av,
                    const int *Lp, const int *Lj, double *Lv, double shift)
{
    for (int i = 0; i < n; ++i) {
        int rb = Ap[i], re = Ap[i + 1];
        /* copy row i of A into L's values (same pattern) */
        for (int p = rb; p < re; ++p) Lv[p] = Av[p];
        double diag = 0.0;
        int    dpos = -1;
        for (int p = rb; p < re; ++p) {
            int j = Lj[p];
            if (j == i) { dpos = p; continue; }
            /* j < i (lower triangle) */
            double s = Lv[p] - sparse_dot_lt(Lj, Lv, rb, re, Lp[j], Lp[j + 1], j);
            int jd = Lp[j + 1] - 1;              /* diagonal of row j is last */
            Lv[p] = s / Lv[jd];
        }
        if (dpos < 0) return 1;                  /* missing diagonal */
        diag = Av[dpos] * (1.0 + shift);
        for (int p = rb; p < re; ++p)
            if (Lj[p] != i) diag -= Lv[p] * Lv[p];
        if (!(diag > 0.0)) return 1;
        Lv[dpos] = sqrt(diag);
    }
    return 0;
}

/* Public entry: retry with a growing shift like PETSc does.
 * Writes the shift actually used into *shift_used.
 * Returns 0 on success, -1 if even the largest shift failed. */
int ic0_factor(int n, const int *Ap, const int *Aj, const double *Av,
               const int *Lp, const int *Lj, double *Lv, double *shift_used)
{
    double shift = 0.0;
    for (int attempt = 0; attempt < 30; ++attempt) {
        if (ic0_pass(n, Ap, Aj, Av, Lp, Lj, Lv, shift) == 0) {
            *shift_used = shift;
            return 0;
        }
        shift = (shift == 0.0) ? 1e-8 : shift * 4.0;
        if (shift > 1e4) break;
    }
    *shift_used = shift;
    return -1;
}

/* Apply M^{-1} = (L L^T)^{-1}:  forward solve then back solve. */
void ic0_solve(int n, const int *Lp, const int *Lj, const double *Lv,
               const double *b, double *x)
{
    /* L y = b */
    for (int i = 0; i < n; ++i) {
        double s = b[i];
        int rb = Lp[i], re = Lp[i + 1] - 1;      /* exclude diagonal (last) */
        for (int p = rb; p < re; ++p) s -= Lv[p] * x[Lj[p]];
        x[i] = s / Lv[Lp[i + 1] - 1];
    }
    /* L^T x = y */
    for (int i = n - 1; i >= 0; --i) {
        x[i] /= Lv[Lp[i + 1] - 1];
        double xi = x[i];
        int rb = Lp[i], re = Lp[i + 1] - 1;
        for (int p = rb; p < re; ++p) x[Lj[p]] -= Lv[p] * xi;
    }
}

/* Batched subdomain apply: the whole ASM sum in one call, so Python never
 * touches the inner loop.
 *   nsub      number of subdomains
 *   off[i]    start of subdomain i inside the flattened idx/w arrays
 *   idx[]     global DOF indices of each subdomain (concatenated)
 *   w[]       per-subdomain weights (concatenated); NULL => all ones (BASIC)
 *   Lp/Lj/Lv  concatenated IC(0) factors, with Lpoff[i]/Lvoff[i] the offsets
 *   r, z      global vectors of length N;  z <- sum_i R_i^T W_i (L L^T)_i^{-1} W_i R_i r
 */
void asm_apply(int N, int nsub,
               const int *off, const int *idx, const double *w,
               const int *Lpoff, const int *Lvoff,
               const int *Lp, const int *Lj, const double *Lv,
               const double *r, double *z, double *buf1, double *buf2)
{
    memset(z, 0, sizeof(double) * (size_t)N);
    for (int i = 0; i < nsub; ++i) {
        int b = off[i], e = off[i + 1], n = e - b;
        for (int l = 0; l < n; ++l) {
            double v = r[idx[b + l]];
            buf1[l] = w ? v * w[b + l] : v;
        }
        ic0_solve(n, Lp + Lpoff[i], Lj + Lvoff[i], Lv + Lvoff[i], buf1, buf2);
        for (int l = 0; l < n; ++l) {
            double v = buf2[l];
            z[idx[b + l]] += w ? v * w[b + l] : v;
        }
    }
}
