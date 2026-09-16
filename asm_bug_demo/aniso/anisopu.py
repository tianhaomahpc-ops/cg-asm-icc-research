"""anisopu.py -- the sigma-tensor axis that the rest of this repo is missing.

Self-contained (numpy/scipy + the local libic0.so) reproduction of the
"overlap up -> CG iter up" anomaly on an *anisotropic* diffusion operator,
plus the two new partition-of-unity families proposed as the fix.

Method mirrors coloring.c / twolevel.c (2D, box subdomains, IC(0) sub-solve,
CG-Lanczos Ritz values for lambda_min / lambda_max) and adds:

  * sigma = R diag(sigma_l, sigma_t) R^T    fiber-aligned conductivity tensor,
    contrast r = sigma_l / sigma_t, det-normalised (sigma_l = sqrt(r),
    sigma_t = 1/sqrt(r)) so the ratio -- not the overall scale -- is the knob.

  * five weights, all in the symmetric sqrt-PU form
        M^-1 = sum_i R_i^T W_i Atil_i^-1 W_i R_i ,  sum_i w_i(k)^2 = 1
    (the form validated in REPORT_weighting_ablation_zh.md sec.12):
        basic   w = 1, no normalisation          (PC_ASM_BASIC)
        flat    w_unnorm = 1        -> 1/sqrt(m) (== sASM, scheme 3)
        graded  w_unnorm = q^depth               (== scheme 7 -pugrade)
        ramp    w_unnorm = 1 - depth/(delta+1)   NEW: |grad chi| ~ 1/delta
        harm    w_unnorm = sigma-harmonic ramp   NEW: minimiser of int sigma|grad chi|^2

  * optional Nicolaides coarse space whose basis is the *same* chi_i
    (coarse-basis reuse), vs the classical 1/m_k basis.
"""
import ctypes, os, sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
_lib = ctypes.CDLL(os.path.join(HERE, "libic0.so"))
_i32 = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
_f64 = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
_lib.ic0_factor.argtypes = [ctypes.c_int, _i32, _i32, _f64, _i32, _i32, _f64,
                            ctypes.POINTER(ctypes.c_double)]
_lib.ic0_factor.restype = ctypes.c_int
_lib.asm_apply.argtypes = [ctypes.c_int, ctypes.c_int, _i32, _i32, _f64,
                           _i32, _i32, _i32, _i32, _f64, _f64, _f64, _f64, _f64]
_lib.asm_apply.restype = None
_lib.ic0_solve.argtypes = [ctypes.c_int, _i32, _i32, _f64, _f64, _f64]
_lib.ic0_solve.restype = None


# ----------------------------------------------------------------- operator
def sigma_tensor(ratio, fiber_deg, det_norm=True):
    """sigma = sigma_t I + (sigma_l - sigma_t) f f^T, f at fiber_deg from +x."""
    if det_norm:
        sl, st = np.sqrt(ratio), 1.0 / np.sqrt(ratio)
    else:
        sl, st = float(ratio), 1.0
    th = np.deg2rad(fiber_deg)
    f = np.array([np.cos(th), np.sin(th)])
    return st * np.eye(2) + (sl - st) * np.outer(f, f)


def build_fem(nx, sig):
    """P1 FEM for -div(sigma grad u) = 1 on [0,1]^2, structured right triangles.
    BC: u = 0 on x = 0 (Dirichlet), natural Neumann elsewhere  -- Sys3 shape.
    Returns (A csr, b, ndof, coords, free-node map)."""
    n1 = nx + 1
    X, Y = np.meshgrid(np.linspace(0, 1, n1), np.linspace(0, 1, n1))
    coords = np.column_stack([X.ravel(), Y.ravel()])
    nid = lambda i, j: j * n1 + i

    tris = []
    for j in range(nx):
        for i in range(nx):
            a, b_, c, d = nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)
            tris.append((a, b_, c))
            tris.append((a, c, d))
    tris = np.asarray(tris, dtype=np.int64)

    p = coords[tris]                                   # (ne,3,2)
    x1, y1 = p[:, 0, 0], p[:, 0, 1]
    x2, y2 = p[:, 1, 0], p[:, 1, 1]
    x3, y3 = p[:, 2, 0], p[:, 2, 1]
    det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    area = 0.5 * np.abs(det)
    # grad of the barycentric hat functions
    bcoef = np.stack([y2 - y3, y3 - y1, y1 - y2], axis=1) / det[:, None]
    ccoef = np.stack([x3 - x2, x1 - x3, x2 - x1], axis=1) / det[:, None]

    ke = np.empty((len(tris), 3, 3))
    for a_ in range(3):
        for b2 in range(3):
            gx_a, gy_a = bcoef[:, a_], ccoef[:, a_]
            gx_b, gy_b = bcoef[:, b2], ccoef[:, b2]
            ke[:, a_, b2] = area * (
                sig[0, 0] * gx_a * gx_b + sig[0, 1] * gx_a * gy_b +
                sig[1, 0] * gy_a * gx_b + sig[1, 1] * gy_a * gy_b)

    rows = np.repeat(tris, 3, axis=1).ravel()
    cols = np.tile(tris, (1, 3)).ravel()
    A = sp.coo_matrix((ke.ravel(), (rows, cols)),
                      shape=(n1 * n1, n1 * n1)).tocsr()
    load = np.zeros(n1 * n1)
    np.add.at(load, tris.ravel(), np.repeat(area / 3.0, 3))

    dirich = np.isclose(coords[:, 0], 0.0)
    free = np.where(~dirich)[0]
    A = A[free][:, free].tocsr()
    A.sort_indices()
    return A, load[free], coords[free], free, n1


# --------------------------------------------------------------- partition
def box_partition(coords, Px, Py):
    """Px x Py non-overlapping box partition of the nodes (matches the regular
    decomposition used for fig8/fig9 in REPORT_interpretability_zh.md)."""
    ix = np.clip((coords[:, 0] * Px).astype(int), 0, Px - 1)
    iy = np.clip((coords[:, 1] * Py).astype(int), 0, Py - 1)
    return iy * Px + ix


def overlap_sets(A, owner, nsub, O):
    """PETSc MatIncreaseOverlap semantics: one BFS layer of the matrix graph per
    call.  Layers are grown with a sparse mat-vec (|A| @ mask), so this is O(nnz)
    per layer instead of a Python loop over the frontier."""
    Abin = sp.csr_matrix((np.ones_like(A.data), A.indices, A.indptr), shape=A.shape)
    N = A.shape[0]
    out = []
    for s_ in range(nsub):
        inset = (owner == s_)
        depth = np.full(N, -1, dtype=np.int64)
        depth[inset] = 0
        cur = inset.astype(np.float64)
        for lev in range(1, O + 1):
            reach = (Abin @ cur) > 0
            new_ = reach & (depth < 0)
            if not new_.any():
                break
            depth[new_] = lev
            cur = new_.astype(np.float64)
        idx = np.where(depth >= 0)[0]
        out.append((idx, depth[idx]))
    return out


# ------------------------------------------------------------------- weights
def harmonic_ramp(Ai, dep):
    """chi: sigma-harmonic extension of 1 on the core into the overlap band,
    with 0 at the artificial boundary (implicit, i.e. outside the subdomain).
    Exactly the minimiser of int sigma |grad chi|^2 for those boundary data."""
    core = np.where(dep == 0)[0]
    band = np.where(dep > 0)[0]
    chi = np.zeros(Ai.shape[0])
    chi[core] = 1.0
    if len(band) == 0:
        return chi
    Abb = Ai[band][:, band].tocsc()
    rhs = -np.asarray(Ai[band][:, core].sum(axis=1)).ravel()
    try:
        chi[band] = spla.spsolve(Abb, rhs)
    except Exception:
        chi[band] = np.exp(-dep[band].astype(float))
    return np.clip(chi, 0.0, 1.0)


def build_weights(kind, sets, Alocs, N, O, q=0.8, power=2):
    """Unnormalised per-subdomain weight, then the EXACT global renormalisation
    sum_i w_i(k)^2 = 1 (same as asm_demo.cpp:623-645)."""
    if kind == "basic":
        return None
    raw = []
    for (idx, dep), Ai in zip(sets, Alocs):
        d = dep.astype(float)
        if kind == "flat":
            w = np.ones_like(d)
        elif kind == "graded":
            w = q ** d
        elif kind == "ramp":
            w = np.maximum(0.0, 1.0 - d / (O + 1.0))
        elif kind == "harm":
            w = harmonic_ramp(Ai, dep)
        else:
            raise ValueError(kind)
        raw.append(w)
    # power=2 -> sum_i w_i^2 = 1 (the symmetric sqrt-PU sandwich)
    # power=1 -> sum_i w_i   = 1 (a genuine partition of unity, for coarse bases)
    ssum = np.zeros(N)
    for (idx, _), w in zip(sets, raw):
        np.add.at(ssum, idx, w ** power)
    ssum[ssum <= 0] = 1.0
    if power == 2:
        return [w / np.sqrt(ssum[idx]) for (idx, _), w in zip(sets, raw)]
    return [w / ssum[idx] for (idx, _), w in zip(sets, raw)]


def coarse_basis_enriched(kind, sets, Alocs, N, O, coords, fiber_deg, q=0.8):
    """MsFEM-style enrichment: instead of one constant per subdomain, use
    {chi_i, chi_i * s_i, chi_i * t_i} where (s,t) are the fiber-aligned local
    coordinates, recentred per subdomain.  3 coarse vectors per subdomain.
    Still symmetric, still tiny; the natural rung between Nicolaides and GenEO."""
    phis = build_weights(kind, sets, Alocs, N, O, q, power=1)
    th = np.deg2rad(fiber_deg)
    ax = np.array([np.cos(th), np.sin(th)])
    ay = np.array([-np.sin(th), np.cos(th)])
    rows, cols, vals = [], [], []
    ncol = 0
    for i, ((idx, _), phi) in enumerate(zip(sets, phis)):
        nz = phi > 1e-14
        gi, pv = idx[nz], phi[nz]
        c = coords[gi]
        s_ = c @ ax; t_ = c @ ay
        s_ = s_ - s_.mean(); t_ = t_ - t_.mean()
        sc = max(np.abs(s_).max(), 1e-12); tc = max(np.abs(t_).max(), 1e-12)
        for mode in (np.ones_like(pv), s_ / sc, t_ / tc):
            rows.append(np.full(len(gi), ncol)); cols.append(gi); vals.append(pv * mode)
            ncol += 1
    return sp.csr_matrix((np.concatenate(vals),
                          (np.concatenate(rows), np.concatenate(cols))),
                         shape=(ncol, N))


def coarse_basis(kind, sets, Alocs, N, O, q=0.8):
    """Nicolaides-type coarse space: one basis function per subdomain, equal to
    that subdomain's partition-of-unity function (sum_i phi_i = 1).
    kind='flat' reproduces twolevel.c:99 exactly (phi_i(k) = 1/m_k);
    kind='harm' REUSES the sigma-harmonic chi_i already built for the fine
    level -- no extra setup at all."""
    phis = build_weights(kind, sets, Alocs, N, O, q, power=1)
    rows, cols, vals = [], [], []
    for i, ((idx, _), phi) in enumerate(zip(sets, phis)):
        nz = phi > 1e-14
        rows.append(np.full(int(nz.sum()), i)); cols.append(idx[nz]); vals.append(phi[nz])
    return sp.csr_matrix((np.concatenate(vals),
                          (np.concatenate(rows), np.concatenate(cols))),
                         shape=(len(sets), N))


# ---------------------------------------------------------- preconditioner
class ASM:
    def __init__(self, A, sets, weights, exact=False):
        self.N = A.shape[0]
        self.nsub = len(sets)
        self.exact = exact
        self.shifts = []
        off, idx_all, w_all = [0], [], []
        Lp_all, Lj_all, Lv_all, Lpoff, Lvoff = [], [], [], [0], [0]
        self.lu = []
        self.Alocs = []
        for s, (idx, dep) in enumerate(sets):
            Ai = A[idx][:, idx].tocsr()
            Ai.sort_indices()
            self.Alocs.append(Ai)
            n = Ai.shape[0]
            idx_all.append(idx.astype(np.int32))
            off.append(off[-1] + n)
            w_all.append(np.ones(n) if weights is None else weights[s])
            if exact:
                self.lu.append(spla.splu(Ai.tocsc()))
                continue
            Lo = sp.tril(Ai, format="csr")
            Lo.sort_indices()
            Lv = np.zeros(Lo.nnz)
            shift = ctypes.c_double(0.0)
            rc = _lib.ic0_factor(n,
                                 Lo.indptr.astype(np.int32), Lo.indices.astype(np.int32),
                                 Lo.data.astype(np.float64),
                                 Lo.indptr.astype(np.int32), Lo.indices.astype(np.int32),
                                 Lv, ctypes.byref(shift))
            if rc != 0:
                raise RuntimeError("IC(0) failed on subdomain %d" % s)
            self.shifts.append(shift.value)
            Lp_all.append(Lo.indptr.astype(np.int32))
            Lj_all.append(Lo.indices.astype(np.int32))
            Lv_all.append(Lv)
            Lpoff.append(Lpoff[-1] + n + 1)
            Lvoff.append(Lvoff[-1] + Lo.nnz)
        self.off = np.array(off, dtype=np.int32)
        self.idx = np.concatenate(idx_all).astype(np.int32)
        self.w = np.concatenate(w_all).astype(np.float64)
        self.use_w = weights is not None
        self.nmax = int(np.max(np.diff(self.off)))
        self.buf1 = np.zeros(self.nmax)
        self.buf2 = np.zeros(self.nmax)
        self.z = np.zeros(self.N)
        if not exact:
            self.Lp = np.concatenate(Lp_all).astype(np.int32)
            self.Lj = np.concatenate(Lj_all).astype(np.int32)
            self.Lv = np.concatenate(Lv_all).astype(np.float64)
            self.Lpoff = np.array(Lpoff, dtype=np.int32)
            self.Lvoff = np.array(Lvoff, dtype=np.int32)
        # coarse level (optional, installed later)
        self.R0 = None

    def apply(self, r):
        if self.exact:
            z = np.zeros(self.N)
            for s in range(self.nsub):
                b, e = self.off[s], self.off[s + 1]
                g = self.idx[b:e]
                v = r[g] * self.w[b:e] if self.use_w else r[g]
                y = self.lu[s].solve(v)
                np.add.at(z, g, y * self.w[b:e] if self.use_w else y)
        else:
            r = np.ascontiguousarray(r, dtype=np.float64)
            _lib.asm_apply(self.N, self.nsub, self.off, self.idx, self.w,
                           self.Lpoff, self.Lvoff, self.Lp, self.Lj, self.Lv,
                           r, self.z, self.buf1, self.buf2)
            z = self.z.copy()
        if self.R0 is not None:
            z += self.R0.T @ self.A0solve(self.R0 @ r)
        return z

    def set_coarse(self, A, R0):
        """R0: (nsub, N) coarse basis (rows = phi_i).  Galerkin A0 = R0 A R0^T."""
        self.R0 = R0
        A0 = (R0 @ A @ R0.T)
        A0 = np.asarray(A0.todense()) if sp.issparse(A0) else np.asarray(A0)
        lu = np.linalg.inv(A0)
        self.A0solve = lambda v: lu @ v


# ------------------------------------------------------------------ Krylov
def pcg(A, b, M, rtol=1e-6, maxit=3000):
    """CG with -ksp_norm_type preconditioned, plus the Lanczos tridiagonal
    (the same construction KSPComputeExtremeSingularValues uses)."""
    x = np.zeros(A.shape[0])
    r = b.copy()
    z = M.apply(r)
    p = z.copy()
    rz = r @ z
    rz0 = rz
    alphas, betas = [], []
    for k in range(maxit):
        if np.sqrt(max(rz, 0.0)) <= rtol * np.sqrt(rz0):
            break
        Ap = A @ p
        pAp = p @ Ap
        if pAp <= 0:
            break
        alpha = rz / pAp
        x += alpha * p
        r -= alpha * Ap
        z = M.apply(r)
        rzn = r @ z
        beta = rzn / rz
        p = z + beta * p
        alphas.append(alpha); betas.append(beta)
        rz = rzn
    # Lanczos tridiagonal; truncate at the first non-finite / non-positive beta
    # (round-off past convergence, exactly what PETSc's Ritz estimate guards).
    n = len(alphas)
    good = n
    for j in range(n):
        if not (np.isfinite(alphas[j]) and alphas[j] > 0
                and np.isfinite(betas[j]) and betas[j] > 0):
            good = j
            break
    n = good
    if n == 0:
        return x, len(alphas), (np.nan, np.nan)
    d = np.empty(n); e = np.empty(max(n - 1, 0))
    for j in range(n):
        d[j] = 1.0 / alphas[j] + (betas[j - 1] / alphas[j - 1] if j > 0 else 0.0)
        if j < n - 1:
            e[j] = np.sqrt(betas[j]) / alphas[j]
    try:
        from scipy.linalg import eigvalsh_tridiagonal
        ev = eigvalsh_tridiagonal(d, e)
    except Exception:
        return x, len(alphas), (np.nan, np.nan)
    return x, len(alphas), (float(ev[0]), float(ev[-1]))


def local_omega(Aloc, steps=120):
    """omega = lambda_max(M_i^{-1} A_i) for the IC(0) sub-solver, and the local
    condition number.  This is the ONLY place the conductivity tensor enters
    the omega * Nhat product, so it is the quantity to watch vs contrast."""
    n = Aloc.shape[0]
    Lo = sp.tril(Aloc, format="csr"); Lo.sort_indices()
    Lp = Lo.indptr.astype(np.int32); Lj = Lo.indices.astype(np.int32)
    Lv = np.zeros(Lo.nnz); shift = ctypes.c_double(0.0)
    rc = _lib.ic0_factor(n, Lp, Lj, Lo.data.astype(np.float64), Lp, Lj, Lv,
                         ctypes.byref(shift))
    if rc != 0:
        return np.nan, np.nan, np.nan

    out = np.zeros(n)

    class _M:
        N = n
        def apply(self, r):
            _lib.ic0_solve(n, Lp, Lj, Lv,
                           np.ascontiguousarray(r, dtype=np.float64), out)
            return out.copy()

    rng = np.random.default_rng(0)
    b = rng.standard_normal(n)
    _, _, (lo, hi) = pcg(Aloc, b, _M(), rtol=1e-13, maxit=steps)
    return hi, lo, shift.value
