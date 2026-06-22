"""asm_spectral.py  --  dimensional + variable-coefficient additive-Schwarz spectral lab.

A transparent NumPy/SciPy reference implementation of one-level additive Schwarz
(BASIC / unscaled  and  sASM / D^{-1/2} partition-of-unity scaled) for the
variable-coefficient operator

        -div( a(x) grad u ) = f      on  [0,1]^d ,
        u = 0 on the x_0 = 0 face (Dirichlet),  zero-flux Neumann elsewhere.

Discretised with a conservative vertex-centred finite-difference stencil and
harmonic averaging of a(x) at the cell faces, so a == 1 reproduces the standard
(2d+1)-point Laplacian used by the prior pure-PETSc demos in this repo.

The point of this lab is to DISSECT the anomaly established earlier
( CG + unscaled-BASIC additive Schwarz + INEXACT IC(0) local solves
  => iteration count rises with overlap )
across spatial dimension d in {1,2,3} and across coefficient fields, by
measuring the EXACT preconditioned spectrum -- not Lanczos estimates:

   * iter      : CG iterations with M^{-1} as preconditioner
   * lam_max   : largest eigenvalue of M^{-1} A   (the over-count / amplification end)
   * lam_min   : smallest eigenvalue of M^{-1} A  (the stable-decomposition end)
   * kappa     : lam_max / lam_min
   * Nhat      : max multiplicity = max_k #{subdomains containing dof k}  (geometric)
   * omega     : max_i lam_max( M_i^{-1} A_i )    (inexact-local amplification)

The central decomposition under test is   lam_max(BASIC) ~= omega * Nhat,
with omega == 1 exactly whenever the local solve is exact, and (the key 1D fact)
omega == 1 even for IC(0) in 1D because IC(0) of a tridiagonal matrix is exact.

No MPI, no PETSc: every quantity here is computed directly so the mechanism is
auditable.  Real-PETSc cross-validation at scale is a separate (HPC) instrument.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg, eigsh, splu


# --------------------------------------------------------------------------
#  Grid / coefficient utilities
# --------------------------------------------------------------------------
def grid_shape(dim, M):
    """M grid points per axis -> tuple of length dim."""
    return tuple([M] * dim)


def node_coords(dim, M):
    """Return (Npts, dim) array of physical coordinates in [0,1]^d, C-order."""
    axes = [np.linspace(0.0, 1.0, M) for _ in range(dim)]
    mesh = np.meshgrid(*axes, indexing='ij')
    return np.stack([m.ravel(order='C') for m in mesh], axis=1)


def coef_field(name, dim, M, contrast=1.0, n_layers=4, rng_seed=0):
    """Nodal coefficient a(x) >= 0, shape (Npts,).  Cases:
       const   : a == 1
       smooth  : a = 1 + 0.5*sin(2 pi x)*sin(2 pi y)... (mild, smooth)
       layers  : piecewise-constant slabs along x0, alternating 1 and `contrast`
                 with `n_layers` layers (a HIGH-CONTRAST, jump coefficient)
       layers_unaligned : same but boundaries placed off the subdomain grid
       checker : d-dim checkerboard of 1 vs contrast (jumps in every direction)
       random  : log-uniform random per node in [1, contrast]  (rough field)
    """
    X = node_coords(dim, M)             # (Npts, dim)
    Npts = X.shape[0]
    if name == 'const':
        return np.ones(Npts)
    if name == 'smooth':
        a = np.ones(Npts) * (1.0 + 0.5)
        s = np.ones(Npts)
        for d in range(dim):
            s = s * np.sin(2.0 * np.pi * X[:, d])
        return 1.0 + 0.5 * s            # in [0.5, 1.5], smooth
    if name in ('layers', 'layers_unaligned'):
        # slabs perpendicular to x0
        x0 = X[:, 0]
        shift = 0.0 if name == 'layers' else 0.5 / n_layers   # nudge jumps off grid lines
        idx = np.floor((x0 - shift) * n_layers).astype(int)
        a = np.where(idx % 2 == 0, 1.0, contrast)
        return a
    if name == 'checker':
        idx = np.zeros(Npts, dtype=int)
        for d in range(dim):
            idx += np.floor(X[:, d] * 4).astype(int)
        return np.where(idx % 2 == 0, 1.0, contrast)
    if name == 'random':
        rng = np.random.default_rng(rng_seed)
        return np.exp(rng.uniform(0.0, np.log(contrast), size=Npts))
    raise ValueError("unknown coef field %r" % name)


# --------------------------------------------------------------------------
#  Operator assembly:  -div(a grad u) = f , Dirichlet on x0=0, Neumann else
# --------------------------------------------------------------------------
def assemble(dim, M, a_nodal, bc='mixed', k=1, sigma=0.0):
    """Return (A csr, b) for  (sigma*Mlump + K) u = f  on an M^dim grid.

    K = conservative vertex-centred FD with harmonic face coefficients (a==1 -> std
    Laplacian).  sigma*Mlump = mass SHIFT (lumped mass Mlump=h^dim) -- mimics the
    monodomain (1/dt)M term (sigma=1/dt).  Boundary:
      bc='full'    Dirichlet on the whole boundary
      bc='kfaces'  Dirichlet on the first k faces (order: x0-,x0+,x1-,x1+,...)
      bc='mixed'   Dirichlet on x0=0 only  (== kfaces, k=1)
      bc='neumann' pure Neumann (NO Dirichlet) -> SINGULAR if sigma==0 (= Sys2)
    Dirichlet by symmetric elimination so A stays symmetric.
    """
    shp = grid_shape(dim, M)
    Npts = int(np.prod(shp))
    h = 1.0 / (M - 1)
    hinv2 = 1.0 / (h * h)
    idx = np.arange(Npts).reshape(shp)
    a = a_nodal.reshape(shp)

    rows = []; cols = []; vals = []
    for d in range(dim):
        lo = idx.take(np.arange(0, M - 1), axis=d).ravel(order='C')
        hi = idx.take(np.arange(1, M),     axis=d).ravel(order='C')
        a_lo = a.take(np.arange(0, M - 1), axis=d).ravel(order='C')
        a_hi = a.take(np.arange(1, M),     axis=d).ravel(order='C')
        w = (2.0 * a_lo * a_hi / (a_lo + a_hi)) * hinv2
        rows += [lo, hi]; cols += [hi, lo]; vals += [-w, -w]
        rows += [lo, hi]; cols += [lo, hi]; vals += [w, w]
    A = sp.csr_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(Npts, Npts))
    # mass/time-step shift: A <- K + sigma*I  (sigma directly regularizes the
    # constant mode; sigma ~ (1/dt)*lumped-mass. sigma large -> well-conditioned
    # (Sys1-like); sigma->0 on pure Neumann -> singular (Sys2-like).
    if sigma != 0.0:
        A = (A + sp.diags(np.full(Npts, float(sigma)))).tocsr()

    # select Dirichlet faces
    face_list = [(ax, side) for ax in range(dim) for side in (0, 1)]
    if bc == 'full':
        faces = face_list
    elif bc == 'mixed':
        faces = [(0, 0)]
    elif bc == 'kfaces':
        faces = face_list[:k]
    elif bc == 'neumann':
        faces = []
    else:
        raise ValueError('bc=%r' % bc)
    multi = np.indices(shp)
    dmask = np.zeros(Npts, bool)
    for (ax, side) in faces:
        val = 0 if side == 0 else (M - 1)
        dmask |= (multi[ax].ravel(order='C') == val)

    if dmask.any():
        keep = (~dmask).astype(float)
        P = sp.diags(keep)
        A = (P @ A @ P + sp.diags(dmask.astype(float))).tocsr()
        A.eliminate_zeros()
    b = np.ones(Npts)
    b[dmask] = 0.0
    return A, b


# --------------------------------------------------------------------------
#  Subdomain partition: S boxes per axis, grown by `overlap` grid points
# --------------------------------------------------------------------------
def box_subdomains(dim, M, S, overlap):
    """Return list of index arrays (global dof ids), one per subdomain.

    Non-overlapping base = split the M points per axis into S contiguous ranges;
    overlap = extend each range by `overlap` points on each side (clipped).
    Mirrors PCASM's MatIncreaseOverlap on a regular box partition.
    """
    shp = grid_shape(dim, M)
    # per-axis base ranges
    edges = np.linspace(0, M, S + 1).astype(int)     # S+1 cut points
    base_ranges = [(edges[s], edges[s + 1]) for s in range(S)]   # [lo, hi)

    # build the d-fold product of axis boxes
    from itertools import product
    subs = []
    idx_all = np.arange(int(np.prod(shp))).reshape(shp)
    for combo in product(range(S), repeat=dim):
        slc = []
        for d in range(dim):
            lo, hi = base_ranges[combo[d]]
            lo = max(0, lo - overlap)
            hi = min(M, hi + overlap)
            slc.append(slice(lo, hi))
        sub = idx_all[tuple(slc)].ravel(order='C')
        subs.append(np.sort(sub))
    return subs


def multiplicity(Npts, subs):
    m = np.zeros(Npts)
    for s in subs:
        m[s] += 1.0
    return m


# --------------------------------------------------------------------------
#  Local solves:  exact (LU)  or  IC(0) incomplete Cholesky (no fill)
# --------------------------------------------------------------------------
def ic0_factor(Aloc):
    """IC(0) of an SPD csr matrix: Cholesky restricted to A's lower pattern.
    Returns lower-triangular L (csr) with L L^T ~ Aloc.  Standard IKJ no-fill.
    """
    A = Aloc.tocsr()
    n = A.shape[0]
    Ad = A.toarray()                       # local blocks are small
    pattern = (Ad != 0.0)
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            if not pattern[i, j]:
                continue
            s = Ad[i, j] - L[i, :j] @ L[j, :j]
            if i == j:
                if s <= 0:
                    s = abs(Ad[i, i]) * 1e-12 + 1e-300   # tiny shift, keep SPD-ish
                L[i, i] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    return L


class LocalSolver:
    """Wraps a subdomain block A_i with either an exact or an IC(0) inverse-apply."""
    def __init__(self, Aloc, mode):
        self.n = Aloc.shape[0]
        self.mode = mode
        if mode == 'exact':
            self.lu = splu(sp.csc_matrix(Aloc))
        elif mode == 'ic0':
            self.L = ic0_factor(Aloc)
        else:
            raise ValueError(mode)

    def apply(self, r):
        if self.mode == 'exact':
            return self.lu.solve(r)
        # M_i^{-1} = (L L^T)^{-1}:  forward then back substitution
        y = np.linalg.solve(self.L, r) if False else _trisolve(self.L, r, lower=True)
        x = _trisolve(self.L.T, y, lower=False)
        return x

    def precond_spectrum(self, Aloc):
        """Eigenvalues of M_i^{-1} A_i  (== 1 if exact)."""
        if self.mode == 'exact':
            return np.ones(self.n)
        Minv = self._dense_inv()
        return np.linalg.eigvalsh(Minv @ Aloc.toarray())

    def _dense_inv(self):
        Linv = np.linalg.inv(self.L)
        return Linv.T @ Linv             # (L L^T)^{-1}


def _trisolve(T, b, lower):
    """Triangular solve T x = b (T dense ndarray)."""
    from scipy.linalg import solve_triangular
    return solve_triangular(T, b, lower=lower)


# --------------------------------------------------------------------------
#  Additive Schwarz preconditioner:  BASIC (unscaled) or sASM (D^{-1/2})
# --------------------------------------------------------------------------
class AdditiveSchwarz(LinearOperator):
    def __init__(self, A, subs, local_mode, scaling):
        self.A = A.tocsr()
        self.N = A.shape[0]
        super().__init__(dtype=np.float64, shape=(self.N, self.N))
        self.subs = subs
        self.scaling = scaling          # 'basic' or 'sasm'
        self.solvers = []
        for s in subs:
            Aloc = self.A[s][:, s]
            self.solvers.append(LocalSolver(Aloc, local_mode))
        m = multiplicity(self.N, subs)
        self.mult = m
        self.Nhat = int(m.max())
        self.inv_sqrt = 1.0 / np.sqrt(np.maximum(m, 1.0))   # D^{-1/2}

    def _basic_apply(self, r):
        z = np.zeros(self.N)
        for s, slv in zip(self.subs, self.solvers):
            z[s] += slv.apply(r[s])
        return z

    def _matvec(self, r):
        if self.scaling == 'basic':
            return self._basic_apply(r)
        # sASM:  D^{-1/2} ( sum R^T A_i^{-1} R ) D^{-1/2}
        z = self._basic_apply(self.inv_sqrt * r)
        return self.inv_sqrt * z

    # omega = max over subdomains of lam_max(M_i^{-1} A_i)
    def omega(self):
        w = 0.0
        for s, slv in zip(self.subs, self.solvers):
            Aloc = self.A[s][:, s]
            ev = slv.precond_spectrum(Aloc)
            w = max(w, ev.max())
        return w

    # exact spectrum of M^{-1}A via a similarity transform to a SYMMETRIC matrix.
    # A = L L^T (Cholesky) => M^{-1}A is similar to S = L^T M^{-1} L (symmetric),
    # so eigvalsh(S) are exactly the (real, positive) eigenvalues of M^{-1}A.
    def sym_similar(self):
        Adense = self.A.toarray()
        L = np.linalg.cholesky(Adense)            # A = L L^T, L lower
        MinvL = np.empty((self.N, self.N))
        for j in range(self.N):
            MinvL[:, j] = self._matvec(L[:, j])
        S = L.T @ MinvL
        return 0.5 * (S + S.T)                     # kill round-off asymmetry only


# --------------------------------------------------------------------------
#  Measurement
# --------------------------------------------------------------------------
def spectrum(asm, dense_limit=5200, full=False):
    """Return (lam_min, lam_max, kappa) of M^{-1} A (exact, real, positive).
    Exact dense eigvalsh of the symmetric similar matrix if N <= dense_limit,
    else eigsh extreme-eigenvalue estimates on the (symmetrised) operator.
    If full, also returns the full eigenvalue array as a 4th element.
    """
    N = asm.N
    if N <= dense_limit:
        S = asm.sym_similar()
        ev = np.linalg.eigvalsh(S)
        ev = ev[ev > 1e-10]
        res = (ev.min(), ev.max(), ev.max() / ev.min())
        return res + (ev,) if full else res
    # large N: extreme eigenvalues of the symmetric similar operator via eigsh.
    Adense = None  # avoid forming dense A; use Cholesky-free route is not symmetric,
    # so we settle for lam_max of M^{-1}A through the A-inner-product Rayleigh op.
    op = LinearOperator((N, N), matvec=lambda x: asm._matvec(asm.A @ x), dtype=np.float64)
    lam_max = float(eigsh(op, k=1, which='LM', return_eigenvectors=False,
                          maxiter=8000, tol=1e-6)[0])
    res = (np.nan, lam_max, np.nan)
    return res + (None,) if full else res


def cg_iters(A, b, asm, rtol=1e-6, maxit=2000):
    it = {'n': 0}
    def cb(xk):
        it['n'] += 1
    # scipy 1.9: cg uses `tol` (relative to ||b||) and `atol`
    x, info = cg(A, b, rtol=rtol, atol=0.0, maxiter=maxit,
                 M=asm, callback=cb) if False else _cg_compat(A, b, asm, rtol, maxit, cb)
    return it['n'], info


def _cg_compat(A, b, asm, rtol, maxit, cb):
    # scipy<1.12 uses tol=, >=1.12 uses rtol=; handle both.
    try:
        return cg(A, b, rtol=rtol, atol=0.0, maxiter=maxit, M=asm, callback=cb)
    except TypeError:
        return cg(A, b, tol=rtol, atol=0.0, maxiter=maxit, M=asm, callback=cb)


# --------------------------------------------------------------------------
#  One experiment cell
# --------------------------------------------------------------------------
def run_cell(dim, M, S, overlap, local_mode, scaling, coef='const',
             contrast=1.0, n_layers=4, want_spectrum=True, rtol=1e-6,
             bc='mixed', k=1, sigma=0.0):
    a = coef_field(coef, dim, M, contrast=contrast, n_layers=n_layers)
    A, b = assemble(dim, M, a, bc=bc, k=k, sigma=sigma)
    subs = box_subdomains(dim, M, S, overlap)
    asm = AdditiveSchwarz(A, subs, local_mode, scaling)
    out = dict(dim=dim, M=M, S=S, overlap=overlap, local=local_mode,
               scaling=scaling, coef=coef, contrast=contrast,
               bc=bc, k=k, sigma=sigma,
               N=A.shape[0], n_sub=len(subs), Nhat=asm.Nhat)
    its, info = cg_iters(A, b, asm, rtol=rtol)
    out['iter'] = its
    out['cg_info'] = info
    if want_spectrum:
        lmn, lmx, kap = spectrum(asm)
        out['lam_min'] = lmn
        out['lam_max'] = lmx
        out['kappa'] = kap
        out['omega'] = asm.omega()
    return out


if __name__ == '__main__':
    # smoke test: 1D vs 2D vs 3D, IC(0) local, BASIC, overlap 0->2
    import sys
    for dim, M, S in [(1, 65, 4), (2, 33, 4), (3, 17, 4)]:
        print("=== dim=%d  M=%d  S=%d  (N=%d) ===" % (dim, M, S, M ** dim))
        for ov in (0, 1, 2):
            r = run_cell(dim, M, S, ov, 'ic0', 'basic', coef='const')
            print("  O=%d  iter=%3d  Nhat=%d  omega=%.3f  lam_max=%.3f "
                  "lam_min=%.4f  kappa=%.1f"
                  % (ov, r['iter'], r['Nhat'], r['omega'],
                     r['lam_max'], r['lam_min'], r['kappa']))
