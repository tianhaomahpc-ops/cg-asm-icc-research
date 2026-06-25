#!/usr/bin/env python3
"""
theory_overlap.py -- closed-form / quantitative analysis of the open theory point
(proposition (5) in REPORT_unified_zh.md sec 7.4):

    "A x B  =>  kappa(M_BASIC^-1 A) strictly increases with overlap delta"
    had a mechanism + empirical support but no closed-form statement.

This is a SELF-CONTAINED numpy/scipy reproduction of the BASIC/sASM x exact/ICC(0)
additive-Schwarz construction of asm_bug_demo/coloring.c (no PETSc/MFEM needed),
used to MEASURE how each factor of the abstract one-level bound

        kappa(M^-1 A)  <=  C0^2 (delta) * (wmax/wmin)(delta) * Nhat(delta)

depends on the overlap delta and the dimension d, and to validate the
closed-form laws derived in REPORT_theory_overlap_zh.md:

  * Prop 1   lambda_max(M_BASIC^-1 A | exact) = Nhat   (over-counting, delta-capped)
  * Law A    omega(delta) = kappa(M_IC0^-1 A_i) = Theta((H+2 delta)^2 / h^2)
             (classical unmodified IC(0) order; 1D: omega == 1 exactly)
  * Criterion  anomaly across an overlap step  <=>  rho_max > rho_min,
             onset rho_max ~ 2^{d-1};  flip needs SIZE-GROWING omega (control below).

Conventions match coloring.c exactly:
  A        d-dim Dirichlet Laplacian on n^d grid (2d diag, -1 neighbours), SPD
  subs     box decomposition into P^d cores, each grown by O layers, clipped
  A_i      = R_i A R_i^T  (principal submatrix)
  BASIC    M_B^-1 = sum_i R_i^T Ahat_i^-1 R_i
  D        = diag(multiplicity m_k);   sASM   M_S^-1 = D^-1/2 M_B^-1 D^-1/2
  exact    Ahat_i^-1 = A_i^-1     ICC   Ahat_i^-1 = (L L^T)^-1, L = no-fill ichol
  filter   exact but smoothest `frac` local modes under-resolved by gamma (control)

Subcommands:
  law            kappa(M_IC0^-1 A_block) vs block size  -> Law A  (fast)
  sweep DIM      lambda_min/max/kappa vs O, BASIC/sASM x exact/ICC  (DIM=1d|2d|3d|2dmany)
  cross          gamma crossover: uniform inexactness does NOT flip kappa  (control)
  fig            regenerate the summary figure  fig14_theory_overlap.png

Run any subcommand with no args for its default. Sequential, dense; keep n modest.
"""
import sys, math
import numpy as np
import scipy.linalg as sla
from itertools import product

np.set_printoptions(suppress=True)


# ---------------------------------------------------------------- operators
def laplace(n, d):
    N = n ** d
    A = np.zeros((N, N))
    def lin(idx):
        s = 0
        for c in idx:
            s = s * n + c
        return s
    for idx in product(range(n), repeat=d):
        k = lin(idx)
        A[k, k] = 2.0 * d
        for ax in range(d):
            for step in (-1, +1):
                nb = list(idx); nb[ax] += step
                if 0 <= nb[ax] < n:
                    A[k, lin(tuple(nb))] = -1.0
    return A


def subdomains(n, d, P, O):
    def lin(idx):
        s = 0
        for c in idx:
            s = s * n + c
        return s
    subs = []
    for cell in product(range(P), repeat=d):
        ranges = []
        for ax in range(d):
            lo = max(0, (cell[ax] * n) // P - O)
            hi = min(n, ((cell[ax] + 1) * n) // P + O)
            ranges.append(range(lo, hi))
        subs.append(np.array(sorted(lin(t) for t in product(*ranges)), dtype=int))
    return subs


# ------------------------------------------------- no-fill incomplete Cholesky
def ichol0(A):
    n = A.shape[0]
    a = A.astype(float).copy()
    S = (A != 0.0)
    for k in range(n):
        a[k, k] = math.sqrt(a[k, k])
        col = np.where(S[k + 1:, k])[0] + (k + 1)
        a[col, k] /= a[k, k]
        for j in col:
            rows = col[(col >= j) & S[col, j]]
            a[rows, j] -= a[rows, k] * a[j, k]
    return np.tril(a) * np.tril(S)


def local_inverse(Ai, mode, gamma=1.0, frac=0.5):
    if mode == "exact":
        return np.linalg.inv(Ai)
    if mode == "icc":
        L = ichol0(Ai); M = L @ L.T
        return np.linalg.inv(M)
    if mode == "filter":
        w, Q = np.linalg.eigh(Ai)
        inv = 1.0 / w
        scale = np.ones_like(w)
        scale[:int(round(frac * len(w)))] = gamma   # damp smallest (smoothest)
        return (Q * (inv * scale)) @ Q.T
    raise ValueError(mode)


# ------------------------------------------------------ preconditioner + spectrum
_SQRT = {}
def sqrtm_psd(A):
    key = (A.shape[0], float(A[0, 0]), float(A.sum()))
    if key not in _SQRT:
        w, V = np.linalg.eigh(A)
        _SQRT[key] = (V * np.sqrt(np.clip(w, 1e-14, None))) @ V.T
    return _SQRT[key]


def extreme(A, Minv):
    As = sqrtm_psd(A)
    S = As @ Minv @ As; S = 0.5 * (S + S.T); N = S.shape[0]
    lmax = sla.eigh(S, subset_by_index=[N - 1, N - 1], eigvals_only=True)[0]
    lmin = sla.eigh(S, subset_by_index=[0, 0], eigvals_only=True)[0]
    return lmin, lmax


def build_prec(A, subs, mode, gamma=1.0, frac=0.5):
    N = A.shape[0]; MB = np.zeros((N, N)); mult = np.zeros(N); locK = []; locL = []
    for idx in subs:
        Ai = A[np.ix_(idx, idx)]
        Minv = local_inverse(Ai, mode, gamma, frac)
        MB[np.ix_(idx, idx)] += Minv; mult[idx] += 1.0
        if mode == "exact":
            locK.append(1.0); locL.append(1.0)
        else:
            MA = Minv @ Ai; w = np.linalg.eigvalsh(0.5 * (MA + MA.T))
            w = w[w > 1e-13]; locK.append(w.max() / w.min()); locL.append(w.max())
    dsq = 1.0 / np.sqrt(mult)
    MS = dsq[:, None] * MB * dsq[None, :]
    return MB, MS, mult, np.array(locK), np.array(locL)


def coloring(subs, N):
    mult = np.zeros(N, dtype=int)
    for idx in subs:
        mult[idx] += 1
    nhat = int(mult.max())
    sets = [set(i.tolist()) for i in subs]; ns = len(subs); adj = [[] for _ in range(ns)]
    for i in range(ns):
        for j in range(i + 1, ns):
            if sets[i] & sets[j]:
                adj[i].append(j); adj[j].append(i)
    col = [-1] * ns
    for i in range(ns):
        used = {col[j] for j in adj[i] if col[j] >= 0}; c = 0
        while c in used:
            c += 1
        col[i] = c
    return nhat, max(col) + 1


# ---------------------------------------------------------------- subcommands
def cmd_law():
    print("Law A:  kappa(M_IC0^-1 A_block) vs block linear size m   (kappa/m^2 -> const)")
    for d in (1, 2, 3):
        print(f"\n=== d={d} ===")
        print(f"{'m':>3} {'lmin':>9} {'lmax':>7} {'kappa':>9} {'kappa/m^2':>10}")
        ms = {1: [16, 32, 64, 128, 256], 2: [8, 12, 16, 24, 32, 40], 3: [6, 8, 10, 12, 14]}[d]
        for m in ms:
            A = laplace(m, d); L = ichol0(A); M = L @ L.T
            MA = np.linalg.solve(M, A); w = np.linalg.eigvalsh(0.5 * (MA + MA.T))
            w = w[w > 1e-12]; k = w.max() / w.min()
            print(f"{m:>3} {w.min():9.5f} {w.max():7.3f} {k:9.2f} {k / m**2:10.4f}")


def cmd_sweep(which="2dmany"):
    cfg = {"1d": (1, 96, 8, [1, 2, 4, 8, 12]),
           "2d": (2, 48, 4, [1, 2, 3, 4, 6, 8]),
           "3d": (3, 16, 2, [1, 2, 3, 4]),
           "2dmany": (2, 64, 8, [0, 1, 2, 3, 4])}[which]
    d, n, P, Os = cfg
    for mode in ("exact", "icc"):
        A = laplace(n, d); N = A.shape[0]
        print(f"\n=== d={d} n={n} P={P} (N={N})  mode={mode} ===")
        print(f"{'O':>2} {'Nhat':>4} {'Nc':>3} | {'lmin_B':>8} {'lmax_B':>8} {'kap_B':>8}"
              f" | {'lmin_S':>8} {'lmax_S':>8} {'kap_S':>8} | {'wmax/wmin':>9}")
        for O in Os:
            subs = subdomains(n, d, P, O); nhat, Nc = coloring(subs, N)
            MB, MS, mult, lk, ll = build_prec(A, subs, mode)
            lmB, lMB = extreme(A, MB); lmS, lMS = extreme(A, MS)
            print(f"{O:>2} {nhat:>4} {Nc:>3} | {lmB:8.4f} {lMB:8.3f} {lMB/lmB:8.1f}"
                  f" | {lmS:8.4f} {lMS:8.3f} {lMS/lmS:8.1f} | {lk.max():9.2f}")


def cmd_cross():
    d, n, P, frac = 2, 56, 7, 0.5
    A = laplace(n, d); extreme(A, np.eye(A.shape[0]))
    print(f"Control: uniform inexactness (gamma) does NOT flip kappa  (2D n={n} P={P})")
    print(f"{'gamma':>6} | {'kapB O0':>8} {'kapB O1':>8} {'B ratio':>7} {'anom':>5}"
          f" | {'kapS O0':>8} {'kapS O1':>8} {'S ratio':>7}")
    def kap(O, g):
        MB, MS, *_ = build_prec(A, subdomains(n, d, P, O), "filter", g, frac)
        lmB, lMB = extreme(A, MB); lmS, lMS = extreme(A, MS)
        return lMB / lmB, lMS / lmS
    for g in [1.0, 0.6, 0.4, 0.3, 0.22, 0.15, 0.1, 0.05]:
        k0B, k0S = kap(0, g); k1B, k1S = kap(1, g)
        rB, rS = k1B / k0B, k1S / k0S
        print(f"{g:6.2f} | {k0B:8.1f} {k1B:8.1f} {rB:7.2f} {'YES' if rB > 1 else '-':>5}"
              f" | {k0S:8.1f} {k1S:8.1f} {rS:7.2f}")


def cmd_fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # (a) Law A: kappa(M_IC0^-1 A_block) vs m, log-log, d=1/2/3
    lawd = {}
    for d in (1, 2, 3):
        ms = {1: [16, 32, 64, 128], 2: [8, 12, 16, 24, 32], 3: [6, 8, 10, 12]}[d]
        ks = []
        for m in ms:
            A = laplace(m, d); L = ichol0(A); M = L @ L.T
            MA = np.linalg.solve(M, A); w = np.linalg.eigvalsh(0.5 * (MA + MA.T))
            w = w[w > 1e-12]; ks.append(w.max() / w.min())
        lawd[d] = (np.array(ms), np.array(ks))

    # (b) anomaly + sASM fix: kappa vs O, exact/ICC, BASIC/sASM (2D many subdomains)
    d, n, P = 2, 64, 8; A = laplace(n, d); extreme(A, np.eye(A.shape[0]))
    Os = [0, 1, 2, 3, 4]; res = {("exact", "B"): [], ("exact", "S"): [],
                                 ("icc", "B"): [], ("icc", "S"): []}
    for mode in ("exact", "icc"):
        for O in Os:
            MB, MS, *_ = build_prec(A, subdomains(n, d, P, O), mode)
            lmB, lMB = extreme(A, MB); lmS, lMS = extreme(A, MS)
            res[(mode, "B")].append(lMB / lmB); res[(mode, "S")].append(lMS / lmS)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    mk = {1: "o", 2: "s", 3: "^"}
    for d in (1, 2, 3):
        ms, ks = lawd[d]
        ax[0].loglog(ms, ks, mk[d] + "-", label=f"d={d}")
    mm = np.array([8, 40]); ax[0].loglog(mm, 0.04 * mm**2, "k--", lw=1, label=r"$0.04\,m^2$")
    ax[0].set_xlabel("subdomain linear size  m  (= (H+2$\\delta$)/h)")
    ax[0].set_ylabel(r"$\omega=\kappa(M_{IC0}^{-1}A_i)$")
    ax[0].set_title("(a) Law A: inexactness grows as $(H{+}2\\delta)^2$\n(d=1: $\\equiv$1, IC0=exact)")
    ax[0].legend(); ax[0].grid(True, which="both", alpha=0.3)

    ax[1].plot(Os, res[("exact", "B")], "o-", color="tab:red", label="BASIC, exact")
    ax[1].plot(Os, res[("icc", "B")], "s--", color="tab:red", label="BASIC, ICC(0)")
    ax[1].plot(Os, res[("exact", "S")], "o-", color="tab:blue", label="sASM, exact")
    ax[1].plot(Os, res[("icc", "S")], "s--", color="tab:blue", label="sASM, ICC(0)")
    ax[1].set_xlabel("overlap O (layers)"); ax[1].set_ylabel(r"$\kappa(M^{-1}A)$")
    ax[1].set_title("(b) over-counting onset O=0$\\to$1:\nBASIC+ICC rises (anomaly); sASM does not")
    ax[1].set_xticks(Os); ax[1].legend(); ax[1].grid(True, alpha=0.3)
    ax[1].annotate("anomaly\n(over-counting\n$\\times$ inexact)", xy=(1, res[("icc", "B")][1]),
                   xytext=(2.0, res[("icc", "B")][1] * 0.80), ha="left",
                   arrowprops=dict(arrowstyle="->"))

    fig.tight_layout()
    out = "infoprop_out/fig14_theory_overlap.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "law"
    if cmd == "law":
        cmd_law()
    elif cmd == "sweep":
        cmd_sweep(sys.argv[2] if len(sys.argv) > 2 else "2dmany")
    elif cmd == "cross":
        cmd_cross()
    elif cmd == "fig":
        cmd_fig()
    else:
        print(__doc__)
