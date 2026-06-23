"""xsys_proto.py -- cross-system shared-coarse proxy (Sys1 -> Sys2), beyond Task-3.

Same mesh for Sys1 (monodomain) and Sys2 (u_e recovery). Model both as the SAME
stiffness K at a mass/time-step shift sigma:
    A(sigma) = K + sigma*I        large sigma ~ Sys1 (1/dt, easy);  sigma->0 ~ Sys2 (singular).

Idea under test: build ONE hybrid coarse on the shared mesh
    V0 = [ few shift-invariant (K,M) low spectral modes ]  (smooth global + constant/nullspace)
       + [ A-harmonic known-coefficient modes ]            (contrast/anisotropy, non-GenEO)
ONCE (sigma-independent, coefficient-derived), and reuse it for the whole Sys1->Sys2
range AND across a timestep sequence. Compare to one-level sASM and to warm-starting,
the analog of Task-3's shared-Nicolaides+warm (avg CG 46->28.9, 1.6x).

Outputs results/xsys_*.csv and figures.
"""
import os, csv
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import asm_spectral as A
import coarse_proto as C

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def build_hybrid_coarse(dim, M, a, Kneu, n_spec=4, use_coef=True):
    """V0 built ONCE from the pure-Neumann stiffness K and the known coefficient."""
    Vsp, _ = C.low_modes(Kneu, n_spec)              # shift-invariant smooth + constant
    if use_coef and (a.max() / a.min() > 2.0):
        Vc = C.coef_harmonic_modes(a, dim, M, Kneu)  # known-coef A-harmonic modes
        return np.column_stack([Vsp, Vc])
    return Vsp


def cg_count(Amat, b, prec, x0=None, rtol=1e-8, maxit=2000):
    it = {'n': 0}
    cb = lambda xk: it.__setitem__('n', it['n'] + 1)
    kw = dict(maxiter=maxit, M=prec, callback=cb)
    if x0 is not None:
        kw['x0'] = x0
    try:
        x, info = cg(Amat, b, rtol=rtol, atol=0.0, **kw)
    except TypeError:
        x, info = cg(Amat, b, tol=rtol, atol=0.0, **kw)
    return x, it['n']


# --------------------------------------------------------------------------
def partA_sigma_sweep(dim=2, M=33, S=4, ov=1):
    """ONE coarse reused across the Sys1->Sys2 shift range, for const & contrast coef."""
    subs = A.box_subdomains(dim, M, S, ov)
    sigmas = [1e3, 1e1, 1e0, 1e-1, 1e-2, 1e-4]
    rows = []
    for coef, rho in [('const', 1.0), ('layers', 10.0)]:
        a = A.coef_field(coef, dim, M, contrast=rho)
        Kneu, _ = A.assemble(dim, M, a, bc='neumann', sigma=0.0)
        V0 = build_hybrid_coarse(dim, M, a, Kneu, n_spec=4, use_coef=(rho > 1))   # built ONCE
        X = A.node_coords(dim, M)
        r2 = (X[:, 0] - 0.3) ** 2 + ((X[:, 1] - 0.5) ** 2 if dim >= 2 else 0)
        b = np.exp(-r2 / (2 * 0.08 ** 2)); b -= b.mean()      # non-trivial compatible RHS
        for sigma in sigmas:
            Amat, _ = A.assemble(dim, M, a, bc='neumann', sigma=sigma)
            one = A.AdditiveSchwarz(Amat, subs, 'ic0', 'sasm')
            two = C.TwoLevel(Amat, one, V0, pinv=True)
            _, it1 = cg_count(Amat, b, one)
            _, it2 = cg_count(Amat, b, two)
            k1 = C.kap(one) if sigma >= 1e-3 else (np.nan,) * 3
            k2 = C.kap(two) if sigma >= 1e-3 else (np.nan,) * 3
            rows.append(dict(coef=coef, rho=rho, sigma=sigma, m=V0.shape[1],
                             it_one=it1, it_two=it2, kap_one=k1[2], kap_two=k2[2]))
            print("  %-7s rho=%-4g sigma=%-7g | one-level it=%3d kap=%.1e | "
                  "two-level it=%3d kap=%.1e  (m=%d, ONE coarse)"
                  % (coef, rho, sigma, it1, k1[2], it2, k2[2], V0.shape[1]))
    with open(os.path.join(RES, 'xsys_sigma.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return rows


def partB_sequence(dim=2, M=33, S=4, ov=1, sigma=1e-2, T=12, coef='layers', rho=10.0):
    """Timestep sequence at the hard (Sys2-like) end: a moving 'front' RHS.
    Compare total CG iters: one-level cold, two-level cold, two-level warm-started."""
    subs = A.box_subdomains(dim, M, S, ov)
    a = A.coef_field(coef, dim, M, contrast=rho)
    Kneu, _ = A.assemble(dim, M, a, bc='neumann', sigma=0.0)
    V0 = build_hybrid_coarse(dim, M, a, Kneu, n_spec=4, use_coef=(rho > 1))   # built ONCE
    Amat, _ = A.assemble(dim, M, a, bc='neumann', sigma=sigma)
    one = A.AdditiveSchwarz(Amat, subs, 'ic0', 'sasm')
    two = C.TwoLevel(Amat, one, V0, pinv=True)

    X = A.node_coords(dim, M)                                  # (N, dim)
    xs = np.linspace(0.2, 0.8, T)                              # front position sweeps
    tot = dict(one_cold=0, two_cold=0, two_warm=0)
    per = []
    xprev = None
    for t, xc in enumerate(xs):
        r2 = (X[:, 0] - xc) ** 2 + (X[:, 1] - 0.5) ** 2 if dim >= 2 else (X[:, 0] - xc) ** 2
        f = np.exp(-r2 / (2 * 0.06 ** 2)); f -= f.mean()       # moving bump, compatible
        _, i1 = cg_count(Amat, f, one)
        _, i2 = cg_count(Amat, f, two)
        xw, i3 = cg_count(Amat, f, two, x0=xprev)              # warm: reuse previous solution
        xprev = xw
        tot['one_cold'] += i1; tot['two_cold'] += i2; tot['two_warm'] += i3
        per.append((t, i1, i2, i3))
        print("  t=%2d front_x=%.2f | one-cold %3d | two-cold %3d | two-warm %3d"
              % (t, xc, i1, i2, i3))
    print("  TOTAL over %d solves: one-cold=%d  two-cold=%d  two-warm=%d"
          % (T, tot['one_cold'], tot['two_cold'], tot['two_warm']))
    print("  speedup vs one-level cold:  two-cold %.2fx   two-warm %.2fx"
          % (tot['one_cold'] / tot['two_cold'], tot['one_cold'] / tot['two_warm']))
    with open(os.path.join(RES, 'xsys_sequence.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['t', 'one_cold', 'two_cold', 'two_warm']); w.writerows(per)
    return per, tot


def figures(rowsA, perB, totB):
    # fig 1: iters & kappa vs sigma (one coarse across Sys1->Sys2)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for coef, col in [('const', 'C0'), ('layers', 'C3')]:
        r = sorted([x for x in rowsA if x['coef'] == coef], key=lambda x: -x['sigma'])
        sg = [x['sigma'] for x in r]
        ax[0].plot(sg, [x['it_one'] for x in r], 'o-', color=col, label='%s one-level' % coef)
        ax[0].plot(sg, [x['it_two'] for x in r], 's--', color=col, alpha=0.65, label='%s two-level' % coef)
        ax[1].plot(sg, [x['kap_one'] for x in r], 'o-', color=col, label='%s one-level' % coef)
        ax[1].plot(sg, [x['kap_two'] for x in r], 's--', color=col, alpha=0.65, label='%s two-level' % coef)
    for a_ in ax:
        a_.set_xscale('log'); a_.invert_xaxis()
        a_.set_xlabel('shift $\\sigma$  (large = Sys1, $\\to0$ = Sys2)'); a_.grid(alpha=0.3)
    ax[0].set_ylabel('CG iterations'); ax[0].set_title('(a) iterations: ONE shared coarse across Sys1$\\to$Sys2')
    ax[0].legend(fontsize=8)
    ax[1].set_yscale('log'); ax[1].set_ylabel('$\\kappa(M^{-1}A)$'); ax[1].set_title('(b) condition number')
    ax[1].legend(fontsize=8)
    fig.suptitle('Cross-system shared coarse: built ONCE, flattens the whole Sys1$\\to$Sys2 range', fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(RES, 'fig_xsys_sigma.png'), dpi=130); plt.close(fig)
    print('  fig_xsys_sigma.png')

    # fig 2: sequence cumulative iters
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    t = [p[0] for p in perB]
    for k, idx, col, lab in [('one_cold', 1, 'C7', 'one-level cold'),
                             ('two_cold', 2, 'C0', 'two-level cold'),
                             ('two_warm', 3, 'C2', 'two-level + warm')]:
        ax[0].plot(t, [p[idx] for p in perB], 'o-', color=col, label=lab)
        ax[1].plot(t, np.cumsum([p[idx] for p in perB]), 'o-', color=col, label=lab)
    ax[0].set_xlabel('timestep'); ax[0].set_ylabel('CG iters / solve'); ax[0].set_title('(a) per-solve iterations')
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)
    ax[1].set_xlabel('timestep'); ax[1].set_ylabel('cumulative CG iters'); ax[1].grid(alpha=0.3)
    sp_cold = totB['one_cold'] / totB['two_cold']; sp_warm = totB['one_cold'] / totB['two_warm']
    ax[1].set_title('(b) cumulative: two-cold %.2fx, two-warm %.2fx vs one-level' % (sp_cold, sp_warm))
    ax[1].legend(fontsize=9)
    fig.suptitle('Sys2-like timestep sequence: shared coarse + warm start (analog of Task-3)', fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(RES, 'fig_xsys_sequence.png'), dpi=130); plt.close(fig)
    print('  fig_xsys_sequence.png')


if __name__ == '__main__':
    print("=== Part A: ONE shared coarse across the Sys1->Sys2 shift range ===")
    rowsA = partA_sigma_sweep()
    print("\n=== Part B: Sys2-like timestep sequence (cold vs warm) ===")
    perB, totB = partB_sequence()
    figures(rowsA, perB, totB)
