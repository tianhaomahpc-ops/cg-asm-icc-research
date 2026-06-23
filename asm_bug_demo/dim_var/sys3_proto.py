"""sys3_proto.py -- Sys3 (torso) cross-mesh interface scheme.

Sys3 lives on a DIFFERENT mesh from Sys1/Sys2 and is coupled only through the
epicardial INTERFACE: the heart-surface face carries Dirichlet data u_e (from Sys2),
the body surface is Neumann. It is therefore NON-singular (no constant nullspace,
unlike Sys2) and its difficulty is size + mixed BC + the overlap anomaly, plus a
Dirichlet datum that moves smoothly with the activation front each timestep.

Cross-mesh reuse (you cannot share the heart-mesh V0):
  1. per-mesh hybrid coarse built ONCE on the torso mesh  -> base conditioning;
  2. temporal coherence: the volume conductor is a smooth (low-rank) map of the
     interface data, so warm-starting across timesteps is effective -- and the front
     position (known from Sys1) localizes where the solution changes ("front-following").

Proxy: torso = a square mesh (different resolution from the heart), Dirichlet on the
x0=0 interface face with a moving-front profile g(y,t), Neumann elsewhere. Solve the
sequence A_ff u_t = -K_{fD} g_t and compare one-level / two-level / two-level+warm,
plus an SVD low-rank diagnostic vs the Sys2 source sequence.
"""
import os, csv
import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 12, 'axes.labelsize': 12,
                     'legend.fontsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10})
import asm_spectral as A
import coarse_proto as C
from xsys_proto import cg_count

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def build_torso(dim, M, coef='const', rho=1.0):
    """Torso operator with x0=0 face = epicardial interface (Dirichlet), rest Neumann.
    Returns A_ff (free-space SPD), K_fD (free<->interface coupling), free idx, coords."""
    a = A.coef_field(coef, dim, M, contrast=rho)
    Kfull, _ = A.assemble(dim, M, a, bc='neumann', sigma=0.0)   # pure operator (no elim)
    X = A.node_coords(dim, M)
    iface = np.where(np.isclose(X[:, 0], 0.0))[0]               # interface face
    free = np.setdiff1d(np.arange(X.shape[0]), iface)
    Aff = Kfull[free][:, free].tocsr()
    KfD = (Kfull[free][:, iface]).tocsr()
    return a, Aff, free, iface, KfD, X


def free_subdomains(dim, M, S, ov, free):
    """Box subdomains intersected with the free (non-interface) nodes, in free-local idx."""
    subs_full = A.box_subdomains(dim, M, S, ov)
    g2l = -np.ones(M ** dim, dtype=int); g2l[free] = np.arange(len(free))
    out = []
    for s in subs_full:
        loc = g2l[s]; loc = loc[loc >= 0]
        if len(loc) > 0:
            out.append(np.sort(loc))
    return out


def front_profile(yc, Xface, w=0.07):
    """Moving-front Dirichlet data on the interface (the u_e the front imprints)."""
    return np.exp(-((Xface - yc) ** 2) / (2 * w ** 2))


def eff_rank(U, tol=0.01):
    s = np.linalg.svd(U, compute_uv=False)
    return int(np.sum(s > tol * s[0])), s


def run(dim=2, Mtorso=41, S=4, ov=1, T=12):
    a, Aff, free, iface, KfD, X = build_torso(dim, Mtorso, 'const')
    subs = free_subdomains(dim, Mtorso, S, ov, free)
    one = A.AdditiveSchwarz(Aff, subs, 'ic0', 'sasm')
    V0, _ = C.low_modes(Aff, 8)                                 # per-mesh coarse, built ONCE
    two = C.TwoLevel(Aff, one, V0)
    Xface = X[iface, 1] if dim >= 2 else np.zeros(len(iface))   # interface coordinate

    ycs = np.linspace(0.2, 0.8, T)
    tot = dict(one_cold=0, two_cold=0, two_warm=0)
    per, sols = [], []
    xprev = None
    for t, yc in enumerate(ycs):
        g = front_profile(yc, Xface)
        b = -(KfD @ g)                                          # Dirichlet lifting RHS
        _, i1 = cg_count(Aff, b, one)
        u2, i2 = cg_count(Aff, b, two)
        uw, i3 = cg_count(Aff, b, two, x0=xprev)
        xprev = uw; sols.append(uw)
        tot['one_cold'] += i1; tot['two_cold'] += i2; tot['two_warm'] += i3
        per.append((t, i1, i2, i3))
        print("  t=%2d front_y=%.2f | one-cold %3d | two-cold %3d | two-warm %3d"
              % (t, yc, i1, i2, i3))
    print("  TOTAL over %d solves: one-cold=%d two-cold=%d two-warm=%d" %
          (T, tot['one_cold'], tot['two_cold'], tot['two_warm']))
    print("  speedup vs one-level cold: two-cold %.2fx  two-warm %.2fx" %
          (tot['one_cold'] / tot['two_cold'], tot['one_cold'] / tot['two_warm']))

    # low-rank diagnostic: torso solution sequence vs a Sys2 (pure-Neumann source) sequence
    U3 = np.array(sols).T
    r3, s3 = eff_rank(U3)
    # Sys2 proxy sequence (same heart-size mesh, pure Neumann, moving source)
    Mh = 33; ah = A.coef_field('const', dim, Mh); A2, _ = A.assemble(dim, Mh, ah, bc='neumann', sigma=1e-2)
    Xh = A.node_coords(dim, Mh); subs2 = A.box_subdomains(dim, Mh, S, ov)
    one2 = A.AdditiveSchwarz(A2, subs2, 'ic0', 'sasm'); V2, _ = C.low_modes(A2, 8); two2 = C.TwoLevel(A2, one2, V2, pinv=True)
    s2sols = []
    for yc in ycs:
        r2 = (Xh[:, 0] - yc) ** 2 + ((Xh[:, 1] - 0.5) ** 2 if dim >= 2 else 0)
        f = np.exp(-r2 / (2 * 0.06 ** 2)); f -= f.mean()
        u, _ = cg_count(A2, f, two2); s2sols.append(u)
    rS2, sS2 = eff_rank(np.array(s2sols).T)
    print("  low-rank: Sys3(torso) eff-rank=%d/%d ; Sys2(heart) eff-rank=%d/%d" % (r3, T, rS2, T))

    # figures
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    t = [p[0] for p in per]
    for idx, col, lab in [(1, 'C7', 'one-level cold'), (2, 'C0', 'two-level cold'), (3, 'C2', 'two-level + warm')]:
        ax[0].plot(t, [p[idx] for p in per], 'o-', color=col, label=lab)
    ax[0].set_xlabel('timestep'); ax[0].set_ylabel('CG iters / torso solve')
    ax[0].set_title('(a) Sys3 torso solve: per-step iterations'); ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)
    ax[1].semilogy(range(1, T + 1), s3 / s3[0], 'o-', color='C3', label='Sys3 torso (eff-rank %d)' % r3)
    ax[1].semilogy(range(1, T + 1), sS2 / sS2[0], 's--', color='C0', label='Sys2 heart (eff-rank %d)' % rS2)
    ax[1].axhline(0.01, color='0.6', ls=':', label='1% threshold')
    ax[1].set_xlabel('singular value index'); ax[1].set_ylabel('normalized singular value')
    ax[1].set_title('(b) solution-sequence spectrum (torso $\\approx$ heart rank)')
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    sp_c = tot['one_cold'] / tot['two_cold']; sp_w = tot['one_cold'] / tot['two_warm']
    fig.suptitle('Sys3 cross-mesh: per-mesh coarse + temporal coherence (two-cold %.2fx, two-warm %.2fx)'
                 % (sp_c, sp_w), fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(RES, 'fig_sys3.png'), dpi=130); plt.close(fig)
    print('  fig_sys3.png')
    with open(os.path.join(RES, 'sys3_sequence.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['t', 'one_cold', 'two_cold', 'two_warm']); w.writerows(per)
    return per, tot, (r3, rS2)


if __name__ == '__main__':
    print("=== Sys3 (torso) cross-mesh interface scheme ===")
    run()
