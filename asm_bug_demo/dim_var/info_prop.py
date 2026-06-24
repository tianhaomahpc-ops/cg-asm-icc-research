"""info_prop.py -- answers three interpretability questions raised on the spectrum slide:

Q1. Does sASM lower lambda_min while it lowers lambda_max? By how much?
    -> measure BOTH ends for BASIC vs sASM over the overlap sweep.

Q2/Q3. How does Dirichlet boundary data 'propagate'? With vs without Dirichlet?
    -> run the STATIONARY Schwarz iteration x_{k+1}=x_k+M^{-1}(b-A x_k) with b=0,
       x0=constant(=the global mode). Each M^{-1} application pushes information ~one
       subdomain. The Dirichlet boundary is the SOURCE: the error drains from it inward.
       Compare full-Dirichlet / 1-face / pure-Neumann (no source -> constant can't drain).
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 12, 'legend.fontsize': 10})
import asm_spectral as A

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def partA(dim=2, M=33, S=4):
    """lambda_min AND lambda_max for BASIC vs sASM over the overlap sweep."""
    a = A.coef_field('const', dim, M)
    ov = [0, 1, 2, 3, 4]
    out = {'basic': {'lmin': [], 'lmax': []}, 'sasm': {'lmin': [], 'lmax': []}}
    for sch in ('basic', 'sasm'):
        for O in ov:
            subs = A.box_subdomains(dim, M, S, O)
            Amat, _ = A.assemble(dim, M, a, bc='mixed')
            asm = A.AdditiveSchwarz(Amat, subs, 'ic0', sch)
            lmn, lmx, _ = A.spectrum(asm)
            out[sch]['lmin'].append(lmn); out[sch]['lmax'].append(lmx)
        print("  %-5s lmax=%s" % (sch, ['%.2f' % v for v in out[sch]['lmax']]))
        print("  %-5s lmin=%s" % (sch, ['%.4f' % v for v in out[sch]['lmin']]))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for sch, col, mk in [('basic', 'C3', 'o-'), ('sasm', 'C0', 's--')]:
        ax[0].plot(ov, out[sch]['lmax'], mk, color=col, label=sch)
        ax[1].plot(ov, out[sch]['lmin'], mk, color=col, label=sch)
    ax[0].set_title('(a) $\\lambda_{max}$: sASM removes the over-count'); ax[0].set_xlabel('overlap O')
    ax[0].set_ylabel('$\\lambda_{max}(M^{-1}A)$'); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].set_title('(b) $\\lambda_{min}$: sASM forgoes part of the overlap gain'); ax[1].set_xlabel('overlap O')
    ax[1].set_ylabel('$\\lambda_{min}(M^{-1}A)$'); ax[1].legend(); ax[1].grid(alpha=0.3)
    # annotate the lambda_min ratio
    rmin = [s / b for s, b in zip(out['sasm']['lmin'], out['basic']['lmin'])]
    ax[1].text(0.05, 0.05, 'sASM/BASIC $\\lambda_{min}$ ratio: ' +
               ', '.join('%.2f' % r for r in rmin), transform=ax[1].transAxes,
               fontsize=9, color='C0')
    fig.suptitle('sASM vs BASIC: BOTH spectral ends (2D, ICC(0))', fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(RES, 'fig_sasm_both_ends.png'), dpi=130); plt.close(fig)
    print('  fig_sasm_both_ends.png'); return out, rmin


def partB(dim=2, M=41, S=4, O=1, K=40, snaps=(2, 8, 20, 40)):
    """Stationary Schwarz: drain the constant global mode; watch info propagate from
    the Dirichlet boundary. Compare full / mixed(1 face) / pure-Neumann."""
    a = A.coef_field('const', dim, M)
    subs = A.box_subdomains(dim, M, S, O)
    bcs = [('full', 'full Dirichlet (source on all sides)'),
           ('mixed', '1 Dirichlet face (source on one side)'),
           ('neumann', 'pure Neumann (NO source)')]
    curves = {}
    fig, axs = plt.subplots(len(bcs), len(snaps), figsize=(3.0 * len(snaps), 2.7 * len(bcs)))
    for i, (bc, title) in enumerate(bcs):
        Amat, _ = A.assemble(dim, M, a, bc=bc, sigma=0.0)
        M1 = A.AdditiveSchwarz(Amat, subs, 'ic0', 'basic')
        # damping theta < 2/lambda_max for a stable stationary iteration (additive
        # Schwarz has lambda_max ~ 4-5 > 1, so undamped Richardson diverges)
        v = np.random.default_rng(0).standard_normal(M ** dim)
        for _ in range(60):
            w = M1._matvec(Amat @ v); v = w / (np.linalg.norm(w) + 1e-300)
        lammax = float(v @ M1._matvec(Amat @ v))
        theta = 0.9 / lammax
        x = np.ones(M ** dim)                       # x0 = constant = the global mode
        nrm = [np.linalg.norm(x)]
        snapsx = {}
        for k in range(1, K + 1):
            r = -(Amat @ x)                         # b=0 -> residual = -A x
            x = x + theta * M1._matvec(r)           # one DAMPED stationary Schwarz sweep
            nrm.append(np.linalg.norm(x))
            if k in snaps:
                snapsx[k] = x.copy()
        curves[bc] = np.array(nrm) / nrm[0]
        vmax = np.abs(np.ones(M ** dim)).max()
        for j, k in enumerate(snaps):
            ax = axs[i, j]
            im = ax.imshow(np.abs(snapsx[k]).reshape(M, M).T, origin='lower',
                           cmap='magma', vmin=0, vmax=1.0)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title('iter %d' % k)
            if j == 0:
                ax.set_ylabel(title, fontsize=9)
    fig.suptitle('Information propagation: |error| draining from the Dirichlet boundary '
                 '(stationary Schwarz, b=0, x0=constant)', fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(RES, 'fig_info_prop_maps.png'), dpi=130); plt.close(fig)
    print('  fig_info_prop_maps.png')

    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    for bc, lab, col in [('full', 'full Dirichlet', 'C2'),
                         ('mixed', '1 Dirichlet face', 'C0'),
                         ('neumann', 'pure Neumann', 'C3')]:
        ax.semilogy(range(len(curves[bc])), curves[bc], 'o-', color=col, label=lab, ms=3)
    ax.set_xlabel('stationary Schwarz sweep'); ax.set_ylabel('$\\|$error$\\|/\\|$error$_0\\|$ (constant mode)')
    ax.set_title('With vs without Dirichlet: the global mode drains only if a source exists')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(RES, 'fig_info_prop_curve.png'), dpi=130); plt.close(fig)
    print('  fig_info_prop_curve.png')
    for bc in ('full', 'mixed', 'neumann'):
        print("  %-8s error after %d sweeps: %.3e" % (bc, K, curves[bc][-1]))
    return curves


if __name__ == '__main__':
    print("=== Q1: sASM effect on BOTH spectral ends ===")
    partA()
    print("\n=== Q2/Q3: information propagation from the Dirichlet boundary ===")
    partB()
