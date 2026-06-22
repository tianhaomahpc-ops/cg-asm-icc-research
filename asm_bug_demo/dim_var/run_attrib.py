"""run_attrib.py -- the attribution study: which factor makes ASM/sASM harder?

Decomposes the difficulty into the two channels along the boundary/shift axis and
the dimension axis, on simple small geometry mirroring the cardiac systems:
  - lambda_max ~= omega*Nhat : the geometric over-count channel (dimension, local solve)
  - lambda_min               : the information-transfer channel (boundary, mass shift sigma)

Boundary/shift axis (ordered most-anchored -> least): full-Dirichlet, k Dirichlet
faces, 1 Dirichlet face (mixed), pure Neumann with shift sigma=10,1,0.1 (-> Sys2).
  Sys1 ~ pure Neumann, large sigma (=1/dt, mass-dominated, easy)
  Sys2 ~ pure Neumann, sigma->0     (singular, hardest)
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import asm_spectral as M

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RES, exist_ok=True)

# (key, bc, k, sigma, label, anchor-order x)
AXIS = [
    ('full',    'full',    0, 0.0,  'full-Dir',        0),
    ('kfaces',  'kfaces',  None, 0.0,'k=d Dir faces',   1),
    ('mixed',   'mixed',   1, 0.0,  '1 Dir (mixed)',    2),
    ('neu_s10', 'neumann', 0, 10.0, 'pureNeu sig=10\n(~Sys1)', 3),
    ('neu_s1',  'neumann', 0, 1.0,  'pureNeu sig=1',    4),
    ('neu_s.1', 'neumann', 0, 0.1,  'pureNeu sig=0.1\n(->Sys2)', 5),
]


def run(dims=(2, 3), Ms=(25, 13), S=4, overlaps=(0, 1, 2, 3)):
    rows = []
    for dim, Md in zip(dims, Ms):
        for key, bc, k, sig, lab, order in AXIS:
            kk = dim if k is None else (k or 1)
            for sch in ('basic', 'sasm'):
                for ov in overlaps:
                    r = M.run_cell(dim, Md, S, ov, 'ic0', sch, coef='const',
                                   bc=bc, k=kk, sigma=sig)
                    r['axis'] = key; r['axis_label'] = lab; r['order'] = order
                    rows.append(r)
                    print("d=%d %-10s %-5s O=%d | it=%3d Nhat=%d w=%.3f "
                          "lmax=%6.2f lmin=%.2e kap=%9.1f" % (
                              dim, key, sch, ov, r['iter'], r['Nhat'], r['omega'],
                              r['lam_max'], r['lam_min'], r['kappa']))
    keys = []
    for r in rows:
        for kk in r:
            if kk not in keys:
                keys.append(kk)
    with open(os.path.join(RES, 'attrib.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, restval=''); w.writeheader(); w.writerows(rows)
    return rows


def sel(rows, **c):
    return sorted([r for r in rows if all(r.get(k) == v for k, v in c.items())],
                  key=lambda r: r['order'])


def fig_channels(rows, dim=2):
    """Headline: along the boundary/shift axis, lam_max is FLAT, lam_min COLLAPSES."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    O = 1
    b = sel(rows, dim=dim, scaling='basic', overlap=O)
    s = sel(rows, dim=dim, scaling='sasm', overlap=O)
    x = [r['order'] for r in b]; labs = [r['axis_label'] for r in b]
    # panel a: lam_max (both schemes) -- flat
    ax[0].plot(x, [r['lam_max'] for r in b], 'o-', color='C3', label='ASM  $\\lambda_{max}$')
    ax[0].plot(x, [r['lam_max'] for r in s], 's--', color='C0', label='sASM $\\lambda_{max}$')
    ax[0].plot(x, [r['lam_min'] for r in b], 'o-', color='C1', label='ASM  $\\lambda_{min}$')
    ax[0].plot(x, [r['lam_min'] for r in s], 's--', color='C2', label='sASM $\\lambda_{min}$')
    ax[0].set_yscale('log'); ax[0].set_xticks(x); ax[0].set_xticklabels(labs, fontsize=7.5)
    ax[0].set_title('(a) %dD: $\\lambda_{max}$ flat (over-count) vs $\\lambda_{min}$ collapses (info-transfer)' % dim)
    ax[0].set_ylabel('eigenvalue of $M^{-1}A$'); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    # panel b: iterations
    for sch, mk, col in [('basic', 'o-', 'C3'), ('sasm', 's--', 'C0')]:
        rr = sel(rows, dim=dim, scaling=sch, overlap=O)
        ax[1].plot([r['order'] for r in rr], [r['iter'] for r in rr], mk, color=col,
                   label='ASM' if sch == 'basic' else 'sASM')
    ax[1].set_xticks(x); ax[1].set_xticklabels(labs, fontsize=7.5)
    ax[1].set_title('(b) %dD: CG iterations track the $\\lambda_{min}$ collapse' % dim)
    ax[1].set_ylabel('CG iterations'); ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)
    fig.suptitle('Attribution: the boundary / shift axis moves $\\lambda_{min}$, NOT $\\lambda_{max}$', fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(RES, 'fig_attrib_channels_%dd.png' % dim), dpi=130)
    plt.close(fig); print('  fig_attrib_channels_%dd.png' % dim)


def fig_sigma(rows):
    """pure-Neumann sigma sweep: kappa ~ 1/sigma (approach to the singular Sys2)."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for dim, col in [(2, 'C0'), (3, 'C1')]:
        pts = [r for r in rows if r['dim'] == dim and r['scaling'] == 'basic'
               and r['overlap'] == 1 and r['axis'].startswith('neu')]
        sigs = {'neu_s10': 10.0, 'neu_s1': 1.0, 'neu_s.1': 0.1}
        pts = sorted(pts, key=lambda r: -sigs[r['axis']])
        xx = [sigs[r['axis']] for r in pts]
        ax[0].plot(xx, [r['kappa'] for r in pts], 'o-', color=col, label='%dD' % dim)
        ax[1].plot(xx, [r['lam_min'] for r in pts], 'o-', color=col, label='%dD' % dim)
    for a in ax:
        a.set_xscale('log'); a.invert_xaxis(); a.set_xlabel('shift $\\sigma$  (large=Sys1, $\\to0$=Sys2)')
        a.grid(alpha=0.3); a.legend(fontsize=9)
    ax[0].set_yscale('log'); ax[0].set_ylabel('$\\kappa(M^{-1}A)$'); ax[0].set_title('(a) $\\kappa \\propto 1/\\sigma$')
    ax[1].set_yscale('log'); ax[1].set_ylabel('$\\lambda_{min}$'); ax[1].set_title('(b) $\\lambda_{min}\\propto\\sigma$ (constant mode unanchored)')
    fig.suptitle('Pure-Neumann shift sweep: approaching the singular Sys2', fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(RES, 'fig_attrib_sigma.png'), dpi=130)
    plt.close(fig); print('  fig_attrib_sigma.png')


if __name__ == '__main__':
    rows = run()
    fig_channels(rows, 2); fig_channels(rows, 3); fig_sigma(rows)
