"""parse_fem.py -- parse schwarz_fem [RESULT] log -> CSV + figures for the PPT.

tag format: case<a|b|c|d>_<d>D_<box|metis>
Produces results/fem.csv and a set of themed PNGs used by build_ppt.py.
"""
import os, re, csv, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
KV = re.compile(r'(\w+)=([^\s]+)')

CASE_TITLE = {
    'a': 'a: full-Dirichlet, isotropic',
    'b': 'b: full-Dirichlet, anisotropic',
    'c': 'c: Dirichlet+Neumann, isotropic',
    'd': 'd: Dirichlet+Neumann, anisotropic',
}


def parse():
    rows = []
    for line in open(os.path.join(RES, 'fem.log')):
        if '[RESULT]' not in line:
            continue
        d = {}
        for k, v in KV.findall(line.split('[RESULT]', 1)[1]):
            try:
                d[k] = float(v) if re.search(r'[.eE]', v) and k != 'tag' else (
                    int(v) if v.lstrip('-').isdigit() else v)
            except ValueError:
                d[k] = v
        m = re.match(r'case([a-d])_(\d)D_(box|metis)', str(d.get('tag', '')))
        if m:
            d['case'], d['dim'], d['part'] = m.group(1), int(m.group(2)), m.group(3)
        rows.append(d)
    return rows


def sel(rows, **c):
    out = [r for r in rows if all(r.get(k) == v for k, v in c.items())]
    return sorted(out, key=lambda r: r.get('overlap', 0))


def write_csv(rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(RES, 'fem.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, restval=''); w.writeheader(); w.writerows(rows)


# ---- Theme 1: ASM vs sASM iterations, 4 cases (use 2D, both partitions) ----
def fig_asm_vs_sasm(rows, metric='iter', dim=2, fname='fig_iter_2d', ylabel='CG iterations',
                    log=False):
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    for ax, case in zip(axs.ravel(), 'abcd'):
        any_data = False
        for part, cstyle in [('box', 'C0'), ('metis', 'C1')]:
            b = sel(rows, case=case, dim=dim, part=part, scheme=0)
            s = sel(rows, case=case, dim=dim, part=part, scheme=3)
            if b:
                ax.plot([r['overlap'] for r in b], [r[metric] for r in b], 'o-',
                        color=cstyle, label='%s ASM' % part); any_data = True
            if s:
                ax.plot([r['overlap'] for r in s], [r[metric] for r in s], 's--',
                        color=cstyle, alpha=0.65, label='%s sASM' % part)
        ax.set_title('case %s' % CASE_TITLE[case], fontsize=10)
        ax.set_xlabel('overlap O'); ax.set_ylabel(ylabel)
        if log:
            ax.set_yscale('log')
        if any_data:
            ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle('%dD unstructured P1 FEM: ASM vs sASM  (%s)' % (dim, ylabel), fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(RES, fname + '.png'), dpi=130)
    plt.close(fig); print('  ', fname)


# ---- Theme: dimensional sweep, given case ----
def fig_dimensional(rows, case='a', part='box', fname=None):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    colors = {1: 'C0', 2: 'C1', 3: 'C2'}
    for d in (1, 2, 3):
        b = sel(rows, case=case, dim=d, part=part, scheme=0)
        s = sel(rows, case=case, dim=d, part=part, scheme=3)
        if b:
            ax[0].plot([r['overlap'] for r in b], [r['iter'] for r in b], 'o-',
                       color=colors[d], label='%dD ASM' % d)
        if s:
            ax[0].plot([r['overlap'] for r in s], [r['iter'] for r in s], 's--',
                       color=colors[d], alpha=0.65, label='%dD sASM' % d)
        if b:
            ax[1].plot([r['overlap'] for r in b], [r['kappa'] for r in b], 'o-',
                       color=colors[d], label='%dD ASM $\\kappa$' % d)
            ax[1].plot([r['overlap'] for r in s], [r['kappa'] for r in s], 's--',
                       color=colors[d], alpha=0.65)
    ax[0].set_xlabel('overlap O'); ax[0].set_ylabel('CG iterations')
    ax[0].set_title('(a) iterations vs overlap'); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].set_xlabel('overlap O'); ax[1].set_ylabel('$\\kappa(M^{-1}A)$'); ax[1].set_yscale('log')
    ax[1].set_title('(b) condition number'); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.suptitle('case %s (%s partition): 1D/2D/3D' % (case, part), fontsize=12)
    fig.tight_layout()
    fn = fname or ('fig_dim_case' + case)
    fig.savefig(os.path.join(RES, fn + '.png'), dpi=130); plt.close(fig); print('  ', fn)


# ---- Theme: BC effect (a vs c) and anisotropy (a vs b, c vs d) ----
def fig_compare(rows, pairs, title, fname, dim=2, part='box'):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for (case, col, lab) in pairs:
        b = sel(rows, case=case, dim=dim, part=part, scheme=0)
        s = sel(rows, case=case, dim=dim, part=part, scheme=3)
        if b:
            ax[0].plot([r['overlap'] for r in b], [r['iter'] for r in b], 'o-', color=col,
                       label='%s ASM' % lab)
            ax[0].plot([r['overlap'] for r in s], [r['iter'] for r in s], 's--', color=col,
                       alpha=0.6, label='%s sASM' % lab)
            ax[1].plot([r['overlap'] for r in b], [r['lam_min'] for r in b], 'o-', color=col,
                       label='%s ASM' % lab)
            ax[1].plot([r['overlap'] for r in s], [r['lam_min'] for r in s], 's--', color=col,
                       alpha=0.6)
    ax[0].set_xlabel('overlap O'); ax[0].set_ylabel('CG iterations')
    ax[0].set_title('(a) iterations'); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].set_xlabel('overlap O'); ax[1].set_ylabel('$\\lambda_{min}(M^{-1}A)$'); ax[1].set_yscale('log')
    ax[1].set_title('(b) $\\lambda_{min}$ = Dirichlet-info propagation'); ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.suptitle(title, fontsize=12); fig.tight_layout()
    fig.savefig(os.path.join(RES, fname + '.png'), dpi=130); plt.close(fig); print('  ', fname)


# ---- Theme: timing ----
def fig_timing(rows, dim=2, part='box', fname='fig_timing_2d'):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for case, col in zip('abcd', ['C0', 'C1', 'C2', 'C3']):
        for sc, mk, a, lab in [(0, 'o-', 1.0, 'case %s' % case), (3, 's--', 0.6, None)]:
            r = sel(rows, case=case, dim=dim, part=part, scheme=sc)
            if r:
                ax[0].plot([x['overlap'] for x in r], [x['t_solve'] for x in r], mk, color=col,
                           alpha=a, label=lab)
                ax[1].plot([x['overlap'] for x in r], [x['t_setup'] for x in r], mk, color=col,
                           alpha=a)
    ax[0].set_xlabel('overlap O'); ax[0].set_ylabel('solve time (s)')
    ax[0].set_title('(a) solve time  (solid=ASM, dashed=sASM)')
    ax[0].legend(fontsize=8, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.16))
    ax[0].grid(alpha=0.3)
    ax[1].set_xlabel('overlap O'); ax[1].set_ylabel('setup time (s)')
    ax[1].set_title('(b) setup time'); ax[1].grid(alpha=0.3)
    fig.suptitle('%dD (%s): wall-clock time, ASM vs sASM' % (dim, part), fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, fname + '.png'), dpi=130, bbox_inches='tight')
    plt.close(fig); print('  ', fname)


# ---- Theme: Ritz spectral distribution ----
def fig_spectrum():
    files = sorted(glob.glob(os.path.join(RES, 'eig_*_box_s*_O2.txt')))
    by_case = {}
    for f in files:
        m = re.search(r'eig_case([a-d])_(\d)D_box_s(\d)_O2', f)
        if m:
            by_case.setdefault((m.group(1), m.group(2)), {})[int(m.group(3))] = f
    if not by_case:
        return
    keys = sorted(by_case)
    n = len(keys)
    fig, axs = plt.subplots(1, n, figsize=(4 * n, 3.8), squeeze=False)
    for ax, k in zip(axs[0], keys):
        for sc, col, lab in [(0, 'C0', 'ASM'), (3, 'C3', 'sASM')]:
            if sc in by_case[k]:
                ev = np.loadtxt(by_case[k][sc])
                ev = ev[ev > 0]
                ax.hist(ev, bins=30, alpha=0.5, color=col, label=lab)
        ax.set_title('case %s %sD, O=2' % k); ax.set_xlabel('Ritz eigenvalue of $M^{-1}A$')
        ax.set_ylabel('count'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle('Preconditioned spectral distribution (CG Ritz values)', fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(RES, 'fig_spectrum.png'), dpi=130)
    plt.close(fig); print('   fig_spectrum')


if __name__ == '__main__':
    rows = parse()
    write_csv(rows)
    print('parsed %d rows' % len(rows))
    fig_asm_vs_sasm(rows, 'iter', 2, 'fig_iter_2d', 'CG iterations')
    fig_asm_vs_sasm(rows, 'kappa', 2, 'fig_kappa_2d', '$\\kappa(M^{-1}A)$', log=True)
    fig_asm_vs_sasm(rows, 'iter', 3, 'fig_iter_3d', 'CG iterations')
    fig_dimensional(rows, 'a', 'box', 'fig_dim_casea')
    fig_dimensional(rows, 'c', 'box', 'fig_dim_casec')
    fig_compare(rows, [('a', 'C0', 'a iso/fullD'), ('c', 'C3', 'c iso/mixed')],
                'BC effect: full-Dirichlet (a) vs Dirichlet+Neumann (c), 2D', 'fig_bc_effect')
    fig_compare(rows, [('a', 'C0', 'a iso'), ('b', 'C3', 'b aniso')],
                'Anisotropy effect (full-Dirichlet): iso (a) vs aniso (b), 2D', 'fig_aniso_fullD')
    fig_compare(rows, [('c', 'C0', 'c iso'), ('d', 'C3', 'd aniso')],
                'Anisotropy effect (mixed BC): iso (c) vs aniso (d), 2D', 'fig_aniso_mixed')
    fig_timing(rows, 2, 'box', 'fig_timing_2d')
    fig_spectrum()
