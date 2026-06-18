"""parse_petsc.py -- parse schwarz_lab [RESULT] logs into CSV + publication figures.

Reads results/petsc_A.log (dimensional, const coef) and results/petsc_C.log
(variable coefficient), writes results/petsc_A.csv, petsc_C.csv and PNG figures.
"""
import os, re, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')

KV = re.compile(r'(\w+)=([^\s]+)')
NUM = re.compile(r'^-?\d')


def parse_log(path):
    rows = []
    if not os.path.exists(path):
        return rows
    for line in open(path):
        if '[RESULT]' not in line:
            continue
        d = {}
        for k, v in KV.findall(line.split('[RESULT]', 1)[1]):
            if NUM.match(v):
                try:
                    d[k] = float(v) if ('.' in v or 'e' in v or 'E' in v) else int(v)
                except ValueError:
                    d[k] = v
            else:
                d[k] = v
        if d:
            rows.append(d)
    return rows


def to_csv(rows, name):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(RES, name), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, restval='')
        w.writeheader()
        w.writerows(rows)


def sel(rows, **cond):
    out = []
    for r in rows:
        if all(r.get(k) == v for k, v in cond.items()):
            out.append(r)
    return sorted(out, key=lambda r: r.get('overlap', 0))


# ----------------------------------------------------------------------
def fig_iter_vs_overlap(rows):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {1: 'C0', 2: 'C1', 3: 'C2'}
    for d in (1, 2, 3):
        b = sel(rows, dim=d, scheme=0, exact=0)
        s = sel(rows, dim=d, scheme=3, exact=0)
        if b:
            ax[0].plot([r['overlap'] for r in b], [r['iter'] for r in b],
                       'o-', color=colors[d], label='%dD BASIC' % d)
        if s:
            ax[0].plot([r['overlap'] for r in s], [r['iter'] for r in s],
                       's--', color=colors[d], alpha=0.7, label='%dD sASM' % d)
    ax[0].set_xlabel('overlap O (grid points)'); ax[0].set_ylabel('CG iterations')
    ax[0].set_title('(a) CG iterations vs overlap  (ICC(0) local)')
    ax[0].legend(fontsize=8, ncol=2); ax[0].grid(alpha=0.3)

    for d in (1, 2, 3):
        b = sel(rows, dim=d, scheme=0, exact=0)
        if b:
            ax[1].plot([r['overlap'] for r in b], [r['lam_max'] for r in b],
                       'o-', color=colors[d], label='%dD BASIC  $\\lambda_{max}$' % d)
        s = sel(rows, dim=d, scheme=3, exact=0)
        if s:
            ax[1].plot([r['overlap'] for r in s], [r['lam_max'] for r in s],
                       's--', color=colors[d], alpha=0.7)
    ax[1].set_xlabel('overlap O'); ax[1].set_ylabel('$\\lambda_{max}(M^{-1}A)$')
    ax[1].set_title('(b) $\\lambda_{max}\\approx\\omega\\cdot\\hat N$ (BASIC) vs flat $\\approx\\omega$ (sASM)')
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, 'fig1_iter_lammax_vs_overlap.png'), dpi=130)
    print('  fig1_iter_lammax_vs_overlap.png')


def fig_nhat_omega(rows):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {1: 'C0', 2: 'C1', 3: 'C2'}
    for d in (1, 2, 3):
        b = sel(rows, dim=d, scheme=0, exact=1)   # exact: lam_max == Nhat
        if b:
            ax[0].plot([r['overlap'] for r in b], [r['Nhat'] for r in b],
                       'o-', color=colors[d], label='%dD  $\\hat N$' % d)
    ax[0].set_xlabel('overlap O'); ax[0].set_ylabel('$\\hat N$ = max multiplicity')
    ax[0].set_title('(a) geometric over-count $\\hat N$ grows with overlap & dimension')
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)

    # omega vs dimension (BASIC ICC0, take a representative overlap=1 value)
    ds, ws = [], []
    for d in (1, 2, 3):
        b = sel(rows, dim=d, scheme=0, exact=0)
        cand = [r for r in b if r.get('overlap') == 1] or b
        if cand:
            ds.append(d); ws.append(cand[0]['omega'])
    ax[1].bar([str(d) + 'D' for d in ds], ws, color=['C0', 'C1', 'C2'][:len(ds)])
    for i, w in enumerate(ws):
        ax[1].text(i, w + 0.01, '%.3f' % w, ha='center', fontsize=10)
    ax[1].axhline(1.0, color='k', ls=':', lw=1)
    ax[1].set_ylabel('$\\omega=\\lambda_{max}(M_i^{-1}A_i)$ (ICC(0))')
    ax[1].set_title('(b) $\\omega$ SATURATES with dimension (the corrected driver)')
    ax[1].set_ylim(0.9, max(ws) * 1.15 if ws else 1.5); ax[1].grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(RES, 'fig2_nhat_omega.png'), dpi=130)
    print('  fig2_nhat_omega.png')


def fig_lambda_race(rows):
    """3D: lam_max and lam_min vs overlap, BASIC vs sASM -> why iters rise."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for sc, name, mk in [(0, 'BASIC', 'o-'), (3, 'sASM', 's--')]:
        b = sel(rows, dim=3, scheme=sc, exact=0)
        if not b:
            continue
        ax[0].plot([r['overlap'] for r in b], [r['lam_max'] for r in b], mk,
                   label='%s $\\lambda_{max}$' % name)
        ax[0].plot([r['overlap'] for r in b], [r['lam_min'] for r in b], mk,
                   alpha=0.5, label='%s $\\lambda_{min}$' % name)
        ax[1].plot([r['overlap'] for r in b], [r['kappa'] for r in b], mk,
                   label='%s $\\kappa$' % name)
    ax[0].set_yscale('log'); ax[0].set_xlabel('overlap O'); ax[0].set_ylabel('eigenvalue')
    ax[0].set_title('(a) 3D spectrum race: $\\lambda_{max}$ vs $\\lambda_{min}$')
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].set_xlabel('overlap O'); ax[1].set_ylabel('$\\kappa(M^{-1}A)$')
    ax[1].set_title('(b) 3D condition number: BASIC rises (anomaly) vs sASM falls')
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, 'fig3_lambda_race_3d.png'), dpi=130)
    print('  fig3_lambda_race_3d.png')


def _contrast_rows(rows_c, dim, coef, sc):
    """C2 contrast-sweep rows: overlap==2, multiple rho, given dim/coef/scheme."""
    r = [x for x in rows_c if x.get('dim') == dim and x.get('coef') == coef
         and x.get('scheme') == sc and x.get('overlap') == 2 and 'rho' in x]
    # keep the contrast block (>=4 distinct rho); dedupe by rho keeping last
    by = {}
    for x in r:
        by[float(x['rho'])] = x
    return [by[k] for k in sorted(by)]


def fig_contrast(rows_c):
    """C2: kappa & lam_min vs contrast (2D, O=2): contrast hits lam_min, not omega."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    styles = {'layers': ('C0', 'aligned slab'),
              'checker': ('C3', 'checkerboard (unaligned)')}
    for coef, (col, lab) in styles.items():
        for sc, mk in [(0, 'o-'), (3, 's--')]:
            r = _contrast_rows(rows_c, 2, coef, sc)
            if len(r) < 3:
                continue
            tag = 'BASIC' if sc == 0 else 'sASM'
            ax[0].plot([x['rho'] for x in r], [x['kappa'] for x in r], mk, color=col,
                       alpha=1.0 if sc == 0 else 0.55, label='%s %s' % (lab, tag))
            ax[1].plot([x['rho'] for x in r], [x['lam_min'] for x in r], mk, color=col,
                       alpha=1.0 if sc == 0 else 0.55, label='%s %s' % (lab, tag))
    for a in ax:
        a.set_xscale('log'); a.set_yscale('log')
        a.set_xlabel('coefficient contrast $\\rho$'); a.grid(alpha=0.3)
    ax[0].set_ylabel('$\\kappa(M^{-1}A)$')
    ax[0].set_title('(a) 2D, O=2: $\\kappa\\propto\\rho$; sASM cannot fix contrast')
    ax[1].set_ylabel('$\\lambda_{min}(M^{-1}A)$')
    ax[1].set_title('(b) contrast collapses $\\lambda_{min}\\propto1/\\rho$ ($\\omega$ stays $\\approx$const)')
    ax[0].legend(fontsize=7); ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, 'fig4_contrast_2d.png'), dpi=130)
    print('  fig4_contrast_2d.png')


def fig_boundary(rows_c):
    """C1: iter vs overlap at rho=1e4, BASIC(solid) vs sASM(dashed), per coef field.
    The headline: sASM fixes the trend for const/aligned, FAILS for unaligned hi-contrast."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    fields = [('const', '1', 'C0', 'const'),
              ('layers', '10000', 'C1', 'aligned slab $\\rho=10^4$'),
              ('checker', '10000', 'C3', 'checkerboard $\\rho=10^4$')]
    for di, (dim, nx, title) in enumerate([(2, 49, '(a) 2D'), (3, 25, '(b) 3D')]):
        for coef, rho, col, lab in fields:
            for sc, mk, a in [(0, 'o-', 1.0), (3, 's--', 0.6)]:
                r = [x for x in rows_c if x.get('dim') == dim and x.get('coef') == coef
                     and x.get('nx') == nx and abs(float(x.get('rho', -9)) - float(rho)) < 1e-6
                     and x.get('scheme') == sc]
                r = sorted(r, key=lambda x: x['overlap'])
                # drop duplicate overlaps (C2 block overlap=2) keeping C1 sweep: need full 0..4
                seen = {}
                for x in r:
                    seen.setdefault(x['overlap'], x)
                r = [seen[k] for k in sorted(seen)]
                if len(r) < 4:
                    continue
                tag = 'BASIC' if sc == 0 else 'sASM'
                ax[di].plot([x['overlap'] for x in r], [x['iter'] for x in r], mk,
                            color=col, alpha=a, label='%s %s' % (lab, tag))
        ax[di].set_xlabel('overlap O'); ax[di].set_ylabel('CG iterations')
        ax[di].set_title('%s: sASM fixes const/aligned, FAILS unaligned hi-contrast' % title)
        ax[di].legend(fontsize=7, ncol=1); ax[di].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, 'fig5_sasm_boundary.png'), dpi=130)
    print('  fig5_sasm_boundary.png')


if __name__ == '__main__':
    A = parse_log(os.path.join(RES, 'petsc_A.log'))
    C = parse_log(os.path.join(RES, 'petsc_C.log'))
    to_csv(A, 'petsc_A.csv'); to_csv(C, 'petsc_C.csv')
    print('parsed A=%d rows, C=%d rows' % (len(A), len(C)))
    if A:
        fig_iter_vs_overlap(A); fig_nhat_omega(A); fig_lambda_race(A)
    if C:
        fig_contrast(C); fig_boundary(C)
