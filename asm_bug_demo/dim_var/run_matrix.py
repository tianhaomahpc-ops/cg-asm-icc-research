"""run_matrix.py -- the dimensional + variable-coefficient experiment matrix.

Experiments
  A  dimensional sweep, constant coefficient  (same M, S across d=1,2,3)
       local in {exact, ic0} x scaling in {basic, sasm} x overlap sweep
  B  IC(0) quality vs connectivity at ~fixed local DOF count  (isolates omega(d))
  C  variable-coefficient sweep  (coef fields x d x overlap, ic0 BASIC + sASM fix)

Outputs CSV under results/ and (if matplotlib present) PNG figures.
Run:  python3 run_matrix.py A|B|C|all
"""
import sys, os, csv, time
import numpy as np
import asm_spectral as M

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
os.makedirs(RES, exist_ok=True)


def write_csv(name, rows):
    if not rows:
        return
    keys = []
    for r in rows:                       # union of keys, preserve first-seen order
        for k in r:
            if k not in keys:
                keys.append(k)
    p = os.path.join(RES, name)
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, restval='')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("  wrote", p)


# ----------------------------------------------------------------------
#  Experiment A : dimensional, constant coefficient
# ----------------------------------------------------------------------
def expA(Ms=(33, 25, 13), S=4, overlaps=(0, 1, 2, 3, 4)):
    """Ms = (M for d=1, M for d=2, M for d=3). Chosen so 3D N stays dense-friendly.
    Same S and overlap-in-gridpoints across dims => fixed #subdomains/axis and
    fixed delta; the local block grows naturally as H^d (the physical reality)."""
    rows = []
    for dim, Md in zip((1, 2, 3), Ms):
        for local in ('exact', 'ic0'):
            for scaling in ('basic', 'sasm'):
                for ov in overlaps:
                    t0 = time.time()
                    r = M.run_cell(dim, Md, S, ov, local, scaling, coef='const')
                    r['sec'] = round(time.time() - t0, 2)
                    rows.append(r)
                    print("A d=%d %-5s %-5s O=%d | iter=%3d Nhat=%2d w=%.3f "
                          "lmax=%7.3f lmin=%.4f kap=%8.1f N=%d (%.1fs)" % (
                              dim, local, scaling, ov, r['iter'], r['Nhat'],
                              r['omega'], r['lam_max'], r['lam_min'], r['kappa'],
                              r['N'], r['sec']))
    write_csv('expA_dimensional_const.csv', rows)
    return rows


# ----------------------------------------------------------------------
#  Experiment B : IC(0) quality vs connectivity at ~fixed total DOFs
# ----------------------------------------------------------------------
def ic0_quality(dim, m, coef='const', contrast=1.0):
    """Build the d-dim operator (one Dirichlet face), measure IC(0) preconditioned
    spectrum of the WHOLE matrix as a stand-in for a local subdomain block."""
    a = M.coef_field(coef, dim, m, contrast=contrast)
    A, b = M.assemble(dim, m, a)
    Ad = A.toarray()
    wA = np.linalg.eigvalsh(Ad)
    wA = wA[wA > 1e-10]
    kapA = wA.max() / wA.min()
    L = M.ic0_factor(A)
    Linv = np.linalg.inv(L)
    Minv = Linv.T @ Linv
    # eigenvalues of Minv A  (symmetric similar S = L_A^T Minv L_A, A=L_A L_A^T)
    LA = np.linalg.cholesky(Ad)
    Sm = LA.T @ Minv @ LA
    w = np.linalg.eigvalsh(0.5 * (Sm + Sm.T))
    w = w[w > 1e-10]
    return dict(dim=dim, m=m, N=A.shape[0], coef=coef, contrast=contrast,
                stencil_nbrs=2 * dim, kappa_A=kapA,
                omega=w.max(), lam_min_prec=w.min(),
                kappa_prec=w.max() / w.min())


def expB():
    """~512 DOFs in every dimension: 1D m=512, 2D m=23 (529), 3D m=8 (512).
    Same problem size, different stencil connectivity 2/4/6."""
    rows = []
    for dim, m in [(1, 512), (2, 23), (3, 8)]:
        r = ic0_quality(dim, m, 'const')
        rows.append(r)
        print("B d=%d nbrs=%d N=%4d | kappa(A)=%9.1f  omega=lam_max(IC0^-1 A)=%.4f"
              "  kappa(IC0^-1 A)=%8.2f" % (
                  dim, r['stencil_nbrs'], r['N'], r['kappa_A'],
                  r['omega'], r['kappa_prec']))
    # also a small contrast scan in each dim to preview variable-coef effect on omega
    for dim, m in [(1, 512), (2, 23), (3, 8)]:
        for c in (1.0, 10.0, 100.0, 1e4):
            r = ic0_quality(dim, m, 'layers', contrast=c)
            r['note'] = 'layers'
            rows.append(r)
            print("B d=%d layers c=%6g | kappa(A)=%10.1f omega=%.4f kappa_prec=%9.2f"
                  % (dim, c, r['kappa_A'], r['omega'], r['kappa_prec']))
    write_csv('expB_ic0_quality.csv', rows)
    return rows


# ----------------------------------------------------------------------
#  Experiment C : variable coefficient
# ----------------------------------------------------------------------
def expC1(Ms=(65, 33, 11), S=4, overlaps=(0, 1, 2, 3),
          coefs=('const', 'smooth', 'layers', 'layers_unaligned'),
          contrast=1e4):
    """Coefficient-field sweep at fixed high contrast: does the overlap anomaly
    (and the sASM fix) survive variable coefficients, across dimensions?"""
    rows = []
    for dim, Md in zip((1, 2, 3), Ms):
        for coef in coefs:
            for scaling in ('basic', 'sasm'):
                for ov in overlaps:
                    c = 1.0 if coef in ('const', 'smooth') else contrast
                    t0 = time.time()
                    r = M.run_cell(dim, Md, S, ov, 'ic0', scaling,
                                   coef=coef, contrast=c)
                    r['sec'] = round(time.time() - t0, 2)
                    rows.append(r)
                    print("C1 d=%d %-16s %-5s O=%d c=%6g | iter=%3d w=%.3f "
                          "kap=%10.1f Nhat=%d" % (
                              dim, coef, scaling, ov, c, r['iter'],
                              r['omega'], r['kappa'], r['Nhat']))
    write_csv('expC1_coef_fields.csv', rows)
    return rows


def expC2(dim=2, M_=33, S=4, ov=2,
          contrasts=(1.0, 1e1, 1e2, 1e3, 1e4),
          coefs=('layers', 'layers_unaligned', 'checker')):
    """Contrast sweep at fixed (dim, overlap): the headline boundary.
    Shows omega (and absolute kappa) blowing up with contrast for UNALIGNED jumps,
    while sASM removes the geometric N-hat over-count (fixes the overlap TREND) but is
    coefficient-blind (absolute kappa still blows up) -> coarse space needed."""
    rows = []
    for coef in coefs:
        for scaling in ('basic', 'sasm'):
            for c in contrasts:
                t0 = time.time()
                r = M.run_cell(dim, M_, S, ov, 'ic0', scaling, coef=coef, contrast=c)
                r['sec'] = round(time.time() - t0, 2)
                rows.append(r)
                print("C2 d=%d %-16s %-5s c=%7g | iter=%3d w=%6.3f "
                      "lam_max=%7.3f lam_min=%.2e kap=%11.1f" % (
                          dim, coef, scaling, c, r['iter'], r['omega'],
                          r['lam_max'], r['lam_min'], r['kappa']))
    write_csv('expC2_contrast_sweep_d%d.csv' % dim, rows)
    return rows


def expC():
    print("--- C1 : coefficient fields at contrast 1e4 ---")
    expC1()
    print("\n--- C2 : contrast sweep, 2D ---")
    expC2(dim=2, M_=33, ov=2)
    print("\n--- C2 : contrast sweep, 3D ---")
    expC2(dim=3, M_=11, ov=2)


# ----------------------------------------------------------------------
if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'A'
    if which in ('A', 'all'):
        print("\n##### Experiment A : dimensional, constant coefficient #####")
        expA()
    if which in ('B', 'all'):
        print("\n##### Experiment B : IC(0) quality vs connectivity #####")
        expB()
    if which in ('C', 'all'):
        print("\n##### Experiment C : variable coefficient #####")
        expC()
