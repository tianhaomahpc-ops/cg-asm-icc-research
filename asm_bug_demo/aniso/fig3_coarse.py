"""Fig III: coarse-basis REUSE.  The same chi_i that weights the fine level is
handed to the Nicolaides coarse space for free.  Compared against the classical
1/m_k coarse basis (twolevel.c:99) and against no coarse space at all."""
import anisopu as ap, numpy as np, csv, time

nx, P, O = 120, 6, 3
CASES = [(1, 45.0), (10, 45.0), (100, 45.0), (100, 0.0)]
rows = []
for r, ang in CASES:
    sig = ap.sigma_tensor(r, ang)
    A, b, coords, free, n1 = ap.build_fem(nx, sig)
    own = ap.box_partition(coords, P, P); nsub = P * P
    sets = ap.overlap_sets(A, own, nsub, O)
    Alocs = [A[i][:, i].tocsr() for i, _ in sets]
    print("\n### r=%d fiber=%.0fdeg  N=%d  nsub=%d  O=%d" % (r, ang, A.shape[0], nsub, O))
    print("  fine PoU  | coarse basis | iter | lambda_min  | lambda_max | kappa")
    for fine in ["flat", "harm"]:
        w = ap.build_weights(fine, sets, Alocs, A.shape[0], O)
        for cb in [None, "flat", "harm"]:
            M = ap.ASM(A, sets, w)
            if cb is not None:
                M.set_coarse(A, ap.coarse_basis(cb, sets, Alocs, A.shape[0], O))
            _, it, (lo, hi) = ap.pcg(A, b, M, rtol=1e-6, maxit=4000)
            print("  %-9s | %-12s | %4d | %.4e | %8.3f   | %7.0f"
                  % (fine, cb or "(none)", it, lo, hi, hi / lo), flush=True)
            rows.append(dict(ratio=r, angle=ang, O=O, fine=fine, coarse=cb or "none",
                             it=it, lmin=lo, lmax=hi))
with open("fig3_coarse.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("\nwrote fig3_coarse.csv")
