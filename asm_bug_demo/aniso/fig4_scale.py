"""Fig IV: weak scaling.  Subdomain size held fixed (nx = 20P), number of
subdomains swept.  One-level methods must grow like sqrt(#subdomains);
a coarse space must flatten them -- and the question is whether the CLASSICAL
1/m coarse basis still flattens them once sigma is a tensor."""
import anisopu as ap, numpy as np, csv

CFG = [("one-level sASM      (flat / none)", "flat", None),
       ("one-level harm PoU  (harm / none)", "harm", None),
       ("two-level classical (flat / flat)", "flat", "flat"),
       ("two-level reuse     (harm / harm)", "harm", "harm")]
rows = []
for r, ang in [(1, 45.0), (100, 45.0)]:
    print("\n### contrast r=%d  fiber=%.0f deg   (O=2, subdomain size fixed at 20x20 cells)" % (r, ang))
    Ps = [2, 3, 4, 5, 6]
    res = {c[0]: [] for c in CFG}
    for P in Ps:
        nx = 20 * P
        sig = ap.sigma_tensor(r, ang)
        A, b, coords, free, n1 = ap.build_fem(nx, sig)
        own = ap.box_partition(coords, P, P); nsub = P * P
        sets = ap.overlap_sets(A, own, nsub, 2)
        Alocs = [A[i][:, i].tocsr() for i, _ in sets]
        for name, fine, cb in CFG:
            w = ap.build_weights(fine, sets, Alocs, A.shape[0], 2)
            M = ap.ASM(A, sets, w)
            if cb: M.set_coarse(A, ap.coarse_basis(cb, sets, Alocs, A.shape[0], 2))
            _, it, (lo, hi) = ap.pcg(A, b, M, rtol=1e-6, maxit=6000)
            res[name].append(it)
            rows.append(dict(ratio=r, angle=ang, P=P, nsub=nsub, N=A.shape[0],
                             cfg=name, it=it, lmin=lo, lmax=hi))
    print("  #subdomains:     " + "".join("%8d" % (P * P) for P in Ps) + "     growth")
    for name, _, _ in CFG:
        v = res[name]
        print("  %-34s" % name + "".join("%8d" % x for x in v)
              + "     %.2fx" % (v[-1] / v[0]))
with open("fig4_scale.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("\nwrote fig4_scale.csv")
