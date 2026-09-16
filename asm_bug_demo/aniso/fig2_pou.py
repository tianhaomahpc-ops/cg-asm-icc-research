"""Fig II: do the two NEW partition-of-unity families beat the multiplicity
scaling, and does lambda_min finally pick up the 1/(1+H/delta) scaling?

  flat   w=1/sqrt(m)            (sASM, what the user runs today)
  graded w=q^depth              (scheme 7 -pugrade, the current best weight)
  ramp   w=1-depth/(delta+1)    NEW-A  |grad chi| ~ 1/delta
  harm   sigma-harmonic ramp    NEW-B  minimiser of int sigma |grad chi|^2
"""
import anisopu as ap, numpy as np, csv, time

nx, P = 120, 6
OVER = [1, 2, 3, 4, 5, 6, 8]
CASES = [(1, 45.0), (10, 45.0), (100, 45.0), (1, 0.0), (100, 0.0)]
KINDS = ["basic", "flat", "graded", "ramp", "harm"]
rows = []
t0 = time.time()
for r, ang in CASES:
    sig = ap.sigma_tensor(r, ang)
    A, b, coords, free, n1 = ap.build_fem(nx, sig)
    own = ap.box_partition(coords, P, P); nsub = P * P
    print("\n### contrast r=%d  fiber=%.0fdeg   N=%d  nsub=%d" % (r, ang, A.shape[0], nsub))
    print("  O |" + "".join("%18s" % k for k in KINDS) + "   exactChol")
    for O in OVER:
        sets = ap.overlap_sets(A, own, nsub, O)
        Alocs = [A[i][:, i].tocsr() for i, _ in sets]
        line, rec = [], dict(ratio=r, angle=ang, O=O)
        for k in KINDS:
            w = ap.build_weights(k, sets, Alocs, A.shape[0], O, q=0.8)
            M = ap.ASM(A, sets, w)
            _, it, (lo, hi) = ap.pcg(A, b, M, rtol=1e-6, maxit=4000)
            rec[f"it_{k}"] = it; rec[f"lmin_{k}"] = lo; rec[f"lmax_{k}"] = hi
            line.append("%5d (%.1e)" % (it, lo))
        Me = ap.ASM(A, sets, None, exact=True)
        _, ite, _ = ap.pcg(A, b, Me, rtol=1e-6, maxit=4000)
        rec["it_exact"] = ite
        rows.append(rec)
        print("  %d |" % O + "".join("%18s" % s for s in line) + "   %6d" % ite, flush=True)
with open("fig2_pou.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("\nwrote fig2_pou.csv  (%.1f s)" % (time.time() - t0))
