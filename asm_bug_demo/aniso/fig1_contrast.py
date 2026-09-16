"""Fig I: does the overlap anomaly track the conductivity contrast, and does it
track omega = lambda_max(M_i^-1 A_i) -- the only channel sigma has into the
omega * Nhat product?"""
import anisopu as ap, numpy as np, csv, sys, time

nx, P = 120, 6
RATIOS = [1, 2, 5, 10, 20, 50, 100]
ANGLES = [0.0, 45.0]
OVER   = [0, 1, 2, 3, 4, 5]
rows = []
t0 = time.time()
for ang in ANGLES:
    for r in RATIOS:
        sig = ap.sigma_tensor(r, ang)
        A, b, coords, free, n1 = ap.build_fem(nx, sig)
        own = ap.box_partition(coords, P, P); nsub = P * P
        for O in OVER:
            sets = ap.overlap_sets(A, own, nsub, O)
            Alocs = [A[i][:, i].tocsr() for i, _ in sets]
            mult = np.zeros(A.shape[0])
            for i, _ in sets: mult[i] += 1
            Nhat = int(mult.max())
            om = [ap.local_omega(Ai)[0] for Ai in Alocs]
            omega = float(np.nanmax(om))
            rec = dict(angle=ang, ratio=r, O=O, Nhat=Nhat, omega=round(omega, 4))
            for kind, ex in [("basic", False), ("flat", False), ("basic", True)]:
                w = ap.build_weights(kind, sets, Alocs, A.shape[0], O)
                M = ap.ASM(A, sets, w, exact=ex)
                _, it, (lo, hi) = ap.pcg(A, b, M, rtol=1e-6, maxit=4000)
                tag = "exact" if ex else kind
                rec[f"it_{tag}"] = it
                rec[f"lmin_{tag}"] = float(lo); rec[f"lmax_{tag}"] = float(hi)
            rows.append(rec)
            print("ang=%2.0f r=%3d O=%d | Nhat=%d om=%.3f | BASIC %4d  sASM %4d  exact %4d "
                  "| lmax_B=%.2f lmax_s=%.2f lmin_s=%.2e" %
                  (ang, r, O, Nhat, omega, rec["it_basic"], rec["it_flat"], rec["it_exact"],
                   rec["lmax_basic"], rec["lmax_flat"], rec["lmin_flat"]), flush=True)
with open("fig1_contrast.csv", "w", newline="") as f:
    wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wtr.writeheader(); wtr.writerows(rows)
print("\nwrote fig1_contrast.csv  (%d rows, %.1f s)" % (len(rows), time.time() - t0))
