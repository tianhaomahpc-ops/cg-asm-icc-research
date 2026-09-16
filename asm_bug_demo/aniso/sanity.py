import anisopu as ap, numpy as np, time
np.set_printoptions(precision=4)

nx, Px, Py = 96, 4, 4
sig = ap.sigma_tensor(1.0, 0.0)
A, b, coords, free, n1 = ap.build_fem(nx, sig)
own = ap.box_partition(coords, Px, Py); nsub = Px*Py
print("N = %d   nsub = %d   isotropic sigma=I" % (A.shape[0], nsub))

def run(O, kind, q=0.8, exact=False):
    sets = ap.overlap_sets(A, own, nsub, O)
    Alocs = [A[i][:, i].tocsr() for i, _ in sets]
    w = ap.build_weights(kind, sets, Alocs, A.shape[0], O, q)
    M = ap.ASM(A, sets, w, exact=exact)
    x, it, (lo, hi) = ap.pcg(A, b, M, rtol=1e-6)
    mult = np.zeros(A.shape[0])
    for i, _ in sets: mult[i] += 1
    return it, lo, hi, int(mult.max())

print("\n--- ANCHOR 1: O=0 => D=I => every variant must be identical ---")
for k in ["basic","flat","graded","ramp","harm"]:
    it, lo, hi, m = run(0, k)
    print("  %-7s iter=%3d  lmin=%.4e lmax=%.4f  Nhat=%d" % (k, it, lo, hi, m))

print("\n--- ANCHOR 2: graded q=1.0 must equal flat (sASM) bit-for-bit ---")
for O in (1,2,3):
    i1,_,_,_ = run(O,"flat"); i2,_,_,_ = run(O,"graded",q=1.0)
    print("  O=%d  flat=%3d   graded(q=1)=%3d   %s" % (O,i1,i2,"EQ" if i1==i2 else "MISMATCH"))

print("\n--- ANCHOR 3: reproduce the anomaly (BASIC up, sASM down) ---")
print("  O   BASIC  sASM   BASIC-exactChol")
for O in range(0,6):
    ib,_,_,m = run(O,"basic"); isa,_,_,_ = run(O,"flat"); ie,_,_,_ = run(O,"basic",exact=True)
    print("  %d   %4d   %4d   %4d      (Nhat=%d)" % (O, ib, isa, ie, m))
