"""Plots for the MFEM/PETSc/MPI sigma-tensor sweep."""
import csv, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "aniso_nx48_n4_L0.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "aniso/figM_mfem_mechanism.png"
R = []
for row in csv.DictReader(open(src)):
    d = {}
    for k, v in row.items():
        if k in ("fiber", "method"): d[k] = v
        else:
            try: d[k] = float(v)
            except Exception: d[k] = np.nan
    R.append(d)
g = lambda m, r, O: next(x for x in R if x["method"] == m and x["ratio"] == r and x["O"] == O)
ratios = sorted({x["ratio"] for x in R}); Os = sorted({x["O"] for x in R})

fig, ax = plt.subplots(1, 4, figsize=(21, 4.6))

amp = [g("BASIC", r, 3)["iter"] / g("BASIC", r, 0)["iter"] for r in ratios]
ax[0].semilogx(ratios, amp, 'o-', lw=2.4, ms=8, color='tab:red')
ax[0].axhline(1.0, color='k', lw=1)
ax[0].fill_between([1, 200], 1, max(amp) * 1.05, color='crimson', alpha=.08)
ax[0].set_xlim(1, 110)
ax[0].set_xlabel("conductivity contrast  $r=\\sigma_\\ell/\\sigma_t$")
ax[0].set_ylabel("iter($O$=3) / iter($O$=0),  PC_ASM_BASIC")
ax[0].set_title("(a) overlap anomaly vs contrast\nMFEM 3D P1 tet, METIS, 4 MPI ranks")
ax[0].grid(alpha=.3)

ax[1].semilogx(ratios, [g("BASIC", r, 3)["omega"] for r in ratios], 's--', lw=2.2, ms=8,
               color='tab:red', label="$\\omega=\\lambda_{max}(M_i^{-1}A_i)$")
ax[1].set_xlabel("contrast $r$"); ax[1].set_ylabel("$\\omega$")
ax[1].set_title("(b) the ICC inexactness $\\omega$")
ax[1].legend(); ax[1].grid(alpha=.3)

fx = [g("BASIC", r, 3)["lmax"] / g("BASIC", r, 0)["lmax"] for r in ratios]
fn = [g("BASIC", r, 3)["lmin"] / g("BASIC", r, 0)["lmin"] for r in ratios]
ax[2].semilogx(ratios, fx, 'o-', lw=2.2, ms=7, color='tab:red',
               label="$\\lambda_{max}$ penalty (over-count)")
ax[2].semilogx(ratios, fn, 's-', lw=2.2, ms=7, color='tab:blue',
               label="$\\lambda_{min}$ payoff (overlap reach)")
ax[2].set_xlabel("contrast $r$"); ax[2].set_ylabel("factor gained, $O$=0 $\\to$ $O$=3")
ax[2].set_title("(c) the race that decides the sign")
ax[2].legend(fontsize=9); ax[2].grid(alpha=.3)

for m, c, mk in [("BASIC", 'grey', 'x:'), ("sASM", 'tab:red', 'o-'),
                 ("ramp", 'tab:blue', 's-'), ("harm", 'tab:green', 'D-')]:
    ax[3].semilogx(ratios, [g(m, r, 3)["iter"] for r in ratios], mk, color=c, lw=2, ms=7, label=m)
ax[3].set_xlabel("contrast $r$"); ax[3].set_ylabel("CG iterations at $O$=3")
ax[3].set_title("(d) the four weights")
ax[3].legend(fontsize=9); ax[3].grid(alpha=.3)

plt.tight_layout(); plt.savefig(out, dpi=135); print("wrote", out)
