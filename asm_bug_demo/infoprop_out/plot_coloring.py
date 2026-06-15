#!/usr/bin/env python3
"""Fig 8: the coloring number N_c IS the over-count, and sASM removes it.
Reads coloring_sweep.txt: O m_max N_c lmax_BE lmin_BE lmax_SE lmin_SE lmax_BI lmin_BI lmax_SI lmin_SI."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACC   = "#8C2D04"   # BASIC (over-counts)
STRUCT= "#1F3A5F"   # sASM  (partition of unity)
NCCOL = "#222222"

d = np.loadtxt("coloring_sweep.txt")
O   = d[:,0]; mmax = d[:,1]; Nc = d[:,2]
lBE = d[:,3]; mBE = d[:,4]
lSE = d[:,5]; mSE = d[:,6]
lBI = d[:,7]; mBI = d[:,8]
lSI = d[:,9]; mSI = d[:,10]
xx = np.arange(len(O))                    # evenly spaced; label with actual O

fig, ax = plt.subplots(1, 2, figsize=(14.5, 6.0))

# ===== (a) lambda_max vs overlap, with N_c overlaid =====
a = ax[0]
a.step(xx, Nc, where="mid", color=NCCOL, lw=2.6, ls=(0,(4,2)),
       label=r"$N_c$ = $\hat N$ = max multiplicity (coincide here)")
a.plot(xx, lBE, "o-",  color=ACC,    lw=2.4, ms=8, label=r"$\lambda_{\max}$  BASIC, exact")
a.plot(xx, lBI, "s--", color="#D7642A", lw=2.2, ms=7, label=r"$\lambda_{\max}$  BASIC, ICC(0)")
a.plot(xx, lSE, "o-",  color=STRUCT, lw=2.4, ms=8, label=r"$\lambda_{\max}$  sASM, exact")
a.plot(xx, lSI, "s--", color="#3B7BBF", lw=2.2, ms=7, label=r"$\lambda_{\max}$  sASM, ICC(0)")
a.set_xticks(xx); a.set_xticklabels([f"{int(o)}" for o in O])
a.set_xlabel("overlap $O$"); a.set_ylabel(r"$\lambda_{\max}(M^{-1}A)$")
a.set_ylim(0, 12)
a.annotate(r"BASIC $\lambda_{\max}=\hat N$ exactly"+"\n(over-count = max multiplicity)",
           xy=(4, 9.0), xytext=(1.4, 10.7), fontsize=9.5, color=ACC,
           arrowprops=dict(arrowstyle="->", color=ACC, lw=1.2))
a.annotate("ICC lifts it to $\\approx\\omega N_c$", xy=(5, 10.74), xytext=(3.1, 7.6),
           fontsize=9, color="#D7642A", arrowprops=dict(arrowstyle="->", color="#D7642A", lw=1))
a.annotate(r"sASM flat: $\hat N$ removed"+"\n(ICC $\\to\\lambda_{\\max}\\approx\\omega\\approx1.25$)",
           xy=(4, 1.25), xytext=(0.9, 3.2), fontsize=9.5, color=STRUCT,
           arrowprops=dict(arrowstyle="->", color=STRUCT, lw=1.2))
a.set_title("(a)  $\\lambda_{\\max}$ tracks $N_c$ for BASIC, is flat for sASM\n"
            "2D Laplace $120^2$, $6\\times6$ subdomains, exact (Cholesky) vs ICC(0)", fontsize=11)
a.grid(alpha=0.3); a.legend(fontsize=8.6, loc="center right", framealpha=0.95)

# ===== (b) condition number kappa = lmax/lmin (the iteration driver) =====
b = ax[1]
b.semilogy(xx, lBE/mBE, "o-",  color=ACC,    lw=2.4, ms=8, label="BASIC, exact")
b.semilogy(xx, lBI/mBI, "s--", color="#D7642A", lw=2.2, ms=7, label="BASIC, ICC(0)")
b.semilogy(xx, lSE/mSE, "o-",  color=STRUCT, lw=2.4, ms=8, label="sASM, exact")
b.semilogy(xx, lSI/mSI, "s--", color="#3B7BBF", lw=2.2, ms=7, label="sASM, ICC(0)")
b.set_xticks(xx); b.set_xticklabels([f"{int(o)}" for o in O])
b.set_xlabel("overlap $O$"); b.set_ylabel(r"$\kappa=\lambda_{\max}/\lambda_{\min}$  (drives CG iters)")
b.set_title("(b)  Condition number: ICC inflates $\\kappa$;\n"
            "sASM keeps the inflation $\\sim3\\times$ smaller (BASIC-ICC vs sASM-ICC)", fontsize=11)
b.grid(alpha=0.3, which="both"); b.legend(fontsize=9, loc="upper right")

fig.suptitle("Fig. 8  —  Measuring the over-count: $\\lambda_{\\max}(\\mathrm{BASIC,exact})=\\hat N$ (max multiplicity $\\leq N_c$); "
             "sASM's $D^{-1/2}$ removes it  ($\\kappa\\leq C_0^2\\,\\omega\\,N_c$)",
             fontsize=12.0, fontweight="bold", color=STRUCT)
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig("fig8_coloring.png", dpi=140)
print("wrote fig8_coloring.png")
print(f"N_c sweep: {[int(x) for x in Nc]}   m_max: {[int(x) for x in mmax]}")
print(f"lmax BASIC-exact: {lBE}  (== N_c?)")
print(f"lmax sASM-ICC: {lSI}  (== omega, flat?)")
