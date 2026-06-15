#!/usr/bin/env python3
"""Fig 9: a coarse space (two-level sASM) lifts lambda_min and makes iterations scalable.
Reads twolevel_fixed.txt and twolevel_scaling.txt."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ONE  = "#8C2D04"   # one-level (red)
TWO  = "#1F3A5F"   # two-level (blue)

fx = np.loadtxt("twolevel_fixed.txt")     # level lmin lmax kappa its
sc = np.loadtxt("twolevel_scaling.txt")   # P nsub N one two
one = fx[0]; two = fx[1]
nsub = sc[:,1]; i1 = sc[:,3]; i2 = sc[:,4]

fig, ax = plt.subplots(1, 2, figsize=(14.0, 5.8))

# ===== (a) fixed config: lambda_min jump + kappa drop =====
a = ax[0]
groups = [r"$\lambda_{\min}$", r"$\lambda_{\max}$", r"$\kappa=\lambda_{\max}/\lambda_{\min}$"]
xv = np.arange(3); w = 0.36
one_vals = [one[1], one[2], one[3]]
two_vals = [two[1], two[2], two[3]]
a.bar(xv-w/2, one_vals, w, color=ONE, edgecolor="k", lw=0.5, label="one-level sASM (121 its)")
a.bar(xv+w/2, two_vals, w, color=TWO, edgecolor="k", lw=0.5, label="two-level sASM (64 its)")
a.set_yscale("log"); a.set_ylim(1e-3, 2e3)
for i,(o,t) in enumerate(zip(one_vals,two_vals)):
    a.text(xv[i]-w/2, o*1.3, f"{o:.3g}", ha="center", fontsize=8.5, color=ONE, fontweight="bold")
    a.text(xv[i]+w/2, t*1.3, f"{t:.3g}", ha="center", fontsize=8.5, color=TWO, fontweight="bold")
a.set_xticks(xv); a.set_xticklabels(groups, fontsize=11)
a.set_ylabel("value (log)")
a.set_title("(a) Fixed 120², 6×6, O=2, ICC(0)\n"
            r"coarse space lifts $\lambda_{\min}$ 14× (0.0021→0.030) → $\kappa$ 8.7× smaller → its 121→64",
            fontsize=10.5)
a.legend(fontsize=9, loc="upper left"); a.grid(alpha=0.3, axis="y", which="both")
a.annotate("the lever:\n$\\lambda_{\\min}$ jumps", xy=(0+w/2, two[1]), xytext=(0.35, 0.3),
           fontsize=9, color=TWO, ha="center",
           arrowprops=dict(arrowstyle="->", color=TWO, lw=1.4))

# ===== (b) scalability: iterations vs #subdomains =====
b = ax[1]
b.plot(nsub, i1, "o-", color=ONE, lw=2.6, ms=9, label="one-level sASM")
b.plot(nsub, i2, "s-", color=TWO, lw=2.6, ms=9, label="two-level sASM")
for xq,yq in zip(nsub,i1): b.text(xq, yq+6, f"{int(yq)}", ha="center", fontsize=8.5, color=ONE, fontweight="bold")
for xq,yq in zip(nsub,i2): b.text(xq, yq-12, f"{int(yq)}", ha="center", fontsize=8.5, color=TWO, fontweight="bold")
b.set_xlabel("number of subdomains  (subdomain size fixed ≈ 20×20)")
b.set_ylabel("CG iterations to $10^{-8}$")
b.set_ylim(0, 240)
b.set_title("(b) Scalability — the decisive property\n"
            "one-level grows with #subdomains; two-level is FLAT (≈64), 3.3× fewer at 144",
            fontsize=10.5)
b.legend(fontsize=10, loc="center right"); b.grid(alpha=0.3)
b.annotate("one-level curse:\nits ∝ √#subdomains", xy=(144, 217), xytext=(60, 200),
           fontsize=9, color=ONE, arrowprops=dict(arrowstyle="->", color=ONE, lw=1.2))
b.annotate("two-level: scalable", xy=(144, 65), xytext=(70, 95),
           fontsize=9.5, color=TWO, fontweight="bold",
           arrowprops=dict(arrowstyle="->", color=TWO, lw=1.4))

fig.suptitle("Fig. 9  —  Improving sASM: add a coarse space.  The bottleneck is $\\lambda_{\\min}$ "
             "(missing global coupling), not $\\lambda_{\\max}$; a two-level sASM fixes it and is scalable.",
             fontsize=12.0, fontweight="bold", color=TWO)
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig("fig9_twolevel.png", dpi=140)
print("wrote fig9_twolevel.png")
