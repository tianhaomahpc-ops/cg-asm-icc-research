#!/usr/bin/env python3
"""Fig 7: reordering turns the subdomain ICC inexact (Q1).
Reads reorder_summary.txt + reorder_{1d,2d}_{nat,rcm,rnd}_reshist.txt."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STRUCT = "#1F3A5F"   # natural / structured
ACC    = "#8C2D04"   # random / damaged
MID     = "#2C7FB8"  # sASM
GREY    = "#9AA0A6"

ONAME = {"nat": "natural", "rcm": "RCM", "rnd": "random"}

# ---- summary ----
summ = {}
for ln in open("reorder_summary.txt"):
    if ln.startswith("#") or not ln.strip():
        continue
    dim, ordr, eta, ib, is_ = ln.split()
    summ[(dim.lower(), ordr)] = (float(eta), int(ib), int(is_))

def hist(dim, ordr):
    a = np.loadtxt(f"reorder_{dim}_{ordr}_reshist.txt")
    k = a[:, 0]
    B = np.where(a[:, 1] > 0, a[:, 1], np.nan)
    S = np.where(a[:, 2] > 0, a[:, 2], np.nan)
    return k, B, S

fig, ax = plt.subplots(2, 2, figsize=(15.2, 7.9))

# ===== (0,0) 1D residual histories =====
a = ax[0, 0]
k, B, _ = hist("1d", "nat"); a.semilogy(k, B, color=STRUCT, lw=2.4, label="natural, BASIC (15 it)")
k, B, _ = hist("1d", "rcm"); a.semilogy(k, B, color=MID, lw=1.6, ls=":", label="RCM, BASIC (15 it)")
k, B, S = hist("1d", "rnd"); a.semilogy(k, B, color=ACC, lw=2.4, label="random, BASIC (113 it)")
a.semilogy(k, S, color="#E08214", lw=2.0, ls="--", label="random, sASM (108 it)")
a.set_title("(a) 1D Laplace, N=256, 8 subdomains, O=2\n"
            "natural/RCM: ICC = exact Cholesky (no fill) → 15 it;  "
            "random: ICC drops fill → 113 it", fontsize=10.5)
a.set_xlabel("CG iteration"); a.set_ylabel(r"relative residual $\|r\|/\|b\|$")
a.set_xlim(0, 120); a.grid(alpha=0.3, which="both"); a.legend(fontsize=8.6, loc="upper right")

# ===== (1,0) 2D residual histories =====
a = ax[1, 0]
k, B, S = hist("2d", "nat")
a.semilogy(k, B, color=STRUCT, lw=2.4, label="natural, BASIC (102 it)")
a.semilogy(k, S, color=MID, lw=2.2, ls="--", label="natural, sASM (67 it)")
k, B, S = hist("2d", "rnd")
a.semilogy(k, B, color=ACC, lw=2.4, label="random, BASIC (147 it)")
a.semilogy(k, S, color="#E08214", lw=2.2, ls="--", label="random, sASM (92 it)")
a.set_title("(c) 2D Laplace, 64², 4×4 subdomains, O=2\n"
            "ICC already inexact (5-pt has fill); random ordering makes it worse;  "
            "sASM fixes both", fontsize=10.5)
a.set_xlabel("CG iteration"); a.set_ylabel(r"relative residual $\|r\|/\|b\|$")
a.set_xlim(0, 160); a.grid(alpha=0.3, which="both"); a.legend(fontsize=8.6, loc="upper right")

# ===== (0,1) inexactness proxy eta (the headline number) =====
a = ax[0, 1]
cases = [("1d", "nat"), ("1d", "rcm"), ("1d", "rnd"),
         ("2d", "nat"), ("2d", "rcm"), ("2d", "rnd")]
labels = [f"{d.upper()}\n{ONAME[o]}" for d, o in cases]
etas = [max(summ[c][0], 1e-16) for c in cases]
cols = [STRUCT, MID, ACC, STRUCT, MID, ACC]
bars = a.bar(range(6), etas, color=cols, edgecolor="k", lw=0.5)
a.set_yscale("log"); a.set_ylim(1e-16, 3)
a.axhline(1e-12, color=GREY, ls=":", lw=1.2)
a.text(5.4, 2e-13, "exact-solve floor", color=GREY, fontsize=8, ha="right")
for i, (b, e) in enumerate(zip(bars, etas)):
    a.text(b.get_x()+b.get_width()/2, e*1.6 if e > 1e-14 else 4e-15,
           f"{e:.1e}" if e > 1e-3 else "≈0\n(exact)", ha="center",
           fontsize=8.4, color="k" if e > 1e-3 else STRUCT, fontweight="bold")
a.set_xticks(range(6)); a.set_xticklabels(labels, fontsize=8.6)
a.set_ylabel(r"$\eta=\mathrm{avg}_i\,\|A_i u_i-v\|/\|v\|$  (ICC inexactness)")
a.set_title("(b) Subdomain-solve inexactness vs ordering\n"
            "1D natural/RCM ≈ 1e-15 (ICC IS exact);  1D random = 0.67 (now inexact)",
            fontsize=10.5)

# ===== (1,1) iteration counts =====
a = ax[1, 1]
xb = np.arange(6); w = 0.38
iB = [summ[c][1] for c in cases]
iS = [summ[c][2] for c in cases]
a.bar(xb - w/2, iB, w, color=ACC, edgecolor="k", lw=0.5, label="BASIC")
a.bar(xb + w/2, iS, w, color=STRUCT, edgecolor="k", lw=0.5, label="sASM")
for i in range(6):
    a.text(xb[i]-w/2, iB[i]+2, str(iB[i]), ha="center", fontsize=8, color=ACC, fontweight="bold")
    a.text(xb[i]+w/2, iS[i]+2, str(iS[i]), ha="center", fontsize=8, color=STRUCT, fontweight="bold")
a.set_xticks(xb); a.set_xticklabels(labels, fontsize=8.6)
a.set_ylabel("CG iterations to $10^{-8}$"); a.set_ylim(0, 170)
a.set_title("(d) CG iterations: BASIC vs sASM\n"
            "1D random: 15→113 (reorder breaks exactness);  sASM helps once inexact",
            fontsize=10.5)
a.legend(fontsize=9, loc="upper left"); a.grid(alpha=0.3, axis="y")

fig.suptitle("Fig. 7  —  Reordering the subdomain unknowns turns the ICC factorization inexact "
             "(Q1: 1D natural ICC = exact; random ICC ≠ exact)",
             fontsize=12.5, fontweight="bold", color=STRUCT)
fig.tight_layout(rect=[0, 0, 1, 0.965])
fig.savefig("fig7_reorder.png", dpi=140)
print("wrote fig7_reorder.png")
