#!/usr/bin/env python3
"""Fig 11: methodological controls.
(a) strip vs box -> lambda_max follows the over-count N_hat, not subdomain size.
(b) two-level BASIC vs sASM -> a coarse space fixes lambda_min, NOT lambda_max;
    it cannot substitute for sASM. Reads rigor_stripbox.txt, rigor_twolevel.txt."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACC="#8C2D04"; STRUCT="#1F3A5F"; GREY="#6E6E6E"

fig, ax = plt.subplots(1, 2, figsize=(14.2, 5.9))

# ---- (a) strip vs box ----
rows = [l.split() for l in open("rigor_stripbox.txt") if l.strip() and not l.startswith("#")]
strip = [(int(O),int(nh),float(le),int(sz)) for lay,O,nh,le,li,sz in rows if lay=="strip"]
box   = [(int(O),int(nh),float(le),int(sz)) for lay,O,nh,le,li,sz in rows if lay=="box"]
a = ax[0]
sz_s=[r[3] for r in strip]; lx_s=[r[2] for r in strip]; nh_s=[r[1] for r in strip]
sz_b=[r[3] for r in box];   lx_b=[r[2] for r in box];   nh_b=[r[1] for r in box]
a.plot(sz_s, lx_s, "s-", color=STRUCT, ms=10, lw=2.2, label="strip ($\\hat N$ stays small)")
a.plot(sz_b, lx_b, "o-", color=ACC, ms=10, lw=2.2, label="box ($\\hat N$ grows)")
for x,y,nh in zip(sz_s,lx_s,nh_s): a.annotate(f"$\\hat N$={nh}", (x,y), textcoords="offset points", xytext=(6,6), color=STRUCT, fontsize=8.5)
for x,y,nh in zip(sz_b,lx_b,nh_b): a.annotate(f"$\\hat N$={nh}", (x,y), textcoords="offset points", xytext=(6,-12), color=ACC, fontsize=8.5)
a.set_xscale("log")
a.set_xlabel("subdomain size (DOF, log)  --- grows with overlap")
a.set_ylabel(r"$\lambda_{\max}$(BASIC, exact)")
a.set_title("(a) $\\lambda_{\\max}$ follows the over-count $\\hat N$, not subdomain size\n"
            "2D Laplace $96^2$, $P{=}6$, exact Cholesky; O=2/4/8/16", fontsize=10.3)
a.grid(alpha=0.3, which="both"); a.legend(fontsize=9, loc="upper left")
a.annotate("strip O=2: 1728 DOF, $\\hat N$=2 $\\to$ $\\lambda_{\\max}$=1.9\nbox O=2: 324 DOF, $\\hat N$=4 $\\to$ $\\lambda_{\\max}$=4.0\n(5$\\times$ the DOF, half the $\\lambda_{\\max}$)",
           xy=(324,4.0), xytext=(360,1.4), fontsize=8.6, color=GREY,
           arrowprops=dict(arrowstyle="->", color=GREY, lw=1))

# ---- (b) two-level BASIC vs sASM ----
tl = {}
for l in open("rigor_twolevel.txt"):
    if l.startswith("#") or not l.strip(): continue
    m,two,lmin,lmax,kap,it,sec = l.split()
    tl[(m,int(two))] = dict(lmin=float(lmin),lmax=float(lmax),kap=float(kap),it=int(it),sec=float(sec))
b = ax[1]
cfgs = [("BASIC",0),("BASIC",1),("sASM",0),("sASM",1)]
labels = ["BASIC\none-level","BASIC\ntwo-level","sASM\none-level","sASM\ntwo-level"]
x = np.arange(4); w=0.38
lmax = [tl[c]["lmax"] for c in cfgs]; lmin=[tl[c]["lmin"] for c in cfgs]; its=[tl[c]["it"] for c in cfgs]
cols = [ACC,ACC,STRUCT,STRUCT]
bars = b.bar(x, lmax, w*1.6, color=cols, edgecolor="k", lw=0.6, alpha=0.92)
for i,c in enumerate(cfgs):
    b.text(x[i], lmax[i]+0.12, f"$\\lambda_{{max}}$={lmax[i]:.2f}", ha="center", fontsize=8.6, fontweight="bold", color=cols[i])
    b.text(x[i], 0.25, f"{its[i]} its", ha="center", fontsize=9, color="white", fontweight="bold")
    b.text(x[i], -0.55, f"$\\lambda_{{min}}$={lmin[i]:.3f}", ha="center", fontsize=7.8, color=GREY)
b.set_xticks(x); b.set_xticklabels(labels, fontsize=8.8)
b.set_ylabel(r"$\lambda_{\max}(M^{-1}A)$ (bars);  iters in bar")
b.set_ylim(-0.9, 5.4)
b.set_title("(b) A coarse space fixes $\\lambda_{\\min}$, NOT $\\lambda_{\\max}$ --- it cannot replace sASM\n"
            "120$^2$, 6$\\times$6, O=2, ICC: two-level BASIC keeps $\\lambda_{\\max}$=4.5 (92 its) vs two-level sASM 2.1 (64 its)",
            fontsize=10.0)
b.axhline(0, color="k", lw=0.6)
b.annotate("coarse lifts $\\lambda_{\\min}$\n(both methods)", xy=(0.5,-0.55), xytext=(0.5,2.6), fontsize=8.4, color=GREY, ha="center",
           arrowprops=dict(arrowstyle="->", color=GREY, lw=1))
b.annotate("only sASM lowers $\\lambda_{\\max}$\n(removes $\\hat N$)", xy=(3,2.07), xytext=(2.2,4.4), fontsize=8.6, color=STRUCT, ha="center",
           arrowprops=dict(arrowstyle="->", color=STRUCT, lw=1.2))

fig.suptitle("Fig. 11  —  Controls: (a) the over-count $\\hat N$ (not size) sets $\\lambda_{\\max}$;  "
             "(b) a coarse space and sASM fix opposite ends of the spectrum --- you need both",
             fontsize=11.6, fontweight="bold", color=STRUCT)
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig("fig11_controls.png", dpi=140)
print("wrote fig11_controls.png")
