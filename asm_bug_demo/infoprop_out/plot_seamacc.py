#!/usr/bin/env python3
"""Fig 13: exact vs inexact BOUNDARY ERROR ACCUMULATION at the seams.
(a) convergence (the Fig-3 symptom); (b) seam-residual fraction (the mechanism):
BASIC traps residual at the over-counted seams; with ICC that trapped error is the
bottleneck; sASM does not trap it. Reads seamacc_<tag>.txt."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = {"BASICexact":("#1F3A5F","BASIC, exact (23 it)","-"),
     "BASICicc":  ("#8C2D04","BASIC, ICC (102 it)","--"),
     "sASMicc":   ("#E08214","sASM, ICC (68 it)","--")}

D = {t: np.loadtxt(f"seamacc_{t}.txt") for t in C}

fig, ax = plt.subplots(1, 2, figsize=(13.6, 5.6))

# (a) convergence
a = ax[0]
for t,(col,lab,ls) in C.items():
    d=D[t]; a.semilogy(d[:,0], np.where(d[:,1]>0,d[:,1],np.nan), color=col, lw=2.4, ls=ls, label=lab)
a.set_xlabel("CG iteration"); a.set_ylabel(r"relative residual $\|r\|/\|b\|$")
a.set_xlim(0,110); a.grid(alpha=0.3, which="both"); a.legend(fontsize=9.5, loc="upper right")
a.set_title("(a) convergence (the Fig-3 symptom)\n"
            "exact fast; BASIC-ICC slow (102); sASM-ICC recovers (68)", fontsize=10.5)

# (b) seam-residual fraction (the mechanism)
b = ax[1]
for t,(col,lab,ls) in C.items():
    d=D[t]; b.plot(d[:,0], d[:,2], color=col, lw=2.4, ls=ls, label=lab)
b.set_xlabel("CG iteration"); b.set_ylabel("seam-residual fraction  "
            r"$\|r\|^2_{\rm overlap}/\|r\|^2$")
b.set_xlim(0,110); b.set_ylim(0,0.95); b.grid(alpha=0.3)
b.legend(fontsize=9.5, loc="upper left")
b.axhspan(0.5,0.95, color="#8C2D04", alpha=0.05)
b.axhspan(0.0,0.42, color="#1F3A5F", alpha=0.05)
b.annotate("BASIC-ICC: over-counted error\nstays trapped at the seams\n(fraction ~0.5–0.8) $\\Rightarrow$ the bottleneck",
           xy=(70,0.6), xytext=(30,0.78), fontsize=9, color="#8C2D04", fontweight="bold",
           arrowprops=dict(arrowstyle="->", color="#8C2D04", lw=1.4))
b.annotate("sASM-ICC: seams not over-counted\n$\\Rightarrow$ residual not trapped (~0.3)",
           xy=(60,0.33), xytext=(45,0.10), fontsize=9, color="#E08214", fontweight="bold",
           arrowprops=dict(arrowstyle="->", color="#E08214", lw=1.4))
b.set_title("(b) where the residual lives --- the boundary error accumulation\n"
            "BASIC traps it at the over-counted seams; sASM does not", fontsize=10.5)

fig.suptitle("Fig. 13  —  Exact vs inexact at the seams: Fig 3 shows the convergence gap (a); "
             "the seam-residual fraction (b) shows WHERE the inexact error accumulates",
             fontsize=11.6, fontweight="bold", color="#1F3A5F")
fig.tight_layout(rect=[0,0,1,0.94])
fig.savefig("fig13_seamacc.png", dpi=132)
print("wrote fig13_seamacc.png")
