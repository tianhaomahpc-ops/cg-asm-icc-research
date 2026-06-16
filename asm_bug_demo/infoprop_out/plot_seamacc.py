#!/usr/bin/env python3
"""Fig 13: exact vs inexact BOUNDARY ERROR ACCUMULATION at the seams.
Top: residual fields at k=15 (shared scale) -- exact converged (dark), BASIC-ICC
stuck at the seams (bright cross-pattern), sASM-ICC less. Bottom: convergence
(the Fig-3 symptom) + seam-residual fraction (the mechanism)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

C = {"BASICexact":("#1F3A5F","BASIC, exact (23 it)","-"),
     "BASICicc":  ("#8C2D04","BASIC, ICC (102 it)","--"),
     "sASMicc":   ("#E08214","sASM, ICC (68 it)","--")}
D = {t: np.loadtxt(f"seamacc_{t}.txt") for t in C}
F = {t: np.abs(np.loadtxt(f"seamacc_field_{t}_k15.txt")) for t in C}
seams = [16,32,48]                                  # 4x4 subdomains on 64 grid

fig = plt.figure(figsize=(13.6, 9.0))
gs = fig.add_gridspec(2, 6, height_ratios=[1.05, 1.0], hspace=0.42, wspace=0.55)

# ---- top: 3 residual-field heatmaps at k=15, shared LogNorm ----
vmax = max(F[t].max() for t in F); vmin = vmax*5e-3
htitle = {"BASICexact":"BASIC, exact: converged\n(max$|r|$=2e-5 — cleared)",
          "BASICicc":  "BASIC, ICC: residual STUCK\nat the seams (max$|r|$=5.4e-2)",
          "sASMicc":   "sASM, ICC: less, not seam-locked\n(max$|r|$=2.0e-2)"}
for j,t in enumerate(["BASICexact","BASICicc","sASMicc"]):
    a = fig.add_subplot(gs[0, 2*j:2*j+2])
    im = a.imshow(F[t], origin="lower", cmap="inferno", norm=LogNorm(vmin=vmin, vmax=vmax))
    for s in seams:
        a.axvline(s, color="w", lw=0.5, ls=":", alpha=0.5); a.axhline(s, color="w", lw=0.5, ls=":", alpha=0.5)
    a.set_title(htitle[t], fontsize=9.2, color=C[t][0]); a.set_xticks([]); a.set_yticks([])
cax = fig.add_axes([0.92, 0.58, 0.013, 0.30]); cb = fig.colorbar(im, cax=cax); cb.set_label("$|r|$ at $k{=}15$ (log)", fontsize=8.5)
fig.text(0.5, 0.955, "residual field at the same iteration $k{=}15$ (white dotted = subdomain seams)",
         ha="center", fontsize=10, color="#444")

# ---- bottom-left: convergence ----
a = fig.add_subplot(gs[1, 0:3])
for t,(col,lab,ls) in C.items():
    d=D[t]; a.semilogy(d[:,0], np.where(d[:,1]>0,d[:,1],np.nan), color=col, lw=2.4, ls=ls, label=lab)
a.set_xlabel("CG iteration"); a.set_ylabel(r"$\|r\|/\|b\|$"); a.set_xlim(0,110)
a.grid(alpha=0.3, which="both"); a.legend(fontsize=9, loc="upper right")
a.set_title("(a) convergence — the Fig-3 symptom (global)", fontsize=10.2)

# ---- bottom-right: seam-residual fraction ----
b = fig.add_subplot(gs[1, 3:6])
for t,(col,lab,ls) in C.items():
    d=D[t]; b.plot(d[:,0], d[:,2], color=col, lw=2.4, ls=ls, label=lab)
b.set_xlabel("CG iteration"); b.set_ylabel(r"$\|r\|^2_{\rm overlap}/\|r\|^2$")
b.set_xlim(0,110); b.set_ylim(0,0.95); b.grid(alpha=0.3); b.legend(fontsize=9, loc="upper left")
b.axhspan(0.5,0.95, color="#8C2D04", alpha=0.05); b.axhspan(0.0,0.42, color="#1F3A5F", alpha=0.05)
b.annotate("BASIC-ICC: error trapped\nat seams (~0.5–0.8)", xy=(70,0.6), xytext=(28,0.80),
           fontsize=8.6, color="#8C2D04", fontweight="bold", arrowprops=dict(arrowstyle="->", color="#8C2D04", lw=1.3))
b.annotate("sASM-ICC: not trapped (~0.3)", xy=(58,0.33), xytext=(40,0.10),
           fontsize=8.6, color="#E08214", fontweight="bold", arrowprops=dict(arrowstyle="->", color="#E08214", lw=1.3))
b.set_title("(b) seam-residual fraction — WHERE the error accumulates", fontsize=10.2)

fig.suptitle("Fig. 13  —  Exact vs inexact at the seams: Fig 3 (panel a) shows only the convergence gap; "
             "the fields + seam fraction (b) show WHERE the inexact error accumulates",
             fontsize=11.4, fontweight="bold", color="#1F3A5F", y=1.0)
fig.savefig("fig13_seamacc.png", dpi=132, bbox_inches="tight")
print("wrote fig13_seamacc.png")
