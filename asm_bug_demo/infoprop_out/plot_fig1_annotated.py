#!/usr/bin/env python3
"""Fig 1 (annotated): make the over-count visible with arrows, and make explicit
that BOTH curves are EXACT-solve (in 1D, ICC(0) = exact => no separate inexact line).
Reads ip1d_apply_O2.txt / O4.txt (cols: x m_k r zB zS)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACC="#8C2D04"; STRUCT="#1F3A5F"; GREY="#9AA0A6"

def load(O):
    d = np.loadtxt(f"ip1d_apply_O{O}.txt")
    return d[:,0], d[:,1], d[:,2], d[:,3], d[:,4]   # x, m_k, r, zB, zS

fig, ax = plt.subplots(2, 2, figsize=(13.6, 7.4),
                       gridspec_kw={"height_ratios":[1,2.6]})

for col,O in enumerate([2,4]):
    x,mk,r,zB,zS = load(O)
    # ---- top: multiplicity m_k ----
    a = ax[0,col]
    a.fill_between(x, 1, mk, where=mk>1.5, color=ACC, alpha=0.18, step="mid")
    a.step(x, mk, where="mid", color=STRUCT, lw=1.8)
    a.set_ylim(0.7,2.4); a.set_yticks([1,2]); a.set_xlim(0,1)
    a.set_ylabel("multiplicity\n$m_k$", fontsize=10)
    a.set_title(f"overlap $O={O}$", fontsize=11)
    if col==1:
        # arrow to an m_k=2 seam
        a.annotate("$m_k$=2: this point sits in\nTWO subdomains (overlap seam)",
                   xy=(0.5,2), xytext=(0.62,2.05), fontsize=8.6, color=ACC,
                   arrowprops=dict(arrowstyle="->", color=ACC, lw=1.3))
        a.annotate("$m_k$=1: interior\n(one subdomain)", xy=(0.30,1), xytext=(0.04,1.45),
                   fontsize=8.6, color=STRUCT, arrowprops=dict(arrowstyle="->", color=STRUCT, lw=1.1))
    # ---- bottom: correction z(x) ----
    b = ax[1,col]
    rs = r/np.max(r)*np.max(zB)*0.62      # residual scaled for display
    b.fill_between(x, 1, 2.4, where=mk>1.5, color=ACC, alpha=0.10, transform=b.get_xaxis_transform())
    b.plot(x, rs, color=GREY, lw=1.6, ls="--", label="input residual $r$ (scaled)")
    b.plot(x, zB, color=ACC, lw=2.6, label=r"BASIC $z=\sum_i R_i^\top A_i^{-1}R_i\,r$")
    b.plot(x, zS, color=STRUCT, lw=2.6, label=r"sASM $z=D^{-1/2}(\cdots)D^{-1/2}r$")
    b.set_xlim(0,1); b.set_xlabel("position $x$"); b.set_ylabel("correction $z(x)$", fontsize=10)
    b.legend(fontsize=8.6, loc="upper right")
    if col==1:
        # over-count bump (BASIC) at the seam
        ib = np.argmin(np.abs(x-0.498))
        b.annotate("BASIC over-counts here:\nadds BOTH subdomains' solves\n$\\Rightarrow$ jumps UP ($\\times m_k$)",
                   xy=(0.498, zB[ib]), xytext=(0.60, zB[ib]+8), fontsize=9, color=ACC, fontweight="bold",
                   arrowprops=dict(arrowstyle="->", color=ACC, lw=1.6))
        b.annotate("sASM normalizes by $1/m_k$\n$\\Rightarrow$ counted once, smooth",
                   xy=(0.498, zS[ib]), xytext=(0.18, zS[ib]-30), fontsize=9, color=STRUCT, fontweight="bold",
                   arrowprops=dict(arrowstyle="->", color=STRUCT, lw=1.6))
    else:
        ib = np.argmin(np.abs(x-0.5))
        b.annotate("seam: red bumps up\n(over-count), blue dips",
                   xy=(0.5, zB[ib]), xytext=(0.60, zB[ib]-22), fontsize=8.4, color=ACC,
                   arrowprops=dict(arrowstyle="->", color=ACC, lw=1.2))

fig.suptitle("Fig. 1 (annotated)  —  BOTH lines are EXACT (Cholesky) solves;  "
             "red BASIC over-counts the overlap,  blue sASM normalizes it",
             fontsize=12.5, fontweight="bold", color=STRUCT, y=0.995)
fig.text(0.5, 0.005,
  "There is NO ‘inexact’ line in this figure: in 1D, ICC(0) = exact Cholesky (a tridiagonal has no fill), so exact and ICC coincide. "
  "The exact-vs-inexact (ICC) contrast first appears in 2D — see Fig. 3.",
  ha="center", fontsize=9.2, color=ACC, style="italic",
  bbox=dict(boxstyle="round,pad=0.4", fc="#FAECE7", ec=ACC, lw=0.8))
fig.tight_layout(rect=[0,0.045,1,0.96])
fig.savefig("fig1b_overcount_annotated.png", dpi=132)
print("wrote fig1b_overcount_annotated.png")
