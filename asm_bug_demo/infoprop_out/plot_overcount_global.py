#!/usr/bin/env python3
"""Fig 1c: the over-count at EVERY seam, with a domain-wide residual (r=1).
Top: z=M^-1 r for BASIC vs sASM (BASIC sits above sASM at every seam).
Bottom: the difference z_BASIC - z_sASM = the pure over-count, one bump per seam.
Reads overcount_global_O{2,4}.txt (cols: x m_k r zB zS diff)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACC="#8C2D04"; STRUCT="#1F3A5F"

fig, ax = plt.subplots(2, 2, figsize=(13.4, 6.8), sharex=True)

for col,O in enumerate([2,4]):
    d = np.loadtxt(f"overcount_global_O{O}.txt")
    x,mk,r,zB,zS,diff = d.T
    seam = mk>1.5
    # top: z curves
    a = ax[0,col]
    a.fill_between(x,0,1, where=seam, transform=a.get_xaxis_transform(), color=ACC, alpha=0.10, step="mid")
    a.plot(x,zB,color=ACC,lw=2.2,label=r"BASIC $z=\sum_i R_i^\top A_i^{-1}R_i\,r$")
    a.plot(x,zS,color=STRUCT,lw=2.2,label=r"sASM $z=D^{-1/2}(\cdots)D^{-1/2}r$")
    a.set_title(f"overlap $O={O}$:  $z=M^{{-1}}r$ for a constant residual $r=1$", fontsize=10.5)
    a.set_ylabel("correction $z(x)$"); a.grid(alpha=0.25)
    if col==0: a.legend(fontsize=8.6, loc="lower center")
    if col==1:
        ix=np.argmin(np.abs(x-0.5))
        a.annotate("BASIC sits ABOVE sASM\nat every seam (over-count)", xy=(0.5,zB[ix]),
                   xytext=(0.30,zB[ix]+60), fontsize=8.8, color=ACC, fontweight="bold",
                   arrowprops=dict(arrowstyle="->", color=ACC, lw=1.3))
    # bottom: difference = pure over-count
    b = ax[1,col]
    b.fill_between(x,0,diff, color=ACC, alpha=0.55, lw=0)
    b.plot(x,diff,color=ACC,lw=1.4)
    b.set_title(f"over-count $=z_{{\\rm BASIC}}-z_{{\\rm sASM}}$:  one bump at every seam", fontsize=10.5)
    b.set_xlabel("position $x$"); b.set_ylabel(r"$z_{\rm BASIC}-z_{\rm sASM}$"); b.grid(alpha=0.25)
    nseam = int(np.sum(np.diff((seam).astype(int))>0))
    if col==1:
        b.annotate(f"7 seams $\\to$ 7 over-count bumps\n(interior $\\approx$ 0)", xy=(0.5,np.max(diff)*0.9),
                   xytext=(0.18,np.max(diff)*0.55), fontsize=8.8, color=ACC, fontweight="bold",
                   arrowprops=dict(arrowstyle="->", color=ACC, lw=1.3))

fig.suptitle("Fig. 1c  —  over-count at EVERY seam (domain-wide residual): BASIC double-counts each of the 7 overlaps; "
             "sASM normalizes",
             fontsize=11.6, fontweight="bold", color=STRUCT)
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig("fig1c_overcount_global.png", dpi=132)
print("wrote fig1c_overcount_global.png")
