#!/usr/bin/env python3
"""How does BASIC's correction z change exact -> ICC (2D, where ICC != exact)?
z_exact = big smooth domes (A^-1 amplifies the smooth small-eigenvalue mode);
z_icc = much weaker (ICC lacks that amplification -> undershoots the smooth mode).
Reads zfield_{exact,icc,diff,mult}.txt (64x64)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ze = np.loadtxt("zfield_exact.txt"); zi = np.loadtxt("zfield_icc.txt")
df = np.loadtxt("zfield_diff.txt");  mu = np.loadtxt("zfield_mult.txt")
n = ze.shape[0]; seams=[16,32,48]; row=32

fig, ax = plt.subplots(1, 4, figsize=(16.0, 4.2))

vmax = ze.max()
for a,(F,t) in zip(ax[:2], [(ze,"$z$ BASIC, EXACT\n(smooth domes, $\\|z\\|$=1458)"),
                            (zi,"$z$ BASIC, ICC(0)\n(undershoots smooth, $\\|z\\|$=142)")]):
    im=a.imshow(F, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
    for s in seams: a.axvline(s,color="w",lw=0.4,ls=":",alpha=0.5);a.axhline(s,color="w",lw=0.4,ls=":",alpha=0.5)
    a.axhline(row,color="r",lw=1.0,ls="--",alpha=0.7)
    a.set_title(t,fontsize=9.5); a.set_xticks([]); a.set_yticks([])
fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)

a=ax[2]
im=a.imshow(df, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
for s in seams: a.axvline(s,color="k",lw=0.4,ls=":",alpha=0.4);a.axhline(s,color="k",lw=0.4,ls=":",alpha=0.4)
a.set_title("$z_{\\rm ICC}-z_{\\rm exact}$\n(the inexactness change, $\\approx$91%)",fontsize=9.5)
a.set_xticks([]); a.set_yticks([]); fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)

b=ax[3]
b.plot(ze[row], color="#1F3A5F", lw=2.2, label="exact (Cholesky)")
b.plot(zi[row], color="#E08214", lw=2.2, label="ICC(0)")
for s in seams: b.axvline(s,color="#8C2D04",lw=0.8,ls=":",alpha=0.6)
b.set_title(f"horizontal slice (row {row}):\nICC undershoots the smooth response",fontsize=9.5)
b.set_xlabel("grid column"); b.set_ylabel("$z$"); b.legend(fontsize=9); b.grid(alpha=0.3)

fig.suptitle("BASIC's correction $z=M^{-1}r$, exact vs ICC subdomain solve (2D Laplace $64^2$, $4{\\times}4$, $O{=}2$, $r{=}1$). "
             "In 1D ICC$=$exact $\\Rightarrow$ z identical; in 2D ICC undershoots the smooth (small-$\\lambda$) mode.",
             fontsize=10.6, fontweight="bold", color="#1F3A5F", y=1.02)
fig.tight_layout()
fig.savefig("fig_zfield_exact_vs_icc.png", dpi=120, bbox_inches="tight")
print("wrote fig_zfield_exact_vs_icc.png")
