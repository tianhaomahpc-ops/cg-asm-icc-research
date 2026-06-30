#!/usr/bin/env python3
"""Reach != accuracy: why Sys2 fills the heart at k=1 yet converges slowly.

The point-source support ("where is the iterate nonzero") saturates almost
immediately, but the relative ERROR ("are the values right") decays slowly.
The slow part is the smooth / near-constant mode of the elliptic operator.

Usage: python3 plot_reach_vs_accuracy.py [ERR_DIR]
"""
import sys, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

DIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/err"

def load(pre):
    fs = [f for f in sorted(glob.glob(os.path.join(DIR, pre + "_r*.txt")))
          if os.path.getsize(f) > 0]
    return np.concatenate([np.loadtxt(f).reshape(-1, 4) for f in fs]) if fs else None

# measured (from forward_ecg -propagation, dt=0.01, real mesh, 4 ranks)
ks = [1, 2, 4, 8, 16, 32, 64]
err2 = [0.933, 0.888, 0.778, 0.454, 0.0613, 0.0109, 2.74e-5]
err3 = [0.984, 0.962, 0.894, 0.619, 0.270, 0.00244, 5.8e-7]
sup2 = [10.7] * 7                                  # heart filled at k=1
sup3 = [7.31, 13.92, 22.18, 39.20, 53.39, 53.39, 53.39]

fig = plt.figure(figsize=(15, 5.2))

# Panel 1: Sys2 reach (saturates) vs accuracy (slow)
ax = fig.add_subplot(1, 3, 1)
ax.plot(ks, np.array(sup2) / 10.7, "s-", color="C1", label="reach (support / heart)")
ax.plot(ks, err2, "o-", color="C3", label="error ||u_k - u*|| / ||u*||")
ax.set_xscale("log", base=2); ax.set_xlabel("iteration k")
ax.set_ylabel("normalized"); ax.set_title(
    "Sys2: reach saturates at k=1,\nbut accuracy needs ~50 iters", fontsize=10)
ax.axhline(1, color="gray", lw=0.5); ax.legend(fontsize=8)
ax.annotate("nonzero everywhere\nyet 93% wrong", (1, 0.93), fontsize=8,
            xytext=(1.5, 0.6), textcoords="data",
            arrowprops=dict(arrowstyle="->", color="C3"))

# Panel 2: error decay Sys2 vs Sys3 (both elliptic -> both slow)
ax = fig.add_subplot(1, 3, 2)
ax.semilogy(ks, err2, "o-", color="C3", label="Sys2 u_e (heart, pure-Neumann)")
ax.semilogy(ks, err3, "^-", color="C0", label="Sys3 torso Laplace")
ax.set_xscale("log", base=2); ax.set_xlabel("iteration k")
ax.set_ylabel("relative error (log)")
ax.set_title("both elliptic -> both slow\n(smooth modes decay slowly)", fontsize=10)
ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)

# Panel 3: WHERE the Sys2 error lives at k=8 (the smooth global mode)
ax = fig.add_subplot(1, 3, 3)
a8 = load("k8_sys2"); aref = load("k400_sys2")
if a8 is not None and aref is not None:
    rk = {(round(x, 4), round(y, 4), round(z, 4)): v for x, y, z, v in aref}
    rows = []
    for x, y, z, v in a8:
        c = (round(x, 4), round(y, 4), round(z, 4))
        if c in rk and abs(z) < 0.6:
            rows.append((x, y, v - rk[c]))           # signed error on z~0 slice
    rows = np.array(rows)
    m = np.abs(rows[:, 2]).max()
    sc = ax.scatter(rows[:, 0], rows[:, 1], c=rows[:, 2], s=8, cmap="coolwarm",
                    norm=TwoSlopeNorm(0, -m, m), marker="s")
    plt.colorbar(sc, ax=ax, fraction=0.04, label="error (u_8 - u*)")
ax.set_title("Sys2 error at k=8 (z=0 slice):\nsmooth, domain-wide -> the slow mode",
             fontsize=10)
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_aspect("equal")

fig.suptitle("Reach is not accuracy: the iterate is nonzero everywhere fast, "
             "but the smooth global mode converges slowly", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = os.path.join(DIR, "fig_reach_vs_accuracy.png")
fig.savefig(out, dpi=120)
print("wrote", out)
