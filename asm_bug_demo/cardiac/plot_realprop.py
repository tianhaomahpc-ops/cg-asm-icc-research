#!/usr/bin/env python3
"""Information propagation on the REAL heart/torso geometry (forward_ecg -propagation).

Point-source impulse, k-th CG iterate dumped per system. Real Niederer params,
real conforming tet mesh. Shows the three systems propagate differently because
they are different PDE types -- on the SAME heart mesh, Sys1 (with the dt mass
term) stays local while Sys2 (pure elliptic) fills the heart at once.

Usage: python3 plot_realprop.py [DUMP_DIR]
"""
import sys, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

DIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/realprop"
FLOOR = 1e-4

def load(prefix):
    fs = [f for f in sorted(glob.glob(os.path.join(DIR, prefix + "_r*.txt")))
          if os.path.getsize(f) > 0]
    return np.concatenate([np.loadtxt(f).reshape(-1, 4) for f in fs]) if fs else None

def slice_xy(a, z0=0.0, dz=0.6):
    return a[np.abs(a[:, 2] - z0) < dz]

def panel(ax, a, src, title, ext, dz=0.6):
    if a is None:
        ax.set_title(title + " (no data)"); return
    sl = slice_xy(a, dz=dz)
    v = np.abs(sl[:, 3]); v = np.maximum(v / (v.max() + 1e-300), FLOOR)
    ax.scatter(sl[:, 0], sl[:, 1], c=v, s=6, cmap="inferno",
               norm=LogNorm(vmin=FLOOR, vmax=1), marker="s")
    ax.plot(src[0], src[1], "co", ms=7, mec="white")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_xlim(-ext, ext); ax.set_ylim(-ext, ext); ax.set_aspect("equal")

fig = plt.figure(figsize=(16, 8))
HEART_EXT, TORSO_EXT = 11, 26

# Row 1: spatial slices (z=0 plane)
panel(fig.add_subplot(2, 4, 1), load("k1_sys1"), (0, 0),
      "Sys1 monodomain (heart), k=1\nlocal blob at source", HEART_EXT)
panel(fig.add_subplot(2, 4, 2), load("k8_sys1"), (0, 0),
      "Sys1 (heart), k=8 (converged)\nSTILL local (r~1.9mm of 20mm)", HEART_EXT)
panel(fig.add_subplot(2, 4, 3), load("k1_sys2"), (0, 0),
      "Sys2 u_e (heart, pure-Neumann), k=1\nfills WHOLE heart at once", HEART_EXT)
panel(fig.add_subplot(2, 4, 4), load("k8_sys3"), (15, 0),
      "Sys3 torso Laplace, k=8\nspreading across 50mm torso", TORSO_EXT, dz=2.0)

# Row 2 left: support radius vs k (the quantitative story)
ax = fig.add_subplot(2, 4, 5)
ks = [1, 2, 4, 8]
r1 = [1.576, 1.660, 1.905, 1.905]
r2 = [10.70, 10.70, 10.70, 10.70]
r3 = [7.305, 13.92, 22.18, 39.20]
ax.plot(ks, r1, "o-", color="C2", label="Sys1 heart (local, saturates)")
ax.plot(ks, r2, "s-", color="C3", label="Sys2 heart (global at once)")
ax.plot(ks, r3, "^-", color="C0", label="Sys3 torso (marches across)")
ax.axhline(10.7, color="C3", ls=":", lw=1, alpha=0.6, label="heart half-size ~10.7mm")
ax.axhline(43.3, color="C0", ls=":", lw=1, alpha=0.6, label="torso half-size ~43mm")
ax.set_title("support radius vs iterations (dt=0.01, real mesh)", fontsize=10)
ax.set_xlabel("iteration k"); ax.set_ylabel("info reach (mm)")
ax.legend(fontsize=7.5); ax.set_ylim(0, 46)

# Row 2 right: text
ax = fig.add_subplot(2, 4, (6, 8)); ax.axis("off")
txt = ("REAL HEART/TORSO, dt=0.01, Niederer params\n\n"
       "Sys1  monodomain  (1/dt)M + (1/2)K   [heart]\n"
       "  the dt mass term SCREENS the coupling.\n"
       "  info reach saturates at ~1.9 mm (heart is 20 mm)\n"
       "  -> LOCAL: a poke only affects nearby tissue,\n"
       "     because one time-step diffuses ~sqrt(D*dt) mm.\n"
       "  -> converges in very few iterations.\n\n"
       "Sys2  u_e recovery  K (pure Neumann)  [SAME heart mesh]\n"
       "  no time term -> instantaneous electrostatics.\n"
       "  reach = 10.7 mm = WHOLE heart already at k=1.\n"
       "  -> GLOBAL. plus a domain-wide constant (the zero-\n"
       "     mean level) that is the slowest thing to settle\n"
       "     -> needs the most iterations / a coarse space.\n\n"
       "Sys3  torso Laplace  K  [torso]\n"
       "  also elliptic, no time term: reach MARCHES out\n"
       "  7 -> 14 -> 22 -> 39 mm, crossing the 50 mm torso\n"
       "  one subdomain-hop per iteration -> many iterations.\n\n"
       "KEY: Sys1 vs Sys2 are on the SAME mesh/geometry;\n"
       "the ONLY difference is Sys1's dt mass term. That\n"
       "alone turns global (elliptic) into local (screened).")
ax.text(0.0, 1.0, txt, va="top", ha="left", fontsize=9, family="monospace")

fig.suptitle("Information propagation on the real heart/torso mesh: "
             "Sys1 local (screened by dt) vs Sys2/Sys3 global (elliptic)",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(DIR, "fig_realprop.png")
fig.savefig(out, dpi=120)
print("wrote", out)
