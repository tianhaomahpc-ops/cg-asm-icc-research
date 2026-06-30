#!/usr/bin/env python3
"""Visualize information propagation in one-level vs two-level domain decomposition.

Reads the per-rank (x y z value) dumps written by asm_demo's -dump_sol probe
(point-source / discrete Green's function, capped at k iterations) and shows:
  (A) the support of the k-th iterate marching outward one subdomain-hop per
      iteration for one-level sASM -- finite "speed of information";
  (B) the support_radius(k) curve: linear for one-level, instantly global for
      the two-level GAMG coarse space (infinite speed).

Usage: python3 plot_propagation.py [DUMP_DIR]
"""
import sys, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

FLOOR = 1e-4   # log color floor; one-level is exactly 0 outside its support ball

DIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/prop"
SRC = np.array([0.5, 0.5, 0.5])

def load(prefix):
    """Concatenate all per-rank files for one (method,k) into one (N,4) array."""
    files = sorted(glob.glob(os.path.join(DIR, prefix + "_r*.txt")))
    if not files:
        return None
    arr = np.concatenate([np.loadtxt(f).reshape(-1, 4) for f in files if os.path.getsize(f) > 0])
    # collapse duplicate (shared-interface) vertices by averaging
    return arr

def radius(arr, tol_frac=1e-3):
    v = np.abs(arr[:, 3]); m = v.max()
    sel = v > tol_frac * m
    d = np.linalg.norm(arr[sel, :3] - SRC, axis=1)
    return d.max() if sel.any() else 0.0

KS = [1, 2, 3, 4, 6, 8, 12]
sasm = {k: load(f"sasm_k{k}") for k in KS}
sasm = {k: a for k, a in sasm.items() if a is not None}
gamg = {k: load(f"gamg_k{k}") for k in (2, 4)}
gamg = {k: a for k, a in gamg.items() if a is not None}

fig = plt.figure(figsize=(15, 9))

# ---- Row 1: y=0.5 slice heatmaps of |u| for one-level sASM at growing k ----
SHOW = [1, 2, 4, 8]
for j, k in enumerate(SHOW):
    ax = fig.add_subplot(2, 4, j + 1)
    a = sasm.get(k)
    if a is None:
        continue
    sl = a[np.abs(a[:, 1] - 0.5) < 0.03]          # y ~ 0.5 slice -> x-z plane
    val = np.abs(sl[:, 3]); val = np.maximum(val / (val.max() + 1e-300), FLOOR)
    sc = ax.scatter(sl[:, 0], sl[:, 2], c=val, s=10, cmap="inferno",
                    norm=LogNorm(vmin=FLOOR, vmax=1), marker="s")
    for xb in np.linspace(0, 1, 9):              # 8 Cartesian slabs along x
        ax.axvline(xb, color="cyan", lw=0.4, alpha=0.5)
    ax.plot(SRC[0], SRC[2], "co", ms=6, mec="white")
    ax.set_title(f"one-level sASM, k={k} iters\nsupport r={radius(a):.3f}", fontsize=10)
    ax.set_xlabel("x"); ax.set_ylabel("z"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")

# ---- Row 2 left: centerline profile |u(x)| at successive k (front marches) ----
ax = fig.add_subplot(2, 4, 5)
for k in KS:
    a = sasm.get(k)
    if a is None:
        continue
    line = a[(np.abs(a[:, 1] - 0.5) < 0.03) & (np.abs(a[:, 2] - 0.5) < 0.03)]
    line = line[np.argsort(line[:, 0])]
    v = np.abs(line[:, 3]); v = v / (v.max() + 1e-300)
    ax.plot(line[:, 0], v, label=f"k={k}")
ax.axvline(SRC[0], color="k", ls=":", lw=1)
ax.set_title("sASM: |u(x)| centerline\n(wavefront advances ~1 slab/iter)", fontsize=10)
ax.set_xlabel("x"); ax.set_ylabel("|u| (norm.)"); ax.legend(fontsize=7, ncol=2)

# ---- Row 2 mid: support_radius(k) -- the speed of information ----
ax = fig.add_subplot(2, 4, 6)
ks = sorted(sasm); rs = [radius(sasm[k]) for k in ks]
ax.plot(ks, rs, "o-", color="C3", label="one-level sASM")
if rs:
    sl, ic = np.polyfit(ks, rs, 1)
    ax.plot(ks, np.array(ks) * sl + ic, "--", color="C3", alpha=0.5,
            label=f"slope={sl:.3f}/iter")
rmax = np.linalg.norm([0.5, 0.5, 0.5])           # center -> corner
ax.axhline(rmax, color="gray", ls=":", label=f"domain diag {rmax:.3f}")
for k, a in gamg.items():
    ax.plot(k, radius(a), "s", color="C0", ms=10,
            label="two-level GAMG" if k == min(gamg) else None)
ax.set_title("speed of information\nlinear (1-level) vs instant (coarse)", fontsize=10)
ax.set_xlabel("iteration k"); ax.set_ylabel("support radius"); ax.legend(fontsize=8)

# ---- Row 2 right: GAMG slice at k=2 (already global) ----
ax = fig.add_subplot(2, 4, 7)
a = gamg.get(2) if 2 in gamg else gamg.get(4)
if a is not None:
    sl = a[np.abs(a[:, 1] - 0.5) < 0.03]
    val = np.abs(sl[:, 3]); val = np.maximum(val / (val.max() + 1e-300), FLOOR)
    ax.scatter(sl[:, 0], sl[:, 2], c=val, s=10, cmap="inferno",
               norm=LogNorm(vmin=FLOOR, vmax=1), marker="s")
    ax.plot(SRC[0], SRC[2], "co", ms=6, mec="white")
ax.set_title("two-level GAMG, k=2  (log scale)\nfaint tail fills WHOLE domain", fontsize=10)
ax.set_xlabel("x"); ax.set_ylabel("z"); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")

# ---- Row 2 far right: text summary ----
ax = fig.add_subplot(2, 4, 8); ax.axis("off")
txt = ("INFORMATION PROPAGATION\n\n"
       "point source (impulse) at center;\n"
       "k-th CG iterate = how far info\n"
       "has spread in k iterations.\n\n"
       "ONE-LEVEL (sASM):\n"
       "  one subdomain-hop / iteration\n"
       "  radius ~ linear in k (finite speed)\n"
       "  -> needs ~D_graph iters to\n"
       "     couple the whole domain\n"
       "     => lambda_min small, slow\n\n"
       "TWO-LEVEL (GAMG coarse space):\n"
       "  global reach in ~1 coarse solve\n"
       "  radius = domain diag immediately\n"
       "  => lambda_min lifted, ~9 iters\n"
       "     independent of #subdomains")
ax.text(0.0, 1.0, txt, va="top", ha="left", fontsize=9, family="monospace")

fig.suptitle("Domain-decomposition convergence as information propagation "
             "(point source, nx=48, 8 slabs)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = os.path.join(DIR, "fig_propagation.png")
fig.savefig(out, dpi=120)
print("wrote", out)
