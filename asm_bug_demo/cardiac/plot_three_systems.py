#!/usr/bin/env python3
"""Information-propagation FORM of the three forward-ECG systems.

Same point-source / discrete-Green's-function probe as plot_propagation.py,
but comparing the CONVERGED Green's function of each system:

  Sys1  monodomain  (1/dt)M + (1/2)K  -> SCREENED Helmholtz, Green ~ exp(-r/l),
                                          l~sqrt(dt): SHORT-RANGE / LOCAL
  Sys3  torso Laplace  K (+ground pin) -> elliptic, Green ~ 1/r: LONG-RANGE / GLOBAL
  Sys2  u_e recovery   K (pure Neumann)-> elliptic 1/r + global constant (nullspace):
                                          LONGEST-RANGE (whole domain + DC mode)

The propagation RANGE (support of the converged Green's function) sets how far
information must travel -> how many iterations a one-level method needs.

Usage: python3 plot_three_systems.py [DUMP_DIR]
"""
import sys, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

DIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/prop"
SRC = np.array([0.5, 0.5, 0.5])
FLOOR = 1e-4

def load(prefix):
    files = sorted(glob.glob(os.path.join(DIR, prefix + "_r*.txt")))
    files = [f for f in files if os.path.getsize(f) > 0]
    if not files:
        return None
    return np.concatenate([np.loadtxt(f).reshape(-1, 4) for f in files])

def decay(arr, nb=24):
    """mean |u| in radial bins -> (r, |u|) decay profile."""
    r = np.linalg.norm(arr[:, :3] - SRC, axis=1)
    v = np.abs(arr[:, 3]); v = v / (v.max() + 1e-300)
    edges = np.linspace(0, r.max(), nb + 1)
    rc, vc = [], []
    for i in range(nb):
        sel = (r >= edges[i]) & (r < edges[i + 1])
        if sel.sum() > 3:
            rc.append(0.5 * (edges[i] + edges[i + 1])); vc.append(v[sel].mean())
    return np.array(rc), np.array(vc)

SYS = [("sys1_dt0.001", "Sys1 monodomain\n(1/dt)M+1/2K, dt=1e-3", 12, 0.0955),
       ("sys3",         "Sys3 torso Laplace\nK (non-singular)",   102, 0.866),
       ("sys2",         "Sys2 u_e recovery\nK pure-Neumann (sing.)", 88, 0.866)]
data = {p: load(p) for p, _, _, _ in SYS}

fig = plt.figure(figsize=(15, 9))

# ---- Row 1: y=0.5 slice of the converged Green's function (log scale) ----
for j, (p, title, it, rng) in enumerate(SYS):
    ax = fig.add_subplot(2, 3, j + 1)
    a = data[p]
    if a is None:
        ax.set_title(title + " (no data)"); continue
    sl = a[np.abs(a[:, 1] - 0.5) < 0.03]
    val = np.abs(sl[:, 3]); val = np.maximum(val / (val.max() + 1e-300), FLOOR)
    ax.scatter(sl[:, 0], sl[:, 2], c=val, s=10, cmap="inferno",
               norm=LogNorm(vmin=FLOOR, vmax=1), marker="s")
    th = np.linspace(0, 2*np.pi, 100)
    ax.plot(SRC[0] + rng*np.cos(th), SRC[2] + rng*np.sin(th), "c--", lw=1.2,
            label=f"range={rng:.2f}")
    ax.plot(SRC[0], SRC[2], "co", ms=6, mec="white")
    ax.set_title(f"{title}\nconverged in {it} iters", fontsize=10)
    ax.set_xlabel("x"); ax.set_ylabel("z"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal"); ax.legend(fontsize=8, loc="upper right")

# ---- Row 2 left: radial decay |u|(r) on semilog-y ----
ax = fig.add_subplot(2, 3, 4)
col = {"sys1_dt0.001": "C2", "sys3": "C0", "sys2": "C3"}
lab = {"sys1_dt0.001": "Sys1 (exp decay, local)",
       "sys3": "Sys3 (1/r, global)", "sys2": "Sys2 (1/r+const, global)"}
for p, *_ in SYS:
    a = data[p]
    if a is None:
        continue
    r, v = decay(a)
    ax.semilogy(r, np.maximum(v, 1e-5), "o-", color=col[p], ms=3, label=lab[p])
ax.set_title("Green's function decay |u|(r)\nexp (straight) = local; 1/r (concave) = global",
             fontsize=10)
ax.set_xlabel("distance from source r"); ax.set_ylabel("|u| (norm., log)")
ax.legend(fontsize=8); ax.set_ylim(1e-5, 1.5)

# ---- Row 2 mid: Sys1 dt -> range -> iters ----
ax = fig.add_subplot(2, 3, 5)
dts = [0.05, 0.02, 0.005, 0.001, 0.0002]
its = [62, 47, 26, 12, 7]
rng = [0.5225, 0.3125, 0.1840, 0.0955, 0.0722]
ax.plot(rng, its, "o-", color="C2", label="Sys1 (vary dt)")
for dt, it, r in zip(dts, its, rng):
    ax.annotate(f"dt={dt}", (r, it), fontsize=6,
                xytext=(3, 3), textcoords="offset points")
ax.plot(0.866, 102, "s", color="C0", ms=10, label="Sys3 (global)")
ax.plot(0.866, 88, "D", color="C3", ms=10, label="Sys2 (global)")
ax.set_title("propagation range -> iterations\n(smaller range = fewer iters)", fontsize=10)
ax.set_xlabel("Green's function range"); ax.set_ylabel("CG iterations"); ax.legend(fontsize=8)

# ---- Row 2 right: text summary ----
ax = fig.add_subplot(2, 3, 6); ax.axis("off")
txt = ("THREE SYSTEMS, THREE PROPAGATION FORMS\n\n"
       "Sys1  monodomain (parabolic step)\n"
       "  A = (1/dt)M + (1/2)K  -- mass term\n"
       "  = screened Helmholtz\n"
       "  Green ~ exp(-r/l),  l ~ sqrt(dt)\n"
       "  -> SHORT-RANGE / LOCAL\n"
       "  -> few iters (7-12 @ cardiac dt)\n"
       "  dt is the knob: smaller dt =>\n"
       "  shorter range => fewer iters\n\n"
       "Sys3  torso Laplace (elliptic)\n"
       "  A = K (+ ground pin)\n"
       "  Green ~ 1/r  -> LONG-RANGE / GLOBAL\n"
       "  -> many iters (~100), needs coarse sp.\n\n"
       "Sys2  u_e recovery (elliptic, singular)\n"
       "  A = K, pure Neumann, ker=span{1}\n"
       "  Green ~ 1/r + GLOBAL CONSTANT\n"
       "  -> LONGEST-RANGE (whole domain +\n"
       "     infinite-wavelength DC mode)\n"
       "  -> coarse space ESSENTIAL")
ax.text(0.0, 1.0, txt, va="top", ha="left", fontsize=8.5, family="monospace")

fig.suptitle("Information-propagation form differs by PDE type: "
             "Sys1 local (screened) vs Sys2/Sys3 global (elliptic)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = os.path.join(DIR, "fig_three_systems.png")
fig.savefig(out, dpi=120)
print("wrote", out)
