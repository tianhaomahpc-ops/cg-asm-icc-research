#!/usr/bin/env python3
"""How the three forward-ECG systems are ANCHORED, and how each is imposed.

Concrete, code-level demonstration on the real heart/torso mesh (dt=0.01):
  - Sys3 (Dirichlet): interface node rows replaced by identity, value = transferred u_e
  - Sys2 (pure Neumann): NO boundary anchor; the constant is pinned by the
    nullspace projection (MatSetNullSpace) -> PETSc subtracts the mean every
    iteration.  Measured: with the anchor ue_mean ~1e-15; without it ue_mean
    DRIFTS to -0.3 mV.  The anchor is imposed fresh each solve by a global
    average (MPI_Allreduce), NOT inherited from the previous time step.

Usage: python3 plot_anchor.py [ANCHOR_DIR]
"""
import sys, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import TwoSlopeNorm

DIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/anchor"

def load_xyzv(pre):
    fs = [f for f in sorted(glob.glob(os.path.join(DIR, pre + "_r*.txt")))
          if os.path.getsize(f) > 0]
    return np.concatenate([np.loadtxt(f).reshape(-1, 4) for f in fs]) if fs else None

on = np.loadtxt(os.path.join(DIR, "mean_on.txt")).reshape(-1, 2)
off = np.loadtxt(os.path.join(DIR, "mean_off.txt")).reshape(-1, 2)

fig = plt.figure(figsize=(16, 9))

# ============ Panel A: schematic of the 3 anchoring mechanisms ============
ax = fig.add_subplot(2, 2, 1); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.set_title("How each system's level is anchored (and how it is imposed)",
             fontsize=11, loc="left")

# Sys3 Dirichlet
ax.add_patch(FancyBboxPatch((0.3, 7.2), 2.3, 1.9, boxstyle="round,pad=0.05",
             fc="#cfe8ff", ec="C0"))
ax.plot([0.3, 0.3], [7.2, 9.1], color="red", lw=4)           # the pinned boundary
ax.text(1.45, 8.15, "Sys3\ntorso", ha="center", va="center", fontsize=9)
ax.text(2.9, 8.5, "DIRICHLET: the interface node rows are\n"
        "replaced by identity (row=1 on diag), RHS=u_e.\n"
        "=> potential pinned to the transferred value.\n"
        "Imposed by: FormLinearSystem(ess_tdofs).",
        va="center", fontsize=8.5)

# Sys2 zero-mean projection
ax.add_patch(FancyBboxPatch((0.3, 4.0), 2.3, 1.9, boxstyle="round,pad=0.05",
             fc="#ffd9d9", ec="C3"))
ax.text(1.45, 4.95, "Sys2\nheart u_e", ha="center", va="center", fontsize=9)
ax.text(2.9, 5.3, "PURE NEUMANN: no boundary anchor. The free\n"
        "constant is pinned by the NULLSPACE PROJECTION:\n"
        "every CG iteration  u <- u - (1/N) sum(u)  (=mean 0).\n"
        "Imposed by: MatSetNullSpace + MPI_Allreduce mean.\n"
        "Fresh each solve -- NOT from the previous step.",
        va="center", fontsize=8.5)

# Warm-start (secondary)
ax.add_patch(FancyBboxPatch((0.3, 0.9), 2.3, 1.7, boxstyle="round,pad=0.05",
             fc="#e8e8e8", ec="gray"))
ax.text(1.45, 1.75, "initial\nguess x0", ha="center", va="center", fontsize=9)
ax.text(2.9, 2.05, "WARM-START (optional, speeds up, does NOT set\n"
        "the level): x0 = u_e(previous step).\n"
        "Imposed by: cg2.iterative_mode=true.\n"
        "Only affects the SOURCE/guess, not the anchor.",
        va="center", fontsize=8.5)

# ============ Panel B: the proof -- ue_mean vs time, anchor ON vs OFF ======
ax = fig.add_subplot(2, 2, 2)
ax.plot(off[:, 0], off[:, 1], "o-", color="C3", label="anchor OFF (no nullspace): DRIFTS")
ax.plot(on[:, 0], on[:, 1], "s-", color="C2", label="anchor ON (nullspace): pinned ~0")
ax.axhline(0, color="k", lw=0.5)
ax.set_title("Proof: the nullspace projection IS the anchor\n"
             "(mean of u_e over the heart vs time)", fontsize=10)
ax.set_xlabel("time (ms)"); ax.set_ylabel("mean(u_e) over heart (mV)")
ax.legend(fontsize=9)
ax.annotate("level wanders, undetermined\n(~ -0.1 to -0.3 mV)", (6, -0.27),
            fontsize=8, color="C3")
ax.annotate("machine zero (1e-15)", (6, 0.02), fontsize=8, color="C2")

# ============ Panel C: the anchored u_e field (converged, zero-mean) =======
ax = fig.add_subplot(2, 2, 3)
a = load_xyzv("ue_sys2")
if a is not None:
    sl = a[np.abs(a[:, 2]) < 0.6]                 # z~0 slice (heart x-y plane)
    v = sl[:, 3]; m = np.abs(v).max()
    sc = ax.scatter(sl[:, 0], sl[:, 1], c=v, s=10, cmap="coolwarm",
                    norm=TwoSlopeNorm(0, -m, m), marker="s")
    plt.colorbar(sc, ax=ax, fraction=0.04, label="u_e (mV)")
ax.set_title("The anchored solution (converged u_e, z=0 slice):\n"
             "a zero-mean dipole -- only DIFFERENCES are physical", fontsize=10)
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_aspect("equal")

# ============ Panel D: code-level summary =================================
ax = fig.add_subplot(2, 2, 4); ax.axis("off")
txt = ("WHERE THE ANCHOR COMES FROM (code-level)\n\n"
       "Sys2 (pure Neumann) -- forward_ecg.cpp:\n"
       "  AttachConstNullSpace(Kiep)   // MatSetNullSpace, ker={1}\n"
       "  ...each time step:\n"
       "    b2 = -Ki * Vm                  // SOURCE from current Vm\n"
       "    RemoveGlobalMean(b2)           // compatibility (redundant:\n"
       "                                   //  PETSc's nullspace does it too)\n"
       "    cg2.Mult(b2, ue_h)             // CG; nullspace projected each iter\n\n"
       "MEASURED (dt=0.01, 4 ranks):\n"
       "  anchor ON : ue_mean ~ 1e-15 every step (pinned)\n"
       "  anchor OFF: ue_mean 0 -> -0.10 -> -0.18 -> -0.30 (drifts)\n"
       "  Sys2 iters ~83-87 both, FLAT in time (no warm-start benefit;\n"
       "   toggling iterative_mode did not change the count here)\n\n"
       "ANSWER to 'is the level inherited from the previous step?':\n"
       "  NO. The level is set FRESH each solve by the global mean\n"
       "  projection (MPI_Allreduce). The previous step enters only via\n"
       "  the RHS (-Ki*Vm) and an optional warm-start guess -- neither\n"
       "  determines the constant. Physically the absolute level of u_e\n"
       "  is meaningless (ECG reads only potential DIFFERENCES).")
ax.text(0.0, 1.0, txt, va="top", ha="left", fontsize=8.6, family="monospace")

fig.suptitle("Anchoring the pure-Neumann system: the zero-mean nullspace "
             "projection (imposed fresh each solve, not inherited in time)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(DIR, "fig_anchor.png")
fig.savefig(out, dpi=120)
print("wrote", out)
