#!/usr/bin/env python3
"""Fig 10: information propagation, quantified. The converged front advances at a
finite speed set by the overlap -> one-level latency grows with #subdomains.
Reads frontprop_profile.txt (O=6 fill-in) and frontprop_front.txt (front vs k)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STRUCT = "#1F3A5F"; ACC = "#8C2D04"; MID = "#2C7FB8"; GREEN = "#117733"

# ---- parse profiles (blocks "# k=K") ----
profs = {}
cur = None
for ln in open("frontprop_profile.txt"):
    ln = ln.rstrip()
    if ln.startswith("# k="):
        cur = int(ln.split("=")[1]); profs[cur] = []
    elif ln and cur is not None:
        x, v = ln.split(); profs[cur].append((float(x), float(v)))
ks = sorted(profs)

fr = np.loadtxt("frontprop_front.txt")   # O k front_subdomains

fig, ax = plt.subplots(1, 2, figsize=(14.2, 5.8))

# ===== (a) the solution crawls in from the boundary =====
a = ax[0]
cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(ks)))
for c, k in zip(cmap, ks):
    arr = np.array(profs[k]); a.plot(arr[:,0], arr[:,1], color=c, lw=2.0, label=f"k={k}")
# exact ramp u* (k -> infinity)
a.plot([0,1],[1,0], color="#999", lw=1.4, ls="--", label="exact $u^*$")
a.set_xlabel("position $x$"); a.set_ylabel("solution $x_k(x)$")
a.set_title("(a) 1D Laplace, left Dirichlet = 1, 12 subdomains, O=6\n"
            "one-level (additive BASIC) solution fills in from the boundary, iteration by iteration",
            fontsize=10.3)
a.set_xlim(0,1); a.set_ylim(-0.02, 1.02); a.grid(alpha=0.3)
a.legend(fontsize=8.4, ncol=2, loc="upper right")

# ===== (b) front position vs iteration: finite speed, grows with overlap =====
b = ax[1]
cols = {2: ACC, 6: MID, 12: STRUCT}
for O in [2,6,12]:
    m = fr[fr[:,0]==O]; k = m[:,1]; f = m[:,2]
    b.plot(k, f, "o-", color=cols[O], ms=4, lw=2.0, label=f"O={O}")
    sel = (k>=2)&(k<=40); slope = np.polyfit(k[sel], f[sel], 1)[0]
    b.text(41, f[k==40][0] if (k==40).any() else f[-1],
           f"{slope:.3f} sub/iter", color=cols[O], fontsize=9, va="center", fontweight="bold")
b.axhline(12, color="#999", ls=":", lw=1.2)
b.text(1.5, 12.4, "couple all 12 subdomains", color="#777", fontsize=8.5)
b.set_xlabel("stationary iteration $k$")
b.set_ylabel("converged front (subdomains reached)")
b.set_xlim(0, 52); b.set_ylim(0, 13)
b.set_title("(b) the converged region grows linearly; speed ∝ overlap\n"
            "stationary iters to couple 12 subdomains: O=2 → 283, O=6 → 137, O=12 → 73",
            fontsize=10.3)
b.legend(fontsize=9, loc="lower right", title="overlap"); b.grid(alpha=0.3)
b.annotate("more overlap →\nfaster propagation\n→ fewer iterations", xy=(38, 5.7), xytext=(20, 9.2),
           fontsize=9, color=STRUCT, ha="center",
           arrowprops=dict(arrowstyle="->", color=STRUCT, lw=1.3))

fig.suptitle("Fig. 10  —  Information propagation, quantified: one-level Schwarz couples the domain "
             "at finite speed (∝ overlap); that latency = the small $\\lambda_{\\min}$ a coarse space removes.",
             fontsize=11.6, fontweight="bold", color=STRUCT)
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig("fig10_frontprop.png", dpi=140)
print("wrote fig10_frontprop.png")
