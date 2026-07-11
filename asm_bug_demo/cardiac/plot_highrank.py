#!/usr/bin/env python3
"""plot_highrank.py -- WHY the moving depolarization front makes Sys3's RHS
trajectory high-rank (which is why -leadvol full-field superposition failed).

WHO is high-rank: the sequence of right-hand sides b(t) = interface data u_e|Gamma(t)
(hence the solution snapshots phi(t)=Kt^-1 b(t)).  NOT the operator Kt (constant).

Illustrative model (numbers in panels B/C are REAL: computed Gram + SVD of the
synthetic snapshots, not hand-drawn):
  - moving front: b_k(x) = localized bump centered at x = v*t_k (the wavefront region
    that "lights up" the interface).  Non-overlapping bumps => near-orthogonal
    snapshots => slowly-decaying singular values => HIGH rank.  This is the
    canonical "translation/advection is not low-rank" (large Kolmogorov n-width).
  - contrast: fixed shape, varying amplitude  b_k(x) = a(t_k)*f(x)  => rank 1 (cliff).

Anchored to the measured -leadvol result: 80 EP steps grew a basis of ~54
directions (rel-drop 1e-3), i.e. empirical rank ~54, not the ~8 we hoped.
Output: fig_highrank.png
"""
import matplotlib, numpy as np
matplotlib.use("Agg")
from matplotlib import font_manager
try:
    font_manager.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
    matplotlib.rcParams["font.family"] = "WenQuanYi Zen Hei"
except Exception: pass
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

# ---- synthetic interface data: a narrow front bump translating across the interface
Nx = 400                      # interface DOFs (1D proxy)
x  = np.linspace(0, 1, Nx)
Nt = 80                       # EP steps (matches the measured run)
w  = 0.022                    # front width (narrow; controls overlap => rank)
centers = np.linspace(0.08, 0.92, Nt)   # front sweeps across
B = np.zeros((Nx, Nt))        # each column = one step's RHS b(t_k)
for k, c in enumerate(centers):
    B[:, k] = np.exp(-0.5*((x - c)/w)**2)   # localized wavefront bump
# normalize columns (compare directions, like the leadvol relative-drop test)
B /= np.linalg.norm(B, axis=0, keepdims=True)

# contrast: fixed shape, only amplitude varies -> rank 1
f0 = np.exp(-0.5*((x - 0.5)/0.18)**2)
Bfix = np.outer(f0, 0.4 + 0.6*np.abs(np.sin(np.linspace(0, 3, Nt))))
Bfix /= np.linalg.norm(Bfix, axis=0, keepdims=True)

# REAL Gram matrix + SVD
G  = B.T @ B
sv_move = np.linalg.svd(B, compute_uv=False)
sv_fix  = np.linalg.svd(Bfix, compute_uv=False)
# effective rank at rel-drop 1e-3 (same idea as the leadvol basis threshold)
eff_move = int(np.sum(sv_move/sv_move[0] > 1e-3))
eff_fix  = int(np.sum(sv_fix /sv_fix[0]  > 1e-3))

fig = plt.figure(figsize=(16.4, 5.3))
gs = fig.add_gridspec(1, 3, width_ratios=[1.06, 0.9, 1.08], wspace=0.32)
fig.suptitle("为什么『移动前沿』让 Sys3 右端项高秩(=为什么 -leadvol 全场叠加失败)——"
             "高秩的是 b(t) 那一叠,不是恒定算子 K_t",
             fontsize=12.4, weight="bold", y=1.02)

# (A) translating front snapshots -- visibly non-overlapping
axA = fig.add_subplot(gs[0])
show = range(0, Nt, 8)
cmap = plt.cm.viridis(np.linspace(0, 1, len(list(show))))
for j, k in enumerate(show):
    axA.plot(x, B[:, k], color=cmap[j], lw=1.8)
axA.set_title("① 窄前沿在界面上平移(几帧):\n错开的帧几乎不重叠 → 两两近正交",
              fontsize=10.3, weight="bold")
axA.set_xlabel("界面位置 x"); axA.set_ylabel("b(t) 剖面(单位化)")
axA.grid(alpha=0.2)
axA.annotate("每挪一步点亮一批新自由度\n= 一个旧帧覆盖不了的新方向", xy=(0.5, 0.9),
             xytext=(0.12, 0.55), fontsize=8.4, color="#c0392b",
             arrowprops=dict(arrowstyle="->", color="#c0392b"))

# (B) Gram matrix -- near-diagonal => mutually orthogonal => high rank
axB = fig.add_subplot(gs[1])
im = axB.imshow(np.abs(G), cmap="magma", origin="lower", vmin=0, vmax=1)
axB.set_title("② Gram 矩阵 |<b_i, b_j>| 近乎对角\n(非对角≈0 = 两两正交 = 满秩)",
              fontsize=10.3, weight="bold")
axB.set_xlabel("步 j"); axB.set_ylabel("步 i")
fig.colorbar(im, ax=axB, fraction=0.046, pad=0.04)

# (C) singular-value spectrum: moving front (slow) vs fixed shape (cliff)
axC = fig.add_subplot(gs[2])
axC.semilogy(sv_move/sv_move[0], "o-", ms=3, color="#c0392b", lw=1.6,
             label=f"移动前沿(行波):慢衰减,有效秩≈{eff_move}")
axC.semilogy(sv_fix/sv_fix[0], "s-", ms=3, color="#27ae60", lw=1.6,
             label=f"固定形状·变幅:悬崖,秩={eff_fix}")
axC.axhline(1e-3, color="0.5", ls="--", lw=1, label="相对阈值 1e-3(leadvol 判据)")
axC.set_title("③ 奇异值谱:行波慢衰减 → 需要几十个基\n"
              "(实测 -leadvol:80 步长出 ~54 个方向,不是 ~8)",
              fontsize=10.3, weight="bold")
axC.set_xlabel("奇异值序号"); axC.set_ylabel("sigma / sigma_max(对数)")
axC.set_ylim(1e-6, 2); axC.grid(alpha=0.25, which="both")
from matplotlib.ticker import FuncFormatter, LogLocator
axC.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
axC.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"1e{int(round(np.log10(v)))}"))
axC.legend(fontsize=8.3, loc="lower left")

fig.savefig("fig_highrank.png", dpi=140, bbox_inches="tight")
print("wrote fig_highrank.png")
print(f"moving-front effective rank (rel-drop 1e-3): {eff_move}  (of {Nt} steps)")
print(f"fixed-shape effective rank                : {eff_fix}")
