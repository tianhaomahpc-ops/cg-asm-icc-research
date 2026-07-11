#!/usr/bin/env python3
"""plot_sys3_opt.py -- how to optimize Sys3 (torso), measured np=8, -fischer3.
Two findings:
 (A) initial-guess tricks are CAPPED for Sys3: warm(-9%) ~ Fischer(-11%) are close
     -- the subspace barely beats a single vector -- because the leftover error is
     in the slow modes regardless of the start.  (Contrast Sys2: warm -5% << Fischer
     -82%, where the subspace IS the win.)  So A1/A2 top out ~10%.
 (B) attack the OPERATOR instead: GAMG cuts Sys3 69->13 iters/step (5.3x); its
     per-iteration cost is ~4x bjacobi+ICC, but 5.3x fewer iters => NET WALL-TIME
     LOWER (3.2s vs 4.1s) even at np=8.  This is the real Sys3 lever.
Output: fig_sys3_opt.png
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

# measured totals over 80 steps
bj_cold, bj_warm, bj_fis, bj_wall = 5544, 4993, 4922, 4.1
gm_cold, gm_warm, gm_fis, gm_wall = 1040, 914, 901, 3.2
NST = 80

fig = plt.figure(figsize=(15.6, 5.8))
gs = fig.add_gridspec(1, 3, width_ratios=[1.05,1.0,1.0], wspace=0.34)
fig.suptitle("Sys3(躯干)怎么优化:初值有上限(左),要治就治算子——GAMG(中/右) 实测 np=8",
             fontsize=13, weight="bold", y=1.0)

# (A) guess ceiling: Sys3 warm~Fischer vs Sys2 warm<<Fischer
axA = fig.add_subplot(gs[0])
groups = ["Sys2\n(u_e)", "Sys3\n(躯干)"]
warm = [5, 9]; fis = [82, 11]
x = np.arange(2); w=0.36
axA.bar(x-w/2, warm, w, color="#e67e22", label="warm(单向量)")
axA.bar(x+w/2, fis, w, color="#27ae60", label="Fischer(子空间)")
for i,(a,b) in enumerate(zip(warm,fis)):
    axA.text(i-w/2, a+1.5, f"-{a}%", ha="center", fontsize=9.5, weight="bold")
    axA.text(i+w/2, b+1.5, f"-{b}%", ha="center", fontsize=9.5, weight="bold")
axA.set_xticks(x); axA.set_xticklabels(groups); axA.set_ylabel("迭代下降(相对冷启动)")
axA.set_title("① 初值的上限:Sys2 子空间大赢单向量(-82 vs -5),\n"
              "Sys3 两者接近(-11 vs -9)→ 初值封顶~10%", fontsize=10.3, weight="bold")
axA.legend(fontsize=9); axA.grid(axis="y", alpha=0.25); axA.set_ylim(0,92)
axA.annotate("差距大=子空间是主角\n(该用回收)", xy=(0,82), xytext=(-0.35,55), fontsize=8.2,
             color="#1e8449", arrowprops=dict(arrowstyle="->",color="#1e8449"))
axA.annotate("差距小=初值没用\n(该治算子)", xy=(1,11), xytext=(0.72,40), fontsize=8.2,
             color="#c0392b", arrowprops=dict(arrowstyle="->",color="#c0392b"))

# (B) iterations per step: bjacobi vs GAMG
axB = fig.add_subplot(gs[1])
labels=["bjacobi+ICC\n(冷)","GAMG\n(冷)"]
vals=[bj_cold/NST, gm_cold/NST]
axB.bar(labels, vals, color=["#e67e22","#2980b9"], edgecolor="0.3")
for i,v in enumerate(vals): axB.text(i, v+1, f"{v:.0f}", ha="center", fontsize=12, weight="bold")
axB.set_ylabel("Sys3 每步迭代数")
axB.set_title(f"② GAMG 迭代 5.3× 少:69 → 13/步\n(治那 8 个慢模,不靠初值)", fontsize=10.3, weight="bold")
axB.grid(axis="y", alpha=0.25); axB.set_ylim(0,80)

# (C) the cost tradeoff: per-iter vs total wall
axC = fig.add_subplot(gs[2])
per_bj = bj_wall/bj_cold*1000; per_gm = gm_wall/gm_cold*1000   # ms/iter
x=np.arange(2); w=0.36
b1=axC.bar(x-w/2, [per_bj, per_gm], w, color="#95a5a6", label="每迭代 ms")
axC2=axC.twinx()
b2=axC2.bar(x+w/2, [bj_wall, gm_wall], w, color="#16a085", label="总墙钟 s")
axC.set_xticks(x); axC.set_xticklabels(["bjacobi+ICC","GAMG"])
axC.set_ylabel("每迭代耗时 (ms)"); axC2.set_ylabel("Sys3 总墙钟 (s)")
axC.text(0-w/2, per_bj+0.05, f"{per_bj:.2f}", ha="center", fontsize=9, weight="bold")
axC.text(1-w/2, per_gm+0.05, f"{per_gm:.2f}", ha="center", fontsize=9, weight="bold")
axC2.text(0+w/2, bj_wall+0.05, f"{bj_wall}", ha="center", fontsize=9, weight="bold", color="#0e6655")
axC2.text(1+w/2, gm_wall+0.05, f"{gm_wall}", ha="center", fontsize=9, weight="bold", color="#0e6655")
axC.set_title(f"③ 成本权衡:GAMG 每迭代贵 {per_gm/per_bj:.1f}×,\n但少 5.3× 迭代 → 总墙钟反而更快(3.2<4.1)",
              fontsize=10.3, weight="bold")
axC.set_ylim(0, per_gm*1.3); axC2.set_ylim(0, 5.2)
lines=[b1,b2]; axC.legend(lines,[l.get_label() for l in lines], fontsize=8.5, loc="upper left")

fig.savefig("fig_sys3_opt.png", dpi=140, bbox_inches="tight")
print("wrote fig_sys3_opt.png")
print(f"per-iter: bjacobi {per_bj:.2f} ms, GAMG {per_gm:.2f} ms ({per_gm/per_bj:.1f}x)")
print(f"iters/step: {bj_cold/NST:.0f} -> {gm_cold/NST:.0f} ({bj_cold/gm_cold:.1f}x fewer)")
print(f"wall: {bj_wall}s -> {gm_wall}s ({bj_wall/gm_wall:.2f}x faster)")
