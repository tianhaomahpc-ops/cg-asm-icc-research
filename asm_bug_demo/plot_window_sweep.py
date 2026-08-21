#!/usr/bin/env python3
"""plot_window_sweep.py -- how big should each system's recycling window be?

Same true sliding window everywhere, only m changes.  Two curves per system:
  solid  = measured CG iterations,
  dashed = iterations plus what maintaining the window itself costs, in units of
           one CG iteration:

     CG iteration    ~ 1 matvec + 1 PC + 2 dots + 3 axpy   ~ 24 N flops
     window step     ~ 2m dots + m axpy + 1 matvec         ~ (6m + 13) N flops
     => window costs (6m+13)/24 ~ 0.25 m + 0.54 iterations per step

The dot products batch into O(1) reductions, so this ratio is the same per rank
at any scale -- it is local flops, not communication.

Usage: python3 plot_window_sweep.py sweep_r0.csv sweep_r2.csv fig_window_sweep.png
"""
import sys, re
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
try:
    font_manager.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
    matplotlib.rcParams["font.family"] = "WenQuanYi Zen Hei"
except Exception:
    pass
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np

r0, r2, out = sys.argv[1], sys.argv[2], sys.argv[3]
MS = [4, 8, 12, 16, 24, 32, 48, 64]

def load(path):
    hdr, rows = None, []
    for line in open(path):
        if line.startswith("#"):
            continue
        if hdr is None:
            hdr = line.strip().split(",");  continue
        p = line.strip().split(",")
        if len(p) == len(hdr):
            rows.append([int(v) for v in p])
    d = {k: np.array([r[i] for r in rows]) for i, k in enumerate(hdr)}
    out = {}
    for s in (1, 2, 3):
        cold = d[f"s{s}_cold"]
        nsolved = int((cold > 0).sum())          # steps where the RHS was not ~0
        out[s] = dict(cold=int(cold.sum()), n=nsolved,
                      tot=[int(d[f"s{s}_m{m}"].sum()) for m in MS])
    return out

d0, d2 = load(r0), load(r2)
COL = {1: "#8e44ad", 2: "#c0392b", 3: "#16a085"}
NAME = {1: "Sys1 单域(质量主导)", 2: "Sys2 $u_e$(奇异)", 3: "Sys3 躯干(Dirichlet 界面)"}
RANK = {1: "轨迹秩 29", 2: "轨迹秩 34", 3: "轨迹秩 9"}
CURRENT = {2: 16, 3: 12}                          # FISCH_MAX / F3MAX today

def wincost(m):                                   # in CG iterations per step
    return (6.0*m + 13.0)/24.0

fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.6))
fig.suptitle("每个系统该配多大的窗口? —— 同一个真滑窗,只扫 m",
             fontsize=13, weight="bold", y=1.005)

for ax, d, lab in ((axes[0], d0, "regime 0:快前沿 + 复极弥散(轨迹高秩)"),
                   (axes[1], d2, "regime 2:低维快变(轨迹 4 维)")):
    for s in (1, 2, 3):
        cold, n, tot = d[s]["cold"], d[s]["n"], np.array(d[s]["tot"], float)
        raw = 100*(1 - tot/cold)
        adj = 100*(1 - (tot + n*np.array([wincost(m) for m in MS]))/cold)
        ax.plot(MS, raw, "o-", color=COL[s], lw=1.8, ms=5,
                label=f"{NAME[s]} · {RANK[s]}")
        ax.plot(MS, adj, "--", color=COL[s], lw=1.4, alpha=0.75)
        best = int(np.argmax(adj))
        ax.plot(MS[best], adj[best], "*", color=COL[s], ms=17,
                markeredgecolor="k", markeredgewidth=0.5, zorder=5)
        ax.annotate(f"m={MS[best]}", (MS[best], adj[best]), textcoords="offset points",
                    xytext=(4, -13), fontsize=9, color=COL[s], weight="bold")
        if s in CURRENT:
            ax.axvline(CURRENT[s], color=COL[s], lw=1.0, ls=":", alpha=0.8)
    ax.set_xscale("log"); ax.set_xticks(MS); ax.minorticks_off()
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("窗口大小 m(保留最近 m 个解)")
    ax.set_ylabel("总迭代降幅 (%)")
    ax.set_title(lab, fontsize=11, weight="bold")
    ax.grid(alpha=0.25, which="both")
axes[0].legend(fontsize=9, loc="lower left")
fig.text(0.5, -0.06,
         "实线 = 实测迭代降幅;虚线 = 再扣掉维护窗口本身的开销(≈ 0.25m + 0.54 次迭代/步,"
         "见脚本头注的 flop 模型);★ = 扣完之后的最优;竖点线 = 现在代码里的值(Sys2 16、Sys3 12)。\n"
         "Sys1 每步只有 16 次迭代,所以窗口开销占比最狠:m=16 就吃掉 4.5 次迭代(28%),m=64 时"
         "维护比求解还贵(降幅变负)。Sys2/Sys3 每步 ~160 次迭代,m=12 只占 2%。",
         ha="center", fontsize=9.4, color="#444444")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)

# ---- text summary ------------------------------------------------------------
for d, lab in ((d0, "regime 0"), (d2, "regime 2")):
    print(f"\n=== {lab} ===")
    print(f"{'sys':<5}{'cold':>8}{'steps':>7}   " + "".join(f"m={m:<7}" for m in MS))
    for s in (1, 2, 3):
        cold, n, tot = d[s]["cold"], d[s]["n"], np.array(d[s]["tot"], float)
        raw = 100*(1 - tot/cold)
        adj = 100*(1 - (tot + n*np.array([wincost(m) for m in MS]))/cold)
        print(f"Sys{s:<2}{cold:>8}{n:>7}   " + "".join(f"{v:7.1f} " for v in raw) + " (raw %)")
        print(f"{'':<5}{'':>8}{'':>7}   " + "".join(f"{v:7.1f} " for v in adj) +
              f" (cost-adjusted, best m={MS[int(np.argmax(adj))]})")
