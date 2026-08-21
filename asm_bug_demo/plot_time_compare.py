#!/usr/bin/env python3
"""plot_time_compare.py -- the complete before/after for all three systems:
iterations AND wall-clock time, cold vs the current implementation vs warm vs
the best sliding window (window maintenance included in its time).

Times are serial, local work only (no MPI).  At scale each avoided iteration
also avoids two Allreduces, so the wall-time gain on a real machine is larger
than what is shown here, not smaller.

Usage: python3 plot_time_compare.py time_r0.csv time_r2.csv fig_time_compare.png
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

def load(path):
    """parse the '# Sys<n> ...' summary block"""
    d, sy = {}, None
    for line in open(path):
        if not line.startswith("#"):
            continue
        t = line.lstrip("# ").rstrip()
        m = re.match(r"Sys(\d)\s+cold\s+(\d+) iters\s+([\d.]+) s", t)
        if m:
            sy = int(m.group(1))
            d[sy] = {"cold": (int(m.group(2)), float(m.group(3)))}
            continue
        if sy is None:
            continue
        m = re.match(r"current\s+(\d+) iters\s+([\d.]+) s", t)
        if m: d[sy]["current"] = (int(m.group(1)), float(m.group(2))); continue
        m = re.match(r"warm\s+(\d+) iters\s+([\d.]+) s", t)
        if m: d[sy]["warm"] = (int(m.group(1)), float(m.group(2))); continue
        m = re.match(r"m=(\d+)\s+(\d+) iters\s+([\d.]+) s", t)
        if m: d[sy][f"m{m.group(1)}"] = (int(m.group(2)), float(m.group(3)))
    # pick the window with the smallest wall time
    for s in d:
        wins = {k: v for k, v in d[s].items() if k.startswith("m")}
        best = min(wins, key=lambda k: wins[k][1])
        d[s]["best"] = wins[best]; d[s]["bestm"] = int(best[1:])
    return d

d0, d2 = load(r0), load(r2)
NAME = {1: "Sys1 单域", 2: "Sys2 $u_e$", 3: "Sys3 躯干"}
VAR = [("cold", "不做回收(冷启动)", "0.62"),
       ("current", "现行实现 ★", "#c0392b"),
       ("warm", "warm(上一步)", "#2980b9"),
       ("best", "最优真滑窗", "#16a085")]

fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.8))
fig.suptitle("三个系统:不做回收 vs 现行实现 vs 最优真滑窗 —— 迭代数与墙钟时间\n"
             "(regime 0 快前沿工况,350 步;窗口的时间已含维护开销)",
             fontsize=12.8, weight="bold", y=1.02)

for ax, key, unit, title in ((axes[0], 0, "总迭代数", "① 迭代数"),
                             (axes[1], 1, "总墙钟时间 (s,串行本地工作)", "② 墙钟时间")):
    x = np.arange(4); w = 0.2
    for vi, (vk, vlab, col) in enumerate(VAR):
        vals = [d0[s][vk][key] for s in (1, 2, 3)]
        vals.append(sum(vals))                       # pipeline total
        ax.bar(x + (vi-1.5)*w, vals, w, color=col, label=vlab)
        for xi, v in zip(x + (vi-1.5)*w, vals):
            ax.text(xi, v*1.02, f"{v:.0f}" if key == 0 else f"{v:.1f}",
                    ha="center", fontsize=7.2)
    base = [d0[s]["cold"][key] for s in (1, 2, 3)]; base.append(sum(base))
    bestv = [d0[s]["best"][key] for s in (1, 2, 3)]; bestv.append(sum(bestv))
    for xi, b, bv in zip(x, base, bestv):
        ax.text(xi, max(base)*1.13, f"{b/bv:.2f}×", ha="center", fontsize=11,
                color="#16a085", weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([NAME[1] + f"\nm={d0[1]['bestm']}", NAME[2] + f"\nm={d0[2]['bestm']}",
                        NAME[3] + f"\nm={d0[3]['bestm']}", "三系统合计"], fontsize=9.5)
    ax.set_ylabel(unit); ax.set_ylim(0, max(base)*1.22)
    ax.set_title(title + "  (绿色数字 = 最优滑窗相对冷启动的加速)", fontsize=11, weight="bold")
    ax.grid(alpha=0.25, axis="y")
axes[0].legend(fontsize=9, loc="upper left")

tot_c = sum(d0[s]["cold"][1] for s in (1, 2, 3))
tot_b = sum(d0[s]["best"][1] for s in (1, 2, 3))
tot_cur = sum(d0[s]["current"][1] for s in (1, 2, 3))
fig.text(0.5, -0.07,
         f"三系统合计:冷启动 {tot_c:.1f} s → 现行实现 {tot_cur:.1f} s "
         f"({'慢' if tot_cur>tot_c else '快'} {abs(tot_cur/tot_c-1)*100:.0f}%)"
         f" → 最优真滑窗 {tot_b:.1f} s({tot_c/tot_b:.2f}× 加速)。\n"
         "★ = forward_ecg.cpp 现在的做法(A-正交基 + erase(begin()),Sys2 m=16 / Sys3 m=12);"
         "Sys1 现在根本没有回收,这里的 ★ 是把同一套现行做法套上去的结果。\n"
         "时间是串行本地工作;在 3000 核上每省一次迭代还额外省 2 次 Allreduce(~50 µs),所以真机上的时间收益只会更大。",
         ha="center", fontsize=9.4, color="#444444")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)

for d, lab in ((d0, "regime 0"), (d2, "regime 2")):
    print(f"\n=== {lab} ===")
    print(f"{'':<12}" + "".join(f"{v[1]:>22}" for v in VAR))
    for s in (1, 2, 3):
        row = f"Sys{s:<11}"
        for vk, _, _ in VAR:
            it, tt = d[s][vk]
            row += f"{it:>10d} /{tt:>7.2f}s "
        print(row + f"   best m={d[s]['bestm']}")
    row = f"{'合计':<10}"
    for vk, _, _ in VAR:
        it = sum(d[s][vk][0] for s in (1, 2, 3)); tt = sum(d[s][vk][1] for s in (1, 2, 3))
        row += f"{it:>10d} /{tt:>7.2f}s "
    print(row)
