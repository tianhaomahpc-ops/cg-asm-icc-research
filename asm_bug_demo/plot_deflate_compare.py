#!/usr/bin/env python3
"""plot_deflate_compare.py -- how much room is left after the window?

Two-level additive coarse correction (the same form as TwoLevelNicolaides in
forward_ecg.cpp) on top of the fixed sliding window, with two candidate coarse
spaces:
  geo   subdomain indicator functions -- LOCAL support, apply is O(N)
  snap  the window's own snapshots    -- GLOBAL dense, apply is O(kN) per iteration

Usage: python3 plot_deflate_compare.py defl_r0.csv defl_r2.csv fig_deflate_compare.png
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
VARS = ["cold(不做)", "窗口 only", "几何粗空间 only",
        "几何粗空间+窗口", "快照粗空间+窗口", "几何+快照+窗口"]

def load(path):
    d, sy = {}, None
    for line in open(path):
        if not line.startswith("#"):
            continue
        t = line.lstrip("# ").rstrip()
        m = re.match(r"--- Sys(\d)", t)
        if m: sy = int(m.group(1)); d[sy] = {}; continue
        if sy is None: continue
        m = re.match(r"(.*?)\s+(\d+) iters \(\s*([-+][\d.]+)%\)\s+([\d.]+) s", t)
        if m and m.group(1).strip() in VARS:
            d[sy][m.group(1).strip()] = (int(m.group(2)), float(m.group(4)))
    return d

d0, d2 = load(r0), load(r2)
SHOW = [("cold(不做)", "不做回收", "0.62"),
        ("窗口 only", "修好的滑窗", "#16a085"),
        ("几何粗空间+窗口", "滑窗 + 几何粗空间", "#1f6f8b"),
        ("快照粗空间+窗口", "滑窗 + 快照当粗空间", "#e67e22")]
NAME = {1: "Sys1 单域", 2: "Sys2 $u_e$", 3: "Sys3 躯干"}

fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.8))
fig.suptitle("窗口之后还剩多少空间? —— 在修好的滑窗之上再加两层粗空间\n"
             "(regime 0 快前沿工况,350 步;粗空间的 apply 成本已计入时间)",
             fontsize=12.8, weight="bold", y=1.02)

for ax, key, unit, title in ((axes[0], 0, "总迭代数", "① 迭代数"),
                             (axes[1], 1, "总墙钟时间 (s,串行本地工作)", "② 墙钟时间")):
    x = np.arange(4); w = 0.2
    for vi, (vk, vlab, col) in enumerate(SHOW):
        vals = [d0[s][vk][key] for s in (1, 2, 3)]
        vals.append(sum(vals))
        ax.bar(x + (vi-1.5)*w, vals, w, color=col, label=vlab)
        for xi, v in zip(x + (vi-1.5)*w, vals):
            ax.text(xi, v*1.02, f"{v:.0f}" if key == 0 else f"{v:.1f}",
                    ha="center", fontsize=7.2)
    base = [d0[s]["cold(不做)"][key] for s in (1, 2, 3)]; base.append(sum(base))
    best = [d0[s]["几何粗空间+窗口"][key] for s in (1, 2, 3)]; best.append(sum(best))
    for xi, b, bv in zip(x, base, best):
        ax.text(xi, max(base)*1.13, f"{b/bv:.1f}×", ha="center", fontsize=11,
                color="#1f6f8b", weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([NAME[1], NAME[2], NAME[3], "三系统合计"], fontsize=10)
    ax.set_ylabel(unit); ax.set_ylim(0, max(base)*1.22)
    ax.set_title(title + "  (蓝色数字 = 滑窗+几何粗空间 相对不做回收的加速)",
                 fontsize=10.6, weight="bold")
    ax.grid(alpha=0.25, axis="y")
axes[0].legend(fontsize=9, loc="upper left")

tc = sum(d0[s]["cold(不做)"][1] for s in (1, 2, 3))
tw = sum(d0[s]["窗口 only"][1] for s in (1, 2, 3))
tg = sum(d0[s]["几何粗空间+窗口"][1] for s in (1, 2, 3))
fig.text(0.5, -0.07,
         f"三系统合计:不做 {tc:.1f} s → 只修滑窗 {tw:.1f} s({tc/tw:.2f}×)"
         f" → 滑窗+几何粗空间 {tg:.1f} s({tc/tg:.2f}×)。窗口之后确实还有约一倍的空间,"
         "但它来自谱(粗空间),不是更好的初值。\n"
         "把回收快照当粗空间几乎没用(橙色):那个子空间已经被当初值用过一遍,再 deflate 是重复劳动;"
         "而且它的 apply 是 O(kN) 稠密,时间反而更差。\n"
         "Sys1 加粗空间纯亏(cond 1.4、0 个慢模,没有慢模可治)——粗空间只该给 Sys2/Sys3 配。",
         ha="center", fontsize=9.3, color="#444444")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)

for d, lab in ((d0, "regime 0"), (d2, "regime 2")):
    print(f"\n=== {lab} ===")
    for s in (1, 2, 3):
        print(f"Sys{s}:")
        for vk in VARS:
            if vk in d[s]:
                it, tt = d[s][vk]
                c_it, c_tt = d[s]["cold(不做)"]
                print(f"   {vk:<20} {it:>8d} ({100*(it/c_it-1):+6.1f}%)  {tt:>7.2f}s "
                      f"({100*(tt/c_tt-1):+6.1f}%)")
    for vk in ("cold(不做)", "窗口 only", "几何粗空间+窗口"):
        it = sum(d[s][vk][0] for s in (1, 2, 3)); tt = sum(d[s][vk][1] for s in (1, 2, 3))
        print(f"   合计 {vk:<18} {it:>8d}  {tt:>7.2f}s")
