#!/usr/bin/env python3
"""plot_fischer_sys13.py -- does the sliding-window fix carry over to Sys1/Sys3?
Three panels:
  (A) per-step CG iterations for Sys3 (torso) in regime 0: cold vs the current
      forward_ecg.cpp recycler vs the true sliding window;
  (B),(C) totals for all three systems, in both regimes.
Usage: python3 plot_fischer_sys13.py r0.csv r2.csv fig_fischer_sys13.png
"""
import sys
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
    hdr, rows = None, []
    for line in open(path):
        if line.startswith("#"):
            continue
        if hdr is None:
            hdr = line.strip().split(",");  continue
        p = line.strip().split(",")
        if len(p) == len(hdr):
            rows.append([float(v) for v in p])
    return {k: np.array([r[i] for r in rows]) for i, k in enumerate(hdr)}

d0, d2 = load(r0), load(r2)
COLD, CUR, SW, SWB, WARM = "0.62", "#c0392b", "#16a085", "#117a65", "#2980b9"

fig = plt.figure(figsize=(15.6, 5.4))
gs = fig.add_gridspec(1, 3, wspace=0.26, width_ratios=[1.25, 1, 1])
fig.suptitle("同一个滑窗修法在 Sys1 / Sys2 / Sys3 上的效果 —— "
             "三个系统同网格、同一个 Vm(t) 驱动、每步一次干净冷启动做基准",
             fontsize=12.8, weight="bold", y=1.0)

# ---- (A) Sys3 per-step, regime 0 --------------------------------------------
ax = fig.add_subplot(gs[0])
t = d0["t"]
ax.plot(t, d0["s3_cold"], "-", color=COLD, lw=1.6,
        label=f"冷启动 (总 {int(d0['s3_cold'].sum())})")
ax.plot(t, d0["s3_cur"], "-", color=CUR, lw=1.0,
        label=f"现行 -fischer3 实现 (总 {int(d0['s3_cur'].sum())}, "
              f"{100*(d0['s3_cur'].sum()/d0['s3_cold'].sum()-1):+.1f}%)")
ax.plot(t, d0["s3_sw"], "-", color=SW, lw=1.3,
        label=f"真滑窗 m=16 (总 {int(d0['s3_sw'].sum())}, "
              f"-{100*(1-d0['s3_sw'].sum()/d0['s3_cold'].sum()):.0f}%)")
ax.set_xlabel("时间 t (ms)"); ax.set_ylabel("该步 CG 迭代数")
ax.set_title("① Sys3(躯干)regime 0:现行实现 ±0%,\n"
             "修好窗口后 -76% —— 『初值封顶 ~10%』不成立", fontsize=10.6, weight="bold")
ax.legend(fontsize=8.4, loc="center right"); ax.grid(alpha=0.25)

# ---- (B),(C) totals ----------------------------------------------------------
sysname = {"s1": "Sys1 单域\n(质量主导)", "s2": "Sys2 $u_e$\n(奇异)", "s3": "Sys3 躯干\n(Dirichlet 界面)"}
variants = [("warm", "warm(上一步)", WARM), ("cur", "现行实现 ★", CUR),
            ("sw", "真滑窗 m=16", SW), ("swb", "真滑窗 m=64", SWB)]
for idx, (d, lab) in enumerate(((d0, "regime 0:快前沿 + 复极弥散"),
                                (d2, "regime 2:低维快变(复现真机基准)"))):
    ax = fig.add_subplot(gs[1+idx])
    x = np.arange(3); w = 0.2
    for vi, (key, vlab, col) in enumerate(variants):
        red = [100*(1-d[f"s{s}_{key}"].sum()/d[f"s{s}_cold"].sum()) for s in (1, 2, 3)]
        ax.bar(x + (vi-1.5)*w, red, w, color=col, label=vlab)
        for xi, v in zip(x + (vi-1.5)*w, red):
            ax.text(xi, v+1.5 if v >= 0 else v-5, f"{v:.0f}", ha="center",
                    fontsize=7, color=CUR if v < 0 else "black")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([sysname[f"s{s}"] for s in (1, 2, 3)], fontsize=9)
    ax.set_ylabel("总迭代降幅 (%)"); ax.set_ylim(-12, 108)
    ax.set_title(f"{'②' if idx==0 else '③'} {lab}", fontsize=10.6, weight="bold")
    ax.grid(alpha=0.25, axis="y")
    if idx == 0: ax.legend(fontsize=8.2, loc="upper left", ncol=2)

fig.text(0.5, -0.03,
         "★ = forward_ecg.cpp 现在的做法(A-正交基 + erase(begin()));三个系统用的是同一份修法。"
         "窗口从 16 加到 64 一律无收益 —— 系数抵消从 7–33 倍涨到 250–1300 倍。",
         ha="center", fontsize=9, color="#555555")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
