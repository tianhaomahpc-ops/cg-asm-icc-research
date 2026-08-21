#!/usr/bin/env python3
"""plot_fields_sys123.py -- what the three solutions actually look like, and why
that decides how well time recycling works.

Rows: three moments of the beat (depolarisation front / plateau / repolarisation).
Cols: Sys1 solution Vm, Sys2 solution u_e, Sys3 solution u_T (mid-plane slice).
Bottom: the normalised Gram spectrum of each solution trajectory -- the sharp
front keeps u_e high rank, the harmonic extension into the volume conductor
low-passes it, so u_T collapses onto a handful of directions.

Usage: python3 plot_fields_sys123.py slices.txt run.csv fig_fields_sys123.png
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
import matplotlib.ticker as mticker
import numpy as np

# WenQuanYi has no U+2212; mathtext tick labels route through the regular font
DEC = mticker.FuncFormatter(lambda v, _: ("1e%d" % round(np.log10(v))) if v > 0 else "0")

slicefile, csvfile, out = sys.argv[1], sys.argv[2], sys.argv[3]

# ---- slices ------------------------------------------------------------------
sl, cur = {}, None
for line in open(slicefile):
    if line.startswith("#"):
        continue
    if line.startswith("SLICE"):
        _, s, st = line.split()
        cur = (int(s), int(st)); sl[cur] = []
        continue
    if cur is not None and line.strip():
        sl[cur].append([float(v) for v in line.split()])
sl = {k: np.array(v) for k, v in sl.items()}

# The Sys3 solution vector carries ZERO on the Dirichlet face (its value lives in
# the right-hand side), so paste the interface data back in for the picture:
# row i=0 of the Sys3 slice is exactly row i=0 of the Sys2 slice.
for (sysid, st) in list(sl):
    if sysid == 3 and (2, st) in sl:
        sl[(3, st)][0, :] = sl[(2, st)][0, :]

# ---- trajectory spectra (the "# spec <label> : v1 v2 ..." lines) --------------
spec = {}
for line in open(csvfile):
    if line.startswith("# spec"):
        head, vals = line.split(":")
        label = head.replace("# spec", "").strip()
        spec[label] = np.array([float(v) for v in vals.split()])

TIMES = [8, 90, 250]
PHASE = {8: "去极化前沿扫过", 90: "平台段(近静止)", 250: "复极前沿扫过"}
COLS = [(1, "Sys1 解 = $V_m$(跨膜电位)", "inferno"),
        (2, "Sys2 解 = $u_e$(心内胞外电位)", "RdBu_r"),
        (3, "Sys3 解 = $u_T$(躯干电位)", "RdBu_r")]

fig = plt.figure(figsize=(13.6, 14.2))
gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 0.95], hspace=0.42, wspace=0.16)
fig.suptitle("三个系统的解长什么样 —— 以及为什么这决定了时间回收好不好使\n"
             "(可复现模型问题,同一张 24³ 网格、同一个 $V_m(t)$ 驱动;中平面切片)",
             fontsize=13.2, weight="bold", y=0.945)

for r, t in enumerate(TIMES):
    for c, (sysid, title, cmap) in enumerate(COLS):
        ax = fig.add_subplot(gs[r, c])
        F = sl.get((sysid, t))
        if F is None:
            ax.axis("off"); continue
        if sysid == 1:
            im = ax.imshow(F, origin="lower", cmap=cmap, vmin=-83, vmax=30, aspect="equal")
            note = f"范围 {F.min():.0f} … {F.max():.0f} mV"
        else:
            lo, hi = F.min(), F.max()
            if lo < 0 < hi:                      # straddles zero -> symmetric
                v = max(-lo, hi); lo, hi = -v, v
            if hi - lo < 1e-30: hi = lo + 1e-30
            im = ax.imshow(F, origin="lower", cmap=cmap, vmin=lo, vmax=hi, aspect="equal")
            note = f"范围 {F.min():.2e} … {F.max():.2e}"
        if sysid == 3:                       # mark where the interface data enters
            ax.axhline(-0.5, color="#16a085", lw=4.0, clip_on=False)
            ax.set_xlabel("↑ 这条边是界面 Γ:Dirichlet = $u_e$ 的迹", fontsize=8,
                          color="#0e6251")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title((title + "\n" if r == 0 else "") + f"t = {t} ms  ·  {note}",
                     fontsize=9.2, weight="bold" if r == 0 else "normal")
        if c == 0:
            ax.set_ylabel(f"{PHASE[t]}", fontsize=10.5, weight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=7)

# ---- spectra -----------------------------------------------------------------
ax = fig.add_subplot(gs[3, :])
style = [("Sys1 solution Vm",  "Sys1 $V_m$",  "#8e44ad", "o"),
         ("Sys2 solution u_e", "Sys2 $u_e$",  "#c0392b", "s"),
         ("Sys3 solution u_T", "Sys3 $u_T$",  "#16a085", "^")]
for key, lab, col, mk in style:
    if key not in spec:
        continue
    ev = np.maximum(spec[key], 1e-18)
    r4 = int((ev > 1e-4).sum()); r8 = int((ev > 1e-8).sum())
    ax.semilogy(np.arange(1, len(ev)+1), ev, mk+"-", color=col, ms=3.4, lw=1.5,
                label=f"{lab}:秩 {r4}(1e-4)/ {r8}(1e-8)")
for lev, txt in ((1e-4, "1e-4"), (1e-8, "1e-8")):
    ax.axhline(lev, color="0.55", lw=0.9, ls=":")
    ax.text(len(spec.get("Sys2 solution u_e", [1]*88))*0.985, lev*1.5, txt,
            fontsize=8, color="0.4", ha="right")
ax.axvline(16, color="#2980b9", lw=1.6, ls="--")
ax.text(16.7, 3e-3, "回收窗口 m = 16", color="#2980b9", fontsize=9.5, weight="bold")
ax.set_ylim(1e-17, 3); ax.yaxis.set_major_formatter(DEC)
ax.set_xlabel("Gram 特征值序号(解轨迹的 88 个等间隔快照)")
ax.set_ylabel("归一化特征值")
ax.set_title("④ 三条解轨迹的谱:心脏里 34 维的东西,到躯干只剩 9 维\n"
             "—— 体导体是低通滤波器,所以 Sys3 反而是最适合回收的那个",
             fontsize=11, weight="bold")
ax.legend(fontsize=9.5, loc="upper right"); ax.grid(alpha=0.25, which="both")

fig.text(0.5, 0.045,
         "读法:$V_m$ 是一条尖锐的行波前沿(高秩);$u_e$ 由 $-K_{\\sigma_i}V_m$ 驱动,"
         "保留了前沿的细结构(次高秩);$u_T$ 是界面数据的调和延拓,离界面越远越平滑(低秩)。"
         "\n窗口只要盖住轨迹的有效维数,投影出来的初值就直接落进容差 —— 这正是 Sys3 修好窗口后能拿 -76% 的原因。",
         ha="center", fontsize=9.6, color="#444444")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
