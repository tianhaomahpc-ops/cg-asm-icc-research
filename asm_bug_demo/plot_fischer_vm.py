#!/usr/bin/env python3
"""plot_fischer_vm.py -- figure for the Sys2 recycling experiments in
fischer_vm_test.c.  Four panels:
  (A) per-step CG iterations: cold vs the current forward_ecg.cpp policy
      (evict the oldest ORTHONORMAL vector) vs the true sliding window;
  (B) the smoking gun -- how much of the PREVIOUS solution the window still
      represents; 0 = "the window really holds the last m solutions";
  (C) total-iteration reduction for every guess, in all three regimes;
  (D) eta (Vm-side least-squares residual, zero matvecs) as a predictor of
      "this guess already lands inside tol".
Usage: python3 plot_fischer_vm.py r0.csv r1.csv r2.csv fig_fischer_vm.png
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
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"   # WQY has no U+2212
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# WenQuanYi has no U+2212, and mathtext tick labels ("$10^{-4}$") route through
# the regular font -- so format decades as plain ASCII instead.
DEC = mticker.FuncFormatter(lambda v, _: ("1e%d" % round(np.log10(v))) if v > 0 else "0")

r0, r1, r2, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

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

d0, d1, d2 = load(r0), load(r1), load(r2)
COLD, OLD, SW = "0.62", "#c0392b", "#16a085"

fig = plt.figure(figsize=(15.6, 9.6))
gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.22)
fig.suptitle("Sys2 时间回收:窗口怎么滑决定一切 —— 可复现模型问题"
             "(奇异 pure-Neumann + 非等各向异性 + 移动前沿)",
             fontsize=13.5, weight="bold", y=0.975)

# ---- (A) per-step iterations, regime 2 (matches the real machine) ------------
ax = fig.add_subplot(gs[0, 0])
t = d2["t"]
ax.plot(t, d2["cold"], "-", color=COLD, lw=1.8, label=f"冷启动 (总 {int(d2['cold'].sum())})")
ax.plot(t, d2["f_old"], "-", color=OLD, lw=1.2,
        label=f"现行实现:淘汰最旧正交基向量 (总 {int(d2['f_old'].sum())}, "
              f"-{100*(1-d2['f_old'].sum()/d2['cold'].sum()):.0f}%)")
ax.plot(t, d2["aopt"], "-", color=SW, lw=1.5,
        label=f"真滑窗:最近 m 个解 + Gram 定系数 (总 {int(d2['aopt'].sum())}, "
              f"-{100*(1-d2['aopt'].sum()/d2['cold'].sum()):.1f}%)")
ax.set_xlabel("时间 t (ms)"); ax.set_ylabel("该步 CG 迭代数")
ax.set_title("① regime 2 —— 复现真机三个基准(warm -2%/physics -5%/现行 Fischer -69%)\n"
             "同一工况下把窗口修对:-69% → -98.6%", fontsize=10.5, weight="bold")
ax.legend(fontsize=8.2, loc="center right"); ax.grid(alpha=0.25)

# ---- (B) span diagnostic -----------------------------------------------------
ax = fig.add_subplot(gs[0, 1])
ax.semilogy(t, np.maximum(d2["span_old"], 1e-17), "-", color=OLD, lw=1.3,
            label="现行:绝对阈值 1e-12 + 淘汰最旧正交向量")
ax.semilogy(t, np.maximum(d2["span_rel"], 1e-17), "-", color="#2980b9", lw=1.3,
            label="只把阈值改成相对(1e-3·‖u‖_A):不再收噪声 → 不再淘汰")
ax.axhline(2e-16, color=SW, lw=2.2, ls=":", label="真滑窗(按定义恒为 0)")
ax.set_ylim(1e-17, 3)
ax.axvspan(0, 16, color="#eafaf1", alpha=0.9)
ax.text(20, 2e-14, "淘汰开始前两者完全一致\n(不淘汰时与真滑窗 90/90 步逐步相同)",
        fontsize=8.5, color="#1e8449")
ax.set_xlabel("时间 t (ms)")
ax.yaxis.set_major_formatter(DEC)
ax.set_ylabel("‖上一步解 - 它在窗口上的投影‖_A / ‖上一步解‖_A")
ax.set_title("② 原因:窗口里已经没有『上一步的解』了\n"
             "正交化让每个留下的向量都含被淘汰者的分量,erase(begin()) 把它删掉",
             fontsize=10.5, weight="bold")
ax.legend(fontsize=8.5, loc="center right"); ax.grid(alpha=0.25, which="both")

# ---- (C) totals --------------------------------------------------------------
ax = fig.add_subplot(gs[1, 0])
keys = [("warm", "warm(上一步 u_e)"), ("f_old", "F 淘汰最旧正交向量 ★"),
        ("f_res", "F 满了就重启"), ("f_div", "F 相位分散淘汰"),
        ("vmfit", "同窗口·用最新 Vm 定系数"), ("aopt", "同窗口·A-最优 = 真滑窗"),
        ("sw_div", "真滑窗+相位分散淘汰"), ("sw_big", "真滑窗·窗口 ×3"),
        ("phys", "physics x0=c*·Vm"), ("warmdelta12", "warm+SGS12(ΔVm)"),
        ("enrich", "真滑窗+ΔVm 富化")]
x = np.arange(len(keys)); w = 0.27
for off, d, lab, col in ((-w, d0, "regime 0:快前沿+复极弥散", "#34495e"),
                         (0.0, d1, "regime 1:慢前沿+近静止平台", "#7f8c8d"),
                         (+w, d2, "regime 2:低维但快变(warm 弱)", "#b3b6b7")):
    red = [100*(1-d[k].sum()/d["cold"].sum()) for k, _ in keys]
    ax.bar(x+off, red, w, color=col, label=lab)
    for xi, v in zip(x+off, red):
        ax.text(xi, v+1.2 if v >= 0 else v-4.0, f"{v:.0f}", ha="center",
                fontsize=6.6, color=OLD if v < 0 else "black")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([l for _, l in keys], rotation=30, ha="right", fontsize=8)
ax.set_ylabel("总迭代数相对冷启动的降幅 (%)")
ax.set_title("③ 所有初值方案(★ = 现行实现):赢的是『窗口是否真的是最近 m 个解』",
             fontsize=11, weight="bold")
ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.25, axis="y")

# ---- (D) eta gate ------------------------------------------------------------
ax = fig.add_subplot(gs[1, 1])
gate = 1e-4
for d, mk, lab, col in ((d0, "o", "regime 0", "#2980b9"),
                        (d1, "^", "regime 1", "#e67e22"),
                        (d2, "s", "regime 2", "#8e44ad")):
    eta = np.maximum(d["eta"], 1e-12)
    hit = d["aopt"] == 0
    sel = d["eta"] < gate
    prec = (sel & hit).sum()/max(1, sel.sum()); rec = (sel & hit).sum()/max(1, hit.sum())
    ax.scatter(eta, d["aopt"], s=10, marker=mk, alpha=0.5, color=col,
               label=f"{lab}: 门内准确率 {prec:.2f} / 召回 {rec:.2f}")
ax.axvline(gate, color=OLD, lw=1.6, ls="--")
ax.set_xscale("log"); ax.xaxis.set_major_formatter(DEC)
ax.set_xlabel("η = 用最新 Vm 在历史 Vm 上做最小二乘的相对残差(零 matvec)")
ax.set_ylabel("真滑窗初值下的该步 CG 迭代数")
ax.set_title("④ 最新 Vm 真正有用的地方:免费预判『这一步会不会 0 迭代』\n"
             "η ≥ 1e-4 时从未命中过(召回 1.00)—— 不改精度,改的是知道自己行不行",
             fontsize=11, weight="bold")
ax.legend(fontsize=8.5, loc="center left"); ax.grid(alpha=0.25, which="both")

fig.text(0.5, 0.005,
         "所有变体都从同一条『干净冷启动解』的历史生长,互不污染(与 forward_ecg.cpp 一致);"
         "停机判据 = 未预条件残差 ≤ 1e-8‖b‖,初值好到位就是 0 迭代。",
         ha="center", fontsize=9, color="#555555")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
