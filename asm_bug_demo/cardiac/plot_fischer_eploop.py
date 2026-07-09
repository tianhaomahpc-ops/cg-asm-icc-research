#!/usr/bin/env python3
"""plot_fischer_eploop.py -- REAL EP time loop (forward_ecg -fischer -T 80 -dt 0.02,
np=8).  Parse the per-step [ITERS] lines and show Sys2's CG iteration count at
every millisecond for cold start (x0=0) vs Fischer recycling (project out the
A-orthonormal history of previous u_e solves).  This is the direct
visualization of "recycling makes the iteration count drop as history builds".
Input : fischer_eploop.txt   Output: fig_fischer_eploop.png
"""
import matplotlib, numpy as np, re
matplotlib.use("Agg")
from matplotlib import font_manager
try:
    font_manager.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
    matplotlib.rcParams["font.family"] = "WenQuanYi Zen Hei"
except Exception: pass
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

pat = re.compile(r"t=(\d+)ms.*Sys2\(singular\)=(\d+) \(cold=(\d+) warm=(\d+) phys=(\d+)\)")
t, fis, cold, warm, phys = [], [], [], [], []
for line in open("fischer_eploop.txt"):
    m = pat.search(line)
    if m:
        t.append(int(m[1])); fis.append(int(m[2])); cold.append(int(m[3]))
        warm.append(int(m[4])); phys.append(int(m[5]))
t=np.array(t); fis=np.array(fis); cold=np.array(cold); warm=np.array(warm); phys=np.array(phys)
print(f"steps={len(t)}  cold sum={cold.sum()}  warm sum={warm.sum()}  fischer sum={fis.sum()}")
print(f"reductions: warm {100*(1-warm.sum()/cold.sum()):.0f}%  fischer {100*(1-fis.sum()/cold.sum()):.0f}%")
print(f"fischer: min={fis.min()} max={fis.max()}  #steps at 0 iters = {(fis==0).sum()}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.0), gridspec_kw={"width_ratios":[1.7,1]})
fig.suptitle("真实 EP 时间循环(T=80ms, np=8):Sys2 每个时间步的 CG 迭代数 —— 冷启动 vs Fischer 回收",
             fontsize=13.5, weight="bold", y=1.0)

# left: per-step iteration counts
ax1.plot(t, cold, "-", color="#c0392b", lw=2.0, label=f"冷启动 x0=0 (总 {cold.sum()})")
ax1.plot(t, warm, "-", color="#e67e22", lw=1.4, alpha=0.8, label=f"warm 上一步 u_e (总 {warm.sum()}, -{100*(1-warm.sum()/cold.sum()):.0f}%)")
ax1.plot(t, fis,  "-o", color="#27ae60", lw=2.2, ms=3.5, label=f"Fischer 回收 (总 {fis.sum()}, -{100*(1-fis.sum()/cold.sum()):.0f}%)")
ax1.axvspan(0, 16, color="#eafaf1", alpha=0.8)
ax1.text(1.0, 8, "前 16 步:历史\n还没攒满\n(基在成长)", fontsize=8.8, color="#1e8449")
ax1.axhline(0, color="0.7", lw=0.8)
ax1.set_xlabel("时间 t (ms) = 第几次求解 Sys2"); ax1.set_ylabel("该步 Sys2 的 CG 迭代数")
ax1.set_title("① 逐步迭代数:冷启动一直 ~90;Fischer 攒够历史后\n大量步 0–几次(初值已在容差内)", fontsize=10.6, weight="bold")
ax1.legend(fontsize=9.5, loc="center right"); ax1.grid(alpha=0.25); ax1.set_ylim(-3, 100)

# right: cumulative work
ax2.plot(t, np.cumsum(cold), color="#c0392b", lw=2.2, label="冷启动累计")
ax2.plot(t, np.cumsum(warm), color="#e67e22", lw=1.6, label="warm 累计")
ax2.plot(t, np.cumsum(fis),  color="#27ae60", lw=2.4, label="Fischer 累计")
ax2.set_xlabel("时间 t (ms)"); ax2.set_ylabel("累计 CG 迭代数(总工作量)")
ax2.set_title(f"② 累计工作量:斜率就是每步成本\nFischer 斜率随历史成熟越压越平", fontsize=10.6, weight="bold")
ax2.legend(fontsize=9.5, loc="upper left"); ax2.grid(alpha=0.25)
ax2.annotate(f"最终:{cold.sum()} → {fis.sum()}\n(-{100*(1-fis.sum()/cold.sum()):.0f}%, {cold.sum()/fis.sum():.1f}x)",
             xy=(t[-1], fis.sum()), xytext=(t[-1]-42, fis.sum()+1800),
             fontsize=10, color="#27ae60", weight="bold", arrowprops=dict(arrowstyle="->",color="#27ae60"))

fig.savefig("fig_fischer_eploop.png", dpi=140, bbox_inches="tight")
print("wrote fig_fischer_eploop.png")
