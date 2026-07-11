#!/usr/bin/env python3
"""plot_transfer_real.py -- DIRECT validation of the interface->torso transfer
operator on the REAL Niederer-parameter, fake-geometry (heart slab + cube torso)
coupled forward-ECG driver (forward_ecg.cpp -transfer), np=8, 80 EP steps.

MEASURED (mpirun -np 8 ./forward_ecg -m heart_torso.msh -transfer -T 80 -dt 0.02):
  N_iface (transfer-operator columns = one-time Sys3 solves) : 4757
  OFFLINE build Z (once)              : 250 s   (4757 solves)
  per-step TRUE solve (bjacobi+ICC)   : 45 ms/step
  per-step TRANSFER apply (matvec)    : 61 ms/step   (naive dense gemv, 0 solve)
  full-field EXACTNESS vs true solve  : mean rel-L2 8.1e-10   max 1.4e-9
  (physics sanity: Niederer P8 activation 40 ms, CV ~0.55 m/s -- real params run)

VERDICT (honest):
 (1) The CORE CLAIM IS VALIDATED on real geometry+params: the fixed operator Z
     reproduces the FULL torso field to solver tolerance (8e-10) EVERY step with
     ZERO per-step solve.  The high-rank moving-front RHS is genuinely irrelevant
     -- you apply the fixed operator, not a reduced basis of the moving field.
 (2) But the NAIVE DENSE operator is NOT a free win: N_iface=4757 => 250 s offline,
     and a dense 4757-column matvec (61 ms) costs about as much as the solve it
     replaces (45 ms).  So exactness is free; SPEED is not, unless you either
       - reduce the OUTPUT to electrodes (few rows) => apply ~ microseconds, exact
         (this is exactly the reciprocity lead-field), or
       - COMPRESS Z with H-matrix/FMM (far-field blocks are low rank) + amortize
         the offline build over the real 10^4-10^5 steps.
Output: fig_transfer_real.png
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
from matplotlib.ticker import FuncFormatter, LogLocator

Niface=4757; t_off=250.0; ms_solve=45.0; ms_apply=61.0
err_mean=8.1e-10; err_max=1.4e-9

fig = plt.figure(figsize=(16.6, 5.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0,1.05,1.05], wspace=0.34)
fig.suptitle("真实参数假几何(Niederer 心脏 slab + 立方体躯干)直接验证 界面→躯干传输算子 —— "
             "精确性成立,但朴素稠密算子不是免费的加速",
             fontsize=12.2, weight="bold", y=1.02)

# (A) EXACTNESS -- the validated win
axA = fig.add_subplot(gs[0])
axA.bar([0,1],[err_mean,err_max], color=["#16a085","#1abc9c"], width=0.55, edgecolor="0.3")
axA.axhline(1e-10, color="0.5", ls="--", lw=1)
axA.text(0,err_mean*1.3,"8.1e-10",ha="center",fontsize=10,weight="bold")
axA.text(1,err_max*1.3,"1.4e-9",ha="center",fontsize=10,weight="bold")
axA.text(1.05,1.1e-10,"求解器容差 ~1e-10",fontsize=8,color="0.4")
axA.set_yscale("log"); axA.set_ylim(1e-12,1e-7)
axA.yaxis.set_major_locator(LogLocator(base=10,numticks=7))
axA.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f"1e{int(round(np.log10(v)))}"))
axA.set_xticks([0,1]); axA.set_xticklabels(["mean","max"])
axA.set_ylabel("全场相对 L2 误差(vs 真解)")
axA.set_title("① 精确性【成立】:Z 对真实高秩 RHS\n每步复现全躯干场到 8e-10(=求解器容差)",
              fontsize=10.3, weight="bold")
axA.grid(axis="y",alpha=0.25,which="both")

# (B) COST reality: offline build + per-step apply vs solve
axB = fig.add_subplot(gs[1])
x=np.arange(3); w=0.6
vals=[ms_solve, ms_apply]
axB.bar([0,1],vals,w,color=["#2980b9","#e67e22"],edgecolor="0.3")
for i,v in enumerate(vals): axB.text(i,v+1.2,f"{v:.0f} ms",ha="center",fontsize=10,weight="bold")
axB.set_xticks([0,1]); axB.set_xticklabels(["每步真解\n(bjacobi+ICC CG)","每步传输 apply\n(朴素稠密 matvec)"])
axB.set_ylabel("每步耗时 (ms)"); axB.set_ylim(0,80)
axB.set_title(f"② 速度【没赢】:N_iface={Niface} 列的稠密\nmatvec(61)≈ 求解(45);离线建 Z 还要 {t_off:.0f}s",
              fontsize=10.3, weight="bold")
axB.grid(axis="y",alpha=0.25)
axB.annotate("稠密+朴素+超订\n未用 BLAS/未压缩", xy=(1,ms_apply), xytext=(0.3,70),
             fontsize=8.2,color="#c0392b",arrowprops=dict(arrowstyle="->",color="#c0392b"))

# (C) what actually makes it win
axC = fig.add_subplot(gs[2]); axC.axis("off"); axC.set_xlim(0,10); axC.set_ylim(0,10)
from matplotlib.patches import FancyBboxPatch
axC.text(5,9.5,"③ 怎样才真的赢(精确已免费,要的是速度)",ha="center",fontsize=10.6,weight="bold")
axC.add_patch(FancyBboxPatch((0.3,6.0),9.4,2.9,boxstyle="round,pad=0.12",fc="#eafaf1",ec="0.4"))
axC.text(5,8.35,"A. 只要电极输出(少数行)",ha="center",fontsize=9.8,weight="bold",color="#1e8449")
axC.text(5,6.95,"取 Z 的电极行 = 互易 lead-field:apply 只是几个点积(~微秒),\n"
        "对任意 RHS 精确,离线只需 电极数 次伴随解 —— 压倒性且现成。",
        ha="center",fontsize=8.6,color="0.15")
axC.add_patch(FancyBboxPatch((0.3,2.7),9.4,2.9,boxstyle="round,pad=0.12",fc="#fef9e7",ec="0.4"))
axC.text(5,5.05,"B. 要全躯干体积场",ha="center",fontsize=9.8,weight="bold",color="#b9770e")
axC.text(5,3.65,"必须 H-matrix/FMM 压缩 Z(远场块低秩,近界面不压)+ 用 BLAS,\n"
        "并把 250s 离线摊到真实 1e4-1e5 步;只有 apply 先压到 < 求解 才谈得上摊销。",
        ha="center",fontsize=8.6,color="0.15")
axC.add_patch(FancyBboxPatch((0.3,0.3),9.4,2.0,boxstyle="round,pad=0.12",fc="#f4ecf7",ec="0.5"))
axC.text(5,1.75,"结论:精确性在真参数真算子上验证通过(8e-10);",ha="center",fontsize=9.2,weight="bold",color="#6c3483")
axC.text(5,0.75,"“近似算子而非近似动解”对——但全场提速要靠压缩,电极输出则立刻兑现。",
        ha="center",fontsize=8.6,color="0.15")

fig.savefig("fig_transfer_real.png",dpi=140,bbox_inches="tight")
print("wrote fig_transfer_real.png")
print(f"N_iface={Niface}, offline={t_off}s, solve={ms_solve}ms, apply={ms_apply}ms, "
      f"exactness mean={err_mean} max={err_max}")
