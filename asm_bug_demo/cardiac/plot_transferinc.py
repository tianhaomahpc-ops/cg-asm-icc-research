#!/usr/bin/env python3
"""plot_transferinc.py -- HONEST negative result for using Sys1/Sys2 info to make
Sys3 incremental.  Measured (-transferinc, real Niederer+cube, np=8, 80 steps):
  active interface DOFs / step (|d_u_e| > 1e-3 max|u_e|): avg 4700 of 4757 = 99%
  => streams ~all of Z every step -> incremental apply 55 ms ~ the 61 ms solve (no win)
  full-field error mean 5.2e-4 (threshold drop + drift between refreshes)

The idea: phi(t)=phi(t-1)+Z*(u_e(t)-u_e(t-1)); only the moving front changes u_e, so
(hoped) only K<<N_iface columns update -> stream a small fraction of the 1.3 GB Z.
Sys1 gives the front location, Sys2 gives the u_e increment.

Why it FAILS: u_e is the solution of an ELLIPTIC equation.  The front (source) is
spatially local, but its effect on u_e (the response) is GLOBAL -- when the front
moves, u_e shifts slightly EVERYWHERE on the interface.  So the increment Delta u_e
is small in norm but DENSE in support (99% above any useful threshold), not sparse.
The same ellipticity that makes the transfer operator's FAR FIELD low-rank (spatial
smoothing) also DELOCALIZES the temporal increment -- there is no local-in-space,
local-in-time update to exploit.  This closes the 'use Sys1/Sys2 to accelerate Sys3'
line: knowing WHERE the front is does not localize the Sys3 update, because the
elliptic Green's function spreads it globally.
Output: fig_transferinc.png
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
from matplotlib.patches import FancyBboxPatch

fig = plt.figure(figsize=(15.2, 5.3))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.24)
fig.suptitle("增量传输(Sys1 前沿 + Sys2 u_e 增量)加速 Sys3 —— 实测负面:u_e 增量是全局稠密,不是局部稀疏",
             fontsize=12.0, weight="bold", y=1.02)

# (A) active fraction: ~99% -- no sparsity
axA = fig.add_subplot(gs[0])
axA.bar([0,1], [99, 1], 0.55, color=["#c0392b","#bdc3c7"], edgecolor="0.3")
axA.text(0, 101, "99%\n(4700/4757)", ha="center", fontsize=10, weight="bold", color="#c0392b")
axA.text(1, 3, "1%", ha="center", fontsize=9)
axA.set_xticks([0,1]); axA.set_xticklabels(["每步『在变』的\n界面 DOF","『没变』的"])
axA.set_ylabel("占界面 DOF 的百分比"); axA.set_ylim(0,112)
axA.set_title("① 每步 99% 的界面都在变\n→ 没有稀疏更新可利用(流~全部 Z)", fontsize=10.4, weight="bold")
axA.grid(axis="y", alpha=0.25)
axA.text(0.5,-0.20,"增量 apply 55ms ≈ 解 61ms(没赢);全场误差 5.2e-4",
         transform=axA.transAxes, ha="center", fontsize=8.4, color="#555")

# (B) mechanism: local source -> global elliptic response
axB = fig.add_subplot(gs[1]); axB.axis("off"); axB.set_xlim(0,10); axB.set_ylim(0,10)
axB.text(5,9.5,"② 为什么:椭圆方程把局部前沿『抹』成全局响应",ha="center",fontsize=10.6,weight="bold")
axB.add_patch(FancyBboxPatch((0.3,6.3),4.4,2.6,boxstyle="round,pad=0.1",fc="#eafaf1",ec="0.4"))
axB.text(2.5,8.5,"源:去极化前沿",ha="center",fontsize=9.6,weight="bold",color="#1e8449")
axB.text(2.5,7.1,"空间局部(窄带)\nSys1 精确知道在哪",ha="center",fontsize=8.6,color="0.15")
axB.annotate("", xy=(5.5,7.6), xytext=(4.8,7.6), arrowprops=dict(arrowstyle="-|>",lw=2,color="#c0392b"))
axB.text(5.15,8.05,"椭圆解",ha="center",fontsize=8,color="#c0392b")
axB.add_patch(FancyBboxPatch((5.6,6.3),4.1,2.6,boxstyle="round,pad=0.1",fc="#fdedec",ec="0.4"))
axB.text(7.65,8.5,"响应:u_e(界面)",ha="center",fontsize=9.6,weight="bold",color="#c0392b")
axB.text(7.65,7.1,"全局稠密:前沿一动,\nu_e 处处都变一点",ha="center",fontsize=8.6,color="0.15")
axB.add_patch(FancyBboxPatch((0.3,3.1),9.4,2.6,boxstyle="round,pad=0.1",fc="#fef9e7",ec="0.4"))
axB.text(5,5.25,"Δu_e 小(范数)但稠密(支撑)→ 阈值挡不住 99% 的 DOF",ha="center",fontsize=9.4,weight="bold",color="#b9770e")
axB.text(5,3.9,"让传输算子远场低秩的『空间光滑』,同样把时间增量『全局化』:\n"
        "没有既局部于空间、又局部于时间的更新可利用。",ha="center",fontsize=8.5,color="0.15")
axB.add_patch(FancyBboxPatch((0.3,0.3),9.4,2.4,boxstyle="round,pad=0.1",fc="#f4ecf7",ec="0.5"))
axB.text(5,2.15,"结论:『用 Sys1/2 加速 Sys3』这条线到此关闭",ha="center",fontsize=9.4,weight="bold",color="#6c3483")
axB.text(5,0.95,"知道前沿在哪 ≠ 能局部化 Sys3 更新——椭圆 Green 函数把它铺满全场。\n"
        "剩下能用的只有:电极 lead-field(ECG)、算子级(GAMG/Chebyshev)、大尺度 H-matrix。",
        ha="center",fontsize=8.3,color="0.15")

fig.savefig("fig_transferinc.png", dpi=140, bbox_inches="tight")
print("wrote fig_transferinc.png")
