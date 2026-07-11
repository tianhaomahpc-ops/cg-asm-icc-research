#!/usr/bin/env python3
"""plot_transferh_real.py -- the COMPLETE H-matrix result on the real Niederer+cube
geometry (forward_ecg.cpp -transfer / -transferh), np=8, measured.

What the full C++ implementation taught us (all validated vs the true solve):
  torso=36229 dofs, N_iface=4757, dense Z storage = 1.3 GB.
  (1) DENSE transfer apply is EXACT (8.8e-10) but MEMORY-BOUND: the per-step BLAS
      matvec streams the whole 172 MB/rank of Z, ~55 ms ~ the 40 ms solve.  So for
      the FULL field the bottleneck is Z's SIZE, not flops -- dense Z is not a clear
      speed win at this scale, and it costs 1.3 GB.
  (2) ELECTRODE ECG via reciprocity lead-field is EXACT (3.5e-9) and streams only 2
      vectors: 0.045 ms/step, ~900x faster than solving, 2 offline solves.  This is
      the deployable answer for the ECG output.
  (3) H-matrix must shrink Z (that is the real lever).  The from-scratch 2-sided
      SINGLE-LEVEL H-matrix is correct (8.2e-7) but only compresses 1.26x on this
      geometry (small central heart => coarse single-level clusters are weak) and its
      scattered-index apply is slower.  A real win needs a RECURSIVE multi-level H2
      matrix + DOF reordering (library-scale).  The 2D POC (fig_hmatrix) shows the
      achievable compression is 4.6-8x and GROWS with N_iface -- the asymptotic regime.
Output: fig_transferh_real.png
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
from matplotlib.patches import FancyBboxPatch

fig = plt.figure(figsize=(16.8, 5.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1.05,1.0,1.12], wspace=0.34)
fig.suptitle("H-matrix 完整落地(真 Niederer+cube,forward_ecg -transfer/-transferh 实测):"
             "全场瓶颈是 Z 的『体积』,电极输出立即赢",
             fontsize=11.8, weight="bold", y=1.02)

# (A) per-step cost, full field -- memory-bound; electrode crushes it (log)
axA = fig.add_subplot(gs[0])
labs=["每步真解\n(CG)","稠密 Z\napply","H-matrix\n单层","电极\nlead-field"]
vals=[40, 55, 110, 0.045]
cols=["#2980b9","#e67e22","#c0392b","#27ae60"]
axA.bar(range(4), vals, 0.62, color=cols, edgecolor="0.3")
for i,v in enumerate(vals):
    axA.text(i, v*1.5, (f"{v:g} ms"), ha="center", fontsize=9, weight="bold")
axA.set_yscale("log"); axA.set_ylim(0.02, 400)
axA.yaxis.set_major_formatter(FuncFormatter(lambda v,_: (f"{v:g}")))
axA.set_xticks(range(4)); axA.set_xticklabels(labs, fontsize=8.4)
axA.set_ylabel("每步耗时 (ms, 对数)")
axA.set_title("① 全场:稠密 Z apply 55ms(内存受限,\n流 172MB/核 的 Z)≈ 解;电极 0.045ms 碾压",
              fontsize=10.0, weight="bold")
axA.grid(axis="y", alpha=0.25, which="both")
axA.annotate("单层 H 未赢\n(压缩弱+散取)", xy=(2,110), xytext=(1.1,200),
             fontsize=7.8, color="#c0392b", arrowprops=dict(arrowstyle="->",color="#c0392b"))

# (B) the real lever = shrink Z's storage
axB = fig.add_subplot(gs[1])
axB.bar([0,1], [1314.9, 1041.9], 0.5, color=["#e67e22","#c0392b"], edgecolor="0.3")
for i,v in enumerate([1314.9,1041.9]): axB.text(i, v+30, f"{v:.0f} MB", ha="center", fontsize=9.5, weight="bold")
axB.set_xticks([0,1]); axB.set_xticklabels(["稠密 Z","H 单层\n(1.26×)"], fontsize=8.8)
axB.set_ylabel("Z 存储 (MB)"); axB.set_ylim(0, 1600)
axB.set_title("② 真瓶颈是 Z 的体积(这里已 1.3GB)。\n单层 2-side 只压 1.26×(中心小心脏→聚类粗)",
              fontsize=10.0, weight="bold")
axB.grid(axis="y", alpha=0.25)
axB.text(0.5,-0.26,"2D POC(多层/有利几何):压 4.6→8.1×,随 N_iface 增长(见 fig_hmatrix)",
         transform=axB.transAxes, ha="center", fontsize=7.8, color="#6c3483")

# (C) verdict: three regimes
axC = fig.add_subplot(gs[2]); axC.axis("off"); axC.set_xlim(0,10); axC.set_ylim(0,10)
axC.text(5,9.5,"③ 完整结论(全部真机验证,均精确)",ha="center",fontsize=10.4,weight="bold")
axC.add_patch(FancyBboxPatch((0.2,6.5),9.6,2.5,boxstyle="round,pad=0.1",fc="#eafaf1",ec="0.4"))
axC.text(5,8.5,"要 ECG(你实际要读的)→ 现成、已完成",ha="center",fontsize=9.4,weight="bold",color="#1e8449")
axC.text(5,7.2,"互易 lead-field:精确 3.5e-9,0.045 ms/步,~900×,离线 2 解。\n直接部署。",
        ha="center",fontsize=8.4,color="0.15")
axC.add_patch(FancyBboxPatch((0.2,3.5),9.6,2.7,boxstyle="round,pad=0.1",fc="#fef9e7",ec="0.4"))
axC.text(5,5.75,"要全场 @ 心脏尺度 → 稠密 Z 精确但内存受限",ha="center",fontsize=9.4,weight="bold",color="#b9770e")
axC.text(5,4.4,"稠密 Z apply 55ms≈解,且占 1.3GB。不是清晰提速;\n真问题是 Z 的体积(存储/内存带宽)。",
        ha="center",fontsize=8.4,color="0.15")
axC.add_patch(FancyBboxPatch((0.2,0.3),9.6,2.9,boxstyle="round,pad=0.1",fc="#f4ecf7",ec="0.5"))
axC.text(5,2.75,"要全场 @ 大尺度 → H-matrix(已实现+验证)",ha="center",fontsize=9.4,weight="bold",color="#6c3483")
axC.text(5,1.3,"2-side 单层已跑通、精确 8e-7,但此几何只压 1.26×;\n"
        "生产级需递归多层 H²+DOF 重排(库级:HLIBpro/H2Lib/STRUMPACK)。\n"
        "POC 已量化可达 4.6-8× 且随规模增长。",
        ha="center",fontsize=8.0,color="0.15")

fig.savefig("fig_transferh_real.png", dpi=140, bbox_inches="tight")
print("wrote fig_transferh_real.png")
