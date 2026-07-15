#!/usr/bin/env python3
"""plot_asmovl.py -- PURE ASM vs sASM overlap sweep on Sys3 (forward_ecg -asmovl).
No coarse space, ICC subdomain solve for BOTH, so the ONLY difference is the Schwarz
scaling.  Measured (real torso Kt, 8 subdomains, synthetic slow-mode RHS, rtol 1e-8):

  overlap :  ASM(unscaled,basic)   sASM(scaled,restrict/RAS)
     0    :        63                    63
     1    :        90                    62
     2    :       105                    50   <- sASM best
     3    :        99                    50
     4    :        90                    50

Textbook picture, all three points confirmed:
 (1) overlap 0: ASM == sASM (no overlap => multiplicity 1 => scaling = I; identical block Jacobi).
 (2) UNSCALED ASM gets WORSE with overlap (63->90->105): it double-counts / over-relaxes the
     overlap region, so adding overlap HURTS.
 (3) SCALED sASM/RAS improves with overlap (63->62->50), best at overlap 2 (50 iters), then
     saturates.  At overlap 2, sASM needs 52% fewer iters than ASM (50 vs 105, ~2.1x).
=> the scaling is what makes overlap useful; overlap alone (unscaled) is not.
Output: fig_asmovl.png
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

ov   = np.array([0,1,2,3,4])
asm  = np.array([63,90,105,99,90])
sasm = np.array([63,62,50,50,50])

fig, ax = plt.subplots(figsize=(9.2, 6.0))
ax.plot(ov, asm,  "o-", color="#c0392b", lw=2.4, ms=8, label="ASM(未缩放, PC_ASM_BASIC)")
ax.plot(ov, sasm, "s-", color="#27ae60", lw=2.4, ms=8, label="sASM(缩放, PC_ASM_RESTRICT/RAS)")
for x,y in zip(ov,asm):  ax.annotate(str(y), (x,y), textcoords="offset points", xytext=(0,9),
                                     ha="center", fontsize=10, weight="bold", color="#c0392b")
for x,y in zip(ov,sasm): ax.annotate(str(y), (x,y), textcoords="offset points", xytext=(0,-16),
                                     ha="center", fontsize=10, weight="bold", color="#1e8449")
# overlap-0 coincide marker
ax.annotate("overlap 0:两者相同\n(无重叠→缩放=恒等\n=块 Jacobi)", xy=(0,63), xytext=(0.35,74),
            fontsize=9, color="0.25", arrowprops=dict(arrowstyle="->", color="0.4"))
# ASM worsens
ax.annotate("未缩放 ASM 加重叠反而更差\n(重复计数/over-relax 重叠区)", xy=(2,105), xytext=(1.4,112),
            fontsize=9, color="#c0392b", arrowprops=dict(arrowstyle="->", color="#c0392b"))
# sASM best at ov=2
ax.annotate("sASM 最佳 @ overlap 2\n50 迭代(比 ASM 少 52%)", xy=(2,50), xytext=(2.55,58),
            fontsize=9.4, color="#1e8449", weight="bold", arrowprops=dict(arrowstyle="->", color="#1e8449"))
ax.scatter([2],[50], s=240, facecolors="none", edgecolors="#1e8449", linewidths=2.2, zorder=5)

ax.set_xlabel("重叠层数 overlap", fontsize=12)
ax.set_ylabel("CG 迭代数 (rtol 1e-8)", fontsize=12)
ax.set_xticks(ov); ax.set_ylim(40, 122); ax.grid(alpha=0.28)
ax.set_title("纯 ASM vs sASM 重叠扫描(Sys3 躯干,8 子域,ICC 子解,无粗空间)\n"
             "缩放才让重叠有用:sASM 随重叠降到 50(ov=2 最佳),未缩放 ASM 反而涨到 105",
             fontsize=12.2, weight="bold")
ax.legend(fontsize=11, loc="center right")
fig.savefig("fig_asmovl.png", dpi=140, bbox_inches="tight")
print("wrote fig_asmovl.png")
