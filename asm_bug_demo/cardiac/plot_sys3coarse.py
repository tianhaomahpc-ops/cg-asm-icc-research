#!/usr/bin/env python3
"""plot_sys3coarse.py -- Sys3 self-acceleration inside the sASM+CG framework
(forward_ecg.cpp -sys3coarse, real torso Kt, 8 subdomains, rtol 1e-8, measured).

Framework: kappa(M^-1 A) <= C0^2 * omega * (Nhat+1).
  Nhat -> scaling (already in the multiplicity-weighted sASM)
  omega -> overlap + subdomain solve
  C0   -> coarse space (global coupling / slow modes)
We swept the C0 knob (coarse spaces) and the omega knob (overlap) on Sys3:

  fine bjacobi+ICC (1-level)          : 63 iters   (0.51 ms/iter)
  C0: + Nicolaides coarse (dim 8)     : 60  (-4%)
  C0: + geometric {1,x,y,z} (dim 32)  : 56  (-11%)
  C0: + recycled A^-1 snapshots (12)  : 61  (-3%)
  C0: + hybrid geom+recycled (43)     : 57  (-9%)
  omega: sASM overlap 0 / 1 / 2       : 63 / 62 / 50  (-0 / -1 / -20%)  (per-iter rises to 1.0 ms)
  best cheap 2-level: overlap2 + geom : 47  (-25%)
  MULTILEVEL: GAMG                    : 13  (-79%, 5.3x)  (~4x per-iter)

READING: cheap coarse spaces barely move Sys3 (~4-11%) because its ill-conditioning
(kappa~172) is the h-refinement MULTISCALE hierarchy of the Laplacian, NOT a low-dim
slow subspace -- so a single crude coarse level cannot scale it (and even A^-1-recycled
'spectral' snapshots give only -3%).  Overlap (omega) is the stronger cheap knob (-20%)
but doubles per-iter cost.  The real cut is the framework's MULTILEVEL extension with a
PROPER coarse space (smoothed-aggregation) = GAMG, 5.3x, at ~4x per-iter.  In-framework
recipe for Sys3: overlap + a GENUINE coarse space (coarse-mesh / GenEO / smoothed-agg),
never a crude subdomain-constant one.
Output: fig_sys3coarse.png
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

fig = plt.figure(figsize=(16.6, 5.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.05], wspace=0.30)
fig.suptitle("Sys3 在 sASM+CG 框架内自我加速:kappa <= C0^2 · omega · (Nhat+1) —— 便宜的粗空间/重叠只 ~10-25%,真降迭代要多层(GAMG)",
             fontsize=11.6, weight="bold", y=1.02)

# (A) iterations per variant
axA = fig.add_subplot(gs[0])
labs = ["基线\nbj+ICC","+Nicolaides\n(C₀,dim8)","+几何{1xyz}\n(C₀,dim32)","+回收快照\n(C₀)",
        "重叠2\n(ω)","重叠2+几何\n(ω+C₀)","GAMG\n(多层)"]
its  = [63, 60, 56, 61, 50, 47, 13]
cols = ["#7f8c8d","#5dade2","#2980b9","#85c1e9","#e67e22","#16a085","#c0392b"]
axA.bar(range(7), its, 0.68, color=cols, edgecolor="0.3")
for i,v in enumerate(its):
    d = 0 if i==0 else -round(100*(63-v)/63)
    axA.text(i, v+1.2, f"{v}"+("" if i==0 else f"\n{d}%"), ha="center", fontsize=8.6, weight="bold")
axA.set_xticks(range(7)); axA.set_xticklabels(labs, fontsize=7.4)
axA.set_ylabel("CG 迭代数 (rtol 1e-8)"); axA.set_ylim(0, 74)
axA.set_title("① 迭代数:便宜的 C₀/ω 只降到 47(-25%);\n只有多层 GAMG 到 13(-79%,5.3×)",
              fontsize=10.0, weight="bold")
axA.grid(axis="y", alpha=0.25)

# (B) why cheap coarse fails: spectrum is multiscale, not low-dim
axB = fig.add_subplot(gs[1]); axB.axis("off"); axB.set_xlim(0,10); axB.set_ylim(0,10)
axB.text(5,9.5,"② 为什么便宜粗空间无效",ha="center",fontsize=10.6,weight="bold")
axB.add_patch(FancyBboxPatch((0.3,5.9),9.4,3.0,boxstyle="round,pad=0.1",fc="#fdedec",ec="0.4"))
axB.text(5,8.3,"Sys3 的病态是『多尺度』,不是『低维』",ha="center",fontsize=9.6,weight="bold",color="#c0392b")
axB.text(5,6.8,"κ~172 来自 Laplacian 的 h 细化层级(~1/h²),\n"
        "是一整片从低到高的频率,不是几个孤立慢模。\n"
        "连 A⁻¹ 回收的『谱』快照也只 -3% → 没有低维子空间可去。",
        ha="center",fontsize=8.4,color="0.15")
axB.add_patch(FancyBboxPatch((0.3,2.6),9.4,2.9,boxstyle="round,pad=0.1",fc="#eafaf1",ec="0.4"))
axB.text(5,4.9,"单层粗空间(常数/{1xyz})太粗",ha="center",fontsize=9.6,weight="bold",color="#1e8449")
axB.text(5,3.5,"它只削掉最低几个频率,主体的多尺度层级留着 →\n"
        "2 层 ASM 的 h 无关界需要真正的粗『网格』空间,不是子域常数。",
        ha="center",fontsize=8.4,color="0.15")
axB.add_patch(FancyBboxPatch((0.3,0.3),9.4,2.0,boxstyle="round,pad=0.1",fc="#fef9e7",ec="0.4"))
axB.text(5,1.7,"重叠(ω):-20% 但每迭代翻倍(墙钟基本抵消)",ha="center",fontsize=9.0,weight="bold",color="#b9770e")
axB.text(5,0.75,"是更强的便宜旋钮,但 3000 核上重叠通信更贵。",ha="center",fontsize=8.2,color="0.15")

# (C) verdict / recipe
axC = fig.add_subplot(gs[2]); axC.axis("off"); axC.set_xlim(0,10); axC.set_ylim(0,10)
axC.text(5,9.5,"③ 框架内的 Sys3 加速配方",ha="center",fontsize=10.6,weight="bold")
axC.add_patch(FancyBboxPatch((0.3,6.3),9.4,2.6,boxstyle="round,pad=0.1",fc="#f4ecf7",ec="0.5"))
axC.text(5,8.4,"真降迭代 = 多层 + 真粗空间",ha="center",fontsize=9.6,weight="bold",color="#6c3483")
axC.text(5,7.0,"GAMG(smoothed-aggregation 多层)= 63→13(5.3×),\n"
        "这就是框架的多层推广(AMG=多层 Schwarz),只是每迭代 ~4×。",
        ha="center",fontsize=8.4,color="0.15")
axC.add_patch(FancyBboxPatch((0.3,3.3),9.4,2.6,boxstyle="round,pad=0.1",fc="#eaf2f8",ec="0.4"))
axC.text(5,5.4,"便宜档(想留在 2 层):重叠2 + 几何粗 = -25%",ha="center",fontsize=9.2,weight="bold",color="#1f618d")
axC.text(5,4.0,"迭代 63→47;但重叠让每迭代变贵,墙钟收益有限,\n大规模上重叠通信更不划算。",
        ha="center",fontsize=8.4,color="0.15")
axC.add_patch(FancyBboxPatch((0.3,0.3),9.4,2.7,boxstyle="round,pad=0.1",fc="#eafaf1",ec="0.4"))
axC.text(5,2.5,"更好的粗空间(未测,需库):GenEO 谱粗空间",ha="center",fontsize=9.2,weight="bold",color="#1e8449")
axC.text(5,1.1,"局部广义特征问题挑出真慢模——理论上对多尺度也 h 无关;\n需 SLEPc(本环境未装)。这是 2 层想真赢的正解。",
        ha="center",fontsize=8.2,color="0.15")

fig.savefig("fig_sys3coarse.png", dpi=140, bbox_inches="tight")
print("wrote fig_sys3coarse.png")
