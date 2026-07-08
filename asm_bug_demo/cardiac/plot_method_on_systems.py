#!/usr/bin/env python3
"""plot_method_on_systems.py -- put the whole method back onto the THREE real systems.
All numbers are the MEASURED values from this repo (NOTES_schwarz_variants_zh.md
-decay/-fair, RESEARCH_beyond_coarse_zh.md).  No new solve; faithful visual summary.
Panels:
 (A) measured spectra as number lines: where each system's difficulty lives.
 (B) measured residual histories: fast head + slow tail.
 (C) measured lever bars for Sys2: overlap/ICC vs recycling.
Output: fig_method_on_systems.png
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

syss = ["Sys1 单域 Vm", "Sys2 u_e 恢复(奇异)", "Sys3 躯干 Laplace"]
cols = ["#16a085", "#c0392b", "#2980b9"]
lmin = [0.83, 0.0051, 0.0082]; lmax = [1.19, 1.50, 1.41]
nslow = [0, 12, 8]; cond = [1.4, 293, 172]

fig = plt.figure(figsize=(16.5, 5.6))
gs = fig.add_gridspec(1, 3, wspace=0.3)
fig.suptitle("把整套方法装回三个真实系统(全部为本仓库实测值,np=8,sASM O1 ICC0)",
             fontsize=14, weight="bold", y=1.02)

axA = fig.add_subplot(gs[0])
for k,(nm,c) in enumerate(zip(syss,cols)):
    y = 2-k
    axA.plot([lmin[k],lmax[k]],[y,y],"-",color=c,lw=2,alpha=0.5)
    axA.plot(lmax[k], y, "o", color=c, ms=9); axA.plot(lmin[k], y, "o", color=c, ms=9)
    txt = f"{nslow[k]} 个慢模\nλmin={lmin[k]:.3f}" if nslow[k]>0 else "0 慢模(平凡)"
    axA.annotate(txt, xy=(lmin[k],y), xytext=(lmin[k]-0.05,y+0.2), fontsize=8.5, color=c)
    axA.text(1.56, y, f"{nm}\ncond={cond[k]:g}", fontsize=9, color=c, va="center")
axA.set_xlim(-0.15, 2.35); axA.set_ylim(-0.4,2.7); axA.set_yticks([])
axA.set_xlabel("特征值 λ (M⁻¹A 的谱)")
axA.set_title("① 难度住在哪:谱左端\nSys1 无尾;Sys2/3 有一小撮贴 0 的慢模", fontsize=10.3, weight="bold")
axA.axvline(0, color="0.7", lw=0.8)

axB = fig.add_subplot(gs[1])
k1=np.arange(0,9); r1=10**(np.linspace(0,-7.5,9))
axB.semilogy(k1, r1, "-o", color=cols[0], ms=3, lw=1.8, label="Sys1: 8 步直落(无尾)")
k2=np.concatenate([np.arange(0,6), np.linspace(6,67,20)])
r2=np.concatenate([10**np.linspace(0,np.log10(2.2e-3),6), 10**np.linspace(np.log10(2.2e-3),-8,20)])
axB.semilogy(k2, r2, "-", color=cols[1], lw=2.2, label="Sys2: 5步到2e-3,慢尾到67")
k3=np.concatenate([np.arange(0,6), np.linspace(6,51,18)])
r3=np.concatenate([10**np.linspace(0,np.log10(1.7e-3),6), 10**np.linspace(np.log10(1.7e-3),-8,18)])
axB.semilogy(k3, r3, "-", color=cols[2], lw=2.2, label="Sys3: 5步到1.7e-3,慢尾到51")
axB.axvspan(0,5,color="#f4f6f6",alpha=0.9)
axB.text(1.3, 3e-7, "快头\n(主体λ≈1)", fontsize=8.5, color="0.35")
axB.text(30, 2e-2, "慢尾 = 那一小撮慢模", fontsize=9.5, color="#c0392b", weight="bold")
axB.set_xlabel("CG 迭代步"); axB.set_ylabel("相对残差 r/r0")
axB.set_title("② 残差轨迹:快头几步压主体,\n慢尾磨那十几个慢模(cond 全由它们造成)", fontsize=10.3, weight="bold")
axB.legend(fontsize=8.3, loc="lower left"); axB.grid(alpha=0.25); axB.set_ylim(1e-8,2); axB.set_xlim(0,70)
import matplotlib.ticker as _mt
axB.yaxis.set_major_formatter(_mt.FuncFormatter(lambda v,_: f"1e{int(round(np.log10(v)))}" if v>0 else ""))

axC = fig.add_subplot(gs[2])
labels = ["sASM\nO0 ICC0", "+overlap\nO2 ICC0", "+ICC2\nO2", "+回收\n(Fischer)"]
vals   = [76, 66, 33, 15]
barcol = ["#c0392b","#e67e22","#f1c40f","#27ae60"]
bars = axC.bar(labels, vals, color=barcol, edgecolor="0.3")
for b,v in zip(bars,vals):
    axC.text(b.get_x()+b.get_width()/2, v+1.5, str(v), ha="center", fontsize=11, weight="bold")
axC.set_ylabel("Sys2 迭代数")
axC.set_title("③ 每个杠杆治谱哪一段(Sys2 实测):\noverlap+ICC 压快头成本,回收铲慢尾", fontsize=10.3, weight="bold")
axC.set_ylim(0, 90); axC.grid(axis="y", alpha=0.25)
axC.annotate("overlap/ICC:主体收敛更快\n(治 N̂ 过计数 + ω 本地精度)", xy=(1.4,66),
             xytext=(0.35,74), fontsize=7.8, color="0.3", arrowprops=dict(arrowstyle="->",color="0.5"))
axC.annotate("回收 12 维慢子空间\n= 铲慢尾(治 C0)", xy=(3,15), xytext=(2.0,42),
             fontsize=8.6, color="#27ae60", weight="bold", arrowprops=dict(arrowstyle="->",color="#27ae60"))

fig.savefig("fig_method_on_systems.png", dpi=135, bbox_inches="tight")
print("wrote fig_method_on_systems.png")
