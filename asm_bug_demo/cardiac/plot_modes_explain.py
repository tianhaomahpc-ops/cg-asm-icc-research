#!/usr/bin/env python3
"""plot_modes_explain.py -- visualize: mode, wavelength, slow mode, Green-function
decay (local vs global), fast-head/slow-tail. Residuals are MEASURED (-decay np=8).
Output: fig_modes_explain.png
"""
import matplotlib, numpy as np
matplotlib.use("Agg")
from matplotlib import font_manager
try:
    font_manager.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
    matplotlib.rcParams["font.family"] = "WenQuanYi Zen Hei"
except Exception:
    pass
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(15, 10.5))
gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.22)
fig.suptitle("模(mode)/ 波长 / 慢模 / Green 函数衰减 / 快头-慢尾", fontsize=17, weight="bold", y=0.985)

Lx = 20.0  # 最长轴长度 (mm)

# ============ (A) 什么是"模" + 波长:把每个模上下错开画 ============
axA = fig.add_subplot(gs[0,0])
x = np.linspace(0, Lx, 500)
modes = [(1,"#c0392b"), (3,"#e67e22"), (8,"#27ae60"), (16,"#2980b9")]
offsets = [9, 6, 3, 0]          # 每个模垂直错开,避免叠在一起
yticks, ylabs = [], []
for (k,col),off in zip(modes, offsets):
    y = 0.9*np.cos(k*np.pi*x/Lx) + off
    axA.plot(x, y, color=col, lw=2.2)
    axA.axhline(off, color="0.85", lw=0.6, zorder=0)
    wl = 2*Lx/k                 # 波长
    yticks.append(off); ylabs.append(f"k={k}")
    # 在右侧标注波长 + 快慢
    tag = "长波·光滑·慢模" if k<=1 else ("短波·振荡·快模" if k>=16 else "")
    axA.text(Lx+0.4, off, f"波长={wl:.1f}mm  {tag}", va="center", fontsize=9.5,
             color=col, weight=("bold" if tag else "normal"))
# 给 k=1 画一个波长双箭头(半周期 = Lx)
axA.annotate("", xy=(0, offsets[0]+1.4), xytext=(Lx, offsets[0]+1.4),
             arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.3))
axA.text(Lx/2, offsets[0]+1.9, "半个波跨过整个域 → 波长 40mm(最长)", ha="center",
         fontsize=9, color="#c0392b")
axA.set_title("① 一个'模'= 域上的一个空间形状(驻波);误差 = 这些形状的叠加",
              fontsize=11.5, weight="bold")
axA.set_xlabel("沿最长轴的位置 x (mm)")
axA.set_yticks(yticks); axA.set_yticklabels(ylabs, fontsize=11)
axA.set_ylabel("不同的模(纵轴 = 哪个模)")
axA.set_xlim(0, Lx*1.05); axA.set_ylim(-1.5, 12.2)
axA.text(0.2, 11.3, "↑ 波数 k 小 = 波长长 = 光滑 = 慢模", fontsize=9.5, color="#c0392b")
axA.text(0.2, -1.2, "↓ 波数 k 大 = 波长短 = 振荡 = 快模", fontsize=9.5, color="#2980b9")

# ============ (B) 特征值 vs 波数:小特征值 = 长波 = 慢模 ============
axB = fig.add_subplot(gs[0,1])
k = np.arange(1, 25)
lam = (np.pi*k/Lx)**2; lam_n = lam/lam.max()
axB.semilogy(k, lam_n, "o-", color="#8e44ad", lw=1.8, ms=5)
axB.axhspan(1e-3, lam_n[11], color="#f9e79f", alpha=0.6)
axB.text(13, 3.5e-3, "≈12 个小特征值\n= 慢模 = 慢尾", fontsize=10, color="#7d6608", weight="bold")
axB.annotate("k=1(最长波)→ 最小特征值\nλ_min ~ σ_L·(π/L_x)²\n= 最慢的模",
             xy=(1, lam_n[0]), xytext=(4.5, 1.8e-3), fontsize=9.5, color="#c0392b",
             arrowprops=dict(arrowstyle="->", color="#c0392b"))
axB.set_title("② 每个模的特征值:波越长 → 特征值越小 → 越慢", fontsize=11.5, weight="bold")
axB.set_xlabel("波数 k(小=长波)"); axB.set_ylabel("该模的特征值 λ(归一化, log)")
axB.grid(alpha=0.3)

# ============ (C) Green 函数衰减:局部 vs 全局 ============
axC = fig.add_subplot(gs[1,0])
r = np.linspace(0.3, Lx, 400); ell = 2.0
G1 = np.exp(-r/ell)/r; G1/=G1.max()
G2 = 1.0/r; G2/=G2.max()
axC.plot(r, G1, color="#16a085", lw=2.4, label="S1(含质量项/屏蔽):G~e^{-r/ℓ}/r → 快衰减=局部")
axC.plot(r, G2, color="#c0392b", lw=2.4, label="S2(纯 Laplace):G~1/r → 慢衰减=全局")
axC.axvline(ell, color="#16a085", ls=":", lw=1.2)
axC.text(ell+0.3, 0.5, "屏蔽长度 ℓ\n(影响半径)", fontsize=9, color="#16a085")
axC.set_title("③ Green 函数 = 一个点源的影响能传多远", fontsize=11.5, weight="bold")
axC.set_xlabel("离点源的距离 r (mm)"); axC.set_ylabel("响应 G(归一化)")
axC.legend(fontsize=8.5, loc="upper right"); axC.grid(alpha=0.3)

# ============ (D) 快头 + 慢尾:实测残差(np=8) ============
axD = fig.add_subplot(gs[1,1])
s2_it=[0,5,10,20,40,67]; s2_r=[1.0,2.2e-3,4.1e-4,3.0e-5,2.0e-6,9.7e-9]
s1_it=[0,5,8]; s1_r=[1.0,5.5e-6,5.6e-9]
s3_it=[0,5,10,20,40,51]; s3_r=[1.0,1.7e-3,2.9e-4,1.2e-5,2.6e-7,9.1e-9]
axD.semilogy(s1_it,s1_r,"s-",color="#16a085",lw=2,ms=6,label="S1:纯快头(8步,0慢模)")
axD.semilogy(s3_it,s3_r,"^-",color="#e67e22",lw=2,ms=6,label="S3:快头+慢尾(51步,8慢模)")
axD.semilogy(s2_it,s2_r,"o-",color="#c0392b",lw=2,ms=6,label="S2:快头+长慢尾(67步,12慢模)")
axD.axvspan(0,5,color="#d5f5e3",alpha=0.6); axD.axvspan(5,67,color="#fadbd8",alpha=0.35)
axD.text(2.5,2e-8,"快头\n(短波/局部模\n被光滑掉)",fontsize=8.5,color="#1e8449",ha="center")
axD.text(33,3e-3,"慢尾:磨那~12个慢模(长波全局模)",fontsize=9.5,color="#922b21",weight="bold")
axD.set_title("④ 残差 = 快头 + 慢尾(实测, np=8)", fontsize=11.5, weight="bold")
axD.set_xlabel("CG 迭代数"); axD.set_ylabel("相对残差 ‖r‖/‖r₀‖ (log)")
axD.legend(fontsize=9, loc="upper right"); axD.grid(alpha=0.3); axD.set_ylim(1e-9,3)

fig.savefig("fig_modes_explain.png", dpi=140, bbox_inches="tight")
print("wrote fig_modes_explain.png")
