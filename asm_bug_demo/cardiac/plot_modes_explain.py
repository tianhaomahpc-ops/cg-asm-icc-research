#!/usr/bin/env python3
"""plot_modes_explain.py -- visualize the 5 concepts: mode, wavelength, slow mode,
Green-function decay (local vs global), fast-head/slow-tail residual decay.
All residual numbers are the MEASURED -decay output (np=8). Output: fig_modes_explain.png
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

fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.22)
fig.suptitle("模(mode)/ 波长 / 慢模 / Green 函数衰减 / 快头-慢尾  —— 一张图串起来",
             fontsize=17, weight="bold", y=0.98)

Lx = 20.0  # slab longest axis (mm)

# ============ (A) 什么是"模"和"波长":slab 上的正弦模 ============
axA = fig.add_subplot(gs[0,0])
x = np.linspace(0, Lx, 400)
for k, col, lab in [(1,"#c0392b","k=1 最长波(慢模)"),
                    (2,"#e67e22","k=2"),
                    (5,"#27ae60","k=5"),
                    (12,"#2980b9","k=12 短波(快模)")]:
    axA.plot(x, np.cos(k*np.pi*x/Lx) + 0*k, color=col, lw=2.0, label=lab)
axA.set_title("① 模 = 算子的特征向量;波长 = 一个模跨域振荡的空间尺度", fontsize=12, weight="bold")
axA.set_xlabel("沿最长轴 x (mm)"); axA.set_yticks([])
axA.legend(fontsize=9, loc="upper right", framealpha=0.9)
axA.text(0.5, -1.9, "波长 λ_wave = 2L/k :  k=1 → 40mm(横跨全域,光滑)   k=12 → 3.3mm(子域内,振荡)",
         fontsize=9.5, color="0.25")
axA.set_ylim(-2.3, 1.5)

# ============ (B) 特征值 vs 波数:小特征值 = 长波 = 慢模 ============
axB = fig.add_subplot(gs[0,1])
k = np.arange(1, 25)
lam = (np.pi*k/Lx)**2                 # stiffness eigenvalue ~ (k pi/L)^2 (Laplace)
lam_n = lam/lam.max()                 # normalize for display
axB.semilogy(k, lam_n, "o-", color="#8e44ad", lw=1.8, ms=5)
# mark the ~12 "slow" modes (small eigenvalue) region
axB.axhspan(1e-3, lam_n[11], color="#f9e79f", alpha=0.5)
axB.text(12, 4e-3, "≈12 个小特征值\n= 慢模(慢尾)", fontsize=10, color="#7d6608", weight="bold")
axB.annotate("k=1 最小特征值\nλ_min ~ σ_L(π/L_x)²\n= 最慢的模",
             xy=(1, lam_n[0]), xytext=(4, 2e-3),
             fontsize=9.5, color="#c0392b",
             arrowprops=dict(arrowstyle="->", color="#c0392b"))
axB.set_title("② 特征值 vs 波数:长波→小特征值→慢模", fontsize=12, weight="bold")
axB.set_xlabel("波数 k(越小=波越长)"); axB.set_ylabel("特征值(归一化,log)")
axB.grid(alpha=0.3)

# ============ (C) Green 函数衰减:Sys1 局部 vs Sys2 全局 ============
axC = fig.add_subplot(gs[1,0])
r = np.linspace(0.3, Lx, 400)
ell = 2.0  # screening length for Sys1 (illustrative): sqrt(sigma*dt/chiCm)
G1 = np.exp(-r/ell)/r; G1/=G1.max()          # screened (mass term) -> exp decay
G2 = 1.0/r; G2/=G2.max()                      # pure Laplace -> 1/r, slow/global
axC.plot(r, G1, color="#16a085", lw=2.4, label="Sys1 (含质量项/屏蔽): G~e^{-r/ℓ}/r  快衰减→局部")
axC.plot(r, G2, color="#c0392b", lw=2.4, label="Sys2 (纯 Laplace): G~1/r  慢衰减→全局")
axC.axvline(ell, color="#16a085", ls=":", lw=1.2)
axC.text(ell+0.3, 0.55, "屏蔽长度 ℓ\n(影响半径)", fontsize=9, color="#16a085")
axC.set_title("③ Green 函数 = 点源的影响范围:衰减快=局部(易),慢=全局(难)",
              fontsize=11.5, weight="bold")
axC.set_xlabel("离点源距离 r (mm)"); axC.set_ylabel("响应 G(归一化)")
axC.legend(fontsize=9, loc="upper right"); axC.grid(alpha=0.3)

# ============ (D) 快头 + 慢尾:实测残差(np=8) ============
axD = fig.add_subplot(gs[1,1])
# MEASURED from -decay, np=8
s2_it = [0,5,10,20,40,67]; s2_r=[1.0,2.2e-3,4.1e-4,3.0e-5,2.0e-6,9.7e-9]
s1_it = [0,5,8];           s1_r=[1.0,5.5e-6,5.6e-9]
s3_it = [0,5,10,20,40,51]; s3_r=[1.0,1.7e-3,2.9e-4,1.2e-5,2.6e-7,9.1e-9]
axD.semilogy(s1_it,s1_r,"s-",color="#16a085",lw=2,ms=6,label="Sys1: 纯快头(8步,0慢模)")
axD.semilogy(s3_it,s3_r,"^-",color="#e67e22",lw=2,ms=6,label="Sys3: 快头+慢尾(51步,8慢模)")
axD.semilogy(s2_it,s2_r,"o-",color="#c0392b",lw=2,ms=6,label="Sys2: 快头+长慢尾(67步,12慢模)")
axD.axvspan(0,5,color="#d5f5e3",alpha=0.6)
axD.axvspan(5,67,color="#fadbd8",alpha=0.4)
axD.text(2.5,3e-8,"快头\n(局部/短波\n模被光滑掉)",fontsize=8.5,color="#1e8449",ha="center")
axD.text(35,2e-3,"慢尾:磨那~12个慢模(长波全局模)",fontsize=9.5,color="#922b21",weight="bold")
axD.set_title("④ 残差 = 快头 + 慢尾(实测 -decay, np=8)", fontsize=12, weight="bold")
axD.set_xlabel("CG 迭代数"); axD.set_ylabel("相对残差 ‖r‖/‖r₀‖ (log)")
axD.legend(fontsize=9, loc="upper right"); axD.grid(alpha=0.3)
axD.set_ylim(1e-9, 3)

fig.savefig("fig_modes_explain.png", dpi=140, bbox_inches="tight")
print("wrote fig_modes_explain.png")
