#!/usr/bin/env python3
"""plot_cg_polynomial.py -- WHY CG converges mode-by-mode: the CG error is a
polynomial p_k(B) applied to e0, so mode i's error = c_i * p_k(lambda_i). The
CG 'filter' polynomial p_k(lambda) is pinned p_k(0)=1 and pressed small on
[lmin,lmax]; the bulk (large lambda) flattens in a few steps, the small
eigenvalues (slow modes) need many. Output: fig_cg_polynomial.png
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

lmin, lmax = 0.005, 1.5           # measured S2 spectrum extremes
# the classical CG residual polynomial (shifted Chebyshev, minimax on [lmin,lmax], p(0)=1)
def cheb(k, x):                    # T_k
    return np.cosh(k*np.arccosh(np.clip(np.abs(x),1,None)))*np.sign(x)**0 if False else np.polynomial.chebyshev.Chebyshev.basis(k)(x)
def pk(k, lam):
    t  = (lmax + lmin - 2*lam)/(lmax - lmin)     # maps [lmin,lmax] -> [-1,1]
    t0 = (lmax + lmin)/(lmax - lmin)             # image of lam=0 (>1)
    return np.polynomial.chebyshev.Chebyshev.basis(k)(t) / np.polynomial.chebyshev.Chebyshev.basis(k)(t0)

# model spectrum: 12 small eigenvalues (measured-ish) + a bulk cluster near lmax
small = np.array([0.005,0.013,0.019,0.032,0.048,0.054,0.059,0.071,0.09,0.11,0.13,0.15])
bulk  = np.linspace(0.35, 1.5, 40)
spec  = np.concatenate([small, bulk])

fig = plt.figure(figsize=(15, 6.2))
gs = fig.add_gridspec(1, 2, wspace=0.24)
fig.suptitle("为什么 CG 逐个波形消误差:误差 = p_k(B)·e0,第 i 个模的误差 = c_i·p_k(λ_i)",
             fontsize=15, weight="bold", y=1.0)

# ---- (A) the CG filter polynomial p_k(lambda) over the spectrum ----
axA = fig.add_subplot(gs[0,0])
lam = np.linspace(0, lmax, 800)
for k, col in [(3,"#2980b9"),(8,"#27ae60"),(20,"#c0392b")]:
    axA.plot(lam, pk(k, lam), color=col, lw=2.0, label=f"k={k} 步后的滤波器 p_k(λ)")
axA.axhline(0, color="0.6", lw=0.8); axA.axhline(1, color="0.85", lw=0.8, ls=":")
axA.plot(0, 1, "ko", ms=7); axA.annotate("p_k(0)=1\n(锚:误差不能免费消)",
          xy=(0,1), xytext=(0.35,0.72), fontsize=9.5, color="k",
          arrowprops=dict(arrowstyle="->"))
# spectrum ticks
axA.plot(small, np.zeros_like(small)-0.05, "v", color="#c0392b", ms=8, label="12 个小特征值(慢模)")
axA.plot(bulk,  np.zeros_like(bulk)-0.05,  "|", color="0.4", ms=8)
axA.text(0.9, -0.16, "bulk(快模)", color="0.4", fontsize=9)
axA.text(0.02, -0.20, "小 λ(慢模)", color="#c0392b", fontsize=9)
axA.annotate("模 i 的误差 ×= p_k(λ_i):\n这里 p_k≈0 → 消掉",
             xy=(1.0, pk(8,1.0)), xytext=(0.75,0.45), fontsize=9, color="#27ae60",
             arrowprops=dict(arrowstyle="->", color="#27ae60"))
axA.annotate("小 λ 处 p_k 还很大\n→ 慢模没消掉",
             xy=(0.02, pk(8,0.02)), xytext=(0.15,0.9), fontsize=9, color="#c0392b",
             arrowprops=dict(arrowstyle="->", color="#c0392b"))
axA.set_title("① CG 每步造一个'滤波器'p_k(λ):p_k(λ_i)=模 i 误差的缩放因子",
              fontsize=11, weight="bold")
axA.set_xlabel("特征值 λ(= 模的'快慢')"); axA.set_ylabel("滤波器值 p_k(λ)")
axA.set_ylim(-0.25, 1.15); axA.legend(fontsize=8.5, loc="upper right")
axA.grid(alpha=0.25)

# ---- (B) per-mode error vs iteration: bulk (fast) vs small (slow) ----
axB = fig.add_subplot(gs[0,1])
ks = np.arange(0, 68)
e_bulk = np.array([abs(pk(k, 1.0)) for k in ks]); e_bulk[0]=1
e_slow = np.array([abs(pk(k, lmin)) for k in ks]); e_slow[0]=1
axB.semilogy(ks, np.clip(e_bulk,1e-12,None), color="#27ae60", lw=2.2, label="快模 λ=1.0(bulk):几步消掉")
axB.semilogy(ks, np.clip(e_slow,1e-12,None), color="#c0392b", lw=2.2, label="慢模 λ=0.005(最小):磨很久")
axB.axvspan(0,5,color="#d5f5e3",alpha=0.5); axB.text(2,1e-9,"快头",color="#1e8449",ha="center",fontsize=10)
axB.text(35,3e-2,"慢尾",color="#922b21",fontsize=11,weight="bold")
axB.set_title("② 每个模的残差因子 |p_k(λ_i)| 随迭代:大 λ 快、小 λ 慢",
              fontsize=11, weight="bold")
axB.set_xlabel("CG 迭代数 k"); axB.set_ylabel("该模残差因子 |p_k(λ_i)| (log)")
axB.set_ylim(1e-10, 2); axB.legend(fontsize=9.5, loc="upper right"); axB.grid(alpha=0.25)

fig.savefig("fig_cg_polynomial.png", dpi=140, bbox_inches="tight")
print("wrote fig_cg_polynomial.png")
