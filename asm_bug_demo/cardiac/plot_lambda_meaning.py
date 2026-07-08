#!/usr/bin/env python3
"""plot_lambda_meaning.py -- the ONE missing link: what an eigenvalue *does* during
a solve.  Use a DIAGONAL matrix A=diag(4,1,0.1) so the eigenvalue IS the number on
the diagonal.  Solving Ax=b decouples into 3 independent 1-variable equations
lambda_i * x_i = b_i.  An iterative solver can only MULTIPLY by A (never divide),
so per coordinate it does  x <- x + alpha(b - lambda x),  whose error shrinks by
(1-alpha*lambda) each step.  lambda = curvature of that coordinate's private
parabola = strength of the pull to the bottom = convergence speed.

Left : the 3 private parabolas (curvature = lambda); a ball rolling to each bottom.
Right: error per coordinate vs iteration = (1-alpha*lambda)^k.  Big lambda snaps,
       tiny lambda crawls.  THAT is 'small eigenvalue = slow mode'.
Output: fig_lambda_meaning.png
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

lams  = [4.0, 1.0, 0.1]                       # the 3 eigenvalues (= diagonal entries)
cols  = ["#8e44ad", "#2980b9", "#c0392b"]
names = ["λ=4 (陡碗, 快模)", "λ=1 (中等)", "λ=0.1 (平碗, 慢模)"]
alpha = 0.25                                  # step: 0 < alpha < 2/lambda_max = 0.5
xstar_all = 1.0                               # put every bowl's bottom at x*=1
                                              # (so only curvature=lambda differs)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(15.5, 6.4))
fig.suptitle("特征值到底在'解'里干什么:每个 λ = 一口碗的陡峭度;解 = 小球滚到碗底",
             fontsize=15, weight="bold", y=1.0)

# ---------- LEFT: the private parabolas + rolling ball ----------
xx = np.linspace(-1.5, 3.2, 400)
for lam, c, nm in zip(lams, cols, names):
    xstar = xstar_all                         # bottom of this parabola (all at x*=1)
    f = 0.5*lam*(xx-xstar)**2                 # private energy: curvature = lambda
    axL.plot(xx, f, color=c, lw=2.2, label=nm)
    axL.plot(xstar, 0, "*", color=c, ms=16, zorder=6)
    # a ball started at x=-1, first gradient-descent step
    x0 = -1.0; g = lam*(x0-xstar); x1 = x0 - alpha*g
    axL.plot([x0],[0.5*lam*(x0-xstar)**2],"o",color=c,ms=9,zorder=5)
    axL.annotate("", xy=(x1,0.5*lam*(x1-xstar)**2), xytext=(x0,0.5*lam*(x0-xstar)**2),
                 arrowprops=dict(arrowstyle="-|>",color=c,lw=1.6))
axL.set_title("① 转到特征向量坐标后,方程拆成 3 口互不相干的碗\n"
              "碗壁陡=λ 大=拉力强;碗壁平=λ 小=拉力弱", fontsize=11, weight="bold")
axL.set_xlabel("这个坐标 x_i  (★=碗底=答案,三口碗答案都在 x*=1,只有陡峭度不同)")
axL.set_ylabel("能量 f_i(x_i) = ½·λ_i·(x_i - x*_i)²")
axL.set_ylim(-0.4, 6); axL.set_xlim(-1.5, 3.2)
axL.legend(fontsize=9.5, loc="upper right"); axL.grid(alpha=0.25)
axL.text(-1.35, 5.2, "小球从同一个起点松手\n陡碗一步冲到底,平碗几乎不动", fontsize=9, color="0.3")

# ---------- RIGHT: error per coordinate vs iteration ----------
ks = np.arange(0, 60)
for lam, c, nm in zip(lams, cols, names):
    factor = abs(1 - alpha*lam)               # per-step shrink factor
    err = factor**ks
    axR.semilogy(ks, np.clip(err,1e-8,2), color=c, lw=2.4,
                 label=f"{nm}: 每步 ×{factor:.2f}")
axR.axhline(1e-6, color="0.6", ls=":", lw=1)
axR.set_title("② 迭代法只会'乘 A'不会'除':每坐标误差每步 ×(1-αλ)\n"
              "λ 大 → 因子小 → 秒收敛;λ 小 → 因子≈1 → 磨很久", fontsize=11, weight="bold")
axR.set_xlabel("迭代步 k"); axR.set_ylabel("该坐标还剩的误差 (log)")
axR.set_ylim(1e-8, 2); axR.legend(fontsize=10, loc="upper right"); axR.grid(alpha=0.25)
import matplotlib.ticker as _mt
axR.yaxis.set_major_formatter(_mt.FuncFormatter(lambda v,_: f"1e{int(round(np.log10(v)))}" if v>0 else ""))
axR.annotate("λ=4:一步 ×0 → 直接到底", xy=(1,1e-7), xytext=(6,3e-6),
             fontsize=9, color="#8e44ad", arrowprops=dict(arrowstyle="->",color="#8e44ad"))
axR.annotate("λ=0.1:每步 ×0.975\n几乎横着走 = 慢尾", xy=(45,0.975**45), xytext=(22,0.06),
             fontsize=9.5, color="#c0392b", arrowprops=dict(arrowstyle="->",color="#c0392b"))

fig.savefig("fig_lambda_meaning.png", dpi=140, bbox_inches="tight")
print("wrote fig_lambda_meaning.png")
for lam in lams:
    print(f"lambda={lam}:  per-step factor |1-alpha*lambda|={abs(1-alpha*lam):.3f}")
