#!/usr/bin/env python3
"""plot_profile_meaning.py -- three panels answering three questions on the n=100 chain:
(A) physical meaning of the profile view: the error is a FIELD over the geometry;
    iteration x<-x+alpha*r relaxes it -- jagged texture dies in a step or two,
    the big smooth arc lingers.  (profiles of e at k=0,1,5,20,150, Richardson a=0.25)
(B) the DEFINITION of a mode's speed: the per-step factor its coefficient is
    multiplied by (|1-alpha*lambda| here; |p_k(lambda)| for CG). slope = speed.
(C) why the profile view is needed even though only lambda matters for speed:
    the slow mode's SHAPE is smooth, so it can be assembled from per-subdomain
    constants (staircase, Nicolaides) or per-subdomain linears ({1,x}) WITHOUT
    ever computing the true eigenvector.  That is exactly the coarse space W.
Output: fig_profile_meaning.png
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
import matplotlib.ticker as mt

n = 100
A = np.diag(2.*np.ones(n)) + np.diag(-np.ones(n-1),1) + np.diag(-np.ones(n-1),-1)
lam, V = np.linalg.eigh(A)
alpha = 0.25
idx = np.arange(1, n+1)

# ---------- (A) error field profiles under Richardson ----------
rng = np.random.default_rng(0)
xstar = rng.standard_normal(n)
e = -xstar.copy()                      # x0 = 0
snaps = {0: e.copy()}
kmax = 150
for k in range(1, kmax+1):
    e = e - alpha*(A@e)                # e <- (I - alpha A) e
    if k in (1, 5, 20, 150): snaps[k] = e.copy()
print("factors: lam_min %.6f -> %.5f | lam_mid %.3f -> %.3f | lam_max %.3f -> %.4f" % (
    lam[0], 1-alpha*lam[0], lam[49], abs(1-alpha*lam[49]), lam[-1], abs(1-alpha*lam[-1])))

fig = plt.figure(figsize=(16.6, 5.9))
gs = fig.add_gridspec(1, 3, wspace=0.27)
fig.suptitle("画法B(把向量放回几何上)的物理意义 · 快慢的定义 · 为什么数轴之外还需要它 (n=100 链)",
             fontsize=14, weight="bold", y=1.02)

axA = fig.add_subplot(gs[0])
shades = ["#111111", "#7d3c98", "#c0392b", "#e67e22", "#f1c40f"]
for (k, ek), c in zip(snaps.items(), shades):
    axA.plot(idx, ek, color=c, lw=1.6 if k else 1.0, label=f"第 {k} 步", alpha=0.95)
axA.set_xlabel("珠子编号 i (= 几何位置)")
axA.set_ylabel("误差场 e_i (珠子 i 还差多少)")
axA.set_title("① 误差是铺在几何上的『场』\n迭代=松弛:毛刺(碎波)一两步抹平,\n大弧(长波)150 步几乎没动", fontsize=10.3, weight="bold")
axA.legend(fontsize=8.6, loc="lower right"); axA.grid(alpha=0.2)

# ---------- (B) definition of speed ----------
axB = fig.add_subplot(gs[1])
ks = np.arange(0, 151)
for li, c, nm in [(0, "#c0392b", "慢模"), (49, "#2980b9", "中间模"), (n-1, "#8e44ad", "快模")]:
    f = abs(1-alpha*lam[li])
    axB.semilogy(ks, np.clip(f**ks, 1e-8, None), color=c, lw=2.3,
                 label=f"{nm} λ={lam[li]:.4g}: 每步 x{f:.4g}")
axB.yaxis.set_major_formatter(mt.FuncFormatter(lambda v,_: f"1e{int(round(np.log10(v)))}" if v>0 else ""))
axB.set_xlabel("迭代步 k"); axB.set_ylabel("该模误差系数剩余比例 (log)")
axB.set_title("② 快慢的定义(不是比喻):\n模的速度 := 它的系数每步被乘的因子\n(这里 |1-αλ|;CG 是 |p_k(λ)|)。斜率即速度", fontsize=10.3, weight="bold")
axB.legend(fontsize=8.6, loc="upper right"); axB.grid(alpha=0.25); axB.set_ylim(1e-8, 2)

# ---------- (C) why the profile view: build W from geometry ----------
axC = fig.add_subplot(gs[2])
v1 = np.sin(idx*np.pi/(n+1)); v1 /= np.abs(v1).max()
nb = 4; blocks = np.array_split(np.arange(n), nb)
stair = np.zeros(n); plin = np.zeros(n)
for bl in blocks:
    stair[bl] = v1[bl].mean()
    t = idx[bl].astype(float)
    cfs = np.polyfit(t, v1[bl], 1)            # per-block linear fit = span{1, x}
    plin[bl] = np.polyval(cfs, t)
axC.plot(idx, v1, color="#c0392b", lw=2.8, label="真·最慢模 v1 (平滑大弧)")
axC.step(idx, stair, where="mid", color="0.25", lw=1.8,
         label="每子域一个常数 (Nicolaides)")
axC.plot(idx, plin, "--", color="#27ae60", lw=2.0, label="每子域一次函数 ({1,x})")
for bl in blocks[:-1]:
    axC.axvline(idx[bl[-1]]+0.5, color="0.75", ls=":", lw=1)
axC.set_xlabel("珠子编号 i (虚线 = 子区域边界)")
axC.set_ylabel("模的形状(剖面)")
axC.set_title("③ 为什么需要画法B:慢模的『形状』平滑\n→ 不算特征向量,用几何件就能拼近似\n→ 这就是粗空间 W(Sys2 实测:常数59步,线性40步)", fontsize=10.3, weight="bold")
axC.legend(fontsize=8.6, loc="lower center"); axC.grid(alpha=0.2)

fig.savefig("fig_profile_meaning.png", dpi=140, bbox_inches="tight")
print("wrote fig_profile_meaning.png")
err_st = np.linalg.norm(v1-stair)/np.linalg.norm(v1)
err_pl = np.linalg.norm(v1-plin)/np.linalg.norm(v1)
print(f"approx error of v1: staircase {err_st:.1%}, piecewise-linear {err_pl:.1%}")
