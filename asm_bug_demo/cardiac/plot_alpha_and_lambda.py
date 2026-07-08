#!/usr/bin/env python3
"""plot_alpha_and_lambda.py -- ONE self-contained arithmetic walkthrough.
No 'energy', no calculus. Only: residual r=b-Ax, update x <- x + alpha*r.

Part 1 (printed): A=diag(4,1,0.1), b=(4,1,0.1), answer x*=(1,1,1).
  Three independent one-unknown equations. Hand-checkable tables:
  error factor per step = (1-alpha*lambda); alpha must be < 2/lambda_max;
  the small-lambda coordinate crawls no matter what alpha is.
Part 2 (printed): non-diagonal A=[[3,1],[1,3]]: the SAME step computed two
  ways (direct matrix arithmetic vs eigen-direction bookkeeping) gives the
  SAME numbers. That equality IS the eigenvalue<->solve correspondence.
Figure: (L) per-coordinate |error| vs step at alpha=0.25;
        (R) shrink factor |1-alpha*lambda| vs alpha: the cap at 2/lambda_max
            and the floor ~0.95 for lambda=0.1. Output: fig_alpha_and_lambda.png
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

print("="*72)
print("PART 1: A=diag(4,1,0.1), b=(4,1,0.1), x*=(1,1,1), start x0=(0,0,0)")
print("rule:  x <- x + alpha*(b - A x)   (only multiply, never divide)")
print("="*72)
lams = np.array([4.0, 1.0, 0.1]); b = lams*1.0   # x* = all ones
alpha = 0.25
print(f"\nalpha = {alpha}: first 3 steps, per coordinate (hand-checkable)")
x = np.zeros(3)
for k in range(1, 4):
    r = b - lams*x
    x = x + alpha*r
    print(f"  step {k}: r={np.round(r,4)}  ->  x={np.round(x,6)}   error e=x-1={np.round(x-1,6)}")
print("\nper-step error factor (1 - alpha*lambda):")
for lam in lams:
    f = 1 - alpha*lam
    print(f"  lambda={lam:4}:  factor = 1 - {alpha}*{lam} = {f:.3f}")
print("\nsteps until |e| < 1e-6  (start |e|=1):")
for lam in lams:
    f = abs(1-alpha*lam)
    if f == 0: k = 1
    else:      k = int(np.ceil(np.log(1e-6)/np.log(f)))
    print(f"  lambda={lam:4}:  factor {f:.3f}  ->  {k} steps")
print("\nwhy alpha is capped: need |1-alpha*lambda|<1 for EVERY lambda")
print("  -> alpha < 2/lambda_max = 2/4 = 0.5")
print("try alpha=0.6 (over the cap): factor for lambda=4 is 1-2.4 = -1.4")
e = 1.0
for k in range(1, 11): e *= -1.4
print(f"  after 10 steps |e| = 1.4^10 = {abs(e):.1f}   -> EXPLODES")
print("\nbest possible alpha = 2/(lam_max+lam_min) = 2/4.1 = 0.4878:")
aopt = 2/(lams.max()+lams.min())
worst = max(abs(1-aopt*lams.max()), abs(1-aopt*lams.min()))
print(f"  worst factor = {worst:.4f}  ->  {int(np.ceil(np.log(1e-6)/np.log(worst)))} steps")
print("  => tuning alpha CANNOT save the small-lambda coordinate.")
print("\ndeflation, toy version: solve row 3 (0.1*x3=0.1) by ONE division -> x3=1.")
print("  remaining slowest is lambda=1: 49 steps instead of 546.")

print()
print("="*72)
print("PART 2: non-diagonal A=[[3,1],[1,3]], b=(6,2), x*=(2,0), x0=(0,0)")
print("="*72)
A = np.array([[3.,1.],[1.,3.]]); b2 = np.array([6.,2.]); xs = np.array([2.,0.])
print("eigen-direction check by plain multiplication:")
print(f"  A*(1,1)  = {A@np.array([1.,1.])}  = 4*(1,1)   -> lambda=4, direction (1,1)")
print(f"  A*(1,-1) = {A@np.array([1.,-1.])} = 2*(1,-1)  -> lambda=2, direction (1,-1)")
e0 = np.array([0.,0.]) - xs
print(f"\ne0 = x0-x* = {e0} = -1*(1,1) + -1*(1,-1)   (check: -1-1=-2, -1+1=0)")
# route 1: direct arithmetic
e1_direct = e0 - alpha*(A@e0)
e2_direct = e1_direct - alpha*(A@e1_direct)
print(f"route 1 (direct):      e1 = e0 - 0.25*A*e0 = {e1_direct}")
print(f"                       e2 = e1 - 0.25*A*e1 = {e2_direct}")
# route 2: eigen bookkeeping
c = np.array([-1.,-1.])                       # components along (1,1),(1,-1)
f = 1-alpha*np.array([4.,2.])                 # per-direction factors 0, 0.5
c1 = f*c; c2 = f*c1
u1 = np.array([1.,1.]); u2 = np.array([1.,-1.])
print(f"route 2 (eigen): factors per direction = {f}  (=1-0.25*4, 1-0.25*2)")
print(f"                       e1 = {c1[0]}*(1,1) + {c1[1]}*(1,-1) = {c1[0]*u1 + c1[1]*u2}")
print(f"                       e2 = {c2[0]}*(1,1) + {c2[1]}*(1,-1) = {c2[0]*u1 + c2[1]*u2}")
print("SAME numbers. After step 1 ALL remaining error lies along (1,-1), the")
print("slow (lambda=2) direction; every further step just multiplies it by 0.5.")

# =================== figure ===================
fig, (axL, axR) = plt.subplots(1, 2, figsize=(15.2, 6.0))
fig.suptitle("只用一条规则 x <- x + α·(b - A·x):特征值 λ = 每个方向误差的'每步折扣率' 1-αλ",
             fontsize=14.5, weight="bold", y=1.0)

# (L) per-coordinate error decay at alpha=0.25
cols = ["#8e44ad", "#2980b9", "#c0392b"]
ks = np.arange(0, 101)
for lam, c in zip(lams, cols):
    f = abs(1-alpha*lam)
    err = np.clip(f**ks, 1e-8, None)
    axL.semilogy(ks, err, color=c, lw=2.4,
                 label=f"λ={lam}: 每步 x{f:.3f}")
axL.axhline(1e-6, color="0.55", ls=":", lw=1.2)
axL.text(72, 2e-6, "目标 1e-6", color="0.4", fontsize=9)
axL.set_title("① α=0.25 时三行方程各自的误差:到 1e-6 分别要 1 / 49 / 546 步",
              fontsize=11, weight="bold")
axL.set_xlabel("迭代步 k"); axL.set_ylabel("该坐标误差 |e| (log)")
import matplotlib.ticker as _mt
axL.yaxis.set_major_formatter(_mt.FuncFormatter(lambda v,_: f"1e{int(round(np.log10(v)))}" if v>0 else ""))
axL.annotate("λ=4: 1 步", xy=(1,1e-8), xytext=(8,1e-7), fontsize=10, color="#8e44ad",
             arrowprops=dict(arrowstyle="->", color="#8e44ad"))
axL.annotate("λ=1: 49 步", xy=(49,1e-6), xytext=(55,1e-4), fontsize=10, color="#2980b9",
             arrowprops=dict(arrowstyle="->", color="#2980b9"))
axL.text(55, 0.35, "λ=0.1: 每步只砍 2.5%\n100 步还剩 8%,共要 546 步", fontsize=10, color="#c0392b")
axL.legend(fontsize=9.5, loc="lower left"); axL.grid(alpha=0.25); axL.set_ylim(1e-8, 2)

# (R) factor vs alpha: the cap and the floor
aa = np.linspace(0, 0.7, 400)
for lam, c in zip(lams, cols):
    axR.plot(aa, np.abs(1-aa*lam), color=c, lw=2.4, label=f"|1-αλ|, λ={lam}")
axR.axhline(1, color="k", lw=1.2)
axR.text(0.62, 1.03, "=1:不收敛", fontsize=9)
axR.axvspan(0, 0.5, color="#d5f5e3", alpha=0.5)
axR.axvline(0.5, color="#c0392b", ls="--", lw=1.5)
axR.text(0.505, 0.25, "上限 α=2/λmax=0.5\n(超过它 λ=4 的因子>1,爆炸)", fontsize=9, color="#c0392b")
axR.axvline(alpha, color="0.4", ls=":", lw=1.4)
axR.text(0.255, 0.06, "本文取 α=0.25\n(让 λ=4 恰好一步归零,\n绿色区间里怎么选都行)", fontsize=9, color="0.25")
axR.annotate("λ=0.1 的因子在整个允许区间\n都贴着 1(最低也只有 0.95)\n-> 调 α 救不了慢方向", xy=(0.45,0.955),
             xytext=(0.13,0.55), fontsize=9.5, color="#c0392b",
             arrowprops=dict(arrowstyle="->", color="#c0392b"))
axR.set_title("② α 不是随便取:必须 < 2/λmax(由最大 λ 定);\n而小 λ 的折扣率在允许区间内永远贴着 1",
              fontsize=11, weight="bold")
axR.set_xlabel("步长 α"); axR.set_ylabel("每步误差因子 |1-αλ|")
axR.set_xlim(0, 0.7); axR.set_ylim(0, 1.45)
axR.legend(fontsize=9.5, loc="upper center"); axR.grid(alpha=0.25)

fig.savefig("fig_alpha_and_lambda.png", dpi=140, bbox_inches="tight")
print("\nwrote fig_alpha_and_lambda.png")
