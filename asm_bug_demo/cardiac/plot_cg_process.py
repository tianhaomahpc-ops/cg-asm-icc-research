#!/usr/bin/env python3
"""plot_cg_process.py -- THE CG PROCESS itself, nothing else.
A=[[3,1],[1,3]], b=(6,2), x*=(2,0).  Two steps, every quantity labeled.
No eigenvalues anywhere.  Output: fig_cg_process.png
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

A=np.array([[3.,1.],[1.,3.]]); b=np.array([6.,2.]); xs=np.array([2.,0.])
x0=np.array([0.,0.]); x1=np.array([1.6667,0.5556]); x2=xs

fig, ax = plt.subplots(figsize=(9.8, 7.2))
fig.suptitle("CG 的过程本身:猜测 x + 残差 r + 方向 p,走两步到解 (A=[[3,1],[1,3]], b=(6,2))",
             fontsize=12.5, weight="bold")
xx=np.linspace(-0.8,3.6,300); yy=np.linspace(-1.6,2.0,300)
XX,YY=np.meshgrid(xx,yy)
# level sets of the error size (A-norm), just as background scenery
F=0.5*(3*XX**2+2*XX*YY+3*YY**2)-(6*XX+2*YY)
ax.contour(XX,YY,F,levels=18,colors="0.82",linewidths=0.8)

# step 1
ax.annotate("", xy=x1, xytext=x0, arrowprops=dict(arrowstyle="-|>",color="#c0392b",lw=3))
ax.plot(*x0,"o",color="k",ms=9); ax.text(x0[0]-0.08,x0[1]-0.28,"x0=(0,0) 起点",fontsize=10)
ax.text(0.55,0.62,"第1步:沿 p0=r0=(6,2)\n走 α0=40/144=0.278\n(这条线上误差最低的点)",
        fontsize=9.5,color="#c0392b")
# r1 perpendicular
ax.annotate("", xy=x1+0.35*np.array([0.4444,-1.3333]), xytext=x1,
            arrowprops=dict(arrowstyle="-|>",color="#2980b9",lw=2))
ax.text(1.98,0.75,"新残差 r1=(0.44,-1.33)\n自动垂直于刚走过的 p0\n(这方向的活干完了)",fontsize=9.5,color="#2980b9")
# step 2
ax.annotate("", xy=x2, xytext=x1, arrowprops=dict(arrowstyle="-|>",color="#27ae60",lw=3))
ax.text(2.28,0.18,"第2步:方向 p1=r1+0.049*p0\n(掺一点旧方向 → 与 p0 共轭,\n不破坏第1步的成果)\n走 α1=0.45",fontsize=9.5,color="#27ae60")
ax.plot(*x1,"o",color="#c0392b",ms=9); ax.text(x1[0]-0.55,x1[1]+0.15,"x1=(1.67, 0.56)",fontsize=10)
ax.plot(*x2,"*",color="#16a085",ms=22)
ax.text(x2[0]-0.15,x2[1]-0.35,"x2=(2,0)=解,残差=0",fontsize=11,weight="bold",color="#16a085")
ax.text(-0.6,-1.35,"背景等高线 = 误差大小的地形(只是布景,CG 不看它,\nCG 只反复算:A·p、两个点积、加减)",fontsize=9,color="0.4")
ax.set_xlim(-0.8,3.6); ax.set_ylim(-1.6,2.0); ax.set_aspect("equal")
ax.set_xlabel("x 的第1个分量"); ax.set_ylabel("x 的第2个分量")
fig.savefig("fig_cg_process.png", dpi=140, bbox_inches="tight")
print("wrote fig_cg_process.png")
