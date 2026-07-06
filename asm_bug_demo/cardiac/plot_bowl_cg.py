#!/usr/bin/env python3
"""plot_bowl_cg.py -- high-school picture: solving Ax=b = finding the bottom of a
bowl. A stretched bowl (spread eigenvalues) is hard: steepest descent zigzags,
CG takes conjugate steps. The bowl's axes = eigenvectors = 'modes'; the long flat
axis = small eigenvalue = slow mode. Output: fig_bowl_cg.png
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

A = np.array([[1.0,0.0],[0.0,8.0]])     # eigenvalues 1 (flat/x) and 8 (steep/y)
def f(X,Y): return 0.5*(A[0,0]*X**2 + A[1,1]*Y**2)

x0 = np.array([9.0, 2.2])

# ---- steepest descent (exact line search) : zigzag ----
def sd(x, n):
    pts=[x.copy()]
    for _ in range(n):
        r = -A@x                       # residual = b - A x (b=0)
        a = (r@r)/(r@A@r)
        x = x + a*r; pts.append(x.copy())
    return np.array(pts)
# ---- CG : 2 steps to the bottom in 2D ----
def cg(x, n):
    pts=[x.copy()]; r=-A@x; p=r.copy()
    for _ in range(n):
        a=(r@r)/(p@A@p); x=x+a*p; rn=r-a*(A@p)
        b_=(rn@rn)/(r@r); p=rn+b_*p; r=rn; pts.append(x.copy())
    return np.array(pts)

sdp = sd(x0, 12); cgp = cg(x0, 2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.6))
fig.suptitle("解 Ax=b = 找'碗'的最低点;碗被拉长(特征值差得远)就难下",
             fontsize=15, weight="bold", y=0.99)

for ax, path, title, col in [(ax1, sdp, "① 最陡下降:朝最陡方向走 → 在细长山谷里来回横跳(慢)", "#c0392b"),
                             (ax2, cgp, "② CG:选'互不干扰'的共轭方向 → 2 步到底(快)", "#2980b9")]:
    xx=np.linspace(-10,10,300); yy=np.linspace(-4,4,300); XX,YY=np.meshgrid(xx,yy)
    ax.contour(XX,YY,f(XX,YY), levels=np.linspace(2,320,16), colors="0.7", linewidths=0.8)
    ax.plot(path[:,0],path[:,1],"o-",color=col,lw=2,ms=6,zorder=5)
    ax.plot(0,0,"*",color="#16a085",ms=20,zorder=6)
    ax.text(0.3,-0.5,"碗底 = 解",color="#16a085",fontsize=10,weight="bold")
    ax.text(path[0,0],path[0,1]+0.4,"起点(猜测)",fontsize=9,ha="center")
    # eigenvector axes = modes
    ax.annotate("", xy=(9,0), xytext=(-9,0), arrowprops=dict(arrowstyle="<->",color="#e67e22",lw=1.4))
    ax.annotate("", xy=(0,3.3), xytext=(0,-3.3), arrowprops=dict(arrowstyle="<->",color="#8e44ad",lw=1.4))
    ax.text(6.2,0.35,"长·平轴 = 小特征值(1) = 慢模",color="#e67e22",fontsize=9)
    ax.text(0.3,3.0,"短·陡轴\n= 大特征值(8)\n= 快模",color="#8e44ad",fontsize=8.5)
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_aspect("equal")
    ax.set_xlim(-10,10); ax.set_ylim(-4.2,4.2)

ax1.text(-9.5,-3.8,"步数多 = 慢尾",color="#c0392b",fontsize=10,weight="bold")
ax2.text(-9.5,-3.8,"碗的轴(特征向量)= 我们说的'模/波形'",color="#333",fontsize=9.5)

fig.savefig("fig_bowl_cg.png", dpi=140, bbox_inches="tight")
print("wrote fig_bowl_cg.png")
