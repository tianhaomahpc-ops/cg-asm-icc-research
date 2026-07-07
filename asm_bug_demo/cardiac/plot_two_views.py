#!/usr/bin/env python3
"""plot_two_views.py -- one 2x2 example, two routes + the bridge.
A=[[3,1],[1,3]], b=[6,2], x*=[2,0]; eigen: lambda=4 dir[1,1] (fast), lambda=2 dir[1,-1] (slow).
(A) eigenvalue route: decompose b, divide each component by its lambda, recombine.
(B) CG route on the energy-bowl contours (no eigenvalues used).
(C) the bridge: CG's per-mode residual = p_k(lambda) evaluated at the eigenvalues.
Output: fig_two_views.png
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

A=np.array([[3.,1.],[1.,3.]]); b=np.array([6.,2.]); xstar=np.array([2.,0.])
u1=np.array([1,1.]); u2=np.array([1,-1.])   # eigen-directions (unnormalized), lam 4, 2

fig=plt.figure(figsize=(16,5.4))
gs=fig.add_gridspec(1,3,wspace=0.28)
fig.suptitle("一个例子两条路:A=[[3,1],[1,3]], b=[6,2], 解 x*=[2,0]", fontsize=14, weight="bold", y=1.0)

# ---- (A) eigenvalue route ----
axA=fig.add_subplot(gs[0,0]); axA.set_aspect("equal")
for u,c in [(u1,"#8e44ad"),(u2,"#e67e22")]:
    axA.plot([-1*u[0],7*u[0]],[-1*u[1],7*u[1]],"--",color=c,lw=1,alpha=0.6)
# b = 4*[1,1] + 2*[1,-1]
axA.annotate("",xy=b,xytext=(0,0),arrowprops=dict(arrowstyle="-|>",color="k",lw=2.2))
axA.text(b[0]+0.1,b[1]+0.1,"b=[6,2]",fontsize=10,weight="bold")
axA.annotate("",xy=4*u1,xytext=(0,0),arrowprops=dict(arrowstyle="-|>",color="#8e44ad",lw=1.6))
axA.annotate("",xy=b,xytext=(4*u1),arrowprops=dict(arrowstyle="-|>",color="#e67e22",lw=1.6))
axA.text(4*u1[0]-1.4,4*u1[1]+0.2,"沿[1,1]:4",color="#8e44ad",fontsize=9)
axA.text(5.1,3.0,"沿[1,-1]:2",color="#e67e22",fontsize=9)
# x* = 1*[1,1] + 1*[1,-1]
axA.annotate("",xy=1*u1,xytext=(0,0),arrowprops=dict(arrowstyle="-|>",color="#8e44ad",lw=2.4))
axA.annotate("",xy=xstar,xytext=(1*u1),arrowprops=dict(arrowstyle="-|>",color="#e67e22",lw=2.4))
axA.plot(*xstar,"*",color="#16a085",ms=18)
axA.text(xstar[0]-0.1,xstar[1]-0.55,"x*=[2,0]",color="#16a085",fontsize=10,weight="bold")
axA.text(0.15,0.55,"÷4",color="#8e44ad",fontsize=11,weight="bold")
axA.text(1.35,0.15,"÷2",color="#e67e22",fontsize=11,weight="bold")
axA.set_title("① 用特征值:b 拆两方向 → 各除以自己的 λ → 拼回\n(快模λ=4:4→1 除4;慢模λ=2:2→1 除2)",fontsize=10)
axA.set_xlim(-1,7); axA.set_ylim(-1.5,4.5); axA.grid(alpha=0.25)

# ---- (B) CG route on bowl ----
axB=fig.add_subplot(gs[0,1]); axB.set_aspect("equal")
xx=np.linspace(-1.5,4.5,300); yy=np.linspace(-3,3,300); XX,YY=np.meshgrid(xx,yy)
F=0.5*(A[0,0]*XX**2+2*A[0,1]*XX*YY+A[1,1]*YY**2)-(b[0]*XX+b[1]*YY)
axB.contour(XX,YY,F,levels=20,colors="0.75",linewidths=0.7)
path=np.array([[0,0],[1.6667,0.5556],[2,0]])
axB.plot(path[:,0],path[:,1],"o-",color="#c0392b",lw=2.2,ms=7,zorder=5)
for i,pt in enumerate(path): axB.text(pt[0]+0.08,pt[1]+0.12,f"x{i}",color="#c0392b",fontsize=9)
axB.plot(2,0,"*",color="#16a085",ms=18,zorder=6)
for u,c,lab in [(u1,"#8e44ad","快轴 λ=4"),(u2,"#e67e22","慢轴 λ=2")]:
    axB.plot([2-1.4*u[0],2+1.4*u[0]],[0-1.4*u[1],0+1.4*u[1]],"--",color=c,lw=1.2)
axB.text(3.2,1.5,"快轴 λ=4",color="#8e44ad",fontsize=9)
axB.text(0.2,1.6,"慢轴 λ=2",color="#e67e22",fontsize=9)
axB.set_title("② CG:在'碗'里走共轭方向,2 步到底\n(只用 A·向量,从不算特征值)",fontsize=10)
axB.set_xlim(-1.5,4.5); axB.set_ylim(-3,3)

# ---- (C) the bridge: p_k(lambda) ----
axC=fig.add_subplot(gs[0,2])
lam=np.linspace(0,5,300)
p1=1-0.2778*lam
p2=(1-lam/2)*(1-lam/4)
axC.axhline(0,color="0.6",lw=0.8)
axC.plot(lam,1+0*lam,color="#95a5a6",lw=1.6,label="p0(λ)=1 (第0步)")
axC.plot(lam,p1,color="#2980b9",lw=2,label="p1(λ)=1−0.278λ (第1步)")
axC.plot(lam,p2,color="#c0392b",lw=2,label="p2(λ)=(1−λ/2)(1−λ/4) (第2步)")
axC.plot(0,1,"ko",ms=7); axC.text(0.08,1.03,"锚 p_k(0)=1",fontsize=9)
# eigenvalues
for lv,c in [(2,"#e67e22"),(4,"#8e44ad")]:
    axC.axvline(lv,color=c,ls=":",lw=1.2)
axC.plot([4,2],[1-0.2778*4,1-0.2778*2],"o",color="#2980b9",ms=8)
axC.text(4.05,-0.16,"λ=4 快模\np1=−0.11",color="#8e44ad",fontsize=8.5)
axC.text(2.05,0.5,"λ=2 慢模\np1=+0.44",color="#e67e22",fontsize=8.5)
axC.plot([2,4],[0,0],"s",color="#c0392b",ms=8)
axC.text(2.4,-0.35,"第2步:p2 的根正好落在两个 λ 上 → 两模都=0 → 到底",
         color="#c0392b",fontsize=8.5)
axC.set_title("③ 桥梁:CG 每步每个模的残余 = p_k(λ) 在该 λ 的高度",fontsize=10)
axC.set_xlabel("特征值 λ"); axC.set_ylabel("残余因子 p_k(λ)")
axC.set_xlim(0,5); axC.set_ylim(-0.6,1.2); axC.legend(fontsize=8,loc="lower left"); axC.grid(alpha=0.25)

fig.savefig("fig_two_views.png",dpi=140,bbox_inches="tight")
print("wrote fig_two_views.png")
