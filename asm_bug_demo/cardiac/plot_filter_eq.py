#!/usr/bin/env python3
"""plot_filter_eq.py -- the CG polynomial p_k(lambda) as an EQ 'filter': gain over
the eigenvalue axis. Mode bars (at lambda=2,4) get multiplied by the filter gain
|p_k(lambda)|; p_k(0)=1 is the pinned knob. k=0/1/2 for A=[[3,1],[1,3]] example.
Output: fig_filter_eq.png
"""
import matplotlib, numpy as np
matplotlib.use("Agg")
from matplotlib import font_manager
try:
    font_manager.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
    matplotlib.rcParams["font.family"]="WenQuanYi Zen Hei"
except Exception: pass
matplotlib.rcParams["axes.unicode_minus"]=False
import matplotlib.pyplot as plt

lam=np.linspace(0,5,400)
def p0(l): return np.ones_like(l)
def p1(l): return 1-0.2778*l
def p2(l): return (1-l/2)*(1-l/4)
polys=[(p0,"p0(λ)=1  (第0步:没开始)"),
       (p1,"p1(λ)=1−0.278λ  (第1步)"),
       (p2,"p2(λ)=(1−λ/2)(1−λ/4)  (第2步)")]
eigs=[(2,"#e67e22","慢模 λ=2"),(4,"#8e44ad","快模 λ=4")]

fig,axes=plt.subplots(1,3,figsize=(15.5,4.8),sharey=True)
fig.suptitle("把 CG 的 p_k(λ) 当'均衡器':每个模的误差 × 该 λ 处的增益 |p_k(λ)|",
             fontsize=14,weight="bold",y=1.02)
for ax,(pf,title) in zip(axes,polys):
    g=np.abs(pf(lam))
    ax.plot(lam,g,color="#2980b9",lw=2.2,zorder=3,label="滤波器增益 |p_k(λ)|")
    ax.fill_between(lam,0,g,color="#2980b9",alpha=0.08)
    # mode bars = |p_k(lambda_i)|
    for lv,c,lab in eigs:
        h=abs(pf(np.array([lv]))[0])
        ax.bar(lv,h,width=0.22,color=c,zorder=4)
        ax.text(lv,h+0.04,f"{h:.2f}",ha="center",color=c,fontsize=9,weight="bold")
        ax.axvline(lv,color=c,ls=":",lw=1,alpha=0.5)
    ax.plot(0,1,"ko",ms=7,zorder=5); ax.text(0.1,1.06,"锚 p_k(0)=1\n(拧不动)",fontsize=8.5)
    ax.set_title(title,fontsize=10.5)
    ax.set_xlabel("特征值 λ  (= 哪个模)"); ax.set_xlim(-0.2,5); ax.set_ylim(0,1.25)
    ax.grid(alpha=0.25)
axes[0].set_ylabel("增益 = 该模误差还剩几分之几")
axes[0].legend(fontsize=8.5,loc="upper right")
axes[0].text(2,0.5,"两个模\n都还剩 1",ha="center",fontsize=8.5,color="0.3")
axes[1].text(3.6,0.06,"凹槽\n(根)",ha="center",fontsize=8,color="#2980b9")
axes[2].text(3.0,0.5,"两个凹槽正好\n卡在两个 λ 上\n→ 都被滤到 0\n→ 收敛",
             ha="center",fontsize=8.5,color="#c0392b",weight="bold")
fig.savefig("fig_filter_eq.png",dpi=140,bbox_inches="tight")
print("wrote fig_filter_eq.png")
