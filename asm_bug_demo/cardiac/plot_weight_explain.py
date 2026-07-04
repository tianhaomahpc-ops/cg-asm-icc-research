#!/usr/bin/env python3
"""plot_weight_explain.py -- why coefficient/diagonal weight == multiplicity
weight for assembled PCASM blocks (so -weightcmp changed 0 iterations).
Output: fig_weight_explain.png
"""
import matplotlib, numpy as np
matplotlib.use("Agg")
from matplotlib import font_manager
font_manager.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
matplotlib.rcParams["font.family"] = "WenQuanYi Zen Hei"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch

fig = plt.figure(figsize=(15, 9.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1.15, 1],
                      hspace=0.22, wspace=0.16)
fig.suptitle("为什么换成\"系数/对角权\",一个迭代都没变", fontsize=18, weight="bold", y=0.985)

# ============================================================ (A) 两种权的定义
axA = fig.add_subplot(gs[0,0]); axA.axis("off"); axA.set_xlim(0,10); axA.set_ylim(0,10)
axA.text(5,9.4,"① 两种单位分解权(都要求 Σ 权 = 1)",ha="center",fontsize=13,weight="bold")
axA.add_patch(FancyBboxPatch((0.3,5.3),9.4,3.4,boxstyle="round,pad=0.1",fc="#eff6ff",ec="#3b82f6",lw=1.5))
axA.text(5,8.15,"重数权(现在用的)",ha="center",fontsize=12,weight="bold",color="#1d4ed8")
axA.text(5,7.35,r"$D_i(j)=\dfrac{1}{m(j)}$,  $m(j)=$ 覆盖 dof $j$ 的子域数",ha="center",fontsize=13)
axA.text(5,6.4,"系数/对角权(想换的)",ha="center",fontsize=12,weight="bold",color="#b45309")
axA.text(5,5.7,r"$D_i(j)=\dfrac{A^{(i)}_{jj}}{\sum_{k}A^{(k)}_{jj}}$,  按各子域在 $j$ 的刚度对角元加权",
         ha="center",fontsize=13)
axA.text(5,4.4,"想法:各向异性 σ 大的方向/子域权重应更大 →",ha="center",fontsize=11,color="0.3")
axA.text(5,3.75,"希望它比\"一律 1/m\"更聪明。",ha="center",fontsize=11,color="0.3")
axA.text(5,2.6,"但关键在于:",ha="center",fontsize=12,weight="bold",color="#b91c1c")
axA.text(5,1.7,r"$A^{(i)}_{jj}$ 到底等不等于 $A^{(k)}_{jj}$?",ha="center",fontsize=13,weight="bold")
axA.text(5,0.8,"(同一个共享点,不同子域看到的对角一样吗?)",ha="center",fontsize=10,color="0.35")

# ============================================================ (B) 主子阵论证
axB = fig.add_subplot(gs[0,1]); axB.axis("off"); axB.set_xlim(0,10); axB.set_ylim(0,10)
axB.text(5,9.4,"② 装配 PCASM:本地块 = 主子阵 → 对角不变",ha="center",fontsize=13,weight="bold")
axB.text(5,8.6,r"$A_i=R_iAR_i^\top$  (只是挑出子域 $i$ 的行列)",ha="center",fontsize=12)
# draw a 5x5 matrix with diagonal, highlight principal submatrix {2,3,4}
d=[3.1,2.0,2.7,2.4,3.3]
x0,y0,cs=2.0,3.0,1.1
for r in range(5):
    for c in range(5):
        val = d[r] if r==c else (0.0)
        inblk = (1<=r<=3 and 1<=c<=3)
        fc = "#fde68a" if inblk else "#f3f4f6"
        if r==c and inblk: fc="#fca5a5"
        axB.add_patch(Rectangle((x0+c*cs, y0+(4-r)*cs), cs,cs, fc=fc, ec="0.6", lw=0.8))
        if r==c: axB.text(x0+c*cs+cs/2, y0+(4-r)*cs+cs/2, f"{d[r]:.1f}", ha="center",va="center",fontsize=10)
        elif abs(r-c)==1: axB.text(x0+c*cs+cs/2, y0+(4-r)*cs+cs/2, "*", ha="center",va="center",fontsize=9,color="0.5")
axB.text(x0-0.35, y0+2.5*cs, "全局\nA", ha="center", va="center", fontsize=11, weight="bold")
axB.text(x0+2*cs, y0-0.45, "红框 = 子域块 A_i 的主子阵", ha="center", fontsize=9.5, color="#b91c1c")
axB.text(5,1.6,r"取主子阵不动对角 → $A^{(i)}_{jj}\equiv A_{jj}$",ha="center",fontsize=12.5,weight="bold",
         bbox=dict(boxstyle="round,pad=0.3", fc="#fca5a5", ec="#b91c1c", lw=1.5))
axB.text(5,0.75,"每个覆盖 j 的子域,看到的对角都是同一个 A_jj。",ha="center",fontsize=10,color="0.3")

# ============================================================ (C) 界面 dof 图解
axC = fig.add_subplot(gs[1,0]); axC.axis("off"); axC.set_xlim(0,10); axC.set_ylim(0,10)
axC.text(5,9.5,"③ 一个共享点 j 上,权到底怎么算",ha="center",fontsize=13,weight="bold")
def scenario(yc, title, v1, v2, concl, ok):
    axC.add_patch(Rectangle((0.6,yc-0.55),3.9,1.1, fc="#dbeafe", ec="#3b82f6", lw=1.4))
    axC.add_patch(Rectangle((5.5,yc-0.55),3.9,1.1, fc="#fee2e2", ec="#dc2626", lw=1.4))
    axC.text(2.55,yc+0.82,"Ω₁ (σ_L)",ha="center",fontsize=9.5,color="#1d4ed8")
    axC.text(7.45,yc+0.82,"Ω₂ (σ_R)",ha="center",fontsize=9.5,color="#b91c1c")
    axC.add_patch(Circle((5.0,yc),0.16, fc="black"))
    axC.text(2.55,yc,v1,ha="center",va="center",fontsize=10,color="#1d4ed8",weight="bold")
    axC.text(7.45,yc,v2,ha="center",va="center",fontsize=10,color="#b91c1c",weight="bold")
    axC.text(0.2,yc+1.15,title,ha="left",fontsize=11,weight="bold",color="#111")
    col = "#15803d" if ok else "#b45309"
    axC.text(5.0,yc-1.05, concl, ha="center", va="center", fontsize=10, color=col, weight="bold")
scenario(7.2, "装配 PCASM(现状):两边看到的对角相同",
         "看到\nA_jj = σ_L+σ_R", "看到\nA_jj = σ_L+σ_R",
         "→ 权 = ½ : ½ = 1/m = 重数权   (σ 信息被抹平)", True)
scenario(2.6, "非装配 Neumann(SORAS):两边看到的对角不同",
         "看到\nσ_L", "看到\nσ_R",
         "→ 权 = σ_L : σ_R ,  σ_L≠σ_R 时 ≠½   (保留各向异性)", False)

# ============================================================ (D) 实测 + 结论
axD = fig.add_subplot(gs[1,1]); axD.axis("off"); axD.set_xlim(0,10); axD.set_ylim(0,10)
axD.text(5,9.5,"④ 实测(-weightcmp, np=8, ICC0)",ha="center",fontsize=13,weight="bold")
rows=[("系统","O","重数","系数"),
      ("Sys1 Vm","0/1/2","13/8/8","13/8/8"),
      ("Sys2 u_e (各向异性)","0/1/2","79/73/71","79/73/71"),
      ("Sys3 φ","0/1/2","66/55/53","66/55/53")]
yt=8.4
for i,(a,b,c,d_) in enumerate(rows):
    y=yt-i*0.82
    wt = "bold" if i==0 else "normal"
    axD.text(0.3,y,a,fontsize=10.5,weight=wt)
    axD.text(4.7,y,b,fontsize=10.5,weight=wt,ha="center")
    axD.text(6.6,y,c,fontsize=10.5,weight=wt,ha="center")
    axD.text(8.6,y,d_,fontsize=10.5,weight=wt,ha="center",
             color=("#15803d" if i>0 else "black"))
    if i>0: axD.text(9.6,y,"同",fontsize=10,color="#15803d",ha="center")
    axD.plot([0.2,9.9],[y-0.34,y-0.34],color="0.9",lw=0.6)
axD.text(5,4.9,"逐格完全相同 —— 一个迭代都没变。",ha="center",fontsize=12,weight="bold",color="#15803d")
axD.add_patch(FancyBboxPatch((0.3,0.5),9.4,3.6,boxstyle="round,pad=0.1",fc="#fef9c3",ec="#ca8a04",lw=1.5))
axD.text(5,3.5,"结论",ha="center",fontsize=12,weight="bold",color="#854d0e")
axD.text(5,2.75,"装配后 σ 已被加进 A_jj、子域间无差别,",ha="center",fontsize=10.5)
axD.text(5,2.15,"系数权自动退化成 1/m → 与重数权恒等。",ha="center",fontsize=10.5)
axD.text(5,1.35,"代数 PCASM 里对角重加权 = 空操作;",ha="center",fontsize=10.5,color="#b91c1c",weight="bold")
axD.text(5,0.8,"各向异性收益要去 SORAS(非装配 Neumann 块)拿。",ha="center",fontsize=10.5,color="#b91c1c",weight="bold")

fig.savefig("fig_weight_explain.png", dpi=140, bbox_inches="tight")
print("wrote fig_weight_explain.png")
