#!/usr/bin/env python3
"""plot_pipeline.py -- one complete figure of the forward-ECG solve pipeline:
which method attacks which problem, the iteration reduction, and the wall-clock
effect.  All numbers are in-container measurements (heart.msh 10085 dof,
forward_ecg, dt=0.1); iters are per-solve, wall-clock is the x-factor vs baseline.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
font_manager.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
matplotlib.rcParams["font.family"] = "WenQuanYi Zen Hei"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(3, 1, height_ratios=[1.05, 1.5, 1.15], hspace=0.34)

# ============================================================= TOP: the problem
axP = fig.add_subplot(gs[0]); axP.axis("off"); axP.set_xlim(0,10); axP.set_ylim(0,3)
axP.text(5,2.86,"问题:三系统耦合前向 ECG(共形四面体 P1-FEM,heart.msh 10085 dof;每个时间步)",
         ha="center",fontsize=13,weight="bold")
def box(ax,x,y,w,h,fc,ec,title,sub,tc="black"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.03,rounding_size=0.08",
                                fc=fc,ec=ec,lw=1.8))
    ax.text(x+w/2,y+h*0.66,title,ha="center",va="center",fontsize=11,weight="bold",color=tc)
    ax.text(x+w/2,y+h*0.26,sub,ha="center",va="center",fontsize=8.5,color=tc)
box(axP,0.2,0.6,2.05,1.6,"#dbeafe","#3b82f6","Sys1  Vm","单域/抛物/局部\nCG+ICC ~13 迭代")
box(axP,2.75,0.6,2.35,1.6,"#fecaca","#dc2626","Sys2  u_e  ★难点","纯Neumann奇异/椭圆/全局\n~84–96 迭代  <- 加速目标")
box(axP,5.6,0.6,2.05,1.6,"#dcfce7","#16a34a","Sys3  φ_torso","躯干Laplace/Dirichlet\n~66 迭代")
box(axP,8.15,0.6,1.65,1.6,"#fef9c3","#ca8a04","体表 ECG","φ_L - φ_R\nQRS + T")
for x0,x1 in [(2.25,2.75),(5.10,5.60),(7.65,8.15)]:
    axP.add_patch(FancyArrowPatch((x0,1.4),(x1,1.4),arrowstyle="-|>",mutation_scale=18,lw=2,color="0.3"))
axP.text(2.5,1.62,"Vm(t)",ha="center",fontsize=8); axP.text(5.35,1.62,"u_e",ha="center",fontsize=8)
axP.text(7.9,1.62,"Dirichlet",ha="center",fontsize=7.5)

# ============================================================ MIDDLE: method map
axM = fig.add_subplot(gs[1]); axM.axis("off"); axM.set_xlim(0,10); axM.set_ylim(0,10)
axM.text(5,9.6,"方法:每个方法解决 Sys2 的哪个子问题(初值 vs 预条件;fine level vs coarse level)",
         ha="center",fontsize=12.5,weight="bold")
# two axes
axM.add_patch(FancyBboxPatch((0.2,5.0),4.6,3.9,boxstyle="round,pad=0.05",fc="#eff6ff",ec="#3b82f6",lw=1.5))
axM.text(2.5,8.55,"轴①  跨时间(修 初值)",ha="center",fontsize=11,weight="bold",color="#1d4ed8")
axM.text(2.5,8.0,"用 Sys2 自己的历史解作初值",ha="center",fontsize=9)
axM.text(0.45,7.35,"· warm-start:上一步 u_e",fontsize=9)
axM.text(0.45,6.75,"· Fischer:历史 A-正交投影(滑窗16)",fontsize=9)
axM.text(0.45,6.05,"机理:u_e(t) 随波光滑漂移",fontsize=8.5,style="italic",color="0.35")
axM.text(0.45,5.55,"-> 平台段解近乎不变、历史是好预测",fontsize=8.5,style="italic",color="0.35")

axM.add_patch(FancyBboxPatch((5.2,5.0),4.6,3.9,boxstyle="round,pad=0.05",fc="#f0fdf4",ec="#16a34a",lw=1.5))
axM.text(7.5,8.55,"轴②  跨系统(修 预条件)",ha="center",fontsize=11,weight="bold",color="#15803d")
axM.text(7.5,8.0,"用 Sys1/Sys2 共享网格的结构",ha="center",fontsize=9)
axM.text(5.45,7.35,"· coarse level:Nicolaides 粗空间",fontsize=9)
axM.text(5.6,6.9,"每子域一列 -> 张成常数=奇异零空间=慢全局模",fontsize=7.8,color="0.3")
axM.text(5.45,6.3,"· fine level:SORAS(重叠+Robin 传输 αM_Γ)",fontsize=9)
axM.text(5.6,5.85,"本地 Neumann 块;局部近似解",fontsize=7.8,color="0.3")
axM.text(5.45,5.35,"fine=绝对迭代数,coarse=可扩展性",fontsize=8.5,style="italic",color="0.35")

# common constraint bar
axM.add_patch(FancyBboxPatch((0.2,3.7),9.6,0.95,boxstyle="round,pad=0.04",fc="#fafafa",ec="0.6",lw=1.2))
axM.text(5,4.35,"公共:测量模式——落盘场永远是干净 baseline 解 => ECG 逐位不变(已核对);",
         ha="center",fontsize=9)
axM.text(5,3.95,"外层 PETSc CG(稳健处理奇异)+ ‖b‖-相对停机 atol=1e-8‖b‖",ha="center",fontsize=9)

# three engineering fixes
axM.text(5,3.15,"三个真实工程坑(修好才量得到收益):",ha="center",fontsize=9.5,weight="bold",color="#b91c1c")
axM.text(5,2.55,"① iterative_mode 不传播到 KSP -> 每步显式 KSPSetInitialGuessNonzero(否则 warm≡cold;微测:精确初值 88->0 迭代)",
         ha="center",fontsize=8.4)
axM.text(5,2.05,"② 默认 atol 在预条件后残差、掩盖初值质量 -> 改 UNPRECONDITIONED",ha="center",fontsize=8.4)
axM.text(5,1.55,"③ Fischer 历史必须滑窗淘汰最旧(append-only 会冻结在早期 QRS 模态、平台段失效)",ha="center",fontsize=8.4)

# ============================================================ BOTTOM: results
axR = fig.add_subplot(gs[2]); axR.axis("off"); axR.set_xlim(0,10); axR.set_ylim(0,10)
axR.text(5,9.5,"效果(实测,np=4,T=40;iters=每解 CG 迭代,wall-clock=对 baseline 的倍数)",
         ha="center",fontsize=12.5,weight="bold")

# data: name, targets, iters/solve, iters_x, time_x(faster>1), verdict color
rows = [
 ("baseline  bjacobi+ICC0", "参照", 87, 1.00, 1.00, "0.85"),
 ("Fischer  (跨时间)",      "初值", 80, 1.12, 1.10, "#3b82f6"),   # time ~= iters (proj overhead <1 SpMV)
 ("Nicolaides 粗空间 (coarse)","全局模/可扩展", 78, 1.11, 1.21, "#16a34a"),
 ("SORAS  local=Cheby4",    "局部+传输", 50, 1.74, 0.43, "#f97316"),
 ("SORAS  local=near-exact","局部+传输", 38, 2.31, 0.05, "#dc2626"),
 ("SORAS+coarse near-exact", "两者", 37, 2.33, 0.05, "#dc2626"),
]
yb = 8.4; dy = 1.30
# column headers
axR.text(0.15,yb+0.42,"方法",fontsize=9.5,weight="bold")
axR.text(3.05,yb+0.42,"解决",fontsize=9.5,weight="bold")
axR.text(4.35,yb+0.42,"迭代↓(×)",fontsize=9.5,weight="bold")
axR.text(7.35,yb+0.42,"墙钟(×)",fontsize=9.5,weight="bold")
axR.text(8.75,yb+0.42,"结论",fontsize=9.5,weight="bold")
imax=2.4; BARW=1.15
for i,(nm,tg,it,ix,tx,c) in enumerate(rows):
    y = yb - i*dy
    axR.text(0.15,y,nm,fontsize=9,weight=("bold" if "baseline" not in nm else "normal"),color=c if c!="0.85" else "0.2")
    axR.text(3.05,y,tg,fontsize=8.3,color="0.3")
    # iteration bar (fewer = longer green-ish), scaled by ix
    axR.add_patch(plt.Rectangle((4.35,y-0.16),BARW*(ix/imax),0.32,fc=c,ec="none",alpha=.85))
    axR.text(4.35+BARW*(ix/imax)+0.08,y,f"{ix:.2f}× ({it})",va="center",fontsize=8.3)
    # wall-clock: green if faster(>1), red if slower(<1)
    fast = tx>=1.0
    axR.text(7.35,y,f"{tx:.2f}×",va="center",fontsize=9,weight="bold",
             color=("#15803d" if fast else "#b91c1c"))
    axR.text(7.9,y,("更快" if fast else f"更慢({1/tx:.0f}×)"),va="center",fontsize=8,
             color=("#15803d" if fast else "#b91c1c"))
    verdict = {"参照":"—",
               "初值":"小赢(次力)",
               "全局模/可扩展":"[OK] 唯一墙钟净赢",
               "局部+传输":("迭代↓但墙钟↑" ),
               "两者":"同 SORAS(粗空间冗余)"}[tg]
    axR.text(8.75,y,verdict,va="center",fontsize=7.8,
             color=("#15803d" if "净赢" in verdict else ("#b91c1c" if "↑" in verdict else "0.25")))
    axR.plot([0.1,9.9],[y-0.55,y-0.55],color="0.9",lw=0.6)

axR.text(5,0.35,"关键:迭代 ≠ 墙钟。SORAS 是最大的迭代杠杆(2.3×)但本地精解太贵(5.5 vs 0.11 ms/迭代)"
         "=> 此规模下墙钟反而慢 20×;Nicolaides 粗空间是唯一墙钟净赢(便宜的 np×np 粗解)。",
         ha="center",fontsize=9,color="#7c2d12",weight="bold")

fig.suptitle("心脏前向 ECG · Sys2 求解加速全景:问题 -> 方法 -> 效果(迭代 & 墙钟,全部真机实测)",
             fontsize=14.5, weight="bold", y=0.995)
fig.savefig("fig_pipeline.png", dpi=125, bbox_inches="tight"); print("wrote fig_pipeline.png")
