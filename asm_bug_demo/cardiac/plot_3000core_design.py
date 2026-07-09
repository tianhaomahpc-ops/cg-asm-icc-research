#!/usr/bin/env python3
"""plot_3000core_design.py -- the 3000-core production design, one picture.
Left : the three-layer Sys2 solver stack and where each communication lives.
Right: division of labour -- which component kills which part of the spectrum /
       cost, with the measured or modeled numbers.
Output: fig_3000core_design.png
"""
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
try:
    font_manager.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
    matplotlib.rcParams["font.family"] = "WenQuanYi Zen Hei"
except Exception: pass
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig = plt.figure(figsize=(16.8, 8.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.16)
fig.suptitle("3000 核生产方案:三层分工的 Sys2 求解器(Sys1 只要最便宜档;Sys3 同 Sys2 缩小版)",
             fontsize=14.5, weight="bold", y=0.99)

# ---------------- left: the stack ----------------
ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,10)
def box(x,y,w,h,fc,title,body,tfs=10.5,bfs=8.6,tc="k"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.12",fc=fc,ec="0.35",lw=1.2))
    ax.text(x+w/2, y+h-0.34, title, ha="center", fontsize=tfs, weight="bold", color=tc)
    ax.text(x+w/2, y+h/2-0.28, body, ha="center", va="center", fontsize=bfs, color="0.15")

box(0.3, 8.15, 9.4, 1.5, "#fdf2e9", "外层 Krylov:pipelined CG(-sys2_ksp_type pipecg,本机已验证可一行切换)",
    "把每迭代的 ~3 次全局 collective(2 点积 + 1 粗解 gather)与 SpMV/预条件 apply 重叠\n"
    "非预条件范数判据 + 挂常数核(奇异 Sys2);回收投影点积打包成 1 次 Allreduce", tfs=10.5)
box(0.3, 5.9, 4.5, 1.9, "#eafaf1", "细层(每迭代)\nsASM + overlap(O1) + ICC(1)",
    "每子域 ~700–7000 dof,本地三角解\n通信:仅近邻 halo(不随 P 恶化)\n治:N̂ 过计数 + ω 本地精度(快头)")
box(5.2, 5.9, 4.5, 1.9, "#eaf2f8", "粗层(每迭代)\nGenEO 粗空间(每子域自适应挑模)",
    "维数 m ≈ (1~5)×P = 3k–15k,列局部支撑\n粗算子 E 稀疏(子域图),setup 一次\n治:C0 = O(P) 慢模 + 薄壁/疤痕硬模")
box(0.3, 3.6, 4.5, 1.9, "#f4ecf7", "回收层(每次求解)\nGCRO-DR / Fischer,窗口 16–32",
    "跨时间步学漂移模,维数 O(1) 不随 P 长\n投影点积打包成 1 次 Allreduce\n治:随 Vm(t) 漂移的余量(时间摊销)")
box(5.2, 3.6, 4.5, 1.9, "#fdedec", "粗问题的解法(关键瓶颈,按 m 分档)",
    "冗余稠密在 m≈300–900 就死(m=3k 超 40 倍)\nm=3k(Nicolaides): 冗余稀疏Chol,~40µs ✓\nm=15k(GenEO): q=32–64 子通信器(TELESCOPE\n/REDUNDANT+MUMPS)~100–230µs;更大→真第三层")
box(0.3, 1.2, 9.4, 1.9, "#f8f9f9", "一次性 setup(被上万次求解摊销 —— 我们的结构性优势)",
    "GenEO 每子域小特征问题 + E 组装/分解 + ICC 分解:全部只做一次;\n"
    "EP 循环解同一算子 O(10³–10⁴) 次 → setup 占比 → 0;逐次求解的边际成本才是账本", tfs=10.5)
for (x0,y0,x1,y1) in [(2.55,8.15,2.55,7.8),(7.45,8.15,7.45,7.8),
                      (2.55,5.9,2.55,5.5),(7.45,5.9,7.45,5.5),(5.0,4.55,5.2,4.55)]:
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle="-|>",color="0.4",lw=1.4,mutation_scale=14))

# ---------------- right: division of labour ----------------
ax2 = fig.add_subplot(gs[1]); ax2.axis("off"); ax2.set_xlim(0,10); ax2.set_ylim(0,10)
rows = [
 ("组件","治什么","代价随 P","依据",True),
 ("sASM+O1+ICC1","快头(主体 λ≈1)","近邻,平","-fair: 76→33",False),
 ("GenEO 粗空间","O(P) 慢模+硬模\n(薄壁/疤痕)","m~P,稀疏解\n子通信器","slowdim: 慢模≈1.5/子域;\n细颈+6、高对比+3",False),
 ("回收 16–32","时间漂移余量","O(1),1 次\nAllreduce/解","EP 实测 −63%\n(fig_fischer_eploop)",False),
 ("pipelined CG","Allreduce 延迟","隐藏 log P","文献/PETSc 标准",False),
 ("Sys1 特例","无尾(cond 1.4)","bjacobi+ICC0\n即可,勿加层","-decay: 0 慢模",False),
]
y=9.6
for r in rows:
    fc = "#d5dbdb" if r[4] else ("#ffffff" if rows.index(r)%2 else "#f7f9f9")
    ax2.add_patch(FancyBboxPatch((0.1,y-1.28),9.7,1.34,boxstyle="round,pad=0.03",fc=fc,ec="0.6",lw=0.6))
    for x,t,fs in [(1.2,r[0],9.2),(3.9,r[1],8.4),(6.6,r[2],8.4),(8.9,r[3],7.6)]:
        ax2.text(x,y-0.62,t,ha="center",va="center",fontsize=fs,
                 weight="bold" if r[4] else "normal")
    y-=1.38
ax2.text(5.0,0.78,"强扩展两大杀手,各有其药:迭代随 P 涨 → GenEO 粗空间钉住;\n每迭代随 P 变贵 → pipelined CG + 稀疏粗解 + 打包点积",
         ha="center", fontsize=10.5, color="#c0392b", weight="bold")
ax2.text(5.0,0.12,"验证阶梯:np=8→64→512→3000,每级测 迭代数(应平)/每步耗时分解/粗解占比(<20%)",
         ha="center", fontsize=9, color="0.35")

fig.savefig("fig_3000core_design.png", dpi=135, bbox_inches="tight")
print("wrote fig_3000core_design.png")
