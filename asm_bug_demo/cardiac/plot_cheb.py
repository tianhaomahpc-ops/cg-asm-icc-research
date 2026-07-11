#!/usr/bin/env python3
"""plot_cheb.py -- Sys3 Chebyshev vs CG: measured np=8 + modeled 3000 cores.
Measured (np=8, 80 steps, persistent solver):
  CG (bjacobi+ICC): 5544 iters (69.3/step), wall 4.1 s -> 0.74 ms/iter; 2 Allreduce/iter
  Chebyshev       : 11108 iters (138.9/step), wall 5.9 s -> 0.53 ms/iter; 0 Allreduce/iter
                    (after a ONE-TIME [emin,emax] estimate; constant operator)
Chebyshev needs ~2x iters but each has NO inner product / NO Allreduce. At np=8 there
is no latency to hide so it loses. At 3000 cores per-iter time is dominated by the
~40-100us Allreduce latency (2/iter for CG, 0 for Chebyshev), so Chebyshev wins.
Model per step: CG = 69*(L + 2*lat); Cheby = 139*L.  L = local work/iter, lat = Allreduce.
Output: fig_cheb.png
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

cg_it, ch_it = 69.3, 138.9          # iters/step
fig,(a1,a2,a3)=plt.subplots(1,3,figsize=(16.4,5.2))
fig.suptitle("Sys3 Chebyshev(零同步迭代)vs CG:小规模慢、大规模靠消 Allreduce 反超(实测+模型)",
             fontsize=12.6,weight="bold",y=1.02)

# (A) iters + collectives per step
ax=a1; x=np.arange(2); w=0.34
ax.bar(x-w/2,[cg_it,ch_it],w,color="#7f8c8d",label="迭代数/步")
ax2=ax.twinx()
ax2.bar(x+w/2,[cg_it*2,0],w,color="#c0392b",label="Allreduce/步")
ax.set_xticks(x); ax.set_xticklabels(["CG","Chebyshev"])
ax.set_ylabel("迭代数/步"); ax2.set_ylabel("全局 Allreduce/步",color="#c0392b")
ax.text(0-w/2,cg_it+3,"69",ha="center",fontsize=10,weight="bold")
ax.text(1-w/2,ch_it+3,"139",ha="center",fontsize=10,weight="bold")
ax2.text(0+w/2,cg_it*2+3,"138",ha="center",fontsize=10,weight="bold",color="#c0392b")
ax2.text(1+w/2,4,"0",ha="center",fontsize=11,weight="bold",color="#c0392b")
ax.set_title("① Chebyshev 迭代 2×(139 vs 69)\n但每迭代零 Allreduce(138 → 0/步)",fontsize=10.2,weight="bold")
ax.set_ylim(0,170); ax2.set_ylim(0,170)
ax.legend(loc="upper left",fontsize=8.5); ax2.legend(loc="upper center",fontsize=8.5)

# (B) measured np=8 wall
ax=a2
ax.bar(["CG","Chebyshev"],[4.1,5.9],color=["#2980b9","#e67e22"],edgecolor="0.3")
for i,v in enumerate([4.1,5.9]): ax.text(i,v+0.1,f"{v}",ha="center",fontsize=11,weight="bold")
ax.set_ylabel("Sys3 总墙钟 (s)"); ax.set_ylim(0,7)
ax.set_title("② np=8 实测:Chebyshev 更慢(5.9>4.1)\n小规模没延迟可藏,2× 迭代直接吃亏",fontsize=10.2,weight="bold")
ax.grid(axis="y",alpha=0.25)

# (C) modeled per-step wall at 3000 cores, sweep local-work/iter L
ax=a3
L=np.linspace(2,40,100)   # us local work per iter
lat=60.0                  # us per Allreduce (mid of 40-100)
cg=cg_it*(L+2*lat)/1000.0     # ms/step
ch=ch_it*L/1000.0             # ms/step
ax.plot(L,cg,color="#2980b9",lw=2.4,label="CG: 69·(L+2·60µs)")
ax.plot(L,ch,color="#e67e22",lw=2.4,label="Chebyshev: 139·L")
ax.fill_between(L,ch,cg,where=(cg>ch),color="#eafaf1",alpha=0.6)
ax.set_xlabel("每迭代本地工作 L (µs)"); ax.set_ylabel("每步墙钟 (ms) @3000核")
ax.set_title("③ 3000 核模型(Allreduce=60µs):\nChebyshev 全 L 区间都更快(消掉 138 次延迟)",fontsize=10.2,weight="bold")
ax.legend(fontsize=9,loc="upper left"); ax.grid(alpha=0.25)
ax.annotate(f"L=5µs: CG {cg_it*(5+120)/1000:.1f}ms vs Cheby {ch_it*5/1000:.1f}ms\n≈ {cg_it*(5+120)/(ch_it*5):.0f}× 更快",
            xy=(5,ch_it*5/1000),xytext=(12,7),fontsize=8.6,color="#c0392b",
            arrowprops=dict(arrowstyle="->",color="#c0392b"))

fig.savefig("fig_cheb.png",dpi=140,bbox_inches="tight")
print("wrote fig_cheb.png")
print(f"3000-core model @L=5us: CG {cg_it*(5+120)/1000:.2f}ms, Cheby {ch_it*5/1000:.2f}ms, {cg_it*(5+120)/(ch_it*5):.1f}x")
print(f"3000-core model @L=20us: CG {cg_it*(20+120)/1000:.2f}ms, Cheby {ch_it*20/1000:.2f}ms, {cg_it*(20+120)/(ch_it*20):.1f}x")
