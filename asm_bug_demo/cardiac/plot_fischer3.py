#!/usr/bin/env python3
"""plot_fischer3.py -- Sys3 recycling result vs Sys2, and WHY it differs.
Sys2 (fischer_after_iters.txt): bimodal -- many steps at 0 iters (guess within
tol during the near-static plateau), a few hard steps at ~90. Total -82%.
Sys3 (fischer3_eploop.txt): uniform ~8-iter shave, NEVER 0 -- the torso potential
tracks the drifting interface data every step, so the guess is good-but-not-within
-tol. Total only -11%. The lesson: recycling's big win needs a near-static solution
(0-iter steps); Sys3 lacks that plateau. Output: fig_fischer3.png
"""
import matplotlib, numpy as np, re
matplotlib.use("Agg")
from matplotlib import font_manager
try:
    font_manager.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
    matplotlib.rcParams["font.family"] = "WenQuanYi Zen Hei"
except Exception: pass
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

# Sys2 (cb-Fischer)
p2 = re.compile(r"t=(\d+)ms.*Sys2\(singular\)=(\d+) \(cold=(\d+)")
t2,f2,c2=[],[],[]
for L in open("fischer_after_iters.txt"):
    m=p2.search(L)
    if m: t2.append(int(m[1])); f2.append(int(m[2])); c2.append(int(m[3]))
# Sys3 (fischer3)
p3 = re.compile(r"t=(\d+)ms.*Sys3\(torso\)=(\d+) \(Fischer=(\d+)\)")
t3,c3,f3=[],[],[]
for L in open("fischer3_eploop.txt"):
    m=p3.search(L)
    if m: t3.append(int(m[1])); c3.append(int(m[2])); f3.append(int(m[3]))
t2,f2,c2,t3,c3,f3=map(np.array,(t2,f2,c2,t3,c3,f3))

fig,(a1,a2)=plt.subplots(1,2,figsize=(15.2,5.6),sharey=True)
fig.suptitle("回收:Sys2 vs Sys3 —— 同一套 cb-Fischer,结果差很多(实测 np=8)",fontsize=13,weight="bold",y=1.0)

a1.plot(t2,c2,"-",color="0.6",lw=1.3,label=f"冷启动 (总 {c2.sum()})")
a1.plot(t2,f2,"-o",color="#27ae60",lw=1.8,ms=3.3,label=f"Fischer (总 {f2.sum()}, -{100*(1-f2.sum()/c2.sum()):.0f}%)")
a1.set_title(f"Sys2(u_e,奇异):双峰 —— {int((f2==0).sum())}/80 步 0 迭代\n平台期解近静止→初值落进容差→0 步",fontsize=10.4,weight="bold")
a1.set_xlabel("时间 t (ms)"); a1.set_ylabel("Sys 每步 CG 迭代数")
a1.legend(fontsize=9,loc="center right"); a1.grid(alpha=0.25); a1.set_ylim(-4,105)

a2.plot(t3,c3,"-",color="0.6",lw=1.3,label=f"冷启动 (总 {c3.sum()})")
a2.plot(t3,f3,"-o",color="#e67e22",lw=1.8,ms=3.3,label=f"Fischer (总 {f3.sum()}, -{100*(1-f3.sum()/c3.sum()):.0f}%)")
a2.set_title(f"Sys3(躯干):均匀削 ~8 步,{int((f3==0).sum())}/80 步 0 迭代\n躯干电位随界面数据每步都变→初值好但够不到容差",fontsize=10.4,weight="bold")
a2.set_xlabel("时间 t (ms)")
a2.legend(fontsize=9,loc="center right"); a2.grid(alpha=0.25)
a2.annotate("从不归零\n= 没有静止平台",xy=(40,f3[40] if len(f3)>40 else 60),xytext=(20,25),
            fontsize=9,color="#c0392b",arrowprops=dict(arrowstyle="->",color="#c0392b"))

fig.text(0.5,-0.03,"结论:回收的大收益靠『很多步 0 迭代』,前提是解近静止;Sys2 有平台期(→82%),"
         "Sys3 的解每步都被漂移的界面数据推动、没有平台(→仅 11%)。",
         ha="center",fontsize=10,color="#c0392b",weight="bold")

fig.savefig("fig_fischer3.png",dpi=140,bbox_inches="tight")
print("wrote fig_fischer3.png")
print(f"Sys2 -{100*(1-f2.sum()/c2.sum()):.0f}%  0-iter {int((f2==0).sum())}/80")
print(f"Sys3 -{100*(1-f3.sum()/c3.sum()):.0f}%  0-iter {int((f3==0).sum())}/80")
