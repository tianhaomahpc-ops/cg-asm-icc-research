#!/usr/bin/env python3
"""plot_cluster_scaling.py -- the USER'S REAL cluster data (384/768/1536 ranks,
recover-ue = Sys2, torso = Sys3; sASM vs ASM + CG) and what it implies.
Three panels: iterations vs ranks (mild power law, extrapolated), total time vs
ranks (near/super-ideal), per-average-iteration time (superlinear = cache =
still compute-bound, latency wall not yet reached).
Output: fig_cluster_scaling.png
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

P   = np.array([384,768,1536])
its = np.array([107,112.7,115.5]); ita = np.array([173.8,177.8,193.8])
ts  = np.array([194.2,97.6,41.8]); ta  = np.array([313.6,159.7,72.2])
t3s = np.array([30.4,17.1,8.2]);   t3a = np.array([50.0,27.8,13.6])

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
fig.suptitle("实测集群数据(384/768/1536 核):一层 sASM 的迭代增长温和(~P^0.055),时间仍近理想扩展",
             fontsize=13, weight="bold", y=1.02)

# (1) iterations
ax=axes[0]
a_s=np.polyfit(np.log(P),np.log(its),1); a_a=np.polyfit(np.log(P),np.log(ita),1)
Px=np.array([384,768,1536,3072,12288])
ax.plot(P, its, "o-", color="#27ae60", lw=2.2, ms=8, label="sASM 实测")
ax.plot(P, ita, "s-", color="#c0392b", lw=2.2, ms=8, label="ASM 实测")
ax.plot(Px, np.exp(np.polyval(a_s,np.log(Px))), "--", color="#27ae60", lw=1.2)
ax.plot(Px, np.exp(np.polyval(a_a,np.log(Px))), "--", color="#c0392b", lw=1.2)
ax.annotate(f"外推 3072: ~120", xy=(3072, 120), xytext=(1400, 140), fontsize=9.5,
            color="#27ae60", weight="bold", arrowprops=dict(arrowstyle="->", color="#27ae60"))
ax.annotate("12288: ~130", xy=(12288, 130), xytext=(4800, 105), fontsize=9,
            color="#27ae60", arrowprops=dict(arrowstyle="->", color="#27ae60"))
ax.set_xscale("log", base=2); ax.set_xticks(Px); ax.set_xticklabels([str(p) for p in Px])
ax.set_xlabel("ranks"); ax.set_ylabel("recover-ue 平均迭代数")
ax.set_title("① 迭代:sASM ~P^0.055(4x 核 +8%)\nASM ~P^0.079;3000 核不会爆炸", fontsize=10.4, weight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.25); ax.set_ylim(80, 230)

# (2) total time
ax=axes[1]
ax.loglog(P, ts, "o-", color="#27ae60", lw=2.2, ms=8, label="recover-ue sASM")
ax.loglog(P, ta, "s-", color="#c0392b", lw=1.8, ms=7, label="recover-ue ASM")
ax.loglog(P, t3s, "o--", color="#2980b9", lw=1.8, ms=7, label="torso sASM")
ax.loglog(P, ts[0]*P[0]/P, ":", color="0.5", lw=1.5, label="理想 1/P")
ax.set_xticks(P); ax.set_xticklabels([str(p) for p in P])
ax.set_xlabel("ranks"); ax.set_ylabel("总求解时间 (s)")
ax.set_title("② 时间:384→1536 加速 4.65x(理想 4x)\n超线性 = cache 效应 = 仍算力主导", fontsize=10.4, weight="bold")
ax.legend(fontsize=8.5); ax.grid(alpha=0.25, which="both")

# (3) per-average-iteration time
ax=axes[2]
pit_s=ts/its; pit_a=ta/ita
ax.loglog(P, pit_s, "o-", color="#27ae60", lw=2.2, ms=8, label="sASM 每(平均)迭代耗时")
ax.loglog(P, pit_a, "s--", color="#c0392b", lw=1.5, ms=6, label="ASM(重合:加权免费)")
ax.loglog(P, pit_s[0]*P[0]/P, ":", color="0.5", lw=1.5, label="理想 1/P")
ax.set_xticks(P); ax.set_xticklabels([str(p) for p in P])
ax.set_xlabel("ranks"); ax.set_ylabel("时间/平均迭代 (s)")
ax.set_title("③ 每迭代耗时:2.10x, 2.39x 每翻倍(超线性)\n→ Allreduce 延迟墙还远;墙到时此线变平", fontsize=10.4, weight="bold")
ax.legend(fontsize=8.5); ax.grid(alpha=0.25, which="both")
ax.annotate("当这条线不再下降\n= 延迟地板到了\n(dof/核 < ~5-10k)", xy=(1536, pit_s[-1]),
            xytext=(500, 0.5), fontsize=9, color="#8e44ad",
            arrowprops=dict(arrowstyle="->", color="#8e44ad"))

fig.savefig("fig_cluster_scaling.png", dpi=135, bbox_inches="tight")
print("wrote fig_cluster_scaling.png")
