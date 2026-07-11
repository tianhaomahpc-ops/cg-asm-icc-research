#!/usr/bin/env python3
"""plot_leadvol.py -- HONEST negative result for the full-field lead-field idea.
Measured (-leadvol, np=8, 80 EP steps):
  lvmax=40: basis grew to 40 (all accepted), 50% free steps, full-field rel-L2 err mean 0.16
  lvmax=80: basis grew to 54/80, 32% free, err mean 0.13
=> the torso RHS trajectory is HIGH-RANK: the moving depolarization front sweeps a
genuinely new interface-data direction almost every step, so NO small basis of
precomputed response fields represents the full volume field (13-16% error).
The KEY asymmetry: the ELECTRODE lead-field (a fixed functional g_lead) is EXACT for
any RHS regardless of rank -- but it only gives scalars. The FULL FIELD by
superposition needs the RHS to be low-rank, which it is NOT. So for the full field
you must actually solve; the model-preserving accelerations are one-time Cholesky
and zero-collective Chebyshev, with GAMG as the iteration-cutter.
Output: fig_leadvol.png
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

fig, (a1, a2) = plt.subplots(1, 2, figsize=(14.6, 5.6))
fig.suptitle("全场 lead-field(响应场叠加)对 Sys3 —— 实测负面结果:RHS 高秩,叠加失败",
             fontsize=13, weight="bold", y=1.0)

# (A) basis growth: it never stabilizes
ax=a1
steps=[40,80]; grown=[40,54]; free=[40,26]; err=[16,13]
x=np.arange(2); w=0.34
ax.bar(x-w/2, grown, w, color="#c0392b", label="新方向(要建响应场)")
ax.bar(x+w/2, free, w, color="#27ae60", label="免解(纯叠加)")
for i,(g,f) in enumerate(zip(grown,free)):
    ax.text(i-w/2,g+1,str(g),ha="center",fontsize=10,weight="bold")
    ax.text(i+w/2,f+1,str(f),ha="center",fontsize=10,weight="bold")
ax.set_xticks(x); ax.set_xticklabels(["80 步\n(基上限40)","80 步\n(基上限80)"])
ax.set_ylabel("步数"); ax.set_ylim(0,62)
ax.set_title("① 基一直在涨:80 步里 54 步都是新方向\n→ RHS 秩 ~54,不是我们盼的 ~12", fontsize=10.4, weight="bold")
ax.legend(fontsize=9); ax.grid(axis="y",alpha=0.25)

# (B) the asymmetry: electrode (exact, any rank) vs full-field (needs low rank, fails)
ax=a2; ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,10)
ax.text(5,9.4,"关键不对称:输出是标量还是全场",ha="center",fontsize=11,weight="bold")
from matplotlib.patches import FancyBboxPatch
ax.add_patch(FancyBboxPatch((0.3,4.9),4.4,3.7,boxstyle="round,pad=0.1",fc="#eafaf1",ec="0.4"))
ax.text(2.5,8.25,"电极 lead-field(标量)",ha="center",fontsize=10,weight="bold",color="#1e8449")
ax.text(2.5,6.55,"ECG = g_lead · b(t)\ng_lead 固定向量\n对任意 RHS 精确\n(不需要低秩)\n但只给标量,不给全场",ha="center",fontsize=8.8,color="0.15")
ax.add_patch(FancyBboxPatch((5.3,4.9),4.4,3.7,boxstyle="round,pad=0.1",fc="#fdedec",ec="0.4"))
ax.text(7.5,8.25,"全场叠加",ha="center",fontsize=10,weight="bold",color="#c0392b")
ax.text(7.5,6.55,"phi = sum a_i(t) Psi_i\n需要 RHS 低秩\n实测 RHS 高秩(动前沿)\n→ 13-16% 场误差\n叠加失败",ha="center",fontsize=8.8,color="0.15")
ax.add_patch(FancyBboxPatch((0.3,0.6),9.4,3.8,boxstyle="round,pad=0.12",fc="#f8f9f9",ec="0.5"))
ax.text(5,3.8,"要全场 → 必须真解;可保留模型的加速:",ha="center",fontsize=10.5,weight="bold")
ax.text(5,2.55,"① 一次性 Cholesky(恒定算子分解一次,每步两次三角解,零迭代)\n"
        "② Chebyshev 迭代(一次性估谱界,每迭代零点积/零 Allreduce)\n"
        "③ GAMG(迭代 5.3×,但每迭代 collective 重,大规模需 telescope+pipeline)",
        ha="center",fontsize=9.2,color="0.15")

fig.savefig("fig_leadvol.png",dpi=140,bbox_inches="tight")
print("wrote fig_leadvol.png")
