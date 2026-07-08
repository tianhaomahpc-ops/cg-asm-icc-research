#!/usr/bin/env python3
"""plot_deflate_reshist.py -- REAL relative-residual histories on Sys2 (Kie),
dumped by forward_ecg -deflate (KSPGetResidualHistory, unpreconditioned).
Shows what deflation/recycling does to the residual CURVE, not just the count.
Files: deflate_hist_{fine,geometric,recycled,hybrid}.txt  (k  r_k/r_0)
Output: fig_deflate_reshist.png
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
import matplotlib.ticker as mt

def load(f):
    d = np.loadtxt(f); return d[:,0], d[:,1]

series = [
    ("deflate_hist_fine.txt",      "#c0392b", "无 deflation (fine=bjacobi+ICC)"),
    ("deflate_hist_recycled.txt",  "#e67e22", "回收(合成快照,弱)"),
    ("deflate_hist_geometric.txt", "#27ae60", "几何粗空间 {1,x,y,z} = 把慢尾铲掉"),
    ("deflate_hist_hybrid.txt",    "#2980b9", "混合(几何+回收)"),
]

fig, ax = plt.subplots(figsize=(10.6, 6.8))
fig.suptitle("真实 Sys2(Kie)上的相对残差:做 deflation/回收前后对比 (np=8, 实测)",
             fontsize=13, weight="bold")
for fn, c, lab in series:
    k, r = load(fn)
    ax.semilogy(k, r, "-", color=c, lw=2.3, label=f"{lab}: {int(k[-1])} 步")
ax.axhline(1e-8, color="0.6", ls=":", lw=1); ax.text(2, 1.4e-8, "收敛判据 1e-8", fontsize=9, color="0.4")
ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v,_: f"1e{int(round(np.log10(v)))}" if v>0 else ""))
ax.set_xlabel("CG 迭代步 k"); ax.set_ylabel("相对残差 r_k / r_0 (log)")
ax.set_xlim(0, 80); ax.set_ylim(1e-9, 2)
ax.grid(alpha=0.25); ax.legend(fontsize=10, loc="upper right")

ax.annotate("红线的长慢尾 = 那 12 个慢模\n(cond 全由它们造成)", xy=(50, 3e-4), xytext=(30, 3e-2),
            fontsize=9.5, color="#c0392b", arrowprops=dict(arrowstyle="->", color="#c0392b"))
ax.annotate("deflation 把慢尾铲掉后\n残差直落,76→40 步", xy=(35, 1e-6), xytext=(42, 6e-4),
            fontsize=9.5, color="#27ae60", weight="bold", arrowprops=dict(arrowstyle="->", color="#27ae60"))
ax.text(2, 5e-9,
        "注:此处『回收』用合成随机快照(会塌到最小模,故仅弱效 −6%);\n"
        "真正的强回收用 EP 时间演化快照(Fischer,整段循环 −77%)——见文字说明。",
        fontsize=8.2, color="0.35")

fig.savefig("fig_deflate_reshist.png", dpi=140, bbox_inches="tight")
print("wrote fig_deflate_reshist.png")
for fn,_,lab in series:
    k,r=load(fn); print(f"{lab:36s}: {int(k[-1])} steps, final r/r0={r[-1]:.2e}")
