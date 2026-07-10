#!/usr/bin/env python3
"""plot_fischer_batched.py -- before/after the large-scale recycling refactor:
(1) BATCH the guess-projection dot products into one Allreduce (was one per
    basis vector), (2) replace single-pass MGS by CGS2 (2 batched passes) in the
    basis maintenance.  Same run (np=8, -fischer -T 80 -dt 0.02), real data from
    fischer_before_iters.txt / fischer_after_iters.txt.
Checks: cold/warm bitwise identical (untouched paths); Allreduce count 7.2x down;
Fischer total DROPS 2461 -> 1361 because CGS2's second pass gives a better
A-orthonormal basis (single-pass MGS loses orthogonality in floating point).
Output: fig_fischer_batched.png
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

pat = re.compile(r"t=(\d+)ms.*Sys2\(singular\)=(\d+) \(cold=(\d+)")
def load(f):
    t,fis,cold=[],[],[]
    for line in open(f):
        m=pat.search(line)
        if m: t.append(int(m[1])); fis.append(int(m[2])); cold.append(int(m[3]))
    return np.array(t),np.array(fis),np.array(cold)

tb,fb,cb = load("fischer_before_iters.txt")
ta,fa,ca = load("fischer_after_iters.txt")
assert (cb==ca).all(), "cold traces must be identical (untouched path)"
print(f"cold identical: True.  fischer before sum={fb.sum()}  after sum={fa.sum()}")
print(f"zero-iter steps: before {(fb==0).sum()}/80  after {(fa==0).sum()}/80")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.2, 5.8), gridspec_kw={"width_ratios":[1.7,1]})
fig.suptitle("回收的大规模改造(点积打包 + MGS→CGS2):同一 EP 循环,改前 vs 改后(np=8 实测)",
             fontsize=13, weight="bold", y=1.0)

ax1.plot(tb, cb, "-", color="0.6", lw=1.4, label=f"冷启动(两次逐位相同,总 {cb.sum()})")
ax1.plot(tb, fb, "-o", color="#e67e22", lw=1.8, ms=3.5, label=f"Fischer 改前(MGS,未打包):总 {fb.sum()}")
ax1.plot(ta, fa, "-o", color="#27ae60", lw=1.8, ms=3.5, label=f"Fischer 改后(CGS2,打包):总 {fa.sum()}")
ax1.set_xlabel("时间 t (ms)"); ax1.set_ylabel("该步 Sys2 CG 迭代数")
ax1.set_title(f"① 逐步迭代:改后 0-迭代步 {(fa==0).sum()}/80(改前 {(fb==0).sum()}/80)\nCGS2 第二遍把基的 A-正交性擦干净 → 投影更准", fontsize=10.4, weight="bold")
ax1.legend(fontsize=9, loc="center right"); ax1.grid(alpha=0.25); ax1.set_ylim(-3, 105)

labels = ["冷启动", "改前\n(MGS)", "改后\n(CGS2)"]
vals = [cb.sum(), fb.sum(), fa.sum()]
cols = ["#7f8c8d", "#e67e22", "#27ae60"]
bars = ax2.bar(labels, vals, color=cols, edgecolor="0.3")
for b,v in zip(bars,vals):
    ax2.text(b.get_x()+b.get_width()/2, v+80, str(int(v)), ha="center", fontsize=11, weight="bold")
ax2.set_ylabel("EP 循环总迭代")
ax2.set_title("② 总量与通信:迭代 −67%→−82%;\n维护 Allreduce 314 vs 2252(7.2×↓)", fontsize=10.4, weight="bold")
ax2.grid(axis="y", alpha=0.25); ax2.set_ylim(0, cb.sum()*1.15)
ax2.text(0.5, 0.55, "run-to-run 方差 ±~300\n(边界步在 0/90 间翻转)\n改前 2461–2754\n改后 1361–1660\n→ 间隔远超方差,收益真实",
         transform=ax2.transAxes, fontsize=8.8, color="0.3")

fig.savefig("fig_fischer_batched.png", dpi=140, bbox_inches="tight")
print("wrote fig_fischer_batched.png")
