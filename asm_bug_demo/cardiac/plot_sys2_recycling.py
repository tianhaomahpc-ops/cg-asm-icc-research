#!/usr/bin/env python3
"""plot_sys2_recycling.py -- CONSOLIDATED Sys2 (u_e recovery) recycling result.
Fresh authoritative run: forward_ecg -fischer -T 80 -dt 0.02, np=8 (sys2_recycling_run.txt).

Sys2 = singular pure-Neumann u_e solve on the heart: K_{si+se} u_e = -K_{si} Vm,
ker = span{1}.  Solved EVERY EP step (~10^3-10^4 steps); the operator is fixed, only
the RHS drifts smoothly.  Recycling = use the A-orthonormal history of previous u_e
solves as a Fischer initial guess x0 = sum_i <p_i,b> p_i.

Totals (summed over the 80 sampled steps = every 1 ms; the 4-way guess measurement
runs on sampled steps to avoid 4 extra solves/step; the kept cold solve runs every step):
  cold  (x0=0)            : 7530 CG iters
  warm  (prev-step u_e)   : 7190   (-4%)      <- a single vector barely helps
  physics (x0 = c* Vm)    : 7338   (-2%)      <- cross-system guess, also weak
  Fischer (u_e history)   : 2273   (-69%)     <- the SUBSPACE is the win
  cb-Fischer collectives  : batched 314 vs unbatched-MGS 2128 = 6.8x fewer Allreduce
Per-ms sample: 55 of 80 sampled steps solve in 0 CG iters (history already spans u_e).

Mechanism: warm -4% << Fischer -69% => it is not proximity of one previous vector but
the LOW-DIMENSIONAL SUBSPACE the drifting u_e lives in; projecting b onto the A-ortho
history lands x0 essentially ON the solution during the quasi-static plateaus (0 iters).
Output: fig_sys2_recycling.png
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
from matplotlib.patches import FancyBboxPatch

# --- parse per-ms iterations from the fresh run ---
pat = re.compile(r"t=(\d+)ms.*Sys2\(singular\)=(\d+) \(cold=(\d+) warm=(\d+) phys=(\d+)\)")
t, fis, cold, warm, phys = [], [], [], [], []
for line in open("sys2_recycling_run.txt"):
    m = pat.search(line)
    if m:
        t.append(int(m[1])); fis.append(int(m[2])); cold.append(int(m[3]))
        warm.append(int(m[4])); phys.append(int(m[5]))
t=np.array(t); fis=np.array(fis); cold=np.array(cold); warm=np.array(warm); phys=np.array(phys)
n0 = int((fis==0).sum())
print(f"sampled steps={len(t)}  fischer 0-iter steps={n0}  fischer sampled sum={fis.sum()}")

# full-run totals (from the [FISCHER] summary block)
tot = {"cold":7530, "warm":7190, "physics":7338, "Fischer":2273}

fig = plt.figure(figsize=(16.6, 5.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 0.85, 1.0], wspace=0.30)
fig.suptitle("Sys2(u_e 恢复,奇异)回收整理:子空间(Fischer 历史)是主角 —— 冷 7530 → Fischer 2273(-69%),"
             "而单向量 warm 只 -4%",
             fontsize=11.6, weight="bold", y=1.02)

# (A) per-ms iteration curve: cold flat vs Fischer collapsing to 0
axA = fig.add_subplot(gs[0])
axA.plot(t, cold, "-", color="#7f8c8d", lw=1.6, label="cold (x0=0)")
axA.plot(t, warm, "-", color="#e67e22", lw=1.2, alpha=0.8, label="warm (上一步 u_e)")
axA.plot(t, fis, "o-", color="#27ae60", lw=1.8, ms=3.2, label="Fischer (u_e 历史)")
axA.axhline(0, color="0.6", lw=0.8)
axA.set_xlabel("EP 时间 (ms)"); axA.set_ylabel("Sys2 每步 CG 迭代数")
axA.set_ylim(-4, 100)
axA.set_title(f"① 每步迭代:cold ~90 平;Fischer 塌到 0\n(80 个采样点里 {n0} 步 0 迭代=历史已张成解)",
              fontsize=10.0, weight="bold")
axA.legend(fontsize=8.4, loc="center right"); axA.grid(alpha=0.25)
axA.annotate("平台段:解落在\n低维历史子空间→0 迭代", xy=(50,0), xytext=(30,34),
             fontsize=8.2, color="#1e8449", arrowprops=dict(arrowstyle="->",color="#1e8449"))

# (B) total iterations: warm/physics ~ cold, Fischer crushes it
axB = fig.add_subplot(gs[1])
labs=["cold","warm","physics","Fischer"]; vals=[tot[k] for k in ["cold","warm","physics","Fischer"]]
cols=["#7f8c8d","#e67e22","#c0392b","#27ae60"]
axB.bar(range(4), vals, 0.62, color=cols, edgecolor="0.3")
for i,v in enumerate(vals):
    d=0 if i==0 else -int(100*(vals[0]-v)/vals[0])   # truncation, matches [FISCHER] print
    axB.text(i, v+120, f"{v}"+("" if i==0 else f"\n{d}%"), ha="center", fontsize=8.8, weight="bold")
axB.set_xticks(range(4)); axB.set_xticklabels(labs, fontsize=8.6)
axB.set_ylabel("总 CG 迭代 (80 采样步,每 1ms)"); axB.set_ylim(0, 8700)
axB.set_title("② 单向量(warm-4%/physics-2%)≈冷;\n只有子空间(Fischer)-69%", fontsize=10.0, weight="bold")
axB.grid(axis="y", alpha=0.25)

# (C) cb-Fischer collectives + mechanism (text only, no fragile inset)
axC = fig.add_subplot(gs[2]); axC.axis("off"); axC.set_xlim(0,10); axC.set_ylim(0,10)
axC.text(5,9.5,"③ cb-Fischer 与机理",ha="center",fontsize=10.6,weight="bold")
axC.add_patch(FancyBboxPatch((0.3,6.2),9.4,2.8,boxstyle="round,pad=0.1",fc="#eafaf1",ec="0.4"))
axC.text(5,8.5,"大规模看的是通信(cb-Fischer)",ha="center",fontsize=9.4,weight="bold",color="#1e8449")
axC.text(5,7.55,"维护 Allreduce:2128(未批 MGS)→ 314(批 CGS2)= 6.8× 更少。",ha="center",fontsize=8.4,color="0.15")
axC.text(5,6.75,"投影点积批成 1 次 Allreduce;CGS2 每趟 1 归约(2 趟)。\n"
        "np=8 看不出,P~3000 延迟受限时省 10-20× 归约。",ha="center",fontsize=8.2,color="0.15")
axC.add_patch(FancyBboxPatch((0.3,3.0),9.4,2.7,boxstyle="round,pad=0.1",fc="#eaf2f8",ec="0.4"))
axC.text(5,5.25,"方法要点",ha="center",fontsize=9.4,weight="bold",color="#1f618d")
axC.text(5,4.0,"x0 = sum_i <p_i,b> p_i(A-正交历史,滑窗 16);奇异系统三处去均值;\n"
        "保留的是干净 cold 解(ECG 逐位一致);warm/physics 只做测量。",
        ha="center",fontsize=8.2,color="0.15")
axC.add_patch(FancyBboxPatch((0.3,0.3),9.4,2.4,boxstyle="round,pad=0.1",fc="#f4ecf7",ec="0.5"))
axC.text(5,2.15,"为何 Sys2 能、Sys3 不能",ha="center",fontsize=9.2,weight="bold",color="#6c3483")
axC.text(5,0.95,"Sys2 解落在低维历史子空间 → -69%;\nSys3 残差在多尺度慢模,历史张不住 → 只 -11%。",
        ha="center",fontsize=8.2,color="0.15")

fig.savefig("fig_sys2_recycling.png", dpi=140, bbox_inches="tight")
print("wrote fig_sys2_recycling.png")
