#!/usr/bin/env python3
"""plot_capstone.py -- two-panel capstone summary.
(A) Sys2 per-solve iterations under each configuration (measured/modeled): shows
    the remaining room -- 67 baseline -> 33 (overlap+ICC2) -> ~10 (AMG / fixed
    spectral deflation), plus Fischer's bimodal EP-loop behaviour.
(B) strong-scaling extrapolation 384 -> 48000 cores: one-level sASM iters
    (P^0.055, mild) vs the coarse-solve death that forces GenEO-multilevel; the
    count is fine, the coarse SOLVE is what breaks.
All numbers from the verified session ledger. Output: fig_capstone.png
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

fig = plt.figure(figsize=(16.6, 6.8))
gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.22)
fig.suptitle("收官:Sys2 迭代数还能降到哪(左)· 3000→48000 核的可扩展性与必然的多层拐点(右)",
             fontsize=13.5, weight="bold", y=1.0)

# ---------- (A) iteration ladder ----------
axA = fig.add_subplot(gs[0])
rows = [   # (label, iters, color, tag)
 ("ASM O1 (过计数异常)", 173, "#c0392b", "measured np=8→384核"),
 ("sASM O1 ICC0 (基线)", 67, "#e67e22", "measured -decay"),
 ("sASM O2 + ICC2", 33, "#f1c40f", "measured -fair"),
 ("+ 几何粗空间 {1,x,y,z}", 40, "#3498db", "measured -deflate(从76)"),
 ("+ GAMG 作粗层", 9, "#27ae60", "measured GAMG,随P平"),
 ("+ 固定谱deflation(12–40模)", 14, "#16a085", "modeled:一次特征解,恒定算子"),
]
y = np.arange(len(rows))[::-1]
for yi,(lab,it,c,tag) in zip(y,rows):
    axA.barh(yi, it, color=c, edgecolor="0.3", height=0.62)
    axA.text(it+2, yi, f"{it}", va="center", fontsize=10.5, weight="bold")
    axA.text(2, yi+0.34, lab, va="center", fontsize=9.2, weight="bold")
    axA.text(178, yi, tag, va="center", fontsize=7.4, color="0.45", ha="right")
axA.set_yticks([]); axA.set_xlim(0, 185); axA.set_ylim(-0.6, len(rows)-0.1)
axA.set_xlabel("Sys2 每次求解 CG 迭代数(越低越好)")
axA.set_title("① 单次求解:sASM 已把 λmax 治到~1,剩下唯一杠杆=λmin 尾巴\n"
              "几何粗空间到 40 见顶;AMG/谱deflation 才是 40→~10 的钥匙", fontsize=10.4, weight="bold")
axA.axvline(67, color="0.6", ls=":", lw=1)
axA.annotate("Fischer 回收(EP循环):\n双峰——易步→0,难步仍~92\n总量 −63~−82%;\n配粗空间才治难步",
             xy=(67,4.3), xytext=(88,3.4), fontsize=8.2, color="#8e44ad",
             arrowprops=dict(arrowstyle="->", color="#8e44ad"))

# ---------- (B) scaling + coarse crossover ----------
axB = fig.add_subplot(gs[1])
P = np.array([384,768,1536,3072,6000,16000,48000])
one = 107*(P/384)**0.055
axB.plot(P, one, "o-", color="#e67e22", lw=2.4, ms=7, label="一层 sASM 迭代(~P^0.055)")
axB.plot([384,768,1536], [107,112.7,115.5], "s", color="#c0392b", ms=9, label="实测点(107/113/116)")
axB.plot(P, np.full_like(P,45,dtype=float), "--", color="#27ae60", lw=2.2, label="GenEO 多层(~45,随P平)")
axB.axvspan(5000, 15000, color="#fdebd0", alpha=0.6)
axB.axvspan(15000, 60000, color="#fadbd8", alpha=0.6)
axB.text(7500, 175, "粗解死亡带\n冗余稀疏在 m≈15k 崩\n(m=c·P → P≈5k–15k)", fontsize=8.2, color="#a04000", ha="center")
axB.text(30000, 165, "必须 3 层\nGenEO-multilevel", fontsize=8.6, color="#c0392b", ha="center", weight="bold")
axB.axvline(6000, color="#c0392b", ls="-.", lw=1.4)
axB.annotate("拐点 P≈5k–8k:\n迭代数不爆(131),\n但粗解 m=O(P) 崩\n→ 换多层", xy=(6000,120), xytext=(1300,150),
             fontsize=8.3, color="#c0392b", arrowprops=dict(arrowstyle="->", color="#c0392b"))
axB.set_xscale("log"); axB.set_xticks(P); axB.set_xticklabels([str(p) for p in P], fontsize=8, rotation=30)
axB.set_xlabel("MPI ranks(核数)"); axB.set_ylabel("每次求解迭代数")
axB.set_ylim(0, 200)
axB.set_title("② 可扩展性:一层迭代数到 16k 核也才~131(不爆),\n但 m=O(P) 的粗解在 P≈5–8k 崩 → 多层是被迫的,不是选的", fontsize=10.4, weight="bold")
axB.legend(fontsize=8.6, loc="lower right"); axB.grid(alpha=0.25, which="both")
axB.text(400, 12, "16k 核要高效需 ≥1e10 dof;\n否则撞强扩展延迟墙(少用核也是答案)", fontsize=8, color="0.35")

fig.savefig("fig_capstone.png", dpi=135, bbox_inches="tight")
print("wrote fig_capstone.png")
