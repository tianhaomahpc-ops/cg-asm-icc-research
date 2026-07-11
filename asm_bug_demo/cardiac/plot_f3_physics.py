#!/usr/bin/env python3
"""plot_f3_physics.py -- Sys3 initial-guess comparison, incl. the PHYSICS DC guess.
Measured (-fischer3, np=8, 80 EP steps, persistent bjacobi+ICC solver):
  cold (x0=0)             : 5544 iters   (cold-solve wall = 3.2 s)
  warm (prev phi)         : 4993   (-9%)
  physics (x0=iface-mean) : 5550   (-0%)   <-- NO help
  Fischer (u_e-BC history): 4922  (-11%)
Why physics DC guess is useless here: the "physics" guess is a constant = the
interface (Dirichlet) mean, which is the harmonic solution's DC component. But
Sys2 (u_e) is SINGULAR / zero-mean, so the interface data itself is ~zero-mean
=> the DC component is ~0 => there is no offset to capture (5550 ~ cold 5544).
Contrast Sys2, where a physics guess DID exist (u_e ~ -c*Vm from the two-scale
bidomain limit) and was worth exploiting. Sys3's harmonic problem has no such
distinct algebraic shortcut: warm start (-9%) already IS its physics-consistent
guess, and the leftover error lives in the slow interface-variation modes, which
only an OPERATOR-level fix (GAMG / Chebyshev) removes -- not any initial guess.
Output: fig_f3_physics.png
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

cold, warm, phys, fis = 5544, 4993, 5550, 4922
NST = 80
labels = ["cold\n(x0=0)", "warm\n(上一步φ)", "physics\n(x0=交界均值)", "Fischer\n(u_e-BC 历史)"]
vals   = [cold, warm, phys, fis]
cols   = ["#95a5a6", "#e67e22", "#c0392b", "#27ae60"]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(14.2, 5.5))
fig.suptitle("Sys3(躯干)初值对比:物理 DC 猜测无效 —— 因为 u_e 零均值(Sys2 奇异),没有直流偏移可利用",
             fontsize=12.4, weight="bold", y=1.0)

# (A) iterations per step for the four guesses
ax = a1
x = np.arange(4)
ax.bar(x, [v/NST for v in vals], color=cols, edgecolor="0.3")
for i, v in enumerate(vals):
    pct = 0 if i == 0 else -round(100*(cold-v)/cold)
    tag = f"{v/NST:.0f}" + ("" if i == 0 else f"\n({pct:+d}%)")
    ax.text(i, v/NST + 1.2, tag, ha="center", fontsize=10, weight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.2)
ax.set_ylabel("Sys3 每步迭代数"); ax.set_ylim(0, 82)
ax.set_title("① physics(69.4) ≈ cold(69.3):完全没用\n"
             "warm/Fischer 也只 −9/−11%(初值封顶)", fontsize=10.4, weight="bold")
ax.grid(axis="y", alpha=0.25)
ax.annotate("物理 DC 猜测 = 冷启动\n(直流分量 ≈ 0)", xy=(2, phys/NST), xytext=(1.4, 40),
            fontsize=8.6, color="#c0392b",
            arrowprops=dict(arrowstyle="->", color="#c0392b"))

# (B) the reasoning: why Sys3 has no physics shortcut but Sys2 did
ax = a2; ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
from matplotlib.patches import FancyBboxPatch
ax.text(5, 9.5, "为什么 Sys3 没有物理捷径,而 Sys2 有", ha="center", fontsize=11, weight="bold")
ax.add_patch(FancyBboxPatch((0.3, 5.2), 9.4, 3.5, boxstyle="round,pad=0.12", fc="#eafaf1", ec="0.4"))
ax.text(5, 8.25, "Sys2(u_e 恢复):有物理捷径", ha="center", fontsize=10, weight="bold", color="#1e8449")
ax.text(5, 6.5, "双域两尺度极限下 u_e ≈ −c·Vm,\n"
        "而 Vm(t) 由 Sys1 刚算出 → 直接给出 u_e 的主形状。\n"
        "但真正的赢家其实是「解落在历史张成的低维子空间」→ Fischer −82%。",
        ha="center", fontsize=8.8, color="0.15")
ax.add_patch(FancyBboxPatch((0.3, 1.2), 9.4, 3.5, boxstyle="round,pad=0.12", fc="#fdedec", ec="0.4"))
ax.text(5, 4.25, "Sys3(躯干 Laplace):没有额外物理捷径", ha="center", fontsize=10, weight="bold", color="#c0392b")
ax.text(5, 2.5, "唯一的解析结构是「常数是调和的」→ 直流分量。\n"
        "但交界数据来自零均值的 u_e → 直流 ≈ 0 → 猜了等于没猜。\n"
        "warm(−9%)已经是它的物理一致初值;剩余误差在慢模,\n"
        "只有算子级(GAMG/Chebyshev)能治,初值治不了。",
        ha="center", fontsize=8.8, color="0.15")

fig.savefig("fig_f3_physics.png", dpi=140, bbox_inches="tight")
print("wrote fig_f3_physics.png")
print(f"cold {cold} ({cold/NST:.1f}/step); warm {warm} (-{round(100*(cold-warm)/cold)}%); "
      f"physics {phys} (-{round(100*(cold-phys)/cold)}%); Fischer {fis} (-{round(100*(cold-fis)/cold)}%)")
