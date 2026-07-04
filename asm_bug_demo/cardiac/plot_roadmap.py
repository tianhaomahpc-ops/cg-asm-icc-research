#!/usr/bin/env python3
"""plot_roadmap.py -- the ASM-anomaly-driven logic route of the project as a
flowchart: anomaly -> diagnosis -> sASM fix -> exposed bottlenecks -> downstream
work.  Pairs with PROMPT_project_zh.md.  Output: fig_roadmap.png
"""
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
font_manager.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
matplotlib.rcParams["font.family"] = "WenQuanYi Zen Hei"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(14.5, 10.2)); ax.axis("off")
ax.set_xlim(0, 14.5); ax.set_ylim(0, 10.2)

def box(x, y, w, h, fc, ec, title, body="", tsz=12, bsz=9, tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.10",
                                fc=fc, ec=ec, lw=2.0))
    if body:
        ax.text(x+w/2, y+h*0.70, title, ha="center", va="center", fontsize=tsz, weight="bold", color=tc)
        ax.text(x+w/2, y+h*0.30, body, ha="center", va="center", fontsize=bsz, color=tc)
    else:
        ax.text(x+w/2, y+h/2, title, ha="center", va="center", fontsize=tsz, weight="bold", color=tc)

def arrow(x0, y0, x1, y1, label="", lx=0.25, color="0.25", lw=2.4):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=22, lw=lw, color=color,
                                 connectionstyle="arc3,rad=0"))
    if label:
        ax.text((x0+x1)/2+lx, (y0+y1)/2, label, ha="left", va="center",
                fontsize=10, style="italic", color="#b45309", weight="bold")

ax.text(7.25, 9.95, "项目逻辑路线:从 ASM 的一个反常出发",
        ha="center", fontsize=16, weight="bold")
ax.text(7.25, 9.58, "anomaly  →  diagnosis  →  sASM fix  →  exposed bottlenecks  →  downstream",
        ha="center", fontsize=10.5, color="0.4", style="italic")

# --- spine ---------------------------------------------------------------
box(4.1, 8.35, 6.3, 0.95, "#fee2e2", "#dc2626",
    "① 引子:ASM 加重叠,迭代反而上升(反常)",
    "Sys3 np=8, O:0->1->2  迭代 66 -> 93 -> 109 (+65%),墙钟更慢", 13, 9.5)
arrow(7.25, 8.35, 7.25, 7.75, "为什么?")

def frame(x, y, w, h, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.10",
                                fc=fc, ec=ec, lw=2.0))

frame(2.4, 6.08, 9.7, 1.57, "#fef3c7", "#d97706")
ax.text(7.25, 7.42, "② 诊断:过计数(over-count) × 不精确(inexact ICC(0))",
        ha="center", va="center", fontsize=13, weight="bold")
ax.text(7.25, 7.06, "重叠 dof 按重数各加一次 => |M^-1|、λmax 放大;ICC(0) 不精确 => 误差不被吸收 => 迭代升",
        ha="center", fontsize=8.8, color="0.15")
ax.text(7.25, 6.70, "上界 (Toselli & Widlund 2005):  λmax(M^-1 A) ≤ (N_c + 1)·ω    [N_c = 着色数 = 过计数,  ω = 不精确本地解]",
        ha="center", fontsize=8.6, color="#7c2d12", weight="bold")
ax.text(7.25, 6.34, "三视角:  信息传播(每迭代一子域跳)  |  Green 函数(Sys1 指数短程 / Sys2,3 代数 1/r 长程)  |  谱桥 iters~½√κ",
        ha="center", fontsize=8.2, color="#7c2d12")
arrow(7.25, 6.08, 7.25, 5.72, "对症")

frame(3.4, 4.35, 7.7, 1.30, "#dcfce7", "#16a34a")
ax.text(7.25, 5.38, "③ 改进:sASM  用单位分解 D^{-1/2}(·)D^{-1/2} 抵消过计数",
        ha="center", va="center", fontsize=13, weight="bold")
ax.text(7.25, 4.92, "=> λmax ≈ 1,重叠恢复\"越多越快\":Sys3 66 -> 55 -> 53 (-20%),O>=1 墙钟也更快",
        ha="center", fontsize=9, color="0.15")
ax.text(7.25, 4.56, "顺手做参数最优(L×O 扫描,按时间):椭圆 Sys2/3 -> L=1,O=1;质量主导 Sys1 -> L=0,O=1;时间最优 ≠ 迭代最优",
        ha="center", fontsize=8.4, color="#14532d")
arrow(7.25, 4.35, 7.25, 3.78, "暴露另一半瓶颈")

box(2.0, 2.85, 10.5, 0.90, "#dbeafe", "#2563eb",
    "④ 一层方法的两个根本瓶颈", "", 12.5, 9)
ax.text(7.25, 3.06, "λmax(重数)= sASM 已修   |   λmin(慢全局模)= 未修   |   绝对迭代数仍偏高",
        ha="center", fontsize=9.5, color="#1e3a8a", weight="bold")

# --- fan-out downstream:  arrows straight down from box4 (per-box center),
#     so the central label sits cleanly in the gap between arrows -----------
ax.text(7.15, 2.62, "顺藤摸瓜:后续工作都是这条线的自然延伸", ha="center",
        fontsize=9.5, style="italic", color="#b45309", weight="bold")
ds = [
    (0.35, "#ede9fe", "#7c3aed", "粗空间(coarse)", "修 λmin / 慢全局模\nNicolaides 1.34× 弱扩展平坦"),
    (3.75, "#ffe4e6", "#e11d48", "SORAS 强 fine", "压绝对迭代\n84->20 (np=2, 4.2×)"),
    (7.15, "#cffafe", "#0891b2", "跨时间历史", "改初值 warm/Fischer\n整拍 -12% (次力)"),
    (10.55, "#fef9c3", "#ca8a04", "迭代 ≠ 墙钟", "SORAS 本地精解贵\n小规模慢 20×,大规模翻正"),
]
for x, fc, ec, t, b in ds:
    cx = x + 1.7
    arrow(cx, 2.85, cx, 2.02, "")
    box(x, 1.05, 3.4, 0.95, fc, ec, t, b, 11, 8.2)

# --- large scale sink -----------------------------------------------------
for x, *_ in ds:
    arrow(x+1.7, 1.05, 7.25, 0.72, "")
box(2.6, 0.05, 9.3, 0.62, "#f3e8ff", "#6b21a8",
    "⑤ 大规模(~3000 核):保留 SORAS(固定本地解)+ 粗空间(telescope/GenEO 可扩展粗解)+ warm-start;丢弃 near-exact 本地解 + 稠密复制粗解",
    "", 9.5, 8, tc="#3b0764")

fig.savefig("fig_roadmap.png", dpi=150, bbox_inches="tight")
print("wrote fig_roadmap.png")
