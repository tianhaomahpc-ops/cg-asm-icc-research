#!/usr/bin/env python3
"""plot_vector_two_views.py -- answer two confusions with the SAME n=3 chain:
(1) a vector is just an ordered list of numbers; there are TWO ways to draw it:
    as an ARROW in 3D (good for angles/perpendicularity) or as a PROFILE over
    the bead index (good for shape: sign flips / waves).  The components are
    NOT spatial coordinates of a point in the room -- component i is the
    displacement of bead i.
(2) the spectrum is a set of numbers, so its natural picture is dots on a
    number line; the POINT of that picture is that mode i's speed depends only
    on the number lambda_i (factor 1-alpha*lambda_i), so the number line IS a
    complete difficulty map.
Output: fig_vector_two_views.png
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

SQ2 = np.sqrt(2.0)
v1 = np.array([1, SQ2, 1]); v2 = np.array([1, 0, -1]); v3 = np.array([1, -SQ2, 1])
lams = [2-SQ2, 2.0, 2+SQ2]
cols = ["#c0392b", "#2980b9", "#8e44ad"]
alpha = 0.25

fig = plt.figure(figsize=(16.2, 5.9))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.15], wspace=0.3)
fig.suptitle("同一组数字的两种画法 + 谱为什么画成数轴上的点 (n=3 链)",
             fontsize=14.5, weight="bold", y=1.02)

# ---- (A) arrow view: 3D ----
axA = fig.add_subplot(gs[0], projection="3d")
for v, c, nm in [(v1,cols[0],"模1 (1, 1.41, 1)"), (v2,cols[1],"模2 (1, 0, -1)"),
                 (v3,cols[2],"模3 (1, -1.41, 1)")]:
    axA.quiver(0,0,0, v[0],v[1],v[2], color=c, lw=2.4, arrow_length_ratio=0.09)
    axA.text(v[0]*1.12, v[1]*1.12, v[2]*1.12, nm, color=c, fontsize=9)
axA.set_xlim(-1.6,1.6); axA.set_ylim(-1.6,1.6); axA.set_zlim(-1.6,1.6)
axA.set_xlabel("第1个数", fontsize=8); axA.set_ylabel("第2个数", fontsize=8)
axA.set_zlabel("第3个数", fontsize=8)
axA.set_title("画法A『箭头』:把 3 个数当一个箭头的坐标\n"
              "用途:看角度 -- 三支箭互相垂直(点积=0)\n"
              "注意:这三根轴是『第几个数』,不是房间里的 x y z", fontsize=9.6, weight="bold")
axA.tick_params(labelsize=7)

# ---- (B) profile view ----
axB = fig.add_subplot(gs[1])
pos = np.array([1,2,3])
for j,(v,c) in enumerate(zip([v1,v2,v3],cols)):
    off = 2-j
    axB.axhline(off, color="0.85", lw=0.8, zorder=0)
    axB.plot(pos, off+v/np.max(np.abs(v))*0.33, "o-", color=c, lw=2.2, ms=9)
    flips = int(np.sum(np.diff(np.sign(v[np.abs(v)>1e-12]))!=0))
    axB.text(3.5, off, f"翻转{flips}次", fontsize=9, color=c, va="center")
axB.set_yticks([2,1,0]); axB.set_yticklabels(["模1","模2","模3"])
axB.set_xticks([1,2,3]); axB.set_xlim(0.6,4.3)
axB.set_xlabel("珠子编号 i (=第几个分量)")
axB.set_title("画法B『剖面』:横轴=第几颗珠子,纵轴=那个数\n"
              "分量 i 的含义 = 珠子 i 的偏移量\n"
              "用途:看形状 -- 鼓包/穿零/锯齿(翻转次数)", fontsize=9.6, weight="bold")

# ---- (C) spectrum as number line + why ----
axC = fig.add_subplot(gs[2])
for lam, c, nm in zip(lams, cols, ["模1","模2","模3"]):
    f = 1-alpha*lam
    axC.plot(lam, 0, "o", color=c, ms=14)
    axC.annotate(f"{nm}\nλ={lam:.3f}", xy=(lam,0), xytext=(lam, -0.5), ha="center",
                 fontsize=9, color=c)
    axC.annotate(f"每步误差\nx{abs(f):.3f}", xy=(lam,0), xytext=(lam, 0.42), ha="center",
                 fontsize=9.5, color=c, weight="bold")
axC.axhline(0, color="0.4", lw=1.2)
axC.set_xlim(0, 4.0); axC.set_ylim(-1.0, 1.0); axC.set_yticks([])
axC.set_xlabel("特征值 λ (数轴)")
axC.set_title("谱 = 每个模的『速度参数』摆上数轴\n"
              "速度只由数 λ 决定(每步 x|1-αλ|),与形状/方向无关\n"
              "所以这张图 = 完整的难度地图", fontsize=9.6, weight="bold")
axC.annotate("最左点 = 最慢模\n= κ 的分母\n= 粗空间要铲的", xy=(lams[0],-0.06),
             xytext=(0.35,-0.92), fontsize=8.8, color=cols[0],
             arrowprops=dict(arrowstyle="->", color=cols[0]))
axC.annotate("最右点定 α 上限\nα<2/λmax", xy=(lams[2],-0.06), xytext=(2.9,-0.92),
             fontsize=8.8, color=cols[2],
             arrowprops=dict(arrowstyle="->", color=cols[2]))

fig.savefig("fig_vector_two_views.png", dpi=140, bbox_inches="tight")
print("wrote fig_vector_two_views.png")
