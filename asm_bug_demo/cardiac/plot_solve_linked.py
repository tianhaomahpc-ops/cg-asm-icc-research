#!/usr/bin/env python3
"""plot_solve_linked.py -- ONE linked story, all panels share x-axis = eigenvalue.

Matrix: 1D Poisson A=tridiag(-1,2,-1), n=40. Eigenpairs closed-form.
We SOLVE Ax=b with CG and record, every iteration, how much error is LEFT in each
mode: e_i(k) = |v_i^T (x_k - x*)|.  Plotting that as a heatmap (rows=iteration,
cols=mode ordered by lambda) literally shows the solve: bright=error left,
dark=solved. The columns line up under the spectrum above (same lambda axis), so
you SEE that the lingering bright columns ARE the small-lambda tail. The coarse
space blacks those columns out at step 0.
Output: fig_solve_linked.png
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

n = 100
A = np.diag(2.0*np.ones(n)) + np.diag(-np.ones(n-1),1) + np.diag(-np.ones(n-1),-1)
lam, V = np.linalg.eigh(A)                       # ascending; columns = modes
xstar = np.random.default_rng(0).standard_normal(n)   # x* excites ALL modes
b = A@xstar

def cg_track(A, b, x0, W=None, kmax=300, tol=1e-10):
    """Plain CG, or TRUE deflated CG (Saad) when W given: the W modes are projected
    out at start and kept out every step -> those modes' error stays ~0 (black).
    Returns per-mode |error| each step and the residual history."""
    if W is None:
        x = x0.copy(); r = b - A@x; p = r.copy(); rr = r@r
        E_hist = [np.abs(V.T@(x-xstar))]; res = [np.linalg.norm(r)]
        for _ in range(kmax):
            Ap = A@p; a = rr/(p@Ap)
            x = x + a*p; r = r - a*Ap
            E_hist.append(np.abs(V.T@(x-xstar))); res.append(np.linalg.norm(r))
            if res[-1] < tol*res[0]: break
            rr2 = r@r; p = r + (rr2/rr)*p; rr = rr2
        return np.array(E_hist), np.array(res)
    # ---- deflated CG ----
    E = W.T@A@W; Einv = np.linalg.inv(E)
    def proj(v):                                 # remove W-modes in A-inner-product
        return v - W@(Einv@(W.T@(A@v)))
    x = x0 + W@(Einv@(W.T@(b - A@x0)))           # coarse solve: kills W modes at step 0
    r = b - A@x                                  # W^T r = 0 by construction
    p = proj(r.copy()); rr = r@r
    E_hist = [np.abs(V.T@(x-xstar))]; res = [np.linalg.norm(r)]
    for _ in range(kmax):
        Ap = A@p; a = rr/(p@Ap)
        x = x + a*p; r = r - a*Ap
        E_hist.append(np.abs(V.T@(x-xstar))); res.append(np.linalg.norm(r))
        if res[-1] < tol*res[0]: break
        rr2 = r@r; p = proj(r) + (rr2/rr)*p; rr = rr2
    return np.array(E_hist), np.array(res)

x0 = np.zeros(n)
E_plain, res_plain = cg_track(A, b, x0)
m = 8
W = V[:, :m].copy()                              # coarse space = 8 smoothest modes
E_defl,  res_defl  = cg_track(A, b, x0, W=W)

# normalise BOTH heatmaps by the SAME reference = plain run's initial per-mode
# error. (Dividing each mode by its own start would make the deflated modes -- which
# begin at ~1e-15 -- blow tiny roundoff up into fake brightness.)
e0ref = E_plain[0].copy(); e0ref[e0ref < 1e-12] = 1e-12
def norm_map(Emap):
    return np.clip(Emap / e0ref, 0.0, 1.0)
H_plain = norm_map(E_plain)
H_defl  = norm_map(E_defl)

# =================== figure: 3 rows, shared lambda axis ===================
fig = plt.figure(figsize=(13.5, 12.0))
gs = fig.add_gridspec(3, 1, height_ratios=[1.15, 2.0, 2.0], hspace=0.42)
fig.suptitle("同一根横轴 = 特征值 λ:上面是谱的形状,下面两张热力图是'怎么解出来的'",
             fontsize=15, weight="bold", y=0.965)

xext = [lam[0], lam[-1]]

# ---- Row A: the spectrum shape ----
axA = fig.add_subplot(gs[0])
axA.stem(lam, np.ones_like(lam), linefmt="0.7", markerfmt=" ", basefmt=" ")
axA.scatter(lam, np.ones_like(lam), c=lam, cmap="viridis", s=55, zorder=5, edgecolor="k", lw=0.3)
axA.scatter(lam[:m], np.ones(m), s=150, facecolor="none", edgecolor="#c0392b", lw=2.2, zorder=6)
axA.set_ylim(0,1.6); axA.set_yticks([])
axA.set_xlim(-0.05, lam[-1]+0.05)
axA.set_title("① 谱的形状 = 100 个 λ 在数轴上怎么分布:右边挤成一坨(好解),"
              "左边稀稀拉拉拖一条尾巴(难解)", fontsize=11, weight="bold")
axA.annotate("左端尾巴:最小的 8 个 λ\n(红圈,平滑慢模)", xy=(lam[1],1.0), xytext=(0.35,1.35),
             fontsize=9.5, color="#c0392b", arrowprops=dict(arrowstyle="->",color="#c0392b"))
axA.annotate("右端一大坨挤在一起\n(锯齿快模)", xy=(lam[-6],1.0), xytext=(2.7,1.35),
             fontsize=9.5, color="#1a5276", arrowprops=dict(arrowstyle="->",color="#1a5276"))
axA.set_xlabel("特征值 λ  (每个点 = 一个模;颜色 = 这个模,一路对应到下面)", fontsize=9.5)

# ---- Row B: plain CG solve as heatmap ----
axB = fig.add_subplot(gs[1])
kB = H_plain.shape[0]-1
im = axB.imshow(H_plain, aspect="auto", origin="upper", cmap="magma",
                extent=[lam[0], lam[-1], kB, 0], vmin=0, vmax=1)
axB.set_ylabel("CG 迭代步 k (从上往下走)", fontsize=10)
axB.set_title("② 普通 CG 怎么解:每一行是一步,颜色 = 这个模还剩多少误差"
              "(亮=没解掉, 黑=已解掉)", fontsize=11, weight="bold")
axB.set_xlabel("特征值 λ  (和上图同一根轴)", fontsize=9.5)
axB.text(1.6, 12, "右边(大 λ)几步就变黑 → 快头", color="0.2", fontsize=10, weight="bold")
axB.text(0.12, kB*0.6, "左边这几列一直亮\n= 慢模拖长尾巴\n(正对上图红圈)", color="#c0392b",
         fontsize=10, weight="bold")
axB.axvline(lam[m-1]+0.02, color="#c0392b", ls="--", lw=1.5)
axB.text(1.6, kB*0.55, f"普通 CG 一共 {len(res_plain)-1} 步", color="#c0392b",
         fontsize=10, weight="bold")

# ---- Row C: deflated CG solve ----
axC = fig.add_subplot(gs[2])
kC = H_defl.shape[0]-1
axC.imshow(H_defl, aspect="auto", origin="upper", cmap="magma",
           extent=[lam[0], lam[-1], kC, 0], vmin=0, vmax=1)
axC.set_ylabel("CG 迭代步 k", fontsize=10)
axC.set_title("③ 粗空间 deflated CG:开工前先用 W·E^-1·W^T 把左边 8 列一次解准"
              "(第 0 行就黑了)", fontsize=11, weight="bold")
axC.set_xlabel("特征值 λ  (同一根轴)", fontsize=9.5)
axC.axvline(lam[m-1]+0.02, color="#27ae60", ls="--", lw=1.5)
axC.text(0.06, kC*0.42, "这 8 列从一开始\n就是黑的\n(被粗空间铲掉)", color="#27ae60",
         fontsize=10, weight="bold")
axC.text(1.6, kC*0.5, f"只剩右边要磨,{len(res_defl)-1} 步搞定", color="#27ae60",
         fontsize=10, weight="bold")

# shared colorbar
cax = fig.add_axes([0.92, 0.12, 0.015, 0.45])
cb = fig.colorbar(im, cax=cax); cb.set_label("该模还剩的相对误差 (亮=1 没解, 黑=0 解掉)", fontsize=9)

fig.savefig("fig_solve_linked.png", dpi=135, bbox_inches="tight")
print("wrote fig_solve_linked.png")
print(f"plain CG  {len(res_plain)-1} steps;  deflated CG {len(res_defl)-1} steps")
print(f"smallest 4 lambda (the tail / removed): {np.round(lam[:4],4)}")
