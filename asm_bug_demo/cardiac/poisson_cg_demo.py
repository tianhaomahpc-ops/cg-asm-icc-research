#!/usr/bin/env python3
"""poisson_cg_demo.py -- the whole story on a REAL Poisson problem + CG.
Domain: elongated 2D grid nx=60 x ny=15 (long axis = x), 5-point stencil:
   4*u_ij - u_(i-1)j - u_(i+1)j - u_i(j-1) - u_i(j+1) = h^2 f_ij
(2D version of the bead chain:每个格点被4个邻居拉).
Closed-form modes: v_(k,l)(i,j) = sin(ik*pi/(nx+1))*sin(jl*pi/(ny+1)),
lambda_(k,l) = [2-2cos(k*pi/(nx+1))] + [2-2cos(l*pi/(ny+1))].
Deliverables:
  fig_poisson_modes.png : f, solution u, Richardson-vs-CG residuals,
                          slow-mode family (smooth along the LONG axis) + fast mode
  fig_poisson_cg.png    : spectrum / CG per-mode heatmap / deflated-CG heatmap
                          (shared lambda axis, honest pcolormesh positioning)
Deflation W is GEOMETRIC (per-slab constants; per-slab {1,x}), like the real system.
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

nx, ny = 60, 15
N = nx*ny
def T(n):
    return np.diag(2.*np.ones(n)) + np.diag(-np.ones(n-1),1) + np.diag(-np.ones(n-1),-1)
A = np.kron(np.eye(ny), T(nx)) + np.kron(T(ny), np.eye(nx))   # index = j*nx+i

lam, V = np.linalg.eigh(A)
kappa = lam[-1]/lam[0]
print(f"grid {nx}x{ny}, N={N};  lambda_min={lam[0]:.4f}  lambda_max={lam[-1]:.4f}  kappa={kappa:.0f}")

# classify smallest modes analytically
lx = 2-2*np.cos(np.arange(1,nx+1)*np.pi/(nx+1))
ly = 2-2*np.cos(np.arange(1,ny+1)*np.pi/(ny+1))
pairs = sorted([(lx[k]+ly[l], k+1, l+1) for k in range(nx) for l in range(ny)])[:10]
print("smallest 10 modes (lambda, k half-waves along LONG x, l along short y):")
for lv,k,l in pairs: print(f"   {lv:.4f}   k={k:2d}  l={l}")

# ---------------- solvers ----------------
rng = np.random.default_rng(0)
xstar = rng.standard_normal(N); b = A@xstar
TOL = 1e-8

def richardson(alpha, kmax=20000):
    x = np.zeros(N); nb = np.linalg.norm(b); res=[nb]
    for k in range(1, kmax+1):
        r = b - A@x; x += alpha*r
        nr = np.linalg.norm(r); res.append(nr)
        if nr < TOL*nb: return k, res
    return kmax, res

def cg(W=None, kmax=2000, track=False):
    if W is not None:
        E = W.T@A@W; Ei = np.linalg.inv(E)
        proj = lambda v: v - W@(Ei@(W.T@(A@v)))
        x = W@(Ei@(W.T@b)); r = b - A@x; p = proj(r.copy())
    else:
        x = np.zeros(N); r = b - A@x; p = r.copy()
    rr = r@r; nb = np.linalg.norm(b)
    H = [np.abs(V.T@(x-xstar))] if track else None
    res=[np.linalg.norm(r)]
    it=0
    for it in range(1, kmax+1):
        Ap = A@p; a = rr/(p@Ap); x += a*p; r -= a*Ap
        if track: H.append(np.abs(V.T@(x-xstar)))
        res.append(np.linalg.norm(r))
        if np.linalg.norm(r) < TOL*nb: break
        r2 = r@r
        pr = r if W is None else proj(r)
        p = pr + (r2/rr)*p; rr = r2
    return it, res, (np.array(H) if track else None)

a_opt = 2/(lam[-1]+lam[0])
it_rich, res_rich = richardson(a_opt)
it_cg, res_cg, Hp = cg(track=True)
print(f"\nRichardson (best alpha={a_opt:.4f}): {it_rich} steps to 1e-8")
print(f"plain CG:                        {it_cg} steps to 1e-8")

# geometric coarse spaces: slabs along long axis
def slab_W(nslab, linear=False):
    cols=[]
    xs = (np.arange(N)%nx).astype(float)          # x-index of each dof
    bounds = np.linspace(0, nx, nslab+1)
    for s in range(nslab):
        m = ((xs>=bounds[s]) & (xs<bounds[s+1])).astype(float)
        cols.append(m)
        if linear:
            xc = xs*m
            xc = np.where(m>0, xs-(bounds[s]+bounds[s+1])/2, 0.0)
            cols.append(xc)
    return np.array(cols).T
W_const = slab_W(6, linear=False)                 # 6 columns
W_lin   = slab_W(6, linear=True)                  # 12 columns
it_c, _, _   = cg(W=W_const)
it_l, _, Hd  = cg(W=W_lin, track=True)
print(f"+ 6 slabs, constants (dim=6):    {it_c} steps")
print(f"+ 6 slabs, {{1,x}} (dim=12):      {it_l} steps")

# ================= figure 1: setup + modes =================
fig = plt.figure(figsize=(16.5, 7.6))
gs = fig.add_gridspec(2, 6, height_ratios=[1.15, 1.0], hspace=0.5, wspace=0.35)
fig.suptitle("Poisson 方程(60x15 长条网格)+ CG:同一套概念的真实版", fontsize=14.5, weight="bold", y=0.99)

f = np.zeros(N); f[7*nx+15] = 1.0                 # point source at (15,7)
u = np.linalg.solve(A, f)
axf = fig.add_subplot(gs[0,0:2])
axf.imshow(f.reshape(ny,nx), cmap="Reds", aspect="auto", origin="lower")
axf.set_title("外源 f(一个点源)\nPoisson: 每格点 4u - 四邻居 = f", fontsize=9.6, weight="bold")
axf.set_xlabel("x (长轴)"); axf.set_ylabel("y")
axu = fig.add_subplot(gs[0,2:4])
im=axu.imshow(u.reshape(ny,nx), cmap="viridis", aspect="auto", origin="lower")
axu.set_title("解 u = A_逆·f(平滑的鼓包场)\n= 稳态温度/薄膜位移", fontsize=9.6, weight="bold")
axu.set_xlabel("x"); plt.colorbar(im, ax=axu, fraction=0.03)
axr = fig.add_subplot(gs[0,4:6])
axr.semilogy(np.array(res_rich)/res_rich[0], color="#7f8c8d", lw=1.8,
             label=f"最优简单迭代: {it_rich} 步")
axr.semilogy(np.array(res_cg)/res_cg[0], color="#c0392b", lw=2.2,
             label=f"CG: {it_cg} 步")
axr.axhline(1e-8, color="0.6", ls=":", lw=1)
axr.yaxis.set_major_formatter(mt.FuncFormatter(lambda v,_: f"1e{int(round(np.log10(v)))}" if v>0 else ""))
axr.set_xlim(0, 300)
axr.set_xlabel("迭代步"); axr.set_ylabel("相对残差")
axr.set_title("CG vs 手选最优步长的简单迭代\n(同一矩阵同一右端)", fontsize=9.6, weight="bold")
axr.legend(fontsize=9); axr.grid(alpha=0.25)

show = [(0,"最慢"),(1,""),(2,""),(3,""),(5,""),(N-1,"最快")]
for col,(mi,tag) in enumerate(show):
    ax = fig.add_subplot(gs[1,col])
    ax.imshow(V[:,mi].reshape(ny,nx), cmap="RdBu_r", aspect="auto", origin="lower")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{tag}模{mi+1}\nλ={lam[mi]:.3f}", fontsize=9, weight="bold",
                 color="#c0392b" if mi<8 else "#333333")
fig.text(0.5, 0.015, "最慢的一族模 = 沿『长轴』的平滑波(l=1,k=1,2,3,...):几何最长方向决定慢模 —— 与实测系统同构",
         ha="center", fontsize=10.5, color="#c0392b", weight="bold")
fig.savefig("fig_poisson_modes.png", dpi=135, bbox_inches="tight")
print("wrote fig_poisson_modes.png")

# ================= figure 2: spectrum + heatmaps =================
e0ref = Hp[0].copy(); e0ref[e0ref<1e-12]=1e-12
Np_ = np.clip(Hp/e0ref,0,1); Nd_ = np.clip(Hd/e0ref,0,1)
edges = np.concatenate([[lam[0]], 0.5*(lam[1:]+lam[:-1]), [lam[-1]]])

fig = plt.figure(figsize=(13.4, 11.6))
gs = fig.add_gridspec(3,1, height_ratios=[1.0,1.9,1.9], hspace=0.46)
fig.suptitle("Poisson+CG:谱 → 求解实况 → 几何粗空间(共用 λ 轴)", fontsize=14, weight="bold", y=0.96)

axS = fig.add_subplot(gs[0])
axS.scatter(lam, np.ones_like(lam), c=lam, cmap="viridis", s=22, edgecolor="none", alpha=0.85)
axS.scatter(lam[:6], np.ones(6), s=150, facecolor="none", edgecolor="#c0392b", lw=2.0)
axS.set_yticks([]); axS.set_xlim(-0.1, 8.1); axS.set_ylim(0.4,1.8)
axS.set_title(f"① 谱:{N} 个 λ;左端红圈 = 沿长轴的慢模家族(最小 {lam[0]:.3f}),κ={kappa:.0f}",
              fontsize=10.5, weight="bold")
axS.annotate("沿长轴平滑的一族\n= 慢尾(几何造成)", xy=(lam[2],1.06), xytext=(0.9,1.42),
             fontsize=9, color="#c0392b", arrowprops=dict(arrowstyle="->",color="#c0392b"))
axS.set_xlabel("特征值 λ", fontsize=9)

for pos, NN, ttl, itn, c in [
    (1, Np_, f"② 普通 CG:{it_cg} 步 -- 右侧大坨几步变黑,左端慢模家族磨到最后", it_cg, "#c0392b"),
    (2, Nd_, f"③ + 几何粗空间(6 段 x {{1,x}},dim=12):{it_l} 步 -- 左端开局即黑", it_l, "#27ae60")]:
    ax = fig.add_subplot(gs[pos])
    kN = NN.shape[0]-1
    X, Y = np.meshgrid(edges, np.arange(kN+2)-0.5)
    pc = ax.pcolormesh(X, Y, NN, cmap="magma", vmin=0, vmax=1, shading="flat")
    ax.invert_yaxis()
    ax.set_xlim(-0.1, 8.1)
    ax.axvline(lam[5]+0.02, color=c, ls="--", lw=1.4)
    ax.set_ylabel("CG 迭代步 k"); ax.set_xlabel("特征值 λ (同一根轴)", fontsize=9)
    ax.set_title(ttl, fontsize=10.5, weight="bold")
cax = fig.add_axes([0.93, 0.11, 0.015, 0.42])
fig.colorbar(pc, cax=cax).set_label("该模剩余误差比例 (亮=在,黑=解掉)", fontsize=9)
fig.savefig("fig_poisson_cg.png", dpi=135, bbox_inches="tight")
print("wrote fig_poisson_cg.png")
