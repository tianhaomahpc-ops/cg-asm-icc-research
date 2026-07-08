#!/usr/bin/env python3
"""tutorial_chain.py -- ONE example (a chain of beads) carried through the whole
story: iterative method, eigenvector/eigenvalue, spectrum, per-mode contraction,
alpha cap, CG, coarse space.  Prints every number quoted in
TUTORIAL_chain_zh.md and writes two figures:
  fig_tut_chain.png     n=3 hand scale: mode shapes + per-mode error decay
  fig_tut_spectrum.png  n=100 grown chain: spectrum / CG heatmap / deflated CG
                        heatmap, all sharing the same lambda axis.
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

SQ2 = np.sqrt(2.0)

# ================= PART A: hand scale n=3 =================
print("="*74)
print("PART A: chain with n=3 beads,  A = [[2,-1,0],[-1,2,-1],[0,-1,2]]")
print("="*74)
A = np.array([[2.,-1.,0.],[-1.,2.,-1.],[0.,-1.,2.]])
v1 = np.array([1., SQ2, 1.]);  v2 = np.array([1., 0., -1.]);  v3 = np.array([1., -SQ2, 1.])
for v, name in [(v1,"(1,r2,1)"), (v2,"(1,0,-1)"), (v3,"(1,-r2,1)")]:
    Av = A@v
    lam = Av[0]/v[0]
    print(f"  A*{name} = {np.round(Av,4)} = {lam:.4f} * {name}")
lam1, lam2, lam3 = 2-SQ2, 2.0, 2+SQ2
print(f"  eigenvalues (the spectrum): {lam1:.4f}, {lam2:.4f}, {lam3:.4f}")
print(f"  condition number kappa = {lam3/lam1:.3f}")
print(f"  alpha cap = 2/lambda_max = {2/lam3:.4f}")

alpha = 0.25
f1, f2, f3 = 1-alpha*lam1, 1-alpha*lam2, 1-alpha*lam3
print(f"\n  per-mode factor 1-alpha*lambda at alpha={alpha}:")
for lam, f in [(lam1,f1),(lam2,f2),(lam3,f3)]:
    steps = 1 if f==0 else int(np.ceil(np.log(1e-6)/np.log(abs(f))))
    print(f"    lambda={lam:.4f}: factor {f:.4f}  -> steps to 1e-6: {steps}")

b = np.array([1.,0.,1.]); xstar = np.array([1.,1.,1.])   # A*(1,1,1)=(1,0,1)
print(f"\n  check: A*(1,1,1) = {A@xstar} = b   -> answer x*=(1,1,1)")
x = np.zeros(3)
print("  iteration x <- x + 0.25*(b - A x), start x0=(0,0,0):")
e2_direct = None
for k in range(1,4):
    r = b - A@x; x = x + alpha*r
    if k == 2: e2_direct = (x-xstar).copy()
    print(f"    step {k}: r={np.round(r,4)}  x={np.round(x,4)}  e=x-x*={np.round(x-xstar,4)}")
# two-route check at k=2
print(f"\n  mode lengths: |(1,r2,1)| = sqrt(1+2+1) = 2;  |(1,0,-1)| = sqrt(2) = {SQ2:.4f}")
n1 = v1/2.0; n2 = v2/SQ2; n3 = v3/2.0                     # unit vectors
print(f"  unit modes: v^1={np.round(n1,4)}  v^2={np.round(n2,4)}  v^3={np.round(n3,4)}")
print(f"  orthogonality: v1.v2={v1@v2:.0f}  v1.v3={v1@v3:.0f}  v2.v3={v2@v3:.0f}   (all zero)")
e0 = -xstar
c1, c2, c3 = e0@n1, e0@n2, e0@n3
print(f"  decompose e0=-(1,1,1): c1=e0.v^1={c1:.4f}(slow) c2={c2:.4f}(mid) c3={c3:.4f}(fast)")
print(f"  after 2 steps, coefficients: c1*f1^2={c1*f1**2:.4f}   c3*f3^2={c3*f3**2:.4f}")
e2_modes = c1*f1**2*n1 + c2*f2**2*n2 + c3*f3**2*n3
print(f"  eigen route  e2 = c1*f1^2*v^1 + c3*f3^2*v^3 = {np.round(e2_modes,4)}")
print(f"  direct route e2 (from step-2 row above)    = {np.round(e2_direct,4)}   <- SAME numbers")

# ---------- figure A ----------
fig, (axm, axd) = plt.subplots(1, 2, figsize=(14.8, 5.8))
fig.suptitle("同一条链(n=3 颗珠子):左=三个模的形状与 λ,右=迭代中每个模按自己的 λ 打折",
             fontsize=13.5, weight="bold", y=1.0)
pos = np.array([1,2,3]); cols = ["#c0392b", "#2980b9", "#8e44ad"]
names = [f"模1 平滑鼓包, λ={lam1:.3f} (慢)", f"模2 一次穿零, λ={lam2:.0f}",
         f"模3 锯齿,     λ={lam3:.3f} (快)"]
for j,(v,c,nm) in enumerate(zip([v1,v2,v3],cols,names)):
    off = 2-j*1.0
    vv = v/np.max(np.abs(v))*0.32
    xs = np.linspace(0.5,3.5,100)
    kk = j+1
    axm.plot(xs, off+0.32*np.sin(kk*np.pi*(xs-0.5)/3)* (1 if j!=2 else 1), color=c, lw=1.0, alpha=0.35)
    axm.plot(pos, off+vv, "o-", color=c, lw=2.2, ms=10, label=nm)
    axm.axhline(off, color="0.8", lw=0.7, zorder=0)
    axm.text(3.62, off, f"{kk} 个半波", fontsize=9, color=c, va="center")
axm.set_yticks([2,1,0]); axm.set_yticklabels(["模1","模2","模3"])
axm.set_xticks([1,2,3]); axm.set_xlabel("珠子编号 (细线 = 对应的正弦驻波)")
axm.set_title("① 三个特殊方向(特征向量)= 链的三种驻波\n波越长 λ 越小,波越碎 λ 越大", fontsize=10.5, weight="bold")
axm.set_xlim(0.4,4.4); axm.legend(fontsize=8.6, loc="lower left")

ks = np.arange(0,61)
for lam,c,lab in [(lam1,"#c0392b",f"模1: 每步 x{f1:.3f} -> 88 步"),
                  (lam2,"#2980b9",f"模2: 每步 x{f2:.3f} -> 20 步 (本例初始为0)"),
                  (lam3,"#8e44ad",f"模3: 每步 x{f3:.3f} -> 8 步")]:
    f = abs(1-alpha*lam)
    axd.semilogy(ks, np.clip(f**ks,1e-8,None), color=c, lw=2.3, label=lab)
axd.axhline(1e-6, color="0.55", ls=":", lw=1.1); axd.text(48,2e-6,"目标 1e-6",fontsize=8.5,color="0.4")
axd.yaxis.set_major_formatter(mt.FuncFormatter(lambda v,_: f"1e{int(round(np.log10(v)))}" if v>0 else ""))
axd.set_xlabel("迭代步 k"); axd.set_ylabel("该模误差剩余比例 (log)")
axd.set_title("② 同一次迭代里,三个模各自独立收敛\n折扣率 = 1-αλ (α=0.25 < 上限 2/λmax=0.586)", fontsize=10.5, weight="bold")
axd.legend(fontsize=9, loc="upper right"); axd.grid(alpha=0.25); axd.set_ylim(1e-8,2)
fig.savefig("fig_tut_chain.png", dpi=140, bbox_inches="tight")
print("\nwrote fig_tut_chain.png")

# ================= PART B: grown chain n=100 =================
print()
print("="*74)
print("PART B: same chain grown to n=100")
print("="*74)
n = 100
A = np.diag(2.*np.ones(n)) + np.diag(-np.ones(n-1),1) + np.diag(-np.ones(n-1),-1)
lam, V = np.linalg.eigh(A)
print(f"  spectrum: lambda_1={lam[0]:.6f} ... lambda_100={lam[-1]:.4f}, kappa={lam[-1]/lam[0]:.0f}")
print(f"  smallest 9: {np.round(lam[:9],4)}   (9th = new lambda_min after deflating 8)")
xs = np.random.default_rng(0).standard_normal(n)
b = A@xs

def cg(W=None, kmax=300, tol=1e-8, track=False):
    if W is not None:
        E = W.T@A@W; Ei = np.linalg.inv(E)
        proj = lambda v: v - W@(Ei@(W.T@(A@v)))
        x = W@(Ei@(W.T@b)); r = b - A@x; p = proj(r.copy())
    else:
        x = np.zeros(n); r = b - A@x; p = r.copy()
    rr = r@r; H = [np.abs(V.T@(x-xs))]
    it = 0
    for it in range(1, kmax+1):
        Ap = A@p; a = rr/(p@Ap); x += a*p; r -= a*Ap
        H.append(np.abs(V.T@(x-xs)))
        if np.linalg.norm(r) < tol*np.linalg.norm(b): break
        r2 = r@r
        p = (r if W is None else (r - W@(np.linalg.inv(W.T@A@W)@(W.T@(A@r))))) + (r2/rr)*p
        rr = r2
    return it, np.array(H)

it_plain, Hp = cg()
m = 8
it_defl, Hd = cg(W=V[:,:m].copy())
print(f"  plain CG: {it_plain} steps;   deflated CG (m={m} smoothest modes): {it_defl} steps")

e0ref = Hp[0].copy(); e0ref[e0ref<1e-12] = 1e-12
Np, Nd = np.clip(Hp/e0ref,0,1), np.clip(Hd/e0ref,0,1)

fig = plt.figure(figsize=(13.2, 11.5))
gs = fig.add_gridspec(3,1, height_ratios=[1.0,1.9,1.9], hspace=0.44)
fig.suptitle("链加长到 n=100:三张图共用同一根横轴 = λ,竖着对齐就是同一个模",
             fontsize=14, weight="bold", y=0.96)
axS = fig.add_subplot(gs[0])
axS.scatter(lam, np.ones_like(lam), c=lam, cmap="viridis", s=42, edgecolor="k", lw=0.3)
axS.scatter(lam[:m], np.ones(m), s=140, facecolor="none", edgecolor="#c0392b", lw=2.0)
axS.set_yticks([]); axS.set_xlim(-0.06, 4.06); axS.set_ylim(0.4,1.8)
axS.set_title(f"① 谱 = 100 个 λ 的集合;左端稀薄的尾巴(红圈 8 个,最小 {lam[0]:.4f})就是慢模",
              fontsize=10.5, weight="bold")
axS.annotate("长波驻波\n= 小 λ = 慢", xy=(lam[2],1.05), xytext=(0.35,1.4), fontsize=9,
             color="#c0392b", arrowprops=dict(arrowstyle="->",color="#c0392b"))
axS.annotate("碎波驻波 = 大 λ = 快\n(挤成一坨,好解)", xy=(lam[-8],1.05), xytext=(2.55,1.4),
             fontsize=9, color="#1a5276", arrowprops=dict(arrowstyle="->",color="#1a5276"))
axS.set_xlabel("特征值 λ", fontsize=9)

for axpos, N, ttl, itn, c in [(1, Np, "② 普通 CG 求解实况:一行=一步,亮=该模误差还在,黑=已解掉", it_plain, "#c0392b"),
                              (2, Nd, f"③ 粗空间(m={m})CG:左边 8 列开局即黑(被 W·E^-1·W^T 直接除掉)", it_defl, "#27ae60")]:
    ax = fig.add_subplot(gs[axpos])
    kN = N.shape[0]-1
    im = ax.imshow(N, aspect="auto", origin="upper", cmap="magma",
                   extent=[lam[0], lam[-1], kN, 0], vmin=0, vmax=1)
    ax.set_xlim(-0.06, 4.06)
    ax.axvline(lam[m-1]+0.02, color=c, ls="--", lw=1.4)
    ax.set_ylabel("迭代步 k"); ax.set_xlabel("特征值 λ (同一根轴)", fontsize=9)
    ax.set_title(ttl + f" -> 共 {itn} 步", fontsize=10.5, weight="bold")
cax = fig.add_axes([0.93, 0.11, 0.015, 0.42])
fig.colorbar(im, cax=cax).set_label("该模剩余误差比例", fontsize=9)
fig.savefig("fig_tut_spectrum.png", dpi=135, bbox_inches="tight")
print("wrote fig_tut_spectrum.png")
