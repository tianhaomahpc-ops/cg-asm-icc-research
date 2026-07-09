#!/usr/bin/env python3
"""fischer_demo.py -- the Fischer method itself, on a small reproducible SPD system.
A = 1D Laplacian (n=80).  A drifting RHS b_t = A x*_t where x*_t moves slowly on a
low-dim manifold (a few slow eigenmodes with time-varying amplitudes) -- mimicking
u_e(t) in the EP loop.  We solve A x = b_t for t=0..59 two ways:
  cold  : x0 = 0 every step
  Fischer: x0 = sum_i <p_i,b_t> p_i  (A-orthonormal history projection); after each
           solve, A-orthonormalise the new solution into the basis (sliding window).
Panels:
 (A) geometry cartoon: solution trajectory, history span, projection guess,
     leftover perpendicular part = what CG still has to solve.
 (B) real per-step CG iteration counts cold vs Fischer (drops as history builds).
Output: fig_fischer_demo.png ; also prints the algorithm's numbers.
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

# ---------- SPD system ----------
n = 80
A = np.diag(2.*np.ones(n)) + np.diag(-np.ones(n-1),1) + np.diag(-np.ones(n-1),-1)
lam, V = np.linalg.eigh(A)

def cg(b, x0, tol=1e-8, kmax=500):
    x = x0.copy(); r = b - A@x; p = r.copy(); rr = r@r
    nb = np.linalg.norm(b); it = 0
    if np.linalg.norm(r) < tol*nb: return x, 0        # guess already good
    for it in range(1, kmax+1):
        Ap = A@p; a = rr/(p@Ap); x += a*p; r -= a*Ap
        if np.linalg.norm(r) < tol*nb: break
        rr2 = r@r; p = r + (rr2/rr)*p; rr = rr2
    return x, it

# ---------- RHS with an action-potential-like time course (like u_e(t)) ----------
# A fixed spatial field s, modulated by an AP envelope e(t): fast upstroke, long
# near-STATIC plateau, repolarisation.  During the plateau x*(t) barely changes
# step-to-step, so a warm/Fischer guess is already within tol -> 0 iters (exactly
# the real EP behaviour).  A slow secondary drift keeps feeding the basis genuine
# new content.  This reproduces the real spiky-then-zero iteration pattern.
T = 60
grid = np.arange(n)
s1 = np.exp(-0.5*((grid-32)/16)**2)          # fixed smooth spatial shape 1
s2 = np.sin(np.pi*grid/(n-1))                # smooth spatial shape 2 (drift)
def env(t):                                  # AP envelope: rise 0-10, plateau, fall 42-55
    if t < 10:   return t/10.0
    if t < 42:   return 1.0 - 0.02*(t-10)     # gentle plateau drift
    if t < 55:   return 0.36 - 0.36*(t-42)/13
    return 0.0
def xstar(t):
    # 0.08*s1 baseline (b never 0) + AP envelope + a slow secondary drift, all inside
    # a low-dim smooth subspace -> Fischer learns it in ~2 steps and then nails every
    # guess (clean -97%).  The REAL EP loop is messier (-63%, fig_fischer_eploop.png)
    # because u_e also has genuine regime changes (QRS) that force periodic re-learning.
    return 0.08*s1 + env(t)*s1 + 0.35*np.sin(0.11*t)*s2

# ---------- Fischer state ----------
FMAX = 8
P, AP = [], []                       # A-orthonormal history and A*that
def fischer_guess(b):
    x = np.zeros(n)
    for pi in P: x += (pi @ b) * pi   # x0 = sum_i <p_i,b> p_i
    return x
def fischer_grow(xnew):
    w = xnew.copy(); Aw = A@w
    ref = np.sqrt(w @ Aw)             # A-norm of the whole new solution
    for i in range(len(P)):           # A-orthogonalise (modified Gram-Schmidt in A)
        c = AP[i] @ w; w -= c*P[i]; Aw -= c*AP[i]
    nrm = np.sqrt(w @ Aw)
    # RELATIVE drop test: only keep the leftover if it is a genuinely new direction
    # (a non-negligible fraction of the solution), not roundoff noise.  Keeping
    # noise and normalising it to unit A-norm is what wrecks a naive basis.
    if nrm > 1e-3 * ref:
        w /= nrm; Aw /= nrm
        if len(P) >= FMAX:            # sliding window: drop oldest
            P.pop(0); AP.pop(0)
        P.append(w); AP.append(Aw)

it_cold, it_fis = [], []
for t in range(T):
    b = A @ xstar(t)
    _, kc = cg(b, np.zeros(n));         it_cold.append(kc)
    xf, kf = cg(b, fischer_guess(b));   it_fis.append(kf)
    fischer_grow(xf)
it_cold = np.array(it_cold); it_fis = np.array(it_fis)
print(f"cold sum={it_cold.sum()}  fischer sum={it_fis.sum()}  "
      f"reduction {100*(1-it_fis.sum()/it_cold.sum()):.0f}%  "
      f"#steps Fischer=0: {(it_fis==0).sum()}/{T}")

# ================= figure =================
fig = plt.figure(figsize=(15.2, 6.0))
gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.25], wspace=0.26)
fig.suptitle("Fischer 回收:用历史解的 A-正交基投影出初值,只解剩下的垂直分量",
             fontsize=13.5, weight="bold", y=1.0)

# (A) geometry cartoon
axA = fig.add_subplot(gs[0])
axA.set_xlim(-0.2, 3.4); axA.set_ylim(-0.4, 3.0); axA.axis("off")
# history span = a shaded band (a line here, "span of p_i")
axA.plot([0.2,3.0],[0.5,1.4], color="#2980b9", lw=3, solid_capstyle="round")
axA.text(2.55,1.05,"历史张成的子空间\nspan{p_1..p_m}", color="#2980b9", fontsize=10)
# true solution point
xs = np.array([1.7, 2.5])
axA.plot(*xs, "*", color="#16a085", ms=20, zorder=6)
axA.text(1.78,2.55,"本步真解 x*(t)", color="#16a085", fontsize=10.5, weight="bold")
# projection (guess) foot
g = np.array([1.55, 1.02])
axA.plot(*g, "o", color="#c0392b", ms=11, zorder=6)
axA.annotate("", xy=g, xytext=(0,0), arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=2.5))
axA.text(0.35,0.28,"Fischer 初值 x0\n= x* 在子空间上的\nA-正交投影\n= Σ⟨p_i,b⟩p_i", color="#c0392b", fontsize=10)
# perpendicular leftover
axA.annotate("", xy=xs, xytext=g, arrowprops=dict(arrowstyle="-|>", color="#8e44ad", lw=2.5))
axA.text(1.75,1.75,"剩下的垂直分量\n= CG 唯一要解的部分\n(历史越全,它越短→迭代越少)",
         color="#8e44ad", fontsize=10)
axA.plot([g[0],g[0]+0.12],[g[1],g[1]+0.08],color="0.5",lw=1)  # right-angle tick hint
axA.set_title("① 几何:初值 = 真解在历史子空间上的投影;\n只有垂直于历史的那点误差还要 CG 解",
              fontsize=10.6, weight="bold")

# (B) real iteration counts
axB = fig.add_subplot(gs[1])
tt = np.arange(T)
axB.plot(tt, it_cold, "-", color="#c0392b", lw=2.0, label=f"冷启动 x0=0 (总 {it_cold.sum()})")
axB.plot(tt, it_fis, "-o", color="#27ae60", lw=2.2, ms=4,
         label=f"Fischer 回收 (总 {it_fis.sum()}, -{100*(1-it_fis.sum()/it_cold.sum()):.0f}%)")
axB.axvspan(0, FMAX, color="#eafaf1", alpha=0.8)
axB.text(0.4, it_cold.max()*0.5, f"前 {FMAX} 步\n基在成长", fontsize=9, color="#1e8449")
axB.set_xlabel("时间步 t (每步解 A x = b_t)"); axB.set_ylabel("该步 CG 迭代数")
axB.set_title("② 真实迭代数(n=80 可复现):历史攒满(~2步)后\nFischer 初值已在容差内 → 0 迭代(理想低维情形)", fontsize=10.6, weight="bold")
axB.legend(fontsize=10, loc="center right"); axB.grid(alpha=0.25); axB.set_ylim(-1, it_cold.max()+3)

fig.savefig("fig_fischer_demo.png", dpi=140, bbox_inches="tight")
print("wrote fig_fischer_demo.png")
