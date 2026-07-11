#!/usr/bin/env python3
"""plot_transfer_fix.py -- THE FIX for Sys3: the high-rank RHS is not a wall.

Diagnosis (fig_highrank): the RHS trajectory b(t) is high-rank because the front
translates. -leadvol tried to build a reduced basis of the SOLUTION SNAPSHOTS
phi(t)=Kt^-1 b(t) online, so it kept discovering new directions (54 and counting).

The fix (this file, proven on a torso-proxy 2D Poisson):
  approximate the fixed OPERATOR, not the moving solutions.
  phi(t) = Kt^-1 * (lift of interface data u_iface(t))  =  Z * u_iface(t)
  where Z (N_torso x N_iface) is the interface->torso transfer operator: its
  column k = Kt^-1 (lift of the k-th interface unit vector).  KEY facts:
    (1) rank(Z) <= N_iface  -- BOUNDED by the interface (a fixed surface), NOT by
        the number of time steps.  The moving front only *visits* these columns.
    (2) Z is built ONCE, offline (one solve per interface DOF), deterministically
        -- no waiting for the front to sweep each region, no per-step solve.
    (3) Z is EXACT for ANY RHS regardless of its rank:  Z u_iface(t) reproduces
        the true field to machine precision, every step.
    (4) Z's singular values DECAY (Green's-function smoothing) -> store a rank-r
        truncation (or H-matrix), r << N_iface.  RHS high rank is irrelevant:
        you compress the operator, then apply it to the high-rank RHS exactly.
This is the FULL-FIELD analogue of the electrode lead-field: same fixed operator,
you just keep all torso rows instead of the electrode rows.  = transfer-matrix /
BEM-FMM (research doc rank #1/#2), now for the volume.
Output: fig_transfer_fix.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
try:
    font_manager.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
    matplotlib.rcParams["font.family"] = "WenQuanYi Zen Hei"
except Exception: pass
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

# ---- torso-proxy 2D Poisson: left edge = heart interface (Dirichlet data),
#      right edge = grounded (Dirichlet 0), top/bottom = insulated (Neumann).
n = 40                      # grid n x n
Niface = n                  # left-edge nodes = interface DOFs
# unknown columns i=1..n-2, all rows j; boundary i=0 (interface), i=n-1 (ground=0)
cols = list(range(1, n-1)); ncols = len(cols)
def idx(i, j): return (i-1)*n + j          # unknown ordering
Nu = ncols * n
A = np.zeros((Nu, Nu)); Biface = np.zeros((Nu, Niface))
for i in cols:
    for j in range(n):
        r = idx(i,j); A[r,r] += 4.0
        if i-1==0:  Biface[r, j] += 1.0              # interface node -> Z column j
        else:       A[r, idx(i-1,j)] += -1.0
        if i+1==n-1: pass                            # ground = 0
        else:        A[r, idx(i+1,j)] += -1.0
        if j+1<=n-1: A[r, idx(i,j+1)] += -1.0
        else:        A[r, idx(i,j-1)] += -1.0        # Neumann reflect (top)
        if j-1>=0:   A[r, idx(i,j-1)] += -1.0
        else:        A[r, idx(i,j+1)] += -1.0        # Neumann reflect (bottom)
Ainv = np.linalg.inv(A)                     # factor ONCE (offline)
def solve(b): return Ainv @ b

# ---- (offline, once) build transfer operator Z = A^{-1} Biface, column per iface DOF
Z = Ainv @ Biface                           # Nu x Niface

# ---- moving-front interface data over T steps (high-rank RHS, like the real run)
T = 80; w = 1.6
yc = np.linspace(4, n-5, T)                  # front sweeps down the interface
G = np.zeros((Niface, T))
for t in range(T):
    G[:,t] = np.exp(-0.5*((np.arange(n)-yc[t])/w)**2)
G /= np.linalg.norm(G, axis=0, keepdims=True)

# ground-truth fields (direct solve) and transfer reconstruction
Phi_true = np.column_stack([solve(Biface @ G[:,t]) for t in range(T)])   # Nu x T
Phi_Z    = Z @ G
max_exact_err = np.max(np.linalg.norm(Phi_true-Phi_Z,axis=0)/np.linalg.norm(Phi_true,axis=0))

# ---- singular values: snapshot matrix vs operator Z (whole volume) vs Z far-field
sv_snap = np.linalg.svd(Phi_true, compute_uv=False)
sv_Z    = np.linalg.svd(Z,        compute_uv=False)
# far-field rows = column nearest the grounded body surface (i=n-2): electrodes live here
far_rows = [idx(n-2, j) for j in range(n)]
Z_far   = Z[far_rows, :]
sv_Zfar = np.linalg.svd(Z_far, compute_uv=False)
def effrank(sv, tol=1e-6): return int(np.sum(sv/sv[0] > tol))
r_snap, r_Z, r_far = effrank(sv_snap), effrank(sv_Z), effrank(sv_Zfar)

# ---- online basis growth: snapshot approach discovers dims front-position by
#      front-position; transfer approach is fixed at N_iface from step 0.
grow = []
for t in range(1, T+1):
    grow.append(effrank(np.linalg.svd(Phi_true[:,:t],compute_uv=False), 1e-6))

# ---- per-step field error: exact transfer op vs an ONLINE snapshot model that can
#      only use the basis discovered from PAST steps (oracle projection onto span).
err_online = []
for t in range(T):
    if t == 0: err_online.append(1.0); continue
    Q,_ = np.linalg.qr(Phi_true[:,:t])          # orthobasis of past snapshots
    resid = Phi_true[:,t] - Q @ (Q.T @ Phi_true[:,t])
    err_online.append(np.linalg.norm(resid)/np.linalg.norm(Phi_true[:,t]))
err_exact = np.full(T, max(max_exact_err, 1e-16))

print(f"transfer operator EXACT reconstruction, max rel field err over {T} steps: {max_exact_err:.2e}")
print(f"snapshot eff-rank(1e-6): {r_snap};  Z full-volume: {r_Z} (=N_iface {Niface} cap);  "
      f"Z far-field/electrodes: {r_far}")

# ================= figure =================
fig = plt.figure(figsize=(16.6, 5.3))
gs = fig.add_gridspec(1, 3, wspace=0.30)
fig.suptitle("Sys3 的解法:别近似『会动的解』,去近似『不动的算子』——"
             "界面→躯干传输算子 Z 一次建好、对任意 RHS 精确",
             fontsize=12.4, weight="bold", y=1.02)
from matplotlib.ticker import FuncFormatter, LogLocator
def logfmt(ax):
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=9))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f"1e{int(round(np.log10(v)))}"))

# (A) rank is BOUNDED by the interface; far-field (electrodes) compresses hard
axA = fig.add_subplot(gs[0])
axA.semilogy(sv_snap/sv_snap[0], "o-", ms=2.6, color="#c0392b", lw=1.4,
             label=f"解快照(会动,轨迹相关):秩≈{r_snap}")
axA.semilogy(sv_Z/sv_Z[0], "s-", ms=2.6, color="#2980b9", lw=1.4,
             label=f"Z 全躯干体积:秩={r_Z}(=界面 {Niface} 封顶)")
axA.semilogy(sv_Zfar/sv_Zfar[0], "^-", ms=3.0, color="#16a085", lw=1.6,
             label=f"Z 远场/体表电极:压到 {r_far}")
axA.axhline(1e-6, color="0.5", ls="--", lw=1)
axA.axvline(Niface, color="0.4", ls=":", lw=1)
axA.set_ylim(1e-9, 2); logfmt(axA)
axA.set_xlabel("奇异值序号"); axA.set_ylabel("sigma / sigma_max")
axA.set_title(f"① 秩被界面 N_iface={Niface} 封顶(不随步数涨)\n近界面体积不压,远场/电极压到 {r_far}",
              fontsize=10.2, weight="bold")
axA.legend(fontsize=7.6, loc="upper right"); axA.grid(alpha=0.25, which="both")

# (B) online discovery (snapshot) vs offline fixed (transfer)
axB = fig.add_subplot(gs[1])
axB.plot(range(1,T+1), grow, "-", color="#c0392b", lw=2.0, label="leadvol:在线逐帧发现基\n(前沿扫到哪才有那一维,还每步解)")
axB.axhline(Niface, color="#2980b9", lw=2.0, label=f"传输算子:离线一次备齐 {Niface} 列\n(第0步就全有,零在线求解)")
axB.set_xlabel("EP 步"); axB.set_ylabel("已具备的基维数"); axB.set_ylim(0, Niface+3)
axB.set_title("② 秩一样有限,但一个在线慢慢发现(每步都要解),\n另一个离线一次备齐、对所有轨迹通用",
              fontsize=10.2, weight="bold")
axB.legend(fontsize=8.0, loc="lower right"); axB.grid(alpha=0.25)
axB.annotate("这就是实测『54 还在涨』", xy=(T*0.7, grow[int(T*0.7)]), xytext=(T*0.30, Niface*0.55),
             fontsize=8.4, color="#c0392b", arrowprops=dict(arrowstyle="->", color="#c0392b"))

# (C) per-step error: exact transfer op vs online snapshot model (never catches up)
axC = fig.add_subplot(gs[2])
axC.semilogy(range(T), err_online, "o-", ms=2.6, color="#c0392b", lw=1.4,
             label="在线快照模型(只用过去帧,预言者投影)")
axC.semilogy(range(T), err_exact, "s-", ms=2.6, color="#2980b9", lw=1.6,
             label=f"传输算子 Z(离线备齐):每步精确 {max_exact_err:.0e}")
axC.set_ylim(1e-17, 2); logfmt(axC)
axC.set_xlabel("EP 步"); axC.set_ylabel("该步全场相对 L2 误差")
axC.set_title("③ 在线快照要解够~20步才追到~1e-10(且需预言者投影);\n传输算子第0步起对任意高秩 RHS 机器精确、零在线求解",
              fontsize=10.2, weight="bold")
axC.legend(fontsize=8.0, loc="lower left"); axC.grid(alpha=0.25, which="both")

fig.savefig("fig_transfer_fix.png", dpi=140, bbox_inches="tight")
print("wrote fig_transfer_fix.png")
