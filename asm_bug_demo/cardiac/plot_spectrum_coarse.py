#!/usr/bin/env python3
"""plot_spectrum_coarse.py -- one concrete example tying together spectrum,
spectral distribution, CG convergence, and coarse-space deflation.

Matrix: 1D Poisson  A = tridiag(-1, 2, -1), size n=40.
Closed form: eigenvalue  lambda_k = 2(1-cos(k*pi/(n+1))),  k=1..n
             eigenvector  v_k[i]  = sin(i*k*pi/(n+1))  (the k-th sine 'mode').
The small-k modes are smooth/long-wavelength (small lambda = slow);
the big-k modes are jagged/short-wavelength (big lambda = fast).

Panels:
 (A) spectrum: the 40 eigenvalues on a line + the smooth/jagged mode shapes.
 (B) spectral distribution: histogram -> most lambda are big, a thin tail is small.
 (C) CG convergence, plain vs deflated (coarse space = 4 smoothest modes).
 (D) the deflated spectrum: the 4 small eigenvalues are GONE -> kappa collapses.
Output: fig_spectrum_coarse.png
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

n = 40
A = np.diag(2.0*np.ones(n)) + np.diag(-np.ones(n-1),1) + np.diag(-np.ones(n-1),-1)
lam, V = np.linalg.eigh(A)          # ascending; V columns = sine modes
kappa = lam[-1]/lam[0]

# ---------- Preconditioned CG that also returns per-mode error ----------
def cg(A, b, x0, M=None, tol=1e-12, kmax=200, track=None):
    """M: linear operator (function) applying preconditioner; track: eigvec matrix
    to record per-mode error coefficients of the true error each step."""
    x = x0.copy(); r = b - A@x
    z = r if M is None else M(r)
    p = z.copy(); rz = r@z
    xstar = np.linalg.solve(A, b)
    hist = [np.linalg.norm(r)]
    modes = [np.abs(track.T@(x-xstar))] if track is not None else None
    for _ in range(kmax):
        Ap = A@p; a = rz/(p@Ap)
        x = x + a*p; r = r - a*Ap
        hist.append(np.linalg.norm(r))
        if modes is not None: modes.append(np.abs(track.T@(x-xstar)))
        if hist[-1] < tol*hist[0]: break
        z = r if M is None else M(r); rz2 = r@z
        p = z + (rz2/rz)*p; rz = rz2
    return x, hist, (np.array(modes) if modes is not None else None)

rng_b = np.sin(np.outer(np.arange(1,n+1), np.arange(1,6))*np.pi/(n+1)).sum(1)  # mix of modes
b = rng_b / np.linalg.norm(rng_b)
x0 = np.zeros(n)

# ---------- Coarse space W = the m smoothest eigenvectors ----------
m = 4
W = V[:, :m]                         # 4 slowest modes (smallest lambda)
E = W.T@A@W                          # coarse operator (m x m), here diag(lam[:m])
Einv = np.linalg.inv(E)
def deflate(r):                      # additive coarse correction  W E^-1 W^T r
    return W@(Einv@(W.T@r))
def M_defl(r):                       # (crude) fine = identity  +  coarse correction
    return r + deflate(r)

_, h_plain, modes_plain = cg(A, b, x0, M=None,     track=V)
_, h_defl , _           = cg(A, b, x0, M=M_defl,   track=V)

# deflated spectrum: eigenvalues of  (I - W E^-1 W^T A) A  restricted off W  == lam[m:]
lam_defl = lam[m:]                   # the 4 smallest are annihilated
kappa_defl = lam_defl[-1]/lam_defl[0]

# =================== figure ===================
fig = plt.figure(figsize=(15.5, 10.5))
gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.22)
fig.suptitle("一个例子:1D 泊松矩阵 A=tridiag(-1,2,-1), n=40 —— 谱 → 谱分布 → CG → 粗空间",
             fontsize=15, weight="bold", y=0.98)

# (A) spectrum on a line + two mode shapes
axA = fig.add_subplot(gs[0,0])
axA.plot(lam, np.zeros_like(lam), "o", color="#2980b9", ms=6)
axA.plot(lam[:m], np.zeros(m), "o", color="#c0392b", ms=9, label=f"{m} 个最小 λ = 慢模")
axA.set_yticks([])
axA.set_xlabel("特征值 λ (谱)")
axA.set_title(f"① 谱:40 个 λ 从 {lam[0]:.4f} 到 {lam[-1]:.3f},κ=λmax/λmin={kappa:.0f}",
              fontsize=11, weight="bold")
axA.annotate("最小 λ=%.4f\n(最平滑/最长波长模)"%lam[0], xy=(lam[0],0), xytext=(0.5,0.6),
             fontsize=9, color="#c0392b", arrowprops=dict(arrowstyle="->",color="#c0392b"))
axA.annotate("最大 λ=%.2f\n(最锯齿/最短波长模)"%lam[-1], xy=(lam[-1],0), xytext=(2.3,0.6),
             fontsize=9, color="#2980b9", arrowprops=dict(arrowstyle="->",color="#2980b9"))
axA.set_ylim(-0.4,1.0)
# inset: mode shapes
axi = axA.inset_axes([0.06,0.06,0.5,0.32])
xx=np.arange(1,n+1)
axi.plot(xx, V[:,0], color="#c0392b", lw=1.6, label="慢模 v1")
axi.plot(xx, V[:,-1], color="#2980b9", lw=0.9, label="快模 v40")
axi.set_xticks([]); axi.set_yticks([]); axi.legend(fontsize=6.5, loc="upper right")
axi.set_title("模的形状",fontsize=7.5)
axA.legend(fontsize=8.5, loc="upper right")

# (B) spectral distribution histogram
axB = fig.add_subplot(gs[0,1])
axB.hist(lam, bins=16, color="#7fb3d5", edgecolor="w")
axB.axvspan(lam[0]-0.01, lam[m-1]+0.01, color="#f5b7b1", alpha=0.6)
axB.text(lam[m-1]+0.05, axB.get_ylim()[1]*0.7,
         "细长的低端尾巴\n= 少数几个慢模\n(卡 CG 的就是它们)", fontsize=9, color="#c0392b")
axB.set_xlabel("特征值 λ"); axB.set_ylabel("有几个 λ 落在这")
axB.set_title("② 谱分布:大部分 λ 挤在高端(好解),\n只有一条稀薄的低端尾巴(难解)",
              fontsize=11, weight="bold")

# (C) CG convergence plain vs deflated
axC = fig.add_subplot(gs[1,0])
axC.semilogy(h_plain/h_plain[0], color="#c0392b", lw=2.2, label=f"普通 CG ({len(h_plain)-1} 步)")
axC.semilogy(h_defl /h_defl[0],  color="#27ae60", lw=2.2, label=f"粗空间 deflated CG ({len(h_defl)-1} 步)")
axC.axhline(1e-8, color="0.6", ls=":", lw=1)
axC.set_xlabel("CG 迭代步 k"); axC.set_ylabel("相对残差 (log)")
axC.set_title("③ CG:慢模拖长尾巴;把 4 个慢模投影掉 → 尾巴消失、步数暴跌",
              fontsize=11, weight="bold")
axC.legend(fontsize=10); axC.grid(alpha=0.25); axC.set_ylim(1e-13,2)
import matplotlib.ticker as mt
axC.yaxis.set_major_formatter(mt.FuncFormatter(lambda v,_: f"1e{int(round(np.log10(v)))}" if v>0 else ""))

# (D) deflated spectrum
axD = fig.add_subplot(gs[1,1])
axD.plot(lam, 1+np.zeros_like(lam), "o", color="#bbb", ms=6)
axD.plot(lam[:m], 1+np.zeros(m), "x", color="#c0392b", ms=11, mew=2.5, label="被粗空间铲掉")
axD.plot(lam_defl, 0+np.zeros_like(lam_defl), "o", color="#27ae60", ms=6)
axD.set_yticks([0,1]); axD.set_yticklabels(["deflated 后\n有效谱","原始谱"])
axD.set_xlabel("特征值 λ")
axD.set_title("④ 粗空间做了什么:把 4 个最小 λ 从有效谱里\n拿掉 → κ 从 %.0f 掉到 %.0f"%(kappa,kappa_defl),
              fontsize=11, weight="bold")
axD.annotate("新的最小有效 λ=%.3f"%lam_defl[0], xy=(lam_defl[0],0), xytext=(0.6,0.45),
             fontsize=9, color="#27ae60", arrowprops=dict(arrowstyle="->",color="#27ae60"))
axD.legend(fontsize=9, loc="upper right"); axD.set_ylim(-0.4,1.4)

fig.savefig("fig_spectrum_coarse.png", dpi=135, bbox_inches="tight")
print("wrote fig_spectrum_coarse.png")
print(f"n={n}  kappa={kappa:.1f}  plain CG={len(h_plain)-1} steps")
print(f"coarse m={m}  kappa_defl={kappa_defl:.1f}  deflated CG={len(h_defl)-1} steps")
print(f"lambda smallest 6: {np.round(lam[:6],4)}")
