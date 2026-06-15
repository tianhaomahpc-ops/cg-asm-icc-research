#!/usr/bin/env python3
"""Plot the information-propagation / over-counting figures from infoprop.c dumps."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

STRUCT="#1F3A5F"; ACC="#8C2D04"; GREEN="#2E6B34"
plt.rcParams.update({"font.size":10,"axes.titlesize":11,"figure.dpi":130})

def seams(ax, nsub=8):
    for i in range(1,nsub):
        ax.axvline(i/nsub, color="0.6", lw=0.6, ls=":")

# ---------- Fig 1: single-apply over-count (factor B made visible) ----------
fig,axes=plt.subplots(2,2,figsize=(10,5.2),height_ratios=[1,2.2])
for col,O in enumerate([2,4]):
    d=np.loadtxt(f"ip1d_apply_O{O}.txt")  # x mult r zB zS
    x,mult,r,zB,zS=d.T
    ax0=axes[0,col]; ax0.fill_between(x,mult,1,where=mult>=2,color=ACC,alpha=.18,step="mid")
    ax0.plot(x,mult,color=STRUCT,lw=1.2,drawstyle="steps-mid")
    ax0.set_ylabel("multiplicity\n$m_k$"); ax0.set_ylim(0.5,2.5); ax0.set_yticks([1,2])
    ax0.set_title(f"overlap $O={O}$  —  shaded = over-counted region ($m_k{{=}}2$)")
    seams(ax0); ax0.set_xticklabels([])
    ax1=axes[1,col]
    ax1.fill_between(x,0,np.maximum(zB,zS),where=mult>=2,color=ACC,alpha=.12,step="mid")
    ax1.plot(x,r/ r.max()*zB.max(),color="0.6",lw=1.0,label="input residual $r$ (scaled)")
    ax1.plot(x,zB,color=ACC,lw=1.8,label=r"BASIC  $\sum_i R_i^TA_i^{-1}R_i\,r$")
    ax1.plot(x,zS,color=STRUCT,lw=1.8,label=r"sASM  $D^{-1/2}(\cdots)D^{-1/2}r$")
    seams(ax1); ax1.set_xlabel("$x$")
    if col==0: ax1.set_ylabel("correction  $z(x)$")
    ax1.legend(fontsize=8,loc="upper right",framealpha=.9)
fig.suptitle("Fig. 1  —  One preconditioner apply: BASIC over-counts the overlap; sASM normalizes it",
             color=STRUCT,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig("fig1_overcount.png"); plt.close(fig)

# ---------- Fig 2: information PROPAGATION (front advances ~1 subdomain/iter) ----------
def read_front(fn):
    blocks={}; k=None
    for line in open(fn):
        line=line.strip()
        if line.startswith("# k="): k=int(line[4:]); blocks[k]=[]
        elif line and k is not None:
            xv,vv=line.split(); blocks[k].append((float(xv),float(vv)))
    return {k:np.array(v) for k,v in blocks.items() if len(v)}

# additive Schwarz with EXACT solves is a clean contraction here -> use it to
# illustrate propagation (method-agnostic: both BASIC/sASM restrict to subdomains).
B=read_front("ip1d_front_O2_exact_BASIC.txt")
fig,ax=plt.subplots(figsize=(9,4.2))
ks=[0,1,2,4,8,16]; cmap=plt.cm.viridis(np.linspace(0,0.88,len(ks)))
for c,k in zip(cmap,ks):
    a=B[k]; ax.plot(a[:,0],a[:,1],color=c,lw=1.9,label=f"$k={k}$")
seams(ax)
ax.plot(B[16][:,0],1-B[16][:,0],color="0.5",ls="--",lw=1.0,label="exact $x^\\star=1-x$")
ax.annotate("", xy=(0.42,0.55), xytext=(0.08,0.55),
            arrowprops=dict(arrowstyle="-|>",color=ACC,lw=2))
ax.text(0.16,0.60,"information front",color=ACC,fontsize=9)
ax.set_xlabel("$x$"); ax.set_ylabel("$x_k(x)$  (left Dirichlet $=1$)")
ax.set_ylim(-0.05,1.05); ax.legend(fontsize=8.5,ncol=2,loc="upper right")
ax.set_title("Each iteration the left boundary's influence reaches $\\sim$1 subdomain further\n"
             "(fast at first $=$ overlap reach; the deep interior lags $=$ the slow global mode a coarse space would fix)",
             fontsize=9.5)
fig.suptitle("Fig. 2  —  Information propagation in one-level Schwarz: a bounded distance per iteration "
             "$\\Rightarrow$ $O(\\#\\mathrm{subdomains})$ iters",color=STRUCT,fontweight="bold",fontsize=10.5)
fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig("fig2_front.png"); plt.close(fig)

# ---------- Fig 3: 2D residual history (the A x B payoff) ----------
fig,ax=plt.subplots(figsize=(7.2,4.6))
styles={"exact":("-","exact subdomain solve"),"icc":("--","ICC(0) subdomain solve")}
for tag,(ls,lab) in styles.items():
    d=np.loadtxt(f"ip2d_reshist_{tag}.txt")  # iter B S
    it,B,S=d.T
    bm=B>0; sm=S>0
    ax.semilogy(it[bm],B[bm],ls,color=ACC,lw=1.8,
                label=f"BASIC, {lab} ({int(it[bm][-1])} it)")
    ax.semilogy(it[sm],S[sm],ls,color=STRUCT,lw=1.8,
                label=f"sASM,  {lab} ({int(it[sm][-1])} it)")
ax.set_xlabel("CG iteration"); ax.set_ylabel(r"relative residual $\|r\|/\|b\|$")
ax.set_title("Fig. 3  —  2D Laplace: sASM beats BASIC ONLY with an inexact solve\n"
             "(exact: 23 vs 27; ICC(0): 67 vs 102) — over-counting bites only when coupled with inexactness",
             color=STRUCT,fontweight="bold",fontsize=10)
ax.grid(True,which="both",alpha=.25); ax.legend(fontsize=8.5)
fig.tight_layout(); fig.savefig("fig3_reshist2d.png"); plt.close(fig)

# ---------- Fig 4: 2D ICC residual fields at MATCHED CG iters (the seam contrast) ----------
from matplotlib.colors import LogNorm
mult2=np.loadtxt("ip2d_mult.txt")
ks=[5,15,30,60]
fig,axes=plt.subplots(2,len(ks)+1,figsize=(2.0*(len(ks)+1),4.6))
for row in range(2):
    axes[row,0].imshow(mult2,origin="lower",cmap="Greys",vmin=1,vmax=mult2.max())
    axes[row,0].set_xticks([]); axes[row,0].set_yticks([])
axes[0,0].set_title("multiplicity $m_k$\n(4$\\times$4 subdomains)",fontsize=8)
axes[0,0].set_ylabel("BASIC (102 it)",fontsize=10,color=ACC,fontweight="bold")
axes[1,0].set_ylabel("sASM (67 it)",fontsize=10,color=STRUCT,fontweight="bold")
# common log color scale across all panels
allf=[np.abs(np.loadtxt(f"ip2d_pcg_{m}_icc_k{k}.txt"))+1e-12
      for m in ["BASIC","sASM"] for k in ks]
vmax=max(f.max() for f in allf); vmin=vmax*1e-5
for row,meth in enumerate(["BASIC","sASM"]):
    for col,k in enumerate(ks,start=1):
        f=np.abs(np.loadtxt(f"ip2d_pcg_{meth}_icc_k{k}.txt"))+1e-12
        im=axes[row,col].imshow(f,origin="lower",cmap="magma",
                                norm=LogNorm(vmin=vmin,vmax=vmax))
        axes[row,col].set_xticks([]); axes[row,col].set_yticks([])
        if row==0: axes[row,col].set_title(f"iter $k={k}$",fontsize=9)
fig.colorbar(im,ax=axes,fraction=0.012,pad=0.01,label="$|r|$ (log)")
fig.suptitle("Fig. 4  —  ICC(0) subdomain solve: residual $|r|$ at MATCHED CG iterations. "
             "BASIC's residual lingers at the overlap seams; sASM clears it $\\Rightarrow$ 67 vs 102 iters",
             color=STRUCT,fontweight="bold",fontsize=10)
fig.savefig("fig4_heatmap2d.png",bbox_inches="tight"); plt.close(fig)

print("wrote fig1_overcount.png fig2_front.png fig3_reshist2d.png fig4_heatmap2d.png")
