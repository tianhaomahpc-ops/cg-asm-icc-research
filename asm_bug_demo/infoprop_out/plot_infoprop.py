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

# ---------- Fig 5: overlap sweep (2D ICC) -- seam localization widens with O ----------
from matplotlib.colors import LogNorm
Ovs=[1,2,4]; iterB={1:182,2:195,4:185}; iterS={1:131,2:130,4:129}
allf=[np.abs(np.loadtxt(f"ip2d_pcg_ovl{m}_O{O}_k40.txt"))+1e-12
      for m in ["B","S"] for O in Ovs]
vmax=max(f.max() for f in allf); vmin=vmax*1e-4
fig,axes=plt.subplots(2,len(Ovs),figsize=(2.3*len(Ovs)+1,4.8))
for col,O in enumerate(Ovs):
    for row,(m,lab,it) in enumerate([("B","BASIC",iterB),("S","sASM",iterS)]):
        f=np.abs(np.loadtxt(f"ip2d_pcg_ovl{m}_O{O}_k40.txt"))+1e-12
        im=axes[row,col].imshow(f,origin="lower",cmap="magma",norm=LogNorm(vmin=vmin,vmax=vmax))
        axes[row,col].set_xticks([]); axes[row,col].set_yticks([])
        if row==0: axes[row,col].set_title(f"overlap $O={O}$  ({iterB[O]} it)",fontsize=9,color=ACC)
axes[0,0].set_ylabel("BASIC",fontsize=11,color=ACC,fontweight="bold")
axes[1,0].set_ylabel("sASM",fontsize=11,color=STRUCT,fontweight="bold")
for col,O in enumerate(Ovs): axes[1,col].set_xlabel(f"sASM {iterS[O]} it",fontsize=8,color=STRUCT)
fig.colorbar(im,ax=axes,fraction=0.012,pad=0.01,label="$|r|$ (log)")
fig.suptitle("Fig. 5  —  Overlap sweep ($128^2$, $8{\\times}8$, ICC(0), residual at fixed $k{=}40$): BASIC's residual sits in the\n"
   "over-counted seams; the over-counted region (and the residual share trapped there, $0.45\\!\\to\\!0.51\\!\\to\\!0.64$) widens with $O$; sASM clears it at every $O$",
   color=STRUCT,fontweight="bold",fontsize=9)
fig.savefig("fig5_overlap_sweep.png",bbox_inches="tight"); plt.close(fig)

# ---------- Fig 6: 3D Laplace (real Sys3) residual + overlap anomaly ----------
def relhist(fn):
    d=np.loadtxt(fn); it=d[:,0]; r=d[:,1]; return it, r/r[0]
fig,(ax,ax2)=plt.subplots(1,2,figsize=(11,4.4),width_ratios=[1.25,1])
for fn,c,ls,lab in [("res3d_BASIC_chol.txt",ACC,"-","BASIC, exact Cholesky (38 it)"),
                    ("res3d_BASIC_icc.txt", ACC,"--","BASIC, ICC(0) (214 it)"),
                    ("res3d_sASM_icc.txt",  STRUCT,"--","sASM, ICC(0) (134 it)")]:
    it,r=relhist(fn); ax.semilogy(it,r,ls,color=c,lw=1.9,label=lab)
ax.set_xlabel("CG iteration"); ax.set_ylabel(r"relative residual $\|r\|/\|r_0\|$")
ax.grid(True,which="both",alpha=.25); ax.legend(fontsize=8.5,loc="upper right")
ax.set_title("3D Laplace, Sys 3 ($n_x{=}48$, $O{=}2$): with ICC, BASIC 214 $\\to$ sASM 134;\n"
             "exact Cholesky is fast (38) $=$ no anomaly",fontsize=9.5)
# right: overlap anomaly (canonical 3D data, rtol 1e-6)
O=[0,1,2]; B=[132,148,179]; S=[132,104,103]
ax2.plot(O,B,"o-",color=ACC,lw=2,label="BASIC + ICC(0)")
ax2.plot(O,S,"s-",color=STRUCT,lw=2,label="sASM + ICC(0)")
ax2.set_xlabel("overlap $O$"); ax2.set_ylabel("CG iterations"); ax2.set_xticks(O)
ax2.grid(True,alpha=.25); ax2.legend(fontsize=9)
ax2.annotate("more overlap\n$\\Rightarrow$ MORE iters\n(error clears slower)",xy=(2,179),xytext=(0.4,168),
             fontsize=8.5,color=ACC,arrowprops=dict(arrowstyle="-|>",color=ACC))
ax2.set_title("3D: overlap $\\uparrow$ makes BASIC+ICC worse (the anomaly);\nsASM reverses it",fontsize=9.5)
fig.suptitle("Fig. 6  —  The same effect on the real 3D Sys 3 (mixed Dirichlet/Neumann Laplace)",
             color=STRUCT,fontweight="bold",fontsize=10.5)
fig.tight_layout(rect=[0,0,1,0.93]); fig.savefig("fig6_3d.png"); plt.close(fig)
print("wrote fig5_overlap_sweep.png fig6_3d.png")
