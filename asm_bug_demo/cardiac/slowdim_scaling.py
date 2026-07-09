#!/usr/bin/env python3
"""slowdim_scaling.py -- how does the SLOW-SUBSPACE dimension (the '12') scale?

The '12' in -decay is the slow tail of the sASM(domain-decomposition)-PRECONDITIONED
operator: #eigenvalues < 0.1*lambda_max of M^{-1}A.  A DD preconditioner already
handles modes that are smooth *within* a subdomain; what stays slow is modes smooth
*across the whole domain*, whose count is set by GEOMETRY + #SUBDOMAINS, not fine N.
(Plain point-Jacobi would instead count ALL smooth modes ~ N -- the wrong proxy.)

Here M^{-1} = block-Jacobi with Dirichlet (principal-submatrix) blocks over a Py x Px
subdomain partition -- a cheap faithful stand-in for sASM.  A = 5-point Neumann
Laplacian (singular, like Sys2).  We count slow modes of M^{-1}A (excluding the
constant nullspace).  Sweeps isolate each factor.  Output: fig_slowdim_scaling.png
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

def neumann_laplacian(mask, sigma=None):
    """5-point Neumann Laplacian. sigma(j,i) optional per-cell conductivity (harmonic
    average on edges) -> lets us test high-contrast (scar/anisotropy) jumps."""
    ny, nx = mask.shape
    idx = -np.ones((ny,nx), int); dofs = np.argwhere(mask)
    for k,(j,i) in enumerate(dofs): idx[j,i]=k
    m=len(dofs); A=np.zeros((m,m)); cell=np.zeros((m,2),int)
    def sig(j,i): return 1.0 if sigma is None else sigma[j,i]
    for k,(j,i) in enumerate(dofs):
        cell[k]=(j,i)
        for dj,di in ((1,0),(-1,0),(0,1),(0,-1)):
            jj,ii=j+dj,i+di
            if 0<=jj<ny and 0<=ii<nx and mask[jj,ii]:
                se=2.0*sig(j,i)*sig(jj,ii)/(sig(j,i)+sig(jj,ii))   # harmonic edge weight
                A[k,k]+=se; A[k,idx[jj,ii]]-=se
    return A, cell

def block_jacobi_slow(A, cell, Py, Px, frac=0.1):
    """M = block diag of Dirichlet (principal-submatrix) blocks over Py x Px grid of
    subdomains (by cell bounding box). Count #(lambda(M^-1 A) < frac*lambda_max)."""
    m=len(cell); jmin,imin=cell.min(0); jmax,imax=cell.max(0)
    by=np.clip(((cell[:,0]-jmin)/(jmax-jmin+1e-9)*Py).astype(int),0,Py-1)
    bx=np.clip(((cell[:,1]-imin)/(imax-imin+1e-9)*Px).astype(int),0,Px-1)
    lab=by*Px+bx
    Minv=np.zeros((m,m))
    for b in np.unique(lab):
        S=np.where(lab==b)[0]
        Ab=A[np.ix_(S,S)].copy(); Ab+=1e-10*np.eye(len(S))   # Dirichlet block is PD
        Minv[np.ix_(S,S)]=np.linalg.inv(Ab)
    w=np.linalg.eigvals(Minv@A).real
    w=np.sort(w); w=w[w>1e-6*w.max()]                        # drop constant nullspace
    return int((w<frac*w.max()).sum()), w.max()/w.min(), len(np.unique(lab))

def rect(ny,nx): m=np.ones((ny,nx),bool); return m
def Lshape(ny,nx):
    m=np.zeros((ny,nx),bool); m[:,:nx//2]=True; m[ny//2:,:]=True; return m
def Stwist(ny,nx):
    m=np.zeros((ny,nx),bool)
    for j in range(ny):
        c=int((nx*0.5)+(nx*0.30)*np.sin(2*np.pi*j/ny)); w=max(2,nx//6)
        m[j,max(0,c-w):min(nx,c+w)]=True
    return m
def cross(ny,nx):
    m=np.zeros((ny,nx),bool); m[ny//3:2*ny//3,:]=True; m[:,nx//3:2*nx//3]=True; return m

print("="*70)
print("(1) MESH REFINEMENT, fixed 4:1 shape, FIXED 2x8=16 subdomains")
r1=[]
for f in (1,2,3,4):
    A,cell=neumann_laplacian(rect(8*f,32*f))
    ns,cond,nb=block_jacobi_slow(A,cell,2,8)
    r1.append((8*f*32*f,ns)); print(f"   N={8*f*32*f:5d}  subdomains={nb}  slow={ns:3d}  cond={cond:7.0f}")
print("   -> FIXED #subdomains: slow count ~FLAT as N grows (refine-invariant)")

print("="*70)
print("(1b) FIXED subdomain SIZE (~8x8), refine => #subdomains grows (strong scaling)")
r1b=[]
for f in (1,2,3,4):
    ny,nx=8*f,32*f; A,cell=neumann_laplacian(rect(ny,nx))
    ns,cond,nb=block_jacobi_slow(A,cell,f,4*f)
    r1b.append((nb,ns)); print(f"   N={ny*nx:5d}  subdomains={nb:2d}  slow={ns:3d}  cond={cond:7.0f}")
print("   -> slow count GROWS ~ #subdomains (this is one-level Schwarz's C0 growth)")

print("="*70)
print("(2) ASPECT RATIO, fixed #subdomains(=16), ~fixed N")
r2=[]
for ratio in (1,2,4,8,16):
    nx=int(round(np.sqrt(1024*ratio))); ny=max(6,int(round(nx/ratio)))
    A,cell=neumann_laplacian(rect(ny,nx)); ns,cond,nb=block_jacobi_slow(A,cell,2,8)
    r2.append((ratio,ns)); print(f"   {ny:3d}x{nx:3d} ratio~{ratio:2d}:1 subdomains={nb} slow={ns:3d} cond={cond:7.0f}")

def dumbbell(ny,nx):
    """two blobs joined by a thin neck -> a near-disconnected connection (like a
    thin cardiac wall / isthmus): classically ONE extra very-slow mode per weak link."""
    m=np.zeros((ny,nx),bool)
    m[:, :nx//2-nx//8]=True; m[:, nx//2+nx//8:]=True      # two blobs
    m[ny//2-1:ny//2+1, :]=True                            # thin neck (2 cells tall)
    return m

print("="*70)
print("(3) GEOMETRIC COMPLEXITY & PHYSICS (fixed ~16 subdomains)")
r3=[]
# regular shape detail (adds little at fixed P)
for nm,mk in [("规则slab",rect(16,64)),("L形",Lshape(40,40)),("S扭曲",Stwist(48,40))]:
    A,cell=neumann_laplacian(mk); ns,cond,nb=block_jacobi_slow(A,cell,4,4)
    r3.append((nm,ns,"#2980b9")); print(f"   {nm:16s} N={int(mk.sum()):5d} sub={nb} slow={ns:3d} cond={cond:7.0f}")
# bottleneck / near-disconnected (thin wall) -- adds hard slow modes
A,cell=neumann_laplacian(dumbbell(40,64)); ns,cond,nb=block_jacobi_slow(A,cell,4,4)
r3.append(("细颈(薄壁/峡部)",ns,"#c0392b")); print(f"   {'细颈(薄壁/峡部)':16s} N={A.shape[0]:5d} sub={nb} slow={ns:3d} cond={cond:7.0f}")
# high-contrast conductivity (scar / anisotropy jump) -- adds hard slow modes
ny,nx=32,64; sg=np.ones((ny,nx)); sg[:, nx//2-2:nx//2+2]=1e-3      # low-conductivity strip
A,cell=neumann_laplacian(rect(ny,nx),sigma=sg); ns,cond,nb=block_jacobi_slow(A,cell,4,4)
r3.append(("高对比(疤痕/跳变)",ns,"#8e44ad")); print(f"   {'高对比(疤痕/跳变)':16s} N={ny*nx:5d} sub={nb} slow={ns:3d} cond={cond:7.0f}")
print("   -> shape detail adds little; BUT thin necks & coeff jumps add EXTRA hard")
print("      slow modes a per-subdomain (Nicolaides) coarse space MISSES -> need GenEO")

# ================= figure =================
fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
fig.suptitle("『慢模维数(那个 12)』如何 scale —— 块-Jacobi(区域分解)预条件后 #(λ<0.1λmax),纯 Neumann 椭圆算子",
             fontsize=12.6, weight="bold", y=1.03)

ax=axes[0]; x=[a for a,_ in r1]; y=[b for _,b in r1]
ax.plot(x,y,"-o",color="#16a085",lw=2.4,ms=8); ax.set_ylim(0,max(y)+6)
ax.set_xlabel("自由度 N"); ax.set_ylabel("慢模个数")
ax.set_title("① 同几何+同子域数,加密网格:\n慢模数缓增并饱和~O(子域数),不 ∝N",fontsize=10,weight="bold")
ax.grid(alpha=0.25); ax.text(0.04,0.08,"→ 远慢于纯Jacobi的∝N\n(那样会76→290)",transform=ax.transAxes,color="#16a085",fontsize=8.5)

ax=axes[1]; x=[a for a,_ in r1b]; y=[b for _,b in r1b]
ax.plot(x,y,"-o",color="#c0392b",lw=2.4,ms=8); ax.set_ylim(0,max(y)+4)
ax.set_xlabel("子域个数(加核)"); ax.set_ylabel("慢模个数")
ax.set_title("② 固定子域大小、加核:\n慢模数~随子域数增长(需粗空间)",fontsize=10,weight="bold")
ax.grid(alpha=0.25); ax.text(0.05,0.85,"→ 强扩展的痛点",transform=ax.transAxes,color="#c0392b",fontsize=9)

ax=axes[2]; x=[a for a,_ in r2]; y=[b for _,b in r2]
ax.plot(x,y,"-o",color="#e67e22",lw=2.4,ms=8); ax.set_ylim(0,max(y)+4)
ax.set_xlabel("长宽比"); ax.set_ylabel("慢模个数")
ax.set_title("③ 拉长几何(固定子域数):\n慢模数随最长轴略增",fontsize=10,weight="bold"); ax.grid(alpha=0.25)

ax=axes[3]; names=[a for a,_,_ in r3]; y=[b for _,b,_ in r3]; bcol=[c for _,_,c in r3]
ax.bar(range(len(names)),y,color=bcol,edgecolor="0.3")
for i,v in enumerate(y): ax.text(i,v+0.3,str(v),ha="center",fontsize=10.5,weight="bold")
ax.set_xticks(range(len(names))); ax.set_xticklabels(names,fontsize=7.5,rotation=20)
ax.set_ylabel("慢模个数"); ax.set_ylim(0,max(y)+5)
ax.set_title("④ 同子域数:规则形状加得少(蓝);\n细颈/高对比(红紫,真实心脏)加硬慢模",fontsize=9.6,weight="bold"); ax.grid(axis="y",alpha=0.25)

fig.savefig("fig_slowdim_scaling.png", dpi=135, bbox_inches="tight")
print("\nwrote fig_slowdim_scaling.png")
