#!/usr/bin/env python3
"""plot_geom_mesh.py -- geometry domains + real unstructured tet mesh from the
Gmsh MSH 2.2 files (heart.msh, torso.msh).  Produces:
  fig_geom.png     : the two conforming domains (heart slab in torso), schematic
  fig_meshview.png : the ACTUAL mesh -- heart boundary-surface triangulation (3D)
                     + a true y=0 plane-slice of the tetrahedra (heart fine,
                     torso coarse), showing the multiscale conforming tet mesh.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

def parse_msh(path):
    nodes=None; tris=[]; tets=[]
    with open(path) as f:
        it=iter(f)
        for line in it:
            line=line.strip()
            if line=="$Nodes":
                n=int(next(it)); nodes=np.zeros((n+1,3))
                for _ in range(n):
                    p=next(it).split(); i=int(p[0])
                    nodes[i]=[float(p[1]),float(p[2]),float(p[3])]
            elif line=="$Elements":
                m=int(next(it))
                for _ in range(m):
                    p=next(it).split(); et=int(p[1]); nt=int(p[2])
                    phys=int(p[3]); ids=list(map(int,p[3+nt:]))
                    if et==2:  tris.append((ids[0],ids[1],ids[2],phys))
                    elif et==4:tets.append((ids[0],ids[1],ids[2],ids[3],phys))
    return nodes, np.array(tris), np.array(tets)

def slice_tets(nodes, tets, axis=1, level=0.0):
    """true planar cross-section of the tet mesh: return closed polygon outlines
    (as (x,z) segment lists) of each tet cut by the plane axis=level."""
    E=[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    keep=[a for a in (0,1,2) if a!=axis]
    segs=[]
    P=nodes[tets[:,:4].astype(int)]                 # (Ntet,4,3)
    D=P[:,:,axis]-level
    for t in range(P.shape[0]):
        d=D[t]; pts=P[t]
        ip=[]
        for a,b in E:
            if d[a]*d[b]<0:
                w=d[a]/(d[a]-d[b]); ip.append(pts[a]+w*(pts[b]-pts[a]))
        if len(ip)>=3:
            ip=np.array(ip)[:,keep]                  # project to (x,z)
            c=ip.mean(0); ang=np.arctan2(ip[:,1]-c[1], ip[:,0]-c[0])
            ip=ip[np.argsort(ang)]
            for k in range(len(ip)):
                segs.append([ip[k], ip[(k+1)%len(ip)]])
    return segs

hn,ht,htet = parse_msh("heart.msh")
tn,tt,ttet = parse_msh("torso.msh")

# ============================= fig_geom.png : domain schematic =================
def box_edges(x0,x1,y0,y1,z0,z1):
    c=np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
                [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]])
    E=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    return [[c[a],c[b]] for a,b in E]

fig=plt.figure(figsize=(13,5.2))
ax=fig.add_subplot(1,2,1,projection='3d')
ax.add_collection3d(Line3DCollection(box_edges(-25,25,-25,25,-25,25),colors='0.6',lw=1.0))
# heart slab as solid faces
def box_faces(x0,x1,y0,y1,z0,z1):
    c=np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
                [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]])
    F=[[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[1,2,6,5],[0,3,7,4]]
    return [c[f] for f in F]
ax.add_collection3d(Poly3DCollection(box_faces(-10,10,-3.5,3.5,-1.5,1.5),
                    facecolor='crimson',edgecolor='darkred',alpha=.85,lw=0.8))
ax.scatter([-25,25],[0,0],[0,0],c='navy',s=40)
ax.text(-25,0,4,'L',color='navy'); ax.text(25,0,4,'R',color='navy')
ax.set_xlim(-25,25);ax.set_ylim(-25,25);ax.set_zlim(-25,25)
ax.set_xlabel('x (mm)');ax.set_ylabel('y (mm)');ax.set_zlabel('z (mm)')
ax.set_title('Torso 50$^3$ mm (gray) + heart slab 20$\\times$7$\\times$3 mm (red)',fontsize=10)
ax.view_init(elev=18,azim=-60)
# cross sections
ax2=fig.add_subplot(1,2,2)
ax2.add_patch(plt.Rectangle((-25,-25),50,50,fill=True,fc='0.9',ec='0.5'))
ax2.add_patch(plt.Rectangle((-10,-1.5),20,3,fill=True,fc='crimson',ec='darkred'))
ax2.annotate('',xy=(10,-4),xytext=(-10,-4),arrowprops=dict(arrowstyle='<->'))
ax2.text(0,-6,'20 mm',ha='center',fontsize=9)
ax2.annotate('',xy=(-13,1.5),xytext=(-13,-1.5),arrowprops=dict(arrowstyle='<->'))
ax2.text(-15,0,'3',ha='center',fontsize=9)
ax2.scatter([-25,25],[0,0],c='navy',s=30); ax2.text(-24,2,'L',color='navy');ax2.text(22,2,'R',color='navy')
ax2.set_xlim(-27,27);ax2.set_ylim(-27,27);ax2.set_aspect('equal')
ax2.set_xlabel('x (mm)');ax2.set_ylabel('z (mm)')
ax2.set_title('x-z cross-section (y=0): conforming heart/torso interface',fontsize=10)
fig.suptitle('Geometry: conforming heart-in-torso (fibers $\\parallel$ x; electrodes L,R at x=$\\pm$25)',
             fontsize=12,weight='bold')
fig.tight_layout(); fig.savefig("fig_geom.png",dpi=140,bbox_inches='tight'); plt.close(fig)

# ============================= fig_meshview.png : the real mesh ================
fig=plt.figure(figsize=(13,5.4))
# (A) 3D heart boundary surface triangulation, colored interface vs outer
ax=fig.add_subplot(1,2,1,projection='3d')
tri_if =[hn[[a,b,c]] for a,b,c,ph in ht if ph==2]   # interface
tri_out=[hn[[a,b,c]] for a,b,c,ph in ht if ph!=2]   # outer heart surface
if tri_out: ax.add_collection3d(Poly3DCollection(tri_out,facecolor='#f4a582',edgecolor='0.3',lw=0.15,alpha=.9))
if tri_if:  ax.add_collection3d(Poly3DCollection(tri_if ,facecolor='#4393c3',edgecolor='0.2',lw=0.15,alpha=.9))
ax.set_xlim(-11,11);ax.set_ylim(-4.5,4.5);ax.set_zlim(-2.5,2.5)
ax.set_box_aspect((20,7,3))
ax.set_xlabel('x');ax.set_ylabel('y');ax.set_zlabel('z')
ax.set_title(f'Heart boundary surface mesh ({len(ht)} tris; blue=interface)',fontsize=10)
ax.view_init(elev=22,azim=-55)
# (B) true y=0 plane-slice of the tets: torso (coarse) + heart (fine)
ax2=fig.add_subplot(1,2,2)
seg_t=slice_tets(tn,ttet,axis=1,level=0.0)
seg_h=slice_tets(hn,htet,axis=1,level=0.0)
ax2.add_collection(LineCollection(seg_t,colors='0.55',lw=0.3))
ax2.add_collection(LineCollection(seg_h,colors='crimson',lw=0.5))
ax2.set_xlim(-26,26);ax2.set_ylim(-26,26);ax2.set_aspect('equal')
ax2.set_xlabel('x (mm)');ax2.set_ylabel('z (mm)')
ax2.set_title('y=0 slice of the tetrahedra: torso coarse (gray) + heart fine (red)',fontsize=10)
fig.suptitle('Unstructured P1 tetrahedral mesh (real connectivity from heart.msh / torso.msh)',
             fontsize=12,weight='bold')
fig.tight_layout(); fig.savefig("fig_meshview.png",dpi=140,bbox_inches='tight'); plt.close(fig)

print("wrote fig_geom.png fig_meshview.png")
print(f"heart: {len(hn)-1} nodes, {len(htet)} tets, {len(ht)} bdr tris; "
      f"torso: {len(tn)-1} nodes, {len(ttet)} tets, {len(tt)} bdr tris")
print(f"y=0 slice cells: torso {len(seg_t)//3}-ish polys, heart {len(seg_h)//3}-ish polys")
