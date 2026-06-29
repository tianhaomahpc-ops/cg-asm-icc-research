#!/usr/bin/env python3
"""plot_results.py -- render the conforming heart-torso mesh and the coupled
forward-ECG fields (Vm, u_e, torso phi, ECG) dumped by
  ./forward_ecg -m heart_torso.msh -T 60 -dt 0.02 -dump_fields
Produces fig_mesh.png, fig_vm.png, fig_ue.png, fig_torso.png, fig_ecg.png.
Units: mm, ms, mV.
"""
import numpy as np, matplotlib, glob
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# Fields are dumped PER-RANK (..._r<rank>.txt); concatenate across ranks to get
# the full field (works for any rank count; falls back to a single file).
def _files(base):
    fs = sorted(glob.glob(f"{base}_r*.txt"))
    return fs if fs else [f"{base}.txt"]
def load_xyz(base):
    a = np.concatenate([np.loadtxt(f).reshape(-1,3) for f in _files(base)])
    return a[:,0], a[:,1], a[:,2]
def load_v(base):
    return np.concatenate([np.loadtxt(f).reshape(-1) for f in _files(base)])

hx,hy,hz = load_xyz("heart_xyz")
tx,ty,tz = load_xyz("torso_xyz")
TIMES = [12,24,36,48]

def slice_tricontour(ax, X, Y, Z, V, zsel, tol, levels, cmap, title, sym=False):
    """tricontourf of field V over (X,Y) for points with |Z-zsel|<tol."""
    m = np.abs(Z - zsel) < tol
    x,y,v = X[m], Y[m], V[m]
    tri = mtri.Triangulation(x, y)
    if sym:
        a = np.percentile(np.abs(v), 99) or 1.0
        cf = ax.tricontourf(tri, v, levels=np.linspace(-a,a,levels), cmap=cmap, extend="both")
    else:
        cf = ax.tricontourf(tri, v, levels=levels, cmap=cmap, extend="both")
    ax.set_aspect("equal"); ax.set_title(title, fontsize=10)
    return cf

# ---------- Fig 1: mesh (conforming heart-in-torso) ----------------------
fig, (a1,a2) = plt.subplots(1,2, figsize=(12,5))
# (a) x-z cross-section at |y|<1.5: heart slab (red) centered in torso (gray)
mt = np.abs(ty) < 1.5; mh = np.abs(hy) < 1.5
a1.scatter(tx[mt], tz[mt], s=2, c="0.75", label="torso nodes")
a1.scatter(hx[mh], hz[mh], s=4, c="crimson", label="heart nodes")
a1.add_patch(plt.Rectangle((-10,-1.5),20,3, fill=False, ec="k", lw=1.5))
a1.set_aspect("equal"); a1.set_xlabel("x (mm)"); a1.set_ylabel("z (mm)")
a1.set_title("conforming mesh: 20x7x3 heart slab in 50^3 torso (y~0 slice)", fontsize=10)
a1.legend(loc="upper right", fontsize=8)
# (b) heart top view (x-y at z~0) node cloud -> shows the unstructured tet nodes
mh0 = np.abs(hz) < 0.4
a2.scatter(hx[mh0], hy[mh0], s=5, c="crimson")
a2.set_aspect("equal"); a2.set_xlabel("x (mm)"); a2.set_ylabel("y (mm)")
a2.set_title("heart unstructured P1 nodes (z~0 slice)", fontsize=10)
fig.tight_layout(); fig.savefig("fig_mesh.png", dpi=130); plt.close(fig)

# ---------- Fig 2: Vm depolarization wave (heart top view) ---------------
fig, axes = plt.subplots(1,4, figsize=(18,3.6))
for ax,tm in zip(axes, TIMES):
    V = load_v(f"heart_vm_{tm:03d}")
    cf = slice_tricontour(ax, hx,hy,hz, V, 0.0, 0.6, np.linspace(-90,30,25),
                          "RdBu_r", f"Vm  t={tm} ms")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
cb = fig.colorbar(cf, ax=axes, shrink=0.8, label="Vm (mV)")
fig.suptitle("Transmembrane potential Vm — depolarization wave (fiber || x), heart z~0 slice", fontsize=12)
fig.savefig("fig_vm.png", dpi=130, bbox_inches="tight"); plt.close(fig)

# ---------- Fig 3: u_e extracellular potential on the heart --------------
fig, axes = plt.subplots(1,4, figsize=(18,3.6))
for ax,tm in zip(axes, TIMES):
    U = load_v(f"heart_ue_{tm:03d}")
    cf = slice_tricontour(ax, hx,hy,hz, U, 0.0, 0.6, 25, "seismic",
                          f"u_e  t={tm} ms", sym=True)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
cb = fig.colorbar(cf, ax=axes, shrink=0.8, label="u_e (mV)")
fig.suptitle("Extracellular potential u_e on the heart (Sys2, singular pure-Neumann), z~0 slice", fontsize=12)
fig.savefig("fig_ue.png", dpi=130, bbox_inches="tight"); plt.close(fig)

# ---------- Fig 4: torso potential u_T (slice through the body) ----------
fig, axes = plt.subplots(1,4, figsize=(18,4.2))
for ax,tm in zip(axes, TIMES):
    P = load_v(f"torso_phi_{tm:03d}")
    cf = slice_tricontour(ax, tx,ty,tz, P, 0.0, 2.0, 25, "seismic",
                          f"phi_torso  t={tm} ms", sym=True)
    ax.add_patch(plt.Rectangle((-10,-3.5),20,7, fill=False, ec="k", lw=1.2))  # heart footprint
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
cb = fig.colorbar(cf, ax=axes, shrink=0.8, label="phi (mV)")
fig.suptitle("Torso potential u_T (Sys3 Laplace), z~0 slice; black box = heart footprint; electrodes at x=+-25", fontsize=12)
fig.savefig("fig_torso.png", dpi=130, bbox_inches="tight"); plt.close(fig)

# ---------- Fig 5: pseudo-ECG + Vm at centre -----------------------------
d = np.loadtxt("fwd_ecg.txt")
t, ecg, vmc = d[:,0], d[:,1], d[:,2]
fig, (e1,e2) = plt.subplots(2,1, figsize=(8,6), sharex=True)
e1.plot(t, ecg, "-", color="navy"); e1.set_ylabel("ECG  phi_L-phi_R (mV)")
e1.set_title("Body-surface ECG lead (torso electrodes x=-25 vs x=+25)"); e1.grid(alpha=.3)
e2.plot(t, vmc, "-", color="crimson"); e2.set_ylabel("Vm @ heart centre (mV)")
e2.set_xlabel("t (ms)"); e2.grid(alpha=.3)
fig.tight_layout(); fig.savefig("fig_ecg.png", dpi=130); plt.close(fig)

print("wrote fig_mesh.png fig_vm.png fig_ue.png fig_torso.png fig_ecg.png")
