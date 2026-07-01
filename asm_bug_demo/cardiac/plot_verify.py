#!/usr/bin/env python3
"""plot_verify.py -- verification plots for the coupled forward-ECG run:
  (A) time curves  Vm, u_e (heart centre) and phi_torso / ECG (body surface)
  (B) whole-field solutions  Vm & u_e over the HEART, phi over the TORSO,
      at snapshots spanning depolarization -> plateau -> repolarization.
Run after:  mpirun -n R ./forward_ecg -m heart.msh -dt 0.1 -T 350 -dump_fields
Units: mm, ms, mV.
"""
import numpy as np, matplotlib, glob
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

def _files(base):
    fs = sorted(glob.glob(f"{base}_r*.txt"));  return fs if fs else [f"{base}.txt"]
def load_xyz(base):
    a = np.concatenate([np.loadtxt(f).reshape(-1,3) for f in _files(base)]);  return a[:,0],a[:,1],a[:,2]
def load_v(base):
    return np.concatenate([np.loadtxt(f).reshape(-1) for f in _files(base)])

hx,hy,hz = load_xyz("heart_xyz")
tx,ty,tz = load_xyz("torso_xyz")

# ---------- (A) time curves: Vm, u_e, phi_torso ---------------------------
d = np.loadtxt("fwd_ecg.txt")
t, ecg, vmc, uec, phiL = d[:,0], d[:,1], d[:,2], d[:,3], d[:,4]
fig, ax = plt.subplots(4,1, figsize=(9,10), sharex=True)
ax[0].plot(t, vmc, color="crimson");  ax[0].set_ylabel("Vm @ centre (mV)")
ax[0].set_title("Sys1  transmembrane potential Vm (TP06 action potential)"); ax[0].grid(alpha=.3)
ax[1].plot(t, uec, color="seagreen"); ax[1].set_ylabel("u_e @ centre (mV)")
ax[1].set_title("Sys2  extracellular potential u_e (singular pure-Neumann recovery)"); ax[1].grid(alpha=.3)
ax[2].plot(t, phiL, color="darkorange"); ax[2].set_ylabel("phi_torso @ x=-25 (mV)")
ax[2].set_title("Sys3  torso surface potential u_T at the left electrode"); ax[2].grid(alpha=.3)
ax[3].plot(t, ecg, color="navy");     ax[3].set_ylabel("ECG  phi_L-phi_R (mV)")
ax[3].set_title("Body-surface ECG lead (QRS on depolarization, T-wave on repolarization)")
ax[3].set_xlabel("t (ms)"); ax[3].grid(alpha=.3)
fig.tight_layout(); fig.savefig("fig_verify_time.png", dpi=130); plt.close(fig)

# ---------- (B) whole-field solutions at snapshots ------------------------
TIMES = [12, 36, 120, 240, 312]   # depol, depol-end, plateau, late-plateau, repol
def tricf(ax, X,Y,Z, V, zsel, tol, levels, cmap, sym=False, vlim=None):
    m = np.abs(Z-zsel) < tol
    tri = mtri.Triangulation(X[m], Y[m]); v = V[m]
    if sym:
        a = vlim or (np.percentile(np.abs(v),99) or 1.0)
        cf = ax.tricontourf(tri, v, levels=np.linspace(-a,a,levels), cmap=cmap, extend="both")
    elif vlim:
        cf = ax.tricontourf(tri, v, levels=np.linspace(vlim[0],vlim[1],levels), cmap=cmap, extend="both")
    else:
        cf = ax.tricontourf(tri, v, levels=levels, cmap=cmap, extend="both")
    ax.set_aspect("equal"); return cf

nT = len(TIMES)
# FIXED symmetric scales per row (max over all shown snapshots) so magnitudes are
# comparable across columns and the shared colorbar is honest -- plateau frames then
# correctly read ~0 (white) instead of being auto-amplified to a spurious dipole.
ue_all = [load_v(f"heart_ue_{tm:03d}") for tm in TIMES]
ph_all = [load_v(f"torso_phi_{tm:03d}") for tm in TIMES]
ue_lim = max(np.percentile(np.abs(u),99.5) for u in ue_all)
ph_lim = max(np.percentile(np.abs(p),99.5) for p in ph_all)
fig, axes = plt.subplots(3, nT, figsize=(3.4*nT, 10))
for j,tm in enumerate(TIMES):
    Vm = load_v(f"heart_vm_{tm:03d}");  Ue = ue_all[j];  Ph = ph_all[j]
    c0 = tricf(axes[0,j], hx,hy,hz, Vm, 0.0, 0.6, 25, "RdBu_r", vlim=(-90,30))
    axes[0,j].set_title(f"t = {tm} ms", fontsize=11)
    c1 = tricf(axes[1,j], hx,hy,hz, Ue, 0.0, 0.6, 25, "seismic", sym=True, vlim=ue_lim)
    c2 = tricf(axes[2,j], tx,ty,tz, Ph, 0.0, 2.0, 25, "seismic", sym=True, vlim=ph_lim)
    axes[2,j].add_patch(plt.Rectangle((-10,-3.5),20,7, fill=False, ec="k", lw=1.0))
    for r in range(3): axes[r,j].set_xlabel("x (mm)")
axes[0,0].set_ylabel("HEART  Vm (mV)\n\ny (mm)")
axes[1,0].set_ylabel("HEART  u_e (mV)\n\ny (mm)")
axes[2,0].set_ylabel("TORSO  phi (mV)\n\ny (mm)")
fig.colorbar(c0, ax=axes[0,:].tolist(), shrink=0.7, label="Vm (mV)")
fig.colorbar(c1, ax=axes[1,:].tolist(), shrink=0.7, label="u_e (mV)")
fig.colorbar(c2, ax=axes[2,:].tolist(), shrink=0.7, label="phi (mV)")
fig.suptitle("Whole-field solutions (z~0 slice): Sys1 Vm & Sys2 u_e on the heart, Sys3 phi on the torso "
             "(black box = heart footprint)", fontsize=13)
fig.savefig("fig_verify_fields.png", dpi=120, bbox_inches="tight"); plt.close(fig)

print("wrote fig_verify_time.png  fig_verify_fields.png")
print(f"Vm range [{vmc.min():.1f}, {vmc.max():.1f}] mV   u_e range [{uec.min():.3f}, {uec.max():.3f}] mV"
      f"   ECG range [{ecg.min():.3f}, {ecg.max():.3f}] mV")
