#!/usr/bin/env python3
"""
plot_slices.py
--------------
Reads the ParaView .pvtu output from asm_demo and produces a side-by-side
image:
  - axial slice (z = 0.5) for source_type=0  (the 1D-degenerate case)
  - axial slice (z = 0.5) for source_type=1  (genuinely 3D case)
Saves PNG to slices.png so the user can SEE the difference.
"""
import sys
import os
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_pvtu(path):
    rdr = vtk.vtkXMLPUnstructuredGridReader()
    rdr.SetFileName(path)
    rdr.Update()
    ug = rdr.GetOutput()
    pts = vtk_to_numpy(ug.GetPoints().GetData())     # (N,3)
    u   = vtk_to_numpy(ug.GetPointData().GetArray("u"))
    return pts, u

def axial_slice(pts, u, z_target=0.5, tol=1e-4):
    """Pick points within |z - z_target| < tol, return x,y,u arrays."""
    mask = np.abs(pts[:, 2] - z_target) < tol
    return pts[mask, 0], pts[mask, 1], u[mask]

def line_probe(pts, u, axis, fixed_a, fixed_b, tol=1e-4):
    """
    Sample the solution along a line.  axis in {0,1,2}.  fixed_a / fixed_b
    are the values of the two OTHER coordinates that must match.
    """
    other = [i for i in (0,1,2) if i != axis]
    mask = ((np.abs(pts[:, other[0]] - fixed_a) < tol) &
            (np.abs(pts[:, other[1]] - fixed_b) < tol))
    coord = pts[mask, axis]
    val   = u[mask]
    idx   = np.argsort(coord)
    return coord[idx], val[idx]

def plot_two():
    files = {
        0: "paraview_nx24_n4_src0/u/Cycle000000/data.pvtu",
        1: "paraview_nx24_n4_src1/u/Cycle000000/data.pvtu",
    }
    titles = {
        0: "source_type=0:  f=1\n=> 1D-degenerate, u(x,y,z) = x - x²/2",
        1: "source_type=1:  f=sin(πy)sin(πz)\n=> genuinely 3D",
    }
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5))

    for col, src in enumerate(files):
        pts, u = load_pvtu(files[src])
        # ----- (row 0, col) axial slice z=0.5 ---------------------------
        xs, ys, vs = axial_slice(pts, u, 0.5)
        ax = axes[0, col]
        sc = ax.tricontourf(xs, ys, vs, levels=20, cmap="viridis")
        ax.set_title(titles[src])
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_aspect("equal")
        plt.colorbar(sc, ax=ax)
        # ----- (row 1, col) three orthogonal line probes --------------
        ax = axes[1, col]
        x_x, ux = line_probe(pts, u, axis=0, fixed_a=0.5, fixed_b=0.5)
        y_y, uy = line_probe(pts, u, axis=1, fixed_a=0.5, fixed_b=0.5)
        z_z, uz = line_probe(pts, u, axis=2, fixed_a=0.5, fixed_b=0.5)
        ax.plot(x_x, ux, "b.-", label="vary x, (y=z=0.5)")
        ax.plot(y_y, uy, "r.-", label="vary y, (x=z=0.5)")
        ax.plot(z_z, uz, "g.--", label="vary z, (x=y=0.5)")
        ax.set_xlabel("coord along line"); ax.set_ylabel("u")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    # The third column shows the analytical reference for src=0
    pts0, u0 = load_pvtu(files[0])
    pts1, u1 = load_pvtu(files[1])
    ax = axes[0, 2]
    # z=0.5 slice of u(x,y,z) - u_analytical, where u_analytical = x - x²/2
    xs, ys, vs = axial_slice(pts0, u0, 0.5)
    err = vs - (xs - 0.5 * xs * xs)
    sc = ax.tricontourf(xs, ys, err, levels=20, cmap="coolwarm")
    ax.set_title("source_type=0 error vs u*(x)=x-x²/2\n(should be O(h²) discretisation noise)")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax)

    ax = axes[1, 2]
    # along x at y=z=0.5: numerical vs analytical
    x_x0, ux0 = line_probe(pts0, u0, axis=0, fixed_a=0.5, fixed_b=0.5)
    ana = x_x0 - 0.5 * x_x0 * x_x0
    ax.plot(x_x0, ux0, "bo-", label="numerical (src=0)")
    ax.plot(x_x0, ana, "k--", label="u*(x)=x-x²/2")
    ax.set_xlabel("x"); ax.set_ylabel("u")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_title("src=0: numerical vs analytical along x")

    plt.suptitle("3D Laplacian on $[0,1]^3$ with 1 Dirichlet face (x=0) "
                 "+ 5 Neumann faces  --  P1 tet, nx=24, 4 ranks",
                 fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = "slices.png"
    plt.savefig(out, dpi=120)
    print(f"wrote {out}  ({os.path.getsize(out)} bytes)")
    print(f"  src=0:  ||u||={np.linalg.norm(u0):.6e}  "
          f"min={u0.min():.4f}  max={u0.max():.4f}  mean={u0.mean():.4f}")
    print(f"  src=1:  ||u||={np.linalg.norm(u1):.6e}  "
          f"min={u1.min():.4f}  max={u1.max():.4f}  mean={u1.mean():.4f}")

if __name__ == "__main__":
    plot_two()
