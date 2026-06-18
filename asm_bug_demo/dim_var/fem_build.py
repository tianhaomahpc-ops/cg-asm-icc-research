"""fem_build.py -- unstructured P1 FEM assembly for the ASM/sASM overlap study.

Generates a genuinely unstructured simplicial mesh (Delaunay of a jittered grid)
on [0,1]^d, assembles the P1 stiffness for  -div(A grad u) = f  with a (possibly
anisotropic, axis-aligned) diffusion tensor A, applies the boundary conditions of
one of the four study cases, and writes everything PETSc-side:

  caseX_dimD_K.petsc   stiffness (PETSc binary AIJ, SPD after BC)
  caseX_dimD_b.petsc   load vector
  caseX_dimD_part.npz  node coords + box-partition + METIS-partition memberships

Cases:
  a : full Dirichlet,  isotropic     (A = I)
  b : full Dirichlet,  anisotropic    (A = diag, axis-aligned; Niederer 3D, 10x 2D)
  c : 1 Dirichlet face + zero-Neumann, isotropic
  d : 1 Dirichlet face + zero-Neumann, anisotropic

The C driver schwarz_fem.c loads K,b and the two partitions and runs PCASM
(BASIC / sASM) + ICC + CG with an overlap sweep.

Usage:  python3 fem_build.py <case a|b|c|d> <dim 1|2|3> <n_per_axis> <nsub>
"""
import sys, os, subprocess
import numpy as np
from scipy.spatial import Delaunay
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'mesh')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, '/opt/homebrew/Cellar/petsc/3.24.6/lib/petsc/bin')
import PetscBinaryIO  # noqa: E402

# Niederer monodomain conductivities (mS/mm), fiber || x  (from monodomain.c)
NIE_sL, NIE_sT = 0.1334, 0.0176          # ratio ~7.58


def tensor(dim, aniso):
    if not aniso:
        return np.eye(dim)
    if dim == 1:
        return np.eye(1)                  # anisotropy degenerate in 1D
    if dim == 2:
        return np.diag([10.0, 1.0])       # 10x, fiber || x
    return np.diag([NIE_sL, NIE_sT, NIE_sT]) / NIE_sT   # normalize T=1 -> diag(7.58,1,1)


# ---------------------------------------------------------------- mesh
def make_mesh(dim, n, jitter=0.35, seed=0):
    """Jittered-grid points on [0,1]^d (boundary kept exact) + Delaunay simplices."""
    rng = np.random.default_rng(seed)
    axes = [np.linspace(0, 1, n) for _ in range(dim)]
    grid = np.stack([g.ravel() for g in np.meshgrid(*axes, indexing='ij')], axis=1)
    h = 1.0 / (n - 1)
    interior = np.all((grid > 1e-12) & (grid < 1 - 1e-12), axis=1)
    grid[interior] += (rng.random(grid[interior].shape) - 0.5) * jitter * h
    if dim == 1:
        order = np.argsort(grid[:, 0])
        pts = grid[order]
        elems = np.stack([np.arange(len(pts) - 1), np.arange(1, len(pts))], axis=1)
        return pts, elems
    tri = Delaunay(grid)
    return grid, tri.simplices


# ---------------------------------------------------------------- assembly
def assemble(dim, pts, elems, A):
    """Vectorised P1 stiffness for -div(A grad u). Returns CSR (Nnode x Nnode)."""
    Nn = pts.shape[0]
    V = pts[elems]                                  # (Ne, d+1, d)
    v0 = V[:, 0, :]
    T = np.stack([V[:, k + 1, :] - v0 for k in range(dim)], axis=2)  # (Ne, d, d)
    detT = np.linalg.det(T)
    # drop degenerate (sliver) simplices that 3D Delaunay produces on near-regular
    # point sets -- they have ~zero volume and a singular T
    good = np.abs(detT) > 1e-10 * np.median(np.abs(detT))
    if not good.all():
        elems = elems[good]; V = V[good]; T = T[good]; detT = detT[good]
    vol = np.abs(detT) / np.math.factorial(dim)
    Tinv = np.linalg.inv(T)                         # (Ne, d, d)
    # grad of basis i=1..d are rows of Tinv; grad_0 = -sum
    G = np.zeros((len(elems), dim + 1, dim))
    for i in range(dim):
        G[:, i + 1, :] = Tinv[:, i, :]
    G[:, 0, :] = -np.sum(G[:, 1:, :], axis=1)
    # K_elem[e,i,j] = vol_e * G[e,i,:] . A . G[e,j,:]
    AG = np.einsum('pq,ejq->ejp', A, G)             # (Ne, d+1, d)
    Kel = np.einsum('eip,ejp->eij', G, AG) * vol[:, None, None]
    rows = np.repeat(elems, dim + 1, axis=1).reshape(len(elems), dim + 1, dim + 1)
    cols = np.repeat(elems[:, None, :], dim + 1, axis=1)
    K = sp.coo_matrix((Kel.ravel(), (rows.ravel(), cols.ravel())),
                      shape=(Nn, Nn)).tocsr()
    K.eliminate_zeros()
    return K


# ---------------------------------------------------------------- BC
def boundary_nodes(pts, mode):
    on = lambda c: (np.isclose(c, 0) | np.isclose(c, 1))
    if mode == 'full':                              # entire boundary
        m = np.zeros(len(pts), bool)
        for d in range(pts.shape[1]):
            m |= on(pts[:, d])
        return m
    # 'mixed': only x0 == 0 face is Dirichlet
    return np.isclose(pts[:, 0], 0.0)


def apply_dirichlet(K, b, dmask):
    keep = (~dmask).astype(float)
    P = sp.diags(keep)
    K2 = (P @ K @ P + sp.diags(dmask.astype(float))).tocsr()
    K2.eliminate_zeros()
    b2 = b.copy(); b2[dmask] = 0.0
    return K2, b2


# ---------------------------------------------------------------- partitions
def box_partition(pts, nsub):
    dim = pts.shape[1]
    S = max(1, int(round(nsub ** (1.0 / dim))))
    part = np.zeros(len(pts), int)
    for d in range(dim):
        idx = np.clip((pts[:, d] * S).astype(int), 0, S - 1)
        part = part * S + idx
    return part, S ** dim


def metis_partition(K, nsub, tag):
    """Partition the nodal graph with the gpmetis CLI."""
    A = K.tocsr()
    n = A.shape[0]
    gfile = os.path.join(OUT, tag + '.graph')
    # METIS graph: 1-based adjacency, no self loops, undirected
    ptr, idx = A.indptr, A.indices
    nedge = 0
    lines = []
    for i in range(n):
        nbrs = [j + 1 for j in idx[ptr[i]:ptr[i + 1]] if j != i]
        nedge += len(nbrs)
        lines.append(' '.join(map(str, nbrs)))
    with open(gfile, 'w') as f:
        f.write('%d %d\n' % (n, nedge // 2))
        f.write('\n'.join(lines) + '\n')
    subprocess.run(['gpmetis', gfile, str(nsub)],
                   check=True, capture_output=True)
    part = np.loadtxt(gfile + '.part.' + str(nsub), dtype=int)
    return part


# ---------------------------------------------------------------- main
def build(case, dim, n, nsub):
    aniso = case in ('b', 'd')
    bcmode = 'full' if case in ('a', 'b') else 'mixed'
    A = tensor(dim, aniso)
    pts, elems = make_mesh(dim, n)
    K = assemble(dim, pts, elems, A)
    # consistent load f=1 : b = K-independent lumped mass row sums ~ use ones (study)
    b = np.ones(K.shape[0])
    dmask = boundary_nodes(pts, bcmode)
    orphan = (np.abs(K.diagonal()) < 1e-300)        # nodes left in no element
    if orphan.any():
        dmask = dmask | orphan                       # pin them (keeps SPD)
    K, b = apply_dirichlet(K, b, dmask)

    tag = 'case%s_%dD' % (case, dim)
    io = PetscBinaryIO.PetscBinaryIO(precision='double', indices='32bit',
                                     complexscalars=False)
    with open(os.path.join(OUT, tag + '_K.petsc'), 'wb') as fh:
        io.writeMatSciPy(fh, K.tocsr())
    with open(os.path.join(OUT, tag + '_b.petsc'), 'wb') as fh:
        io.writeVec(fh, b.view(PetscBinaryIO.Vec))
    boxp, nbox = box_partition(pts, nsub)
    metp = metis_partition(K, nbox, tag)            # same #parts as boxes
    # node->part membership as PETSc IS binaries for the C driver
    for nm, arr in (('box', boxp), ('metis', metp)):
        with open(os.path.join(OUT, tag + '_' + nm + '.is'), 'wb') as fh:
            io.writeIS(fh, np.asarray(arr, dtype=np.int32).view(PetscBinaryIO.IS))
    np.savez(os.path.join(OUT, tag + '_part.npz'),
             coords=pts, box=boxp, metis=metp, nbox=nbox,
             dmask=dmask, A=A, case=case, dim=dim)
    print("[build] %s : N=%d elems=%d Ndir=%d nsub=%d (box=%d, metis parts=%d) Aniso=%s ratio=%.3g"
          % (tag, K.shape[0], len(elems), int(dmask.sum()), nbox,
             len(np.unique(boxp)), len(np.unique(metp)), aniso,
             (A[0, 0] / A[-1, -1]) if dim > 1 else 1.0))
    return K, b, pts


if __name__ == '__main__':
    case = sys.argv[1] if len(sys.argv) > 1 else 'a'
    dim = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    nsub = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    build(case, dim, n, nsub)
