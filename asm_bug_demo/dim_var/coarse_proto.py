"""coarse_proto.py -- prototype of the shift-invariant (K,M) spectral coarse space.

Idea (non-GenEO, problem-flow-aligned): the three cardiac systems are the SAME
stiffness K at different mass shifts sigma (Sys1: K+sigma*I large sigma; Sys2: K,
sigma->0 singular). The low eigenvectors of K are shift-INVARIANT: they are also the
low modes of K+sigma*I for ANY sigma. So build ONE coarse basis V0 = lowest m modes
of K (the lowest is the constant = the pure-Neumann nullspace = the collapsed
lambda_min mode), and reuse it across all sigma.

Two-level additive (symmetric, CG-safe):
    M2^-1 = M1^-1  +  V0 (V0^T A V0)^-1 V0^T          (M1 = one-level ASM/sASM)

Validate: (i) it lifts the collapsed lambda_min on pure-Neumann small-sigma,
(ii) the SAME V0 works across sigma (shift-invariance), (iii) iterations drop.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
import asm_spectral as A


def low_modes(K, m):
    """m lowest eigenvectors of K (dense; small problems). For pure-Neumann K the
    lowest is ~constant (eigenvalue ~0) = the nullspace mode."""
    w, V = np.linalg.eigh(K.toarray())
    return V[:, :m], w[:m]


class TwoLevel(LinearOperator):
    def __init__(self, Amat, onelevel, V0):
        self.A = Amat.tocsr(); self.N = Amat.shape[0]
        super().__init__(dtype=np.float64, shape=(self.N, self.N))
        self.M1 = onelevel                       # AdditiveSchwarz (M1^-1)
        Q, _ = np.linalg.qr(V0)                  # orthonormal coarse basis
        self.V0 = Q
        self.A0 = Q.T @ (self.A @ Q)             # coarse operator (m x m)
        self.A0inv = np.linalg.inv(self.A0)
        self.Nhat = onelevel.Nhat

    def _matvec(self, r):
        z1 = self.M1._matvec(r)                  # fine (one-level) correction
        z0 = self.V0 @ (self.A0inv @ (self.V0.T @ r))   # coarse correction
        return z1 + z0

    def sym_similar(self):                       # exact spectrum of M2^-1 A
        Ad = self.A.toarray()
        L = np.linalg.cholesky(Ad)
        ML = np.empty((self.N, self.N))
        for j in range(self.N):
            ML[:, j] = self._matvec(L[:, j])
        S = L.T @ ML
        return 0.5 * (S + S.T)


def kap(asm):
    ev = np.linalg.eigvalsh(asm.sym_similar())
    ev = ev[ev > 1e-10]
    return ev.min(), ev.max(), ev.max() / ev.min()


if __name__ == '__main__':
    dim, M, S, ov = 2, 25, 4, 1
    a = A.coef_field('const', dim, M)
    subs = A.box_subdomains(dim, M, S, ov)

    # build the pure-Neumann stiffness ONCE (sigma=0) and its low modes (shift-invariant)
    Kneu, _ = A.assemble(dim, M, a, bc='neumann', sigma=0.0)
    print("=== shift-invariant (K,M) spectral coarse: ONE V0 reused across sigma ===")
    for m in (1, 4, 8, 16):
        V0, lammodes = low_modes(Kneu, m)
        print("\n m=%d coarse modes (lowest K-eigenvalues: %s)"
              % (m, np.array2string(lammodes, precision=3, max_line_width=120)))
        print("  sigma | one-level sASM        | two-level (sASM + coarse)")
        print("        | lmin     kappa   iter | lmin     kappa   iter")
        for sigma in (10.0, 1.0, 0.1, 0.01):
            Amat, b = A.assemble(dim, M, a, bc='neumann', sigma=sigma)
            one = A.AdditiveSchwarz(Amat, subs, 'ic0', 'sasm')
            two = TwoLevel(Amat, one, V0)
            l1 = kap(one); l2 = kap(two)
            it1, _ = A.cg_iters(Amat, b, one)
            it2, _ = A.cg_iters(Amat, b, two)
            print("  %5g | %.2e %7.1f %4d | %.2e %7.1f %4d"
                  % (sigma, l1[0], l1[2], it1, l2[0], l2[2], it2))
