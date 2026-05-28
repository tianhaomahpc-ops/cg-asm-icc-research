# ASM + CG on a Laplacian cube — bug isolation + single-level fix

A minimal, self-contained MFEM 4.9 + PETSc 3.24 investigation of the
**"PCASM overlap up → CG iter up"** phenomenon observed when solving the
`u_e` recovery and torso elliptic systems inside an upstream
electrophysiology code (cardioid). The same behaviour is reproduced
here on two model problems:

1. **`asm_demo`** — cube Laplacian with **1 Dirichlet + 5 Neumann** faces
   (asymmetric BC, non-singular). The full investigation including
   matrix bit-identity vs an independent pure-PETSc P1 FEM assembler
   is in [`INVESTIGATION.md`](./INVESTIGATION.md).

2. **`recoverue_demo`** — cube Laplacian with **all-Neumann** BC (singular,
   null-space = constants). Mirrors cardioid's `Sys2 / u_e recovery`
   case structurally: a singular SPD system that needs `MatSetNullSpace`
   to be solvable. Compares scheme 0 (ASM) vs scheme 3 (sASM) directly.

If you only want the headline answer:

> `PC_ASM_BASIC + sub_pc=ICC + KSP=CG` on an elliptic problem genuinely
> produces an iter count that grows with overlap; it is not an MFEM or
> Hypre or matrix-conversion artefact. Within the single-level
> framework, the honest fix that keeps CG symmetric is **scaled additive
> Schwarz** (`PCSHELL`: `D^{-1/2} · M_BASIC^{-1} · D^{-1/2}`, where
> `D = diag(multiplicity)`). See `asm_bug_demo/asm_demo.cpp` — `-scheme 3`.

The full investigation log (problem statement, what was tried, what was
falsified, three sources of MFEM↔PETSc disagreement and how each was
removed, headline tables, reproduction recipe) is in
[`INVESTIGATION.md`](./INVESTIGATION.md).

---

## Repository layout

```
asmcg-laplace-demo/
├── README.md                  # this file (quick orientation)
├── INVESTIGATION.md           # full chronological write-up + tables
└── asm_bug_demo/
    ├── asm_demo.cpp           # MFEM 4.9 + PETSc 3.24 reproducer
    │                          # (1 Dirichlet + 5 Neumann faces)
    ├── recoverue_demo.cpp     # all-Neumann singular Laplacian, mirrors
    │                          # cardioid Sys2 / u_e recovery shape.
    │                          # Compares scheme 0 (ASM) vs scheme 3 (sASM)
    ├── pure_petsc_fem.c       # Independent pure-PETSc P1 hex→tet FEM
    ├── pure_petsc_demo.c      # Earlier 7-pt FD reference (kept for sanity)
    ├── pure_petsc_load.c      # PETSc driver that loads MFEM's partitioned
    │                          # matrix+rhs so iter counts match bit-for-bit
    ├── sweep.sh / sweep_pure.sh / bench.sh / bench_recoverue.sh
    └── Makefile               # uses Spack-installed MFEM/PETSc/HYPRE/METIS
```

## What this code actually solves

```
PDE:        -Δu = 1     on  [0,1]^3
BC:          u = 0      on  x = 0   (Dirichlet, attribute 5 in MFEM)
            ∂u/∂n = 0   on the other 5 faces  (natural / homogeneous Neumann)
Discretisation:
    structured nx³ hex mesh, each hex split into 6 tetrahedra sharing the
    (0,6) main diagonal — identical to MFEM Mesh::MakeCartesian3D with
    Element::TETRAHEDRON.  P1 (linear) H1 element, (nx+1)³ vertex DOFs.
Solver:
    -ksp_type cg  -ksp_norm_type preconditioned  -ksp_rtol 1e-6
    -pc_type asm  -pc_asm_type basic  -pc_asm_overlap N
    -sub_ksp_type preonly  -sub_pc_type icc  -sub_pc_factor_levels L
```

## Build

The `Makefile` reads MFEM's installed `config.mk` to pick up the same
mpic++/PETSc/HYPRE/METIS toolchain that the upstream cardioid project
links against. **Hard-coded paths** (edit per host):

```make
MFEM_DIR          /Users/tianhaoma/spack/.../mfem-4.9.0-...
PETSC_DIR_DERIVED /Users/tianhaoma/spack/.../petsc-3.24.4-...
MPICC, MPICXX     /Users/tianhaoma/spack/.../openmpi-5.0.9-.../bin/mpicc[xx]
```

Build all four binaries:

```bash
cd asm_bug_demo
make            # -> asm_demo, pure_petsc_fem, pure_petsc_demo, pure_petsc_load
```

## Run

```bash
# Reproduce the bug (MFEM side)
./sweep.sh 1 0 24 4            # fix_level=1, scheme=0 (CG+ASM_BASIC), nx=24, 4 ranks

# Three single-level fixes, in increasing "stays-CG-ness":
./sweep.sh 1 1 24 4            # GMRES + ASM RESTRICT (RAS)         — non-symmetric Krylov
./sweep.sh 1 2 24 4            # BCGS  + ASM RESTRICT (RAS)         — non-symmetric Krylov
./sweep.sh 1 3 24 4            # CG    + sASM via PCSHELL           — RECOMMENDED, stays symmetric

# Independent verification with a pure-PETSc P1 FEM assembler (no MFEM):
./sweep_pure.sh 0 24 4         # bug reproduces here too
./sweep_pure.sh 3 24 4         # sASM fix works here too

# Bit-identity demonstration (load MFEM's METIS-partitioned matrix into pure PETSc):
mpirun -n 4 ./asm_demo -fix_level 2 -scheme 0 -nx 24 \
  -ksp_norm_type preconditioned -ksp_initial_guess_nonzero false \
  -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 1000 \
  -pc_type asm -pc_asm_type basic -pc_asm_overlap 1 \
  -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels 0
mpirun -n 4 ./pure_petsc_load -nx 24 -scheme 0 \
  -ksp_norm_type preconditioned -ksp_initial_guess_nonzero false \
  -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 1000 \
  -pc_type asm -pc_asm_type basic -pc_asm_overlap 1 \
  -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels 0
# Both print iters=76 final_pnorm=1.747e-07  (bit-for-bit identical)
```

## Headline results (nx=24, 4 ranks, P1 tet, METIS partition)

Per-scheme iter at `L=0` (no ICC fill):

|   | O=0 | O=1   | O=2     | trend  |
|:--|:-:|:-:|:-:|:--|
| scheme 0 — CG + ASM_BASIC          | 71 | **76 ↑**  | **99 ↑↑** | reproduces the bug |
| scheme 1 — GMRES + ASM_RESTRICT     | 125 | 77 ↓ | 76 ↓ | overlap fixes itself (non-sym Krylov) |
| scheme 2 — BCGS + ASM_RESTRICT      | 48  | 36 ↓ | 41 ↓ | best absolute iter (non-sym Krylov) |
| **scheme 3 — CG + sASM (PCSHELL)**  | 71  | **59 ↓** | **59 ↓** | **stays CG, single-level, fixes overlap** |

Cross-check vs. independent pure-PETSc implementation, after METIS
partition + per-rank layout alignment:

```
scheme=0 (CG+ASM_BASIC):  9/9 EQUAL  (includes 4 cells where both stagnate at 1000)
scheme=1 (GMRES+RAS):     9/9 EQUAL
scheme=2 (BCGS+RAS):      9/9 EQUAL
scheme=3 (CG+sASM):       9/9 EQUAL
                          ─────────
                          36/36 cells iter bit-identical
```

## Why two implementations were necessary

To rule out the possibility that the bug is an MFEM artefact (a custom
HypreParMatrix → PETSc conversion, an unusual sparsity allocation, a
nullspace handling quirk, …), we hand-wrote an independent pure-PETSc
P1 hex→tet FEM assembler that uses **no MFEM, no Hypre, no shared
helpers**. It uses MFEM's `hex_to_tet[6][4]` decomposition table
verbatim (lifted from `mfem-4.9/mesh.cpp:986`) so the discrete operator
matches. The two implementations are mathematically equal:

| nx | MFEM ‖Ax‖     | pure-PETSc ‖Ax‖ |
|:--:|:--|:--|
| 16 | 2.409188e+01  | 2.409188e+01    |
| 17 | 2.137087e+01  | 2.137087e+01    |
| 24 | 3.128297e+01  | 3.128297e+01    |

`pure_petsc_load.c` then closes the partition gap: it `MatSetSizes` to
MFEM's per-rank row count **before** `MatLoad`, which is the one detail
that PETSc's binary I/O does not preserve by default (defaulting to
`PETSC_DECIDE` and silently re-chunking). With all three layers
(matrix-zero tolerance, METIS partition, row layout) aligned, the
ksp iter counts match bit-for-bit.

## `recoverue_demo`: all-Neumann (singular) Laplacian — ASM vs sASM

This second demo mirrors the structural shape of cardioid's
`hack/femheart.cpp` `Sys2 / u_e recovery` solve:

```
PDE:        -∇·(σ ∇u) = -∇·(σ_i ∇V_m)    on  [0,1]^3
BC:          ∂u/∂n = 0                   on all 6 faces
Discrete:   A u = A V_m   (here σ = σ_i = 1, so same stiffness on both sides)
Choice:     V_m(x,y,z) = cos(πx) cos(πy) cos(πz)     (zero mean, Neumann-compatible)
            ⇒  analytical  u = V_m + const
```

The matrix `A` is **singular** (kernel = span{1}); the demo attaches
`MatNullSpace` and gauges the solution to mean zero. This is exactly
the configuration Sys2 lives in.

### Headline iter count (`nx = 24`, 4 ranks, 15625 DOFs, run via `bench_recoverue.sh`):

| | (O,L) | ASM (scheme 0)<br>iter / time(s) | sASM (scheme 3)<br>iter / time(s) | Δ iter | Δ time |
|:-:|:-:|:-:|:-:|:-:|:-:|
| | (0,0) | 64 / 0.022 | 64 / 0.018 | 0 (D=I at O=0)| −18% |
| | (0,1) | 56 / 0.016 | 56 / 0.014 | 0 | −13% |
| | (0,2) | 50 / 0.018 | 50 / 0.015 | 0 | −17% |
| **★** | **(1,0)** | **68 / 0.020** | **52 / 0.012** | **−24%** | **−40%** |
| | (1,1) | 52 / 0.019 | 40 / 0.011 | −23% | −42% |
| | (1,2) | 45 / 0.023 | 37 / 0.014 | −18% | −39% |
| **★** | **(2,0)** | **87 / 0.029** | **51 / 0.022** | **−41%** | **−24%** |
| | (2,1) | 53 / 0.024 | 38 / 0.028 | −28% | noise (small problem) |
| | (2,2) | 44 / 0.031 | 34 / 0.014 | −23% | −55% |

Trend (`L = 0` column):

```
       O=0    O=1    O=2          iter
ASM    64 --> 68 --> 87           ↑↑  iter UP with overlap (the bug)
sASM   64 --> 52 --> 51           ↓   iter DOWN (theory recovered)
```

### Solution-correctness checks (every cell)
- `meanU ~ 10^-17` after solve → null-space gauge clean
- true residual `||r||/||b|| ~ 10^-6` consistent with the rtol target
- error vs analytical: `||u_0 - V_m_0|| / ||V_m_0|| ~ 10^-7 ~ 10^-6` —
  the discrete solution **equals the analytical V_m up to a constant**
  to KSP-residual accuracy, both for ASM and sASM, regardless of (O,L)

### Why this matters for cardioid
The "iter-up-with-overlap" mis-behaviour persists for this singular
all-Neumann case, and the same `InstallScaledASM(ksp, A, overlap, icc)`
PCSHELL patch fixes it without touching CG or introducing a coarse
correction. Dropping the patch into `hack/femheart.cpp` after each
`PetscPCGSolver(...)` construction is a ~5-line change per system.

### Reproduction

```bash
cd asm_bug_demo
make recoverue_demo

# Single-cell sanity check, baseline ASM
mpirun -n 4 ./recoverue_demo -nx 24 -scheme 0 -overlap 1 -icc 0 \
  -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 2000 -ksp_converged_reason

# Same with sASM
mpirun -n 4 ./recoverue_demo -nx 24 -scheme 3 -overlap 1 -icc 0 \
  -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 2000 -ksp_converged_reason

# Full sweep over (overlap, ICC level), iter + min time (5 repeats)
./bench_recoverue.sh 24 4 5
```

## License

Mixed:
- The new source under `asm_bug_demo/` is permissively licensed (MIT-style).
- `asm_demo.cpp` borrows the `ConvertHypreToPetscAIJSafe` shape from
  cardioid's `hack/femheart.cpp` (MIT, LLNL); the `hex_to_tet[6][4]`
  table in `pure_petsc_fem.c` is reproduced from MFEM `mesh/mesh.cpp`
  (BSD-3, LLNL).

Acknowledgements to the upstream cardioid project (`hack/femheart.cpp`)
for the original observation that triggered this investigation.
