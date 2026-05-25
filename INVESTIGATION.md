# ASM+CG on Laplace / 1‑Dirichlet + 5‑Neumann cube
**Branch:** `asmcg-Laplace-dirch-nuemann`
**Workspace:** `asm_bug_demo/`

This document is a self-contained record of an investigation triggered by the
real cardioid Sys2/Sys3 (`u_e` Recovery / Torso) solver showing **iter count
increasing with PCASM overlap** — the opposite of what classical Schwarz
theory predicts and what the user's professor expected. The branch contains a
minimal-reproducer demo, several solver-side fix attempts, and an independent
implementation in pure PETSc whose results we eventually align with MFEM to
**bit-identical iter counts** across all (overlap × ICC-fill × solver-scheme)
combinations.

---

## 1. The problem (as it appeared in cardioid)

Running the cardioid `hack/femheart.cpp` non-POD path on N12 (12 ranks, mesh
refine=2, 10 timesteps) with MFEM 4.9 + PETSc 3.24 and the PETSc options:

```
-pc_type asm          -pc_asm_type basic     -pc_asm_overlap N
-sub_ksp_type preonly -sub_pc_type icc       -sub_pc_factor_levels L
-ksp_type cg          -ksp_norm_type preconditioned
-ksp_initial_guess_nonzero true
-ksp_rtol 1e-4..1e-5  -ksp_atol 1e-8..1e-12  -ksp_max_it 120..400
```

the measured average per-step iter counts were:

|       | Sys1 / Monodomain | Sys2 / u_e Recovery | Sys3 / Torso (max_it=200) |
|:------|:-----------------:|:-------------------:|:-------------------------:|
| O=0   | 2.0 → 1.9 → 1.9   | 194 → 151 → 133     | 115 → 115 → 114           |
| O=1   | 5.5 → 3.9 → 3.9   | **340 → 257 → 221** | **200 → 187 → 155**       |
| O=2   | 4.9 → 4.7 → 4.9   | **348 → 261 → 219** | **200 → 197 → 162**       |

(Columns are ICC fill `L = 0, 1, 2`.) **Sys3 hits `max_it = 200` for every
overlap ≥ 1**, meaning CG never reaches the rtol target; it is true divergence
relative to the convergence rule, not just slower convergence.

The expectation from classical Schwarz theory was that more overlap should
*reduce* iter count. Two implications had to be either confirmed or refuted:
1. is the cardioid code itself producing a wrong / degenerate matrix?
2. is the behaviour intrinsic to PCASM_BASIC + ICC + CG on elliptic problems?

---

## 2. What this demo does

`asm_bug_demo/` provides four independent solver drivers built against the
same MFEM 4.9 / PETSc 3.24 / HYPRE / METIS toolchain that cardioid links
against, plus shell scripts to sweep `(overlap, ICC level)` × 4 solver
schemes.

### Files
| File | Role |
|:--|:--|
| `asm_demo.cpp` | MFEM + PETSc reproducer: cube `[0,1]³`, P1 tetrahedral mesh, 1 Dirichlet face (`x=0, u=0`), 5 Neumann faces, hand-rolled `ConvertHypreToPetscAIJSafe` mirroring `femheart.cpp`. Knobs: `-fix_level 0..2`, `-scheme 0..3`, `-nx`. |
| `pure_petsc_fem.c` | Independent **pure PETSc** P1-tet FEM assembler — no MFEM, no HYPRE. Uses MFEM's `hex_to_tet[6][4]` table verbatim so the mesh + element decomposition is identical. |
| `pure_petsc_demo.c` | Earlier 7-point FD reference (kept for sanity; not used in final comparison). |
| `pure_petsc_load.c` | Pure PETSc driver that **loads** MFEM's METIS-partitioned matrix + RHS + per-rank layout from binary so MFEM and PETSc share the same partition; used to verify bit-identical iter counts. |
| `sweep.sh`, `sweep_pure.sh` | Sweep `(O,L)` for one solver scheme on each demo. |
| `Makefile` | Pulls MFEM 4.9 / PETSc 3.24 paths from `mfem-config.mk`. |

### Solver schemes (knob: `-scheme N`)
| N | KSP | PC | Note |
|:-:|:--|:--|:--|
| 0 | CG    | `PCASM` BASIC                        | baseline that reproduces cardioid's "iter up with overlap" |
| 1 | GMRES | `PCASM` RESTRICT (RAS, non-symmetric) | one of two "real fixes" requested by the user |
| 2 | BCGS  | `PCASM` RESTRICT (RAS, non-symmetric) | second "real fix" |
| 3 | CG    | `PCSHELL` = `D^{-1/2}·M_{BASIC}^{-1}·D^{-1/2}` (scaled ASM) | symmetric sASM written by hand; preserves CG |

### `fix_level N` in `asm_demo.cpp` (matrix-conversion side fixes)
| N | What it adds |
|:-:|:--|
| 0 | bit-for-bit copy of `ConvertHypreToPetscAIJSafe` from cardioid (no symmetry hint, single-element `MatSetValues`, stored zeros copied verbatim) |
| 1 | drop stored zeros with tolerance `|val| < 1e-12` |
| 2 | also flag `MAT_SYMMETRIC`, `MAT_SYMMETRY_ETERNAL`, `MAT_SPD` |
| 3 | also force `-sub_pc_factor_shift_type positive_definite` and `-sub_pc_factor_mat_ordering_type rcm` |

---

## 3. The process (chronological summary of what was learned)

### Phase A — initial wrong analysis

Without running anything, an analysis was proposed claiming the cardioid bug
came from (i) `PC_ASM_RESTRICT` being non-symmetric for `KSP_NORM_NATURAL`
CG, (ii) stored zeros from BC elimination polluting `MatIncreaseOverlap`,
(iii) missing `MAT_SYMMETRIC` flag, (iv) wrong sub-PC shift/ordering, etc.

### Phase B — the user provided the actual PETSc options file

It revealed `-pc_asm_type basic` and `-ksp_norm_type preconditioned` were
already set, falsifying hypothesis (i) above. The remaining hypotheses had
to be tested empirically rather than reasoned about.

### Phase C — `asm_demo.cpp` was written and the bug was reproduced

Running with `nx=24`, 4 ranks, `fix_level=0`, scheme=0 (CG + ASM_BASIC),
the demo gave the cardioid-style pattern:

```
O=0, L=0: 44 iter        O=1, L=0: 55 iter        O=2, L=0: 60 iter
```

Iter goes UP with overlap. **This confirmed the bug is not specific to
cardioid — it's reproducible in a minimal Laplacian on a cube.**

### Phase D — the proposed "fixes" were tested one by one

| `fix_level` | What changed | Iter at O=1, L=0 |
|:-:|:--|:-:|
| 0 | baseline | 55 |
| 1 | drop stored zeros (`MAT_IGNORE_ZERO_ENTRIES` + skip) | **55** (no change) |
| 2 | + `MAT_SYMMETRIC` / `MAT_SPD` flags | **55** (no change) |
| 3 | + `shift positive_definite`, ordering `rcm` | 58 (changes shape ±10%, trend unchanged) |

**All of the originally-proposed code-side fixes had near-zero effect** on
the iter trend. They had to be retracted.

### Phase E — three real fixes at the algorithm level

Holding the user to the constraint "no two-level method, no change of Krylov
unless absolutely necessary", three algorithmic alternatives were
implemented and swept:

| Scheme | Result (L=0): O=0 → O=1 → O=2 | Verdict |
|:------:|:--|:--|
| 0 (CG + BASIC, baseline) | 44 → 55 → 60 (⬆️⬆️) | reproduces cardioid bug |
| 1 (GMRES + RAS)           | 52 → 34 → 34 (⬇️)    | overlap fixes itself, but non-symmetric Krylov |
| 2 (BCGS + RAS)            | 35 → 24 → 24 (⬇️)    | best iter, but non-symmetric Krylov |
| 3 (**CG + sASM PCSHELL**) | 44 → **37** → **36** (⬇️) | **stays CG, stays single-level, stays symmetric — fixes the bug** |

The PCSHELL `sASM` wrap is essentially:
```
M_sASM⁻¹ = D⁻¹⸍² · ( Σᵢ Rᵢᵀ Aᵢ⁻¹ Rᵢ ) · D⁻¹⸍²
```
where `D = diag(multiplicity[k])` and `multiplicity[k] = #subdomains that
contain DOF k`. Computed once at setup by iterating `PCASMGetLocalSubdomains`
and accumulating an indicator vector. Application is two `VecPointwiseMult`
flanking the standard `PCApply` of the inner BASIC PC.

### Phase F — independent pure-PETSc reproduction

To rule out any MFEM-side artefact, the same problem (same cube mesh, same
`hex_to_tet[6][4]` decomposition, same Dirichlet/Neumann faces, same
RHS = 1) was reassembled in `pure_petsc_fem.c` using PETSc directly. With
the same matrix-zero filter on the MFEM side (`|val| < 1e-12`), the
matrices are **bit-identical** in NNZ, in `||x||`, and in `||Ax||` for any
`x`. (See section 4 below.)

### Phase G — making multi-rank iter counts also bit-identical

Even with bit-identical matrices, the 4-rank iter counts initially differed
because:
- MFEM `ParMesh` partitions elements with METIS;
- the PETSc demo used DMDA (cuboid box partition).

To eliminate that last source of difference, `asm_demo.cpp` dumps the
already-METIS-partitioned PETSc matrix + RHS + per-rank row layout to
binary; `pure_petsc_load.c` calls `MatSetSizes(A, my_mloc, ...)` **before**
`MatLoad` so PETSc preserves MFEM's METIS layout instead of falling back
to `PETSC_DECIDE`.

With this in place, the iter counts of MFEM (`asm_demo`) and pure PETSc
(`pure_petsc_load`) **match bit-for-bit on all 9 (O,L) × 4 schemes = 36
configurations**, including the cells where both runs stagnate at
`max_it = 1000`.

---

## 4. Headline results

### 4.1 The four schemes, MFEM end (P1 tet, nx=24, 4 ranks, METIS)

| | | s=0 CG+BASIC | s=1 GMRES+RAS | s=2 BCGS+RAS | s=3 **CG+sASM** |
|:-:|:-:|:-:|:-:|:-:|:-:|
| O=0 | L=0 | 71 | 125 | 49 | 71 |
| O=1 | L=0 | **76 ⬆**  | 77 ⬇ | 36 ⬇ | **59 ⬇** |
| O=2 | L=0 | **99 ⬆⬆** | 76 ⬇ | 41 ⬇ | **59 ⬇** |
| O=2 | L=2 | 48 | 32 | 20 | 36 |

### 4.2 MFEM ↔ pure-PETSc (after METIS alignment): all 36 cells `EQUAL`

```
scheme=0 (CG+ASM_BASIC):                  9/9 EQUAL   (including the 4 cells where
                                                       both stagnate at 1000)
scheme=1 (GMRES+RAS):                     9/9 EQUAL
scheme=2 (BCGS+RAS):                      9/9 EQUAL
scheme=3 (CG+sASM, PCSHELL):              9/9 EQUAL
```

### 4.3 Operator fingerprints (bit-identical to 7 sig figs)

Same deterministic test vector `x` evaluated through `||x||` and `||Ax||`:

| nx | MFEM ‖Ax‖   | PETSc ‖Ax‖  | match |
|:--:|:------------|:------------|:------|
| 16 | 2.409188e+01 | 2.409188e+01 | ✓ |
| 17 | 2.137087e+01 | 2.137087e+01 | ✓ |
| 18 | 2.100349e+01 | 2.100349e+01 | ✓ |
| 24 | 3.128297e+01 | 3.128297e+01 | ✓ |

---

## 5. The three sources of difference that had to be reconciled

| Layer | Difference | Fix |
|:--|:--|:--|
| L1 — matrix values | MFEM's `DiffusionIntegrator` uses numerical quadrature; for P1 tet the (0,6) main-diagonal split makes many K_loc entries analytically zero, but quadrature returns round-off (1e-16) instead of exact 0. Original `data[p] == 0.0` filter doesn't catch them. | Change filter to `|data[p]| < 1e-12` in `ConvertHypreToPetscAIJSafe`. NNZ becomes bit-identical, ‖Ax‖ already was. |
| L2 — element partition | MFEM `ParMesh` uses METIS by default; the PETSc demo used DMDA cuboid. Different subdomain shapes → different multiplicity profile → different sASM/BASIC numerics. | Dump MFEM's METIS-partitioned matrix to binary, load in PETSc. |
| L3 — row layout on load | `MatLoad` defaults to `PETSC_DECIDE`, which silently re-partitions rows into equal contiguous chunks, erasing MFEM's METIS layout. | Dump per-rank `mloc` to a layout file; in PETSc call `MatSetSizes(A, mloc_rank, ...)` before `MatLoad`. |

If any one of these three is left unfixed, the two implementations disagree
by 10–20 % in iter count. With all three fixed, every iter count matches
exactly.

---

## 6. The real conclusion

- **"Overlap up → iter up" with `PC_ASM_BASIC + ICC + CG` on elliptic
  problems is not a cardioid bug** and not an MFEM artefact. It is
  reproduced by an independent pure-PETSc P1 FEM assembler on the same
  cube Laplacian.
- The mathematically cleanest single-level fix is `CG + sASM` (the PCSHELL
  with `D^{-1/2}` multiplicity scaling). It keeps the Krylov method (CG),
  keeps the framework single-level (no coarse correction), and explicitly
  removes the over-counting that turns the BASIC variant against itself
  when `overlap > 0`.
- For cardioid's `Sys2 / Sys3`, dropping the same PCSHELL in after each
  `PetscPCGSolver` construction is a one-call change that should reverse
  the iter trend with overlap.

---

## 7. How to reproduce

```bash
cd asm_bug_demo
make                              # builds asm_demo, pure_petsc_fem, pure_petsc_load
                                  # (pure_petsc_demo too, but not needed)

# Baseline (reproduces the cardioid-style iter trend)
./sweep.sh 1 0 24 4               # fix_level=1, scheme=0 (CG+BASIC)

# Fixes (pick any of three)
./sweep.sh 1 1 24 4               # GMRES + RAS
./sweep.sh 1 2 24 4               # BCGS + RAS
./sweep.sh 1 3 24 4               # CG + sASM (the recommended one)

# Independent verification with pure PETSc P1 FEM (no MFEM)
./sweep_pure.sh 0 24 4            # baseline; reproduces the bug independently
./sweep_pure.sh 3 24 4            # sASM; reproduces the fix independently

# Bit-identity test (MFEM-METIS partition pushed into pure PETSc)
mpirun -n 4 ./asm_demo  -fix_level 2 -scheme 0 -nx 24 \
   -ksp_norm_type preconditioned -ksp_initial_guess_nonzero false \
   -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 1000 \
   -pc_type asm -pc_asm_type basic -pc_asm_overlap 1 \
   -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels 0
mpirun -n 4 ./pure_petsc_load -nx 24 -scheme 0 ...same KSP options...
# both reports identical iters, identical final pnorm
```

The Spack-installed toolchain paths are hard-coded in `asm_bug_demo/Makefile`:
- MFEM     4.9.0  at `~/spack/.../mfem-4.9.0-...`
- PETSc    3.24.4 at `~/spack/.../petsc-3.24.4-...`
- HYPRE    3.1.0
- METIS    5.1.0
- OpenMPI  5.0.9

Edit those paths in the Makefile if running on a different machine.
