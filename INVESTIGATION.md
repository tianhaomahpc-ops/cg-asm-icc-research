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

## 7. A cheaper, faster single-level variant: sASM + Chebyshev block solve (scheme 4)

### 7.1 Motivation, straight from the diagnosis

Section 3/6 established the disease as a **product** of two factors:

```
   iter-up-with-overlap   =   (BASIC overcounting)   x   (inexact ICC(0) local solve)
```

and proved (BASIC + *exact* Cholesky block solve makes iter go **down** with
overlap, e.g. 40 -> 30 -> 23) that removing **either** factor cures it.
`scheme 3` (sASM) removes the first factor for free. The natural next step is
to also shrink the **second** factor — but exact Cholesky is `O(n_i^3)` in
memory/time and unusable at cardioid scale. The cheap surrogate is a
**fixed low-degree Chebyshev iteration over ICC(0)** on each subdomain block:

```
   scheme 4  =  CG  +  D^{-1/2} ( sum_i R_i^T  S_i  R_i ) D^{-1/2}
   with       S_i  =  k steps of Chebyshev preconditioned by ICC(L) on block i
```

A *fixed* number of Chebyshev steps with **frozen** eigenvalue bounds and a
symmetric (ICC) smoother is a fixed SPD linear operator, so the outer CG stays
valid. It adds **no fill / no extra memory** beyond ICC(L), only a couple of
block mat-vecs, and — crucially — the block Chebyshev runs on `PETSC_COMM_SELF`,
so it performs **zero global reductions**. Default degree `k = 2`
(`-localcheby N` to override).

### 7.2 (a) Full (O, L) sweep — scheme 4 is stable and correct everywhere

`nx = 48`, 4 ranks, MFEM `asm_demo`. `sch0` = BASIC, `sch3` = sASM,
`sch4` = sASM + Chebyshev(2). All `sch4` cells converge (`CONVERGED_RTOL`) to
the same discretization error (`||u-u*||/||u*|| = 5.42e-05`) as the baseline,
so the solution is correct at every overlap.

| O | L | sch0 (BASIC) | sch3 (sASM) | **sch4 (sASM+Cheby2)** |
|:-:|:-:|:-:|:-:|:-:|
| 0 | 0 | 132 | 132 | **85** |
| 0 | 1 | 102 | 102 | **73** |
| 0 | 2 | 87  | 87  | **66** |
| 1 | 0 | 148 | 104 | **68** |
| 1 | 1 | 102 | 75  | **54** |
| 1 | 2 | 92  | 70  | **47** |
| 2 | 0 | **179** | 103 | **65** |
| 2 | 1 | 117 | 73  | **48** |
| 2 | 2 | 90  | 67  | **42** |

Notes:
- At `O = 0` we have `sch0 == sch3` exactly (multiplicity `D = I`, so
  sASM degenerates to BASIC) — a correctness self-check.
- `sch0` overlap trend at `L=0`: 132 -> 148 -> **179** (the bug, climbing).
- `sch4` overlap trend at `L=0`: 85 -> 68 -> **65** (cured, and lowest).
- `sch4` helps even at `O = 0` (85 vs 132) because the Chebyshev attacks the
  *inexactness* factor, which is present at every overlap.

**Cross-check vs. the independent pure-PETSc implementation** (same MFEM-METIS
matrix loaded via `pure_petsc_load`, scheme 4 added there too): all 9 (O, L)
cells match **bit-for-bit**.

```
scheme 4: MFEM(asm_demo)  vs  pure-PETSc(pure_petsc_load)   ->   9/9 EQUAL
```

### 7.3 (b) Hard proof of reduced synchronization (`-log_view`, noise-free)

Wall-clock time on a loaded laptop is unreliable (same config varies up to
1.7x run-to-run; a bigger problem can even appear "faster"). So the time
argument is made with a **deterministic** metric instead: the global-reduction
count from PETSc `-log_view`. Each reduction is one `MPI_Allreduce` — the
latency-bound global synchronization that dominates CG at HPC scale.

`nx = 48`, 4 ranks, `O = 2, L = 0` (the baseline's worst overlap):

| config | iter | MPI Reductions | VecTDot | = 2·iter + 2 ? |
|:--|:-:|:-:|:-:|:-:|
| BASELINE BASIC+ICC(0) | 179 | **668** | 360 | yes (360) |
| sASM+ICC(0)           | 103 | 443 | 208 | yes (208) |
| sASM+Cheby2/ICC(0)    | 65  | 329 | 132 | yes (132) |
| **sASM+Cheby2/ICC(1)**| 48  | **278** | 98 | yes (98) |

Two facts are proven here:

1. **`VecTDot = 2·iter + 2` holds exactly for every config**, including
   scheme 4. The CG dot-products (the global reductions) track only the
   **outer** iteration count — the inner Chebyshev block solves add **zero**
   global reductions (they run on `PETSC_COMM_SELF`). So scheme 4 trades
   *local* flops for *fewer global syncs*; it never inflates the sync count.
2. At `O = 2`, scheme 4 (`sASM+Cheby2/ICC1`) cuts
   - **iterations 179 -> 48  (3.7x fewer)**, and
   - **global synchronizations 668 -> 278  (2.4x fewer)**.

Because `MPI_Allreduce` is latency-bound and gets relatively more expensive as
the rank count grows, this 2.4x reduction in synchronizations is the concrete,
machine-independent basis for the "reduces time at scale" claim — exactly the
regime cardioid runs in (N12 = thousands of ranks).

### 7.4 Honest scope of the contribution

This is a **synthesis, not a new algorithm**. Multiplicity/partition-of-unity
scaled additive Schwarz exists; Chebyshev-smoothed Schwarz / polynomial block
smoothers exist. The reusable intellectual content here is the **diagnosis**
(disease = overcounting x inexactness, multiplicatively coupled) which then
*prescribes* the cheapest fix: attack each factor with the cheapest tool —
`D^{-1/2}` scaling for overcounting (free, symmetric), a fixed low-degree
Chebyshev block solve for inexactness (no extra memory, no extra global sync).
Stronger published relatives (RASHO, SORAS) reach better spectral constants
with harmonic/Robin local problems at higher implementation cost; deflation of
the constant mode is powerful but is a (1-dimensional) coarse space, i.e.
two-level.

### 7.5 Recommendation for cardioid Sys2 / Sys3

| scenario | recommended | why |
|:--|:--|:--|
| minimal change, safest | **sASM + ICC(1)** (scheme 3, L=1) | one PCSHELL, ~2x fewer iters, symmetric, zero risk |
| memory-tight, large scale | **sASM + Cheby2/ICC(0)** (scheme 4) | no extra fill, ~2.6x fewer iters, fewer global syncs |
| push iters lowest | sASM + Cheby2/ICC(1) (scheme 4) | ~3x fewer iters |
| latency-bound network | any of the above + `-ksp_type pipecg` | overlaps the allreduce with the mat-vec |

Start with scheme 3 + ICC(1) (certain, low-risk win); if profiling shows
`MPI_Allreduce` dominates (very likely at thousands of ranks), upgrade to
scheme 4 to cut the outer iteration count — and thus the synchronization
count — further.

## 8. Monodomain regime: why a cheap PC wins, and the fancy machinery is wasted

Everything above (overcounting, sASM, Chebyshev block solve) targets the
**elliptic** systems `Sys2 / Sys3` (pure stiffness `K`, ill-conditioned,
`kappa ~ O(h^-2)`, 100–350 iters). The **monodomain** diffusion step is a
*different* matrix and needs the *opposite* recommendation.

### 8.1 The matrix structure

With operator splitting (reaction handled by an ODE integrator, e.g.
Rush–Larsen, no linear solve), the Crank–Nicolson diffusion step gives

```
   A  =  (1/dt) M  +  (1/2) K          (theta = 1/2, M = mass, K = stiffness)
```

This is **mass-dominated** for small `dt`. Its condition number is
`kappa(A) ~ 1 + O(dt / h^2)`, i.e. bounded and small — unlike the pure `K`
whose `kappa ~ O(h^-2)` blows up under refinement. The cardioid data already
showed this: `Sys1 (Monodomain)` converges in **2–5 iters** while the elliptic
`Sys2/Sys3` need 100–350.

`asm_demo` now models it via `-dt VALUE`:
`dt <= 0` → pure `K` (elliptic, default); `dt > 0` → `(1/dt) M + (1/2) K`.

### 8.2 The comparison (P1, nx=48, 4 ranks; `iter (MPI Reductions)`)

| regime | Jacobi | Block-Jacobi/ICC(0) | sASM-O1 (sch3) | sASM+Cheby-O1 (sch4) |
|:--|:--|:--|:--|:--|
| pure-K (elliptic) | 223 (793) | 132 (520) | 104 (445) | **68 (337)** |
| dt = 1e-2         | 77 (355)  | 33 (223)  | 27 (214)  | 17 (184) |
| dt = 1e-3         | 22 (190)  | 12 (160)  | 9 (160)   | 8 (157) |
| **dt = 1e-4** (cardiac-realistic) | **6 (142)** | **6 (142)** | 4 (145) | 4 (145) |

### 8.3 What it shows

1. **As `dt -> 0` (mass-dominated), every method collapses to single-digit
   iterations and the gaps vanish.** The elliptic-regime advantage of the
   sophisticated preconditioners *evaporates*: pure-K gap Jacobi/sch4 = 223/68
   (3.3x) shrinks to 6/4 (1.5x) at `dt = 1e-4`.

2. **At `dt = 1e-4`, Jacobi (6) == Block-Jacobi (6).** A mass-dominated matrix
   is nearly diagonally dominant, so the plain diagonal is already as good as a
   block-ICC solve. The block solve buys nothing.

3. **The counter-intuitive part:** at `dt = 1e-4`, sASM/scheme 4 record *more*
   total reductions (145) than Jacobi/Block-Jacobi (142) for a single solve —
   their PCSHELL setup + Chebyshev eigenvalue estimation cost out-weighs the
   tiny `6 -> 4` iteration saving. (With fixed `dt` the matrix is constant and
   that setup amortizes over thousands of steps, but even then sASM saves only
   `2*iter+2 = 14 -> 10 = 4` reductions per solve while *adding* overlap halo
   exchange that Jacobi/Block-Jacobi do not have.)

### 8.4 Recommendation, by system

| system | matrix | kappa | iters | best PC |
|:--|:--|:--|:--|:--|
| **Monodomain (C-N)** | `(1/dt) M + (1/2) K` | `~1 + O(dt/h^2)`, small | 4–6 | **Jacobi or Block-Jacobi/ICC(0)** — zero PC communication, trivial setup; add `-ksp_type pipecg` at scale |
| Sys2 / Sys3 (`u_e` / torso) | pure `K` | `O(h^-2)`, large | 100–350 | **sASM / scheme 4** (sections 3–7), or AMG |

For the monodomain step at ≤3000 cores: **use plain Jacobi-PCG (or
Block-Jacobi/ICC(0)) — no overlap, no sASM, no Chebyshev.** The matrix is so
well-conditioned that the cheap preconditioner with zero inter-process PC
communication wins; spend the sophisticated-preconditioner budget on the
genuinely hard elliptic `Sys2/Sys3` solves instead. This matches the cardiac
HPC literature consensus that the parabolic monodomain step is "cheap" and the
elliptic bidomain step is "expensive".

Reproduce: `mpirun -n 4 ./asm_demo -fix_level 1 -scheme 0 -nx 48 -dt 1e-4
-pc_type jacobi -ksp_norm_type preconditioned -ksp_rtol 1e-6 -ksp_max_it 2000`
(append `-log_view`, read `MPI Reductions:`). Vary `-dt` and swap
`-pc_type jacobi` / `-pc_type bjacobi -sub_pc_type icc` / `-scheme 3 ...` /
`-scheme 4 ...`.

## 9. Large-scale monodomain: eliminating global synchronization

> **STATUS: NOT YET ADOPTED — design notes only.** The ideas in this section
> are theoretically grounded and the per-iteration synchronization counts were
> verified on this 4-rank laptop, but the actual *time* benefit only appears at
> large rank counts and has **not been measured at ~3000 cores**. Treat this as
> a roadmap to validate later, not a recommendation to switch to today. At small
> rank counts plain CG (section 8) is still fastest.

Section 8 said: for the mass-dominated monodomain step, use a cheap PC
(Jacobi / Block-Jacobi), iteration count is already only 4–6. The natural
follow-up: at ~3000 cores, is there anything left to improve? Yes — but the
lever changes completely.

### 9.0 Why global synchronization becomes the enemy at scale

Each timestep solves `A x = b` with `A = M/dt + (1/2) K`. One CG iteration does:

| operation | communication type | cost at ~3000 cores |
|:--|:--|:--|
| SpMV (sparse mat-vec)   | **neighbour point-to-point** halo exchange | ~1–5 us, ~flat in rank count |
| dot product (for CG alpha/beta) | **global `MPI_Allreduce`** over all ranks | ~10–30 us, *grows* with rank count (log P tree); a hard synchronization barrier |

CG does **2 dot products = 2 global syncs per iteration**. At 3000 cores those
two allreduces cost an order of magnitude more than the local SpMV and dominate
the solve. The iteration count is already 4–6 and cannot be pushed lower by a
stronger PC — so the only remaining lever is to **remove or hide the global
synchronization**. The five techniques below all attack that, from different
angles. (Per-iteration sync counts measured here on 4 ranks, monodomain
`dt=1e-4`, Jacobi: CG = 6 iters x 2 allreduce; Chebyshev = 21 iters x **0**
allreduce; Richardson = 8 iters x **0** allreduce — confirmed by running
Chebyshev to 2000 iters and seeing the `MPI Reductions` count stay flat at the
one-time setup value.)

### 9.1 Temporal extrapolation initial guess

**What it is.** Any iterative solver starts from a guess `x_0`; the iteration
count depends on the initial error `||x_0 - x*||`. In time-stepping the
solution evolves smoothly, so `V^{n+1}` is close to `V^n`. Instead of starting
from `x_0 = 0`, extrapolate from previous solutions:

```
   0th order:  x_0 = V^n
   1st order:  x_0 = 2 V^n - V^{n-1}        (linear extrapolation of the trend)
   2nd order:  x_0 = 3 V^n - 3 V^{n-1} + V^{n-2}
```

The linear guess predicts where V is heading from its recent trajectory; for a
smoothly propagating wavefront the initial error drops from O(1) to O(dt^2).

**Why it helps at scale.** Closer start -> fewer iterations (CG 6 -> 2–3). Every
iteration saved removes 2 global allreduces.

**How.** Store one extra vector `V^{n-1}`; each step set the solution to
`2 V^n - V^{n-1}` (one local AXPY, no communication) before the solve; use
`-ksp_initial_guess_nonzero true`.

**Cost.** Essentially free (one extra vector, one AXPY). Slightly less accurate
during the sharp depolarization upstroke, but still helps. **Do this first —
best effort/reward.**

### 9.2 Synchronization-free stationary solver (Chebyshev / Richardson) — the main lever

**What it is.** CG is a *Krylov* method: it computes optimal step lengths
`alpha, beta` from **inner products** of residual vectors, and those inner
products *require* a global allreduce. That is intrinsic to CG.

A *stationary* iteration uses **fixed, precomputed coefficients** instead of
optimal ones, so it needs **no inner products and no allreduce**.

(a) **Richardson (= damped preconditioned Jacobi):**
```
   x_{k+1} = x_k + omega * M_pc^{-1} (b - A x_k)
```
Each step = 1 SpMV (halo only) + 1 diagonal solve (local) + 1 AXPY (local).
**No dot products -> zero allreduce.** Converges when the spectral radius of
`(I - omega M_pc^{-1} A) < 1`; for the well-conditioned mass-dominated matrix
this is small, so a handful of steps suffice (measured: 8). Optimal
`omega = 2/(lambda_min + lambda_max)`; for a well-conditioned matrix
`omega ~ 1`, which is why `omega = 1` worked.

(b) **Chebyshev (= polynomial-accelerated Richardson):** uses a sequence of
coefficients from Chebyshev polynomials tuned to the eigenvalue interval
`[lambda_min, lambda_max]` of `M_pc^{-1}A` — the optimal stationary polynomial.
It needs the eigenvalue bounds, estimated **once** at setup (a few Lanczos /
GMRES steps that do a few allreduces *once*, not per timestep); then every
timestep runs a fixed-degree Chebyshev sweep with **zero allreduce**. It needs
more iterations than CG (21 vs 6) because the fixed polynomial cannot adapt to
the right-hand side, but it does zero global synchronization.

**Why it helps at scale (the arithmetic).** Per timestep:
```
   CG:         6 iter x (1 SpMV + 2 allreduce) =  6 SpMV + 12 allreduce
   Richardson: 8 iter x (1 SpMV + 0 allreduce) =  8 SpMV +  0 allreduce
```
With illustrative ~3000-core costs (allreduce ~15 us, SpMV ~2 us):
`CG ~ 6*2 + 12*15 = 192 us` (allreduce-dominated) vs `Richardson ~ 8*2 = 16 us`.
Richardson trades 8 cheap local SpMVs for eliminating 12 expensive global
syncs and wins by ~10x at that scale.

**How.** `-ksp_type chebyshev -ksp_chebyshev_esteig 0,0.1,0,1.1
-ksp_norm_type none -ksp_max_it <frozen count>` (the `-ksp_norm_type none`
removes even the residual-norm allreduce — run a fixed count). Or
`-ksp_type richardson -ksp_richardson_scale <omega> -ksp_norm_type none`.

**Cost / caveat.** More iterations (8–21 vs 6 -> more SpMV halo exchanges), so
this only wins when allreduce >> SpMV, i.e. at large scale — **at 4 cores CG
still wins**; the crossover is roughly hundreds-to-thousands of ranks depending
on the network. Richardson is sensitive to `omega` (too large -> divergence)
and the optimal `omega` shifts with anisotropy, so it is risky; **Chebyshev is
more robust** (only needs eigenvalue bounds, estimated once) and is the
preferred zero-sync solver. Run a *conservative* frozen iteration count, or
keep an occasional residual check.

### 9.3 Pipelined CG (if staying within CG)

**What it is.** Standard CG has a strict dependency: dot product -> **block on
allreduce** -> use result -> next vector -> dot product -> block again. During
each allreduce the rank idles while all 3000 ranks finish the reduction.
Pipelined CG (Ghysels & Vanroose 2014, PETSc `KSPPIPECG`) **reorders** the
algorithm so the global allreduce can **overlap** with the SpMV and PC work: it
issues a non-blocking `MPI_Iallreduce` and does the SpMV while the reduction is
in flight. Same CG, same iteration count, same convergence.

**Why it helps.** Does not reduce the *number* of allreduces but **hides their
latency** behind useful work, and merges the per-iteration 2 reductions into 1.
At 3000 cores where allreduce latency is large, the hiding is significant.
(Measured here: reductions 143 -> 122, with the remainder overlapped.)

**How.** `-ksp_type pipecg`.

**Cost / caveat.** A little more local flops (extra AXPYs, one extra SpMV-like
op); slightly less numerically stable than CG (the reordering amplifies
rounding) — irrelevant for the well-conditioned monodomain matrix. Needs MPI-3
non-blocking collectives (standard now).

### 9.4 Fixed-dt amortization

**What it is.** `A = M/dt + (1/2) K` depends only on `dt` and the mesh /
conductivity. If `dt` is held constant (standard for monodomain) then **A is
the same matrix every timestep**. So assemble `A` once, build the PC once
(Jacobi = store `1/diag`; Block-Jacobi/ICC = factor blocks once; Chebyshev =
estimate eigenvalue bounds once); each step only the right-hand side `b`
(containing `V^n` and the reaction current) changes, which is cheap.

**Why it helps.** Setup cost (assembly + factorization + eigenvalue estimation)
is paid once and amortized over thousands of timesteps -> per-step setup ~ 0.
This is what makes a slightly-more-expensive PC setup (ICC factorization,
Chebyshev eigenvalue estimation) affordable — it does not recur.

**How.** Build `A` and the PC/KSP objects *outside* the time loop; inside the
loop only recompute `b` and call solve. Do not rebuild them per step.

**Cost / caveat.** Valid only for fixed `dt` (adaptive `dt` forces a refactor
when `dt` changes, but that is rare). This is implementation discipline rather
than an algorithm, but it is essential — without it the per-step PC setup would
dominate.

### 9.5 Mass lumping

**What it is.** The consistently-assembled P1 mass matrix `M` is **not
diagonal** (it couples each node to its neighbours). Mass *lumping* replaces `M`
with a diagonal `M_L` whose entries are the row sums of `M` (or a nodal
quadrature) — physically, the element mass is lumped onto its vertices. Then
`A_lumped = M_L/dt + (1/2) K` with `M_L` diagonal.

**Why it helps.** The matrix becomes **even more diagonally dominant** (the
mass term, dominant for small `dt`, is now purely diagonal) -> smaller spectral
radius / condition number -> Jacobi / Richardson / Chebyshev converge in fewer
steps (fewer SpMV halo exchanges); and the mass term contributes no off-diagonal
nonzeros -> cheaper SpMV. In operator splitting, lumped mass is also what makes
the reaction step a pointwise (solve-free) update.

**Why it is standard.** Lumped mass is the norm in cardiac EP codes; the
accuracy loss is small and the same order O(h^2) as the P1 discretisation, so
it does not degrade the convergence order.

**Cost / caveat.** Slight discretisation-error change (consistent vs lumped);
some literature argues consistent mass is marginally more accurate for the
propagation speed, but lumped mass is widely accepted in practice.

### 9.6 How to combine them (the target form at ~3000 cores)

Stack the five so each removes a different piece of the global communication:

1. **Fixed-dt amortization (9.4) + mass lumping (9.5)** — foundation: matrix
   assembled once, strongest diagonal dominance, PC setup paid once.
2. **Temporal extrapolation initial guess (9.1)** — drives the *required*
   iteration count to its minimum.
3. **Chebyshev as the solver (9.2)** — drives the *per-iteration* global
   synchronization to **zero**.

Result: each timestep has almost no global communication — only the neighbour
SpMV halo exchange remains; the Krylov allreduce bottleneck is bypassed
entirely.

A more conservative variant that stays inside CG: **9.1 + 9.3 + 9.4 + 9.5**
(extrapolated guess + pipelined CG + amortization + lumping) — a large step
with lower risk (keeps CG's robustness, only hides rather than eliminates the
allreduce).

> **Again: not adopted yet.** These are design notes; the time benefit must be
> confirmed at real scale (~3000 cores), which we have not done. The
> synchronization *counts* are verified; the *wall-time* crossover is not.

## 10. How to reproduce

```bash
cd asm_bug_demo
make                              # builds asm_demo, pure_petsc_fem, pure_petsc_load
                                  # (pure_petsc_demo too, but not needed)

# Baseline (reproduces the cardioid-style iter trend)
./sweep.sh 1 0 24 4               # fix_level=1, scheme=0 (CG+BASIC)

# Fixes (pick any)
./sweep.sh 1 1 24 4               # GMRES + RAS
./sweep.sh 1 2 24 4               # BCGS + RAS
./sweep.sh 1 3 24 4               # CG + sASM (recommended, symmetric)

# scheme 4 = CG + sASM + Chebyshev(2) block solve (cheapest iter/sync)
mpirun -n 4 ./asm_demo -fix_level 1 -scheme 4 -nx 48 \
   -ksp_norm_type preconditioned -ksp_initial_guess_nonzero false \
   -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 2000 \
   -pc_type asm -pc_asm_type basic -pc_asm_overlap 2 \
   -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels 1
#   -localcheby 3   # to use degree-3 Chebyshev instead of the default 2

# Prove the synchronization reduction with a noise-free metric:
#   append -log_view and read "MPI Reductions:" and the VecTDot Count.

# Independent verification with pure PETSc P1 FEM (no MFEM)
./sweep_pure.sh 0 24 4            # baseline; reproduces the bug independently
./sweep_pure.sh 3 24 4            # sASM; reproduces the fix independently

# Bit-identity test (MFEM-METIS partition pushed into pure PETSc); works for
# scheme 0/3/4.  Seed the matrix dump with asm_demo, then load in pure PETSc:
mpirun -n 4 ./asm_demo  -fix_level 2 -scheme 4 -nx 48 \
   -ksp_norm_type preconditioned -ksp_initial_guess_nonzero false \
   -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 2000 \
   -pc_type asm -pc_asm_type basic -pc_asm_overlap 1 \
   -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels 0
mpirun -n 4 ./pure_petsc_load -nx 48 -scheme 4 ...same KSP options...
# both report identical iters, identical final pnorm
```

The Spack-installed toolchain paths are hard-coded in `asm_bug_demo/Makefile`:
- MFEM     4.9.0  at `~/spack/.../mfem-4.9.0-...`
- PETSc    3.24.4 at `~/spack/.../petsc-3.24.4-...`
- HYPRE    3.1.0
- METIS    5.1.0
- OpenMPI  5.0.9

Edit those paths in the Makefile if running on a different machine.
