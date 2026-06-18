# Preliminary dimensional results (validated NumPy instrument)

Instrument: `asm_spectral.py` — variable-coefficient `-div(a grad u)=f` on `[0,1]^d`,
1 Dirichlet face + Neumann elsewhere, S boxes/axis, BASIC vs sASM additive Schwarz,
exact (LU) vs IC(0) local solves. Spectrum is the **exact** eigenspectrum of M⁻¹A
(via the similarity transform S = Lᵀ M⁻¹ L, A = LLᵀ), validated to machine precision
against brute-force `eig(M⁻¹A)` (max imag part 0–4e-16).

## Constant-coefficient, S=4 boxes/axis (sizes: 1D M=129, 2D M=33, 3D M=13)

```
dim local ov   | iter  Nhat  omega lam_max lam_min   kappa
----------------------------------------------------------------------
1   exact 0    |    7     1  1.000   1.997  0.0031   640.8
1   exact 1    |    8     2  1.000   2.000  0.0093   214.5
1   exact 2    |    8     2  1.000   2.000  0.0155   129.0
1   exact 3    |    8     2  1.000   2.000  0.0216    92.4
1   ic0   *    |  (identical to exact — IC(0) is exact on a tridiagonal block)

2   exact 0    |   45     1  1.000   1.994  0.0064   311.0
2   exact 1    |   30     4  1.000   4.000  0.0208   191.9
2   exact 2    |   24     4  1.000   4.000  0.0373   107.2
2   exact 3    |   21     4  1.000   4.000  0.0564    70.9
2   ic0   0    |   62     1  1.356   1.658  0.0029   575.0
2   ic0   1    |   70     4  1.357   4.301  0.0043  1002.3
2   ic0   2    |   74     4  1.357   4.542  0.0054   848.2
2   ic0   3    |   69     4  1.357   4.662  0.0067   698.0

3   exact 0    |   39     1  1.000   1.988  0.0116   171.3
3   exact 1    |   29     8  1.000   8.000  0.0544   147.2
3   exact 2    |   38    27  1.000  27.000  0.1988   135.8
3   exact 3    |   39    27  1.000  32.452  0.5429    59.8
3   ic0   0    |   41     1  1.185   1.689  0.0083   204.0
3   ic0   1    |   54     8  1.188   8.314  0.0236   353.0
3   ic0   2    |   64    27  1.188  27.131  0.0568   477.6
3   ic0   3    |   59    27  1.188  32.481  0.1177   275.9
```

## Findings so far

1. **The anomaly (iter ↑ with overlap) requires INEXACT local solves.** With *exact*
   local solves, overlap reduces κ and iterations in **every** dimension 1/2/3 —
   classical behaviour — even though the geometric over-count makes λmax explode.

2. **λmax(BASIC, exact local) = N̂ exactly**, where N̂ = max multiplicity = **2^d** at
   the first corner overlap (1D→2, 2D→4, 3D→8), then 3^d=27 in 3D at O=2. The
   geometric over-count grows *exponentially* with dimension. (Earlier memory note
   confirmed cleanly.)

3. **N̂ alone does NOT cause the anomaly.** In 3D-exact, λmax jumps 2→8→27 yet κ still
   *drops* 171→60, because λmin improves fast enough (0.012→0.54). Overlap's classical
   benefit (better stable-decomposition constant → larger λmin) compensates the
   over-count — **when local solves are exact.**

4. **Inexactness does TWO dimension-dependent damages:**
   (a) inflates λmax by ω = max_i λmax(M_i⁻¹A_i): ω=1 in 1D (IC(0) exact), >1 in 2D/3D.
   (b) *cripples the λmin improvement*: e.g. 3D O=2 λmin = 0.199 (exact) vs 0.057 (IC0).
   So the overlap no longer buys the full stable-decomposition gain. κ then rises:
   2D 575→1002, 3D 204→478.

5. **Dimensional ordering of the anomaly strength (iter at O=0 → worst):**
   1D none (7→8); 2D +19% (62→74); 3D +56% (41→64). **Confirms H2 direction.**

CAVEAT: subdomain sizes are NOT yet controlled across dimensions (1D H≈32/axis,
2D H≈8, 3D H≈3), so ω is not yet a clean function of d here (2D ω=1.36 > 3D ω=1.19,
a sizing artifact). The controlled study (fix per-axis H, and separately fix total
local DOFs) is the next step to isolate ω(d) from block size — pending.
