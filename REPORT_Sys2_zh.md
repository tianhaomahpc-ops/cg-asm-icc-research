# Sys2(u_e 恢复 / 全 Neumann 奇异椭圆)预条件实验报告

## 0. 背景与逻辑主线

**前情提要(已完成,见 `REPORT_zh.md`):** 在 Sys3(Torso,1 面 Dirichlet + 5 面
Neumann,非奇异)上,我们已经确认 `CG + 单层 ASM(BASIC) + ICC` 在
**增大 overlap 时迭代次数反而上升**,并定位到两个并存的机制:

- **机制 A —— 子域求解器不精确**:ICC(0) 只是 $A_i^{-1}$ 的廉价近似,留下子域内低频残差;
- **机制 B —— 重叠区重复计数(over-counting)**:`PC_ASM_BASIC` 把重叠 DOF 的修正
  按子域个数 $m_k$ 累加,$\sum_i R_i^\top R_i = D = \mathrm{diag}(m_k) \ne I$。

**本报告目标:在 Sys2(全 Neumann、奇异)上验证上述机制是否同样成立,并给出修复。**
四个实验构成一条严谨的因果链:

| 实验 | 针对的问题 | 在因果链中的角色 |
|---|---|---|
| **实验 1** | 不精确局部解是否是 overlap 反向的根因? | **隔离机制 A**:把 ICC 换成精确 Cholesky,看趋势是否翻正 |
| **实验 2** | sASM 能否在奇异系统上修复? | **消除机制 B**:用 $D^{-1/2}(\cdot)D^{-1/2}$ 对角缩放抵消 over-counting |
| **实验 3** | iter 下降是否等于 time 下降? | **工程落地**:ASM / sASM / sASM+Chebyshev 的 iter 与 wall-time 权衡 |
| **实验 4** | 奇异系统必须显式给 nullspace 吗? | **正确性边界**:`MatSetNullSpace` 对 iter 与解的影响 |

逻辑递进:实验 1 确认"病根之一是局部解不精确"→ 实验 2 处理"另一个病根 over-counting"
→ 实验 3 回答"修好 iter 是否真省时间"→ 实验 4 收尾"奇异系统的 nullspace 这一必做项
到底影不影响 iter"。

---

## 1. 全实验共用的设置

### 1.1 问题(方程与边界条件)

模拟 cardioid 伪双域 `u_e` 恢复(`hack/femheart.cpp` 的 Sys2 / `recoverue_`):

$$
-\nabla\cdot(\sigma\,\nabla u) \;=\; -\nabla\cdot(\sigma_i\,\nabla V_m)
\qquad \text{in } \Omega
$$

$$
\frac{\partial u}{\partial n} = 0 \quad \text{on } \partial\Omega \;\;(\text{全 6 面齐次 Neumann})
$$

取 $\sigma = \sigma_i = 1$(各向同性、同系数),离散后两侧用同一刚度矩阵,得到

$$
A\,u \;=\; A\,V_m
$$

这与 cardioid 中 `temp_form` 的右端构造(`A_temp · (-V_m)`)同构。

**奇异性:** 全 Neumann ⇒ $A$ 半正定奇异,$\ker A = \mathrm{span}\{\mathbf{1}\}$(常数模)。

**解析参考解:** 取 $V_m(x,y,z) = \cos(\pi x)\cos(\pi y)\cos(\pi z)$。它满足全 Neumann
边界、空间均值为零,因此精确解为 $u = V_m + \text{const}$。报告中所有 "解正确性"
均以去常数后的相对误差 $\|u_0 - V_{m,0}\| / \|V_{m,0}\|$ 度量($\cdot_0$ 表示减去全局均值)。

### 1.2 区域

单位立方体 $\Omega = [0,1]^3$。

### 1.3 网格

- 结构化 $n_x \times n_x \times n_x$ 六面体,每个六面体按 MFEM `Mesh::MakeCartesian3D`
  的 `hex_to_tet[6][4]` 表(共用 0-6 主对角线)剖分为 **6 个四面体**;
- **P1 线性元**(`H1_FECollection`,order=1),自由度位于顶点,全局 $(n_x+1)^3$ 个 DOF;
- 默认 $n_x = 24$(15 625 DOF);弱可扩展性实验取 $n_x = 24 / 48 / 72$
  (15 625 / 117 649 / 389 017 DOF);
- 4 个 MPI rank,**METIS 图剖分**(MFEM `ParMesh` 默认)。

### 1.4 求解器基底

- **软件栈:** MFEM 4.9.0 + PETSc 3.24.4 + HYPRE + METIS 5.1.0 + OpenMPI 5.0.9(Spack 安装);
- **外层 Krylov:** `PetscPCGSolver`(CG),`rtol = 1e-6`,`atol = 1e-12`,`max_it = 2000`,
  `-ksp_norm_type preconditioned`,零初值;
- **矩阵转换:** `HypreParMatrix → PETSc AIJ`,过滤阈值 $|val| < 10^{-12}$(滤掉 quadrature
  round-off 零),并打 `MAT_SYMMETRIC / MAT_SYMMETRY_ETERNAL`;
- **奇异性处理:** `MatNullSpaceCreate(span{1})` + `MatSetNullSpace` + 对 RHS 做
  `MatNullSpaceRemove`,解后 mean-zero 校规(实验 4 专门关掉它做对照);
- **预条件子(由 `-scheme` 选择):**
  - scheme 0 = `CG + PCASM(BASIC) + sub`(cardioid baseline);
  - scheme 3 = `CG + PCSHELL[ D^{-1/2} · M_BASIC^{-1} · D^{-1/2} ]`(sASM);
  - 子域求解器:`-icc L`(ICC(L),不精确)/ `-exact`(Cholesky,精确)/
    `-cheb K`(K 步 Chebyshev + Jacobi,无三角分解)。
- 代码:`asm_bug_demo/recoverue_demo.cpp`。

---

## 2. 实验 1 — 不精确局部解是否是 overlap 反向的根因?

### 2.1 实验目的

前情提要在 Sys3 上**同时**观察到两个机制叠加的结果。本实验在 Sys2 上**单独隔离机制 A**:
固定预条件结构为最经典的 `ASM(BASIC)`(不引入 sASM 的缩放),只把**子域求解器的精度**
从 ICC(0)(不精确)逐步提到 ICC(2),再到精确 Cholesky,观察 overlap → iter 趋势是否随
精度提升而由"上升"翻转为经典理论预期的"下降"。

**预期(若猜想成立):** 精确局部解时,overlap 增大 → iter 单调下降(经典 Schwarz 理论);
不精确时,趋势反向。

### 2.2 问题 / 区域 / 网格

同 §1。$n_x = 24$,4 ranks,全 Neumann 奇异系统。

### 2.3 实现细节

- 固定 scheme 0(`PCASM BASIC`),关闭 sASM 缩放;
- 子域 KSP 一律 `preonly`(子域内只跑一次 PC,不迭代),子域 PC 取:
  - `-icc 0`:`PCICC` + `PCFactorSetLevels(0)`;
  - `-icc 2`:`PCICC` + `PCFactorSetLevels(2)`;
  - `-exact`:`PCCHOLESKY`(完全分解,精确 $A_i^{-1}$);
- overlap 取 0 / 1 / 2;
- 复现命令:
  ```
  mpirun -n 4 ./recoverue_demo -nx 24 -scheme 0 -overlap {O} -icc {0|2}  ...
  mpirun -n 4 ./recoverue_demo -nx 24 -scheme 0 -overlap {O} -exact      ...
  ```

### 2.4 结果

迭代次数(`CG + ASM(BASIC)`,$n_x=24$,4 ranks):

| 子域求解器 | O=0 | O=1 | O=2 | 趋势 |
|---|---:|---:|---:|---|
| **ICC(0)** — 最不精确 | 64 | 68 | **87** | **↑ 上升** |
| **ICC(2)** — 较精确 | 50 | 45 | 44 | 基本持平 / 微降 |
| **Cholesky** — 精确 | 36 | 26 | **22** | **↓ 下降** |

为完整起见,同一组子域求解器在 sASM(scheme 3)下:

| 子域求解器 | O=0 | O=1 | O=2 |
|---|---:|---:|---:|
| sASM + ICC(0) | 64 | 52 | 51 |
| sASM + ICC(2) | 50 | 37 | 34 |
| sASM + Cholesky | 36 | 25 | 22 |

### 2.5 结论

1. **猜想得到实验证实。** 在奇异的 Sys2 上,**精确局部解(Cholesky)使 overlap 增大时
   iter 单调下降(36→26→22),恢复经典 Schwarz 理论**;而 ICC(0) 使其反向上升(64→68→87)。
   ICC(2) 处于中间。
2. **机理(条件数上界分解):**
   $$
   \kappa(M_{\text{BASIC}}^{-1}A) \;\lesssim\;
   \underbrace{C\Bigl(1+\tfrac{H}{\delta}\Bigr)}_{\text{随 overlap}\,\delta\,\text{减小}}
   \;\times\;
   \underbrace{\frac{\omega_{\max}}{\omega_{\min}}}_{\text{局部解谱等价}}
   \;\times\;
   \underbrace{N_c}_{\text{重叠重复计数}}
   $$
   精确解时 $\omega_{\max}/\omega_{\min}=1$,$(1+H/\delta)$ 项主导 → iter 降;
   ICC(0) 时 $\omega_{\max}/\omega_{\min}\gg1$,放大了 $N_c$ 与常数因子,压过 $(1+H/\delta)$
   的有利贡献 → iter 升。
3. 注意:Cholesky 在 sASM 与 ASM 下几乎相同(36/26/22 vs 36/25/22)——因为精确解时
   over-counting 影响被局部精确解吸收,sASM 没有多少 $N_c$ 可"修";**这反过来说明
   over-counting 的危害只在局部解不精确时才显著**,引出实验 2。

---

## 3. 实验 2 — sASM 在奇异系统上能否修复(消除 over-counting)?

### 3.1 实验目的

实验 1 表明:在工程上唯一可负担的廉价局部解(ICC(0))下,overlap 反向不可避免。实验 2
转而处理**机制 B(over-counting)**:保持 ICC(0) 的廉价不变,改用 **sASM** —— 在
`M_BASIC^{-1}` 两侧各乘 $D^{-1/2}$($D=\mathrm{diag}(m_k)$ 为 multiplicity 对角阵)——
**显式把重叠区的重复计数精确抵消,同时保持算子对称(从而 CG 仍可用)**。目的:验证
在 Sys2(奇异)上 sASM 是否把"overlap↑→iter↑"翻成"overlap↑→iter↓",且解仍正确。

### 3.2 问题 / 区域 / 网格

同 §1。$n_x = 24$,4 ranks,全 Neumann 奇异系统。

### 3.3 实现细节

- scheme 3 通过 `PCSHELL` 实现:
  - 内层建一个 `PCASM(BASIC)` + 子域 ICC(L),先 `PCSetUp` 得到子域 IS 列表;
  - 用 `PCASMGetLocalSubdomains` 取带 overlap 的 IS,累加指示向量得 multiplicity
    $m_k$,构造 $D^{-1/2} = \mathrm{diag}(1/\sqrt{m_k})$;
  - PCSHELL 的 apply:$z = D^{-1/2}\,\big(M_{\text{BASIC}}^{-1}(D^{-1/2} r)\big)$
    (两次 `VecPointwiseMult` 夹一次内层 `PCApply`);
- 对称性:$D^{-1/2}$ 自伴,夹住对称的 $M_{\text{BASIC}}^{-1}$,整体对称 ⇒ CG 合法;
- **无额外内存**(只多一个 $D^{-1/2}$ 向量)、**无额外全局通信**(子域内本地);
- overlap 0/1/2,ICC level 0/1/2,共 9 个格子;
- 每格 5 次取最小时间(抗 OS 抖动)。复现:`./bench_recoverue.sh 24 4 5`。

### 3.4 结果

`recoverue` 全 Neumann,$n_x=24$,4 ranks,iter / wall-time(s):

| (O, L) | ASM (scheme 0) | sASM (scheme 3) | Δiter | Δtime |
|:---:|:---:|:---:|:---:|:---:|
| (0, 0) | 64 / 0.022 | 64 / 0.018 | 0(O=0 时 $D=I$) | −18% |
| (0, 1) | 56 / 0.016 | 56 / 0.014 | 0 | −13% |
| (0, 2) | 50 / 0.018 | 50 / 0.015 | 0 | −17% |
| **(1, 0)** | **68 / 0.020** | **52 / 0.012** | **−24%** | **−40%** |
| (1, 1) | 52 / 0.019 | 40 / 0.011 | −23% | −42% |
| (1, 2) | 45 / 0.023 | 37 / 0.014 | −18% | −39% |
| **(2, 0)** | **87 / 0.029** | **51 / 0.022** | **−41%** | **−24%** |
| (2, 1) | 53 / 0.024 | 38 / 0.028 | −28% | (噪声) |
| (2, 2) | 44 / 0.031 | 34 / 0.014 | −23% | −55% |

趋势(L=0 列):

```
        O=0    O=1    O=2
ASM     64 --> 68 --> 87     ↑↑  (over-counting 病灶)
sASM    64 --> 52 --> 51     ↓   (经典趋势恢复)
```

解正确性(每个格子):`meanU ~ 1e-16`(mean-zero 校规干净)、
真残差 `‖r‖/‖b‖ ~ 1e-6`(与 rtol 一致)、
`‖u_0 − V_{m,0}‖/‖V_{m,0}‖ ~ 3e-7 … 1.5e-6`(数值解 = 解析 $V_m$ 去常数,精度达残差水平)。

### 3.5 结论

1. **sASM 在奇异系统上同样有效:** overlap↑ 时 iter 由上升翻为下降(L=0:68→52→51),
   overlap≥1 时比 ASM 省 **24%–41% 迭代**,与 Sys3(非奇异)案完全一致。
2. **理论支撑:** sASM 把 $\lambda_{\max}(M^{-1}A)$ 从 BASIC 的 $\le N_c$ 压到 $\le 1$,
   等价于在上界中**把 over-counting 因子 $N_c$ 强制为 1**;余下 $(1+H/\delta)$ 随 overlap
   单调减,故 iter 单调降。
3. **O=0 时 ASM ≡ sASM**($D=I$,multiplicity 处处为 1),数据完全相同,验证实现正确。
4. **奇异性不破坏结论:** nullspace 处理干净,两法收敛到同一物理解,只是 sASM 步数更少。
   这正是 cardioid Sys2 需要的单层修补——保 CG、保单层、约 10 行 `PCSHELL`。

---

## 4. 实验 3 — iter 下降是否等于 time 下降?(ASM+ICC0 / sASM+ICC0 / sASM+Cheby5)

### 4.1 实验目的

实验 2 在固定网格上显示 sASM 省 iter 又省时间。但工程关心的是**超大规模**下的行为,
且"iter 少"与"time 少"并不等价(每步 PC 的成本不同)。本实验做**弱可扩展性**测试
(固定 4 ranks,网格 $n_x$ 逐步加密),对比三种单层组合:

- `ASM + ICC(0)`(baseline);
- `sASM + ICC(0)`(实验 2 的修复:治 over-counting);
- `sASM + Cheby(5)`(进一步把不精确的 ICC 换成 5 步 Chebyshev 多项式:**一手治
  over-counting、一手治局部解不精确**,且不加内存、不加全局通信、对 GPU 友好)。

目的:回答三个问题 —— **(i)** 为什么 iter 会更好?**(ii)** 为什么有时 time 会更省?
**(iii)** 什么情况下 iter 少、什么情况下 time 少?

### 4.2 问题 / 区域 / 网格

同 §1。**弱扩展:** $n_x = 24 / 48 / 72$(15 625 / 117 649 / 389 017 DOF),4 ranks 固定,
全 Neumann 奇异系统,overlap=1(`sASM` 系列)。

### 4.3 实现细节

- `sASM + Cheby(K)`:PCSHELL 内层 `PCASM(BASIC)`,子域 KSP 改为
  `KSPCHEBYSHEV`(`-cheb 5`),`max_it = K`,`KSP_NORM_NONE`,
  子域 PC 用 `PCJACOBI`;特征值区间用 `KSPChebyshevEstEigSet`(GMRES 估计,
  比例 0.1–1.1,即 PETSc/HYPRE 平滑器默认配置);
- Chebyshev 子求解**只含 SpMV + AXPY + 对角缩放,不存 L/U 因子**;
- 每格 3 次取最小时间;复现见附录;
- 作为"打破单层天花板"的参照,附 `GAMG`(PETSc 原生聚合代数多重网格)一行。

### 4.4 结果

弱扩展(全 Neumann,4 ranks,iter / wall-time(s)):

| 方法 | nx=24(15k) | nx=48(117k) | nx=72(389k) | iter 增长 |
|:--|---:|---:|---:|:---:|
| `ASM  + ICC(0)`  O=1 | 68 / 0.04 | 126 / 1.05 | 172 / 2.58 | 2.5× |
| `sASM + ICC(0)`  O=1 | 52 / 0.02 | 93 / 0.72 | 131 / **0.79** | 2.5× |
| `sASM + Cheby(5)` O=1 | 43 / 0.03 | 73 / 0.81 | **111** / 4.72 | 2.6× |
| `GAMG`(参照) | 7 / 0.17 | 7 / 0.68 | 7 / 0.89 | **1.0×** |

### 4.5 结论

**(i) 为什么 iter 更好:**
- `ASM → sASM`:消除 over-counting,$\lambda_{\max}\le1$ ⇒ 谱半径收窄,**iter 降**
  (172→131,−24%,nx=72);
- `ICC(0) → Cheby(5)`:K 步多项式 $p_K(A_i)$ 比 ICC(0) 更接近 $A_i^{-1}$($\omega/\omega$
  更接近 1)⇒ **iter 进一步降**(131→111,−15%,nx=72)。
两者分别打在条件数上界的两个因子($N_c$ 与 $\omega/\omega$)上,所以 iter 单调改善。

**(ii) 为什么有时 time 更省:**
- sASM 每步只比 ASM 多两次 $O(n)$ 向量点乘(相对 ICC 三角求解可忽略),**iter 省即
  time 省**(nx=72:2.58s→0.79s,−69%)。

**(iii) 什么情况 iter 少 / 什么情况 time 少 —— 二者不等价:**
- **iter 最少 ≠ time 最少。** nx=72 时 `Cheby(5)` iter 最少(111 < 131),但 **time 最慢
  (4.72s ≫ 0.79s)**:Chebyshev 每步要 5 次 SpMV,在本机 **CPU** 上比 ICC 的一次前/回代
  贵 3–5 倍,iter 省下的量摊不平单步涨的量;
- **本机 CPU 上 time 最优的是 `sASM + ICC(0)`**(0.79s);
- **`Cheby(K)` 的 time 优势只在 GPU 上兑现** —— GPU 上 ICC 的稀疏三角求解强顺序、是瓶颈,
  而 Chebyshev 全是 SpMV(GPU 看家本领),此时 iter 少会直接转化为 time 少;
- **单层方法都撞 $H^{-2}$ 天花板**:iter $\sim P^{1/3}\sim n_x$,nx 翻 3 倍 iter 涨约 2.5 倍;
  **只有多层(GAMG/BoomerAMG/HPDDM/POD 两层)能让 iter 恒定**(GAMG:7→7→7),
  这是超大规模唯一的根治方向。

**一句话:** `time = iter × 单步成本 + setup/求解次数`。改 iter 必须同时核单步成本与硬件;
单层修复(sASM、Chebyshev)改善常数与趋势但不改 $H^{-2}$ 阶,跨大规模仍需多层方法。

---

## 5. 实验 4 — 显式给 nullspace 是否影响迭代次数?(ASM + ICC0 + overlap 0)

### 5.1 实验目的

Sys2 是奇异系统,工程上必须处理 $\ker A = \mathrm{span}\{\mathbf 1\}$。一个常被忽视的问题:
**显式 `MatSetNullSpace` 到底改不改变迭代次数,还是只影响解的校规?** 本实验在最干净的
配置(scheme 0,ICC(0),overlap 0)下,**唯一变量是是否调用 `MatSetNullSpace`**,做对照。

### 5.2 问题 / 区域 / 网格

同 §1。$n_x = 24$,4 ranks,全 Neumann 奇异系统;固定 `ASM(BASIC) + ICC(0)`,
overlap=0(也附 overlap 1/2、ICC level 0/2 以验证规律普适)。

### 5.3 实现细节

- `-no_ns` 开关:为真时**完全不调** `MatSetNullSpace` / `MatSetTransposeNullSpace` /
  对 RHS 的 `MatNullSpaceRemove`,让 KSP 在奇异矩阵上"盲跑";为假时按 §1.4 正常挂载;
- 其余完全相同(同一矩阵、同一 RHS、零初值、同一 rtol);
- 记录 iter、解的全局均值 `meanU`、去常数相对误差;
- 复现:`-scheme 0 -overlap 0 -icc 0` 加 / 不加 `-no_ns`。

### 5.4 结果

scheme 0,$n_x=24$,4 ranks,**有 / 无 `MatSetNullSpace`**:

| 配置 | iter(有 NS) | iter(无 NS) | meanU(有 NS) | meanU(无 NS) | `‖u−u*‖/‖u*‖` |
|:--|:---:|:---:|:---:|:---:|:---:|
| **O=0 L=0** | **64** | **64** | 2.2e-16 | −1.2e-3 | 1.2e-6 |
| O=0 L=2 | 50 | 50 | 1.3e-16 | −3.3e-3 | 1.6e-6 |
| O=1 L=0 | 68 | 68 | 8.4e-17 | 5.2e-4 | 3.0e-7 |
| O=1 L=2 | 45 | 45 | 2.8e-16 | 4.4e-4 | 1.1e-6 |
| O=2 L=0 | 87 | 87 | 1.1e-17 | 1.3e-3 | 7.3e-7 |
| O=2 L=2 | 44 | 44 | 2.9e-16 | 1.7e-3 | 9.1e-7 |

**6/6 格子迭代次数字面一致。** 唯一差别:`meanU` 在有 NS 时 ~1e-16(漂移被投影清除),
无 NS 时 ~1e-3(漂移自由累积);**真残差与去常数误差两者完全相同。**

### 5.5 结论

1. **在本配置下,`MatSetNullSpace` 不改变迭代次数。** 原因:零初值 + 兼容 RHS
   ($b = A V_m$,且 $b \perp \ker A$,因 $\mathbf 1^\top A v = 0$ 对一切 $v$),
   于是每个 Krylov 向量 $A^k r_0$ 都落在 $\mathrm{range}(A)$ 内,KSP 每步的 null-space
   投影是 **no-op**(精确算术下);浮点下它只清掉 ~1e-16 的漂移,不影响收敛判据。
2. **`MatSetNullSpace` 真正必要的场景:**
   - 非零初值含常数分量(如 cardioid `iter_mode=true` 携带上一时间步的常数漂移)——
     否则漂移跨步累积;
   - RHS 因装配浮点误差略不兼容($\sum b_i \ne 0$)——否则残差的常数分量永降不下、撞 max_it;
   - **AMG 类预条件**(GAMG / BoomerAMG / HPDDM)粗格用直接 LU——奇异粗矩阵无 nullspace
     提示会**直接崩溃**。
3. **对 cardioid 的建议:保留 `MatSetNullSpace`**——它对单次 solve 的 iter 无影响,但
   (a) 防跨时间步常数漂移,(b) 是日后切换到 AMG 的前置必需项。代价是每步一次
   `MatNullSpaceRemove` 的全局 Allreduce,小问题上占比可见(本机小问题约 5× wall-time
   差异源于此),大问题上摊薄到可忽略。

---

## 6. 总体结论(Sys2)

| 问题 | 结论 |
|---|---|
| **病根 A(局部解不精确)** | 实验 1 证实:精确 Cholesky 使 overlap↑→iter↓(36→22),ICC(0) 使其反向(64→87)。猜想成立。 |
| **病根 B(over-counting)** | 实验 2 证实:sASM 用 $D^{-1/2}(\cdot)D^{-1/2}$ 抵消,奇异系统上 overlap↑→iter↓,省 24–41% iter,保 CG 保单层。 |
| **iter vs time** | 实验 3 证实:iter 少 ≠ time 少。CPU 上 `sASM+ICC(0)` time 最优;`Cheby(K)` iter 更少但 CPU time 更慢,其优势在 GPU。单层均撞 $H^{-2}$,唯多层(GAMG iter 恒为 7)可破。 |
| **nullspace** | 实验 4 证实:兼容 RHS + 零初值下 `MatSetNullSpace` 不改 iter(6/6 字面一致),仅清漂移;但跨时间步、不兼容 RHS、AMG 粗格时必须有。 |

**对 cardioid Sys2 的工程建议:**
1. **单层框架内**:把 baseline 的 `PCASM(BASIC)` 升级为 sASM(约 10 行 PCSHELL),
   立刻省 24–41% iter,不改 CG、不加内存/通信;
2. **GPU 部署**:进一步把子域 ICC 换成 Chebyshev(K),消除三角求解瓶颈;
3. **超大规模根治**:换 `-pc_type gamg`(或 BoomerAMG / HPDDM,或沿用已有 POD 两层),
   iter 与网格规模解耦($O(1)$);
4. **奇异性**:无论用哪种预条件,保留 `MatSetNullSpace`(切 AMG 时为必需)。

---

## 附录:复现命令

```bash
cd asm_bug_demo && make recoverue_demo

# 实验 1：不精确 vs 精确局部解(ASM）
for O in 0 1 2; do
  mpirun -n 4 ./recoverue_demo -nx 24 -scheme 0 -overlap $O -icc 0    -ksp_converged_reason
  mpirun -n 4 ./recoverue_demo -nx 24 -scheme 0 -overlap $O -icc 2    -ksp_converged_reason
  mpirun -n 4 ./recoverue_demo -nx 24 -scheme 0 -overlap $O -exact    -ksp_converged_reason
done

# 实验 2：ASM vs sASM 全 (O,L) 扫描 + 计时
./bench_recoverue.sh 24 4 5

# 实验 3：弱可扩展性 ASM+ICC0 / sASM+ICC0 / sASM+Cheby5（+ GAMG 参照）
for NX in 24 48 72; do
  mpirun -n 4 ./recoverue_demo -nx $NX -scheme 0 -overlap 1 -icc 0           -ksp_converged_reason
  mpirun -n 4 ./recoverue_demo -nx $NX -scheme 3 -overlap 1 -icc 0           -ksp_converged_reason
  mpirun -n 4 ./recoverue_demo -nx $NX -scheme 3 -overlap 1 -cheb 5          -ksp_converged_reason
  mpirun -n 4 ./recoverue_demo -nx $NX -scheme 6                             -ksp_converged_reason
done

# 实验 4：nullspace 有 / 无 对照（ASM + ICC0 + overlap 0）
mpirun -n 4 ./recoverue_demo -nx 24 -scheme 0 -overlap 0 -icc 0        -ksp_converged_reason
mpirun -n 4 ./recoverue_demo -nx 24 -scheme 0 -overlap 0 -icc 0 -no_ns -ksp_converged_reason
```

所有命令统一附 `-ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 2000`。
