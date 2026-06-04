# cardioid 三系统的统一抽象:reaction-diffusion / pure-Neumann / Laplace 的完整预条件讨论

**对象**:cardioid `hack/femheart.cpp` 每个时间步求解的三个线性系统
**方法**:把它们抽象为同一立方体、同一 P1 网格、同一 CG 求解器基底上的三个**标准模型问题**,统一讨论 `ASM(ICC)+CG` 的行为、病根、单层修复与大规模取舍。
**三份分报告**:[`REPORT_Sys1_zh.md`](./REPORT_Sys1_zh.md)(Sys1)、[`REPORT_Sys2_zh.md`](./REPORT_Sys2_zh.md)(Sys2)、[`REPORT_zh.md`](./REPORT_zh.md)(Sys3)。本报告是三者的综合与抽象,数据均引自分报告。

---

## 0. 抽象:从 cardioid 耦合求解到三个模型问题

cardioid 用**算子分裂**求解心脏电生理:反应项(ODE,用 Rush–Larsen 类积分,无需解线性系统)与扩散/椭圆项分开。**每个时间步**要解三个线性系统,它们在数学上分别退化为三类经典 PDE:

| cardioid 系统 | 物理 | 抽象模型问题 | 离散矩阵 | 边界 | 关键性质 |
|---|---|---|---|---|---|
| **Sys1 / Monodomain** | Vm 扩散步(C–N) | **reaction-diffusion** | `A = (1/Δt)M + (1/2)K` | 同 Vm | **质量主导,良态**(κ≈1.3–2) |
| **Sys2 / u_e Recovery** | 伪双域 u_e 恢复 | **pure Neumann(奇异)** | `A = K` | 全 6 面 Neumann | 椭圆,**奇异**(ker=span{1}) |
| **Sys3 / Torso** | 躯干电位 | **Laplace(混合)** | `A = K` | 1 面 Dirichlet + 5 面 Neumann | 椭圆,**非奇异** |

**统一变量**:三个问题差异的本质是**"质量主导 ↔ 刚度(椭圆)主导"**与**"边界是否提供锚点"**。本报告将证明:

> cardioid 最初的反常现象——**"增大 ASM overlap,CG 迭代次数反而上升"**——是
> **椭圆、刚度主导系统(Sys2/Sys3)** 在 **不精确 ASM(BASIC)+ICC** 下的内在性质,
> **在质量主导的 Sys1 上不发病**;其单层修复(sASM、Chebyshev 块解)对 Sys2/Sys3
> 有效,而对 Sys1 多余。

---

## 1. 全实验共用框架

**区域**:单位立方体 Ω=[0,1]³。
**网格**:结构化 nx³ 六面体,每个六面体按 MFEM `hex_to_tet[6][4]`(共用 0–6 主对角线)剖成 6 个四面体;**P1 线性元**,顶点自由度 (nx+1)³。
**求解器基底**:`CG`(`-ksp_norm_type preconditioned`,rtol=1e-6,atol=1e-12,零初值),子域 `preonly + ICC(L)`(L=0/1/2),overlap O=0/1/2。
**两套独立实现**:`asm_demo`(MFEM 4.9 装配)与 `pure_petsc_fem`/`pure_petsc_load`(纯 PETSc,无 MFEM/HYPRE);专用奇异系统驱动 `recoverue_demo`。
**软件栈**:MFEM 4.9.0 + PETSc 3.24.4 + HYPRE 3.1.0 + METIS 5.1.0 + OpenMPI 5.0.9,4 ranks,METIS 图剖分。

**四个预条件方案(`-scheme`)**:

```
scheme 0 : CG + PCASM(BASIC) + ICC                         ← cardioid baseline(复现 bug)
scheme 3 : CG + PCSHELL[ D^{-1/2} · M_BASIC^{-1} · D^{-1/2} ]   ← sASM(对称缩放,消 over-counting)
scheme 4 : CG + sASM,子域用固定阶 Chebyshev/ICC 块解         ← 同时治两个病灶
(另:Jacobi / Block-Jacobi 用于 Sys1;GAMG 作多层参照)
```

---

## 2. 贯穿三系统的统一诊断

### 2.1 理论标尺

预条件 CG 迭代数 `n_iter ≲ (1/2)√κ·ln(2/ε)`,`κ = κ(M⁻¹A)`。**"迭代为何多/少"完全归结为预条件算子的条件数 κ。** 对单层加性 Schwarz,条件数上界可分解为三个因子:

```
                  ┌ (1 + H/δ) ┐   ┌ ω_max/ω_min ┐   ┌  N_c  ┐
  κ(M_BASIC⁻¹A) ≲ │ 随 overlap │ × │  局部解谱   │ × │ 重叠区 │
                  │  δ 增大而  │   │  等价比     │   │ 重复  │
                  │  减小      │   │ (=1 当精确) │   │ 计数  │
                  └───────────┘   └─────────────┘   └───────┘
                     有利因子        病灶 A           病灶 B
```

- **病灶 A(局部解不精确)**:ICC(0) 只是 A_i⁻¹ 的廉价近似 ⇒ ω_max/ω_min ≫ 1;且 δ 增大时子块变大、ICC(0) 相对更差,该因子**随 overlap 增大**。
- **病灶 B(over-counting)**:BASIC 把重叠 DOF 修正按子域数 m_k 累加,`Σ R_iᵀR_i = D = diag(m_k) ≠ I`,把 λ_max 抬向 N_c。

**核心机理**:两个病灶**相乘**。当 A 与 B 同时存在,它们的乘积**随 δ 上升快于 (1+H/δ) 的有利下降** ⇒ κ↑ ⇒ **overlap↑ iter↑**(反常)。去掉任一因子即恢复经典"overlap↑ iter↓"。

### 2.2 为什么 Sys1 不发病、Sys2/Sys3 发病

| | 矩阵 | κ(A) | 椭圆病灶是否显现 |
|---|---|---|---|
| Sys1 | `(1/Δt)M + (1/2)K`,**质量主导** | ≈1.3–2(良态) | **否**:矩阵近对角占优,Jacobi 已足够,iter 仅 2–6,overlap 的影响淹没在噪声里 |
| Sys2/Sys3 | `K`,**刚度(椭圆)主导** | O(h⁻²)(病态) | **是**:需强预条件,A×B 乘积病灶充分暴露,overlap↑ iter↑ |

这条"质量项把病态椭圆变良态"的分界,是理解三系统差异的总钥匙(Sys1 §3.1 实测:同用 Jacobi,纯 K 要 223 步,加质量项后 dt=1e-4 只要 6 步)。

---

## 3. Sys1 — reaction-diffusion(质量主导,良态)

**问题**:`A x = b`,`A = (1/Δt)M + (1/2)K`(Crank–Nicolson θ=1/2);心脏典型参数 Δt≈0.025 ms、h≈0.2 mm 下刚度项仅占质量项 10–20%,故 **κ≈1.3–2**。固定 Δt 时 A 为常数矩阵,整个仿真只装配/预条件 setup 一次。

**结果**(nx=48,4 ranks,`迭代次数 (MPI 全局同步数)`;数据见 Sys1 报告 §3.2):

| regime | Jacobi | Block-Jacobi/ICC(0) | sASM-O1 (sch3) | sASM+Cheby-O1 (sch4) |
|:--|:--|:--|:--|:--|
| 纯 K(椭圆,对照) | 223 (793) | 132 (520) | 104 (445) | **68 (337)** |
| dt = 1e-2 | 77 (355) | 33 (223) | 27 (214) | 17 (184) |
| dt = 1e-3 | 22 (190) | 12 (160) | 9 (160) | 8 (157) |
| **dt = 1e-4(心脏真实区)** | **6 (142)** | **6 (142)** | 4 (145) | 4 (145) |

**结论**:
1. **dt 越小(质量越主导),所有方法迭代坍缩到个位数,差距消失**:纯 K 时 Jacobi 223 vs scheme4 68(3.3×,复杂预条件值钱);dt=1e-4 时 6 vs 4(仅 1.5×,且都是个位数)。椭圆区"复杂预条件大幅领先"的优势**随质量主导蒸发**。
2. **dt=1e-4 时 Jacobi(6)== Block-Jacobi(6)**:近对角占优,花哨块解零增益。
3. **反直觉**:质量主导区 sASM/scheme4 迭代少 2 步,但 PCSHELL setup + Chebyshev 特征值估计的额外 reduction + overlap halo 通信,使**单次求解的总同步反而更多**(145 vs 142)。
4. **建议**:Sys1 用 **Jacobi 或 Block-Jacobi/ICC(0)**,**不要 overlap/sASM/Chebyshev**;把复杂预条件预算全留给真正难的椭圆 Sys2/Sys3。这与心脏 HPC 文献共识一致(parabolic monodomain 步"便宜",elliptic 步才"贵")。

---

## 4. Sys2 — pure Neumann(奇异椭圆)

**问题**:`-∇·(σ∇u) = -∇·(σ_i∇V_m)`,**全 6 面齐次 Neumann** ⇒ `A=K` 半正定**奇异**,ker A=span{1}。解析参考 V_m=cos(πx)cos(πy)cos(πz)(满足 Neumann、均值零),精确解 u=V_m+const,误差以去常数相对误差度量。**奇异性处理**:`MatNullSpaceCreate(span{1})`+`MatSetNullSpace`+RHS `MatNullSpaceRemove`+解后 mean-zero 校规。

**实验 1(隔离病灶 A,nx=24,BASIC)**:

| 子域求解器 | O=0 | O=1 | O=2 | 趋势 |
|---|---:|---:|---:|---|
| ICC(0) 最不精确 | 64 | 68 | **87** | **↑ 上升(发病)** |
| ICC(2) 较精确 | 50 | 45 | 44 | 基本持平 |
| Cholesky 精确 | 36 | 26 | **22** | **↓ 下降(恢复经典理论)** |

**实验 2(消病灶 B:sASM,nx=24,L=0)**:`ASM 64→68→87 (↑)` vs `sASM 64→52→51 (↓)`;overlap≥1 省 24–41% 迭代;O=0 时 ASM≡sASM(D=I,自检)。解正确(去常数误差 3e-7…1.5e-6)。

**实验 3(弱可扩展,O=1,`iter / wall-time(s)`)**:

| 方法 | nx=24(15k) | nx=48(117k) | nx=72(389k) | iter 增长 |
|:--|---:|---:|---:|:---:|
| ASM + ICC(0) | 68 / 0.04 | 126 / 1.05 | 172 / 2.58 | 2.5× |
| sASM + ICC(0) | 52 / 0.02 | 93 / 0.72 | **131 / 0.79** | 2.5× |
| sASM + Cheby(5) | 43 / 0.03 | 73 / 0.81 | **111** / 4.72 | 2.6× |
| GAMG(多层参照) | 7 / 0.17 | 7 / 0.68 | 7 / 0.89 | **1.0×** |

**实验 4(nullspace 对照,nx=24)**:有/无 `MatSetNullSpace` **6/6 格子迭代字面一致**;唯一差别是解的均值漂移(有 NS ~1e-16,无 NS ~1e-3),真残差与去常数误差完全相同。

**结论**:
1. 病灶 A、B 在**奇异系统上同样成立**:精确解使 overlap↑ iter↓(36→22),sASM 消 over-counting 使其翻正(68→52→51,保 CG 保单层约 10 行 PCSHELL)。
2. **iter 少 ≠ time 少**:CPU 上 `sASM+ICC(0)` time 最优(nx=72:2.58→0.79s,−69%);`Cheby(5)` iter 最少(111)但 CPU time 最慢(4.72s,每步 5 次 SpMV),**其优势只在 GPU 兑现**(GPU 上 ICC 三角求解是瓶颈,Chebyshev 全 SpMV)。
3. **单层撞 H⁻² 天花板**:iter ∼ P^{1/3} ∼ nx,nx 翻 3 倍 iter 涨 2.5 倍;**唯多层(GAMG iter 恒为 7)可破**。
4. **nullspace**:兼容 RHS+零初值下 `MatSetNullSpace` 不改 iter,只清漂移;但跨时间步漂移、RHS 不兼容、AMG 粗格直接 LU 时**必须保留**。

---

## 5. Sys3 — Laplace(1 面 Dirichlet + 5 面 Neumann,非奇异)

**问题**:`-Δu=1`,x=0 面 Dirichlet u=0,其余 5 面齐次 Neumann ⇒ `A=K` 非奇异。解析解 u=x−x²/2(由对称性退化为 1D,但为 3D 精确解;max=0.5,均值 1/3)。这是 cardioid 最初报 bug 的系统。

**结果**(nx=48,4 ranks,fix_level=1;MFEM 与纯 PETSc **27/27 逐位一致**;完整数据见 Sys3 报告附录 B):

| O | L | scheme 0 (BASIC) | scheme 3 (sASM) | scheme 4 (sASM+Cheby2) |
|:-:|:-:|:-:|:-:|:-:|
| 0 | 0 | 132 | 132 | 85 |
| 1 | 0 | **148** | 104 | 68 |
| 2 | 0 | **179** | 103 | **65** |
| 2 | 2 | 90 | 67 | **42** |

- **病灶对照(BASIC 子域解)**:精确 Cholesky `52→34→30 (↓)` vs ICC(0) `132→148→179 (↑)`——精确-Cholesky 对照组**证明经典理论本身正确**,问题在 cardioid 配置同时踩中 A、B 两个假设破坏点。
- **趋势(L=0 列)**:scheme 0 `132→148→179 (↑)`;scheme 3 `132→104→103 (↓)`;scheme 4 `85→68→65 (↓)`。
- **全局同步(`-log_view`,L=0)**:O=2 时 scheme 0 = 668 reductions → scheme 4 = 329(**≈2× 削减**);`VecTDot=2·iter+2` 严格成立,证明内层 Chebyshev **零全局同步**。

**结论**:Sys3 上 sASM(消 B)恢复下降趋势;scheme 4(sASM+Chebyshev,廉价再消 A)处处最低(≈baseline 的 1/3),且 O=0 也受益(因子 A 零 overlap 时也在);两独立实现逐位一致,排除 MFEM/装配嫌疑。

---

## 6. 横向综合:三系统对照

### 6.1 一张表看懂三系统

| | Sys1 reaction-diffusion | Sys2 pure Neumann | Sys3 Laplace(混合) |
|---|---|---|---|
| 矩阵 | (1/Δt)M+(1/2)K | K(奇异) | K(非奇异) |
| κ(A) | ≈1.3–2(良态) | O(h⁻²)(病态) | O(h⁻²)(病态) |
| 边界锚点 | — | 无(ker=span{1}) | 有(x=0 Dirichlet) |
| baseline 迭代量级 | 2–6 | 60–170 | 130–180 |
| overlap↑ iter↑ 反常 | **不发病** | 发病(64→87) | 发病(132→179) |
| 病灶 A×B 适用 | 不显著 | 显著 | 显著 |
| 单层最优 PC | **Jacobi / BJ** | sASM(+Cheby on GPU) | sASM / scheme 4 |
| 奇异性处理 | 无需 | **MatSetNullSpace 必备** | 无需 |
| 大规模根治 | 消同步(外推初值/Chebyshev/pipeCG) | 多层(GAMG/POD) | 多层(GAMG/POD) |

### 6.2 统一的三条规律

1. **质量主导 → 良态 → 简单 PC 最优**(Sys1);**刚度主导 → 病态 → 需强 PC**(Sys2/Sys3)。质量项是分水岭。
2. **椭圆系统上 overlap↑ iter↑ = 病灶 A(不精确)× 病灶 B(over-counting)**;去掉任一即愈。精确-Cholesky 对照组在 Sys2、Sys3 都证明经典理论正确(36→22、52→30,均↓)。
3. **单层修复改善常数与趋势,但不改 H⁻² 阶**:sASM(消 B)+ Chebyshev(廉价消 A)是保 CG、保单层、约 10 行 PCSHELL 的最优单层组合;但 iter 仍 ∼nx 增长,**跨大规模唯多层方法(GAMG iter 恒 7、或已有 POD 两层)可让 iter 与网格解耦**。

### 6.3 iter 与 time 的关系(三系统统一)

`time = iter × 单步成本 + setup`。改 iter 必须同时核单步成本与硬件:
- Sys1:iter 已个位数,瓶颈是 CG 每步 2 次 `MPI_Allreduce` → 转向**消同步**(时间外推初值、Chebyshev 定常解法零 allreduce、pipeCG、固定 Δt 摊销、mass lumping)。
- Sys2/Sys3:sASM 每步只多两次 O(n) 点乘 ⇒ **iter 省即 time 省**(CPU);Chebyshev iter 更少但 CPU 每步贵(5× SpMV),**time 优势在 GPU**。
- 时间的**机器无关**度量用 `-log_view` 的全局 reduction 计数(wall-time 在负载机器上不可信)。

---

## 7. 对 cardioid 的统一工程建议(按系统)

| 系统 | 单层框架内 | GPU 部署 | 超大规模根治 | 奇异性 |
|---|---|---|---|---|
| **Sys1 Monodomain** | **Jacobi / Block-Jacobi**(零 PC 通信、setup 平凡);不要 overlap/sASM | 同左 + mass lumping | 消同步:时间外推初值 + Chebyshev 定常解法 + pipeCG + 固定 Δt 摊销 | 无需 |
| **Sys2 u_e Recovery** | baseline → **sASM**(≈10 行 PCSHELL,省 24–41% iter) | sASM + **Chebyshev(K)** 块解(消三角求解瓶颈) | **GAMG / BoomerAMG / HPDDM 或已有 POD 两层**(iter→O(1)) | **保留 `MatSetNullSpace`**(切 AMG 时必需) |
| **Sys3 Torso** | baseline → **sASM**;或 **scheme 4**(sASM+Cheby,iter≈1/3) | scheme 4 | 同 Sys2 | 无需 |

**一句话总纲**:**把预条件预算按系统难度分配**——Sys1 用最便宜的 Jacobi-PCG 并主攻"消同步";Sys2/Sys3 用 sASM(必要时 + Chebyshev 块解)修掉单层椭圆病灶,并保留通往多层方法(GAMG / POD)的接口以根治 H⁻² 天花板;奇异的 Sys2 始终保留 `MatSetNullSpace`。

---

## 附录:数据来源与复现

| 系统 | 分报告 | 驱动程序 | 主网格 |
|---|---|---|---|
| Sys1 reaction-diffusion | `REPORT_Sys1_zh.md` | `asm_demo`(`-dt` 开关) | nx=48 |
| Sys2 pure Neumann | `REPORT_Sys2_zh.md` | `recoverue_demo` | nx=24/48/72 |
| Sys3 Laplace | `REPORT_zh.md` | `asm_demo` / `pure_petsc_load` | nx=48 |

各分报告含完整逐 case 数据表、理论分析与复现命令。统一工具链(Spack 安装,路径写死在 `asm_bug_demo/Makefile`):MFEM 4.9.0 / PETSc 3.24.4 / HYPRE 3.1.0 / METIS 5.1.0 / OpenMPI 5.0.9。所有命令统一附 `-ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 2000`。
