# ASM + CG 在椭圆问题上 "overlap 增大、迭代反增" 的诊断与单层改进
## 实验报告

---

## 0. 背景与逻辑主线

**起源问题.** 在心脏电生理代码 cardioid 的 `hack/femheart.cpp` 不开 POD 的
baseline 路径中,`Sys2`(u_e 恢复)与 `Sys3`(torso)用
`CG + 加性 Schwarz(BASIC) + 不完全 Cholesky(ICC)` 求解时观察到:**增大子域
overlap,CG 迭代次数不降反升**(`Sys3` 甚至撞 `max_it=200` 不收敛)。这违反
经典 Schwarz 理论"overlap↑ ⇒ 条件数↓ ⇒ 迭代↓"的预期。

**本报告的逻辑链**(每个实验的结论是下一个实验的前提):

```
   复现现象 (asm_demo 上 BASIC+ICC+CG, overlap↑ iter↑)
        │
   实验1: 这是"实现/装配的人为产物"吗? ── 两独立实现共享 METIS 分区做 bit-identical 对照
        │   结论: 不是。现象是方法内在的,且两码可互换 → 后续实验可在任一码上做
        ▼
   实验2: 病根在哪? ── BASIC+精确Cholesky vs BASIC+ICC(0)
        │   结论: 病根 = "重叠区过度计数" × "局部解不精确"(乘积)。去掉任一因子即愈
        ▼
   实验3: 精确解太贵,选择去掉"过度计数"因子 ── scheme 3 = CG + sASM (D^{-1/2}缩放)
        │   结论: 对称、保 CG、单层,趋势翻回下降
        ▼
   实验4: 同时廉价地去掉两个因子 ── scheme 4 = sASM + Chebyshev 块解
        │   结论: 迭代最低,无额外内存; 加进纯 PETSc 交叉验证 9/9 一致
        ▼
   实验5: "降迭代"是否等于"降时间"? ── -log_view 全局 reduction 计数(非 wall-time)
            结论: scheme 4 内层 Chebyshev 零全局同步,外层迭代降 = 全局同步降,大规模降时间
```

---

## 1. 全实验共用的设置(区域 / 方程 / 网格 / 求解器基底)

为保证实验之间可比,除非另行说明,所有实验共用下列设置。

**区域.** 单位立方体 Ω = [0,1]³。

**方程(强形式).**

```
   -∇·(σ ∇u) = f      在 Ω 内,  σ ≡ 1,  f ≡ 1   (泊松方程 -Δu = 1)
```

**边界条件(对应 cardioid 的混合 Dirichlet/Neumann 结构).**

```
   u = 0            在 x = 0 面             (Dirichlet,MFEM 边界属性 5)
   ∂u/∂n = 0        在其余 5 个面            (齐次 Neumann,自然边界)
```

**解析解(用于校验解的正确性).** 由对称性,所有数据(σ, f, 边界值)均与
(y,z) 无关,故 3D 解退化为一维函数,且其为 3D 定解问题的**精确解**:

```
   u(x,y,z) = x − x²/2,   max u = 0.5 (在 x=1),  cube 均值 = 1/3
```

**网格.** 结构化 nx×nx×nx 六面体网格,每个六面体按 MFEM
`Mesh::MakeCartesian3D(..., TETRAHEDRON)` 的 `hex_to_tet[6][4]` 表(共享
顶点 0–6 主对角线)剖分为 6 个四面体。**P1(线性)H1 单元**,顶点自由度
(nx+1)³。

**求解器基底.**

```
   -ksp_type cg              外层 Krylov: 共轭梯度(SPD)
   -ksp_norm_type preconditioned
   -ksp_initial_guess_nonzero false
   -ksp_rtol 1e-6  -ksp_atol 1e-12
   -pc_type asm  -pc_asm_type basic  -pc_asm_overlap O    (加性 Schwarz, BASIC 变体)
   -sub_ksp_type preonly  -sub_pc_type icc  -sub_pc_factor_levels L  (子域: ICC(L))
```

子域数 = MPI rank 数。扫描参数:overlap `O ∈ {0,1,2}`,ICC 填充层
`L ∈ {0,1,2}`。

**软件环境.** MFEM 4.9.0 + PETSc 3.24.4 + HYPRE 3.1.0 + METIS 5.1.0 +
OpenMPI 5.0.9(与 cardioid 同一套 Spack 工具链)。

**两套独立实现.**
- `asm_demo`(MFEM 装配 + 自写的 Hypre→PETSc AIJ 转换,镜像 cardioid 的
  `ConvertHypreToPetscAIJSafe`);
- `pure_petsc_fem` / `pure_petsc_load`(纯 PETSc,无 MFEM、无 HYPRE,自写
  P1 四面体单元装配,复用 MFEM 的 `hex_to_tet` 表)。

四个求解方案(代码内 `-scheme`):

```
   scheme 0 : CG + PCASM(BASIC)                                    ← 复现 bug 的 baseline
   scheme 1 : GMRES + PCASM(RESTRICT/RAS, 非对称)
   scheme 2 : BCGS  + PCASM(RESTRICT/RAS, 非对称)
   scheme 3 : CG + PCSHELL[ D^{-1/2} · M_BASIC · D^{-1/2} ]        ← sASM
   scheme 4 : CG + sASM,且每个子域用固定阶 Chebyshev/ICC 块解        ← 本工作
```

---

## 2. 实验 1 — 两实现共享 METIS 同一分区,bit-identical 对照

**实验目的.**
排除"现象是某个实现(MFEM 的矩阵转换、稀疏分配、nullspace 处理…)的人为
产物"这一可能。若两套**完全独立**的实现,在**同一离散算子、同一并行分区**
下迭代数逐位一致,则现象必为方法内在,且二者可互换用于后续实验。

**问题 / 区域 / 网格.** 同 §1。nx=24(顶点自由度 15 625),4 ranks,P1 四面体。

**实现细节(三层差异的消除).**
两码默认会因三处不同而给出不同迭代数,逐一对齐:

1. **矩阵数值零.** MFEM 的 `DiffusionIntegrator` 用数值积分,P1 单元中解析
   为零的 K 元被算成舍入级(~1e-16)非零;纯 PETSc 走解析公式得到字面 0。
   将转换处的过滤阈值由 `== 0.0` 改为 `|val| < 1e-12`。
2. **单元分区.** MFEM `ParMesh` 默认用 METIS 剖分;由 `asm_demo` 把已按
   METIS 分布的 PETSc 矩阵 + 右端项 + **逐 rank 行数** dump 为二进制。
3. **载入时的行布局.** `MatLoad` 默认 `PETSC_DECIDE` 会重新等分行,抹掉
   METIS 布局;在 `pure_petsc_load` 中**先 `MatSetSizes(A, m_local, …)`
   再 `MatLoad`**,保住 MFEM 的逐 rank 行数。

**结果.**

(i) 算子指纹(同一确定性向量 x 的 ‖Ax‖,与并行无关):

| nx | MFEM ‖Ax‖    | 纯 PETSc ‖Ax‖ |
|:--:|:------------|:-------------|
| 16 | 2.409188e+01 | 2.409188e+01 |
| 17 | 2.137087e+01 | 2.137087e+01 |
| 24 | 3.128297e+01 | 3.128297e+01 |

(ii) 三层对齐后,在 nx=48 下对 scheme 0/3/4 × 9 个 (O,L) = **27/27 个格子
迭代数逐位一致**(完整数字见附录 B,B.1)。

(iii) 用 `-ksp_monitor_true_residual` 核对:两码在每一步的**预条件残差范数
逐位相同**,且在同一步收敛(例:scheme 0,O=0,L=0 两码均在第 71 步
`CONVERGED_RTOL`,残差历史每位一致)。这是比"迭代数相等"更强的等价证据。

> 说明:迭代数在 rtol 阈值附近对舍入极敏感,不加间隔地连续提交上百次
> mpirun 时,因机器争用可能出现 ±1 的**假性**差异;带间隔的干净复跑给出
> 上述 27/27 逐位一致。残差历史逐位相同则不受此影响。

**结论.**
"overlap↑ ⇒ iter↑" **不是 MFEM / HYPRE / 矩阵转换的人为产物** —— 一个无关
的纯 PETSc P1 实现独立复现同一现象,且两码在同一算子、同一分区下残差历史
逐位一致。因此现象是 `BASIC + ICC + CG` 方法本身的性质;后续实验可在任一
实现上进行。

---

## 3. 实验 2 — BASIC + 精确 Cholesky 块解 vs BASIC + ICC(0)

**实验目的.**
现象既为方法内在,定位其根因。经典 Schwarz 理论默认**子域问题精确求解**;
本实验把唯一变量设为"子域解是否精确",其余(BASIC 变体、矩阵、分区)全部
不变,以判定"局部不精确"是否为根因。

**问题 / 区域 / 网格.** 同 §1。nx=48,4 ranks(与全报告统一;nx=24 亦得到
相同定性趋势)。

**实现细节.**
仅替换子域求解器:`-sub_pc_type icc -sub_pc_factor_levels 0`(不精确)
↔ `-sub_pc_type cholesky`(完整 Cholesky,数学上等价于精确块逆
A_i⁻¹)。外层仍 `CG + PCASM(BASIC)`,overlap 取 0/1/2。

**结果**(nx=48,4 ranks;完整数据见附录 B):

| 子域解 | O=0 | O=1 | O=2 | 趋势 |
|:--|:-:|:-:|:-:|:--|
| BASIC + **精确 Cholesky** | 52 | 34 | **30** | **↓ 单调下降(符合经典理论)** |
| BASIC + ICC(0)(不精确) | 132 | 148 | **179** | ↑ 上升(即 bug) |

**结论.**
仅把局部解从 ICC(0) 换成精确 Cholesky,趋势即由"上升"翻为"下降",回到
经典 Schwarz 理论的预言。故病根可表述为一个**乘积**:

```
   iter 随 overlap 上升  =  (BASIC 在重叠区过度计数)  ×  (ICC 局部解不精确)
```

其中"过度计数"指被 m_k 个子域共享的自由度 k 的修正被加了 m_k 次
(D := Σ R_iᵀR_i = diag(m_k) ≠ I)。**去掉任一因子即可治愈**:经典理论
之所以成立,是因为它默认了第二个因子为 1(精确解);cardioid 的配置同时
踩中两个因子。

---

## 4. 实验 3 — scheme 3:CG + sASM(对称缩放加性 Schwarz)

**实验目的.**
实验 2 表明去掉任一因子即可,但精确 Cholesky 在大规模上 O(n_i³) 内存/时间
不可行。本实验改去**第一个因子(过度计数)**:用对角缩放精确抵消重叠区的
重复加权,同时**保持对称**(从而保住外层 CG),且保持**单层**(不引入粗
空间)。

**问题 / 区域 / 网格.** 同 §1。nx=48,4 ranks。

**方法与实现细节.**
预条件子取

```
   M_sASM⁻¹  =  D^{-1/2} ( Σ_i R_iᵀ A_i⁻¹ R_i ) D^{-1/2},   D = diag(multiplicity m_k)
```

对共享度为 m_k 的自由度,过度计数因子 m_k 被两侧 1/√m_k 精确抵消。因 D 为
对角阵,D^{-1/2} 自伴,夹住的 BASIC 算子对称 ⇒ 整体对称 ⇒ CG 合法。
实现为 PCSHELL:每次作用 = `VecPointwiseMult(D^{-1/2})` → 内层 PCASM(BASIC)
作用 → `VecPointwiseMult(D^{-1/2})`;multiplicity 向量由
`PCASMGetLocalSubdomains` 的 IS 列表累加指示子得到,setup 时算一次。
(注:为何不是单标量松弛?**CG 对预条件子的标量缩放不变**,单个 ω 对 CG
无效;D^{-1/2} 是逐自由度不同的对角,才真正改变预条件谱。)

**结果**(nx=48,4 ranks,L=0 无 ICC 填充;完整 (O,L) 数据见附录 B):

| | O=0 | O=1 | O=2 | 趋势 |
|:--|:-:|:-:|:-:|:--|
| scheme 0 (BASIC) | 132 | 148 | **179** | ↑(bug) |
| scheme 3 (sASM) | 132 | 104 | **103** | **↓(治愈)** |

自检:O=0 时 multiplicity D=I,sASM 退化为 BASIC,两者迭代数完全相等
(132=132),符合预期。解经 §1 解析解校验正确。

**结论.**
sASM 在**保持 CG、保持单层、无额外求解成本**的前提下,把"overlap↑ iter↑"
翻回经典预期的"overlap↑ iter↓"。这是单层框架内最小代价、零风险的修法
(一个 PCSHELL)。代价:它只去掉了第一个因子,第二个因子(局部不精确)仍在,
故仍有进一步压缩迭代的空间 → 实验 4。

---

## 5. 实验 4 — scheme 4:sASM + 固定阶 Chebyshev 块解

**实验目的.**
在 sASM(已去掉过度计数)基础上,**廉价地**缩小第二个因子(局部不精确),
即用低成本手段逼近精确块逆,但不付出精确 Cholesky 的 O(n_i³) 代价、也不
增加 ICC 填充带来的内存。

**问题 / 区域 / 网格.** 同 §1。nx=48,4 ranks,全 (O,L) 扫描。

**方法与实现细节.**
把 sASM 的子域求解器由"`preonly` + ICC(L)"换为"**k 步 Chebyshev 迭代,
以 ICC(L) 为光滑子**":

```
   scheme 4 = CG + D^{-1/2} ( Σ_i R_iᵀ S_i R_i ) D^{-1/2},
              S_i = 固定 k 步 Chebyshev(默认 k=2),预条件为 ICC(L) 的块 i
```

要点:固定步数 + **冻结特征值边界** + 对称(ICC)光滑子 ⇒ S_i 是**固定的
SPD 线性算子** ⇒ 外层 CG 仍合法。它**不增加任何 fill / 内存**(超出 ICC(L)
本身),只多几次块内 mat-vec;且块 Chebyshev 在 `PETSC_COMM_SELF` 上运行
(见实验 5,**零全局通信**)。PETSc 实现:子 KSP 设 `KSPCHEBYSHEV`、
`max_it=k`、`KSP_NORM_NONE`、`KSPChebyshevEstEigSet` 冻结边界。`-localcheby N`
改阶数。在 `asm_demo` 与 `pure_petsc_load` 中均实现。

**结果(a):全 (O,L) 扫描(nx=48,4 ranks).**
sch0=BASIC,sch3=sASM,sch4=sASM+Chebyshev(2)。所有 sch4 格子均
`CONVERGED_RTOL`,且与解析解的 L² 相对误差均为 5.42e-05(与 baseline 相同
离散误差),故解处处正确。

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

观察:(i) O=0 时 sch0=sch3(D=I 自检);(ii) sch0 在 L=0 上升
132→148→179,sch3/sch4 翻为下降;(iii) sch4 即使在 O=0 也优于 sch0/sch3
(85 vs 132),因为 Chebyshev 攻的是"不精确"这个在任何 overlap 都存在的因子;
(iv) sch4 在每个格子均为三者最低。

**结果(b):双实现交叉验证.**
把 scheme 4 同样加入纯 PETSc 的 `pure_petsc_load`(载入 MFEM 同一 METIS
分区矩阵),9 个 (O,L) 格子迭代数与 MFEM 端 **9/9 逐位一致**。

**结论.**
scheme 4 同时(且廉价地)去掉乘积病根的两个因子:`D^{-1/2}` 治过度计数
(免费、对称),固定阶 Chebyshev/ICC 块解治局部不精确(无额外内存)。其迭代
数为四方案最低(约为 baseline 的 1/3),解正确,且独立实现交叉验证一致。
**诚实定位:这是已有要素的合成,而非全新算法**——multiplicity/单位分解
缩放的加性 Schwarz、Chebyshev 光滑的块解均已见于文献;可复用的贡献是那个
**诊断(病根=过度计数×不精确)与处方(用最便宜的工具分别打两个因子)**。
更强的文献亲属(RASHO、SORAS)用调和/Robin 局部问题可得更好的谱常数,但
实现成本更高;常数模 deflation 很强但属(一维)粗空间即两层方法。

---

## 6. 实验 5 — 用 -log_view 全局 reduction 计数度量同步开销

**实验目的.**
"降迭代"未必等于"降时间"。本实验证明 scheme 4 也降**时间**,但**不**用
不可信的 wall-time(同一配置在负载机器上 run-to-run 可差 1.7×,更大的问题
甚至可能"更快"),而用**确定性**指标:PETSc `-log_view` 的全局 reduction
计数。每个 reduction = 一次 `MPI_Allreduce`,即 CG 在大规模并行下延迟受限
的真正瓶颈。

**问题 / 区域 / 网格.** 同 §1。nx=48,4 ranks,O=2(baseline 最差的 overlap),
L=0/1。各配置在命令行尾附加 `-log_view`,读取 `MPI Reductions:` 与
`VecTDot` 的 Count。

**结果(实测).**

| 配置 | iter | MPI Reductions | VecTDot | 是否 = 2·iter+2 |
|:--|:-:|:-:|:-:|:-:|
| BASELINE BASIC+ICC(0) | 179 | **668** | 360 | 是 (360) |
| sASM+ICC(0) | 103 | 443 | 208 | 是 (208) |
| sASM+Cheby2/ICC(0) | 65 | 329 | 132 | 是 (132) |
| **sASM+Cheby2/ICC(1)** | 48 | **278** | 98 | 是 (98) |

**结论.**
两点被严格证明:

1. **`VecTDot = 2·iter + 2` 对每个配置精确成立**(含 scheme 4)。CG 的点积
   (即全局 reduction)只跟随**外层**迭代数;内层 Chebyshev 块解在
   `PETSC_COMM_SELF` 上运行,**贡献零全局 reduction**。故 scheme 4 是"用
   本地浮点换更少全局同步",绝不灌水同步计数。
2. 在 O=2,scheme 4(sASM+Cheby2/ICC1)把**迭代 179→48(3.7×)**、**全局
   同步 668→278(2.4×)**。

因 `MPI_Allreduce` 延迟受限、且 rank 数越多相对越贵,这 2.4× 的同步削减是
"大规模降时间"论断的**机器无关**依据 —— 正是 cardioid(N12 = 数千 rank)
所处的区间。

---

## 7. 迭代次数的理论分析(为什么多 / 为什么少 / 是否符合理论)

本节把上述所有实验的迭代数统一到一个理论量上解释,并逐一判断"是否与理论
预想一致"。所有数字取自附录 B(nx=48,4 ranks,fix_level=1,单一代码状态)。

### 7.1 理论标尺:CG 迭代数 ~ √κ

对 SPD 系统,预条件 CG 的迭代数满足

```
   n_iter ≲ (1/2) · √κ · ln(2/ε),     κ = κ(M⁻¹A) = λ_max(M⁻¹A) / λ_min(M⁻¹A)
```

故"迭代为何多/少"**完全归结为预条件算子 M⁻¹A 的条件数 κ**。下面分析每个
旋钮(overlap O、ICC 填充 L、是否缩放、块解精度)如何改变 λ_max、λ_min,
从而改变 κ。

### 7.2 经典单层加性 Schwarz 理论(假设子域精确求解)

对重叠 Schwarz,经典结果(Dryja–Widlund;Toselli–Widlund 教材)为

```
   λ_max(M_AS⁻¹A) ≤ N_c                         (N_c = 子域着色数,与 overlap δ 无关,有界)
   λ_min(M_AS⁻¹A) ≳ 1 / (1 + 1/(Hδ))            (随 overlap δ 增大而增大)
   ⇒ κ ≈ N_c · (1 + 1/(Hδ))                      (随 δ 增大而减小)
```

(H = 子域尺寸;无粗空间时另有一个 ~1/H² 的、随子域数变差的因子,但本报告
P=4 固定,该因子为常数。)**结论:子域精确求解时,overlap↑ ⇒ κ↓ ⇒ 迭代↓。**
这正是教授与教科书的预期。

**实验 2 验证了这条理论(对照组).** 把子域解换成精确 Cholesky(满足理论
假设),实测 **52 → 34 → 30(O=0→1→2),单调下降**,与理论定性一致。这条
对照实验证明:**理论本身是对的**,问题出在它的假设上。

### 7.3 为什么 BASIC + ICC(0) 反而 overlap↑ 迭代↑(scheme 0)

实测 scheme 0(L=0):**132 → 148 → 179,上升**。这并不与 §7.2 矛盾 —— 而是
§7.2 的**两个前提同时被破坏**:

- **因子 A(过度计数).** BASIC 用 R_iᵀ 延拓,重叠区自由度 k 的修正被相加
  m_k 次(D = ΣR_iᵀR_i = diag(m_k) ≠ I)。这把 λ_max 往 N_c 方向抬;且
  重叠区(m_k>1 的区域)**随 δ 增大而增大**。
- **因子 B(局部不精确).** ICC(0) 只是 A_i 的近似;δ 增大 ⇒ 子块 A_i 变大
  ⇒ 同样零填充的 ICC(0) 对更大块的逼近**相对更差**,即局部误差**随 δ 增大**。

两者**相乘**:不精确误差恰好落在被过度计数放大的重叠区。于是 λ_max 被
(过度计数 × 局部误差)抬高,且**随 δ 上升快于 λ_min 的改善** ⇒ κ↑ ⇒ 迭代↑。
这就是 132→148→179 的成因。**与"正确的(含不精确解的)理论"一致**;只是
经典有界 λ_max 的结论因假设失效而不适用。

### 7.4 为什么 sASM 把趋势翻回下降(scheme 3)

实测 scheme 3(L=0):**132 → 104 → 103,下降**。sASM 用 D^{-1/2}(·)D^{-1/2}
使 ΣR_iᵀW_iR_i = I(单位分解),**精确消除因子 A** ⇒ λ_max 回到 ≈1(不再被
过度计数抬高)。此时 δ↑ 只通过改善 λ_min 起作用,不再抬 λ_max ⇒ κ↓ ⇒ 迭代↓,
**恢复 §7.2 的经典走势**。细看:

- **O=0:132 = scheme 0 的 132**(D=I,sASM 退化为 BASIC,自检通过)。
- **O=0→1:132→104 大降**(过度计数去掉后,overlap 终于"按理论"帮上忙)。
- **O=1→2:104→103 几乎持平**(overlap 收益递减:1/(Hδ) 项饱和,理论也预言
  overlap 从 0→1 收益最大、再加收益小)。

### 7.5 为什么 ICC 填充 L 越大迭代越少(所有方案)

提高 L ⇒ M_i 更接近 A_i⁻¹ ⇒ **因子 B 收缩** ⇒ κ↓ ⇒ 迭代↓,且在每个
(scheme,O) 都单调成立。实测例:scheme 0 在 O=1 上 **148→102→92**(L=0→1→2);
scheme 4 在 O=2 上 **65→48→42**。与"更精确的局部解 ⇒ κ 更小"完全一致。

### 7.6 为什么 scheme 4 处处最低,且 O=0 也受益

scheme 4 用 2 步 Chebyshev(以 ICC 为光滑子)做块解,是对 A_i⁻¹ 的**多项式
加速逼近**,比纯 ICC(0) 好得多却**不增填充**——即**廉价地收缩因子 B**。叠加
sASM 已消除因子 A,两个因子同时变小 ⇒ κ 最小 ⇒ 迭代处处最低(85→68→65,
且每个 (O,L) 均低于 scheme 0/3)。注意 **O=0 也降(132→85)**:因子 B 在
**零 overlap 时也存在**(对角块的 ICC(0) 本身就不精确),故改进块解与 overlap
无关地有效。

### 7.7 是否符合理论 —— 汇总判定

| 配置 | 局部解 | 过度计数(因子A) | 不精确(因子B) | overlap 趋势(实测) | 与理论 |
|:--|:--|:-:|:-:|:--|:--|
| BASIC + 精确 Cholesky | 精确 | 有 | 无 | 52→34→30 ↓ | **符合经典 AS 理论(对照组)** |
| scheme 0 BASIC+ICC(0) | 不精确 | 有 | 有 | 132→148→179 ↑ | 经典定理假设被破坏,故不适用;符合含不精确解的分析 |
| scheme 3 sASM+ICC(0) | 不精确 | **无** | 有 | 132→104→103 ↓ | **符合单位分解/缩放 AS 理论** |
| scheme 4 sASM+Cheby | 较精确 | **无** | 小 | 85→68→65 ↓ | 符合,且 κ 最小 |

**一句话**:"overlap↑ 迭代↑"只在**因子 A 与因子 B 同时存在**时出现;去掉任一
(精确解去 B、或 sASM 去 A)即恢复理论预期的下降。实验 2 的精确-Cholesky
对照组**证明理论本身正确**,问题在 cardioid 的配置同时踩中两个假设破坏点。

---

## 8. 总体结论与对 cardioid 的建议

**诊断结论.**
1. cardioid 看到的"overlap↑ iter↑"**不是代码 bug**,也非 MFEM/HYPRE/装配
   产物(实验 1:两独立实现 27/27 逐位一致,残差历史逐位相同)。
2. 其根因是一个**乘积**:`(BASIC 重叠区过度计数) × (ICC 局部解不精确)`
   (实验 2:换精确 Cholesky 即翻回下降)。经典 Schwarz 理论默认第二因子为 1,
   故不适用于 ICC 不精确解。

**修法结论(单层、保 CG).**
3. **scheme 3 = sASM**:去掉过度计数因子,对称、保 CG、零额外求解成本,迭代
   翻回下降(实验 3)。
4. **scheme 4 = sASM + Chebyshev 块解**:同时廉价去掉两个因子,迭代最低
   (~baseline 的 1/3)、无额外内存、零额外全局同步(实验 4、5)。

**对 cardioid Sys2/Sys3 的落地建议(按场景).**

| 场景 | 推荐 | 理由 |
|:--|:--|:--|
| 最小改动、最稳 | **scheme 3 + ICC(1)** | 一个 PCSHELL,迭代约减半,对称,零风险 |
| 内存紧、大规模 | **scheme 4 (sASM+Cheby2/ICC0)** | 无额外 fill,迭代约减 2.6×,全局同步更少 |
| 迭代压到最低 | scheme 4 (sASM+Cheby2/ICC1) | 迭代约减 3× |
| 网络延迟主导 | 上述任一 + `-ksp_type pipecg` | 把 allreduce 与 mat-vec 重叠 |

建议先上 **scheme 3 + ICC(1)**(确定、低风险);若 profiling 显示
`MPI_Allreduce` 占比高(数千 rank 下大概率),再升级 **scheme 4** 进一步削减
外层迭代数,从而削减全局同步。

---

## 附录 A. 复现命令

详见仓库 `INVESTIGATION.md` §8 与 `asm_bug_demo/` 下的
`sweep.sh / sweep_pure.sh / bench.sh / collect.sh`。其中附录 B 的全部数字由
`collect.sh` 一次性采集(输出 `collect_nx48_n4.csv`)。关键命令:

```bash
cd asm_bug_demo && make

# 实验 2: BASIC + 精确 Cholesky(把 sub_pc 换成 cholesky)
mpirun -n 4 ./asm_demo -fix_level 1 -scheme 0 -nx 24 \
  -ksp_norm_type preconditioned -ksp_initial_guess_nonzero false \
  -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 1000 \
  -pc_type asm -pc_asm_type basic -pc_asm_overlap 2 \
  -sub_ksp_type preonly -sub_pc_type cholesky

# 实验 3/4: scheme 3 / 4
mpirun -n 4 ./asm_demo -fix_level 1 -scheme 4 -nx 48 \
  -ksp_norm_type preconditioned -ksp_initial_guess_nonzero false \
  -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 2000 \
  -pc_type asm -pc_asm_type basic -pc_asm_overlap 2 \
  -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels 1

# 实验 5: 追加 -log_view,读 "MPI Reductions:" 与 VecTDot Count
#   ...（同上命令）... -log_view 2>&1 | grep -E "MPI Reductions:|^VecTDot"
```

---

## 附录 B. 详细数据表(单一代码状态:nx=48, 4 ranks, fix_level=1)

下列数字均由脚本 `asm_bug_demo/collect.sh` 在**同一代码状态、同一参数、带
间隔避免争用**下采集(原始 CSV:`collect_nx48_n4.csv`)。迭代数取 PETSc 内部
`CONVERGED_RTOL ... iterations N` 计数。

### B.1 逐 (O,L) 迭代次数 — scheme 0 / 3 / 4(MFEM 与纯 PETSc 逐位一致)

下表为 MFEM(`asm_demo`)数值;纯 PETSc(`pure_petsc_load`,载入同一 METIS
矩阵)在全部 27 格**逐位相同**(27/27,见 §2)。

| O | L | scheme 0 (BASIC) | scheme 3 (sASM) | scheme 4 (sASM+Cheby2) |
|:-:|:-:|:-:|:-:|:-:|
| 0 | 0 | 132 | 132 | 85 |
| 0 | 1 | 102 | 102 | 73 |
| 0 | 2 | 87  | 87  | 66 |
| 1 | 0 | 148 | 104 | 68 |
| 1 | 1 | 102 | 75  | 54 |
| 1 | 2 | 92  | 70  | 47 |
| 2 | 0 | 179 | 103 | 65 |
| 2 | 1 | 117 | 73  | 48 |
| 2 | 2 | 90  | 67  | 42 |

按 overlap 看 L=0 列的趋势:scheme 0 **132→148→179 (↑)**;scheme 3
**132→104→103 (↓)**;scheme 4 **85→68→65 (↓)**。按 ICC 填充看(固定 O),
每个方案均随 L 增大单调下降。

### B.2 实验 2 — BASIC 子域解:精确 Cholesky vs ICC(0)

| O | BASIC + 精确 Cholesky | BASIC + ICC(0) (= scheme 0, L=0) |
|:-:|:-:|:-:|
| 0 | 52 | 132 |
| 1 | 34 | 148 |
| 2 | 30 | 179 |
| 趋势 | **↓(符合经典理论)** | ↑(因子 A×B,见 §7.3) |

### B.3 实验 5 — 全局同步计数(`-log_view`,L=0)

| 方案 | O | 迭代 | MPI Reductions |
|:--|:-:|:-:|:-:|
| scheme 0 (BASIC) | 0 | 132 | 525 |
| scheme 0 (BASIC) | 1 | 148 | 574 |
| scheme 0 (BASIC) | 2 | 179 | **668** |
| scheme 3 (sASM)  | 0 | 132 | 528 |
| scheme 3 (sASM)  | 1 | 104 | 445 |
| scheme 3 (sASM)  | 2 | 103 | 443 |
| scheme 4 (sASM+Cheby2) | 0 | 85 | 387 |
| scheme 4 (sASM+Cheby2) | 1 | 68 | 337 |
| scheme 4 (sASM+Cheby2) | 2 | 65 | **329** |

在 O=2:scheme 4 相对 baseline 把迭代 179→65、全局同步 668→329(**约 2 倍**)。
`MPI Reductions` 与外层迭代数同步下降,印证内层 Chebyshev 不产生全局通信
(见 §6)。

### B.4 wall-time(O=1,3 次取中位数)

> **重要警示:wall-time 在负载笔记本上不可靠,仅供定性参考;定量的"降时间"
> 论断以 B.3 的全局同步计数(机器无关)为准。**

| 方案 | L | 迭代 | 中位 wall-time (s) |
|:--|:-:|:-:|:-:|
| scheme 0 (BASIC) | 0 | 148 | 0.231 |
| scheme 0 (BASIC) | 1 | 102 | 0.168 |
| scheme 0 (BASIC) | 2 | 92  | 0.171 |
| scheme 3 (sASM)  | 0 | 104 | 0.149 |
| scheme 3 (sASM)  | 1 | 75  | 0.112 |
| scheme 3 (sASM)  | 2 | 70  | 0.119 |
| scheme 4 (sASM+Cheby2) | 0 | 68 | 0.130 |
| scheme 4 (sASM+Cheby2) | 1 | 54 | 0.112 |
| scheme 4 (sASM+Cheby2) | 2 | 47 | 0.113 |

定性看:wall-time 大体跟随迭代数下降(scheme 0,L=0 的 0.231s →
scheme 4,L=0 的 0.130s)。scheme 4 每步本地浮点略高(Chebyshev 块解),
但迭代数大幅减少,净时间更短;严格的、与机器无关的时间依据见 B.3。
