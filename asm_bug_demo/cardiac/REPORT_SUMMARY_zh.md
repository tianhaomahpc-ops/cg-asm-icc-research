# 心脏前向 ECG 三系统:问题定义 · 逻辑 · 方法 · 结论

> 真几何、变分一致 FEM 上的三系统耦合前向 ECG,及其求解加速研究的完整梳理。
> 全部数字为本容器内(MFEM 4.9 + PETSc 3.19,`forward_ecg`,`heart.msh` 10085 dof)实测。
> 相关文件:`forward_ecg.cpp`、`REPORT_T2_cardiac_zh.md`(离散/物理)、`REPORT_T3_xsys_zh.md`
> (跨系统预条件明细)、`plot_verify.py`(验证图)。

---

## 一、问题定义

### 1.1 物理与几何

在**共形非结构四面体 P1 FEM** 网格上做**耦合前向 ECG**:心脏 slab(20×7×3 mm)居中嵌入
50³ mm 躯干,心脏-躯干交界面 conforming(`heart_torso.py` 的 Gmsh `BooleanFragments` 生成;
`heart.msh` / `torso.msh` 共享同一组交界面节点)。单位 mm / ms / mV / mS/mm;χ=140/mm、
Cm=0.01、σ_i=(0.17,0.019)、σ_e=(0.62,0.236)、σ_o=0.22(纤维 ∥ x)。

### 1.2 三个系统(每个时间步)

| 系统 | 方程 | 类型 | 难点 |
|---|---|---|---|
| **Sys1** 单域 Vm | $(\tfrac1{\Delta t}M+\tfrac12K)V_m^{n+1}=\text{rhs}(V_m^n,I_\text{ion})$ | 抛物 / 质量主导 / **局部** | 良态,CG+ICC ~13 迭代 |
| **Sys2** u_e 恢复 | $K_{\sigma_i+\sigma_e}\,u_e=-K_{\sigma_i}V_m$ | 椭圆 / **纯 Neumann 奇异** / **全局** | ker=span{1},慢全局模,~84–96 迭代 |
| **Sys3** 躯干 Laplace | $K_{\sigma_o}\phi=0$,交界面 Dirichlet=u_e | 椭圆 / 非奇异 / 全局 | Dirichlet 定态,~66 迭代 |

耦合:Sys1 的 $V_m(t)$ 驱动 Sys2 → u_e 经共形交界面作为 Sys3 的 Dirichlet BC → 体表电位差
= ECG。反应用 TP06 细胞模型,时间推进 IMEX(显式反应 + C–N 扩散)。

### 1.3 研究目标

**Sys2 是唯一的真难点**(奇异 + 椭圆 + 全局),也是"用前系统给后系统做预条件"研究的目标算子。
目标:在**不改变解**(ECG 逐位不变)的前提下,把 Sys2 的迭代数尽可能打下来,并搞清**每种加速的
机理、量级、可扩展性**。

---

## 二、逻辑(为什么这么做)

### 2.1 为什么 Sys2 难、迭代多

- **奇异**:纯 Neumann,$K_{ie}\mathbf1=0$,零空间 = 常数 = 一个慢全局模。必须去均值(RHS ⊥ 常数)
  + 挂 `MatSetNullSpace` 才能解。
- **椭圆 + 全局**:信息要传遍全心脏才收敛;一层区域分解**每迭代只传一个子域跳**,子域越多越慢。
- **弱 fine level**:`forward_ecg` 默认 PC = block-Jacobi + ICC(0),**无重叠、无 Robin**,对 1 万 dof
  椭圆问题 ~84–96 迭代是正常的。

### 2.2 两条正交的加速轴

迭代数 = f(初值好坏, 预条件强弱)。对应两条**互相正交、可叠加**的轴:

| 轴 | 用什么 | 修的是 | 机理 |
|---|---|---|---|
| **跨时间** | Sys2 **自己**上一步/历史解作初值 | **初值** | u_e(t) 随波光滑漂移 ⟹ 历史是好预测 |
| **跨系统** | Sys1/Sys2 **共享网格**上的粗空间 + 强 fine level | **预条件** | 粗空间补全局模;SORAS 补局部+传输 |

跨系统这条又分两层:
- **fine level(局部)**:决定**绝对迭代数**。弱(bjacobi)→ ~90;强(SORAS,重叠+Robin)→ ~20–54。
- **coarse level(全局)**:决定**可扩展性**(迭代随子域数是否平坦)。补的是奇异常数模/慢全局模。

**逻辑主线**:先量出弱 baseline,再分别上跨时间(次力)、共享粗空间(补全局)、SORAS(强 fine,大头),
最后组合;每一步都对比 baseline、保证 ECG 不变、给出机理解释。

---

## 三、方法(做了什么,怎么实现)

所有加速都实现为**测量模式**:落盘场永远是干净的 baseline 解 ⟹ ECG 逐位不变(已核对);
加速解只在丢弃向量上测迭代数,永不污染物理。外层统一用 PETSc CG(稳健处理奇异),
统一 $\|b\|$-相对停机(`KSP_NORM_UNPRECONDITIONED` + `atol=10^{-8}\|b\|`)。

### 3.1 跨时间:warm-start + Fischer(`-fischer`)

- **warm**:上一步 u_e 作初值。
- **Fischer**:把历史解做 **A-正交归一**(A-内积 Gram–Schmidt),初值 = $\sum_i\langle p_i,b\rangle p_i$
  (历史张成上的 A-正交投影);历史用**滑窗**(淘汰最旧,16 步)。

### 3.2 跨系统 · 共享 Nicolaides 粗空间(`-coarse`)

- 两层加性 $M^{-1}=M_\text{fine}^{-1}+R_0A_0^{+}R_0^\top$;$R_0$ **每个 MPI 子域一列**(该子域 true-dof
  的示性函数,单位分解 $\sum_kR_0e_k=1$)⟹ **张成常数 = Sys2 奇异零空间 = 慢全局模**;这正是 Sys1
  在同一网格上会建的空间,一次构建、跨系统复用。
- $A_0=R_0^\top K_{ie}R_0$($n_c\times n_c$,奇异 $A_0\mathbf1=0$)在去均值子空间求逆。

### 3.3 跨系统 · 强 fine level SORAS(`-soras`)

- 优化 Schwarz:本地 **Neumann 块** $K_\text{loc}$(shared 面自然 BC,免费)+ 共享面**真实质量**
  $\alpha M_\Gamma$(Robin 传输条件,α≈0.2)+ 单位分解对称组合
  $M^{-1}=P^\top D(K_\text{loc}+\alpha M_\Gamma)^{-1}DP$。
- 本地块用 COMM_SELF 上的 CG+ICC 近似精确解。可与 `-coarse` 叠成两层 SORAS+coarse。

### 3.4 实现要点(三个真实工程坑)

1. **`iterative_mode` 不传播**:MFEM `PetscPCGSolver` 只在**构造函数**里 `KSPSetInitialGuessNonzero`,
   之后改 `iterative_mode` 无效 ⟹ 初值被静默忽略、warm≡cold。必须**每步显式**
   `KSPSetInitialGuessNonzero`。(微测:从精确解热启动,修前 88 迭代、修后 **0**。)
2. **停机范数**:PETSc 默认 `atol` 在**预条件后**残差 ⟹ 掩盖初值质量;改 `UNPRECONDITIONED` +
   $\|b\|$-相对。
3. **Fischer 滑窗**:历史必须淘汰最旧;append-only 基填满后冻结在早期 QRS 模态、平台段失效。

所有算子(SORAS、两层)都写成 `mfem::Solver`,经 `SetPreconditioner` 挂成 PETSc `PCShell`。

---

## 四、结论(实测,全部真机验证)

### 4.1 加速效果(Sys2 每解 CG 迭代)

**跨时间(整拍 T=350ms,总迭代):**

| 起点 | 总迭代 | 相对 cold |
|---|---|---|
| cold | 31892 | — |
| warm | 28758 | −9% |
| **Fischer** | 28029 | **−12%** |

分相位:QRS 段波前快移、相邻解差别大 ⟹ 仅省 3–6%(Fischer 偶尔 0 迭代);平台段解近乎不变 ⟹
warm 省 ~13%。**次力**,与合成低维 demo 的 −66% 差距大——真 u_e(t) 在 QRS 段高秩(平流主导)。

**跨系统 · fine level 弱扩展(每解迭代):**

| np(=n_c) | baseline(bj+ICC0) | +Nicolaides 粗空间 | **SORAS** | SORAS+coarse |
|---|---|---|---|---|
| 2 | 84 | 81 | **20**(4.2×) | 20 |
| 4 | 87 | 77 | 35 | 34 |
| 8 | 91 | 73 | 41 | 41 |
| 16 | 96 | 72(1.34×) | 54(1.8×) | 53 |

### 4.2 三条核心结论

1. **SORAS 是大杠杆**:84→20(np=2,**4.2×**)、96→54(np=16,1.8×)。**强 fine level(重叠+优化
   Robin 传输)才是把 ~90 变 ~20–54 的那一步**,和 `soras_par` 独立实测(27–41)一致。

2. **粗空间的价值取决于 fine level 已有多少全局耦合**:
   - 在**弱 baseline** 上,Nicolaides 粗空间加 25%(np=16,1.34×)——补上缺失的全局模。
   - 在 **SORAS** 上几乎不加分(434→421,~3%)——SORAS 的 Robin 传输已提供全局耦合,粗模冗余。

3. **一层方法仍随子域数增长**(SORAS 每解 20→54):要**真正平坦**需更富的 **GenEO 谱粗空间**
   ($n_c=k\cdot$np,每子域取若干本征模),这是文档化的下一步。

**层级(全部真机验证,np=16 每解迭代)**:
```
baseline 96 → +Nicolaides 72 (1.34×,补全局模) → SORAS 54 (1.8×,强fine,大头) → SORAS+coarse 53
```
外加正交的跨时间历史(−12%)可再叠。

### 4.3 物理正确性(整拍验证,见 `fig_verify_time/fields.png`)

| 判据 | 期望 | 实测 |
|---|---|---|
| AP 形态 | TP06 静息 −85 / 平台 / 复极 250–320ms | ✓ |
| **u_e ~ ∇Vm**(最强判据) | 波前偶极、平台整片归零 | ✓ |
| 躯干场 | 光滑偶极远场、平台归零、驱动 ECG | ✓ |
| ECG 时序 | 负 QRS 对去极化、正 T 对复极 | ✓ |
| 传导速度 | Niederer ~0.6–0.7 m/s | 0.5 m/s,P1→P8 43ms ✓ |
| 纤维各向异性 | 纵向(x)快 | P1 0.8ms → P8 44ms ✓ |

u_e 平台期归零是最强判据:u_e 由 ∇Vm 驱动,平台期全心脏均匀去极化、无梯度 ⟹ u_e≈0,同时解释
ECG 的平坦 ST 段。三系统耦合、时序、量级全部自洽。

### 4.4 未决 / 下一步

- **GenEO 谱粗空间**:让 SORAS 也弱可扩展(迭代随 np 平坦)——每子域解局部广义特征值问题取低模。
- 跨时间与跨系统正交,可进一步组合(SORAS + Fischer)端到端叠加。
- 当前 $n_c=$np(每 rank 一列);细分成 $n_c\gg$np 会让 Nicolaides 粗空间更强。
