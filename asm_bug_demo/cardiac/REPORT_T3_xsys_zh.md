# Task 3:用前面的系统给后面的系统做预条件(Sys 1 → Sys 2)

> **【升级:跨系统预条件搬到真几何 conforming FEM 算子上】**
> 原 Task 3 在 **20×7×3 mm 规则盒子**上用**结构化 FD 刚度阵**,且驱动是**合成 sigmoid
> 前沿**(`xsys_precond.c`,保留为基线)。现已升级:
> - **算子**:Sys2 的 $K_{\sigma_i+\sigma_e}$、$K_{\sigma_i}$ 改为**心脏 submesh 上的 P1-FEM
>   `DiffusionIntegrator`**(各向异性,σ 值 S/m 数值不变,因 1 S/m = 1 mS/mm),在
>   `heart_torso.py` 的 conforming 网格上装配(`forward_ecg.cpp`)。奇异性处理沿用
>   asm_demo 的均值去除 + `MatSetNullSpace`。
> - **驱动**:不再是合成前沿,而是**真实 Sys1 单域解 $V_m(t)$**(在同一心脏网格上跑出来的
>   序列),$b(t)=-K_{\sigma_i}V_m(t)$ 去均值。
> - **已实现策略**(`forward_ecg.cpp -xsys`,跑在 FEM 算子上):baseline(ICC)/
>   warm-start(上一解作初值)/ POD 历史投影。输出 `fwd_xsys.txt`。
> - **待补**:Nicolaides **shared-coarse / 两层**(原 strategies 4–5)在非结构 FEM 上需要
>   一个**心脏 submesh 分区粗空间**(不再是结构化 4×2×1 盒子);这是文档化的下一步。
> - **状态(已在容器内验证)**:`forward_ecg.cpp -xsys` 用 MFEM 4.9 + PETSc 3.19 实测跑通 FEM Sys2:
>   单细胞 → 真实 Vm(t) 序列驱动 → baseline/warm/POD 三策略迭代数正常输出(`fwd_xsys.txt`)。
>   短跑(NT=5)warm-start 最优(增益随序列变长放大)。下方五策略表用 Mac 上的细网格/长序列重跑回填。
>   `xsys_precond.c` 的结构化-FD + 合成前沿数字保留为对照基线。

## 0. 问题定义(先写清楚,具体是哪个例子)

在心脏管道里,**单域 Sys 1**(monodomain $V_m$)在**每个时间步**驱动一个椭圆"恢复细胞外电位"求解:
$$\textbf{Sys 2}:\quad K_{\sigma_i+\sigma_e}\,u_e=-K_{\sigma_i}\,V_m(t)\qquad(\text{纯 Neumann,奇异}).$$
随去极化波扫过板,**RHS 与解 $u_e(t)$ 随时间光滑变化**,解序列近似落在一个低维子空间。
**目标**:实现我们讨论过的所有"用前系统给后系统做预条件/加速"的方案,**找出最好的**。

**具体例子**:Niederer 板 $20{\times}7{\times}3$ mm,$h{=}0.5$($41{\times}15{\times}7{=}4305$ 节点)。
$\sigma_i{=}(0.17,0.019)$、$\sigma_e{=}(0.62,0.236)$ S/m(纵/横)。
驱动:合成的移动去极化前沿 $V_m(t,x)=-85+110\,\mathrm{sigmoid}((x-c\,t)/0.6)$(前沿 0.12 mm/解,模拟 0.6 m/s 波被采样),
给出光滑移动、一致(去均值)的 RHS 序列(40 个解)。
**度量**:每个 recover 解到 $\|r\|/\|b\|<10^{-8}$ 的 CG 迭代数(及总数)。
**代码**:`asm_bug_demo/cardiac/xsys_precond.c`;图 `fig_xsys.png`。

## 1. 实现的所有方案(逐项)

| 方案 | 思路(用前系统的什么) | 平均迭代/解 | 总迭代(40) |
|---|---|---|---|
| **(0) baseline** | 无迁移(CG+ICC,零初值) | 46.0 | 1839 |
| **(1) warm start** | **解的历史**:上一步 $u_e$ 作初值(时间相干) | 42.0 | 1679 |
| **(2) POD deflation** | **解的历史(Fischer–POD)**:前若干解建 POD 基,Galerkin 投影作初值 | 38.5 | 1541 |
| **(3) shared coarse** | **几何/结构迁移**:Sys 1 同网格上建的 Nicolaides 粗空间,直接用于 Sys 2 的两层 CG | 32.9 | 1316 |
| **(4) warm + coarse** | **历史 + 结构 一起** | **28.9** | **1154** |

**最佳:warm + shared coarse(1154 迭代,比 baseline 快 1.6×)。**

## 2. 结论:哪个方案最好、为什么

**层级**:baseline < warm < POD < 共享粗空间 < (warm + 粗空间)。

- **共享粗空间是主力**(46→33,1.4×)。Sys 1 和 Sys 2 在**同一心脏网格**上,Nicolaides 粗空间($\sum_i\phi_i=1$)
  **一次构建、跨系统复用**;它**张成常数 = Sys 2 奇异 Neumann 的零空间 = 慢全局模**(正是 Task 1 的结论)。
  所以它既消除奇异性、又压低端 $\lambda_{\min}$——这是最干净、最可迁移的"前系统给后系统"的结构。
- **解的历史(warm / POD)是次力**(46→42→38.5)。波是**移动前沿**(平流主导 ⟹ 解序列在解空间里**高秩**),
  所以 POD/历史复用收益有限——这是平流型问题的已知局限。
- **二者正交、可叠加**:历史复用"时间相干",粗空间修"空间低端" ⟹ **叠加得最优(1.6×)**。

### 2.1 真机实测:`forward_ecg -fischer`(真网格 + 真 Vm(t),非合成前沿)

上表的 warm/POD 数字来自**合成移动前沿**驱动的 `xsys_precond.c`。为验证这些结论在**真正的
耦合前向 ECG** 里成不成立,把跨时加速直接接进 `forward_ecg.cpp` 的 EP 主循环:每个 ECG 采样步
(每 ~1 ms)重解一次奇异 Sys2 $K_{\sigma_i+\sigma_e}u_e=-K_{\sigma_i}V_m(t)$,驱动是**真实 TP06 +
IMEX 单域解出来的 $V_m(t)$**。`-fischer` 每步测三种起点的 CG 迭代数:cold($x_0{=}0$)、warm
(上一步 $u_e$)、Fischer(把历史解做 A-正交归一,投影 $x_0=\sum_i\langle p_i,b\rangle p_i$)。
**真正落盘的场永远是 cold 解**,warm/Fischer 只在丢弃向量上测迭代数 ⟹ ECG 与非加速路径**逐位相同**
(已核对),加速永远不可能污染物理或历史基。

**实测(heart.msh,np=4,dt=0.1,一整拍 T=350 ms,350 个采样步,Sys2 CG 总迭代数):**

| 起点 | 总迭代 | 相对 cold |
|---|---|---|
| cold($x_0{=}0$) | 31892 | — |
| warm(上一步 $u_e$) | 28758 | **−9%** |
| Fischer(16 步滑窗历史) | 28029 | **−12%** |

**分相位看**(这才是关键):
- **去极化(QRS,0–40 ms)**:前沿快速扫过(CV≈0.6 mm/ms),相邻 1 ms 采样的 $u_e$ 差别大 ⟹
  warm 仅省 3–6%(cold 88 → warm 84);Fischer 偶尔命中历史张成($x_0$ 已达 $10^{-8}$ ⟹ **0 迭代**),
  偶尔不命中(≈cold),呈双峰。
- **平台/复极(t>50 ms)**:$V_m$ 时间上近似不变 ⟹ 相邻解很近,warm 稳定省 ~13%(cold 92 → warm 79);
  Fischer 与 warm 相当(慢 1D 漂移下"上一步"已是极好预测,多加历史向量近共线、增益有限)。

**结论(诚实):** 真机上跨时加速给 Sys2 的总收益是 **warm −9% / Fischer −12%**——远小于合成低维轨迹
demo 的 −66%。原因正是 §2 说的:真波是**移动前沿(平流主导)**,采样出来的 $u_e(t)$ 在 QRS 段**高秩**、
不是慢变低维流形,所以历史复用收益本就有限;平台段才接近低维、收益才显现。Fischer 略胜 warm(靠
QRS 段的 0-迭代命中 + 平台段的滑窗),但两者都是"次力",与 §2 的层级判断完全一致。

**两个真实的工程坑(修好才量得到收益)**:
1. **`iterative_mode` 不生效**:MFEM 的 `PetscPCGSolver` 只在**构造函数**里 `KSPSetInitialGuessNonzero`,
   之后改 `cg.iterative_mode=true` **不传播到 KSP**,初值被静默忽略 ⟹ warm≡cold(精确相等)。修法:每步
   显式 `KSPSetInitialGuessNonzero((KSP)cg2, PETSC_TRUE/FALSE)`。(独立微测:从**精确解**热启动,修前
   88 迭代、修后 **0 迭代**。)
2. **停机范数约定**:PETSc KSP 默认 `atol` 作用在**预条件后**残差上,会掩盖初值质量。改
   `KSP_NORM_UNPRECONDITIONED` + 每步 `atol=10^{-8}\|b\|`,初值好坏才反映到迭代数。
3. **滑窗**:Fischer 历史必须**滑窗淘汰最旧**;append-only 基填满后冻结在早期 QRS 模态,平台段完全失效
   (实测冻结基平台段 Fischer 91–92 ≈ cold,滑窗后降到 79–89)。

**诚实说明 / 分区结论**:
- **算子级(把 Sys 1 的算子 $\tfrac{1}{\Delta t}M+\tfrac12K$ 直接当 Sys 2 的预条件)我们之前判定不直接可行**——
  Sys 1 有质量位移(小 $\Delta t$ 时质量主导、谱与纯 Laplace 的 Sys 2 不匹配);实测也证实粗空间/历史是更好的迁移途径。
- **收益随"波速/容差"变**:前沿越慢(真实 $\Delta t{=}0.02$ ms 时前沿仅动 ~0.012 mm/步)或容差越松,
  **历史复用(warm)收益越大**;前沿快/容差紧时,**粗空间**主导。**最稳的通用最优是"warm + 共享粗空间"**。

**一句话**:**最好的跨系统预条件 = 共享粗空间(把 Sys 1 网格的 Nicolaides 粗空间迁移给奇异的 Sys 2,精确消零模)+ 解的历史 warm-start**,在本例快 1.6×;粗空间是主力(结构/几何迁移),历史是叠加增益。
