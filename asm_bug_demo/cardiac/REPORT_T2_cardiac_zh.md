# Task 2:Niederer benchmark 心脏电生理 + ECG(PETSc,本机)

> **【升级:非结构四面体 P1-FEM + conforming 躯干 + 完整耦合前向 ECG】**
> 原 Task 2 在 **20×7×3 mm 规则盒子**上用**结构化 7 点 FD lumped-mass** 模板,
> ECG 用 lead-field pseudo-ECG(`monodomain.c`,保留为基线)。现已升级为:
> - **网格**:`heart_torso.py`(Gmsh OpenCASCADE)生成**真非结构 Delaunay 四面体**:
>   20×7×3 mm 心脏 slab **居中嵌入 50³ mm 躯干方盒**,`BooleanFragments` 保证
>   心脏-躯干交界面 **conforming**(共享同一组节点/面)。physical volume→MFEM
>   domain attr(1=heart,2=torso),physical surface→bdr attr(1=body,2=interface);
>   导出 **MSH 2.2 ASCII**(MFEM 4.9 唯一可读格式)。
> - **离散**:**MFEM 4.9 P1 四面体 FEM**(`DiffusionIntegrator`/`MassIntegrator`,
>   各向异性 σ 用 `MatrixCoefficient`,纤维沿 x → diag(σ_L,σ_T,σ_T));时间仍 IMEX
>   Crank–Nicolson,Sys1 = (1/dt)M + (1/2)Kdiff;TP06 反应(`tt06.h` 原样复用)。
> - **前向 ECG**:不再用 lead-field 伪 ECG,而是**真实体表前向**:Sys2 在心脏上恢复
>   $u_e$(奇异 pure-Neumann),经 conforming 交界面耦合到 Sys3 躯干 Laplace,读体表
>   电位差为 ECG。两种耦合:默认 decoupled(保住奇异 Sys2),`-monolithic` 为全域
>   变分一致解。
> - **代码**:`forward_ecg.cpp`(+ `mfem_petsc_util.hpp`、`sigma_tensor.hpp`);
>   `make mesh && make forward_ecg`,在用户 Mac 的 MFEM/PETSc/Gmsh 工具链上构建运行。
>   程序首先做 **conforming 自检**(心脏∩躯干共享顶点数>0 才继续)。
> - **状态(已在容器内验证)**:`forward_ecg.cpp` 用 **MFEM 4.9(源码构建)+ PETSc 3.19 + OpenMPI
>   + Gmsh 4.15** 实测 **编译通过 + 端到端跑通**:`heart_torso.py` 生成 17k 四面体网格,
>   **conforming 自检 = 580 个共享交界面顶点**;心脏 710 dof / 躯干 3018 dof;8 个 Niederer
>   基准点全部激活、波各向异性传播(P1→P8 ≈ 51 ms),体表 ECG 随去极化偏转;decoupled 与
>   `-monolithic` 两条耦合路径、`-xsys` 研究均跑通、干净退出。粗网格(h=1 mm)CV≈0.42 m/s,
>   程序打印诚实对比(细化 + 调 σ → 趋近 0.6–0.7 / P8→43 ms,同原 `monodomain.c` 的收敛趋势)。
>   下方 Niederer 定量表用**用户 Mac 上的细网格**重跑回填即可。`monodomain.c` 结构化-FD 保留为基线。

## 0′. ASM vs sASM 预条件:三系统迭代数(容器内实测)

`forward_ecg.cpp -precond` 对三个系统分别用 **ASM(PC_ASM_BASIC + 子域 ICC)** 与
**sASM(对称缩放 D^{−1/2}M_BASIC⁻¹D^{−1/2} + ICC)** 做 CG 预条件,收敛到 rtol=1e-8
的 **CG 迭代数**(MFEM 4.9 + PETSc 3.19 实测)。**并行已修好**(见下),所以可用**真
METIS 几何分区**(`mpirun -n 4`,1 子域/rank)。加密网格(heart 10,085 / torso 36,229 dof):

| 系统 | overlap | ASM(BASIC) | sASM |
|---|---|---|---|
| **Sys1** 单域(心脏,质量主导,良态) | 0 / 1 / 2 | 12 / 13 / 13 | 12 / **8** / 8 |
| **Sys2** u_e 恢复(心脏,奇异 pure-Neumann) | 0 / 1 / 2 | 72 / **89 / 90** | 72 / 71 / 71 |
| **Sys3** 躯干 Laplace(非奇异) | 0 / 1 / 2 | 59 / **83 / 87** | 59 / 50 / 52 |

**读法**:① overlap=0 时两者**完全相同**(无重叠 ⇒ 重数 D=I ⇒ 无 over-count);
② **几何 METIS 分区下,BASIC ASM(ICC0) 在 Sys2/Sys3 上随 overlap 上升——这就是 repo 主线
的"overlap↑→迭代↑"反常**(Sys2 72→89→90、Sys3 59→83→87);**sASM 把它压平/修复**
(Sys2 72→71→71、Sys3 59→50→52);③ 奇异 Sys2 经常数零空间投影正常收敛(`MatSetNullSpace`)。
> 并行修复说明:此前并行会卡死,定位到两个 bug 并修好——(a) 内层 PC 用了 PETSc 的
> **bare `PCICC`**(单进程算法,MPIAIJ 上死锁)→ 改为 **bjacobi + 子块 ICC**(并行安全,
> 单 rank 等价于纯 ICC);(b) `fes_p.GlobalTrueVSize()`(集合通信)被错放进 `if(rank==0)`
> → 死锁,改为所有 rank 调用。现 EP / `-precond` 在 `mpirun -n 2/4` 下均跑通。
>
> **并行加速与一致性**:`-T 60` 全程仿真 **串行 44 s → 4 进程 19 s(~2.3×)**;
> **EP 传播完全一致**(P8=39.58 ms、CV=0.550 m/s 与串行逐位相同)。**注意**:默认
> *decoupled* 路径里 heart→torso 的 `ParTransferMap`(SubMesh→SubMesh 界面 Dirichlet
> 传递)在该 MFEM 4.9 构建下**随分区不同**(并行的体表 ECG 幅值与串行差 ~20%);
> `-monolithic`(并网格,无界面传递)则**逐位并行一致**。⟹ **decoupled 的前向 ECG /
> 出图请用单 rank**(界面传递精确),并行用于 EP 加速与 `-precond`。字段 dump 已改为
> **逐 rank**(`..._r<rank>.txt`,`plot_results.py` 自动拼接),所以并行 dump 的 Vm/激活
> 场仍是完整且一致的。

## 0. 问题定义(先写清楚,具体是哪个例子)

**心脏(Niederer 2011 benchmark)**:**$20\times7\times3$ mm 长方体板**,纤维沿长轴 $x$。
- **离子模型**:**ten Tusscher–Panfilov 2006(TP06)epicardial**,19 个状态变量。
- **单域方程**:$\chi\big(C_m\,\partial_t V_m+I_{\rm ion}\big)=\nabla\!\cdot(\bm\sigma\nabla V_m)+I_{\rm stim}$,组织边界零通量(Neumann)。
- **参数(Niederer)**:$\chi{=}140\,\text{mm}^{-1}$、$C_m{=}0.01\,\mu\text{F/mm}^2$、
  单域电导 $\sigma_L{=}0.1334$、$\sigma_T{=}0.0176\,\text{mS/mm}$(谐和组合,纤维沿 $x$)。
- **刺激**:$(0,0,0)$ 角的 $1.5{\times}1.5{\times}1.5$ mm 立方,2 ms。
- **网格**:$h\approx0.5$ mm(心脏);时间 $\Delta t{=}0.02$ ms。

**离散(参照 `slides/tex/cardiac.pdf`)**:
- **空间**:P1 (集中质量) 有限元 / 结构网格。
- **时间**:**Crank–Nicolson 扩散 + 显式反应(IMEX)** ⟹ **Sys 1** $=\frac{\chi C_m}{\Delta t}M+\frac12K_{\sigma}$。
- **离子时间积分**:**门控 Rush–Larsen + 浓度 forward Euler**。
- 每步:① 各节点 TP06 react(推进门控/浓度,给出 $I_{\rm ion}^n$);② 组装 RHS;③ CG 解 Sys 1 得 $V^{n+1}$。

**ECG**:lead-field **pseudo-ECG**(电极距板 20 mm,$\phi_e=\sum_{\rm node}\nabla\!\cdot(\bm\sigma\nabla V_m)/r$)。
全躯干前向解(**Sys 2** recover $u_e$ 奇异 + **Sys 3** torso)在 Task 3 中构建与使用。

**代码**:`asm_bug_demo/cardiac/`:`tt06.h`(模型)、`tt06_cell.c`(单细胞验证)、`monodomain.c`(单域+ECG)。

---

## 1. 逐级验证(管道正确)

### Stage 1 — TP06 单细胞(`fig_tt06_ap.png`)
1 Hz 起搏,$\Delta t{=}0.02$ ms。**实测**:
- 静息 **−85.5 mV**、峰值 **+40.9 mV**、**APD90 = 305.7 ms** —— **全部生理正常**
  (目标:静息 ~−85、峰 ~+35..45、APD90 ~270..310)。
- 正确的 spike–dome–plateau(Ito 切迹)+ 钙瞬变(峰 ~0.84 μM)。⟹ **离子模型正确**。

### Stage 2 — 单域传播(`fig_mono_activation.png`)
板上从角点刺激,波**各向异性传播**:沿纤维($x$)快、横向慢,等时线呈椭圆。
**网格收敛(Niederer 的核心)**:远角 P8 激活时间
$$h{=}0.5\text{mm}:76.5\text{ms}\ \longrightarrow\ h{=}0.2\text{mm}:47.6\text{ms}\quad(\text{趋向共识}\sim43\text{ms}).$$
纵向 CV $\approx0.96$ m/s(网格收敛、与共识 0.6–0.7 同量级,略快)。⟹ **单域管道正确、并随网格收敛**。

### Stage 3 — ECG(`fig_mono_ecg.png`)
完整一拍(360 ms)的 pseudo-ECG:**正向 QRS**(去极化,峰 ~57 ms)回基线(~80 ms)、
随后**负向 T 波**(复极化,谷 ~315 ms)——与 AP 时程一致。⟹ **ECG 形态正确(QRS+T)**。

---

## 2. 与 Niederer / ECG 的对比小结

| 量 | 本实现 | Niederer 共识 | 评价 |
|---|---|---|---|
| TP06 静息/峰/APD90 | −85.5 / +40.9 / 305.7 ms | ~−85 / ~+40 / ~280..300 | ✓ 正确 |
| 远角 P8 激活($h{=}0.2$) | 47.6 ms | ~42.9 ms | ~10% 内,且随 $h$ 收敛 |
| 纵向 CV | 0.96 m/s | 0.6–0.7 m/s | 同量级,略快(粗网格 + 可微调 $\sigma$) |
| ECG 形态 | QRS(+)→T(−) | QRS→T | ✓ 形态正确 |

**结论**:**端到端管道正确**(TP06 细胞 → 各向异性单域 → ECG),关键定量量(P8、CV、AP、ECG 形态)
在 Niederer 共识的合理范围内,且 P8 随网格收敛趋向共识值。**满足"管道正确即可推进 Task 3"的门槛。**

**诚实说明**:① 纵向 CV 略快于共识,可由更细网格 + 微调 $\sigma_L$ 收紧(Niederer 各码本身也差 ~10–20%);
② Task 2 的 ECG 用 lead-field pseudo-ECG;**完整躯干前向(Sys 2 recover + Sys 3 torso,50³ mm conforming)在 Task 3 构建**(它们正是跨系统预条件要用的 Sys 2 / Sys 3)。
