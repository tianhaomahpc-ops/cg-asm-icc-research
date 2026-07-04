# 项目 Prompt:心脏前向 ECG 三系统求解器 —— 区域分解与跨系统预条件研究

> 用途:把整个研究(问题构造 → 反常现象 → 想法与可解释性 → 数值结果 → 结论)浓缩成一份可交接、
> 可复现、可续做的 prompt。分支 `cardiac-sim-fakegeo`;所有数字均为容器内实测(MFEM 4.9 + PETSc 3.19,
> `heart.msh` 10085 dof)。**本研究是算法(数值线性代数 / 区域分解)研究**,不涉及药物/医学。

你是一名数值线性代数 / 区域分解方向的研究者。目标:把一个耦合 PDE 前向 ECG 管线里的三个线性系统解得
**又快又可扩展**,并解释清楚每个加速手段"解决了哪个子问题、为什么有效、代价如何"。

---

## 一、三个系统的构造(方程 + 边界条件 + 离散)

**几何**:心脏 slab $\Omega_h=20\times7\times3$ mm **共形嵌入**躯干 $\Omega_t=50^3$ mm(Gmsh
`BooleanFragments`,交界面 $\Gamma$ 共享同一组节点);连续 P1 Galerkin 非结构四面体(heart 47582 tets,
torso 206647 tets);各向异性 $\sigma$,纤维 $\parallel x$。前身是"假几何"(结构化 FD + 合成 sigmoid 前沿),
本项目换成真几何、变分一致的 FEM 算子。

每个时间步依次解三个 SPD(半)正定系统:

**Sys1 —— 单域 $V_m$(抛物 / 质量主导 / 局部,非奇异)**
$$\chi C_m\,\partial_t V_m=\nabla\!\cdot(\sigma_{\text{mono}}\nabla V_m)-\chi I_{\text{ion}}(V_m,\cdot)+I_{\text{stim}},
\qquad \sigma_{\text{mono}}=\tfrac{\sigma_i\sigma_e}{\sigma_i+\sigma_e}$$
- 边界:$\partial_n V_m=0$(绝缘,纯 Neumann);反应 $I_{\text{ion}}$ 用 TP06 细胞模型,显式。
- 离散(IMEX Crank–Nicolson):$\big(\tfrac1{\Delta t}M+\tfrac12 K\big)V_m^{n+1}=\big(\tfrac1{\Delta t}M-\tfrac12 K\big)V_m^{n}+M(-I_{\text{ion}}-I_{\text{stim}})$。
- **关键**:质量项 $\tfrac1{\Delta t}M$ 使算子 SPD **非奇异且良态**(screened/shifted Laplacian);CG+ICC ~13 迭代。

**Sys2 —— $u_e$ 恢复(椭圆 / 纯 Neumann **奇异** / 全局)**
$$\nabla\!\cdot\big((\sigma_i+\sigma_e)\nabla u_e\big)=-\nabla\!\cdot(\sigma_i\nabla V_m)
\quad\Longleftrightarrow\quad K_{\sigma_i+\sigma_e}\,u_e=-K_{\sigma_i}V_m$$
- 边界:$\partial_n u_e=0$ 全 Neumann ⇒ **奇异**,$\ker=\mathrm{span}\{\mathbf1\}$(常数)。
- 相容性:RHS 必须 $\perp\mathbf1$(即 $\int=0$)⇒ **去均值**;解在相差一常数下唯一 ⇒ 零均值投影 / `MatSetNullSpace`。
- **这是唯一的真难点**(奇异 + 椭圆 + 全局),也是所有跨系统预条件研究的目标算子;baseline ~82–96 迭代。

**Sys3 —— 躯干 Laplace(椭圆 / 非奇异,Dirichlet 锚定)**
$$\nabla\!\cdot(\sigma_o\nabla\varphi)=0\ \text{in }\Omega_t,\qquad
\varphi|_\Gamma=u_e\ (\text{Dirichlet, 来自 Sys2}),\quad \partial_n\varphi|_{\partial\Omega_t\setminus\Gamma}=0$$
- 交界面 Dirichlet 数据使其**非奇异**;baseline ~62–66 迭代。体表 ECG $=\varphi(x_L)-\varphi(x_R)$。

**耦合链**:$V_m(t)$【Sys1】$\to$ Sys2 的 RHS $\to u_e\to$ Sys3 的 Dirichlet BC $\to\varphi\to$ ECG。
子区域 $=$ MPI rank(METIS 几何划分),$n_p=$ rank 数 $=$ 子域数。

---

## 二、观察到的反常现象(anomalies)

1. **重叠反常(核心)**:一层 **ASM(BASIC)** 加重叠 $O$($0\to2$),CG **迭代不降反升**;**sASM**(scaled /
   restricted)加重叠则**迭代下降**。实测最狠:Sys3 $n_p{=}8$ ASM $66\to93\to109$(+65%),sASM $66\to55\to53$(−20%)。
   $O{=}0$ 时二者相等(无重叠无过计数)。

2. **分块方式决定重叠是好是坏**:早期"矩阵连续索引分块"重叠让迭代**下降**,而 METIS 几何分块的 BASIC 重叠让迭代
   **上升** —— 差别在重叠区的**重数(multiplicity)结构**不同。

3. **触达 ≠ 精度(reach ≠ accuracy)**:Sys2 的 $u_e$ 支集在 $k{=}1$ 迭代就**铺满整个心脏**(椭圆全局触达),
   却仍**缓慢收敛** —— 慢的是那个**光滑全局低模**,不是触达范围。

---

## 三、想法与可解释性(信息传播)

- **信息传播 = 每迭代传一个子域跳**:一层 Schwarz/CG 每步只把边界势的信息推进一个子域;要传遍全域需
  $\sim$(子域直径)步 ⇒ **迭代数随 $n_p$ 增长 ⇒ 一层方法不可扩展**。**粗空间**给出全局耦合(粗解一次就全局),
  在**一步内**消掉慢全局模 ⇒ 恢复可扩展性。

- **重叠反常 = 过计数 × 不精确**:BASIC 把重叠 dof 的贡献按重数 $M>1$ **各加一次** ⇒ $M^{-1}$ 在重叠区**放大**
  ($\|M^{-1}\|$ 偏大),谱恶化;又因 ICC(0) 本地解**不精确**,放大误差不被吸收 ⇒ 迭代升。sASM 用对角单位分解
  $D^{-1/2}(\cdot)D^{-1/2}$($D=$ 重数)**精确抵消**过计数 ⇒ $\lambda_{\max}\!\approx\!1$,重叠恢复"越多越快"的正常行为。

- **Green 函数衰减(为什么三系统信息传播不同)**:看边界势扰动传多远。**Sys1**(抛物 / 质量项 $\sim1/\Delta t$
  = 屏蔽)⇒ **指数衰减**,长度 $\sim\!\sqrt{D\Delta t}$(短程、局部,天然可扩展,无需粗空间)。**Sys2/Sys3**
  (纯椭圆 Laplace)⇒ **代数 $1/r$ 衰减**(长程、全局 ⇒ 需要全局粗空间)。

- **谱桥**:$\text{iters}\approx\tfrac12\sqrt{\kappa}\,\ln(2/\varepsilon)$,$\kappa=\lambda_{\max}/\lambda_{\min}$。
  重叠过计数抬高 $\lambda_{\max}$(重数);子域尺寸/距离决定 $\lambda_{\min}$。**sASM 修 $\lambda_{\max}$,粗空间修 $\lambda_{\min}$**。

- **两条正交的加速轴**:迭代数 $=f(\text{初值},\text{预条件})$。
  ① **跨时间(改初值)**:用 Sys2 自己的历史解 —— warm-start / Fischer(A-正交历史投影)。次力。
  ② **跨系统(改预条件)**:共享网格的 **Nicolaides 粗空间**(补常模/全局模)+ **SORAS** 强 fine level。
  其中 **fine level 定绝对迭代数,coarse level 定可扩展性**。

---

## 四、数值结果(全部真机实测;所有加速为测量模式,落盘场恒为 baseline 解 ⇒ ECG 逐位不变)

**(A) ASM vs sASM(rtol $10^{-8}$,ICC(0),子域=rank;迭代 | 计算时间 ms):**
Sys3 $n_p{=}8$,$O{=}0/1/2$:ASM 迭代 $66/93/109$,sASM $66/55/53$;solve-only 墙钟 ASM $35/60/83$ ms,
sASM $39/36/40$ ms。$O{\ge}1$ 时 sASM **迭代更少且墙钟更快**。

**(B) sASM 参数扫描 $L$(ICC level)$\times O$(overlap),按 solve-only 时间取优(setup 在 EP 循环里摊销):**
- Sys2/Sys3(椭圆全局):**时间最优 $L{=}1,O{=}1$**(Sys2 12.5 ms/47 it,Sys3 30.7 ms/36 it)。
- Sys1(质量主导):**$L{=}0,O{=}1$**(ICC0 已足,任何多余 fill 只加成本)。
- **时间最优 $\ne$ 迭代最优**:迭代最少是 $L{=}2,O{=}2$(35/30 it)但慢约 40%;$O{=}2$ 相对 $O{=}1$ 只省几步迭代却更贵。
- 普适规律:**$O{=}1$ 三系统通用最优**;ICC level 按算子性质分(局部→L0,椭圆→L1)。

**(C) 跨时间(整拍,Sys2 CG 总迭代):** cold 31892 → warm −9% → Fischer(16 步滑窗)−12%。真波是移动前沿
(平流主导、QRS 段高秩)⇒ 收益远小于合成低维轨迹 demo 的 −66%;平台段才低维、收益才显。

**(D) 跨系统(每解迭代,弱扩展):**

| $n_p$ | baseline | +Nicolaides | SORAS | SORAS+coarse |
|---|---|---|---|---|
| 2 | 84 | 81 | **20 (4.2×)** | 20 |
| 16 | 96 | 72 (1.34×) | 54 (1.8×) | 53 |

**(E) 迭代 $\ne$ 墙钟(关键)**:$n_p{=}4$ 实测 —— Nicolaides 粗空间 1.11× 迭代 / **1.21× 更快**(便宜的
$n_p\times n_p$ 粗解);SORAS 近似精解本地块 2.31× 迭代 / **慢 20×**(本地精解 5.5 vs 0.11 ms/迭代)。
**小规模墙钟结论到 3000 核会翻转**(每迭代 = 一次全局 Allreduce,迭代数就是货币)。

**(F) 物理/数值一致性验证(整拍)**:TP06 动作电位形态;**$u_e\!\sim\!\nabla V_m$(平台期整片归零 —— 最强判据)**;
躯干偶极远场驱动 ECG;CV $\approx0.5$ m/s,P1$\to$P8 43 ms(与共识一致)。

---

## 五、结论 + 三个工程坑 + 大规模取舍

**结论**
1. **重叠区域分解要用 sASM / RAS(单位分解加权),不是 BASIC**;BASIC 的重叠在此设置下有害。
2. **时间最优参数用"适度 ICC + overlap-1",不是迭代最少的配置**(时间最优 $\ne$ 迭代最优)。
3. 加速有两条正交轴:**历史复用(次力)+ 共享粗空间(补全局,主力)+ SORAS 强 fine(最大的迭代杠杆)**;层级
   baseline < 历史 < 粗空间 < SORAS。
4. **迭代 $\ne$ 墙钟**:SORAS 是迭代主力但本地精解贵,小规模墙钟更慢;大规模(通信主导)才翻正。

**三个工程坑(修好才量得到收益)**
1. MFEM `PetscPCGSolver.iterative_mode` 只在构造函数生效 ⇒ 每步显式 `KSPSetInitialGuessNonzero`(否则
   warm≡cold;微测:精确初值 88→**0** 迭代)。
2. PETSc 默认 `atol` 作用在预条件后残差、掩盖初值质量 ⇒ 改 `KSP_NORM_UNPRECONDITIONED` + `atol=1e-8‖b‖`。
3. Fischer 历史必须**滑窗淘汰最旧**(append-only 基填满后冻结在早期 QRS 模、平台段失效)。

**大规模(~3000 核,几十 k dof/rank,强可扩展)技术取舍**
- **保留**:粗空间(强扩展命根子;但稠密复制 $A_0^{-1}$ 是 $O(n_c^3)$ 串行瓶颈 ⇒ 改 `PCTELESCOPE`+MUMPS 或
  三层,优选 **GenEO 谱粗空间**);SORAS(本地解改**固定次数 Chebyshev/ICC**,丢 CG-to-1e-10);warm-start(免费)。
- **丢弃**:near-exact CG 本地解;稠密复制粗解。
- 实测:MUMPS/SuperLU\_DIST/TELESCOPE/GAMG/ML 可用,**PCHPDDM 未编译**(需 `--with-hpddm` 重配)。
- 用户现状"全系统 ICC+CG 已有 ~90% 强扩展"**合理**:大子区域计算主导 + 迭代温和增长;粗空间/SORAS 是继续
  往更多核 / 更细网格推时保住 90% 的保险。

**代码入口(`forward_ecg.cpp`,skips EP loop)**:`-precond`(ASM vs sASM,iters + tot/solve-only ms)、
`-sweep`(sASM $L\times O$ 扫描 + 时间最优)、`-fischer`(跨时间)、`-coarse`(Nicolaides 两层)、
`-soras [-soras_local K]`(强 fine level)。交付:`REPORT_SUMMARY_zh.md`、`REPORT_T3_xsys_zh.md`、
`slides_sim_fakegeo_en.{tex,pdf}`、`plot_geom_mesh.py`/`fig_geom.png`/`fig_meshview.png`。
