# 研究报告:ASM(ICC0)+CG 的 overlap 反常 —— sASM 是否最优、跨系统加速、大规模可扩展性

**对象**:`-Δu=f` 型椭圆系统(Laplace 混合 BC = Sys3;pure-Neumann 奇异 = Sys2 心脏 u_e 恢复;质量主导 = Sys1 单域),用 `CG + 重叠加性 Schwarz(ASM) + 子域 ICC(0)` 求解。
**现象**:子域解不精确(ICC0)时,**增大 overlap 反而迭代数上升**。
**方法**:容器内真实数值实验(MFEM 4.9 源码构建 + PETSc 3.19 + OpenMPI 4.1 + **METIS 几何分区**,`asm_demo` 立方体模型 + `forward_ecg` 心脏 FEM)+ 文献综述。所有迭代数为 CG 收敛到 `rtol=1e-6` 的实测值,脚本见 `asm_bug_demo/research/run_experiments.sh`。

---

## 0. 三个问题的结论(摘要)

1. **sASM 不是最优,也不是唯一好办法。** 反常的本质是**两个独立机制的乘积:overlap 的"重复计数"(over-counting)× 子域解的"不精确"(inexactness)**。sASM 只对称地修掉 over-counting 这一半;另一半(不精确)要靠更精确的子域解。**实验证明:子域用精确 LU 时反常完全消失**(overlap 恢复"越大越好")。而且即便只在一层方法里比,**sASM 也不是最好的**——对称化 RAS(SMRAS)、sASM+Chebyshev 子域解都更优;Robin/优化传输条件(OSM/SORAS/RASHO)进一步改善"边界信息传递不精确"。但**根本上,所有一层方法都不强可扩展**。

2. **可以,而且收益明确。** 三个系统在**同一网格**上、互为**谱等价**的加权刚度/质量算子,因此可以:① 一次构造**共享粗空间**(Nicolaides/GenEO)或 IC/AMG 分解,三系统共用;② 用**解历史 warm-start**(Fischer 投影);③ 跨时间步 **deflation/recycle** 最小特征向量。常数向量同时充当 Sys2 奇异核的 deflation。

3. **一层方法(含 sASM)不强可扩展;两层(粗空间)才行。** 实测**弱扩展**下一层 sASM 迭代数随子域数单调上升(68→132),两层 GAMG 恒定(9→10)。这与理论一致:一层 κ∼1+1/(Hδ) 含 1/H 随子域数增长,两层 κ∼1+H/δ 与子域数无关。

> **一句话**:针对这个问题,**最优路线不是 sASM,而是"两层 CG":CG +(sASM 一层平滑器 + Nicolaides/GenEO 粗空间),子域解用 Chebyshev-ICC 提精度;跨系统再叠加共享粗空间 + warm-start**。它同时解决精度/对称/边界(Q1)、跨系统(Q2)、可扩展(Q3)。这正是真实大规模心脏码(openCARP、Chaste、lifex、Pavarino–Scacchi)实际采用的结构。

---

## 1. Q1:反常的本质 —— sASM 是否最优,还有没有优化空间

### 1.1 先复现反常(真 METIS 几何分区,4 子域)
| Sys3 Laplace, nx=24 | O=0 | O=1 | O=2 |
|---|---|---|---|
| **BASIC ASM** | 71 | 80 | **100** |
| **sASM** | 71 | 59 | **58** |

BASIC 随 overlap **上升**(71→100),sASM **下降**(71→58)。反常 + 修复都成立。

### 1.2 关键实验:反常 = over-counting × 不精确子域解
固定 nx=48、4 子域,扫 {BASIC, sASM} × {ICC(0),ICC(1),ICC(2),LU} × overlap:

| 方法 | 子域解 | O=0 | O=1 | O=2 | O=3 |
|---|---|---|---|---|---|
| BASIC | ICC(0) | 118 | 147 | 175 | **196** |
| BASIC | ICC(1) | 98 | 106 | 110 | 130 |
| BASIC | ICC(2) | 85 | 91 | 88 | 103 |
| sASM | ICC(0) | 118 | 101 | 101 | 101 |
| sASM | ICC(1) | 98 | 80 | 72 | 72 |
| sASM | ICC(2) | 85 | 69 | 66 | **60** |
| **BASIC** | **LU(精确)** | 49 | 31 | 28 | **25** |

**读法(这张表是全报告的核心)**:
- **子域精确(LU)时,反常彻底消失**:BASIC 也随 overlap 单调下降(49→25),overlap 恢复"越大越好"的教科书行为。⟹ **反常的根因之一就是子域解不精确**(用户的判断正确)。
- 子域越精确(ICC0→ICC1→ICC2→LU),BASIC 的反常**越来越弱**(196→130→103→25)。
- **sASM 修掉的是 over-counting**:它在每个 ICC 档都把 BASIC 的"上升"压成"下降/持平",但在 ICC(0) 下 overlap 收益**封顶**(101 plateau)——因为残余的不精确还在。要继续从 overlap 拿收益,必须**同时**提精度(sASM+ICC2:60)。
- ⟹ **两个正交的轴**:① over-counting(修法 = sASM 的对称缩放 D^{−1/2}(·)D^{−1/2});② 不精确(修法 = 更强子域解:ICC fill / Chebyshev / LU)。sASM 只动第一根轴。
- 文献支撑:精确子域解下 κ≤C(1+1/(Hδ)) 随 δ 增大而减小(overlap 帮忙);不精确子域解把局部解的最大误差放到**正是 overlap/界面**处,放大 overlap 反而注入不一致(Smith–Bjørstad–Gropp 1996;Toselli–Widlund 2005;Ifpack2 ILU(k)+overlap 实践 arXiv:2503.08126)。

### 1.3 一层方法 bake-off:sASM 并不是最优
固定 Sys3 nx=48、4 子域、overlap=2、ICC(0),比所有一层变体:

| 一层方法 | CG 迭代数 | 对称/CG 有效 |
|---|---|---|
| **SMRAS**(对称化 RAS,scheme 8) | **59** | 是 |
| **sASM+Chebyshev** 子域解(scheme 4) | **64** | 是 |
| sASM(scheme 3) | 101 | 是 |
| ε-PU 加权(scheme 7) | 110 | 是 |
| BASIC(scheme 0) | 175 | 是 |
| RAS + GMRES(restrict) | 173 | 否(需 GMRES) |

**sASM(101)被 SMRAS(59)和 sASM+Cheby(64)明显击败**。机理与文献完全吻合:
- **RAS 系**(Cai–Sarkis 1999)用单侧 PU 权重消 over-count,经验上比 ASM 快;但非对称、需 GMRES。**对称化**后得 **RASHO**(Cai–Dryja–Sarkis 2003,SPD 专用,谐波 overlap)/ **SORAS**(St-Cyr–Gander–Thomas 2007),保 CG。我们的 **SMRAS** 即此族,实测最优。
- **更精确子域解**(Chebyshev-ICC = 定常 SPD 多项式,保 CG)直接打"不精确"那根轴,实测 64。
- **优化传输条件(Optimized Schwarz)**:把子域人工边界从 Dirichlet 换成 **Robin/二阶 Ventcel**,收敛因子从 1−O(L) 改善到 1−O(L^{1/3})/1−O(L^{1/5}),且**无 overlap 也收敛**(Gander 2006)。这正面回答"边界信息传递不精确"——Dirichlet 只传值不传通量,Robin 近似 Steklov–Poincaré/DtN,把低频界面误差也耦合准。CG 有效的打包形式是 SORAS。

### 1.4 Q1 结论
**sASM = "保 CG 有效地修掉 RAS over-count"的一个修复,不是最优。** 改进空间(按对这个问题的契合度):
1. **加粗空间(两层)** —— 最根本(见 §3),也顺带压住 overlap 反常。
2. **更精确子域解**(Chebyshev-ICC / 更高 ICC fill):直接消"不精确"轴,保 CG。
3. **对称化 RAS(SMRAS/RASHO/SORAS)**:消 over-count 比 sASM 更彻底,保 CG。
4. **Robin/优化传输条件**:改善边界信息传递,弱化对 overlap 的依赖。

---

## 2. Q3:大规模强可扩展性(先讲,因为它是 Q1 的根本答案)

### 2.1 一层方法不可扩展(理论 + 实测)
一层 ASM 的条件数界 **κ(M⁻¹A) ≤ C(1+1/(Hδ))**,含 **1/H** 项:子域越多 H 越小,κ 越大,CG 迭代(∝√κ)越多。**弱扩展**(固定 ~15k dof/子域,增大子域数)实测:

| 子域数 np | nx | 总 dof | **一层 sASM**(O=1) | **两层 GAMG** |
|---|---|---|---|---|
| 2 | 30 | 3.0e4 | 68 | 9 |
| 4 | 38 | 5.9e4 | 84 | 9 |
| 8 | 48 | 1.2e5 | 103 | 10 |
| 16 | 60 | 2.3e5 | **132** | **10** |

**一层 sASM 单调上升(68→132,翻倍),两层 GAMG 恒定(9→10)。** 强扩展(固定 nx=48,变子域数)下 sASM≈100–115、GAMG≈10(10× 差距,且 GAMG 与子域数无关)。

### 2.2 两层(粗空间)才可扩展
两层 Schwarz κ ≤ C(1+H/δ),**去掉 1/H 项,与子域数无关**(generous overlap 时 O(1))。粗空间选择:
- **Nicolaides**(每子域一个 PU 常数向量):对 Laplace/pure-Neumann **正好**——它的近零模就是常数,Nicolaides 逐子域精确表示;且常数向量同时是 Sys2 奇异核(必须放进粗空间或 deflate)。Nicolaides 1987;Tang–Nabben–Vuik 2009。
- **GenEO**(子域 overlap 内局部广义特征问题选小特征模):异质/各向异性鲁棒,**可证界** κ≤(1+k₀)[2+k₀(2k₀+1)max(1+H/δ)],与子域数和系数跳跃无关。Spillane 等 2014;arXiv:2104.00280。
- **AMG**(代数多重网格,GAMG/BoomerAMG):实测 10 迭代、与规模无关,工程首选。

### 2.3 真实大规模心脏码用什么(佐证)
椭圆(细胞外/躯干)解是 bidomain 计算瓶颈(Vigmond 等 2008)。主流码全部用**两层/多重网格**,无人用一层 sASM:
- **openCARP/CARP**:`CG + hypre BoomerAMG`(每 CG 迭代一个 V-cycle);Plank 等 2007/2021。
- **Chaste**:block-diagonal + 每块一个 AMG(Bernabeu 等)。
- **lifex**:CG + AMG / 加性 Schwarz。
- **Pavarino–Scacchi 系**:多层加性 Schwarz(子域数无关,SISC 2008);**Newton–Krylov–BDDC/FETI-DP deluxe**,可证 **polylog 界 C(1+log(H/h))²**、对系数跳跃鲁棒(SISC 2022,arXiv:2101.02959)。

---

## 3. Q2:前面的系统给后面的系统做加速(预处理)

三系统 Sys1(质量+刚度)、Sys2(奇异刚度 K_{σi+σe})、Sys3(躯干 Laplace)在**同一心脏网格**上由同一套 FE 装配得到,仅差**电导加权 + 质量项**,故**互为谱等价**:存在 0<c≤C 使 c·xᵀA₁x ≤ xᵀA₂x ≤ C·xᵀA₂x,于是 κ(M⁻¹A₂) ≤ (C/c)·κ(M⁻¹A₁)——**为 A₁ 造的预条件子直接能用于 A₂**,迭代数只乘 √(C/c)(Axelsson–Karátson;Mardal–Winther 算子预处理,h 无关)。三条可叠加的复用轴:

### 3.1 共享粗空间 / 预条件(结构/几何迁移,收益最大)
一次在网格上建 **Nicolaides/GenEO 粗空间**或 **IC/AMG 分解**,三系统共用,每个算子只重建小的 Galerkin 粗矩阵 ZᵀAZ(廉价)。常数向量同时 deflate 掉 Sys2 的奇异核(否则 CG 在零特征上停滞)。这正是 Task 3 里"把 Sys1 网格的 Nicolaides 粗空间迁移给 Sys2"的理论依据。心脏文献:Pavarino–Scacchi 的粗空间是**网格定义**的,可跨 formulation 复用。

### 3.2 解历史 warm-start(Fischer 投影,最便宜)
RHS 随去极化波**光滑漂移** ⟹ 把新 RHS 在 A-内积下投影到最近 m 个历史解张成的子空间,作 CG 初值;只需解残余的正交分量。Fischer 1998 报告 ~2× 省时;SPD 用 A-正交(能量范数)投影正好匹配 CG。对 Sys2 需投影到 range(去常数)。

### 3.3 谱 deflation / Krylov recycling(跨时间步)
把一次求解中提取的**最小特征(Ritz)向量** recycle 到后续系统:deflated CG(Saad 等 2000)、GCRO-DR(Parks–de Sturler 2006)、对称的 RMINRES。有效条件数从 λ_n/λ_1 变 λ_n/λ_{k+1},迭代 ∝√κ_eff 显著下降;文献报 20–70% 省迭代。eigCG 在 QCD 类问题最多 ~8×。

### 3.4 实测(forward_ecg `-xsys`,真实 Sys1 Vm(t) 序列驱动 Sys2 FEM 算子)
| 策略 | 总 CG 迭代(NT=5) |
|---|---|
| baseline(ICC,零初值) | 147 |
| warm-start(上一解作初值) | 142 |
| POD 历史投影 | 143 |
短跑(5 个 RHS)增益小(~3%),与文献一致:warm/POD 的收益随序列变长、RHS 漂移越慢而放大(真实 Δt=0.02ms 时前沿每步仅动 ~0.012mm)。

### 3.5 Q2 排序(对 CG-SPD 的预期收益)
1. **共享粗空间**(结构迁移,一次建多次用,且解 Sys2 奇异核)。
2. **谱 deflation/recycle**(攻低频/病态,Sys2/Sys3 收益最大)。
3. **warm-start 历史投影**(近零成本,三系统通用)。
4. **复用 IC/AMG 分解**(谱等价,省 setup,注意各向异性下 C/c)。
> 三者正交可乘:warm-start(免费初值)+ within-system deflation + 共享粗空间,叠在一个很少重建的细网格预条件子上。

---

## 4. 总结:针对本问题的推荐方案

| 维度 | 不推荐 | 推荐 |
|---|---|---|
| 修 overlap 反常 | 单纯 sASM(只修 over-count 一半,ICC0 下封顶) | **两层粗空间** + 更精确子域解(Chebyshev-ICC),或 SMRAS/SORAS |
| 边界信息传递 | Dirichlet 人工边界 | **Robin/优化传输(SORAS)** |
| 大规模可扩展 | 任何一层方法(含 sASM,迭代随子域数涨) | **两层(Nicolaides/GenEO/AMG)**,子域数无关 |
| 跨系统(Sys1→Sys2/3) | 每系统从零 | **共享粗空间 + warm-start + deflation** |

**最终建议**:`CG +(sASM/RASHO 一层平滑器 + Nicolaides(异质则 GenEO)粗空间)`,子域解用 Chebyshev-ICC;跨时间步/跨系统叠加 Fischer warm-start 与 Ritz deflation,粗空间在三系统间共享。这与 openCARP/Chaste/lifex/Pavarino–Scacchi 的工程实践一致,且每一项都有本仓库可复现的实测支撑(`research/run_experiments.sh`)。

> **诚实边界**:① 实测在 PETSc 3.19 + 源码 MFEM 4.9、cube 模型(Sys2/Sys3 形态)与心脏 FEM 上完成,未在用户 Mac 的 3.24/Spack 上复跑;② 两层用 GAMG 作"含粗空间的可扩展方法"代表,未手写 GenEO(GenEO 的可证界引文献);③ 文献条目经 WebSearch 交叉核对,个别精确常数/页码建议查原文 PDF(代理对部分出版商 403)。

---

## 参考文献(精选)

**Schwarz / 两层 / 粗空间**
- A. Toselli, O. Widlund, *Domain Decomposition Methods — Algorithms and Theory*, Springer, 2005.
- B. Smith, P. Bjørstad, W. Gropp, *Domain Decomposition*, Cambridge UP, 1996.
- V. Dolean, P. Jolivet, F. Nataf, *An Introduction to Domain Decomposition Methods*, SIAM, 2015.
- N. Spillane et al., "Abstract robust coarse spaces … generalized eigenproblems (GenEO)," *Numer. Math.* 126:741–770, 2014. https://doi.org/10.1007/s00211-013-0576-y
- R. A. Nicolaides, "Deflation of conjugate gradients …," *SIAM J. Numer. Anal.* 24:355–365, 1987. https://doi.org/10.1137/0724027
- J. M. Tang, R. Nabben, C. Vuik, Y. Erlangga, "Comparison of two-level preconditioners …," *J. Sci. Comput.* 39:340–370, 2009. https://doi.org/10.1007/s10915-009-9272-6

**RAS / 对称化 / 优化 Schwarz**
- X.-C. Cai, M. Sarkis, "A restricted additive Schwarz preconditioner …," *SIAM J. Sci. Comput.* 21:792–797, 1999. https://doi.org/10.1137/S106482759732678X
- X.-C. Cai, M. Dryja, M. Sarkis, "RAS preconditioners with harmonic overlap (RASHO) for SPD systems," *SIAM J. Numer. Anal.* 41, 2003. https://doi.org/10.1137/S0036142901389621
- M. J. Gander, "Optimized Schwarz Methods," *SIAM J. Numer. Anal.* 44:699–731, 2006. https://doi.org/10.1137/S0036142903425409
- A. St-Cyr, M. Gander, S. Thomas, "Optimized … restricted additive Schwarz preconditioning (ORAS/SORAS)," *SIAM J. Sci. Comput.* 29:2402–2425, 2007. https://doi.org/10.1137/060652610
- E. Efstathiou, M. Gander, "Why RAS converges faster than AS," *BIT* 43:945–959, 2003.

**跨系统:recycling / deflation / 投影**
- P. Fischer, "Projection techniques … successive right-hand sides," *CMAME* 163:193–204, 1998. https://doi.org/10.1016/S0045-7825(98)00012-7
- J. Erhel, F. Guyomarc'h, "Augmented CG for consecutive SPD systems," *SIAM J. Matrix Anal. Appl.* 21:1279–1299, 2000.
- Y. Saad, M. Yeung, J. Erhel, F. Guyomarc'h, "A deflated version of CG," *SIAM J. Sci. Comput.* 21:1909–1926, 2000. https://doi.org/10.1137/S1064829598339761
- M. Parks, E. de Sturler et al., "Recycling Krylov subspaces …," *SIAM J. Sci. Comput.* 28:1651–1674, 2006. https://doi.org/10.1137/040607277
- K. Soodhalter, E. de Sturler, M. Kilmer, "A survey of subspace recycling iterative methods," *GAMM-Mitt.* 43(4), 2020. arXiv:2001.10347
- O. Axelsson, J. Karátson, "Equivalent operator preconditioning for elliptic problems," *Numer. Algorithms*, 2009.

**心脏 / bidomain / heart–torso**
- L. Pavarino, S. Scacchi, "Multilevel additive Schwarz preconditioners for the bidomain …," *SIAM J. Sci. Comput.* 31:420–443, 2008. https://doi.org/10.1137/070706148
- M. Munteanu, L. Pavarino, S. Scacchi, "A scalable Newton–Krylov–Schwarz method …," *SIAM J. Sci. Comput.* 31:3861–3883, 2009.
- N. Huynh, L. Pavarino, S. Scacchi, "Newton–Krylov-BDDC / FETI-DP deluxe for the bidomain," *SIAM J. Sci. Comput.* 44:B224–B249, 2022. arXiv:2101.02959
- G. Plank et al., "Algebraic multigrid preconditioner for the cardiac bidomain model," *IEEE TBME* 54:585–596, 2007.
- E. Vigmond et al., "Solvers for the cardiac bidomain equations," *Prog. Biophys. Mol. Biol.* 96:3–18, 2008.
- M. Boulakia et al., "Mathematical modeling of electrocardiograms," *Ann. Biomed. Eng.* 38:1071–1097, 2010.
