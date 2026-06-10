# 对角加权 ASM 变体与外层 Krylov 的消融实验(实验 1–5)

本报告在 sASM(scaled Additive Schwarz)的基础上,设计五个对照实验,从
**对称性**、**归一化剂量**、**权重形式**、**GMRES 参数**、**Krylov 方法**
五个方向系统地考察"增大 overlap 反而迭代数上升"这一现象的成因与修复边界。

所有数字均在 MFEM(`asm_demo`)与纯 PETSc(`pure_petsc_load`)两套实现上
**逐位一致**(bit-identical),新方案 scheme 5/6 的交叉校验为 15/15 EQ
(见 `asm_bug_demo/xcheck_exp.log`)。

---

## 0. 统一框架

经典 BASIC 加性 Schwarz

$$M_{\text{BASIC}}^{-1}=\sum_{i=1}^{P}R_i^{T}\tilde A_i^{-1}R_i ,$$

其重叠节点被重复计数,形式化为

$$\sum_{i=1}^{P}R_i^{T}R_i=D=\operatorname{diag}(m_k),\qquad
m_k=\#\{i:\text{DOF }k\in\Omega_i^{\text{ovl}}\}.$$

要抵消这层重复,总共需要一份 $D^{-1}$ 的权重;**怎么放(两侧/单侧)、放多少
(幂次)、配什么 Krylov** 就构成下面这张表:

| 记号 | 预条件子 $M^{-1}$ | 归一化总量 | 对称 | 适配 Krylov | scheme |
|---|---|---|---|---|---|
| BASIC | $\sum_i R_i^{T}\tilde A_i^{-1}R_i$ | 无 | 是 | CG | 0 |
| RAS | $\sum_i \tilde R_i^{T}\tilde A_i^{-1}R_i$(0/1 所有权) | $D^{-1}$ | 否 | GMRES/FCG | 0 + `restrict` |
| **sASM** | $D^{-1/2}M_{\text{BASIC}}^{-1}D^{-1/2}$ | $D^{-1}$ | 是 | CG | 3 |
| exp2 | $D^{-1}M_{\text{BASIC}}^{-1}D^{-1}$ | $D^{-2}$(过量) | 是 | CG | 5 |
| exp3 | $D^{-1}M_{\text{BASIC}}^{-1}$(单侧) | $D^{-1}$ | 否 | GMRES/FCG | 6 |

---

## 1. 实验设置(五个实验共用)

- **问题**:单位立方体 Laplace,1 面 Dirichlet + 5 面 Neumann,源项 $f=1$。
- **离散**:MFEM P1 四面体元,`nx=48`($49^3\approx1.18\times10^5$ DOF)。
- **并行**:4 进程,串行 METIS 分区后 `ParMesh`;两套实现共用同一分区
  (`multiplicity range = [1,3]`)。
- **求解**:外层 Krylov,`rtol=1e-6`,`max_it=2000`,
  `-ksp_norm_type preconditioned`;子域解 ICC($L$),`preonly`,除非特别说明取 $L=0$。
- **参照量**:每次都记录收敛 reason、$\lVert b-Ax\rVert/\lVert b\rVert$、与解析解的相对误差
  $\lVert u-u^\*\rVert/\lVert u^\*\rVert\approx5.42\times10^{-5}$(确认收敛到正解,而非"假收敛")。
- 数据文件:实验 1–3 见 `asm_bug_demo/exp_nx48_n4.csv`,实验 4–5 见
  `asm_bug_demo/exp45_nx48_n4.csv`;脚本 `collect_exp.sh` / `collect_exp45.sh` / `xcheck_exp.sh`。

---

## 2. 实验 1:RAS + ICC(0),GMRES vs CG

**目的**:RAS 非对称,标准配对是 GMRES;把它强行配 CG,演示对称性对 CG 的必要性。
**方法**:`-pc_asm_type restrict`,外层分别 `-ksp_type gmres` 与 `-ksp_type cg`,$O=0,1,2$,$L=0$。

| $O$ | BASIC+CG | RAS+GMRES | RAS+CG | sASM+CG |
|---|---|---|---|---|
| 0 | 132 | 265 | 132 | 132 |
| 1 | 148 | 232 | **2000(DIVERGED)** | 104 |
| 2 | 179 | 136 | 111 | 103 |

**结论**:
- RAS+GMRES(正确配对):$265\to232\to136$,overlap↑ → iter↓,经典 Schwarz 行为。
- RAS+CG(非法配对):$O=0$ 收敛(此时 `restrict=basic`,对称)、**$O=1$ 直接发散**、
  $O=2$ 又收敛——非对称 PC 配 CG **没有保证,行为飘忽**。
- $O=0$ 时 RAS=BASIC=sASM=132,是正确性自检。

---

## 3. 实验 2:$D^{-1}M_{\text{BASIC}}^{-1}D^{-1}$(对称,过度归一化)

**目的**:把 sASM 两侧的 $D^{-1/2}$ 都换成 $D^{-1}$(总量从 $D^{-1}$ 变 $D^{-2}$),
做"剂量对照",检验 $D^{-1/2}$ 是否为正确剂量。
**方法**:scheme 5,CG,全 $(O,L)$ 网格。

| $O$ | $L$ | BASIC | sASM $D^{-1/2}$ | exp2 $D^{-1}$ |
|---|---|---|---|---|
| 0 | 0 | 132 | 132 | 132 |
| 0 | 1 | 102 | 102 | 102 |
| 0 | 2 | 87 | 87 | 87 |
| 1 | 0 | 148 | 104 | 123 |
| 1 | 1 | 102 | 75 | 92 |
| 1 | 2 | 92 | 70 | 80 |
| 2 | 0 | 179 | 103 | 128 |
| 2 | 1 | 117 | 73 | 93 |
| 2 | 2 | 90 | 67 | 81 |

**结论**:
- $O=0$ 时 $D=I$,三者恒等(自检通过)。
- $O\ge1$ 时 exp2 **恒在 BASIC 与 sASM 之间**:过度归一化($D^{-2}$)比 BASIC 好,
  但**不如正确剂量 $D^{-1/2}$**。
- $D^{-1/2}$(单位分解的那一份)是正确剂量,**不是越多越好**。

---

## 4. 实验 3:$D^{-1}M_{\text{BASIC}}^{-1}$(单侧左乘,非对称)

**目的**:同样总量 $D^{-1}$,但单侧放(非对称),与对称 sASM、与 RAS 的 0/1 权重对照。
**方法**:scheme 6,分别 GMRES 与 CG,$O=0,1,2$,$L=0$。

| $O$ | exp3+GMRES | exp3+CG | RAS+GMRES | sASM+CG |
|---|---|---|---|---|
| 0 | 265 | 132 | 265 | 132 |
| 1 | 215 | **2000(DIV)** | 232 | 104 |
| 2 | 221 | **2000(DIV)** | 136 | 103 |

**结论**:
- $O=0$:$D=I$,$D^{-1}M=M_{\text{BASIC}}$,对称 → 等同 RAS、CG 正常。
- exp3+CG:$O\ge1$ **一致发散**(真残差 $\sim10^{-1}$,误差也错)——非对称稳定地破坏 CG。
- 与 RAS+GMRES 比:overlap 增大后 RAS(136)明显优于分数权重单侧缩放(221),
  说明 **0/1 唯一所有权 ≳ 分数 $1/m_k$ 单侧**。

---

## 5. 实验 4:GMRES 参数(restart × 正交化)

**目的**:实验 1/3 的 GMRES 迭代数偏高,是 restart 太短还是正交化丢失?
**方法**:RAS+GMRES,`-ksp_gmres_restart` $\in\{30,60,120,240,600\}$,
经典 GS(默认)与修正 GS(`-ksp_gmres_modifiedgramschmidt`)各一遍,$O=0,1,2$,$L=0$。

| $O$ | r=30 | r=60 | r=120 | r=240 | r=600 |
|---|---|---|---|---|---|
| 0 | 265 | 195 | 130 | 129 | 129 |
| 1 | 232 | 147 | 95 | 95 | 95 |
| 2 | 136 | 127 | 92 | 92 | 92 |

(每个 restart 下**经典 GS 与修正 GS 迭代数逐行完全相同**,故只列一列。)
次要:scheme 6 + GMRES,r=30→600:$O=1$ 215→101,$O=2$ 221→99。

**结论**:
- **restart 是主导因素**:30→120 大幅下降($O=1$:232→95),之后饱和(已达满 GMRES)。
  实验 1/3 里"GMRES 看着比 CG 差"很大程度是 **restart=30 的假象**。
- **修正 GS vs 经典 GS:完全相同**——预条件后条件数不够坏,正交性损失不影响收敛,
  **正交化不是瓶颈**。
- 满 GMRES(r=600)下 RAS 随 overlap:$129\to95\to92$,**单调下降**,经典 Schwarz 行为显现。

---

## 6. 实验 5:RAS + FCG(Flexible CG)

**目的**:CG 在非对称 RAS 上垮了(实验 1),换 flexible CG(`-ksp_type fcg`,
容忍可变/非对称预条件)能否救活,且是否比 GMRES 便宜。
**方法**:`-pc_asm_type restrict` + `-ksp_type fcg`,$O=0,1,2$,$L=0$;并扫 `-ksp_fcg_mmax`。

| $O$ | RAS+FCG | RAS+CG | RAS+GMRES(r30) | RAS 满GMRES(r600) | sASM+CG |
|---|---|---|---|---|---|
| 0 | 132 | 132 | 265 | 129 | 132 |
| 1 | **97** | **2000(DIV)** | 232 | 95 | 104 |
| 2 | **93** | 111 | 136 | 92 | 103 |

`mmax` 扫描($O=2$):30/60/120/600 → **全为 93**。
scheme 6 + FCG:$O=1$ 103,$O=2$ 102(也被救活)。

**结论**:
- **FCG 把 CG 在非对称 RAS 上救活**:$O=1$ 普通 CG 发散,FCG 收敛 97;overlap 单调
  $132\to97\to93$;$O=0$(对称)FCG=CG,不多花。
- **FCG ≈ 满 GMRES**(97/93 ≈ 95/92),但 **mmax=30 就够**(与 mmax=600 同为 93)——
  即用很短的截断窗口达到满 GMRES 级收敛,**正交化代价远低于满 GMRES**。
- 代价提醒:FCG 单步需存 $\sim$mmax 个方向、做 mmax 个内积,**单步比 CG 贵**;
  墙钟时间不能仅按迭代数和 sASM 直接比。

---

## 7. 更大范围参数扫描:BASIC 的翻转点与 sASM 的全域有效

**目的**:把 $(O,L)$ 扫描范围扩大,定位 BASIC"overlap 趋势翻转"的 ICC level,
并确定 sASM 的迭代数下限。
**方法**:scheme 0(BASIC)与 scheme 3(sASM),外层 CG,$O=0\text{–}5$。
数据 `asm_bug_demo/sweep2_nx48_n4.csv`(脚本 `collect_sweep2.sh`)。

**7.1 BASIC + ICC + CG(迭代次数),$L=0\text{–}6$:**

| $O\backslash L$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| 0 | 132 | 102 | 87 | 77 | 72 | 66 | 64 |
| 1 | 148 | 102 | 92 | 75 | 65 | 56 | 50 |
| 2 | 179 | 117 | 90 | 74 | 65 | 57 | 51 |
| 3 | 199 | 130 | 103 | 80 | **59** | 56 | 50 |
| 4 | 203 | 133 | 105 | 83 | 66 | 55 | **46** |
| 5 | 202 | 133 | 106 | 83 | 68 | 56 | 48 |
| **随 $O$** | ↑ | ↑ | ↑ | ≈ | ↓ | ↓ | ↓ |

**翻转点 $L\approx4\text{–}5$**:以判据"$O{=}3$ 是否优于 $O{=}0$"看,$L=0,1,2,3$ 为否
(199/130/103/80 $>$ 132/102/87/77),$L=4$ 起为是(59 $<$ 72),$L=5,6$ 单调下降。
机理:ICC level 升高 $\Rightarrow$ 子域解趋于精确 $\Rightarrow$ 因素 A 的 $\omega\to1$
$\Rightarrow$ $C_0^2(\delta)$ 的下降占上风。**与第 6 节"成因"一致**。

**7.2 sASM + ICC + CG(迭代次数),$L=0\text{–}5$:**

| $O\backslash L$ | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 0 | 132 | 102 | 87 | 77 | 72 | 66 |
| 1 | 104 | 75 | 70 | 60 | 51 | 48 |
| 2 | 103 | 73 | 67 | 57 | 50 | 43 |
| 3 | 102 | 74 | 66 | 55 | 48 | 42 |
| 4 | 102 | 73 | 65 | 55 | 47 | 41 |
| 5 | 101 | 74 | 65 | 54 | 47 | **40** |
| **随 $O$** | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ |

sASM **在每一个 $L$(含 $L{=}0$)overlap 都有益**,无需等到 $L=4\text{–}5$;且每个 $(O,L)$
都不劣于 BASIC(见 7.1),$O{=}0$ 时两者相同($D=I$)。

**7.3 sASM 迭代数的饱和与下限**(`sweep3/4`):

- **overlap 饱和**:固定 $L=8$,$O=5,6,7,8,10$ 全为 **32**($O\ge5$ 再加 overlap 零增益)。
- **ICC level 触底**:固定 $O=5$,$L=6\text{–}20$ 为 $36,34,32,31,30,28(L{=}12),29,29,29$
  ——$L\approx12$ 触底 $\approx\mathbf{28}$,再加填充不再降(ICC 逼近精度封顶)。
- **真下界(精确 Cholesky 子域解)**:BASIC$+$精确 Cholesky,$O=0/2/3/5/8 = 52/30/27/22/\mathbf{18}$
  ($O{=}8$ 仍在降),但每次 **4–9 s**(比 ICC 慢 $20\text{–}40\times$)。3D 下 level-fill ICC
  即便 $L=20$ 也远非完整分解,故迭代卡在 $\sim28$,高于精确解的 $\sim18$。

---

## 8. 迭代次数与时间的最优对比(ASM vs sASM)

**目的**:在公平计时下比较 BASIC 与 sASM 的**最优时间**。
**计时口径**:`time=` 只括 `pcg.Mult`(KSPSolve);为公平,加 `-warmup`(先做一次不计时
的解,触发所有 `PCSetUp`/ICC 分解),使两边的 `time=` 都是**纯迭代(solve-only)**——
即预条件复用、setup 摊销的口径(心脏代码每个时间步复用同一矩阵,正合此口径)。
取 **best-of-5** 最小值压噪声。数据 `asm_bug_demo/time_nx48_n4.csv`(脚本 `collect_time.sh`)。

**8.1 每个 overlap 的最优(取最佳 $L$),solve-only 秒 / 迭代:**

| $O$ | BASIC iter | BASIC s | sASM iter | sASM s |
|---|---|---|---|---|
| 0 | 87($L2$) | 0.145 | 87($L2$) | 0.150 |
| 1 | 102($L1$) | 0.169 | 75($L1$) | 0.127 |
| 2 | 90($L2$) | 0.175 | **73($L1$)** | **0.124** |
| 3 | — | — | 66($L2$) | 0.132 |
| **随 $O$** | — | ↑ 越慢 | — | ↓ $O{=}2$ 触底 |

**8.2 全局最优:**

| 方法 | 最优配置 | iter | solve-only |
|---|---|---|---|
| BASIC | $O{=}0,\ L{=}2$ | 87 | 0.145 s |
| **sASM** | $O{=}2,\ L{=}1$ | **73** | **0.124 s** |

**结论**:
- **BASIC 的时间最优被逼到 $O=0$**:overlap 让它迭代与时间都上升($0.145\to0.169\to0.175$),
  只能靠加 ICC 填充($L{=}2$)压迭代。
- **sASM 的时间最优在 $O=2$**:把 overlap 变成净收益($0.150\to0.127\to\mathbf{0.124}\to0.132$)。
- sASM 最优 **0.124 s / 73 步**,比 BASIC 最优 **0.145 s / 87 步** 快约 **14\%** 且迭代更少。
- 每步成本几乎相同(1.70 vs 1.67 ms)——sASM 靠 overlap 省迭代取胜,而这条路 BASIC 走不了;
  且 sASM 最优用**更低**的 ICC 填充($L{=}1$ vs $L{=}2$)。

**8.3 计时口径声明 + 构造摊销下的精确 Cholesky 对照**

**口径声明**:本报告(§8–§10)所有 "solve-only(warm)" 时间均为
**预条件只构造一次、随后纯求解**的口径——`-warmup` 先做一次不计时的解,触发全部构造
(ICC/Cholesky 分解、overlap 延拓、sASM multiplicity、Chebyshev 特征值估计、GMRES workspace),
计时的第二次解完全复用。心脏代码中矩阵固定、每个时间步复用同一预条件,**正是此场景**:
构造成本被摊销,**solve 时间才是决定量**。各方法的最优 $(O,L)$ 也是在此口径下选出的。

**精确 Cholesky 在"建一次"假设下仍出局**。早前报的"精确 Cholesky 4–9 s"含分解;
摊销后重测(warm,best-of-5,Sys3):

| BASIC+精确 Cholesky | $O=2$ | $O=3$ | $O=5$ | $O=8$ |
|---|---|---|---|---|
| iter | 33 | 29 | 24 | 19 |
| solve-only | 0.441 s | 0.424 s | 0.439 s | 0.423 s |
| 每步 apply | ~13 ms | ~15 ms | ~18 ms | ~22 ms |

(迭代数与早前冷启动测量有 ±3 出入,系达容差边缘的浮点次序差,不影响结论。)

即使分解免费,完整 3D Cholesky 因子的**每步三角回代 ~13–22 ms 是 ICC 方法(~2 ms)的
约 10×**:迭代最少(19)却 solve-only 仍 ~0.42 s,比 ICC 四方法的最优(0.102–0.145 s)
**慢约 3–4 倍**,且因子内存极大。**决定 solve 时间的是"迭代数 × 每步 apply 成本",
低填充 ICC 恰好两头都便宜**——这就是为何最优解始终落在 ICC 一侧。

---

## 9. 四方性能对比:BASIC / sASM / sASM+Chebyshev / RAS+GMRES

**目的**:在公平计时下,把四种单层方法各自调到最优,比较迭代、时间、全局归约、内存。
**口径**:solve-only(`-warmup`,best-of-5);归约由 `-log_view` 的 `MPI Reductions` 读出。
数据 `ras_time_nx48_n4.csv`、`cheby_time_nx48_n4.csv`(脚本 `collect_ras_time.sh` / `collect_cheby_time.sh`)。

**9.1 RAS+GMRES 的 restart × $L$ 时间权衡**(solve-only 秒,括号内迭代;$O=2$):

| restart | 30 | 60 | 120 | 240 | 600 |
|---|---|---|---|---|---|
| $L=0$(ICC0,PCApply 便宜) | **0.175**(136) | 0.219(127) | 0.201(92) | 0.195(92) | 0.193(92) |
| $L=2$(ICC2,PCApply 贵) | 0.126(79) | **0.113**(57) | 0.112(57) | — | — |

- $L=0$:正交化相对昂贵 → **小 restart 最快**;$L=2$:正交化被昂贵 ICC2 求解掩盖 → **大 restart 最快**。
- 经典 Gram–Schmidt 把一步所有正交化内积合并成**一次** `VecMDot` 全局归约,故 GMRES 归约**不比 CG 多**
  ("GMRES 归约多 10×"只对**修正** GS 成立)。**RAS+GMRES 最优:$O3,L2,r60$ → 51 步 / 0.102 s。**

**9.2 sASM+Chebyshev 的阶数权衡**(solve-only 秒,括号内迭代;$O=2,L=1$):

| Chebyshev 阶 | 2 | 3 | 4 |
|---|---|---|---|
| iter | 48 | 39 | 39 |
| time | **0.115** | 0.118 | 0.143 |

阶数越高 → 外层迭代越少,但每步多做 deg 次 ICC,时间反增。**deg=2 为时间甜点**;
Chebyshev 步在子域内(`COMM_SELF`),**零额外全局归约**。**最优:$O2,L1,\text{deg}2$ → 48 步 / 0.115 s。**

**9.3 四方最优对比**(Sys3,$n_x{=}48$,4 ranks):

| 方法 | 最优配置 | 外层 iter | solve-only | 全局归约 | Krylov 内存 | 对称/CG |
|---|---|---|---|---|---|---|
| BASIC | $O0,L2$ | 87 | 0.145 s | 310 | ~6 向量 | 是 |
| sASM | $O2,L1$ | 73 | 0.124 s | 285 | ~6 向量 | 是 |
| **sASM+Cheby(d2)** | $O2,L1$ | **48** | **0.115 s** | 278 | ~10 向量 | **是** |
| **RAS+GMRES(r60)** | $O3,L2$ | 51 | **0.102 s** | **233** | ~62 向量 | 否 |

- **时间**:RAS+GMRES (0.102) < sASM+Cheby (0.115) < sASM (0.124) < BASIC (0.145)。
- **外层迭代**:sASM+Cheby (**48,最少**) < RAS+GMRES (51) < sASM (73) < BASIC (87)。
- **归约**:四者在各自最优点落在 233–310;RAS+GMRES 最低(经典 GS 合并归约 + 迭代少)。

**选型**:
- **要对称 / CG / 低内存(心脏 HPC 常见约束)**:**sASM+Chebyshev(deg 2)** 最优——外层迭代最少、
  时间快过纯 sASM、内存仍低;每步多做本地 ICC、零额外全局归约,换更少外层同步。
- **内存充裕、可放弃对称**:RAS+GMRES 最快、归约最少,但内存约 $10\times$。
- **最省事**:纯 sASM(无 deg/restart 可调)。
- **BASIC 最差**(困在 $O{=}0$)。

---

## 10. 跨系统验证:四方法在 Sys1 / Sys2 / Sys3 上的最优

**目的**:把四种方法在三个模型问题上各自调到最优,看排名是否一致。
**方法**:同一公平口径(`-warmup` solve-only,best-of-3/5)。
- **Sys1** = reaction–diffusion $(1/\Delta t)M+\tfrac12K$,$\Delta t=10^{-2}$(`asm_demo -dt 1e-2`);
- **Sys2** = 全 Neumann 奇异 Laplace,$\ker=\mathrm{span}\{1\}$(`asm_demo -pure_neumann`,
  `MatNullSpaceCreate(const)` + RHS 去均值;真残差 $1.3\times10^{-6}$,解 mean $\approx0$);
- **Sys3** = 1 面 Dirichlet + 5 面 Neumann Laplace(主线)。
数据 `sys1_nx48_n4.csv` / `sys2_nx48_n4.csv`(脚本 `collect_sys1.sh` / `collect_sys2.sh`)。

| 方法 | Sys1(dt=1e-2) | Sys2(pure Neumann) | Sys3(1D+5N) |
|---|---|---|---|
| BASIC | $O0,L1$ · 26 · 0.048 s | $O1,L2$ · 67 · 0.121 s | $O0,L2$ · 87 · 0.145 s |
| sASM | $O2,L1$ · 18 · 0.041 s | $O2,L2$ · 54 · 0.110 s | $O2,L1$ · 73 · 0.124 s |
| **sASM+Cheby(d2)** | $O3,L1$ · **11** · 0.042 s | $O2,L1$ · **42** · 0.106 s | $O2,L1$ · **48** · 0.115 s |
| **RAS+GMRES(r60)** | $O3,L1$ · 15 · **0.024 s** | $O2,L2$ · 40 · **0.072 s** | $O3,L2$ · 51 · **0.102 s** |

(每格为 最优配置 · 外层迭代 · solve-only 时间。)

**结论(三系统一致)**:
1. **时间排名恒为** RAS+GMRES $<$ sASM+Cheby $\lesssim$ sASM $<$ BASIC。
2. **外层迭代**:sASM+Cheby 最少(或与 RAS+GMRES 并列),BASIC 最多。
3. **RAS+GMRES 的领先在良态问题上最大**(Sys1 ~1.7$\times$、Sys2 ~1.5$\times$),Sys3 最小($\sim$1.13$\times$);
   但其代价(~10$\times$ 内存、非对称)与系统无关。
4. **sASM+Cheby $\approx$ sASM 在良态 Sys1**(Chebyshev 省的迭代被每步 deg$\times$成本抵消),
   但迭代恒最少;在 ill-conditioned 的 Sys3 上 Chebyshev 的时间优势更明显。
5. **反常 + 修复在三系统都成立**:如 Sys2 BASIC($L{=}0$)随 overlap $102\to110\to134$(↑,**奇异系统也发病**),
   sASM $102\to75\to74\to74$(↓,修好)。

**10.1 Sys1 在更小(心脏量级)$\Delta t$ 下的稳定性检验**($\Delta t=10^{-4}$;
数据 `sys1b_nx48_n4.csv`,脚本 `collect_sys1b.sh`):

| 方法 | 最优配置 | iter | solve-only |
|---|---|---|---|
| BASIC | $O0,L0$ | 6 | 0.012 s |
| sASM | $O1,L0$ | 4 | 0.009 s |
| sASM+Cheby(d2) | $O1,L0$ | 4 | **0.014 s(最慢)** |
| **RAS+GMRES(r30)** | $O2,L1$ | 2–3 | **0.005 s** |

- **反常仍在但很轻**:BASIC($L0$)随 $O$:$6\to8\to9$;sASM 修复:$6\to4\to4$。
- **排名的稳定部分**:RAS+GMRES 恒最快(此处优势最大,~1.8×)、sASM $<$ BASIC 恒成立。
- **排名的不稳部分**:sASM+Cheby 从病态端的第 2 名**掉到末位**——系统只需 ~4 步,
  Chebyshev 省不出迭代($4\to4$),却照付每步 ~1.6× 加价(0.014 vs 0.009 s)。
  "省的迭代 × 每步加价"的权衡在良态端变成净亏:**Chebyshev 的价值只在病态端**
  (Sys3、大 $\Delta t$);心脏小 $\Delta t$ 下要对称/CG 就用纯 sASM,sASM+Chebyshev 属过度设计。

---

## 11. 边界条件对照:全 Dirichlet Laplace 上的 overlap 行为

**目的**:排除"反常与 Neumann 边界有关"的假设——把 Sys3 的 1 Dirichlet + 5 Neumann
换成 **6 面全 Dirichlet**($u=0$ 于全部边界,完全钉死、非奇异、条件数更好),
看 BASIC+CG+ICC 的迭代数随 overlap 的趋势是否仍反常。
**方法**:`asm_demo -all_dirichlet`(新开关;`ess_bdr` 全标记)。正确性:实测
$\max u = 0.05618$,与单位立方体全 Dirichlet Poisson($f{=}1$)的级数解中心值
$u(\tfrac12,\tfrac12,\tfrac12)\approx0.0562$ 一致;真残差 $1.5\times10^{-6}$。
数据 `sysD_nx48_n4.csv`(脚本 `collect_sysD.sh`),$n_x{=}48$,4 ranks。

**11.1 BASIC + ICC(迭代次数):**

| $O\backslash L$ | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 0 | 60 | 51 | 46 | 42 | 40 | 38 |
| 1 | 74 | 50 | 42 | 36 | 32 | 29 |
| 2 | 82 | 53 | 43 | 35 | 30 | 27 |
| 3 | 93 | 60 | 46 | 36 | 31 | 27 |
| 4 | 94 | 62 | 48 | 37 | 30 | 26 |
| 5 | 92 | 61 | 48 | 37 | 30 | 26 |
| **随 $O$** | ↑ | ↑(浅谷后升) | ≈ | ↓后平 | ↓ | ↓ |

**11.2 对照行**(同问题):sASM($L{=}0$)$60\to51\to51\to50\to50\to50$(↓);
sASM($L{=}2$)$46\to34\to33\to32\to32\to32$(↓);
BASIC+精确 Cholesky $34\to21\to20\to18\to15$(↓,经典行为)。

**结论**:
1. **全 Dirichlet 同样发病**:$L{=}0$ 时 $60\to74\to82\to93\to94$,
   $O{=}4$ 比 $O{=}0$ 高 **+57%**——与 Sys3(1D+5N,$132\to203$,+54%)**相对幅度几乎相同**。
   反常与 Neumann 面**无关**。
2. **绝对迭代数约为 Sys3 的一半**(60 vs 132 起点):全钉死 ⇒ 条件数更好,
   但这只平移了水平,**不改变随 overlap 的方向**。
3. **翻转点略提前**:$L\approx3$(Sys3 为 $L\approx4\text{–}5$)——子块条件更好 ⇒ 同样填充下
   ICC 的 $\omega$ 更小 ⇒ 因素 A 更早退场,与机理一致。
4. sASM 与精确 Cholesky 在全 Dirichlet 上同样恢复"overlap 有益"。
   至此,反常 + 修复已在**四种边界/算子变体**上复现:混合边界(Sys3)、全 Neumann 奇异(Sys2)、
   reaction–diffusion(Sys1)、全 Dirichlet(本节)——**该现象由 A(ICC 不精确)× B(重叠重复计数)
   决定,与边界条件类型无关**。

---

## 12. 综合结论

| 方法 | 对称 | 配 CG | 配 GMRES | 配 FCG | overlap 趋势 |
|---|---|---|---|---|---|
| BASIC | 是 | ✓ 但**反常↑** | — | — | 132→148→179 ↑ |
| sASM $D^{-1/2}$ | 是 | ✓ **最优** | — | — | 132→104→103 ↓ |
| exp2 $D^{-1}$ 两侧 | 是 | ✓ 次优 | — | — | 132→123→128 ~ |
| exp3 $D^{-1}$ 单侧 | 否 | **✗ 发散** | ✓(restart 限) | ✓ 103/102 | — |
| RAS | 否 | 飘忽($O{=}1$ 发散) | ✓ 满GMRES 129→95→92 ↓ | ✓ **97→93 ↓** | — |

1. **对称性是 CG 的硬约束**:非对称 PC(exp3 / RAS)配 CG 不行;**FCG 能救**,且短截断
   (mmax=30)即达满 GMRES 水平 → RAS+FCG 是"保 CG 族 + 处理非对称"的最佳折中。
2. **GMRES 高迭代数主要来自 restart=30**,不是正交化;放大 restart 或换 FCG 回落到 ~95。
3. **对称路线里 sASM($D^{-1/2}$)最优**;过量($D^{-2}$)次优;无归一化(BASIC)反常上升。
4. 同样总量 $D^{-1}$,**RAS 的 0/1 权重 ≳ 分数 $1/m_k$ 单侧**(FCG 97/93 vs 103/102;
   满 GMRES 95/92 vs 101/99)。
