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

## 7. 综合结论

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
