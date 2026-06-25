# 推进开放理论点 ⑤:为什么 "overlap↑ ⇒ iter↑" —— 闭式判据与可复现谱测量

**对象**:[`REPORT_unified_zh.md`](./REPORT_unified_zh.md) §7.4 命题 ⑤
> "**病灶 A(子域解不精确)与病灶 B(over-counting)同时存在 ⇒ κ(M_BASIC⁻¹A) 随 overlap δ 严格上升**"
> ——当时状态:**机制 + 实证,无闭式定理**。缺口在于:抽象上界
> $\kappa\le C_0^2\,\omega\,N_c$ 把 $\omega、N_c$ 当作**与 δ 无关的常数**,
> 故上界本身**只预测 overlap "少帮忙",不预测严格上升**。

**本报告做了什么**:把那一行抽象上界里的三个常数,逐个换成**随 δ 的定量律**,
给出一条**闭式反常判据**,并用一套**自包含、无需 PETSc/MFEM 的 numpy 复现**
(`asm_bug_demo/theory_overlap.py`)把每一步**实测验证到位**。结论:

> 反常的充要驱动是 **"不精确度本身随 overlap 增大"**,即 $\omega=\omega(\delta)\uparrow$,
> 其闭式律是经典**未修正 IC(0) 的 $\Theta(L^2)$ 阶**作用在被 overlap 撑大的子块
> $L=H+2\delta$ 上;**仅仅 $\omega>1$(常数不精确)不足以反常**(本报告给出反例对照)。
> over-counting B 提供一个**与 δ 无关、被 $\hat N$ 封顶的台阶**;真正"随 δ 变坏"的是 A。

新增代码与图:`asm_bug_demo/theory_overlap.py`、`asm_bug_demo/infoprop_out/fig14_theory_overlap.png`。

---

## 0. 复现工具与口径(先说清楚"测的是什么")

`theory_overlap.py` 用 numpy/scipy **逐字复现** `asm_bug_demo/coloring.c` 的
BASIC/sASM × 精确/ICC(0) 构造(无 PETSc):

- $A$ = $n^d$ 网格上的 **Dirichlet Laplacian**(对角 $2d$、邻居 $-1$),SPD;
- 子域 = $P^d$ 个核盒,各向外扩 $O$ 层、裁剪到边界(= coloring.c 的盒式分解);
- $A_i=R_iAR_i^\top$(主子矩阵);**精确** $\widehat A_i^{-1}=A_i^{-1}$;
  **ICC(0)** $\widehat A_i^{-1}=(LL^\top)^{-1}$,$L$=无填充不完全 Cholesky;
- $M_B^{-1}=\sum_iR_i^\top\widehat A_i^{-1}R_i$(BASIC);
  $\mathbf D=\mathrm{diag}(m_k)$;$M_S^{-1}=\mathbf D^{-1/2}M_B^{-1}\mathbf D^{-1/2}$(sASM);
- $\lambda_{\min},\lambda_{\max}$ 由对称相似 $A^{1/2}M^{-1}A^{1/2}$ 的端点特征值精确算出
  (非 CG 估计,**直接稠密对角化**,故是真值)。

**自检(与三份既有报告对齐)**:1D 自然序 ICC **逐位等于**精确
(下 §2 表 $\omega\equiv1$);$\lambda_{\max}(\text{BASIC,精确})$ **逐位等于** $\hat N$
(Fig 8 的核心测量,本报告 §2 重测得 $2,4,8$);sASM-ICC 的 $\lambda_{\max}\approx1.25$
(Fig 8);over-counting 惩罚比 $\kappa_B/\kappa_S\approx2.8$(相图 §3.6)。**口径一致,可信。**

---

## 1. 精确变分恒等式(分析的总抓手)

两个 SPD 矩阵的预条件谱由广义 Rayleigh 商给出。因 $M_B^{-1}$ 显式已知(而 $M_B$ 不),用
$\mathrm{eig}(M_B^{-1}A)=\mathrm{eig}(A^{1/2}M_B^{-1}A^{1/2})$ 得到**只含 $M_B^{-1}$ 的恒等式**:

$$
\boxed{\;\lambda_{\max}(M_B^{-1}A)=\max_{x\neq0}\frac{x^\top M_B^{-1}x}{x^\top A^{-1}x},
\qquad x^\top M_B^{-1}x=\sum_i \|R_ix\|^2_{\widehat A_i^{-1}}.\;}\tag{$\star$}
$$

($\lambda_{\min}$ 同式取 min。)这条恒等式把"谱端点"变成"挑一个 $x$ 去试",
是下面所有上/下界的来源。

---

## 2. 病灶 B 的闭式:$\lambda_{\max}(\text{BASIC,精确})=\hat N$,**与 δ 无关、被封顶**

**命题 1(over-counting 的精确刻画).** 取精确局部解,则
$M_B^{-1}A=\sum_iP_i$,其中 $P_i=R_i^\top A_i^{-1}R_iA$ 是到子域空间
$V_i=\mathrm{range}(R_i^\top)$ 的 **$A$-正交投影**($P_i^2=P_i$,$A$-自伴)。于是

$$
\hat N\;\le\;\lambda_{\max}(M_B^{-1}A)\;\le\;N_c .
$$

- **上界 $N_c$**(着色数):经典 Toselli–Widlund。
- **下界 $\hat N$(本报告给出的干净见证).** 取最大重数 DOF $k$($m_k=\hat N$),
  令 $x=e_k$。$e_k$ 落在含 $k$ 的那 $\hat N$ 个子域空间内,故对这些 $i$ 有 $P_ie_k=e_k$。
  在 $A$-内积下
  $$\frac{\langle\sum_iP_ie_k,e_k\rangle_A}{\langle e_k,e_k\rangle_A}
   =\sum_i\frac{\|P_ie_k\|_A^2}{\|e_k\|_A^2}\;\ge\;\hat N$$
  (其余项 $\|P_je_k\|_A^2\ge0$ 只会更大)。$\blacksquare$

**实测(`theory_overlap.py sweep`,精确列)**:盒式分解里 $\hat N=N_c$,故上下界夹紧,
$\lambda_{\max}=\hat N$ **逐位成立**:

| d | 分解 | $\hat N=\lambda_{\max}(\text{BASIC,精确})$ | 随 δ 变化? |
|---|---|---|---|
| 1 | P=8 | 2(O 大时台阶到 3,4) | **台阶,封顶**,非连续增长 |
| 2 | P=4/8 | 4(O 极大才到 9) | **台阶,封顶** |
| 3 | P=2 | 8 | **常数** |

**要点**:病灶 B 的 $\lambda_{\max}$ 贡献是一个**阶梯式、被 $\hat N=2^d$ 封顶的常数**
(overlap 一旦触达 $2^d$ 子域交汇点就到顶,之后不再随 δ 长)。**它本身不是"随 δ 严格上升"的来源。**
真正随 δ 单调变坏的,是病灶 A。

---

## 3. 病灶 A 的闭式:$\omega(\delta)=\Theta\big((H+2\delta)^2/h^2\big)$ —— 缺失的那条律

**Law A.** 不精确因子 $\omega=\kappa(\widehat A_i^{-1}A_i)$ = ICC(0) 对子块的谱等价常数,
等于**经典未修正不完全 Cholesky 在 $d$ 维 Laplacian 上的条件数阶**:

$$
\boxed{\;\omega(\delta)=\kappa(M_{IC0}^{-1}A_i)=\Theta\!\big((L_i/h)^2\big),
\qquad L_i=H+2\delta\;}
$$

即子块**线性尺寸的平方**;$d=1$ 时 $\omega\equiv1$(三对角无填充,IC(0)=精确)。
其增长**全部来自 $\lambda_{\min}(M_{IC0}^{-1}A_i)\propto h^2/L_i^2$**,而局部 $\lambda_{\max}\approx$ 常数(O(1))。

**实测(`theory_overlap.py law`,单子块扫尺寸,无裁剪噪声)**:$\kappa/m^2\to$ 常数:

| $m$ | d=1 $\kappa$ | d=2 $\kappa$ ($\kappa/m^2$) | d=3 $\kappa$ ($\kappa/m^2$) |
|---:|---:|---:|---:|
| 8  | 1.00 | 3.70 (0.058) | 4.05 (0.063) |
| 16 | 1.00 | 11.16 (0.044) | — |
| 24 | 1.00 | 23.20 (0.040) | — |
| 32 | 1.00 | 39.82 (0.039) | — |
| 40 | 1.00 | 61.03 (**0.038**) | — |
| 12 | — | — | 7.69 (0.053) |
| 14 | — | — | 10.01 (0.051) |

$d=2$ 常数 $\to0.038$,$d=3$ 约 $0.05$(前渐近,$m$ 更大才落到常数);
局部 $\lambda_{\max}$ 在 $d=2$ 恒 $\approx1.20$、$d=3$ 恒 $\approx1.10$。
**这就是 §7.4 说"未在文献查到闭式"的那条 $\omega(\delta)$ 单调律**:
它不是新现象,而是 **Gustafsson 经典 IC(0) 的 $\Theta(L^2)$ 阶**,套在被 overlap 撑大的子块上。
overlap 每加一层,子块边长 $+2$,$\omega$ 随 $L^2$ 单调上升——**这是唯一真正"随 δ 变坏"的因子**。

---

## 4. 闭式反常判据:$\kappa$ 比 = $\lambda_{\max}$ 比 / $\lambda_{\min}$ 比

把 ($\star$) 用在相邻两档 overlap $\delta\to\delta'$,定义乘法增量
$\rho_{\max}=\lambda_{\max}(\delta')/\lambda_{\max}(\delta)$、
$\rho_{\min}=\lambda_{\min}(\delta')/\lambda_{\min}(\delta)$,则恒等地

$$
\frac{\kappa_B(\delta')}{\kappa_B(\delta)}=\frac{\rho_{\max}}{\rho_{\min}},
\qquad\boxed{\ \text{反常(}\kappa\uparrow\text{)}\iff \rho_{\max}>\rho_{\min}.\ }
$$

**onset 档(O=0→1,over-counting 开关打开)**:
- $\lambda_{\max}$ 从 block-Jacobi 基线($\approx2$,**与 $d$ 无关**:点/块 Jacobi 的 $\lambda_{\max}(D^{-1}A)\to2$)
  跳到 $\hat N\cdot\omega_{\rm top}$,故 $\rho_{\max}\approx \hat N/2=2^{d-1}$;
- $\lambda_{\min}$ 的改善 $\rho_{\min}$ = overlap 带来的"光滑模被局部解吸收"的程度——
  **精确解全额兑现,ICC 因为恰好抓不住光滑模而被打折**。

于是 onset 反常判据为 $\;2^{\,d-1}>\rho_{\min}^{\rm 局部解}\;$。

**实测验证(`sweep 2dmany`,2D n=64 P=8,O=0→1,逐位对上)**:

| 局部解 | $\rho_{\max}$ | $\rho_{\min}$ | $\kappa$ 比 = $\rho_{\max}/\rho_{\min}$ | 实测 $\kappa$:O0→O1 | 反常? |
|---|---:|---:|---:|---:|---|
| **精确** | $4.00/1.99=2.01$ | $0.0309/0.0092=3.36$ | **0.60** | $216\to129$ ✓ | **否**(↓) |
| **ICC(0)** | $4.30/1.66=2.59$ | $0.0083/0.0050=1.66$ | **1.56** | $334\to519$ ✓ | **是**(↑) |

**两件事一目了然**:
1. $\rho_{\max}$ 在精确/ICC 下**几乎一样**($\approx2=2^{d-1}$,d=2)——over-counting 跳变与精确性无关;
2. **唯一区别是 $\rho_{\min}$**:精确 3.36(overlap 充分改善 $\lambda_{\min}$)→ $\kappa$ 降;
   ICC 仅 1.66(光滑模没被吸收,改善被打掉一半)→ $\kappa$ 升。
   **"A×B 相乘"在此落为一句精确的话:over-counting 把 $\lambda_{\max}$ 抬同样的台阶,
   是不精确把 $\lambda_{\min}$ 该有的 overlap 改善"砍掉",才使台阶压不下去 → 反常。**

维度解释也随之而来:$\rho_{\max}\approx2^{d-1}$,
$d=1$ 时 $=1$(over-counting 台阶恰被 block-Jacobi 基线 2 吃掉)⟹ **1D 永不反常**(与 1D 自然序一致);
$d=2$ 时 $=2$,需 $\rho_{\min}<2$,ICC 勉强触发(弱、非单调,与 §Fig5 "2D 弱"一致);
$d=3$ 时 $=4$,$\rho_{\min}$ 被 ICC 压得更低 ⟹ **强、单调反常**(与真实 Sys3 $132\to148\to179$ 一致)。

---

## 5. 决定性对照:**仅 $\omega>1$ 不够,必须 $\omega(\delta)\uparrow$**

命题 ⑤ 的核心其实是"$\omega$ **随 δ 变坏**",而非"$\omega>1$"。本报告用一个**可调不精确度**
对照实验把这一点钉死。`filter` 局部解 = 精确解,但把最光滑的一半局部模**统一**衰减系数 $\gamma\le1$
(模拟"ICC 抓不住光滑模",但**衰减强度与子块尺寸无关**)。

**实测(`theory_overlap.py cross`,2D n=56 P=7,O=0→1)**:

| $\gamma$ | $\kappa_B$ O0→O1 比 | 反常? | $\kappa_S$(sASM)比 |
|---:|---:|---|---:|
| 1.00(精确) | 0.60 | 否 | 0.41 |
| 0.40 | 0.90 | 否 | 0.36 |
| 0.22 | 0.91 | 否 | 0.36 |
| 0.10 | 0.91 | 否 | 0.36 |
| 0.05 | **0.91** | **否** | 0.36 |

**结论(本报告最关键的新结果)**:**统一(尺寸无关)不精确度** $\gamma$ 再小,
$\kappa$ 比也只**单调逼近并饱和在 0.91 < 1,永不反常**——因为 $\gamma$ 把 O=0 与 O=1 的
$\lambda_{\min}$ 等比例压低,比值不变。**反观真实 ICC(§4)$\gamma$ 不是常数:子块越大越不精确
($\omega\propto L^2$,§3),O=1 的更大子块被压得比 O=0 更狠,$\rho_{\min}$ 才掉到 1.66<2 → 反常。**

> 这就**把命题 ⑤ 的假设精确化**:反常的充要驱动不是"局部解不精确"($\omega>1$),
> 而是"**不精确度随 overlap 单调变坏**"($\omega(\delta)\uparrow$,闭式即 $\Theta(L^2)$)。
> 这是文献里没有显式点出、而本组实验干净隔离出来的一条。

---

## 6. sASM 为何稳健消反常(同一框架的推论)

sASM 用 $\mathbf D^{-1/2}$ 单位分解把 §2 的 over-counting 台阶 $\hat N$ 从 $\lambda_{\max}$ 拿掉
($\lambda_{\max}(\text{sASM,ICC})\approx\omega_{\rm top}=O(1)$,实测恒 $\approx1.25$),于是 onset 的
$\rho_{\max}$ 不再有 $\hat N$ 跳变,反常判据右端塌掉。实测(§4/§5 的 sASM 列):
$\kappa_S$ 在 O=0→1 **不升反降**(334→202),且在整条 $\gamma$ 扫上比值恒 $\approx0.36$、**与不精确度无关**。
sASM **不修 $\omega$ 本身**(§3 的 $L^2$ 增长仍在,只是不再被 $\hat N$ 放大),所以它在
"解不精确"处值钱、精确处中性——与 Fig 3/相图完全一致。

---

## 7. 综合:δ-显式上界与它的非单调性

把三条律代回抽象界,得到一条**δ-显式**的上界(而非把常数当 δ-无关):

$$
\kappa_B(\delta)\;\lesssim\;\underbrace{\hat N}_{\text{B:台阶,封顶}}\;\cdot\;
\underbrace{c_d\,(H+2\delta)^2/h^2}_{\text{A:}\;\omega(\delta)\uparrow}\;\cdot\;
\underbrace{C_0^2(\delta)}_{\text{有利,}\downarrow\text{但被不精确打折}} .
$$

- 经典叙事把 $\omega、\hat N$ 当常数,只剩 $C_0^2(\delta)\downarrow$ ⟹ "overlap 总归有利或中性";
- 代入 $\omega(\delta)=\Theta((H+2\delta)^2)$ 后,**$A$ 项随 δ 单调上升**;它与 $C_0^2(\delta)$ 的下降相争,
  乘积**非单调(U 形)**:存在内点极小 $\delta^\*$,$\delta>\delta^\*$ 后上界本身**严格上升**。
  小 δ 端的反常则由 §4 的 onset 台阶($\hat N$ 跳变未被打折的 $\rho_{\min}$ 吸收)驱动。
- **维度**通过两处进入:$\hat N=2^d$(台阶高度)与 $\rho_{\max}\approx2^{d-1}$(onset 阈值)——
  解释了 1D 不发病、2D 弱、3D 强且单调。

---

## 8. 已证 / 模型 / 实测 边界(诚实声明,更新 §7.4 的表)

| 命题 | 本报告前状态 | **本报告后状态** |
|---|---|---|
| ① $\lambda_{\max}(M_{\rm AS}^{-1}A)\le N_c$ | 已证 | 已证(并补 $\ge\hat N$ 见证下界,命题 1) |
| ② 不精确以 $\omega$ 进入 κ 上界 | 已证 | 已证 |
| ③ 精确解 overlap↑⇒κ↓ | 已证 | 已证 + 实测复现(§4 精确列 216→129→…) |
| ④ 单位分解令 $\lambda_{\max}\le\omega$(sASM) | 已证 | 已证 + 实测($\lambda_{\max}^{\rm sASM,ICC}\approx1.25$,§2/§6) |
| **⑤ A×B ⇒ κ 随 δ 严格上升** | **机制+实证,无闭式** | **闭式判据 $\rho_{\max}>\rho_{\min}$(§4)+ 闭式律 $\omega(\delta)=\Theta(L^2)$(§3)+ 充要性对照(§5)** |

**⑤ 仍未完全闭合的一小块(留给后续/可发表点)**:
本报告给出的是 **(i)** 反常的**充要驱动**($\omega(\delta)\uparrow$,且统一不精确不触发——已实测严格隔离);
**(ii)** $\omega(\delta)=\Theta(L^2)$ 的**阶**(经典 IC(0))与**实测常数**;
**(iii)** onset 的**闭式阈值** $\rho_{\max}\approx2^{d-1}$。**尚缺**的是把 $\rho_{\min}^{\rm IC}(\delta)$
(ICC 下 $\lambda_{\min}(M_B^{-1}A)$ 随 δ 的改善率)写成闭式下界——它需要 ICC 谱误差在重叠带上的
精细局部化估计。目前 $\rho_{\min}^{\rm IC}$ 是**实测**(2D onset 1.66、精确 3.36)。
把这一项闭式化(进而把判据 $2^{d-1}>\rho_{\min}^{\rm IC}(\delta)$ 变成纯解析不等式),
是一个**界定清晰、规模小、可发表**的剩余理论点。

---

## 9. 复现

全部自包含(numpy/scipy,**无需 PETSc/MFEM**;约 1 分钟内逐条):

```bash
cd asm_bug_demo
python3 theory_overlap.py law          # Law A: omega(delta)=Theta(L^2)  (fast)
python3 theory_overlap.py sweep 1d     # 自检:1D ICC==exact,无反常
python3 theory_overlap.py sweep 2dmany # 反常 + sASM 修复 (2D n=64 P=8, ~3 min)
python3 theory_overlap.py cross        # 对照:统一不精确不反常 (~2 min)
python3 theory_overlap.py fig          # 重画 infoprop_out/fig14_theory_overlap.png
```

谱端点用稠密对称对角化(真值,非 CG 估计);全部构造逐字对应 `coloring.c` 的
BASIC/sASM × 精确/ICC(0)。图 `fig14_theory_overlap.png`:(a) Law A 的 $\Theta(L^2)$ 双对数律
(d=1/2/3);(b) 2D 多子域 onset:BASIC+ICC 升(反常)、sASM 不升(修复)。
