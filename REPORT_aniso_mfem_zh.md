# σ 张量下的 ASM overlap 反常:原理、分析与新方法(MFEM / PETSc / MPI 实测)

本报告在**真实生产栈**上重做全部测量:MFEM 4.9(源码编译,MPI + PETSc)+ PETSc 3.19.6
+ OpenMPI + HYPRE 2.28 + METIS 5.1,3D P1 四面体元,METIS 分区,`mpirun -n P` 真并行。

它**取代** [`REPORT_aniso_pu_zh.md`](./REPORT_aniso_pu_zh.md) 的结论部分。那份报告的数字来自
2D numpy 谐振器,其中**三条主要结论在 3D 下不成立**,本报告第 6 节逐条列出撤回内容。

---

## 0. 栈验证(在测任何新东西之前)

| 检查 | 结果 |
|---|---|
| 基线复现 `nx=24, 4 ranks, scheme 0, ICC(0)` | `O=0,1,2 → 71, 80, 100`(README 记录 71, 76, 99) |
| `-pugrade 1` 是否恒等于 scheme 3(sASM) | **逐位一致**,含 `final_pnorm`,$O=1,2,3$ |
| σ=I 时 `-puharm` vs `-puramp` | 相差 ±1 步(**不应**逐位相同:METIS 子域不规则,离散调和斜坡 ≠ 图距离斜坡;2D 谐振器里两者重合只因为子域是规则盒子) |
| scheme 3 是否修好反常 | `71 → 59 → 58 → 58` ✓ |
| 新方法是否解对 | 所有方法收敛到**同一个解**:$\lVert u\rVert$、$\max u$、$\mathrm{mean}\,u$ 在每个 $P$ 上 7 位一致 |

$O=0$ 与 README 逐位一致;$O\ge1$ 差 ≤4 步,因为这台机器的 METIS/PETSc 版本不同 ⇒ 分区不同
⇒ 子域形状不同。**反常本身完整复现。**

---

## 1. 原理

### 1.1 单层加性 Schwarz 的三因子界

$$\kappa(M^{-1}A)\;\le\;\underbrace{C_0^2}_{\lambda_{\min}\ \text{侧}}\cdot\underbrace{\omega}_{\text{ICC 不精确}}\cdot\underbrace{\hat N}_{\text{重叠重复计数}}$$

$\hat N=\max_k m_k$ 是最大重数,$\omega=\lambda_{\max}(M_i^{-1}A_i)$ 是子域解的失真度。

### 1.2 增大 overlap 是一场**赛跑**

overlap 从 $\delta$ 增大,同时做两件相反的事:

- **代价**:被多个子域覆盖的 DOF 变多 ⇒ $\hat N\uparrow$ ⇒ $\lambda_{\max}\uparrow$;
- **收益**:信息跨越子域边界的能力变强 ⇒ $C_0^2\downarrow$ ⇒ $\lambda_{\min}\uparrow$。

$\kappa=\lambda_{\max}/\lambda_{\min}$ 往哪边走,取决于哪一项赢。**"overlap 增大、迭代数上升"
就是收益跑输代价。** 这不是 bug,是这两项的比值。

### 1.3 实测的账本(BASIC + ICC(0),$O{=}0\to3$,`nx=48`,4 ranks)

| $r=\sigma_\ell/\sigma_t$ | 1 | 2 | 5 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|---|---|
| $\lambda_{\max}$ 代价 | 2.78× | 2.82× | 2.86× | 2.85× | 2.85× | 2.92× | 3.02× |
| $\lambda_{\min}$ 收益 | 1.20× | 1.18× | 1.16× | 1.15× | 1.15× | 1.17× | 1.21× |
| $\kappa$ 净变 | 2.32× | 2.39× | 2.46× | 2.47× | 2.47× | 2.50× | 2.49× |
| 迭代比 | 1.66 | 1.62 | 1.60 | 1.58 | 1.49 | 1.50 | 1.49 |
| $\omega$ | 1.163 | 1.132 | 1.147 | 1.194 | 1.276 | 1.487 | 1.657 |

**读法:**

1. $\lambda_{\max}$ 代价 $\approx\hat N\approx3$,**对 σ 免疫**(2.78→3.02)。
2. $\lambda_{\min}$ 收益**又平又小**($\approx1.2\times$)。
3. **赛跑在每一个对比度都输** ⇒ 3D 下反常恒存在(幅度 1.49–1.66),**从不翻转**。
4. $\omega$ 确实随对比度上升(1.16→1.66),但**幅度动态范围只有 1.49–1.66**,
   拿它和 $\omega$ 做相关没有统计意义,本报告**不**声称任何相关系数。

### 1.4 为什么 overlap 的收益这么小 —— 两个直接检验

**检验 A:把 overlap 加到 $O=6$。** 收益几乎不动:

```
r=1,  lambda_min:   O=0 ......................... O=6        增益
  BASIC   8.93e-4 9.81e-4 1.02e-3 1.07e-3 1.12e-3 1.17e-3 1.22e-3   1.37x
  sASM    8.93e-4 9.29e-4 9.31e-4 9.31e-4 9.31e-4 9.31e-4 9.31e-4   1.04x  <- O>=2 后完全冻结
  harm    8.93e-4 9.37e-4 9.47e-4 9.50e-4 9.52e-4 9.53e-4 9.54e-4   1.07x
迭代:  BASIC 118→196→203→202→201   sASM 118→101→…→100   harm 118→93→…→91
```

**在这个配置里 overlap 买不到它该买的东西,多宽都一样。**

**检验 B:增加子域数(缩小 $H$)。** 收益确实涨了,但代价涨得更快:

| $r=1$,$O{=}0\to3$ | $P{=}2$ | $P{=}4$ | $P{=}8$ | $P{=}16$ |
|---|---|---|---|---|
| $\lambda_{\min}$ 收益 | 1.10× | 1.20× | 1.34× | **1.52×** |
| $\lambda_{\max}$ 代价 | 1.48× | 2.78× | 2.86× | **3.43×** |
| 反常幅度 | 1.18 | 1.66 | 1.66 | 1.58 |

$\lambda_{\max}(\text{BASIC})$ 跟踪 $\hat N$,而 METIS 把立方体切得越碎、最大重数越大。
**所以这是一个"规模越大越严重"的问题,不只是"低对比度"的问题。**

### 1.5 权重能做什么、不能做什么

$\sqrt{\text{PU}}$ 对称加权

$$M^{-1}=\sum_i R_i^\top W_i\,\tilde A_i^{-1}\,W_i R_i,\qquad \sum_i w_i(k)^2=1$$

的理论上界是 $\lambda_{\max}\le\omega$。实测($O=3$):

| $r$ | $\omega$ | $\lambda_{\max}$ BASIC | sASM$/\omega$ | **harm$/\omega$** |
|---|---|---|---|---|
| 1 | 1.163 | 4.351 | 1.08 | **1.00** |
| 10 | 1.194 | 4.475 | 1.02 | **1.00** |
| 100 | 1.657 | 5.624 | 1.00 | **1.00** |

**σ-调和 PoU 在 $O\ge2$ 时精确达到理论上界**;$1/\sqrt{m_k}$ 在低对比度下超标 8%。

**但这就是权重的天花板。** 权重只能动 $\lambda_{\max}$ 那一项,而 $\lambda_{\max}$ 已经被压到
$\omega=O(1)$。剩下的全部在 $\lambda_{\min}$,**只有粗空间动得了它**。

---

## 2. 分析:五种方法的横向对比

**口径**:`nx=48`(117k DOF),$O=2$,ICC(0),纤维 $(1,1,1)$,
**`-ksp_norm_type unpreconditioned`,rtol $10^{-6}$ 作用在真残差 $\lVert b-Ax\rVert/\lVert b\rVert$ 上**。

> **为什么必须用真残差判据。** 跨预条件子比较时,`-ksp_norm_type preconditioned` 是**不公平的**:
> 容差作用在 $\lVert r\rVert_M$ 上,而加粗空间会改变 $M$($\lambda_{\max}$ 从 1.17 涨到 1.99),
> 同样的 $\lVert r\rVert_M$ 对应**更松**的真残差。实测:`nx=48, P=8, r=100` 下
> `harm+coarse` 停在 $3.1\times10^{-5}$,而 `harm`/`sASM` 停在 $2.6\times10^{-6}$
> ——松 10 倍,凭空多出约 30 步的"优势"。本报告早期版本(commit `c7aba44`)受此影响,已撤回。

### 2.1 $r=1$(各向同性)

| 方法 | $P{=}2$ | $P{=}4$ | $P{=}8$ | $P{=}16$ | $P_{16}/P_2$ |
|---|---|---|---|---|---|
| BASIC | 151 | 182 | 205 | 218 | 1.44 |
| sASM(你现在) | 112 | 109 | 115 | 118 | 1.05 |
| **harm(NEW-B)** | **108** | **98** | **99** | 109 | 1.01 |
| **harm+coarse** | 114 | 119 | 114 | **104** | **0.91** |
| mult+coarse | 126 | 130 | 122 | 114 | 0.90 |

### 2.2 $r=100$(强各向异性)

| 方法 | $P{=}2$ | $P{=}4$ | $P{=}8$ | $P{=}16$ | $P_{16}/P_2$ |
|---|---|---|---|---|---|
| BASIC | 243 | 305 | 333 | 355 | 1.46 |
| sASM(你现在) | 178 | 191 | 201 | 218 | 1.22 |
| harm(NEW-B) | **168** | **188** | 188 | 201 | 1.20 |
| **harm+coarse** | 184 | 184 | **166** | **143** | **0.78** |
| mult+coarse | 255 | 253 | 235 | 231 | 0.91 |

**正确性**:所有方法在每个 $P$ 上 $\lVert u\rVert$ 完全一致。

### 2.3 四条结论

1. **所有单层方法随子域数变差,所有两层方法随子域数变好。**
   $P_{16}/P_2$:BASIC 1.44/1.46,sASM 1.05/1.22,harm 1.01/1.20 **vs** harm+coarse 0.91/**0.78**。
   这是可扩展性的分水岭。
2. **harm 在单层里恒优于 sASM**:$r{=}1$ 快 4–14%,$r{=}100$ 快 2–8%。
3. **粗基复用在各向异性下是决定性的**:$r{=}100$, $P{=}16$ 时
   σ-调和粗基 **143** vs 经典 $1/m_k$ 粗基 **231**,**少 38%**。
   而且 `mult+coarse`(231)**比单层 sASM(218)还差** —— 经典 Nicolaides 粗基在各向异性下
   主动帮倒忙,其 $\lambda_{\max}$ 会炸到 7.4(σ-调和粗基恒为 1.98)。
4. **粗空间的回本点与对比度有关**:$r{=}100$ 时 $P{=}8$ 起回本;$r{=}1$ 时要到 $P{=}16$ 才勉强回本。
   机理:加粗空间在 $\lambda_{\max}$ 上是**固定代价 $\approx+0.8$**(1.17→1.99,即两层 ASM 理论里那个 "+1"),
   只有 $\lambda_{\min}$ 的提升超过它才划算。

### 2.4 迭代数不是时间

solve-only 墙钟(`-warmup`,best-of-3,同一判据):

| $r$ | $P$ | BASIC | sASM | harm | harm+coarse |
|---|---|---|---|---|---|
| 1 | 8 | 205 / 0.215s | 115 / 0.134s | **99 / 0.116s** | 115 / 0.256s |
| 1 | 16 | 218 / 0.316s | 118 / 0.174s | **109 / 0.169s** | 104 / 0.452s |
| 100 | 8 | 333 / 0.378s | 201 / 0.256s | **188 / 0.242s** | 166 / 0.390s |
| 100 | 16 | 355 / 0.584s | 218 / 0.377s | **201 / 0.371s** | 143 / 0.713s |

**粗空间每步贵约 2×,在这台机器上净亏**($r{=}100,P{=}16$:143 步/0.713 s vs 201 步/0.371 s)。
当前实现用 `PCREDUNDANT` 做粗解 —— 每次 apply 一次全局 gather/scatter,加上两次
带跨进程列的 `MatMult`。**注意计时环境不代表真实 HPC**:4 物理核上跑 8/16 进程是超订的,
通信代价被严重放大。**这条是实现与环境的问题,不是算法的问题**,但在优化之前它就是事实。

---

## 3. 新方法

$$\boxed{\;M^{-1}\;=\;\underbrace{R_0^\top A_0^{-1}R_0}_{\text{粗:抬 }\lambda_{\min}}\;+\;\underbrace{\sum_i R_i^\top X_i^{\sigma}\,\tilde A_i^{-1}\,X_i^{\sigma} R_i}_{\text{细:压 }\lambda_{\max}\text{ 到 }\omega}\;}$$

**两层都由同一个 $\chi_i$ 生成**:

- $\chi_i$ = **σ-调和延拓**:$A_i\chi_i=0$ 于重叠带,$\chi_i=1$ 于 owner 核心,
  $\chi_i=0$ 于人工边界(隐式,在子域之外)。
  这是 $\int\sigma|\nabla\chi|^2$ 的极小元 —— 而这个泛函正是 $C_0^2$ 界里出现的量。
  你现有的所有权重($1/m_k$、flat $\varepsilon$、$q^{\text{depth}}$)都是**图距离**造的,与 σ 无关;
  用各向同性的尺子量各向异性的算子,必然沿纤维掐过头、横向掐不够。
- **细层**:$X_i^\sigma=\mathrm{diag}(\sqrt{\chi_i})$,精确全局重归一 $\sum_i w_i^2=1$
  ⇒ 对称 SPD,CG 合法,$\lambda_{\max}=\omega$ **精确达界**。
- **粗层**:$\phi_i=\chi_i$ 线性重归一 $\sum_i\phi_i=1$(Nicolaides),
  $A_0=R_0AR_0^\top$。**复用同一个 $\chi_i$,粗空间构造零增量成本。**
- 成本:setup 期每子域一次 `COMM_SELF` 的 CG+ICC(0)(松容差即可),**每步零加价**;
  粗空间维数 = 子域数。心脏代码矩阵跨时间步不变,setup 完全摊销。

### 实现(已在 `asm_demo.cpp`,已编译、已运行)

| 开关 | 含义 |
|---|---|
| `-aniso r` / `-aniso_raw` / `-fiber ax,ay,az` | σ 张量($\sigma_t I+(\sigma_\ell-\sigma_t)ff^\top$,默认 det 归一) |
| `-scheme 7 -puramp` | δ-归一 ramp 权重(NEW-A) |
| `-scheme 7 -puharm` | **σ-调和权重(NEW-B)** |
| `-scheme 7 -puharm -pucoarse` | **+ 粗基复用(完整新方法)** |
| `-spectrum` / `-probe_omega` | 测 $\lambda_{\min},\lambda_{\max}$ / $\omega$ |

### 诚实的选型建议

| 场景 | 推荐 | 依据 |
|---|---|---|
| **现在就换、无条件更好** | **`-puharm`**(单层) | 迭代与墙钟在**所有**测试配置里都优于 sASM;改动局限在 setup |
| 子域数多($P\gtrsim8$)且各向异性强 | `-puharm -pucoarse` | 迭代少 29%($r{=}100,P{=}16$),且是唯一随 $P$ 下降的 |
| 但要先做 | **优化粗解** | 当前每步贵 2×,墙钟净亏;`PCREDUNDANT` 应换成更便宜的粗层通信 |
| 绝不要 | 经典 $1/m_k$ Nicolaides 粗基 | 各向异性下 $\lambda_{\max}$ 炸到 7.4,比单层还差 |

---

## 4. 一句话原理

> 增大 overlap 同时抬高 $\lambda_{\max}$(重复计数,$\approx\hat N$,与 σ 无关)和 $\lambda_{\min}$
> (跨子域传播)。3D 真实配置下前者涨 ~2.8–3.0×、后者只涨 ~1.2×,**赛跑恒输**,所以反常在
> 每个对比度、每个 overlap 宽度上都存在,并且**随并行规模加重**。
> 权重(包括你现在的 multiplicity scaling)只能修 $\lambda_{\max}$ 那一项,
> σ-调和 PoU 把它修到理论极限 $\lambda_{\max}=\omega$ 后**再无余地**;
> 要继续,只能上粗空间 —— 而经典粗基在 σ 张量下会失效,**必须换成同一个 σ-调和 $\chi_i$**。

---

## 5. 复现

```bash
# 工具链(Ubuntu 24.04)
apt-get install -y libopenmpi-dev openmpi-bin libpetsc-real-dev libhypre-dev libmetis-dev
PD=/usr/lib/petscdir/petsc3.19/x86_64-linux-gnu-real
git clone --depth 1 -b v4.9 https://github.com/mfem/mfem
cd mfem && make config MFEM_USE_MPI=YES MFEM_USE_METIS_5=YES MFEM_USE_PETSC=YES \
     PETSC_DIR=$PD PETSC_ARCH= HYPRE_OPT=-I/usr/include/hypre HYPRE_LIB="-lHYPRE" \
     METIS_OPT= METIS_LIB="-lmetis" CXXFLAGS="-O2 -std=c++17" && make -j4 && \
     make install PREFIX=<prefix>

cd asm_bug_demo
make asm_demo MFEM_DIR=<prefix> PETSC_DIR_DERIVED=$PD MPICC=mpicc

./collect_aniso.sh 48 4 1,1,1 0        # 对比度 × overlap,含谱与 omega
./collect_aniso_overlap.sh 48 4 1,1,1 0 # overlap 到 O=6
./collect_aniso_ranks.sh 48 0           # 2/4/8/16 进程
./collect_twolevel_fair.sh 48 0 16      # 两层,真残差判据(主表)
python3 aniso/plot_final.py             # aniso/figD_mfem_final.png
```

数据:`aniso_nx48_n4_L0.csv`、`aniso_overlap_nx48_n4_L0.csv`、`aniso_ranks_nx48_L0.csv`、
`twolevel_fair_nx48_L0.csv`、`twolevel_time_nx48.csv`。

---

## 6. 撤回清单(2D 谐振器的结论在 3D 不成立)

| [`REPORT_aniso_pu_zh.md`](./REPORT_aniso_pu_zh.md) 的说法 | 3D 实测 | 处理 |
|---|---|---|
| "反常是低对比度病,$r\approx10$ 翻转" | 幅度 1.66→1.49,**从不翻转** | **撤回**,仅适用于 2D |
| "高对比度下 sASM 变成净负担(比 BASIC 慢)" | sASM 恒优于 BASIC(−31%～−48%) | **撤回** |
| "harm 比 sASM 快 1.85×" | 快 2–14% | **撤回**,量级改写 |
| "$\omega$ 与反常无关(相关 +0.08)" | 3D 下 $\omega$ 随对比度升(1.16→1.66),但幅度动态范围太小不足以做相关 | **不再声称相关系数** |
| "δ-ramp 能解开 λ_min 的饱和" | $O$ 加到 6,λ_min 仍只涨 1.07× | **撤回** |
| "低对比度下粗空间给 3.8×" | $r{=}1$ 时粗空间要到 $P{=}16$ 才勉强回本 | **撤回**,量级改写 |
| "σ-调和 PoU 优于图距离权重" | 成立,且优势随 $P$ 拉大;$\lambda_{\max}$ 精确达界 | **保留** |
| "经典 $1/m_k$ 粗基在各向异性下失效" | 成立,且更严重($\lambda_{\max}\to7.4$,比单层还差) | **保留并加强** |

---

## 7. 未做 / 已知局限

- **墙钟**只在 4 物理核上测,8/16 进程是超订,通信代价被放大;粗层的 2× 每步开销
  有多少是实现、有多少是环境,**未分离**。
- 粗解用 `PCREDUNDANT`,**未优化**。这是墙钟结论的主要疑点。
- 只测到 $P=16$,$n_x=48$(117k DOF)。真正的可扩展性论断需要更大规模。
- 纤维方向是**常张量** $(1,1,1)$;真实心脏纤维是逐点旋转的场,
  做逐点 `MatrixFunctionCoefficient` 是直接的下一步。
- 只测了 ICC(0)。$L\ge1$ 会降低 $\omega$,可能整体平移结论(§1.3 的 $\omega$ 列已显示它是活变量)。
- GenEO 未实现。$r{=}100$ 下 `harm+coarse` 虽随 $P$ 下降,但 Nicolaides 型粗空间理论上
  对高对比度不稳健,更大规模下可能仍需谱粗空间。
- PETSc 版本为 3.19.6,你的 Mac 是 3.24.4。已知差异:`PCApplyTranspose` 对
  `PC_ASM_RESTRICT` 仅 ≥3.20 是精确转置(见 `REPORT_weighting_ablation_zh.md` §13.2),
  本报告未使用该路径,不受影响。
