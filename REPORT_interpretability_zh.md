# sASM 的可解释性:逐图详解(信息传播 / 重复计数 / 不精确放大 / 重排序)

最小算例(C+PETSc:`asm_bug_demo/infoprop.c` 与 `asm_bug_demo/reorder.c`;
绘图 `infoprop_out/plot_infoprop.py`、`plot_reorder.py`)。
把"因素 B = 重叠重复计数"、"信息传播"、"不精确被放大"、"重排序制造不精确"变成可见的图。
**每张图都标注它对应哪个问题/设定,并逐一解释每个坐标轴、颜色、曲线、参数。**

---

## 0. 三个设定 + 两个概念前提(务必先读)

**三个设定(贯穿全文,对号入座)**:
- **1D / 2D-exact**(精确子域解,Cholesky):**对照组,没有不精确,因此没有反常**。
- **2D-ICC / 3D-ICC**(不精确子域解,ICC(0)):**反常发生的设定**——sASM 在这里才有用。
- **3D = 真实 Sys3**(1 面 Dirichlet + 5 面 Neumann Laplace,$n_x{=}48$,4 ranks);
  1D/2D 是把同一线性代数现象抽象出来的最小算例。

**概念前提 1 — 精确/不精确求解子区域 = 解什么?(回答你的 Q2)**
- 子矩阵 $A_i=R_iAR_i^\top$ = 把全局 Laplace 限制到子域 $i$(含 overlap)的 DOF。
  它**隐含**:子域之间的人工切割面 → 齐次 **Dirichlet**(丢掉跨界耦合);落在物理边界的面 → 保留物理 BC。
- **精确求解 $A_i^{-1}$(Cholesky)= 精确解一个"带人工 Dirichlet 边界的局部 Laplace 子问题"**。✓ 你说得对。
- **不精确求解(ICC(0))= 解一个被扰动算子 $\tilde A_i=A_i-E_i$ 的精确解**,其中 $E_i$ = ICC 丢掉的填充。
  等价地,是同一个 Dirichlet 子问题**只解到近似**。这个扰动 $E_i$ 就是**因素 A($\omega>1$)的来源**——
  它把局部解推离真实 Dirichlet 解,而误差恰好落在重叠/缝区。

**概念前提 2 — 重排序为什么改变 ICC?(回答你的 Q1,Fig 7 实测)**
- ICC(0) = 不完全 Cholesky,**只保留 $A_i$ 原有的非零位置,丢掉所有填充**。
- 因此 ICC 的质量 = "排序产生多少填充"。**自然序的 1D 子块是三对角 → Cholesky 零填充 → ICC(0) 字面上=精确**。
- 把子域内 DOF **随机重排** → $\hat A_i=PA_iP^\top$ 不再带状 → 精确 Cholesky 产生大量填充 →
  ICC(0) 把这些填充全丢掉 → **ICC 变不精确**($\omega$ 从 1 跳到很大)。

**概念前提 3 — 上界里到底是哪个常数?(订正:$N_c$ vs 重数 $\hat N$)**
抽象界 $\kappa(M^{-1}A)\le C_0^2\,\omega\,N_c$(单层无粗空间;Gander–Halpern–Santugini-Repiquet,
*ESAIM:M2AN* **49**(2):713,2015,定理 2.7;= Toselli–Widlund 定理 2.7)。其中三个整数有严格链
(Gander 等,Remark 2.8):
$$\hat N\ \le\ N_c\ \le\ N_k.$$
- $\hat N$ = **最大重数** = 一个点最多属于几个子域 = $\max_k m_k$ = $\max\mathrm{diag}(\mathbf D)$。**这才是 $\mathbf D$ 度量的"过度计数"**。
- $N_c$ = **着色数** = 子域"相互作用图"(两子域共享 DOF 即相邻)的染色数,$\ge\hat N$。
- $N_k$ = 连通度(邻居数上限),$\rho(E)\le N_k$。
- **订正**:之前写"$N_c$ = 最大重数"是把两者混为一谈。正确说法是 **$m_{\max}=\hat N\le N_c$**;
  **sASM 的 $\mathbf D^{-1/2}$ 单位分解去掉的是重数过度计数 $\hat N$**(而非染色数 $N_c$)。
  在规则盒状分解里 $\hat N=N_c$(本文 Fig 8 实测二者都 = 4 然后 9),所以数值上恰好重合,但一般 $\hat N\le N_c$。

**两个不精确度量**(都在 Fig 7 panel b):
- $\eta=\mathrm{avg}_i\|A_iu_i-v\|/\|v\|$(对一个随机 $v$ 的一次 ICC 求解相对残差)——**便宜、只能当"精确/不精确"的定性开关**,
  $\eta\approx0$=精确、$\eta>0$=不精确;**它不是 $\omega$ 的标定值**(单样本、用的是 $\ell_2$ 范数而非能量范数)。
- $\kappa(M_i^{-1}A_i)=\lambda_{\max}/\lambda_{\min}$(用 KSPCG+ICC 估计)——**严格的局部条件数**;
  上界里的 $\omega$ 严格说 = $\lambda_{\max}(M_i^{-1}A_i)$(单侧上界,$\lambda_{\min}$ 那侧归入 $C_0^2$)。

**总览**(CG 到 $\|r\|/\|b\|<10^{-8}$):

| 设定 | $\eta$(定性) | $\kappa(M_i^{-1}A_i)$(严格) | BASIC | sASM | 说明 |
|---|---|---|---|---|---|
| 1D 自然序 | 1.8e-15 | **1.0** | 15 | 25 | **ICC=精确**(三对角无填充)⟹ 无反常;sASM 略亏 |
| 1D RCM | 2.5e-15 | **1.0** | 15 | 25 | RCM 对 1D 仍带状 ⟹ 仍精确 |
| **1D 随机序** | 0.67 | **88.7** | **113** | 108 | **重排把 ICC 变不精确**;迭代 15→113 |
| 2D 自然序 | 0.17 | 15.1 | 102 | 67 | 2D 五点本就有填充 ⟹ ICC 不精确 ⟹ 反常+修复 |
| 2D RCM | 0.17 | 15.1 | 102 | 67 | RCM≈自然(质量相当) |
| 2D 随机序 | 0.26 | 28.5 | 147 | 92 | 随机使 ICC 更差;sASM 修复幅度更大 |
| **3D ICC(0)**($O{=}2$ 真实 Sys3) | — | — | **214** | **134** | 反常+修复(真实问题) |
| 3D 精确 Cholesky | — | — | 38 | — | 对照:精确很快,无反常 |

**注**:$\eta$ 与 $\kappa$ 完全同向:$\eta\to0\Leftrightarrow\kappa\to1$(精确),$\eta$ 大 $\Leftrightarrow\kappa$ 大(不精确)。
$\kappa$ 是驱动 CG 迭代数的量(迭代 $\sim\sqrt{\kappa}$),$\eta$ 只做定性判断。

---

## 1. 逐图详解(每个参数、每条曲线、结论)

### Fig 1 `fig1_overcount.png` — 一次预条件 apply:重复计数可见
- **【对应问题】** 1D Laplace,$N{=}256$,8 个等分子域,overlap $O$,**精确解**(纯机制,与 ICC 无关)。
- **【坐标/参数】**
  - 横轴 = 物理坐标 $x\in[0,1]$($x_k=(k{+}1)/(N{+}1)$)。
  - 上排纵轴 = multiplicity $m_k$=覆盖点 $k$ 的子域数;阴影带 = $m_k{=}2$ 的重叠区(共 7 条缝)。
  - 下排纵轴 = 对同一个高斯包 residual 施加 $\sum_iR_i^\top A_i^{-1}R_i$ 后的修正 $z(x)$:
    **红 = BASIC**($\sum_iR_i^\top A_i^{-1}R_i$),**蓝 = sASM**($D^{-1/2}(\cdot)D^{-1/2}$)。
- **【看什么】** 红线在每条重叠缝处**隆起**(那里被叠加了 $m_k{=}2$ 次);蓝线被 $1/\sqrt{m_k}$ 归一、保持平滑。
- **【结论】** 这是"因素 B = 重复计数"的**直接可视化**:$\sum_iR_i^\top R_i=\mathrm{diag}(m_k)$,
  BASIC 在重叠区把局部贡献多加了 $m_k$ 倍。**注意:这本身无害**——下面的图说明它何时才致命。

### Fig 2 `fig2_front.png` — 信息传播:为什么单层要 $O(\#\text{子域})$ 次迭代
- **【对应问题】** 1D Laplace,$N{=}256$,8 子域,左边界 Dirichlet$=1$(RHS 只在最左点非零),
  **精确解**(传播是单层 Schwarz 的固有机制,与 BASIC/sASM 无关)。
- **【坐标/参数】**
  - 横轴 = 物理坐标 $x$;纵轴 = 第 $k$ 次平稳迭代后的近似解 $x_k(x)$。
  - 多条曲线 = 迭代步 $k=0,1,2,4,8,16$(由浅到深);灰色虚线 = 精确解 $u^\star=1-x$。
- **【看什么】** 椭圆问题是全局耦合的(改一点影响全场),但每次 Schwarz 只解局部子问题 ⟹
  **左边界的影响每步只向右推进约一个子域**;解从左向右逐步"填进来",深部内区严重滞后。
- **【结论】** 单层 Schwarz 的固有代价 $\approx O(\#\text{子域})$ 次迭代;那条**滞后的慢全局模正是两层法的粗空间**要补的。
  overlap 越大,每步推进越远 ⟹ 迭代越少(这解释了"overlap 在经典精确情形下有益")。**与 BASIC/sASM 无关。**

### Fig 3 `fig3_reshist2d.png` — 核心图:sASM 的收益只在不精确解时出现
- **【对应问题】** 2D Laplace,$64^2$,$4\times4$ 子域,$O{=}2$;**精确(实线)与 ICC(0)(虚线)对比**。
- **【坐标/参数】**
  - 横轴 = CG 迭代次数 $k$;纵轴(对数)= 相对残差 $\|r_k\|/\|b\|$。
  - 四条线:实线深蓝 = **精确 BASIC**(23 it),实线浅蓝 = **精确 sASM**(27 it),
    虚线红 = **ICC BASIC**(102 it),虚线橙 = **ICC sASM**(67 it)。
- **【看什么】** 比较"实线两条"与"虚线两条"各自内部的 BASIC↔sASM 间距。
- **【结论】**
  - **实线(精确,$\omega{=}1$)**:BASIC 23 与 sASM 27 几乎重合——**过度计数本身不拖慢收敛**(甚至 sASM 略亏)。
  - **虚线(ICC,$\omega{>}1$)**:BASIC **102** 明显变慢,sASM **67** 收回近 1/3。
  - **一句话**:**去掉因素 B(sASM)只在因素 A(ICC 不精确)存在时值钱**——这是 A×B 相乘耦合最干净的证据。

### Fig 4 `fig4_heatmap2d.png` — 2D-ICC 残差场:缝处残差清得快慢
- **【对应问题】** 2D Laplace,$64^2$,$4\times4$,$O{=}2$,**ICC(0)**(反常设定)。
- **【坐标/参数】**
  - 每个小图 = 把 $64\times64$ 网格上的 $|r_k|$ 画成热力图;**色标为对数(LogNorm)**,亮=残差大、暗=已清。
  - 上排 = BASIC,下排 = sASM;四列 = **相同的 CG 步数** $k=5,15,30,60$(横向对齐才公平)。
- **【看什么】** 同一列(同一 $k$)上下对比;尤其看子域缝(规则网格线)处和左边界附近的亮度。
- **【结论】** 两者前期都在左边界+缝处堆残差;到 **$k{=}60$,BASIC 仍明显发亮**(它还要到 102 才收敛),
  **sASM 已近乎全暗**(67 已收敛)。直观看到 **sASM 把重叠/缝区的残差清得更快**。

### Fig 5 `fig5_overlap_sweep.png` — 变 overlap:误差在缝处的占比随 overlap 增大
- **【对应问题】** 2D Laplace,$128^2$,$8\times8$ 子域,**ICC(0)**,overlap $O=1,2,4$;固定 $k{=}40$ 的残差场。
- **【坐标/参数】**
  - 上排 = BASIC,下排 = sASM;三列 = $O=1,2,4$;每图是 $128\times128$ 的 $|r_{40}|$ 热力图(LogNorm)。
  - "缝区"= multiplicity $>1$ 的重叠带;其面积占比随 $O$ 为 0.21 / 0.39 / 0.68。
- **【看什么】** 随 $O$ 增大,亮的残差越来越**集中到变宽的缝带**里。
- **【结论】**
  - **被困在缝区的残差能量占比随 $O$ 增大:$0.45\to0.51\to0.64$**(即"误差越来越集中在重叠缝处")。
  - sASM 在**每个 overlap** 下都更优(迭代 BASIC 182/195/185 vs sASM 131/130/129)。
  - **诚实说明**:2D 的迭代数随 $O$ **非单调**(2D 的 ICC 不精确度本就较弱);
    "overlap↑ ⟹ 迭代↑"的**单调反常在 3D 才干净**(见 Fig 6)。

### Fig 6 `fig6_3d.png` — 真实 3D Sys3:反常与修复
- **【对应问题】** **真实 Sys3**(1 Dir + 5 Neu Laplace,$n_x{=}48$,4 ranks)。
- **【坐标/参数】**
  - 左图:横轴 = CG 迭代,纵轴(对数)= 相对残差;三条线 = **BASIC 精确 Cholesky**(38 it,实线)、
    **BASIC ICC(0)**(214 it,虚线)、**sASM ICC(0)**(134 it,虚线)。
  - 右图:横轴 = overlap $O\in\{0,1,2\}$,纵轴 = CG 迭代数;
    **红 = BASIC+ICC**($132\to148\to179$),**蓝 = sASM+ICC**($132\to104\to103$)。
- **【看什么】** 左图比较三条曲线的斜率;右图看两条折线随 $O$ 的走向(上升 vs 下降)。
- **【结论】**
  - **左**:ICC 下 BASIC 214 慢、sASM 134 修复;精确 Cholesky 38 很快(=无反常,对照基线)。
  - **右**:**这才是"误差随 overlap 越积越多"的干净版**——overlap 越大,BASIC+ICC 迭代单调上升
    (误差清得越慢);**sASM 把它反转成"越大越好"**。
  - 与 1D/2D 完全一致:**反常发生在 ICC(不精确)设定,精确解下无反常,sASM 恰好修不精确带来的放大。**

### Fig 7 `fig7_reorder.png` — 重排序把子域 ICC 变不精确(直接验证你的 Q1)
- **【对应问题】** 同一个 1D($N{=}256$,8 子域,$O{=}2$)和 2D($64^2$,$4\times4$,$O{=}2$)Laplace,
  **唯一变量 = 子域内 ICC 的排序**(自然 / RCM / 随机);**子域、overlap、multiplicity 权重全部不变**,
  所以四个子图严格隔离"排序 → ICC 质量"这一条因果。
- **【坐标/参数】**
  - **(a) 左上 = 1D 残差史**:横轴 CG 迭代、纵轴(对数)相对残差;
    深蓝实线 = 自然序 BASIC(15 it)、蓝点线 = RCM BASIC(15 it,与自然重合)、
    红实线 = 随机序 BASIC(113 it)、橙虚线 = 随机序 sASM(108 it)。
  - **(b) 右上 = 不精确度(头条)**:6 组(1D/2D × 自然/RCM/随机)。
    **柱(左轴,对数)= 便宜代理 $\eta$**;**绿色菱形(右轴,对数)= 严格条件数 $\kappa(M_i^{-1}A_i)$**。
    两者同向:1D 自然/RCM $\eta\approx$1e-15、$\kappa=1$(ICC 精确);1D 随机 $\eta=0.67$、$\kappa=89$(不精确)。
    灰虚线 = 精确求解地板。
  - **(c) 左下 = 2D 残差史**:深蓝 = 自然 BASIC(102)、蓝虚 = 自然 sASM(67)、
    红 = 随机 BASIC(147)、橙虚 = 随机 sASM(92)。
  - **(d) 右下 = 迭代数柱状**:6 组,红 = BASIC、深蓝 = sASM,柱顶标数字。
- **【看什么】** 重点看 (b) 的 1D 自然/RCM 两根柱 vs 1D 随机柱;再看 (a)(d) 里 1D 随机如何从 15 炸到 113。
- **【结论】**
  - **(b) 1D 自然序 $\eta=1.8\text{e-}15$、RCM $\eta=2.5\text{e-}15$ = 机器零 = ICC 字面上就是精确 Cholesky**
    (三对角无填充);**1D 随机序 $\eta=0.67$ = ICC 变得严重不精确**——
    **这正面回答你的 Q1:"reorder 之后 1D 的 ICC 就不一样了" ✓**。
  - **(a)(d) 1D**:自然/RCM 都是 15 次收敛;随机序炸到 113 次(因为 ICC 不再精确)。
    **RCM 对 1D 仍是带状 ⟹ 保持精确**(这点和直觉略不同:不是所有重排都破坏精确,只有增带宽的随机序才破坏)。
  - **(c)(d) 2D**:五点 Laplace 本就有填充($\eta\approx0.17$),随机序使其更差($\eta=0.26$),迭代 102→147;
    **sASM 在两种排序下都修复**(102→67、147→92),且 ICC 越差(随机)sASM 修复幅度越大。
  - **大结论**:**"不精确"不是 ICC 的玄学,而是"排序产生的填充被丢弃"这件具体的事**;
    重排序能人为地把 1D 的 ICC 从精确推成不精确,从而把反常/sASM 的价值"打开"。

### Fig 8 `fig8_coloring.png` — 直接测出过度计数 $\hat N$,并证明 sASM 去掉它($\omega\times\hat N$ 相乘)
- **【对应问题】** 2D Laplace,$120^2$,$6\times6$ 子域,扫 overlap $O\in\{1,2,4,8,12,16\}$;
  四种组合 BASIC/sASM × 精确(Cholesky)/ICC(0)。用 **PCSHELL 包住 BASIC/sASM 的 apply + PETSc KSPCG**,
  由 CG-Lanczos 的 Ritz 值经 `KSPComputeExtremeSingularValues` 取出 $\lambda_{\max},\lambda_{\min}$。
- **【坐标/参数】**
  - **(a) 左 = $\lambda_{\max}(M^{-1}A)$ vs $O$**:黑虚阶梯 = 着色数/重数($N_c=\hat N$,本几何二者重合,= 4 然后在 $O{=}12$ 跳到 9——
    此时 overlap 触达第二邻居,角点被 $3\times3{=}9$ 个子域共享);
    红圆实 = BASIC 精确、橙方虚 = BASIC ICC、蓝圆实 = sASM 精确、蓝方虚 = sASM ICC。
  - **(b) 右 = 条件数 $\kappa=\lambda_{\max}/\lambda_{\min}$ vs $O$(对数)**:四条同色线;这是真正驱动 CG 迭代的量。
- **【看什么】** (a) 里 BASIC 两条是否贴着阶梯 $\hat N$、sASM 两条是否压平;(b) 里 ICC 把 $\kappa$ 抬多少、sASM 收回多少。
- **【结论】**
  - **$\lambda_{\max}(\text{BASIC,精确})=\hat N$ 精确成立**:实测 4,4,4,4,**9**,8.7,逐点等于最大重数(过度计数的字面定义)。
  - **$\lambda_{\max}(\text{BASIC,ICC})\approx\omega\hat N$**:略高于 $\hat N$(4.3→10.7),即不精确 $\omega$($\approx$1.1–1.2)**乘上** $\hat N$。
  - **$\lambda_{\max}(\text{sASM,}\cdot)$ 平**:精确 $\approx2$、**ICC 恒 $\approx1.25=\omega$**——
    $\mathbf D^{-1/2}$ 单位分解**去掉 $\hat N$**,只剩 $\omega$。**$\omega\times\hat N$ 的乘积被打断,这是整套说法最直接的测量证据。**
  - (b) ICC 把 $\kappa$ 抬约 $5$–$10\times$;sASM-ICC 的 $\kappa$ 比 BASIC-ICC 小约 $3\times$ ⟹ 迭代 $\sim\sqrt3\approx1.7\times$ 更少(与观测一致)。

---

## 2. 两个核心问题(订正版,带文献)

抽象界(单层、无粗空间):$\boxed{\ \kappa(M^{-1}A)\le C_0^2\,\omega\,N_c\ }$,其中 $\lambda_{\min}\ge C_0^{-2}$、$\lambda_{\max}\le\omega N_c$;
$m_{\max}=\hat N\le N_c\le N_k$。(Gander–Halpern–Santugini-Repiquet,*M2AN* **49**(2):713,2015,定理 2.7 + Remark 2.8;Toselli–Widlund 定理 2.7。)

**Q(inexact 为什么不好)**:$\lambda_{\max}\le\omega N_c$ 里 **$\omega$(不精确)与过度计数因子相乘**。
精确解 $\omega{=}1$,过度计数只把 $\lambda_{\max}$ 抬到 $\hat N$(有界、与网格无关)、CG 容忍(无反常,Fig 3 实线、Fig 8 蓝圆);
ICC 时 $\omega{>}1$,BASIC 把**带误差的修正在重叠区叠加 $m_k$ 次** ⟹ $\lambda_{\max}\approx\omega\hat N$(Fig 8 橙方),
且 overlap 越大 $\omega,\hat N$ 越大 ⟹ 迭代随 overlap 上升(Fig 6 右),残差堆在缝处清不掉(Fig 4/5)。

**Q(sASM 为什么能克服)**:$\mathbf D^{-1/2}(\cdot)\mathbf D^{-1/2}$ 单位分解($\sum_i$ 加权 $R_i^\top R_i=I$)把重叠区计一次、
**去掉重数过度计数 $\hat N$**($\lambda_{\max}\le\omega$;Fig 8 蓝方恒 $\approx1.25=\omega$),**打断 $\omega\times\hat N$ 相乘**
⟹ 不精确修正不再被放大叠加,缝处不累积(Fig 4/5),收敛恢复(102→67、214→134)。
sASM **不修不精确本身**($\omega$ 仍 $>1$),只阻止过度计数去放大它——所以**恰好在子域解不精确时有用**,精确时中性甚至略亏。

> **订正说明**:$\mathbf D=\mathrm{diag}(m_k)$ 度量的过度计数是**最大重数 $\hat N$**(=$\max\mathrm{diag}\mathbf D$),
> 不是染色数 $N_c$;严格链 $\hat N\le N_c\le N_k$。sASM 去掉的是 $\hat N$。规则盒状分解里 $\hat N=N_c$(Fig 8 实测重合),
> 故早期把二者等同在数值上无害,但概念上应区分。$\eta$ 只是定性开关,严格不精确度量是 $\kappa(M_i^{-1}A_i)$、上界常数 $\omega=\lambda_{\max}(M_i^{-1}A_i)$。

---

## 3. 如何"很好地"降低迭代数:给 sASM 加粗空间(两层)—— Fig 9

**诊断**(由 Fig 8 直接读出):sASM 已经把 $\lambda_{\max}\approx\omega=O(1)$ 压住(去掉了 $\hat N$),
**剩下的瓶颈是 $\lambda_{\min}$ 极小**(~0.002)。$\kappa=\lambda_{\max}/\lambda_{\min}\approx590$ 几乎全由 $\lambda_{\min}$ 决定。
$\lambda_{\min}\ge C_0^{-2}$,而单层无粗空间时 $C_0^2$ 随**子域个数增大而爆**(= Fig 2 的"每步只传一个子域"、缺全局耦合)。
**所以再怎么调权重都没用**($\lambda_{\max}$ 已最小);**唯一的大杠杆是抬 $\lambda_{\min}$ = 加粗空间**。

**改进 = 两层对称 sASM**(保持加性+对称 ⟹ 仍可用 CG、仍低内存):
$$\mathbf M_{2}^{-1}=\underbrace{R_0^\top A_0^{-1}R_0}_{\text{粗(抬 }\lambda_{\min})}+\underbrace{\mathbf D^{-1/2}\Big(\textstyle\sum_iR_i^\top A_i^{-1}R_i\Big)\mathbf D^{-1/2}}_{\text{sASM 细(保持 }\lambda_{\max}\approx\omega)}.$$
粗空间用 **Nicolaides 单位分解**:每个子域一个粗基 $\phi_i(k)=1/m_k\ (k\in\Omega_i)$,$\sum_i\phi_i=1$;
$A_0=R_0AR_0^\top$ Galerkin(维数=子域数,Cholesky 直接解)。代码 [asm_bug_demo/twolevel.c](asm_bug_demo/twolevel.c)。

### Fig 9 `fig9_twolevel.png` — 加粗空间把瓶颈 $\lambda_{\min}$ 抬起来、迭代变可扩展
- **【对应问题】** 2D Laplace,ICC(0)子域解。(a)固定 $120^2$、$6\times6$、$O{=}2$;(b)固定子域尺寸($n{=}20P$)、扫子域数。
- **【(a)左 = 固定配置,one-level vs two-level 的 $\lambda_{\min},\lambda_{\max},\kappa$】**(对数柱)
  - 粗空间把 **$\lambda_{\min}$ 抬 14×**(0.0021→0.030),$\lambda_{\max}$ 几乎不动(1.25→2.07),
    **$\kappa$ 降 8.7×**(596→69),**迭代 121→64**(≈ 砍半)。
- **【(b)右 = 可扩展性(决定性)】** 横轴子域数(16→144,子域尺寸固定),纵轴 CG 迭代:
  - **one-level**:81→121→158→188→**217**(随 $\sqrt{\#\text{子域}}$ 增长——单层的"诅咒")。
  - **two-level**:59→64→64→65→**65**(**平**,与子域数无关 = 可扩展)。
- **【结论】**
  - **瓶颈是 $\lambda_{\min}$ 不是 $\lambda_{\max}$**:调权重动不了 $\lambda_{\min}$,**加粗空间**才动得了。
  - **两层 sASM 把迭代数变得与子域数无关**(144 子域时 3.3× 少,且差距随规模继续拉大)——这才是大规模并行真正要的可扩展性。
  - 仍是**对称、加性、CG 可用、低内存**(粗空间维数=子域数,极小),契合"新对称 CG 低内存方法"的目标。

**还能再小幅改进的方向**(次要,带 tradeoff):
- 降 $\omega$:子域用更富的 ICC(level≥1)或 RCM 排序减少丢弃的填充($\omega$ 1.25→~1),代价是内存/时间。
- 增 overlap $\delta$:$\lambda_{\min}$ 随 $\delta$ 略升(Fig 8b 中 $\kappa$ 随 $O$ 下降),但不改子域数的标度——治标不治本。
- 更强粗空间:Nicolaides 已够;难问题(强各向异性/跳系数)可换 **GenEO**(局部广义特征向量),更稳健但设置更贵。

**一句话**:sASM 把"过度计数 $\hat N$"那一项解决了;要"很好地"再降迭代,就**加一个粗空间**把 $\lambda_{\min}$ 那一项也解决——
两层 sASM 迭代砍半且可扩展,是直接面向大规模并行的改进。
