# 项目 Prompt:从 ASM 的一个反常出发 —— 区域分解预条件的诊断、改进与推广

> 用途:把整个研究浓缩成一份可交接、可复现、可续做的 prompt。**这是算法(数值线性代数 / 区域分解)研究**,
> 不涉及药物/医学。分支 `cardiac-sim-fakegeo`;所有数字均为容器内实测(MFEM 4.9 + PETSc 3.19,`heart.msh` 10085 dof)。
> **叙事是因果驱动的**:一切从 ASM 的一个"bug"式反常开始,sASM 是它的修正,后续全部工作是顺着病因的延伸。

你是一名数值线性代数 / 区域分解方向的研究者。载体是一个耦合前向 ECG 管线里的三个线性系统,但**真正的对象是
"一层 Additive Schwarz 为什么会反常、怎么修、修完后还差什么"**。

---

## 逻辑主线(一句话)

**ASM 加重叠迭代反而上升(反常)** → 诊断出病因 **"过计数 × 不精确"**(用信息传播 / Green 函数 / 谱来解释)
→ **sASM 用单位分解 $D^{-1/2}$ 精确抵消过计数(改进)** → 由此看清一层方法的**两个根本瓶颈**
$\big(\lambda_{\max}$ 被重数抬高、$\lambda_{\min}$ 是慢全局模$\big)$ → 顺势展开:**参数最优、粗空间(修
$\lambda_{\min}$)、SORAS 强 fine level、跨时间历史、迭代≠墙钟、大规模取舍**。

---

## 〇、舞台:三个待解的线性系统(载体,先交代清楚)

几何:心脏 slab $20\times7\times3$ mm **共形嵌入**躯干 $50^3$ mm(Gmsh `BooleanFragments`,交界面 $\Gamma$ 共享节点);
P1 非结构四面体 FEM(heart 47582 tets,torso 206647 tets);各向异性 $\sigma$,纤维 $\parallel x$。前身是"假几何"
(结构化 FD + 合成前沿),本项目换成真几何、变分一致的 FEM 算子。每步依次解:

- **Sys1 单域 $V_m$**(抛物 / 质量主导 / 局部,**非奇异**):
  $\chi C_m\partial_t V_m=\nabla\!\cdot(\sigma_{\text{mono}}\nabla V_m)-\chi I_{\text{ion}}+I_{\text{stim}}$;绝缘 Neumann;
  IMEX 离散 $(\tfrac1{\Delta t}M+\tfrac12K)V_m^{n+1}=(\tfrac1{\Delta t}M-\tfrac12K)V_m^n+M(-I_{\text{ion}}-I_{\text{stim}})$。
  质量项 $\tfrac1{\Delta t}M$ 使其 SPD 良态(screened Laplacian),~13 迭代。
- **Sys2 $u_e$ 恢复**(椭圆 / 纯 Neumann **奇异** / 全局):$K_{\sigma_i+\sigma_e}u_e=-K_{\sigma_i}V_m$;全 Neumann
  ⇒ $\ker=\mathrm{span}\{\mathbf1\}$;RHS 去均值 + `MatSetNullSpace`。**唯一真难点**,~82–96 迭代。
- **Sys3 躯干 Laplace**(椭圆 / **非奇异**):$\nabla\!\cdot(\sigma_o\nabla\varphi)=0$,$\varphi|_\Gamma=u_e$(Dirichlet),
  体表绝缘;~62–66 迭代。ECG $=\varphi(x_L)-\varphi(x_R)$。
- 耦合:$V_m(t)\to$ Sys2 RHS $\to u_e\to$ Sys3 Dirichlet $\to\varphi\to$ ECG。子域 $=$ MPI rank(METIS),$n_p=$ 子域数。

---

## 一、引子:ASM 出了问题(反常现象)

在这三个算子上用**一层 Additive Schwarz(ASM, BASIC)+ 重叠 $O$ + 本地 ICC(0)** 做 CG 预条件,发现一个"bug"式反常:

> **加重叠 $O$($0\to2$),ASM 的 CG 迭代数不降反升。**

实测最狠:Sys3 $n_p{=}8$,$O{=}0/1/2$:ASM 迭代 $66\to93\to109$(**+65%**),墙钟也更慢。这违反直觉 —— 重叠越多、
子域间信息应该传得越快、迭代应越少。

同时观察到两个佐证线索:
- **分块方式决定重叠是好是坏**:矩阵连续索引分块的重叠让迭代**下降**,METIS 几何分块的 BASIC 重叠让迭代**上升**。
- **触达 ≠ 精度**:Sys2 的 $u_e$ 在 $k{=}1$ 迭代就铺满整个心脏,却仍缓慢收敛 —— 慢的不是触达范围,是光滑全局低模。

**这个反常就是整个研究的起点。**

---

## 二、诊断:为什么反常?(思考 / 可解释性)

**病因 = 过计数(over-count)× 不精确(inexact)。**
- BASIC 把重叠区 dof 的贡献按**重数** $M>1$ **各加一次** ⇒ 预条件 $M^{-1}$ 在重叠区**放大**($\|M^{-1}\|$、
  $\lambda_{\max}$ 偏大),谱被拉坏。
- 本地 ICC(0) 解**不精确**,这份放大误差不被本地精解吸收 ⇒ 迭代升。(两条缺一不可:精确本地解或无重叠都不会反常。)

三个互补的解释视角(为报告/汇报准备):
1. **信息传播**:一层 Schwarz 每迭代把边界势推进**一个子域跳**;要传遍全域需 $\sim$(子域直径)步 ⇒ 迭代随 $n_p$ 增长。
   重叠本应加速这个传播,但 BASIC 的过计数把增益抵消还倒赔。
2. **Green 函数衰减(三系统信息传播不同)**:**Sys1**(抛物 / 质量项 $\sim1/\Delta t$ 屏蔽)⇒ **指数衰减** $\sim\sqrt{D\Delta t}$
   (短程、局部、天然可扩展);**Sys2/Sys3**(纯椭圆)⇒ **代数 $1/r$**(长程、全局)。
3. **谱桥**:$\text{iters}\approx\tfrac12\sqrt{\kappa}\ln(2/\varepsilon)$,$\kappa=\lambda_{\max}/\lambda_{\min}$。
   **过计数抬高 $\lambda_{\max}$(重数);子域尺寸/距离决定 $\lambda_{\min}$(慢全局模)。**

---

## 三、改进:sASM(对症下药)+ 参数最优

**sASM(scaled / restricted ASM)= 用对角单位分解 $D^{-1/2}(\cdot)D^{-1/2}$($D=$ 重数)精确抵消过计数:**
$$M_{\text{sASM}}^{-1}=D^{-1/2}\Big(\textstyle\sum_i R_i^\top A_i^{-1}R_i\Big)D^{-1/2}.$$
效果:$\lambda_{\max}\!\approx\!1$,重叠恢复"越多越快"的正常行为。同一 Sys3 $n_p{=}8$:sASM 迭代 $66\to55\to53$(**−20%**),
$O{\ge}1$ 时迭代更少且**墙钟也更快**。$O{=}0$ 时无过计数,ASM$\equiv$sASM。**结论:重叠区域分解要用 sASM/RAS,不是 BASIC。**

**顺手做参数最优**(sASM 的 $L$=ICC level $\times\,O$=overlap 扫描,按 solve-only 时间取优;setup 在 EP 循环里摊销):
- **椭圆系统(Sys2/Sys3):时间最优 $L{=}1,O{=}1$**(Sys2 12.5 ms/47 it,Sys3 30.7 ms/36 it)。
- **质量主导 Sys1:$L{=}0,O{=}1$**(ICC0 已足,多余 fill 只加成本)。
- **$O{=}1$ 三系统通用最优**;**时间最优 $\ne$ 迭代最优**(迭代最少是 $L{=}2,O{=}2$,但慢约 40%)。

---

## 四、顺藤摸瓜:病因看清后,一层方法还差什么

sASM 修好了 $\lambda_{\max}$(过计数那一半),但诊断也暴露了一层方法的**另一半瓶颈没动**:$\lambda_{\min}$ = 慢全局模,
以及绝对迭代数还偏高。于是这条逻辑线自然往下长出后续所有工作:

- **一层不可扩展(信息一跳/迭代)⇒ 粗空间**:共享网格的 **Nicolaides 粗空间**(每子域一列,张成常数 = Sys2 奇异零空间
  = 慢全局模)给全局耦合、**修 $\lambda_{\min}$**。实测弱扩展:baseline 随 $n_p$ 升(96),两层压平(72,**1.34×**)。
- **想把绝对迭代数打到底 ⇒ SORAS 强 fine level**(重叠 + Robin 传输 $\alpha M_\Gamma$,本地 Neumann 块):
  Sys2 $84\to20$(np=2,**4.2×**)、$96\to54$(np=16,1.8×)。
- **RHS 随时间光滑 ⇒ 跨时间历史复用**:warm-start / Fischer(A-正交历史投影)。整拍 Sys2:cold 31892 → warm −9% →
  Fischer −12%。**次力**(真波是移动前沿、平流主导、QRS 段高秩,远小于合成低维 demo 的 −66%)。
- **但迭代 ≠ 墙钟(关键转折)**:$n_p{=}4$ 实测 —— 粗空间 1.11× 迭代 / **1.21× 更快**(便宜的 $n_p\times n_p$ 粗解);
  SORAS 近似精解本地块 2.31× 迭代 / **慢 20×**(本地精解 5.5 vs 0.11 ms/迭代)。
- **⇒ 大规模(~3000 核,几十 k dof/rank)取舍**:小规模墙钟结论会**翻转**(每迭代 = 一次全局 Allreduce,迭代数就是货币)。
  - **保留**:粗空间(强扩展命根子;稠密复制 $A_0^{-1}$ 是 $O(n_c^3)$ 串行瓶颈 ⇒ 改 `PCTELESCOPE`+MUMPS 或三层,优选
    **GenEO 谱粗空间**);SORAS(本地解改**固定次数 Chebyshev/ICC**,丢 CG-to-1e-10);warm-start(免费)。
  - **丢弃**:near-exact CG 本地解;稠密复制粗解。
  - 实测:MUMPS/SuperLU\_DIST/TELESCOPE/GAMG/ML 可用,**PCHPDDM 未编译**(需 `--with-hpddm` 重配)。
  - 用户现状"全系统 ICC+CG ~90% 强扩展"**合理**(大子区域计算主导 + 迭代温和增长);粗空间/SORAS 是继续往更多核 /
    更细网格推时保住 90% 的保险。

**加速的两条正交轴(总纲)**:迭代数 $=f(\text{初值},\text{预条件})$。① 跨时间(改初值,历史,次力);② 跨系统(改预条件:
sASM 修 $\lambda_{\max}$、粗空间修 $\lambda_{\min}$、SORAS 压绝对迭代)。层级:baseline < 历史 < 粗空间 < SORAS。

---

## 五、结论 + 三个工程坑 + 正确性验证

**结论**:① ASM 的重叠反常 = 过计数×不精确;**重叠 DD 要用 sASM/RAS,不用 BASIC**。② 时间最优参数 = 适度 ICC +
overlap-1(**时间最优 ≠ 迭代最优**)。③ 一层的两个瓶颈:$\lambda_{\max}$(sASM 修)、$\lambda_{\min}$(粗空间修);
绝对迭代靠 SORAS。④ **迭代 ≠ 墙钟**,大规模才翻正。

**三个工程坑(修好才量得到收益)**:
1. MFEM `PetscPCGSolver.iterative_mode` 只在构造函数生效 ⇒ 每步显式 `KSPSetInitialGuessNonzero`(否则 warm≡cold;
   微测:精确初值 88→**0** 迭代)。
2. PETSc 默认 `atol` 作用在预条件后残差、掩盖初值 ⇒ 改 `KSP_NORM_UNPRECONDITIONED` + `atol=1e-8‖b‖`。
3. Fischer 历史必须**滑窗淘汰最旧**(append-only 填满后冻结在早期 QRS 模、平台段失效)。

**正确性(整拍验证,加速为测量模式 ⇒ ECG 逐位不变)**:TP06 动作电位形态;**$u_e\!\sim\!\nabla V_m$(平台期整片归零 ——
最强判据)**;躯干偶极远场驱动 ECG;CV $\approx0.5$ m/s,P1$\to$P8 43 ms。

**代码入口(`forward_ecg.cpp`,均 skip EP loop)**:`-precond`(ASM vs sASM,iters + tot/solve-only ms)、
`-sweep`(sASM $L\times O$ 扫描 + 时间最优)、`-fischer`(跨时间)、`-coarse`(Nicolaides 两层)、
`-soras [-soras_local K]`(强 fine)。交付:`REPORT_SUMMARY_zh.md`、`REPORT_T3_xsys_zh.md`、
`slides_sim_fakegeo_en.{tex,pdf}`(13 页)、`plot_geom_mesh.py`/`fig_geom.png`/`fig_meshview.png`。
