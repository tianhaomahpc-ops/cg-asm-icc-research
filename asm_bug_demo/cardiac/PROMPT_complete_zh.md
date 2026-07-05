# Prompt:从"overlap 反常"到"低维几何慢模"——心脏前向 ECG 三系统的强可扩展预条件研究

> 这是一个**逻辑路线完整**的研究 prompt:问题(含边界条件)→ 发现问题 → 可解释性诊断 → 文献 → 我们的解法 → 结果 → 结论。
> 所有数字均为本仓库实测(`forward_ecg.cpp` 的 `-precond/-sweep/-transmit/-transmiti/-overlap/-fair/-neumann/-decay/-anisocmp/-fischer/-anisocmp` 诊断)。

---

## 1. 问题描述(含边界条件)

**物理**:心脏电生理前向 ECG。真实非结构四面体 P1-FEM(MFEM),心脏 slab(20×7×3 mm,纤维沿 x)conforming 嵌入躯干。每个 EP 时间步顺序解**三个线性系统**:

| 系统 | 方程 | 类型 | **边界条件** | 谱(实测,np=8) |
|---|---|---|---|---|
| **Sys1** 单域 $V_m$ | $A_1V_m=b_1$,$A_1=\tfrac{1}{\Delta t}M+\tfrac12 K_{\sigma_\text{mono}}$ | 抛物、**质量主导**、SPD | Neumann(绝缘心脏) | cond **1.4**,8 迭代,**0 慢模** |
| **Sys2** $u_e$ 恢复 | $K_{\sigma_i+\sigma_e}u_e=-K_{\sigma_i}V_m$ | 椭圆、各向异性、**纯 Neumann → 奇异**(核=常数) | 纯 Neumann(绝缘),奇异 | cond **293**,67 迭代,**12 慢模** |
| **Sys3** 躯干 $\varphi$ | $K_{\sigma_o}\varphi=0$ | 椭圆、各向同性 | **界面 Dirichlet**($\varphi=u_e$)+ 体表 Neumann | cond **172**,51 迭代,**8 慢模** |

- $\sigma_\text{mono}=\sigma_i\sigma_e/(\sigma_i+\sigma_e)$ 是 **harmonic mean**(纵 0.1334、横 0.0176)。
- Sys2 奇异用**零均值 RHS + `MatSetNullSpace`(投影常数核)**处理。
- 规模目标:**~3000 核强可扩展**,数十 k dof/核 → 通信受限区。
- 预条件族:**CG + 加性 Schwarz(ASM/sASM)+ 本地 ICC**,1 子域/rank(METIS)。

---

## 2. 发现问题:overlap 不帮收敛(ASM 反常)

**现象**:经典 **ASM(BASIC,不缩放)**在椭圆 Sys2/Sys3 上,**迭代数随 overlap 上升**(理应下降)——"overlap 反常"。

---

## 3. 可解释性诊断:把迭代数拆到谱上

**理论界(Gander–Halpern–Santugini 2015, Thm 2.7)**:
$$\kappa(M^{-1}A)\ \le\ \underbrace{C_0^2}_{\text{粗空间/全局}}\cdot\underbrace{\omega}_{\text{本地解精度}}\cdot\underbrace{(\hat N+1)}_{\text{过计数}}$$

| 因子 | 含义 | 谱位置 | 治法 | 实测状态 |
|---|---|---|---|---|
| $\hat N$ 过计数 | 重叠区被重复修正 | **λmax 侧** | **sASM**($D^{-1/2}$ 重数缩放) | ✅ λmax→1.5 |
| $\omega$ 本地不精确 | ICC 不完全 | λmax 侧 | ICC level(L=1–2) | ✅ 便宜杠杆(L0→L2 约减半) |
| $C_0$ 慢全局模 | 横跨全域的光滑模 | **λmin 侧** | **粗空间** | ❌ 未治(瓶颈) |

**overlap 反常的根因 = 过计数(λmax 侧)**:重叠区修正被计数 $\hat N$ 次 → λmax 涨 →CG 放大高频误差。**sASM 除以重数 → λmax→~1 → overlap 转为帮忙**。

**`-decay` 实测(Sys2,sASM O1 ICC0)**:λmax=**1.50**(已好)、λmin=**0.0051**、cond=**293**。残差**快头(5 迭代到 e-3)+ 慢尾**。最小特征值 `0.005,0.013,0.019,…` —— **主体聚在 ~1,外加 12 个孤立极小模**。

**核心可解释结论**:**cond=293 全部来自 λmin 那 12 个孤立小模**;它们是**沿最长几何轴(slab 的 x,20mm)的光滑全局模**。难度是**低维(~12)几何全局慢子空间**,不是过计数(已治)、不是各向异性、不是传输。

---

## 4. 系统性排除(每条都带可解释理由 + 实测)

| 试过什么 | 结果 | 可解释原因 |
|---|---|---|
| 系数/对角 PU 替重数 PU | 装配块 **0%** | 装配对角逐子域相同 |
| **传输条件优化(Robin,调 α)** | 近精确 −30%(常数因子);**ICC0 下归零** | **λmax 侧的"一跳局部"修正**;慢模是"多跳全局",够不着;ICC0 的 ω 又盖过它 |
| cw-SORAS(非装配 Neumann+Robin) | 被 overlapping sASM 支配 | Robin 收益要贵本地解买;overlap+Dirichlet 更简单更好 |
| 纯 Neumann 子域(不调 Robin) | 打不过 Dirichlet,"越准越糟" | 奇异+RHS 不相容;钉点/挂核都是绕过;归宿是粗空间(Neumann-Neumann/FETI) |
| overlap(sASM) | 帮忙,但只常数因子 | 也是 λmax 侧;不碰 λmin 地板 |
| 物理初值 $u_e\approx-cV_m$(Sys1→Sys2) | **−2%** | 单向量张不成 12 维尾;各向异性失配 |
| **各向异性 vs 各向同性** | **迭代不变**(67=67) | 难在几何长轴慢模($\sigma_L$+区域长度),非纤维比 |

**统一图景**:overlap、传输、PU、Neumann/Robin 块**全在 λmax(fine level / 高频 / 一跳)侧**;**sASM 已把这一侧做满**。真正卡住的是 **λmin(低频 / 全局 / 多跳)**——**局部方法一律够不着**。

---

## 5. 别人怎么解决(文献)

- **AMG(GAMG / BoomerAMG)**:bidomain 椭圆的**金标准多层粗空间**(Pennacchio-Simoncini;cardiac 强扩展标配)。
- **GenEO 粗空间 / SORAS**(Haferssas–Jolivet–Nataf 2017;GHS 2015):对 **Helmholtz/不定/高对比**;良态 SPD 椭圆(我们)用不到其特长。
- **Optimized Schwarz for bidomain**(Gerardo-Giorda et al, M2AN 2013)。
- **monodomain 预处理 bidomain**(Gerardo-Giorda et al, JCP 2009):块三角预条件,harmonic-mean σ,**等各向异性比下精确**。治**耦合**,不治椭圆扩展性;我们已分裂解、$\sigma_\text{mono}$ 正好是那个 harmonic mean。
- **Krylov 回收 / GCRO-DR / deflation**:对**系统序列**(同算子、变 RHS)。
- **POD 降基**(你们自己的 POD-solver 分支)。

---

## 6. 我们怎么解决(量身定做)

瓶颈 = 低维几何全局慢子空间(λmin)。**关键结构馈赠:Sys2/Sys3 在 EP 循环里每步重解同一算子** → **从求解过程"学"出慢子空间(回收),而非构造粗空间**。

- **轴 A(★核心,治迭代):Krylov 回收 / deflation** 那 ~12(Sys2)/~8(Sys3)个慢模(维度取自 `-decay`)。**"时间摊销的粗空间"**,几乎免费,随步成熟。**实测:滑窗 Fischer −77%(4.5×)**,正好对上 `-decay` 预言。升级:显式谱 deflation / GCRO-DR + 几何慢模播种治冷启动。
- **轴 B(治时间,3000 核命根):pipelined / CA-CG** 隐藏 Allreduce 延迟(每迭代 2–3 个全局归约、~log P,是通信受限区的主导项),不改迭代、砍每迭代墙钟。
- **轴 C(治时间/扩展):Chebyshev-Jacobi 光滑子**替 ICC(无分解 setup、通信平、GPU 友好)。
- **轴 D(已下调):按几何切分区**(subdomain 长宽比≈1,减少沿长轴界面);**不是**按各向异性(实测无关)。
- **轴 E:跨时间/系统 warm start**(单向量,弱,−2%;当锦上添花)。
- **杂交**:轻量 Nicolaides 粗空间(治冷启动/给强扩展下界保证)+ 回收(把维度和常数学到最优)。

**逐系统**:Sys1 平凡(只求最便宜迭代,不上回收/粗);Sys2/Sys3 **回收是主角** + overlap + ICC1/Cheby + warm start。

---

## 7. 结果(实测)

| 项 | 结果 |
|---|---|
| sASM+overlap+ICC(fine level)| λmax 侧做满(≈1.5) |
| **回收(Fischer)Sys2 EP 循环** | **−77%(4.5×)**;≈ `-decay` 预言的 deflate 12 维 |
| 物理初值(Sys1→Sys2)| −2%(单向量弱) |
| 各向异性 | 迭代不变(几何才是难度源)|
| Sys2/Sys3 地板 | 28–36 迭代 = λmin = 待粗空间/回收压平 |
| overlapping sASM vs cw-SORAS | sASM 两端(便宜/近精确)都赢 |

---

## 8. 结论

1. **所有 fine-level 手段(overlap / 传输 / PU / Neumann-Robin 块)都在 λmax 侧;sASM 已把这一侧做满,它们碰不到 λmin 瓶颈。** overlap 反常的根因(过计数)与传输优化的失效,**是同一侧的事**。
2. **难度是一个低维(~12)几何全局慢子空间**(沿最长轴的光滑模),与各向异性无关、与过计数无关。
3. **正确解法利用"重解同一算子"的结构:回收慢子空间(时间摊销的粗空间)**;实测 −77%。局部传输/overlap 永远够不着,只有全局耦合(粗空间/回收)能治 λmin。
4. **强可扩展两大杀手各有对策**:轴 A(回收/粗空间)钉住"迭代随 np 涨"、轴 B(pipelined)治"每迭代随 np 变贵"。

---

## 9. 最大的未验证空白(诚实)

**所有结论都在 np=4/8 单机;真正的强扩展曲线(iters & time vs np)从未跑过。** "回收/粗空间让迭代随 np 钉住"是**核心命题,尚未验证**。下一步最该做:pipelined CG(一行)+ sASM+粗空间+回收的**强扩展曲线**——否则"适合 3000 核"仍是外推。
