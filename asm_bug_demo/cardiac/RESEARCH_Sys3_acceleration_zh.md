# Sys3(躯干正问题)加速:文献 + 我们实测的综合

> **诚实前提(先读)**:这份是深度检索(5 角度 × 并行搜索 × 抓取 × 对抗验证 × 综合)的产物,
> 但**抓取/声明抽取阶段失败(0 条外部声明被验证)**——很可能 WebFetch 被代理挡了。
> 所以:**下面的排名与机理是基于问题结构 + 我们自己的实测推理出来的,不是逐条验证的文献结论;
> §末的论文清单是搜索真实 surfacing 出来的 36 篇(标题/链接可核对),但我没有逐篇读证。**
> 数字若来自我们的实测会标 [实测];来自推理标 [推理];文献线索标 [文献线索,待核]。
>
> **更新(实测后):§1/§2 的排名 #1「传输矩阵消掉每步求解」经 §7 实测修正——它只对
> _电极标量_ 输出成立;对你需要的 _躯干体积场_,全场叠加因 RHS 高秩而失败(实测 13–16% 误差),
> 必须每步真解。请连同 §7 一起读。**

---

## 1. 一句话结论

**Sys3 最根本的加速不是"更好的预条件",而是根本不每步解 PDE——用预计算的"界面→体表"传输矩阵(lead-field)替代整个求解。**
Sys3 三个性质合起来决定了这一点:算子 K_t **恒定**、每步只有界面 Dirichlet 数据变、最终**只要体表电极处**的电位(不要躯干体积场)。于是"界面电位 → 电极导联"是**一个固定线性映射 Z**,一次性建好(用少数界面基右端解 Sys3 得到 Z 的列),之后 10³–10⁴ 步每步只做**一次通信轻的矩阵×向量,没有内层 CG、没有每迭代 Allreduce、没有 AMG 最粗层全局通信**——正好把我们实测到的 3000 核瓶颈(40–100 µs Allreduce 延迟)从根上消掉。

**这与我们之前讨论的 GAMG/回收是不同层级**:GAMG/回收只是**加速 FEM Krylov 求解**,仍受 log P 同步地板限制;传输矩阵/BEM 是**从根本上改变问题**(消掉每步体积求解)。只有当你**真的需要躯干体积场**时,才回到 Krylov 加速那一档。

---

## 2. 排名(标注:改变问题 vs 只是加速)

| # | 方案 | 改变问题? | 3k–16k 核预期收益 | 置信 |
|---|---|---|---|---|
| **1** | **界面→体表传输/lead-field 矩阵 Z**(K_t 恒定 + 只要体表 → 预计算,低秩/H-matrix 压缩,摊销 10⁴ 步) | **是**:消掉每步体积求解 + 全部内层归约 | **最大**:51–69 次延迟受限 CG/步 → 1 次通信轻 apply,秩被那 ~8 个慢模界定 | 机理高[推理];**无验证文献引用** |
| **2** | **BEM 解躯干 Laplace**(+ FMM / H-matrix) | **是**:无体积网格/DOF(纯 Laplace);恒定算子 → 边界逆也可预计算成传输映射 | 去掉躯干体积 DOF,每步塌成边界 apply(近线性) | 机理中[推理];**无验证引用** |
| **3** | **保留 GAMG,但粗层 telescope + 外层 pipelined/CA-CG** | 否:只加速 FEM Krylov | 修我们实测的 GAMG 两个短板(粗层全局通信 → PCTELESCOPE;每迭代 Allreduce → PIPECG),让 5.3× 少迭代**真的**兑现成墙钟 | 中高;各部件有文献线索 |
| **4** | **降阶 / POD / reduced-basis** | 部分:若解流形低维 → 小投影解 | 有上限:去极化前沿是对流主导 → 快照高秩 → 饱和(我们 T3 实测 baseline 46→POD 38.5,短序列仅 ~3%) | 中[实测边界] |
| **5** | **warm/Fischer/回收-deflation 那 ~8 慢模** | 否:只加速 Krylov | 小:Sys3 实测封顶 ~10%(warm −9%≈Fischer −11%,残差留在 8 慢模) | 高[实测] |

---

## 3. 分角度要点

### 3.1 传输矩阵 / lead-field(根本性改变)
离散上把躯干 DOF 分成界面 Γ 和内部/表面,谐波延拓给出 **φ_电极(t) = Z · u_e|Γ(t)**,Z = 由 K_t 建的离散 Poincaré–Steklov(谐波延拓)算子在电极行的限制。K_t 恒定 → Z 一次建好(解 Sys3 若干次,每次一个界面基右端 = Z 的一列),之后每步复用。**这就是经典正问题 ECG 的"传输矩阵/lead field"构造**;我们仓库其实已有一个简化形式(T2 里的 lead-field pseudo-ECG,但那是心源近似,不是完整躯干算子)。压缩:界面数据平滑、我们实测只有效张 ~8 个慢模 → **Z 的有用列秩很小**,低秩/H-matrix 天然。

### 3.2 BEM vs FEM + FMM/H-matrix
躯干内部**无体源**(纯 Laplace)= BEM 的教科书前提:只离散边界,消掉体积网格与 DOF。配恒定算子 → 边界逆预计算成同样的传输算子。BEM 的稠密边界矩阵正好用 **FMM / H-矩阵**压到近线性 apply/存储。M/EEG 正问题这条路很成熟(见清单),躯干 ECG 同理。

### 3.3 降阶 / POD
若 u_e(t) 与躯干解在低维流形上 → POD 每步塌成小投影解。**但对流主导的去极化前沿产生高秩快照 → 饱和**(我们跨系统 T3 实测:baseline 46 → POD 38.5;真实短序列仅 ~3%)。适合前沿慢/容差松的场景,不是通用解。

### 3.4 通信规避 Krylov + 多层(若必须解体积场)
pipelined CG(KSPPIPECG,Cornelis–Cools–Vanroose)把 Allreduce 与 SpMV 重叠(**须开 MPI 异步进度**);GAMG 粗层用 PCTELESCOPE 挪到子通信器。这两条正好修我们实测的 GAMG 两个短板,让 69→13 的迭代收益在 3000 核兑现成墙钟——但**这是加速求解,不改变问题**,仍不如传输矩阵。

### 3.5 心脏 EP 的 HPC 实践
清单里 openCARP、Cardioid(SC12,Sequoia 上近实时)、Chaste、lifex-ep、Potse 2018(可扩展 ECG 模拟)、AMG-bidomain 预条件等,是真实的大规模心脏求解器工作,可看它们对躯干/双域外电位/正问题 ECG 到底用什么方法与预条件。

---

## 4. 和我们实测怎么合

- **回收(−11%)、warm(−9%)封顶**:文献侧(GCRO-DR 回收、reduced-basis)也表明,对流主导 + 移动边界的问题,子空间/降阶会饱和——与我们 Sys3 实测一致。
- **GAMG 迭代 −5.3× 但通信重**:文献(pipelined CG、telescope 粗层)给的正是我们 3000 核模型缺的那两块补丁;方向一致。
- **传输矩阵为何胜出**:它绕开我们整个 3000 核成本模型的前提(每步有迭代求解)——**没有迭代就没有 Allreduce 墙**。这是我们之前的 GAMG/回收讨论都没触及的层级。

---

## 5. 诚实的空白

- 抓取失败 → **传输矩阵压缩经济学、BEM+FMM 躯干加速的具体数字没有验证文献支撑**,是从结构推的。
- 论文清单**我没有逐篇读证**,是搜索 surfacing 的标题/链接,供你核对。
- 传输矩阵的实际瓶颈(Z 的构造成本、电极数 × 界面模数、并行 apply 的通信)在**极大规模**下需要自己测——它把"每步迭代求解"换成了"一次稠密/H-matrix apply",apply 本身的通信要另算(但远小于几十次 Allreduce)。

---

## 6. 论文清单(搜索真实 surfacing,36 篇,待逐篇核对)

**传输矩阵 / lead-field / 正问题综述**
- Potse, *Scalable and Accurate ECG Simulation for Reaction-Diffusion Models of the Human Heart*, Front. Physiol. 2018 — https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2018.00370/full
- Stenroos & Haueisen, *Lead field computation for the ECG inverse problem — FEM vs BEM*, CMPB 2008 — https://www.sciencedirect.com/science/article/abs/pii/S0169260704002068
- Wang & Rudy 等, *Acceleration of FEM-based transfer matrix computation for forward/inverse ECG* — https://pubmed.ncbi.nlm.nih.gov/19590912
- *Learning geometry-dependent lead-field operators for forward ECG* (arXiv:2602.22367,ID 待核) — https://arxiv.org/abs/2602.22367
- *A Review of the Forward Problem in Electrocardiographic Imaging*, J. Imaging 2025 — https://www.mdpi.com/2313-433X/12/6/224

**BEM / FMM / H-matrix**
- Kybic, Clerc 等, *Symmetric BEM formulation for the M/EEG forward problem* — https://pubmed.ncbi.nlm.nih.gov/15344485
- Kybic 等, *Fast multipole acceleration of the MEG/EEG BEM* — https://iopscience.iop.org/article/10.1088/0031-9155/50/19/018
- Makarov 等, *BEM-FMM for neurophysiological recordings* — https://pmc.ncbi.nlm.nih.gov/articles/PMC7704617
- Makarov, Noetscher 等, *Fast EEG/MEG BEM forward solution for high-res head models*, 2024 — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11185788
- Rahmouni 等, *Calderón-regularized symmetric formulation for EEG forward* — https://arxiv.org/pdf/1903.08405
- Fischer, Tilg 等, *Bidomain BEM-FEM coupling for anisotropic cardiac tissue* — https://link.springer.com/article/10.1114/1.1318927
- *ECG forward solution by BEM* — https://www.jstage.jst.go.jp/article/jsmbe1963/22/5/22_5_318/_article
- Liu, Ghysels, Li 等, *Parallel Hierarchical Blocked ACA (H-BACA)* — https://arxiv.org/pdf/1901.06101
- Zapata 等, *GPU parallelization of H-Matrix accelerated BEM* — https://arxiv.org/abs/1711.01897

**Poincaré–Steklov / DtN / HPS**
- *Schur complement form of the Dirichlet-to-Neumann operator*, Syst. Control Lett. 2007 — https://www.sciencedirect.com/science/article/abs/pii/S0165212507000856
- *Towards modular Hierarchical Poincaré–Steklov (HPS) solvers*, arXiv:2510.26945 — https://www.arxiv.org/pdf/2510.26945

**降阶 / POD / reduced-basis / DL-ROM**
- Pagani, Manzoni, Quarteroni, *Local reduced basis for parametrized cardiac EP*, CMAME 2018 — https://www.sciencedirect.com/science/article/abs/pii/S0045782518303001
- Boulakia, Schenone, Gerbeau, *Reduced-order modeling for cardiac EP*, arXiv:1111.5926 — https://arxiv.org/abs/1111.5926
- Fresca, Manzoni, Dedè, Quarteroni, *Deep learning-based ROM in cardiac EP*, PLOS One 2020 — https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0239416
- *POD-Enhanced DL-ROM for real-time atrial EP*, Front. Physiol. 2021 — https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2021.679076/full
- *ROM + ML for forward UQ in cardiac EP* — https://pmc.ncbi.nlm.nih.gov/articles/PMC8244126
- *Accelerated subspace recycling via MOR for parametric linear systems*, CMAME 2022 — https://www.sciencedirect.com/science/article/abs/pii/S0045782522007216
- *DEIM-based data projection across non-conforming interfaces*, ACOM 2022 — https://link.springer.com/article/10.1007/s10444-022-10008-w

**Krylov 回收 / 通信规避**
- Parks, de Sturler 等, *Recycling Krylov Subspaces for Sequences of Linear Systems (GCRO-DR)*, SIAM — https://epubs.siam.org/doi/10.1137/040607277
- Soodhalter, de Sturler, Kilmer, *A survey of subspace recycling iterative methods*, arXiv:2001.10347 — https://arxiv.org/abs/2001.10347
- Cornelis, Cools, Vanroose, *Improving Strong Scaling of CG using Global Reduction Pipelining* — https://arxiv.org/pdf/1905.06850

**心脏 EP 的 HPC 求解器**
- Plank 等, *The openCARP Simulation Environment for Cardiac EP*, CMPB 2021 — https://www.biorxiv.org/content/10.1101/2021.03.01.433036v2.full.pdf
- Richards, Mirin 等, *Cardioid: Towards Real-Time Cardiac EP at High Resolution* (SC12) — https://research.ibm.com/publications/towards-real-time-simulation-of-cardiac-electrophysiology-in-a-human-heart-at-high-resolution
- Bernabeu 等, *Chaste: parallelisation of an open-source FE cardiac EP solver* — https://www.researchgate.net/publication/262281678
- *lifex-ep: Robust and Efficient Software for Cardiac EP*, arXiv:2308.01651 — https://arxiv.org/pdf/2308.01651
- *Algebraic Multigrid Preconditioner for the Cardiac Bidomain Model* — https://pmc.ncbi.nlm.nih.gov/articles/PMC5428748

---

## 7. 我们随后实测了这些方案(4 张图,验证/修正上面的排名)

上面 §2 的排名是**推理**的;检索后我们把其中三档真的实现并在 np=8、80 步上跑了,结果**修正了排名 #1 的一个关键前提**,并确认了 #3/#5。逐一对上:

### 7.1 传输矩阵(排名 #1)——必须拆成两种,全场那种**实测失败** → `fig_leadvol.png`
`-leadvol`:把 §3.1 的传输思想用到**整个躯干体积场**(不只电极)——预计算响应场 Ψᵢ=K_t⁻¹Bᵢ,每步做 φ(t)=Σ⟨Bt,Bᵢ⟩Ψᵢ 纯叠加,不解 PDE。
- **实测负面**:80 步里 **54 步都是新方向**(基长到 54 还不收敛),全场相对 L2 误差 **13–16%**。躯干右端项(界面数据)由**移动去极化前沿**驱动,是**高秩**的,不是我们盼的 ~8 维。
- **关键不对称(排名 #1 要改写)**:
  - **电极 lead-field(标量输出)**:ECG = g_lead·b(t),g_lead 固定向量,**对任意 RHS 精确、不要求低秩** —— §3.1 那一档在这里成立,但**只给电极标量**。
  - **全场叠加(体积输出)**:需要 RHS 低秩,而它**不是** → 失败。
  - 所以 §2 排名 #1「消掉每步体积求解」**只有当你要的输出是电极标量时才成立**;**你明确说过要全身体积解** → 这条对你不适用,必须真解。§1 的"一句话结论"应据此降级:它是**电极 ECG** 的最优解,不是**体积场**的。

### 7.2 算子级零同步:Chebyshev(§3.4 的通信规避一档)——小规模慢、大规模反超 → `fig_cheb.png`
`-f3cheb`:一次性估 [emin,emax] 后用 Chebyshev 迭代,**每迭代零点积 / 零 Allreduce**。
- **实测(np=8)**:139 迭代/步(CG 的 69 的 2×),墙钟 5.9 s > CG 4.1 s —— 小规模没延迟可藏,2× 迭代直接吃亏。
- **3000 核模型**:每迭代被 40–100 µs Allreduce 延迟主导,CG 每步付 69×2 次延迟、Chebyshev **0 次** → 模型全区间 Chebyshev 更快(L=5 µs 时 ~12×)。这正是 §3.4「若必须解体积场」的最实用一档:不改问题、不改预条件、只把同步点消掉。

### 7.3 算子级迭代削减:GAMG + 初值封顶(§3.4 vs §3.5)→ `fig_sys3_opt.png`
`-fischer3gamg`:
- **GAMG 迭代 5.3× 少**(69→13/步),np=8 总墙钟 3.2 s < bjacobi+ICC 4.1 s;但**每迭代贵 4.2×**、collective 更多 → 3000 核要靠 §3.4 的 PCTELESCOPE(粗层)+ pipeline 才能兑现。
- **初值封顶**(印证 §2 排名 #5):Sys3 warm(−9%)≈ Fischer(−11%),子空间几乎不比单向量强,残差留在那 ~8 个慢模里 → **初值类一律 ~10% 天花板**。对比 Sys2 warm(−5%)≪ Fischer(−82%),那里子空间才是主角。

### 7.4 物理初值 = 0% → `fig_f3_physics.png`
`-fischer3` 里加了 physics DC 猜测 x0=常数=交界面均值(调和解的直流分量)。
- **实测 −0%**(5550 vs cold 5544)。因为交界数据来自**零均值的 Sys2 u_e** → 直流 ≈ 0 → 没有偏移可捕。**warm(−9%)已经是 Sys3 的物理一致初值**;Sys3 的调和问题没有 Sys2 那种 `u_e≈−c·Vm` 的额外代数捷径。

### 7.6 直接验证:真参数假几何(Niederer + cube)上实现并跑 `-transfer` → `fig_transfer_real.png`
把"近似算子而非近似动解"做成 `forward_ecg.cpp -transfer`:离线一次建界面→躯干传输算子 Z
(每个界面 DOF 一次 Sys3 解,列 = K_t⁻¹ 该单位界面值的提升),之后每步 φ=Z·u_iface(t) 用一次
本地稠密 matvec。实测(`mpirun -np 8 ./forward_ecg -m heart_torso.msh -transfer -T 80`,
真 Niederer 参数,P8 激活 40 ms、CV~0.55 m/s 说明物理跑对):
- **精确性【成立】**:Z 对真实高秩 RHS **每步复现全躯干场到 mean 8.1e-10 / max 1.4e-9**(=求解器容差)。
  高秩 RHS 确实**无关**——我们 apply 的是不动算子,不是动解的降基。核心主张在真参数真算子上验证通过。
- **速度【没赢】**:N_iface=**4757** → 离线建 Z 要 **250 s**;每步稠密 matvec **61 ms** ≈ 每步真解 **45 ms**。
  即**精确是免费的,提速不是**——4757 列的稠密算子 apply 和它替代的求解一样贵(还没算 250s 离线)。
- **电极输出【立刻大赢,已实测】**:同一次运行里也建了互易 lead-field(每电极一次伴随解 w_e=Kt⁻¹·单位@电极,每步 φ(e)=w_e·Bt)。实测:**离线仅 2 次伴随解 = 0.11 s**,**每步 2 点积 = 0.05 ms/step = 求解的 910× 更快**,ECG 精确到 **max|ECG_lead−ECG_true|=3.5e-9(ECG 尺度~5.2 → 相对 ~7e-10)**。你真正要读的 ECG 基本免费且精确。
- **要全躯干体积场**:必须 **H-matrix/FMM 压缩** Z(远场块低秩、近界面不压)+ 用 BLAS,并把 240s 摊到真实 1e4–1e5 步(且 apply 得先压到 < 求解才谈得上摊销)。

这条把 §7.1 的负面结论**补全**:leadvol(降基)失败是因为动解高秩;`-transfer`(降算子)在精确性上成功,
但朴素稠密版不提速,验证了"全场需压缩、电极输出立即兑现"的判断——**都在真参数假几何上直接测过**。

### 7.7 H-matrix 完整落地(`-transferh`,真几何实测)→ `fig_transferh_real.png`
把 H-matrix 压缩真写进 `forward_ecg.cpp -transferh`(2-side:torso 目标聚成叶盒、界面源聚成组,
每个 叶×组 块近则稠密、远则随机化+dsyev 截断成低秩 U·V),真机跑通并对真解验证。关键发现(都实测):
- **稠密 Z apply 是内存受限**:每步 BLAS matvec 要**流完整 172 MB/核 的 Z(总 1.3 GB)**,~55 ms ≈ 解(40 ms)。
  所以全场的瓶颈**不是 flops 而是 Z 的『体积』**——稠密 Z 在心脏尺度**不是清晰的提速**,还占 1.3 GB。
- **电极 lead-field 只流 2 个向量** → 0.045 ms/步、~900×、精确 3.5e-9 —— 你要读的 ECG 立即赢(已完成、可部署)。
- **H-matrix 必须缩小 Z**(这才是真杠杆):从零写的 **2-side 单层**正确(精确 8.2e-7),但此几何(心脏居中且小 → 单层聚类粗)**只压 1.26×**,且散取索引使 apply 反而更慢(110 ms)。**真正的生产级收益需递归多层 H² + DOF 重排(库级:HLIBpro/H2Lib/STRUMPACK)**;2D POC(§见 fig_hmatrix)已量化可达 4.6–8× 且随 N_iface 增长——那是大尺度区间。

**三档最终结论**:① 要 ECG → 互易 lead-field(现成、精确、~900×);② 要全场 @ 心脏尺度 → 稠密 Z 精确但内存受限(≈解、占 1.3 GB),不划算;③ 要全场 @ 大尺度 → H-matrix(已实现+验证,单层此几何压 1.26×,生产级需递归多层+重排)。

### 7.8 用 Sys1/Sys2 信息做增量传输(`-transferinc`)—— 实测负面 → `fig_transferinc.png`
最后一个"用前面系统加速 Sys3"的想法:维护 φ(t)=φ(t−1)+Z·Δu_e(t),只更新前沿在变的界面列
(Sys1 给前沿位置,Sys2 给 Δu_e)。若前沿只碰 K≪N_iface 列,每步就只流那一小片 Z(而非 1.3GB)。
- **实测(真几何,eps=1e-3·max|u_e|,每 20 步全刷新)**:每步"在变"的界面 DOF **平均 4700 / 4757 = 99%**!
  → 流~全部 Z,增量 apply 55 ms ≈ 解 61 ms(**没赢**),全场误差 5.2e-4。
- **机理(重要)**:u_e 是**椭圆方程的解**。前沿(源)空间局部、Sys1 精确知道在哪,但它对 u_e(响应)的
  影响是**全局**的——前沿一动,u_e 在**整个界面处处变一点**。所以 Δu_e **范数小但支撑稠密**,阈值挡不住 99%。
  **让传输算子远场低秩的那个"空间光滑",恰恰同样把时间增量"全局化"**:没有既局部于空间、又局部于时间的更新可利用。
- **这条线到此关闭**:知道前沿在哪 ≠ 能局部化 Sys3 更新——椭圆 Green 函数把它铺满全场。这也从机理上解释了
  为何初值/历史/增量全都封顶:跨系统耦合是椭圆的 → 全局 → 没有低维/局部的时间结构可榨。

**Sys1/2 → Sys3 的最终裁决**:能用的只有 ① 电极 lead-field(ECG 输出,精确、~900×)、② 算子级(GAMG/Chebyshev,不依赖 Sys1/2)、③ 大尺度 H-matrix。**Sys1/2 的解信息本身不能加速 Sys3 的全场求解**(初值封顶 ~10%,增量因椭圆全局化而失效)。

### 7.9 Sys3 在 sASM+CG 框架内自我加速(`-sys3coarse`,真机实测)→ `fig_sys3coarse.png`
回到项目核心框架 κ ≤ C₀²·ω·(N̂+1),直接在 Sys3 的躯干算子 Kt(SPD、Dirichlet 锚定、非奇异)上
扫 C₀(粗空间)和 ω(重叠)两个旋钮(8 子域,rtol 1e-8):

| 变体 | 迭代 | 相对基线 | 每迭代 |
|---|---|---|---|
| 基线 bjacobi+ICC(一层) | 63 | — | 0.51 ms |
| +Nicolaides 粗空间(C₀,dim 8) | 60 | −4% | ~0.5 ms |
| +几何 {1,x,y,z} 粗空间(C₀,dim 32) | 56 | −11% | ~0.55 ms |
| +回收 A⁻¹ 快照(C₀,谱近似) | 61 | −3% | — |
| sASM 重叠 0 / 1 / 2(ω) | 63 / 62 / 50 | −0 / −1 / −20% | 升到 1.0 ms |
| 重叠2 + 几何粗(便宜档最佳) | 47 | **−25%** | — |
| **GAMG(多层)** | **13** | **−79%(5.3×)** | ~4× |

**关键发现(有点反直觉)**:**便宜的粗空间对 Sys3 几乎没用**(−4~−11%),连 A⁻¹ 回收的"谱"快照也只 −3%。原因:
**Sys3 的病态(κ~172)是 Laplacian 的 h 细化『多尺度』层级(~1/h²),不是几个孤立的低维慢模。** 单层
粗空间(子域常数/{1,x,y,z})只削掉最低几个频率,主体多尺度层级留着——2 层 ASM 的 h 无关界需要真正的
粗『网格』空间,不是子域常数。**ω(重叠)是更强的便宜旋钮(重叠2 −20%),但每迭代翻倍**(墙钟基本抵消,
大规模上重叠通信更贵)。便宜档最佳 = 重叠2+几何粗 = −25%。

**框架内的 Sys3 加速配方**:① **真降迭代 = 多层 + 真粗空间 = GAMG**(smoothed-aggregation,63→13,5.3×),
这就是框架的多层推广(AMG = 多层 Schwarz),代价每迭代 ~4×;② 想留在 2 层真赢 → 需 **GenEO 谱粗空间**
(局部广义特征问题挑真慢模,对多尺度也 h 无关),但要 SLEPc(本环境未装)。**结论:Sys3 靠框架内的
便宜旋钮只能 ~25%,真正的杠杆是多层(GAMG)或谱粗空间——单层几何粗空间治不了它的多尺度病态。**

### 7.11 纯 ASM vs sASM 重叠扫描(`-asmovl`)—— 缩放才让重叠有用 → `fig_asmovl.png`
不掺任何其他技术(无粗空间,ICC 子解对两者相同),只对比**未缩放 ASM(PC_ASM_BASIC)**与
**缩放 sASM/RAS(PC_ASM_RESTRICT)**,扫 overlap 0–4(Sys3 躯干 Kt,8 子域,合成慢模 RHS,rtol 1e-8):

| overlap | ASM(未缩放) | sASM(缩放) | sASM 省 |
|---|---|---|---|
| 0 | 63 | 63 | 0%(重合) |
| 1 | 90 | 62 | 31% |
| **2** | 105 | **50**(最佳) | **52%** |
| 3 | 99 | 50 | 49% |
| 4 | 90 | 50 | 44% |

**三点全对上理论**:① overlap=0 两者相同(无重叠 → multiplicity=1 → 缩放=恒等 → 同一个块 Jacobi);
② **未缩放 ASM 加重叠反而更差**(63→90→105),因为重复计数/over-relax 重叠区;③ **缩放 sASM 随重叠稳步降到 50**,
**ov=2 达到最佳**后饱和,比同重叠的 ASM 少 **52%**(50 vs 105,~2.1×)。**结论:让重叠真正减少迭代的是缩放,不是重叠本身;
sASM 在 ov=2 达到最佳迭代数。**

### 7.5 修正后的一句话
- **要电极 ECG**:用 §3.1 电极 lead-field,精确、对任意 RHS、无每步求解 —— 排名 #1 成立。
- **要躯干体积场**(你的需求):全场传输**失败(RHS 高秩)**,必须每步真解;**初值/回收/物理封顶 ~10%**;真正的杠杆是**算子级**——Chebyshev(消同步,3000 核模型胜)与 GAMG(削 5.3× 迭代,需 telescope+pipeline 兑现)。BEM+FMM(§3.2)是尚未实现、值得一试的"改变问题"档,但同样**只有电极/边界输出**时才免体积。
