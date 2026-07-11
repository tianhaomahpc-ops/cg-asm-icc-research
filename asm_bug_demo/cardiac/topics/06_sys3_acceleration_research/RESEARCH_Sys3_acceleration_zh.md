# Sys3(躯干正问题)加速:文献 + 我们实测的综合

> **诚实前提(先读)**:这份是深度检索(5 角度 × 并行搜索 × 抓取 × 对抗验证 × 综合)的产物,
> 但**抓取/声明抽取阶段失败(0 条外部声明被验证)**——很可能 WebFetch 被代理挡了。
> 所以:**下面的排名与机理是基于问题结构 + 我们自己的实测推理出来的,不是逐条验证的文献结论;
> §末的论文清单是搜索真实 surfacing 出来的 36 篇(标题/链接可核对),但我没有逐篇读证。**
> 数字若来自我们的实测会标 [实测];来自推理标 [推理];文献线索标 [文献线索,待核]。

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
