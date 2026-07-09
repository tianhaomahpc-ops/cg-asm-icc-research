# 3000 核生产方案:三层分工的求解器设计

> 回答"3000 核时应该怎么做"。所有组件可用性**已对本机 PETSc 3.19.6 实际验证**(grep 头文件
> + nm 库符号 + 运行时测试);成本数字来自逐项算术核对的模型(见 §5 账本)。
> 配图:`fig_3000core_design.png`。依据:`NOTES_schwarz_variants_zh.md`(-fair/-decay 等实测)、
> `slowdim_scaling.py`(慢模维数 ~O(子域数))、`fig_fischer_eploop.png`(回收实测 −63%)。

---

## 0. 一句话答案

**Sys2/Sys3 用三层分工:sASM+overlap+ICC(细层,治快头)→ GenEO 粗空间(治 O(P) 慢模+薄壁/疤痕硬模,粗解按 m 分档:稀疏冗余或子通信器)→ 有界回收窗口 16–32(治时间漂移),外层 pipelined CG 隐藏 collective 延迟;Sys1 平凡,只用最便宜档。一次性 setup 被上万次重解摊销。**

---

## 1. 先做前提检查:3000 核配多大的问题

成本模型的最硬结论(逐项算术核对):

| 每迭代项 | N=2e6(700 dof/核) | N=2e7(7000 dof/核) |
|---|---|---|
| SpMV | ~1 µs | ~10 µs |
| ICC(1) apply(含 overlap) | ~4 µs(实际可能 3–10×) | ~26 µs |
| halo(近邻) | ~4 µs | ~4 µs |
| **2 次点积 Allreduce** | **40–100 µs(主导)** | 40–100 µs |
| 合计(不含粗解) | **~65 µs** | ~100 µs |

- **700 dof/核已过强扩展甜点**:overlap 一层就把本地工作放大 1.84×,浮点效率 ~5%,每迭代几乎全是延迟。
- **结论:N≈2e6 的网格用 ~500–1000 核就到头;3000 核要配 N≥1e7–2e7(≥5000–7000 dof/核)才划算。** 若网格就是 2e6,答案是"别用 3000 核",这也是一种工程正确。

---

## 2. 逐系统处方

| 系统 | 谱(实测 -decay) | 3000 核处方 |
|---|---|---|
| **Sys1** | cond 1.4,0 慢模 | bjacobi+ICC0(或 Chebyshev-Jacobi)+ pipelined CG + warm start。**不加粗空间/回收**(无尾可治,加层纯浪费 setup) |
| **Sys2** | cond 293,慢模 ~1.5×子域数 | **三层栈(§3)+ 粗解分档(§4)** —— 全文主角 |
| **Sys3** | cond 172,同类略轻 | 同 Sys2,回收窗口略小(~12) |

---

## 3. Sys2 三层栈(每层:做什么 / 治什么 / 通信)

### 细层:sASM + overlap(O1) + ICC(1)
- 每子域 Dirichlet 块 + 1 层 overlap,本地 ICC(1) 三角解;重数权对称缩放(CG 合法)。
- 治:κ 界里的 (N̂+1) 过计数 + ω 本地精度 = **快头**(实测 -fair:76→33)。
- 通信:仅近邻 halo(几百字节–几 KB,纯延迟,**不随 P 恶化**)。

### 粗层:GenEO 粗空间(维数 m ≈ (1~5)×P)
- 为什么不是 Nicolaides 就完:`slowdim_scaling.py` 实测慢模 ≈1.5/子域,且**薄壁/峡部 +6、高对比疤痕 +3 个硬模**——这些是每子域常数抓不到的,GenEO(每子域小广义特征问题,自适应挑 λ<τ 的模)有可证界。
- 列局部支撑 → 每 rank 只存自己的几列;**粗算子 E 稀疏**(子域连接图,~7 块/块行)。
- setup(每子域特征问题 + E 组装/分解)**只做一次**,被 EP 循环 O(10³–10⁴) 次重解摊销。
- 治:C0 = O(P) 全局慢模 + 几何/系数硬模 → **迭代数随 P 钉住**(强扩展的命根)。

### 回收层:Fischer/GCRO-DR,窗口 16–32(有界!)
- 跨时间步学 Vm(t) 漂移余量;维数 **O(1) 不随 P 长**——它扛的是"粗空间之外的时间漂移",不是 O(P) 主体(那是粗空间的活,别让回收硬扛)。
- **实现铁律:投影的 16–32 个点积必须打包成 1 次 Allreduce**(单发 = 320–1600 µs 灾难;打包 = 1 次 ~20–50 µs,与窗口大小无关)。
- 实测:EP 循环 −63%(fig_fischer_eploop),50/80 步 0 迭代。

### 外层:pipelined CG
- 每迭代全局 collective ≈ 3 次(2 点积 + 1 粗解 gather),P=3000 时每次 20–50 µs,是**主导项**;PIPECG 把它们与 SpMV/PC apply 重叠。
- 非预条件范数判据(回收初值质量才显形)+ 挂常数核(Sys2 奇异)。

---

## 4. 粗问题怎么解(关键瓶颈,按 m 分档,数字已核)

| 策略 | m=3000(Nicolaides) | m=15000(GenEO×5) |
|---|---|---|
| (a) 冗余**稠密** Chol | **1800 µs,超预算 ~40×,死**(36 MB/rank) | 45 ms + 900 MB/rank,不可行 |
| (b) 冗余**稀疏** Chol(E=子域图) | **22–43 µs ✓ 正解** | 540–1080 µs,不可用 |
| (c) q=32–64 子通信器直接解 | ~100–230 µs(2 次 collective,反而亏) | **~100–230 µs ✓ 正解** |

- **冗余稠密在 m≈300–900 就死**(即 P~300–900 就不能再用它)——np=8 时代的"稠密小 E 随手求逆"到 3000 核完全失效。
- m 再大(P>1e4 或 GenEO 挑得多):(c) 也撑不住,需要**真第三层**(对 E 上 AMG / 递归粗空间)。3000 核还不必,但要在软件上留门。

---

## 5. 软件落地(已对本机 PETSc 3.19.6 验证)

**本机现状(验证过)**:PCHPDDM/SLEPc **没编进去**(petscconf.h 无 PETSC_HAVE_HPDDM,`-pc_type hpddm` 运行时报 Unknown type);但 **PIPECG/PIPECG2/GROPPCG、PCASM、PCTELESCOPE、PCREDUNDANT(+MUMPS)、PCGAMG、内建 KSPGuess fischer/pod 全部可用且运行验证通过**。

### 路线 A:不重编 PETSc(现有组件,立即可跑)
```
-sys2c_ksp_type pipecg                        # 外层(sys2c_ 前缀未被代码钉死,CLI 可调)
-sys2c_ksp_norm_type unpreconditioned
# 细层:代码里现成 sASM/SORAS;粗层:代码里现成 TwoLevelCoarse/GlobalDeflate
# 粗解子通信器化:PCTELESCOPE
-telescope_pc_type lu -telescope_pc_factor_mat_solver_type mumps
-pc_telescope_reduction_factor 64
# 或冗余稀疏:-pc_type redundant -pc_redundant_number 1 -redundant_pc_type cholesky
# 回收(PETSc 内建!):
-sys2c_ksp_guess_type fischer -sys2c_ksp_guess_fischer_model 1,16
```
注意(已验证的坑):`forward_ecg.cpp:523-528` 用 `PetscOptionsSetValue` 把 `sys1_/sys2_/sys3_` 的 pc_type 钉死成 bjacobi,**命令行改不动**,要改这几个前缀的 PC 需改代码;`sys2c_/sys2fine_/soraspu_/defl_` 前缀完全 CLI 可调。MFEM 的 PetscPCGSolver 在**第一次 Mult 才执行 KSPSetFromOptions**,所以 `-sys2c_ksp_type pipecg` 能覆盖构造函数里的 CG——但只在首解前有效。

### 路线 B:重编 PETSc(集群上推荐,一步到位)
```
./configure --download-hpddm --download-slepc ...
# GenEO 两级 Schwarz(粗空间自动建,粗解自动子通信器化):
-pc_type hpddm -pc_hpddm_levels_1_eps_nev 5 -pc_hpddm_levels_1_sub_pc_type cholesky
-pc_hpddm_coarse_pc_type redundant   (或 mumps + coarse_p 子通信器)
# GCRO-DR 回收:
-ksp_type hpddm -ksp_hpddm_type gcrodr -ksp_hpddm_recycle 24
```
路线 B 把 §3 的粗层+回收层全部变成一行选项,是生产首选;路线 A 是我们已有代码的直接延伸,用来在小集群上先验证阶梯。

---

## 6. 每迭代通信账本(P=3000)

| collective | 次数/迭代 | 成本 | 对策 |
|---|---|---|---|
| CG 点积 Allreduce | 2 | 各 20–50 µs | PIPECG 重叠 |
| 粗解 gather/allgather(m 双精度) | 1 | 30–60 µs(m=3k) | 与 PC apply 重叠;稀疏冗余则只此一次 |
| 回收投影 | 1/**解**(非/迭代) | 20–50 µs(打包后) | 必须打包 |
| halo | 1–2 | ~4 µs | 无需处理 |

**验收线:粗解 ≤ 每迭代总预算的 20–30%(N=2e7 时约 20–30 µs 档)。**

---

## 7. 验证阶梯(怎么知道做对了)

np = 8 → 64 → 512 → 3000,每级测四件事:
1. **迭代数应平**(粗空间生效的直接证据;np=8 已见两级迭代 51→48→40 的苗头);
2. **慢模数 vs P**(-decay 式诊断):无粗空间应 ~1.5×P 增长,有粗空间后残余慢模应 O(1);
3. **每步耗时分解**(本地/halo/Allreduce/粗解),粗解占比 <20–30%;
4. **回收叠加收益**(EP 循环总迭代,对比 cold/fischer),应保持 −50% 量级且不随 P 退化。

任何一级不达标,先查:分区质量(子域长宽比、薄壁是否被劈开)→ GenEO τ 阈值(挑模不足)→ 粗解档位(§4 换档)。

---

## 8. 风险与后备

| 风险 | 触发条件 | 后备 |
|---|---|---|
| GenEO setup 过重 | 子域特征问题慢 | Nicolaides+{1,x,y,z}(4/子域,免特征解)先顶,硬模区域局部加 GenEO |
| 粗解占比超线 | m 过大 / 网络差 | 换 §4 档位;GenEO τ 收紧少挑模;真第三层 |
| PIPECG 数值稳定性 | 病态 + 大 P | PIPECG2 / GROPPCG(都已验证可用) |
| 回收基污染 | 长时间运行漂移 | 滑动窗口(已实现)+ 相对丢弃阈值(fischer_demo 教训) |
| 分区劈开薄壁 | 真实心脏几何 | 分区约束(ParMETIS 权重)或靠 GenEO 吸收 |
| dof/核过小 | 网格只有 2e6 | **降核数到 500–1000**,不硬上 3000 |

---

## 9. 一句话总纲

**3000 核 = 三层各司其职:细层(sASM+O1+ICC1)扫快头、GenEO 粗空间把 O(P) 慢模和薄壁/疤痕硬模钉住(粗解 m=3k 用冗余稀疏、m=15k 用子通信器,冗余稠密早已死)、有界回收窗口(点积打包)吃时间漂移;外层 pipelined CG 把每迭代 ~3 次 collective 藏进计算;全部 setup 被上万次重解摊销。前提是给 3000 核配 ≥1e7 dof 的问题——不够大就少用核,这也是答案的一部分。**
