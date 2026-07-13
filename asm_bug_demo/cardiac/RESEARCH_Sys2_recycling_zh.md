# Sys2(u_e 恢复)回收:结果整理

> 整理对象:Sys2 时间回收(Fischer)在真几何(Niederer 心脏 slab + 立方体躯干)上的全部结果。
> 权威数字来自 `forward_ecg -fischer -T 80 -dt 0.02`(np=8)的新鲜重跑(`sys2_recycling_run.txt`),
> 汇总图 `fig_sys2_recycling.png`。所有数字均标注了代码/图/日志出处。

---

## 1. 一句话结论

**Sys2 的解随时间落在一个低维子空间里,所以回收(Fischer)几乎白拿 −69% 的迭代;而"单个上一步向量"(warm)只 −4%——赢的是子空间,不是邻近。** 大规模上真正省的是**通信**:collective-batched Fischer(cb-Fischer)把维护归约从 2128 压到 314(**6.8× 更少 Allreduce**)。

---

## 2. Sys2 是什么、为什么是回收的靶子

- **Sys2**:心脏上的胞外电位恢复,**奇异 pure-Neumann**:`K_{σi+σe} u_e = −K_{σi} Vm`,核 = span{1}(常数)。(`forward_ecg.cpp:9-12, 710-717`)
- **为什么值得回收**:算子 `K_{σi+σe}` **时间不变**,每个 EP 步都要解一次(~10³–10⁴ 步),**只有右端项 b(t)=−K_{σi}Vm(t) 随时间平滑漂移**。→ 历史解之间高度相关,是回收的教科书场景。
- **奇异处理(三处去均值)**:矩阵挂常数零空间(`AttachConstNullSpace`,每次 CG 迭代投影掉,`:721-722`)、右端每步去均值(`RemoveGlobalMean(b2)`,`:2408-2409`)、每个新基向量去均值(`:2473`)。用 `PetscPCGSolver`(`MatSetNullSpace`)稳健处理奇异,MFEM 的 `CGSolver` 会发散(`:723-728`)。

---

## 3. 方法:Fischer 回收怎么做

每步在**丢弃向量**上测四种初值(保留的物理场始终是干净的 cold x0=0 解,保证 ECG 逐位一致、坏初值不污染历史)(`:2428-2436`):

| 初值 | 定义 | 代码 |
|---|---|---|
| **cold** | x0 = 0(保留的解) | `:2441-2448` |
| **warm** | x0 = 上一步 u_e(`ue_prev`) | `:2441-2448` |
| **Fischer** | x0 = Σᵢ ⟨pᵢ,b⟩ pᵢ,{pᵢ} 是 A-正交历史 | `:2449-2457` |
| **physics** | x0 = c\*·Vm,c\* = ⟨b,K·Vm⟩/⟨K·Vm,K·Vm⟩(单域约化 u_e≈−c Vm,用**当前步** Vm,跨系统) | `:2458-2467` |

**基的构造与维护**(`:2469-2492`):
- 从**干净 cold 解去均值**长出新基向量;
- **CGS2**(经典 Gram-Schmidt,2 趟)做 A-正交化,系数**批成 1 次 Allreduce/趟**;接受阈值 A-范数 > 1e-12;
- **滑窗** `FISCH_MAX = 16`:满了淘汰最旧的(`erase(begin())`)——冻结的 append-only 基会填满早期 QRS 模态、平台段失效,所以必须滑窗(`:747, 2486-2492`)。
- 停机:`RelTol=0, AbsTol=1e-8·‖b‖`,并把 KSP 范数切成 `KSP_NORM_UNPRECONDITIONED`,否则预条件后残差会掩盖初值质量(`:744, 2437-2442`)。

---

## 4. 结果(新鲜实测,80ms,80 个采样步 = 每 1ms)

> 四路测量在采样步(每 1ms)上做,避免每步多解 4 次;保留的 cold 解每步都解。`[FISCHER]` 汇总(`:2814-2827`):

| 初值 | 总 CG 迭代 | 相对 cold |
|---|---|---|
| cold (x0=0) | **7530** | — |
| warm (上一步 u_e) | 7190 | **−4%** |
| physics (x0=c·Vm) | 7338 | **−2%** |
| **Fischer (u_e 历史)** | **2273** | **−69%** |

- **80 个采样步里 55 步 Fischer = 0 迭代**:历史子空间已张成解,投影后的 x0 基本就是解。剩下 ~25 步是快速移动的 QRS 前沿(历史还没覆盖的新方向)。
- **单向量 vs 子空间**:warm(−4%)、physics(−2%)都几乎等于 cold;**只有 Fischer 子空间(−69%)大赢**。→ 关键不是"离上一步近",而是"解落在低维历史子空间里"。

---

## 5. cb-Fischer:大规模真正省的是通信

`fig_fischer_batched.png` / `plot_fischer_batched.py` 的重构(投影点积批成 1 次 Allreduce;单趟 MGS → 2 趟 CGS2):

- **维护 Allreduce:2128(未批 MGS-等价)→ 314(批处理 CGS2)= 6.8× 更少**(本次运行的 `fisch_red_old/fisch_red_new`,`:2824-2827`)。
- CGS2 第二趟给出更好的 A-正交基,Fischer 迭代本身也从早期版本的 2461 降到 1361(采样口径,`plot_fischer_batched.py` 头注)。
- **np=8 看不出**(局部工作主导);但 **P~3000 延迟受限**时,未批的"每基向量 1 次归约"比投影本身贵 10–20×(`:750-761`)。这是把回收从"小规模能用"变成"大规模可扩展"的关键。

---

## 6. 为什么 Sys2 能、Sys3 不能(对照)

| | Sys2(u_e,奇异) | Sys3(躯干 Laplace) |
|---|---|---|
| warm | −4% | −9% |
| Fischer | **−69%** | −11% |
| 机理 | 解落在**低维历史子空间**(平台段 0 迭代) | 残差留在**多尺度慢模**,历史张不住 |

Sys3 的病态是 Laplacian 的 h 细化**多尺度层级**,不是低维子空间——所以回收/初值一律封顶 ~10%(详见 `RESEARCH_Sys3_acceleration_zh.md` §7.9)。**这正是"子空间是主角"的反证**:有低维子空间(Sys2)回收就大赢,没有(Sys3)就没用。

---

## 7. 相关文件

- 代码:`forward_ecg.cpp` `-fischer` 块(选项 `:429`;循环 `:2428-2495`;汇总 `:2814-2827`;`ipbatch` `:750-761`)。
- 图:`fig_sys2_recycling.png`(本整理汇总)、`fig_fischer_eploop.png`(每 ms 迭代曲线)、`fig_fischer_batched.png`(cb-Fischer 前后)、`fig_fischer_demo.png`(教学演示)。
- 数据:`sys2_recycling_run.txt`(本次新鲜运行)、`fischer_eploop.txt`、`fischer_before/after_iters.txt`。
- 脚本:`plot_sys2_recycling.py`、`plot_fischer_eploop.py`、`plot_fischer_batched.py`。
- 报告:`REPORT_T3_xsys_zh.md` §2.1(Sys2 fischer)、`DESIGN_3000core_zh.md`(cb-Fischer / 3000 核)、`topics/04_recycling_fischer/`。
