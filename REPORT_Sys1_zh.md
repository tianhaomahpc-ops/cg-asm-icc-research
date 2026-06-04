# Sys1 Monodomain（Crank–Nicolson）求解器选型报告

**对象**：cardioid `hack/femheart.cpp` 中的 Sys1 Monodomain 扩散步
**时间离散**：Crank–Nicolson（θ = 1/2）
**空间离散**：P1（线性）有限元
**问题**：在 ≤3000 核 CPU 上，哪种预条件 / 求解器最优？
**结论先行**：**用最便宜的 Jacobi 或 Block-Jacobi（不要 overlap / sASM / Chebyshev 块解）**；
为 Sys2/Sys3 椭圆系统准备的复杂预条件在这里反而多余。

> 本报告对应整体调研记录 [`INVESTIGATION.md`](./INVESTIGATION.md) 第 8 节，
> 数据均来自 `asm_demo`（MFEM 4.9 + PETSc 3.24，P1 四面体立方体，4 ranks）。

---

## 1. 背景：Sys1 和 Sys2/Sys3 是两种完全不同的矩阵

cardioid 用 ASM(ICC)+CG 解三个系统。最初的疑问是“增大 overlap，迭代次数反而增加”
——但那个反常现象只发生在**椭圆型** Sys2/Sys3 上。Sys1 Monodomain 的原始数据
迭代次数只有 **2–5**，overlap 的影响几乎看不出来。

原因在于**矩阵结构不同**：

| 系统 | 矩阵 | 条件数 κ | 随网格 h | 性质 |
|---|---|---|---|---|
| Sys2 / Sys3（u_e / torso） | 纯刚度 `K` | `O(h⁻²)` | 加密变差 | 病态，需强预条件 |
| **Sys1 Monodomain（C-N）** | `(1/Δt) M + (1/2) K` | `~ 1 + O(Δt/h²)` | 受 Δt/h² 控制 | **质量主导，良态** |

算子分裂把反应项（ODE，用 Rush–Larsen 之类积分，无需解线性系统）和扩散项分开后，
C-N 扩散步要解：

```
A x = b ,   A = (1/Δt) M + (1/2) K
```

- `M` = 质量矩阵，`K` = 刚度矩阵（含 σ），θ = 1/2
- 心脏典型参数（Δt ~ 0.025 ms，h ~ 0.2 mm，D ~ 0.1–1），刚度项只占质量项约
  10–20%，故 **κ(A) ≈ 1.3–2，极其良态**
- **固定 Δt 时 A 是常数矩阵**：整个仿真（几千上万时间步）只装配一次、预条件 setup 一次

---

## 2. 实验设置

`asm_demo` 新增 `-dt` 选项：

- `dt ≤ 0`：装配纯刚度 `A = K`（椭圆，作对照）
- `dt > 0`：装配 C-N monodomain 矩阵 `A = (1/dt) M + (1/2) K`

固定网格 nx=48（约 11.7 万 DOF），4 ranks，CG，rtol=1e-6，扫：

- **dt 区间**：纯 K（椭圆）、1e-2、1e-3、1e-4（心脏真实区）
- **预条件子**：Jacobi、Block-Jacobi/ICC(0)、sASM overlap-1（scheme 3）、
  sASM+Chebyshev overlap-1（scheme 4）

指标用**迭代次数**和 `-log_view` 的 **MPI Reductions**（全局同步数，确定性、
不受 wall-time 噪声影响）。

---

## 3. 实验结果

### 3.1 烟雾测试：质量项把矩阵从病态变良态

同样用最朴素的 Jacobi-PCG：

| 矩阵 | 迭代次数 |
|---|---|
| 纯刚度 K（椭圆） | **223** |
| monodomain dt=1e-4 | **6** |

→ 仅仅加上质量项，Jacobi 的迭代次数从 223 掉到 6，和 cardioid Sys1 实测的 2–5 吻合。

### 3.2 完整对比（格式：`迭代次数 (MPI 全局同步数)`）

| regime | Jacobi | Block-Jacobi/ICC(0) | sASM-O1 (sch3) | sASM+Cheby-O1 (sch4) |
|:--|:--|:--|:--|:--|
| 纯 K（椭圆） | 223 (793) | 132 (520) | 104 (445) | **68 (337)** |
| dt = 1e-2 | 77 (355) | 33 (223) | 27 (214) | 17 (184) |
| dt = 1e-3 | 22 (190) | 12 (160) | 9 (160) | 8 (157) |
| **dt = 1e-4（心脏真实区）** | **6 (142)** | **6 (142)** | 4 (145) | 4 (145) |

---

## 4. 三个关键结论

### 4.1 dt 越小（质量越主导），所有方法迭代数坍缩到个位数，差距消失

- 纯 K（椭圆）：Jacobi 223 vs scheme4 68 → **3.3× 差距**（这里复杂预条件值钱）
- dt=1e-4（心脏区）：Jacobi 6 vs scheme4 4 → **1.5× 差距，绝对值都是个位数**

椭圆区那种“复杂预条件大幅领先”的优势，**随质量项主导而蒸发**。

### 4.2 dt=1e-4 时 Jacobi (6) == Block-Jacobi (6)，块解给零增益

质量主导 ⇒ 矩阵近似对角占优 ⇒ **纯对角（Jacobi）就和块 ICC 一样好**。花哨的块解白费。

### 4.3 反直觉：质量主导区 sASM/scheme4 的总同步反而更多（145 vs 142）

scheme 3/4 迭代少 2 步（4 vs 6），但**单次求解**的全局同步总数反而比 Jacobi 高：

- 每次求解的同步 = `2·iter + 2`（CG 点积）：Jacobi 14 次，sASM 10 次 —— sASM 确实每解省 4 次
- **但** sASM 的 PCSHELL setup + Chebyshev 特征值估计本身要做额外 reduction，在质量
  主导区这点 setup 开销抵消甚至超过那 4 次的节省
- 而且 sASM/scheme4 **overlap=1 需要 halo 交换**，Jacobi/Block-Jacobi **进程间零 PC 通信**

> 注：固定 Δt 时矩阵不变，setup 摊销到几千步后这点开销会消失；但即便摊销后，sASM 每解
> 只省 4 次同步，却换来 overlap halo 通信，对一个 4–6 步的问题**不划算**。

---

## 5. 最终建议（按系统）

| 系统 | 矩阵 | κ | 迭代量级 | 最优预条件 |
|---|---|---|---|---|
| **Sys1 Monodomain (C-N)** | `(1/Δt) M + (1/2) K` | `~1+O(Δt/h²)`，小 | 4–6 | **Jacobi 或 Block-Jacobi/ICC(0)** —— PC 零通信、setup 平凡；3000 核下加 `-ksp_type pipecg` |
| Sys2 / Sys3（u_e / torso） | 纯 `K` | `O(h⁻²)`，大 | 100–350 | sASM / scheme 4（见 INVESTIGATION.md 第 3–7 节）或 AMG |

**一句话**：monodomain 步用 Jacobi-PCG（或 Block-Jacobi/ICC(0)），不要 overlap、
不要 sASM、不要 Chebyshev 块解；把复杂预条件的预算全留给真正难的椭圆 Sys2/Sys3。
这也符合心脏 HPC 文献（Pavarino–Scacchi–Zampini、Niederer benchmark 等）的共识：
parabolic monodomain 步“便宜”，elliptic bidomain 步才“贵”。

---

## 6. 大规模（~3000 核）进一步改进方向 — 暂不采用

> **状态：设计草案，尚未在 ~3000 核上测试，暂不采用。**
> 同步**计数**已在 4 核验证，但**时间收益只在大规模才出现**，crossover 未实测。
> 详见 [`INVESTIGATION.md`](./INVESTIGATION.md) 第 9 节。

迭代数已只有 4–6，无法再靠更强预条件压低；大规模下真正的瓶颈是 **CG 每步那 2 次
全局 `MPI_Allreduce`**。改进方向从“降迭代”转为“消同步”：

1. **时间外推初始猜测** `x₀ = 2Vⁿ − Vⁿ⁻¹` —— 解随时间光滑，起点更近，CG 6→2-3 步，近乎免费
2. **同步无关定常解法（Chebyshev / Richardson + Jacobi）** —— 每步**零 allreduce**；
   实测 Richardson 8 步 / Chebyshev 21 步，均 0 全局同步/步（vs CG 6 步 × 2 同步）
3. **Pipelined CG**（`-ksp_type pipecg`）—— 把 allreduce 与 matvec 重叠（实测 143→122）
4. **固定 Δt 摊销** —— A 不变，装配 + 分解 + 特征值估计只做一次
5. **Mass lumping** —— M 对角化，对角占优更强，Jacobi/Chebyshev 更快（心脏 EP 标准做法）

**组合（理想形态）**：固定 Δt 摊销 + mass lumping（地基）→ 时间外推初始猜测（最小化步数）
→ Chebyshev 当解法（每步零全局同步）。每个时间步几乎无全局通信，只剩邻居 SpMV halo 交换。

**零同步实测佐证**：Chebyshev 用 `-ksp_norm_type none` 跑满 2000 步，
`MPI Reductions` 仍停在 setup 的 ~142（与 CG 跑 6 步的 143 相同）——证明每步零全局 reduction。

---

## 7. 复现命令

```bash
cd asm_bug_demo
make asm_demo

# 烟雾测试：椭圆 vs monodomain（同用 Jacobi），看迭代次数差异
mpirun -n 4 ./asm_demo -fix_level 1 -scheme 0 -nx 48 -dt -1 \
  -pc_type jacobi -ksp_norm_type preconditioned -ksp_rtol 1e-6 -ksp_max_it 2000
mpirun -n 4 ./asm_demo -fix_level 1 -scheme 0 -nx 48 -dt 1e-4 \
  -pc_type jacobi -ksp_norm_type preconditioned -ksp_rtol 1e-6 -ksp_max_it 2000

# 扫 dt × 预条件子（追加 -log_view 读 "MPI Reductions:"）：
#   -dt  取  -1 / 1e-2 / 1e-3 / 1e-4
#   预条件子：
#     -pc_type jacobi
#     -pc_type bjacobi -sub_pc_type icc -sub_pc_factor_levels 0
#     -scheme 3 -pc_type asm -pc_asm_type basic -pc_asm_overlap 1 -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels 0
#     -scheme 4 -pc_type asm -pc_asm_type basic -pc_asm_overlap 1 -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels 0

# 零同步解法实测（monodomain dt=1e-4）：
mpirun -n 4 ./asm_demo -fix_level 1 -scheme 0 -nx 48 -dt 1e-4 -pc_type jacobi \
  -ksp_type chebyshev -ksp_chebyshev_esteig 0,0.1,0,1.1 -ksp_norm_type none \
  -ksp_max_it 2000 -log_view        # 看 MPI Reductions 不随迭代增长
```

工具链（Spack 安装，路径写死在 `asm_bug_demo/Makefile`）：MFEM 4.9.0 / PETSc 3.24.4 /
HYPRE 3.1.0 / METIS 5.1.0 / OpenMPI 5.0.9。换机器需改 Makefile 里的路径。
