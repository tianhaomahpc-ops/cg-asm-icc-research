# 心脏 EP 前向管线的跨时间步回收:问题、方法与实测

> 自洽的完整版。把 `REPORT_fischer_vm_zh.md`(Sys2)与 `REPORT_fischer_sys13_zh.md`(Sys1/Sys3、
> 窗口大小、计时、粗空间)统一成一份数学表述 + 一张结果总表。
> 代码:`fischer_vm_test.c`、`fischer_sys13_test.c`、`fischer_gram_selftest.cpp`;
> 补丁:`fischer_sliding_window.patch`(改 `forward_ecg.cpp` 的 Sys2 与 Sys3);
> 数据:`fischer_*_regime{0,2}.csv`;图:`fig_fischer_vm.png`、`fig_fischer_sys13.png`、
> `fig_fields_sys123.png`、`fig_window_sweep.png`、`fig_time_compare.png`、`fig_deflate_compare.png`。

---

## 1. 问题

### 1.1 每个时间步要解的三个线性系统

P1 有限元、IMEX 分裂(扩散隐式 Crank–Nicolson、反应显式),每步 $t_n\to t_{n+1}$:

$$
\textbf{Sys1 (单域)}\qquad
\underbrace{\Big(\tfrac{\chi C_m}{\Delta t}\mathbf M+\tfrac12\mathbf K_{\sigma_m}\Big)}_{\textstyle \mathbf A_1}\mathbf V^{n+1}
=\Big(\tfrac{\chi C_m}{\Delta t}\mathbf M-\tfrac12\mathbf K_{\sigma_m}\Big)\mathbf V^{n}
-\chi\mathbf M\mathbf I_{\rm ion}^{n}+\mathbf M\mathbf I_{\rm stim}^{n}
$$

$$
\textbf{Sys2 (胞外恢复)}\qquad
\underbrace{\mathbf K_{\sigma_i+\sigma_e}}_{\textstyle \mathbf A_2}\mathbf u_e^{n+1}=-\mathbf K_{\sigma_i}\mathbf V^{n+1},
\qquad \ker \mathbf A_2=\operatorname{span}\{\mathbf 1\},\quad \textstyle\int_{\Omega_H}u_e=0
$$

$$
\textbf{Sys3 (躯干)}\qquad
\mathbf K_{\sigma_T}\mathbf u_T^{n+1}=\mathbf 0,\quad \mathbf u_T|_\Gamma=\Pi\big(\mathbf u_e^{n+1}|_\Gamma\big)
\;\Longleftrightarrow\;
\underbrace{\mathbf A_{II}}_{\textstyle \mathbf A_3}\mathbf u_I=-\mathbf A_{I\Gamma}\,\mathbf g,\quad
\mathbf g=\Pi(\mathbf u_e|_\Gamma)
$$

三者都是 SPD(Sys2 为 SPSD),而且**算子在整个时间循环里不变,只有右端随 $t$ 变**——这正是回收(recycling)成立的前提。

### 1.2 三条解轨迹都是同一个驱动场的线性像

记 $\mathbf V(t)$ 为跨膜电位的轨迹。则

$$
\mathbf u_e=\mathbf S_2\mathbf V,\quad \mathbf S_2=-\mathbf A_2^{+}\mathbf K_{\sigma_i}\ \ (\text{在零均值子空间上}),
\qquad
\mathbf u_T=\mathbf S_3\,\mathbf g=\mathbf S_3\Pi\mathcal T\,\mathbf u_e
$$

其中 $\mathcal T$ 是迹算子,$\mathbf S_3$ 是"界面数据 → 内部调和延拓"的固定线性映射。于是

$$
\operatorname{span}\{\mathbf u_e^{k}\}=\mathbf S_2\operatorname{span}\{\mathbf V^{k}\},\qquad
\operatorname{span}\{\mathbf u_T^{k}\}=\mathbf S_3\Pi\mathcal T\operatorname{span}\{\mathbf u_e^{k}\}
$$

**推论(后面反复用到)**:回收能拿多少,取决于**解轨迹的有效维数**,而不是"相邻两步像不像"。而 $\mathbf S_3$ 是体导体的调和延拓——一个低通滤波器——所以 $\mathbf u_T$ 的轨迹维数**严格低于** $\mathbf u_e$(实测 9 vs 34,见 §4.2)。

### 1.3 停机判据(不修这条,后面全白做)

PETSc 默认判据是 $\|r_k\|\le \max(\texttt{rtol}\cdot\|r_0\|,\ \texttt{atol})$,而 $r_0=b-Ax_0$ **依赖初值**。初值越好 $\|r_0\|$ 越小,要下降的位数**一点不变** ⇒ 在 `rtol` 判据下任何初值都省不了迭代。所以必须

$$
\texttt{rtol}=0,\qquad \texttt{atol}=\varepsilon\|b\|,\qquad \text{范数取}\ \texttt{KSP\_NORM\_UNPRECONDITIONED}
$$

`forward_ecg.cpp` 的 Sys2/Sys3 路径已经这么做了(`:2547-2548`、`:2651`、`:765`、`:2298`);**Sys1 还没有**(`cg1.SetRelTol(1e-10)`、`iterative_mode=false`,`:688`)。

---

## 2. 方法

### 2.1 滑窗投影(Fischer 类)的数学

设窗口 $\mathbf U=[\mathbf u^{n-m+1},\dots,\mathbf u^{n}]\in\mathbb R^{N\times m}$(Sys2 去均值、Sys3 在 $\Gamma$ 上取零)。初值取

$$
\boxed{\ \mathbf x_0=\arg\min_{\mathbf x\in\operatorname{range}(\mathbf U)}\|\mathbf x^\star-\mathbf x\|_{\mathbf A}\ }
\qquad \|\mathbf y\|_{\mathbf A}^2:=\mathbf y^\top\mathbf A\mathbf y
$$

正规方程:因为 $\langle\mathbf u^k,\mathbf x^\star\rangle_{\mathbf A}=\langle\mathbf u^k,\mathbf A\mathbf x^\star\rangle=\langle\mathbf u^k,\mathbf b\rangle$,

$$
\mathbf G\,\mathbf c=\mathbf f,\qquad
\mathbf G=\mathbf U^\top\mathbf A\mathbf U,\quad \mathbf f=\mathbf U^\top\mathbf b,
\qquad \mathbf x_0=\mathbf U\mathbf G^{+}\mathbf U^\top\mathbf b
$$

**注意 $\mathbf x_0$ 只依赖 $\mathbf b$**,而 $\mathbf b^{n+1}=-\mathbf K_{\sigma_i}\mathbf V^{n+1}$ —— 也就是说这个投影**已经用了本步最新的 $V_m$**,而且用得最优。

若改用 A-正交归一基 $\mathbf P$($\mathbf P^\top\mathbf A\mathbf P=\mathbf I$,$\operatorname{range}\mathbf P=\operatorname{range}\mathbf U$),则 $\mathbf x_0=\sum_i\langle\mathbf p_i,\mathbf b\rangle\mathbf p_i=\mathbf P\mathbf P^\top\mathbf b$ —— **同一个算子的两种写法**。

### 2.2 为什么"用最新 $V_m$ 定系数"不可能更好

任何替代的系数规则 $\mathbf c_{\rm vm}=\arg\min_{\mathbf c}\|\mathbf V^{n+1}-\sum_k c_k\mathbf V^{k}\|_{\mathbf W}$ 诱导的解误差是

$$
\mathbf x^\star-\textstyle\sum_k c_k\mathbf u^{k}=\mathbf S_2\Big(\mathbf V^{n+1}-\sum_k c_k\mathbf V^{k}\Big)
$$

于是**正确的** $V_m$ 侧范数是 $\mathbf W^\star=\mathbf S_2^\top\mathbf A_2\mathbf S_2=\mathbf K_{\sigma_i}\mathbf A_2^{+}\mathbf K_{\sigma_i}$ —— 求它本身要解一次 Sys2。用 $\mathbf M$ 或 $\mathbf K$ 范数是替代品,**必然次优**;而 §2.1 的 A-正交投影在同一子空间里是**唯一最优**。实测 A-范数误差比 $1.26$–$1.44$(§4.1)。

### 2.3 现行实现错在哪:三个不同的 $(m-1)$ 维子空间

A-正交化的定义给出 $\mathbf p_1=\mathbf u^1/\|\mathbf u^1\|_{\mathbf A}$,而 $\mathbf p_2,\mathbf p_3,\dots$ 都**先减掉了 $\mathbf p_1$ 方向的分量**。于是

$$
\operatorname{span}\{\mathbf p_1..\mathbf p_m\}=\operatorname{span}\{\mathbf u^1..\mathbf u^m\}
\quad\text{(完整窗口,没问题)}
$$
$$
\operatorname{span}\{\mathbf p_2..\mathbf p_m\}
=\operatorname{span}\{\mathbf u^1..\mathbf u^m\}\ominus_{\mathbf A}\operatorname{span}\{\mathbf p_1\}
\quad\text{(}\texttt{erase(begin())}\text{ 得到的)}
$$
$$
\operatorname{span}\{\mathbf u^2..\mathbf u^m\}\quad\text{(真正的"忘掉最旧一个解")}
$$

后两个都是 $(m-1)$ 维,但**是完全不同的子空间**。由于一般地 $\langle\mathbf p_1,\mathbf u^k\rangle_{\mathbf A}\neq0\ \forall k$,第二个子空间里**一个 $\mathbf u^k$ 都没有**。诊断量

$$
\eta_{\rm span}=\frac{\big\|\mathbf u^{n-1}-\Pi^{\mathbf P}_{\mathbf A}\mathbf u^{n-1}\big\|_{\mathbf A}}{\|\mathbf u^{n-1}\|_{\mathbf A}}
\qquad(\text{真滑窗恒为 }0)
$$

实测均值 $0.191/0.109/0.099$、峰值达 $0.98$(§4.3)。

### 2.4 修法:保留原始快照 + 增量 Gram

维护 $\mathbf U$、$\mathbf{AU}=\mathbf A\mathbf U$、$\mathbf G=\mathbf U^\top\mathbf A\mathbf U$:

- **滑窗**:删掉 $\mathbf G$ 的第 0 行/列,$\mathbf U,\mathbf{AU}$ 各弹出最旧一列;
- **追加**:新行 $\mathbf g_{\rm new}=[\mathbf U^\top\mathbf A\mathbf u_{\rm new};\ \mathbf u_{\rm new}^\top\mathbf A\mathbf u_{\rm new}]$(一次归约即可);
- **取初值**:$\mathbf f=\mathbf U^\top\mathbf b$(一次归约)+ 截断伪逆 $\mathbf G^{+}$(丢弃 $\lambda\le\tau\lambda_{\max}$)。

每步成本对比(见 §2.7 的 flop 模型):

| 每步 | 现行(CGS2 + 正交基) | 修后(Gram) |
|---|---|---|
| Allreduce | 4 | **2** |
| 向量运算 | $\sim 5m$ | $\sim m$ |
| matvec | 1 | 1 |
| 额外 | — | 一次 $m\times m$ 局部特征分解 |

**没有淘汰发生时两者逐位等价**(实测 90/90 步相同),**有淘汰时才分道扬镳**。

### 2.5 $\eta$ 门控:零 matvec 的命中预判

$$
\eta=\frac{\big\|\mathbf V^{n+1}-\Pi_{\operatorname{span}\{\mathbf V^{k}\}}\mathbf V^{n+1}\big\|}{\|\mathbf V^{n+1}\|}
$$

只需 $m$ 个点积 + 一个 $m\times m$ 小解,**零 matvec**。实测:$\eta<10^{-4}$ 时窗口初值必落进容差(召回 $1.00$,门内准确率 $0.71$–$0.95$)。用途是**调度**:$\eta\ge10^{-4}$ 的步别指望初值,直接上粗空间。

### 2.6 两层加性粗空间(治谱)

$$
\mathbf M^{-1}=\mathbf M_{\rm fine}^{-1}+\mathbf Z\mathbf E^{+}\mathbf Z^\top,\qquad \mathbf E=\mathbf Z^\top\mathbf A\mathbf Z
$$

两种候选 $\mathbf Z$:

| | 构造 | apply 成本 | $\mathbf E$ |
|---|---|---|---|
| **几何**(Nicolaides) | 每子域一列示性函数,**局部支撑** | $O(N)+O(k^2)$ | 一次组装、**一次分解** |
| **快照** | 窗口里的 $\mathbf U$,**全局稠密** | $O(kN)$ **每迭代** | 就是已维护的 $\mathbf G$ |

**为什么快照当粗空间几乎没用(可证)**:若初值已取 $\mathbf x_0=\mathbf U\mathbf G^{+}\mathbf U^\top\mathbf b$,则初始残差满足

$$
\mathbf U^\top\mathbf r_0=\mathbf U^\top\mathbf b-\mathbf U^\top\mathbf A\mathbf U\mathbf G^{+}\mathbf f
=\mathbf f-\mathbf G\mathbf G^{+}\mathbf f=\mathbf 0\quad(\mathbf f\in\operatorname{range}\mathbf G)
$$

即**初始残差已经与整个快照空间正交**,粗校正 $\mathbf Z\mathbf E^{+}\mathbf Z^\top\mathbf r_0=\mathbf 0$。它只能作用在迭代中重新生成的那部分分量上 —— 所以在**同一个子空间已经被当初值用过**之后,再拿它做 deflation 是重复劳动。实测只多 $3.4$ 个百分点,而时间因为 $O(kN)$ 的 apply 反而更差(§4.5)。

### 2.7 成本模型

**串行/本地 flop**(7 点格式、Jacobi 预条件):

$$
\text{一次 CG 迭代}\approx 24N,\qquad
\text{一步窗口维护}\approx (6m+13)N
\;\Longrightarrow\;
\text{窗口}\approx 0.25\,m+0.54\ \text{次迭代/步}
$$

点积可批成 $O(1)$ 次归约,所以这个比值是**每 rank 的本地 flop**,与规模无关。

**并行**(强扩展):

$$
t_{\rm iter}(P)=t_{\rm local}(N/P)+t_{\rm halo}+2\,t_{\rm ar}(P),
\qquad
E(P)=\frac{t_{\rm iter}(P_0)\,P_0}{t_{\rm iter}(P)\,P}
$$

用 `DESIGN_3000core_zh.md` §1 的实测拟合 $t_{\rm local}\approx1.6+4.9\times10^{-3}n_{\rm loc}\ \mu s$、$t_{\rm halo}\approx4\ \mu s$、$t_{\rm ar}\approx2.8\log_2P-7\ \mu s$:

| 网格 | $P=3000$ 的 $n_{\rm loc}$ | 本地 | 通信 | 通信占比 | $E(3000)$ |
|---|---|---|---|---|---|
| $2\times10^6$ | 667 | 4.9 µs | 54 µs | 92% | ~9% |
| $2\times10^7$ | 6667 | 34 µs | 54 µs | **61%** | **~39%** |
| $4\times10^7$ | 13333 | 67 µs | 54 µs | 45% | **~55%** |

**Sys1 的算法侧不会随 $P$ 退化**:屏蔽长度

$$
\ell=\sqrt{\frac{\sigma_m\Delta t}{2\chi C_m}}
=\sqrt{\frac{0.133\times0.01}{2\times1.4}}\ \text{mm}\approx 22\ \mu m\ (\text{横向}\ \approx8\ \mu m)
$$

远小于任何网格尺寸 ⇒ $\mathbf A_1$ 近似对角(实测 cond 1.4、0 个慢模)⇒ 块 Jacobi 近似精确、迭代数与 $P$ 无关。**所以 3000 核上观测到的 ~50% 效率应当全部记在 collective 延迟上**;可检验的判据是:迭代数是否仍是 ~15。

---

## 3. 测试台

容器里没有 MFEM/PETSc,故在**结构同构**的可复现模型问题上测(`fischer_sys13_test.c`):$24^3=13824$ 自由度,同一张网格、同一个 $V_m(t)$ 驱动,350 步,每步一次干净冷启动做基准。

| | 算子 | 右端 | 性质 | cold 迭代/步 |
|---|---|---|---|---|
| Sys1 | $c\mathbf I+\tfrac12\mathbf K_{\sigma_m}$,$c$ 调到 cold≈16(对齐真机 15) | 人造成使精确解 $=\mathbf V(t)$ | SPD、质量主导 | 16.0 |
| Sys2 | $\mathbf K_{\sigma_i+\sigma_e}$,自然 Neumann | $-\mathbf K_{\sigma_i}\mathbf V$,去均值 | 奇异,$\ker=\operatorname{span}\{\mathbf 1\}$ | 169.8 |
| Sys3 | 立方体 $\mathbf K_{\rm iso}$,面 $i{=}0$ 作 Dirichlet 界面 | 界面数据的提升,数据 = **本步 Sys2 解的迹** | 非奇异、病态 | 159.3 |

两个工况:**regime 0** = 快前沿 + 复极弥散(轨迹高秩,保守);**regime 2** = 4 个固定模态、幅值快变(轨迹 4 维,**复现真机的三个基准**:warm −2%、physics −5%、现行 Fischer −69%)。

**两处校验**:
1. **不淘汰时**,正交基实现与真滑窗**逐步完全一致**(90/90 步,迭代数与 A-范数误差都相同)⇒ 后面所有差异只来自淘汰规则;
2. 补丁的增量 Gram 簿记有独立单测(`fischer_gram_selftest.cpp`):增量 vs 从头重算 $9.5\times10^{-16}$;初值满足 A-正交投影的定义性质 $\langle\mathbf u^k,\mathbf b-\mathbf A\mathbf x_0\rangle$ 相对 $2.3\times10^{-11}$。

---

## 4. 结果

### 4.1 "用最新 $V_m$ 定系数":确认无收益

同一窗口、两种系数规则的平均 $\|\mathbf x^\star-\mathbf x_0\|_{\mathbf A}$:

| | regime 0 | regime 1 | regime 2 |
|---|---|---|---|
| A-最优(§2.1) | 4.70 | 28.7 | 0.472 |
| 用最新 $V_m$ 定系数 | 6.78 | 39.5 | 0.597 |
| **比值** | **1.44** | **1.38** | **1.26** |

迭代数上两者几乎相同 —— 因为迭代由**残差 2-范数**和误差的**谱分布**决定,与 A-范数误差不单调相关。
另外 `physics` 初值 $\mathbf x_0=c^\star\mathbf V$(用当前步 $V_m$)实测 $-3.0/-2.8/-5.0\%$,复现真机的 $-2\%$ —— $\mathbf u_e\approx-c\mathbf V$ 只在**等各向异性** $\sigma_e=\lambda\sigma_i$ 时成立,而你们的参数是 $\sigma_i=(0.17,0.019)$、$\sigma_e=(0.62,0.236)$,各向异性比 $8.9$ vs $2.6$。

### 4.2 解轨迹的谱(决定一切的量)

88 个等间隔快照的 Gram 特征值,归一化后计数(regime 0):

| 相对阈值 | $10^{-4}$ | $10^{-6}$ | $10^{-8}$ |
|---|---|---|---|
| Sys1 $\mathbf V_m$ | 10 | 14 | 29 |
| Sys2 $\mathbf u_e$ | 10 | 18 | **34** |
| **Sys3 $\mathbf u_T$** | **3** | **6** | **9** |

**心脏里 34 维的东西,到躯干只剩 9 维** —— $\mathbf S_3$ 是低通滤波器(§1.2)。这解释了为什么 $m{=}8$ 的窗口对 Sys3 绰绰有余,而对 Sys2 只能盖住一部分。

### 4.3 淘汰规则:窗口里连"上一步的解"都没有

| $\eta_{\rm span}$(均值 / 峰值) | regime 0 | regime 1 | regime 2 |
|---|---|---|---|
| 现行实现 | 0.191 / 0.766 | 0.109 / 0.964 | 0.099 / 0.979 |
| 真滑窗 | 0 | 0 | 0 |

逐步轨迹(regime 2,窗口第 16 步填满):第 17 步第一次淘汰,$\eta_{\rm span}$ 从 $5\times10^{-9}$ 跳到 $4.5\times10^{-2}\to0.75$,迭代从 0 跳到 106→166;之后 ~5 步重新学回来,周期 ~15 步。**"学会 → 被淘汰 → 塌成冷启动 → 重新学"。**

### 4.4 窗口大小:各自定,而且都很小

扣掉窗口自身开销(§2.7)后的最优 $m$:

| | 现在的代码 | 迭代最优 | **时间最优(实测)** |
|---|---|---|---|
| Sys1 | 无窗口 | 4 | **4** |
| Sys2 | `FISCH_MAX=16` | 4–12 | **4** |
| Sys3 | `F3MAX=12` | 8–12 | **8** |

两个决定因素:**轨迹有效维数**(盖不住就退化成时间外插器,4 个最近解就够)与**每步迭代数**(决定开销占比:Sys1 每步只有 16 次迭代,$m{=}16$ 就吃掉 4.5 次 = 28%,$m{=}64$ 时维护比求解还贵)。**⇒ 窗口大小并没有配错,错的是淘汰规则。**

### 4.5 完整对比(regime 0,350 步)

迭代数是精确值(运行确定性,两次运行逐位相同);时间是串行本地工作,run-to-run 方差约 10%,
所以下面分成**两组**,每组内部来自同一次运行,不跨组比时间。

**(a) 初值这一层**(`fischer_time_regime0.csv`)

| | 迭代 | 时间 | 相对不做回收 |
|---|---|---|---|
| 不做回收(冷启动) | 118437 | 31.5 s | 1.00× |
| **现行实现** | 119085 | **33.9 s** | **0.93×(慢 8%)** |
| warm(上一步) | 85743 | 22.8 s | 1.38× |
| **最优真滑窗**(m = 4/4/8) | **40736** | **11.7 s** | **2.68×** |

分系统(现行实现 vs 冷启动的**时间**):Sys1 **+43%**、Sys2 **+7.8%**、Sys3 **+4.0%** —— 三个系统全是负收益。

**(b) 再叠一层粗空间**(`fischer_deflate_regime0.csv`)

| | 迭代 | 时间 | 相对不做回收 |
|---|---|---|---|
| 不做回收 | 118437 | 33.0 s | 1.00× |
| 修好的滑窗 | 40736 | 11.9 s | 2.77× |
| **滑窗 + 几何粗空间** | **17901** | **6.21 s** | **5.31×** |
| 滑窗 + 快照当粗空间 | 36763 | 15.6 s | 2.12× |

分系统(相对不做回收,迭代 / 时间):

| | 只修滑窗 | + 几何粗空间 |
|---|---|---|
| Sys1 | −63.5% / −52.2% | −62.7% / −47.4%(**变差**) |
| Sys2 | −56.3% / −55.3% | **−81.6% / −78.3%** |
| Sys3 | −75.3% / −73.6% | **−90.5% / −87.3%** |

**regime 2**(复现真机基准)合计:不做 113064 / 32.0 s → 滑窗 5797 / 2.31 s → 滑窗+几何粗空间 4263 / 2.04 s。

### 4.6 五条结论

1. **现行实现在高秩工况下是负收益**:墙钟比"根本不做回收"还慢 8%(Sys1 +43%、Sys2 +7.8%、Sys3 +4.0%)。它照付窗口维护,却因为 §2.3 的淘汰规则拿不回迭代。
2. **把窗口修对**:迭代 2.91×、时间 2.77×;在复现真机基准的工况下 Sys2 从 −69% 变成 −98.6%。
3. **窗口之后还有约一倍,来自谱**:加几何粗空间后 Sys2 迭代 −81.6%/时间 −78.3%,Sys3 −90.5%/−87.3%,合计 **5.31×**。窗口与粗空间**正交**:前者把容易的步变成 0 迭代,后者把难步变便宜。
4. **别把回收快照当粗空间**(§2.6 的正交性论证):只多 3.4 个百分点,时间反而从 −55.3% 恶化到 −44.1%。已知的"快照 deflation 76→15"成立的前提是**没有同时把它当初值**。
5. **别给 Sys1 配粗空间**:−3.6% 迭代 / +7.3% 时间(cond 1.4、0 慢模)。但 Sys1 该有一个 $m{=}4$ 的小窗口,并且**先修停机判据**(§1.3)。

---

## 5. 落地

### 5.1 补丁

`fischer_sliding_window.patch`,6 个 hunk,对 `cardiac-sim-fakegeo` `git apply --check` 干净通过:

| | Sys2 (`-fischer`) | Sys3 (`-fischer3`) |
|---|---|---|
| 声明 | `:766` | `:2239` |
| 投影 | `:2564` | `:2678` |
| 窗口更新 | `:2586` | `:2692` |

Sys3 那边已经有相对阈值和干净的停机判据,只有淘汰规则要改;改完 Gram–Schmidt 整段消失,阈值也不用再调。

### 5.2 建议顺序

1. **apply 补丁**,跑 `forward_ecg -fischer` 与 `-fischer3`,核对 `[FISCHER]/[FISCHER3]` 汇总;
2. **Sys1**:`SetRelTol(0)+SetAbsTol(10^{-10}\|b\|)`+`KSP_NORM_UNPRECONDITIONED`+`KSPSetInitialGuessNonzero`,先只开 warm,再考虑 $m{=}4$ 的窗口;**不要**给它加粗空间;
3. **Sys2/Sys3 上粗空间**(Nicolaides / GenEO),这是剩下的那一倍;3000 核上粗解要按 `DESIGN_3000core_zh.md` §4 分档(m=3000 用稀疏 Chol);
4. **延迟层**:pipelined CG;Sys1 因为 cond 1.4 最适合 Chebyshev(每迭代零点积);
5. 真机上量一次 **Sys3 解快照的谱**,验证 §4.2 的低通结论(直接回答 `-leadvol` 的"54 维"是不是初值的障碍)。

---

## 6. 诚实边界

- **模型问题**:结构(奇异性、非等各向异性、移动前沿、单向耦合、多拍)对得上,但网格、预条件(Jacobi 而非 bjacobi+ICC)、并行都不同。**绝对数字不可直接搬**,可搬的是**排序与机理**。
- 立方体不是真躯干,**Sys3 的轨迹秩很可能被低估**;regime 2 的 −98% 带上界性质(4 维合成流形)。
- 时间是**串行、只含本地工作**。3000 核上:每省一次迭代额外省 2 次 Allreduce(收益变大),但粗解本身是 collective(收益变小)——**迭代数上的 6.6× 才是可搬的部分**。
- $\eta$ 门控的召回 1.00 是在这三个工况、2400+ 步上测的,不是定理。
