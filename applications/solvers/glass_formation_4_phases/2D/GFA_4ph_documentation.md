# GFA_4ph 求解器文档（玻璃形成能力 · 4 相 Cu–Zr · 2D）

> 源文件：`applications/solvers/glass_formation_4_phases/2D/GFA_4ph.cu`
> 理论推导：`doc/GFA_theory/variational_equation_to_differential_equation.md`（下文引用其公式编号 G1–G16、F1–F7）
> 作者：Wang Hanwei

本文档分两部分：
1. **求解的方程是什么** —— 物理模型、自由能泛函、变分推导到差分形式的完整链条；
2. **逐函数说明** —— `GFA_4ph.cu` 中每个函数 / lambda / 核函数的作用。

---

## 第一部分：这个求解器在解什么方程

### 1.1 物理图景

模拟二元 Cu–Zr 合金的**玻璃形成与晶化竞争**。用一组相场变量描述每个网格点的物相状态：

| 变量 | 含义 | 类型 | 取值约定 |
|------|------|------|----------|
| `φ₀` | 液相 / 非晶相 | Allen–Cahn 序参量 | 0 缺席, 1 存在 |
| `φ₁` | Cu₁₀Zr₇ 晶体 | Allen–Cahn 序参量 | 0 缺席, 1 存在 |
| `φ₂` | CuZr (B2) 晶体 | Allen–Cahn 序参量 | 0 缺席, 1 存在 |
| `φ₃` | CuZr₂ 晶体 | Allen–Cahn 序参量 | 0 缺席, 1 存在 |
| `η`  | 非晶有序化序参量 | Allen–Cahn 序参量 | 0 液体, 1 非晶玻璃 |
| `c`  | Zr 的摩尔分数 | Cahn–Hilliard 守恒场 | 0–1 |
| `μ`  | 化学势 `μ = ∂f/∂c` | 辅助场 | —— |

四个 `φᵢ` 满足单纯形约束 `Σφᵢ = 1`（由每步的 Gibbs 单纯形投影强制，见 §1.6）。`η` 是独立序参量，**不**进单纯形。

> **当前代码状态**：`c` 在程序里被**固定为 0.5**（Cu₅₀Zr₅₀），Cahn–Hilliard 方程（理论里的 G9）**尚未在主循环中求解**。文件头注释提到 `μ`/CH，但实现是常成分等温晶化。`G_phi*` 的成分依赖也因此被忽略（晶体相按化学计量比处理，本就与 `c` 无关）。

### 1.2 自由能泛函（G10）

整个模型由一个 Ginzburg–Landau 型自由能泛函驱动 `F = ∫_Ω f dΩ`，密度为：

```
f = φ₀ ( f₀(c,T) + h(η)·Δf^SR(T) )            ← 液相体自由能 + 短程有序驱动
  + Σ_{i=1..3} φᵢ fᵢ(T)                         ← 三个晶体相的体自由能（线性插值）
  + Σ_{i<j} w_ij φᵢ² φⱼ²                         ← 相间双阱势垒（界面能的“高度”）
  + w_η η²(1−η)²                                ← η 的双阱
  + w_ex η² Σ_{i=1..3} φᵢ²                       ← 非晶-晶体互斥耦合
  + Σ_{i<j} (ε_ij²/2)|φᵢ∇φⱼ − φⱼ∇φᵢ|²           ← 各向同性梯度能（界面能的“宽度”）
  + (β/2)|∇η|²                                  ← η 的梯度能
```

各项的物理意义：
- `f₀, f₁, f₂, f₃`：CALPHAD 给出的各相**体自由能密度** `[J/m³]`，由 `fᵢ = Gₘ^φᵢ / Vₘ^φᵢ`（F1）。
- `Δf^SR`：Inden 型**短程有序驱动力**，只作用于液相、并经 `h(η)` 调制——只有当 η 长大（液→非晶）时才释放，保证液相在 η=0 处是亚稳的。
- `w_ij φᵢ²φⱼ²`：双阱势垒，配合梯度能决定界面厚度 `δ ≈ √(2ε²/w)` 和界面能。
- `w_ex η²Σφᵢ²`：让非晶序与晶体序互斥（晶体里不能同时是玻璃）。
- 梯度能写成 `|φᵢ∇φⱼ − φⱼ∇φᵢ|²` 的**反对称形式**，是多相场里抑制虚假第三相在两相界面析出的标准构造。

### 1.3 控制方程（G7–G9）

对泛函做变分（理论文档 Eq.1–15 给出 `δF/δq = ∂f/∂q − ∇·(∂f/∂∇q)`），得到：

**非守恒序参量 `φᵢ`（Allen–Cahn，Einstein 求和 over j）：**
```
∂φᵢ/∂t = −Σⱼ L_ij · δF/δφⱼ
        = −Σⱼ L_ij · ( ∂f/∂φⱼ − ∇·(∂f/∂∇φⱼ) )          [G7]
```

**非晶序参量 `η`（Allen–Cahn）：**
```
∂η/∂t = −L_η · ( ∂f/∂η − β∇²η )                          [G8]
```

**成分 `c`（Cahn–Hilliard，标量迁移率）—— 理论存在，代码暂未启用：**
```
∂c/∂t = ∇·( M_c ∇μ ),   μ = δF/δc = ∂f/∂c               [G9]
```

### 1.4 变分导数的展开（G11–G16）

代码把 `δF/δφⱼ` 拆成**体项（bulk）**与**梯度能项（gradient energy）**两部分。

**对 φ₀（G11 + G15, i=0）：**
```
δF/δφ₀ = f₀(c,T) + h(η)Δf^SR
       + 2φ₀(w₀₁φ₁² + w₀₂φ₂² + w₀₃φ₃²)
       + Σₖ ε₀ₖ²[ 2φ₀|∇φₖ|² − 2φₖ(∇φ₀·∇φₖ) − φₖ²∇²φ₀ + φ₀φₖ∇²φₖ ]
```

**对 φₛ（s=1,2,3）（G12 + G15）：**
```
δF/δφₛ = fₛ + 2φₛ Σ_{k≠s} w_sk φₖ² + 2 w_ex η² φₛ
       + Σ_{k≠s} ε_sk²[ 2φₛ|∇φₖ|² − 2φₖ(∇φₛ·∇φₖ) − φₖ²∇²φₛ + φₛφₖ∇²φₖ ]
```

**对 η（G13 + G16）：**
```
δF/δη = 30 φ₀ η²(1−η)² Δf^SR + 2 w_η η(1−η)(1−2η) + 2 w_ex η Σ_{i=1..3} φᵢ²  − β∇²η
```

### 1.5 梯度能的面通量重构（本求解器的关键数值技巧）

直接在格点中心用中心差分离散梯度能里的 `∇²` 与 `∇·∇`，会出现奇偶解耦（棋盘格震荡）。本求解器把每个相对 `(j,k)` 的梯度能变分**精确地**拆成两部分：

```
δe/δφⱼ = ε²(∇φₖ)·A + ε²∇·(φₖ A),      其中 A = φⱼ∇φₖ − φₖ∇φⱼ
```

于是对 `∂φᵢ/∂t` 的贡献 `−L_ij·δe/δφⱼ` 变为：

- **非散度部分（格点中心，`gradE_nd`）**：`−L_ij ε² φⱼ|∇φₖ|² + L_ij ε² φₖ(∇φₖ·∇φⱼ)`
- **散度部分（交错面通量，`divFace`）**：`+L_ij ε² ∇·( φₖ²∇φⱼ − φₖφⱼ∇φₖ )`

两者相加在连续极限下逐项还原 G15，但**扩散性的散度项**现在是守恒的、紧致 5 点耦合的交错面通量（无棋盘格），只剩下不可约的旋转源项（真正的 `∇·∇` 点积，不是散度）留在格点中心。这套 `cell → interp/faceGrad → facePW → divFace` 的面通量链来自 `FaceOps`，与 `dendrite_growth.cu` 同构。

### 1.6 迁移率矩阵 = 图拉普拉斯（Steinbach 等价）

`L_ij` **不是**裸的两两迁移率邻接矩阵（那会让能量可能增长、发散），而是其**加权图拉普拉斯** `L = D − A`：

```
        j=0           j=1           j=2           j=3
i=0 [ L01+L02+L03    −L01          −L02          −L03        ]
i=1 [ −L01           L01+L12+L13   −L12          −L13        ]
i=2 [ −L02           −L12          L02+L12+L23   −L23        ]
i=3 [ −L03           −L13          −L23          L03+L13+L23 ]
```

- 对角线 = 该相所有两两迁移率之和（自弛豫项），非对角 = `−`两两迁移率。
- 这等价于 Steinbach 的两两形式 `∂φᵢ/∂t = −Σ_{j≠i} M_ij(δF/δφᵢ − δF/δφⱼ)`。
- 性质：`L` 半正定 ⇒ `dF/dt = −gᵀLg ≤ 0`（耗散、稳定）；行和为 0 ⇒ `Σᵢ ∂φᵢ/∂t = 0`（`Σφᵢ` 守恒）。

因此 `buildCellRHS(i)` 和 `assembleAllFlux()` 都对**所有 j（包括 j=i 自项）**求和，而不是只对 `j≠i`。

### 1.7 数值方案总览

| 环节 | 方法 |
|------|------|
| 时间积分 | 显式 Euler（`TimeScheme::EULER`） |
| 耦合方式 | `EquationSystem` **同步更新**：所有 RHS 从同一时间层 n 算完再统一推进（全耦合多相 AC 的正确选择，而非算子分裂） |
| 空间离散 | 体项格点中心；梯度能扩散部分用交错面通量（`FaceOps`） |
| φ 约束 | 每步后 `k_proj_simplex4`：裁负 + 归一化到 `Σφᵢ=1`（Gibbs 单纯形投影，恢复相竞争） |
| η 约束 | 每步后 `k_clamp01`：硬钳到 `[0,1]` |
| 形核激活 | 每步对 η 加高斯热噪声 `k_noiseClamp`（η 源项 ∝η² 在 η=0 处为零，无噪声无法形核） |
| 边界 | 由配置构造（测试用周期边界） |

**稳定性**（见测试配置注释）：`dt ≤ dx²/(2·L_eff·ε²_max)`。测试取 `dx=0.6 nm, dt=5e-10 s`，约 1.8× 安全系数。

---

## 第二部分：逐函数说明

文件结构：①文件级物理常数 → ②CALPHAD 热力学函数 → ③短程有序 → ④开关/势垒函数 → ⑤GPU 约束/噪声核 → ⑥`main`（含一系列构造 RHS 的 lambda）。

### 2.1 文件级常数（`GFA_4ph.cu:78–110`）

非函数，但是所有方程的物理参数，全部来自理论文档参数表（Wang & Napolitano 2012）：
- `R_gas = 8.314` 气体常数 `[J/(mol·K)]`
- `eps01_sq … eps23_sq`, `beta`：梯度能系数 `ε_ij²`、`β` `[J/m]`
- `w01 … w23`, `w_eta`, `w_ex`：双阱势垒高度 `[J/m³]`
- `L01 … L23`, `L_eta`：两两界面迁移率 `[m³/(s·J)]`

### 2.2 CALPHAD 热力学函数

均为 `__host__ __device__ inline`，可在 CPU 与 GPU 两端调用。

| 函数 | 行 | 作用 |
|------|----|------|
| `G_Cu_liq(T)` | 116 | 纯 Cu 液相的 SGTE Gibbs 自由能 `°G_Cu^liq(T)`，分 `T≤1357.77 K` 与高温两段多项式 `[J/mol]`。 |
| `G_Zr_liq(T)` | 126 | 纯 Zr 液相 Gibbs 自由能 `°G_Zr^liq(T)`，分 `T≤2128 K` 两段 `[J/mol]`。 |
| `L_CuZr_liq(T)` | 136 | 液相 Cu–Zr 正规溶液**相互作用参数** `L = −68890 + 16.20T`（对应 F7 末项系数）。 |
| `compute_Gliq(c,T)` | 141 | 液相摩尔 Gibbs 自由能 `G_liq(c,T)`（F7）：参考项 `(1−c)G_Cu + c G_Zr` + 理想混合熵 `RT[(1−c)ln(1−c)+c ln c]` + 过剩项 `c(1−c)L`。先把 `c` 钳到 `[1e-12, 1−1e-12]` 防 `log` 奇点。**注意**：返回的是 `[J/mol]`，使用处再除以 `Vm_liq` 转成 `[J/m³]`。 |
| `compute_dGliq_dc(c,T)` | 150 | 上式对 `c` 的解析导数 `∂G_liq/∂c`，即液相中 Zr 相对 Cu 的化学势 `μ₀`。CH 方程启用后会用到；当前 `c` 固定故未实际进入演化。 |
| `G_phi1(T)` | 163 | Cu₁₀Zr₇ 化学计量相 Gibbs 自由能（F3），三段温区多项式，仅依赖 `T`。 |
| `G_phi2(T)` | 180 | CuZr (B2) 化学计量相（F4），三段。 |
| `G_phi3(T)` | 197 | CuZr₂ 化学计量相（F5），三段。 |

> 三个 `G_phiN` 是化学计量化合物，不含 `c` 依赖；在 `main` 里被一次性换算为体能量密度 `fN_val = G_phiN(T)/Vm_phiN`。

### 2.3 短程有序驱动力

**`compute_delta_f_SR(T, Tg, alpha, p, Vm)`（`224`，纯 host）**
计算 Inden 型短程有序驱动力 `Δf^SR = −R Tg ln(1+α) f(τ) / Vm`，`τ = T/Tg`：
- 先算归一化常数 `A`（与 Inden 级数系数 79/140、474/497 等相关）；
- `τ<1`（过冷液体）与 `τ≥1` 分别用不同的级数展开 `f(τ)`；
- 末尾除以摩尔体积 `Vm` 把 `[J/mol]` 转成能量密度 `[J/m³]`。
此值在 `main` 里预计算为标量 `dFSR`，进入 φ₀ 与 η 的方程。

### 2.4 开关 / 势垒函数（`__host__ __device__ inline`）

| 函数 | 行 | 公式 | 用途 |
|------|----|------|------|
| `h_func(x)`  | 249 | `x³(10−15x+6x²)` | 标准 5 阶插值函数 `h`，`h(0)=0, h(1)=1, h'(0)=h'(1)=0`。用于 φ₀ 体项里 `h(η)Δf^SR`，保证液相在 η=0 亚稳。 |
| `h_prime(x)` | 252 | `30x²(1−x)²` | `h` 的导数。**当前未被直接调用**——η 方程里把 `30φ₀η²(1−η)²` 内联展开了（数值等价于 `φ₀·h'(η)`）。保留以备复用。 |
| `g_prime(x)` | 255 | `2x(1−x)(1−2x)` | 双阱势 `g(x)=x²(1−x)²` 的导数，用于 η 方程的 `2w_η g'(η)` 项。 |

### 2.5 GPU 核函数（约束 / 噪声）

| 核 | 行 | 作用 |
|----|----|------|
| `k_clamp01(d_curr, …)` | 267 | 把某个场的物理格点硬钳到 `[0,1]`。显式 Euler 会让序参量冲出 `[0,1]`，进而让刚性双阱/CALPHAD 项发散；每步钳回保证物理性与稳定性。用于 **η**。 |
| `k_proj_simplex4(p0,p1,p2,p3, …)` | 290 | 四个相分数的 **Gibbs 单纯形投影**：先把每个 φ 裁到 ≥0，再按和归一化到 `Σφᵢ=1`；若一格四相全≈0（退化）则默认设为纯液相 `φ₀=1`。恢复相竞争——否则对角-L 近似下每个晶体相会各自饱和到 1。 |
| `k_initStates(states, seed, n)` | 319 | 为每个物理格点初始化一个 cuRAND 状态（`curand_init`），整次运行只调用一次。 |
| `k_noiseClamp(d_curr, states, …, mean, std)` | 327 | 给 **η** 的每个物理格点加一次高斯噪声 `N(mean, std²)` 再钳回 `[0,1]`。η 从 0 出发、源项 ∝η² 在 0 处为零，没有这个随机踢动就永远停在液相、无法形核。`noise_std=0` 时主循环跳过它。 |

> 注意三个核的索引换算 `idx = (ix+ghost) + sx*((iy+ghost) + sy*ghost)`：2D 场里 z 的存储索引固定为 `ghost`，`sx=storedDims[0]`、`sy=storedDims[1]` 含 ghost halo。

### 2.6 `main`（`348`）

总体流程：读配置 → 建网格 → 读时间/物理参数 → 建场并上传 GPU → 建边界 → 建交错面场 → 用一组 lambda 拼出每个方程的 RHS → 组装 `EquationSystem` → 时间循环（组装面通量 → 同步推进 → 投影/钳制 → 输出 → η 噪声）。

#### 2.6.1 配置与初始化（`352–429`）
- `IO::ConfigFile::fromArgs`：读 JSONC 配置（默认 `settings/settings.jsonc`，可由 `argv[1]` 覆盖）。
- 第 1 块：`Mesh::makeUniform2D` 建均匀笛卡尔网格。
- 第 2 块：时间步 `dt`、步数 `nSteps`。
- 第 3 块：温度 `T`、各相摩尔体积 `Vm_*`、短程有序参数 `T_g/alpha/p_SR`、噪声参数 `noise_mean/std/seed`；预计算 `f1_val,f2_val,f3_val`（晶体体能量密度）与 `dFSR`。
- 第 4 块：建 6 个 `ScalarField`（`phi0..3, eta, c`，ghost=1），`c` 填 0.5；按 `start_from` 解析重启步并 `IO::initField` 读入初值；`allocUp` lambda 给每个场分配设备内存并上传。

#### 2.6.2 边界与交错面场（`434–468`）
- `buildBCs(mesh, cfg["boundary_conditions"])`：从配置构造边界条件集合 `bcs`（框架 `BCFactory`）。
- `makeFaceVec(ax, tag)`（lambda）：为 4 个相各建一个指定轴的 `FaceField`，返回 `vector<FaceField>`。据此建：
  - `pX[m]/pY[m]`：φₘ 插值到 x/y 面的值；
  - `gX[m]/gY[m]`：φₘ 在 x/y 面上的面梯度；
  - `Gx[i]/Gy[i]`：第 i 个方程的梯度能**总面通量**累加器；
  - `t1x,t2x,t1y,t2y`：逐对 `(j,k)` 计算的面端 scratch。
- `allocUpFace`（lambda）：把面场清零、分配设备内存并上传。

#### 2.6.3 构造 RHS 的 lambda（核心，`511–603`）

| lambda | 行 | 返回/作用 |
|--------|----|-----------|
| `gradE_nd(phi_j, phi_k, eps_jk_sq, Lij)` | 511 | 返回梯度能的**非散度残差** `RHSExpr`：`−Lij ε² φⱼ|∇φₖ|² + Lij ε² φₖ(∇φₖ·∇φⱼ)`。用 `mul` + `grad_dot` 组合（见 §2.7）。散度部分另由面通量处理。 |
| `bulk_phi1(Lij)` | 521 | 返回 `−Lij·δF/δφ₁` 的**体项**（去掉梯度能）：`−Lij f1_val` − 双阱 `2Lij φ₁(w01φ₀²+w12φ₂²+w13φ₃²)` − 互斥耦合 `2Lij w_ex η²φ₁`。 |
| `bulk_phi2(Lij)` | 529 | φ₂ 的体项（结构同上，换系数）。 |
| `bulk_phi3(Lij)` | 537 | φ₃ 的体项。 |
| `bulk_phi0(Lij)` | 548 | φ₀ 的体项：液相 CALPHAD 驱动 `−Lij(G_liq(c,T)/Vm_liq + h(η)dFSR)`（用 2 场 `pw(c, eta, …)`）− 双阱 `2Lij φ₀(w01φ₁²+w02φ₂²+w03φ₃²)`。φ₀ **无** `w_ex` 耦合。 |
| `bulk(j, Lij)` | 559 | 按相序号 `j` 分派到上面四个 `bulk_phiN`。 |
| `buildCellRHS(i)` | 592 | 拼出第 i 个 φ 方程的**全部格点中心 RHS**：对所有 `j`（含 `j=i` 自项，因为 Lmat 是图拉普拉斯）累加 `bulk(j,Lij)`，再对所有 `(j,k), k≠j` 累加 `gradE_nd`。散度部分不在这里。 |

矩阵常量：`Lmat[4][4]`（图拉普拉斯，`568–580`）、`eps2[4][4]`（对称 `ε_jk²` 表，`581–586`）。

#### 2.6.4 方程对象（`619–643`）
- `eqPhi0..3`：4 个 `Equation`，`setRHS( buildCellRHS(i) + divFace(Gx[i], Gy[i]) )`——格点中心残差 + 梯度能散度面通量。
- `eqEta`：η 的 Allen–Cahn 方程，`setRHS(...)` 直接写出：
  - 2 场 `pw(phi0, eta, …)` 给短程有序 `−L_η·30φ₀η²(1−η)²dFSR` + 双阱 `−L_η·2w_η g'(η)`；
  - `−mul(eta, φ₁²+φ₂²+φ₃², 2 L_η w_ex)` 互斥耦合；
  - `+L_η β lap(eta)` 梯度能（格点中心 Laplacian，η 不走面通量）。

#### 2.6.5 每步面通量组装（`654–694`）

| lambda | 行 | 作用 |
|--------|----|------|
| `addPairFlux(accX, accY, j, k, eps_jk_sq, Lij)` | 654 | 把一个 `(j,k)` 对的梯度能散度通量 `+Lij ε²∇·(φₖ²∇φⱼ − φₖφⱼ∇φₖ)` 累加进 `(accX, accY)`。面上通量分量 = `Lij ε²[ (φₖ_f)²(∂φⱼ)_f − (φₖ_f)(φⱼ_f)(∂φₖ)_f ]`，系数用面插值 `pX/pY`、导数用面梯度 `gX/gY`，是标准守恒有限体积通量。用 `facePWGPU`（2 场 / 3 场）实现。 |
| `assembleAllFlux()` | 673 | 每个时间步开头调用：①对所有 φ 应用边界（刷新 ghost，供面梯度与格点残差使用）；②对 4 个相做 `interpGPU`+`faceGradGPU` 得到共享的 `pX/gX/pY/gY`；③对每个方程 i 先把 `Gx[i]/Gy[i]` 清零，再对所有 `j`（含自项）、所有 `k≠j` 调 `addPairFlux` 累加。 |

#### 2.6.6 耦合系统与时间循环（`701–786`）
- `EquationSystem sys(dt, TimeScheme::EULER)`：把 5 个方程 `add(eq, bcs)` 进去，设置 `step/time` 起点。**同步**推进——所有 RHS 取自同一时间层。
- `IO::OutputWriter writer(cfg["output"])`：按配置控制打印/写出节奏与重启；`start_step==0` 时先写初始场。
- 钳制核启动配置 `clampBlk/clampGrd` 与 `cg/csx/csy`；`cudaMalloc` 出 cuRAND 状态并 `k_initStates` 播种。
- **主循环 `for s in [start_step, nSteps)`**：
  1. `assembleAllFlux()`（同时刷新所有 φ ghost）；
  2. `sys.advance()` 显式 Euler 同步推进全系统；
  3. `k_proj_simplex4`（φ 投影到单纯形）+ `k_clamp01`（η 钳到 `[0,1]`）；
  4. `writer.shouldPrint/shouldWrite` 控制诊断打印与写场；
  5. 若 `noise_std≠0`，`k_noiseClamp` 给 η 加高斯噪声为下一步的形核做准备。
- 收尾 `cudaFree(d_states)`。

### 2.7 用到的框架算子（非本文件定义，速查）

| 算子 | 头文件 | 含义 |
|------|--------|------|
| `pw(f, fn[, coeff])` / `pw(f1,f2,fn)` / `pw(f1,f2,f3,fn)` | `equation/Term.h` | 逐点变换，1/2/3 场输入；functor 用 `PHIX_FN` 宏保证 host+device 双端编译。 |
| `mul(a, b[, coeff])` | `equation/Term.h` | 逐点乘 `coeff·(a⊙b)`，a/b 可为场或表达式。 |
| `grad_dot(f, g[, coeff])` | `equation/Term.h` | `∇f·∇g`，二阶中心差分（需 ghost）。 |
| `lap(f[, coeff])` | `equation/Term.h` | Laplacian `∇²f`（格点中心）。 |
| `interpGPU(cell, axis, face)` | `operators/FaceOps.h` | 格点值插值到指定轴的面。 |
| `faceGradGPU(cell, axis, face)` | `operators/FaceOps.h` | 在面上算 `∂_axis(cell)`。 |
| `facePWGPU(out, a[, b[, c]], fn)` | `operators/FacePW.inl` | 对 1/2/3 个 `FaceField` 做逐面逐点变换。 |
| `divFace(flux_x, flux_y[, coeff])` | `operators/FaceOps.h` | 从面通量做守恒散度，返回 `Term` 进 RHS。 |
| `buildBCs(mesh, json)` | `boundary/BCFactory.h` | 从配置块构造边界条件集合。 |

---

## 附录 A：已知局限与物理状态

- **`c` 固定、CH 未求解**：当前是常成分等温晶化；要复现论文的成分动力学需启用 G9（`compute_dGliq_dc`/`M_c`/`kappa_c` 已就绪）。
- **温度调度未实现**：`T` 取静态 `constants.T`；配置里的 `T_start/T_end/cooling_rate` 暂不生效。
- **线性体自由能 ⇒ 液相线性不稳定**：体能量用 `Σφᵢ fᵢ`（线性插值），故 `∂f/∂φᵢ|_{φᵢ=0}=fᵢ≠0`——过冷液体角点不是局部极小，整个域会自发体晶化（无形核势垒）。这是 GFA 范式的预期行为（液相本就该不稳定并转变），晶体 vs 玻璃的竞争由 η 噪声 + `w_ex` 耦合体现。若想要严格的形核势垒，可改用 `h(φᵢ)` 插值体能量或双障碍势（模型决策，未实现）。
- **`w_ex` 致 η 衰减偏刚**：晶化后 `w_ex η²Σφᵢ²` 把 η 快速压回 0（`Δη≈−1.6η/步 @ dt=5e-10`，靠 `[0,1]` 钳制兜底）；若该过程重要应减小 `dt`。

## 附录 B：如何运行（测试案例）

从 `test/` 目录执行（详见 `test/settings/settings.jsonc` 顶部注释）：
```bash
cd applications/solvers/glass_formation_4_phases/2D/test
python3 gen_initial_field.py          # 生成初始场（液相背景 + CuZr 晶核）
mkdir -p output
../../../../../../build/applications/solvers/glass_formation_4_phases/2D/GFA_4ph
```
配置要点：`mesh` 256×256、`dx=dy=6e-10 m`、`dt=5e-10 s`、周期边界、`T=1100 K`、`c=0.5`。
