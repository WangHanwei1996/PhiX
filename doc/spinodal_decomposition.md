# 旋节分解：二维 Cahn–Hilliard 方程

本算例求解二维周期域上的 **Cahn–Hilliard（CH）方程**，模拟二元混合物在淬火进入不稳定区域后发生的**旋节分解**——自发相分离过程。

---

## 问题描述

求浓度场 $c(\mathbf{x}, t)$ 满足：

$$\frac{\partial c}{\partial t} = M \nabla^2 \mu$$

$$\mu = 2\rho\,(c - c_a)(c - c_b)(2c - c_a - c_b) - \kappa\,\nabla^2 c$$

其中 $\mu$ 为化学势，$f'(c) = 2\rho(c-c_a)(c-c_b)(2c-c_a-c_b)$ 是双井自由能密度的导数，$-\kappa\nabla^2 c$ 是梯度能惩罚项。

**计算域**：$\Omega = [0,\,200]^2$，$\Delta x = \Delta y = 1.0$

**边界条件**（两个方向均为周期）：

$$c(0, y, t) = c(L_x, y, t), \qquad c(x, 0, t) = c(x, L_y, t)$$

**初始条件**（围绕平均浓度 $c_0$ 的余弦扰动）：

$$c(\mathbf{x}, 0) = c_0 + \varepsilon \Bigl[\cos(0.105x)\cos(0.11y)
+ \bigl(\cos(0.13x)\cos(0.087y)\bigr)^2
+ \cos(0.025x - 0.15y)\cos(0.07x - 0.02y)\Bigr]$$

**参数**：

| 符号 | 取值 | 含义 |
|------|------|------|
| $M$ | 5.0 | 迁移率 |
| $\rho$ | 5.0 | 自由能势垒高度 |
| $c_a$ | 0.3 | 左侧平衡浓度 |
| $c_b$ | 0.7 | 右侧平衡浓度 |
| $\kappa$ | 2.0 | 梯度能系数 |
| $c_0$ | 0.5 | 平均浓度 |
| $\varepsilon$ | 0.01 | 扰动幅值 |
| $\Delta t$ | 0.001 | 时间步长 |

---

## 为什么需要手动时间循环？

标准的 `solver.run()` 接口针对形如 $\partial\phi/\partial t = \text{RHS}(\phi)$ 的单场方程设计。  
Cahn–Hilliard 方程每个时间步包含**两个耦合步骤**：

1. 由 $c^n$ 计算 $\mu^n$（辅助计算——$\mu$ **不**参与时间积分）。
2. 推进 $c^{n+1} = c^n + \Delta t\,M\nabla^2\mu^n$。

因此，需要构造两个 `Equation` 对象，并手动驱动循环：

```
for each step:
    对 c   施加边界条件   ← 填充 ∇²c 模板所需的 ghost 单元
    eq_1.computeRHS(mu)  ← 计算 μ = f'(c) − κ∇²c，写入 mu 的物理格点
    对 mu  施加边界条件   ← 填充 ∇²μ 模板所需的 ghost 单元
    solver.advance()     ← c ← c + dt · M∇²μ
```

---

## 数值稳定性

显式 Euler 格式作用于 CH 方程中的双调和算子 $\nabla^4$ 时，稳定性约束为：

$$\Delta t \leq \frac{\Delta x^4}{8\,M\,\kappa}$$

对本算例参数（$\Delta x = 1$，$M = 5$，$\kappa = 2$）：

$$\Delta t_\text{max} = \frac{1}{8 \times 5 \times 2} = 0.0125$$

选取 $\Delta t = 0.001$ 远小于此上限，保证了计算稳定。

---

## 代码详解

### 1. 网格（Mesh）

```cpp
Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                200, 1.0, 0.0,   // nx, dx, x0
                                200, 1.0, 0.0);  // ny, dy, y0
```

$200 \times 200$ 的均匀笛卡尔网格，间距为 1。

### 2. 物理场（Field）

两个场：时间积分场 $c$（浓度）和辅助场 $\mu$（化学势，每步覆写）。

```cpp
Field c(mesh, "c", /*ghost=*/1);
Field mu(mesh, "mu", /*ghost=*/1);
```

**$c$ 的初始化**（利用 `mesh.coord` 获取物理坐标）：

```cpp
for (int j = 0; j < mesh.n[1]; ++j)
for (int i = 0; i < mesh.n[0]; ++i) {
    double x = mesh.coord(0, i);
    double y = mesh.coord(1, j);
    double val = c0 + eps * (
          std::cos(0.105 * x) * std::cos(0.11 * y)
        + std::pow(std::cos(0.13 * x) * std::cos(0.087 * y), 2)
        + std::cos(0.025 * x - 0.15 * y) * std::cos(0.07 * x - 0.02 * y)
    );
    c.curr[c.index(i, j)] = val;
}
c.allocDevice();
c.uploadAllToDevice();

mu.fill(0.0);
mu.allocDevice();
mu.uploadAllToDevice();
```

两个场均需在循环开始前完成**分配与上传**，否则 GPU 端数据为空。

### 3. 边界条件（BoundaryCondition）

```cpp
PeriodicBC bc_x(Axis::X);
PeriodicBC bc_y(Axis::Y);
```

两个方向均为周期边界。循环中，同一对 BC 对象先后作用于 $c$ 和 $\mu$。

### 4. 方程（Equation）

**`eq_1`** 计算化学势 $\mu = f'(c) - \kappa\nabla^2 c$，目标场为 `mu`，读取 `c`：

```cpp
Equation eq_1(mu, "CH_1");
eq_1.params["rho"]   = 5.0;
eq_1.params["ca"]    = 0.3;
eq_1.params["cb"]    = 0.7;
eq_1.params["kappa"] = 2.0;

const double rho   = eq_1.params["rho"];
const double ca    = eq_1.params["ca"];
const double cb    = eq_1.params["cb"];
const double kappa = eq_1.params["kappa"];

eq_1.setRHS(
    pw(c, [rho, ca, cb] __host__ __device__ (double c_val) {
        return 2.0 * rho * (c_val - ca) * (c_val - cb) * (2.0 * c_val - ca - cb);
    })
    - kappa * lap(c)
);
```

> **注意**：`pw()` 对场逐格点施加纯量函数，其 lambda **必须**标注 `__host__ __device__` 才能在 GPU 上运行。`lap()` 是场级算子，利用 2 阶中心差分计算 $\nabla^2 c$——它不能在 `pw` 的 lambda 内部调用。

**`eq_2`** 描述时间演化 $\partial c/\partial t = M\nabla^2\mu$，目标场为 `c`，读取 `mu`：

```cpp
Equation eq_2(c, "CH_2");
eq_2.params["M"] = 5.0;
const double M = eq_2.params["M"];

eq_2.setRHS(M * lap(mu));
```

### 5. 求解器与时间循环

只有 `eq_2` 传入 `Solver`，因为只有 $c$ 参与时间积分：

```cpp
Solver solver(eq_2, {&bc_x, &bc_y}, dt, TimeScheme::EULER);
```

输出时刻按对数等间距排列（0.1、1、10、100、1000、$10^4$、$10^5$ s），转换为对应步号：

```cpp
const std::vector<double> out_times = {0.1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5};
std::set<int> out_steps;
for (double t : out_times)
    out_steps.insert(static_cast<int>(std::round(t / dt)));
```

手动时间循环：

```cpp
for (int s = 0; s < nSteps; ++s) {
    // 第一步：为 c 填充 ghost 单元，计算 μ
    bc_x.applyOnGPU(c);
    bc_y.applyOnGPU(c);
    eq_1.computeRHS(mu);

    // 第二步：为 μ 填充 ghost 单元，推进 c
    bc_x.applyOnGPU(mu);
    bc_y.applyOnGPU(mu);
    solver.advance();

    if (out_steps.count(solver.step)) {
        c.downloadCurrFromDevice();
        c.write("output/c_" + std::to_string(solver.step) + ".field");
    }
}
```

`solver.advance()` 在内部推进前会自动对 $c$ 施加构造时注册的边界条件；循环顶部对 $c$ 的显式 BC 调用仅用于填充 `eq_1` 所需的 ghost 单元。  
`bc_x/bc_y.applyOnGPU(mu)` 则为 `eq_2` 中 $\nabla^2\mu$ 模板填充 $\mu$ 的 ghost 单元。

---

## 目录结构

```
develop/Spinodal Decomposition/
├── Cahn-Hillard.cu        # 求解器源文件
├── CMakeLists.txt         # 构建配置
├── postProcess.py         # 可视化脚本
└── output/                # 运行时生成的 .field 快照
    ├── c_0.field          # t = 0
    ├── c_100.field        # t = 0.1
    ├── c_1000.field       # t = 1
    ├── c_10000.field      # t = 10
    ├── c_100000.field     # t = 100
    ├── c_1000000.field    # t = 1000
    ├── c_10000000.field   # t = 1e4
    └── c_100000000.field  # t = 1e5
```

---

## 编译与运行

```bash
# 编译（在 PhiX 根目录下执行）
cd build
touch "../develop/Spinodal Decomposition/Cahn-Hillard.cu"   # 更新时间戳以触发重编译
make spinodal_decomposition

# 运行
cd "../develop/Spinodal Decomposition"
rm -f output/*.field
./spinodal_decomposition
```

每 10 000 步打印一次进度：

```
Starting Cahn-Hilliard simulation (100000000 steps, dt=0.001)
  step 0  t=0  written: output/c_0.field
  [progress] step=10000  t=10
  ...
  step 100000  t=100  written: output/c_100000.field
  ...
```

---

## 后处理

在 Windows 的 ICE conda 环境中运行：

```bash
conda activate ICE
python "postProcess.py"
```

`postProcess.py` 依次读取每个 `.field` 快照，用 `coolwarm` 色图（$c \in [0, 1]$）绘制浓度云图，并将 PNG 保存至 `output/png/`。

---

## 预期结果

| 时间 | 物理现象 |
|------|---------|
| $t = 0$ | 平均场 $c = 0.5$，余弦扰动幅值仅 0.01，几乎均匀 |
| $t \sim 1$ | 不稳定模式放大，$c$ 开始偏离 0.5 |
| $t \sim 10$–$100$ | 相分离清晰可见，形成相互连通的 A 富集域与 B 富集域 |
| $t \sim 10^3$–$10^5$ | 粗化阶段：小液滴合并，特征长度 $\ell \sim t^{1/3}$ |

后期 $t^{1/3}$ 粗化律是守恒序参量动力学（Ostwald 熟化）的经典特征。
