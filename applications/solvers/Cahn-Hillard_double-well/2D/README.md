# Cahn–Hilliard 求解器 — 双井自由能（2D）

## 数学模型

### 控制方程

求浓度场 $c(\mathbf{x}, t)$ 满足 Cahn–Hilliard 方程：

$$\frac{\partial c}{\partial t} = M \nabla^2 \mu$$

化学势 $\mu$ 由下式给出：

$$\mu = f'(c) - \kappa \nabla^2 c$$

其中双井自由能密度的导数为：

$$f'(c) = 2\rho\,(c - c_a)(c - c_b)(2c - c_a - c_b)$$

上式来源于双井自由能密度：

$$f(c) = \rho\,(c - c_a)^2(c - c_b)^2$$

该函数在 $c = c_a$ 与 $c = c_b$ 处各有一个极小值（"双井"），驱动相分离至两个平衡浓度。

### 方程汇总

| 符号 | 含义 |
|------|------|
| $c$ | 浓度（序参量） |
| $\mu$ | 化学势 |
| $M$ | 迁移率 |
| $\kappa$ | 梯度能系数（界面能量） |
| $\rho$ | 自由能势垒高度 |
| $c_a,\,c_b$ | 两相的平衡浓度 |

### 边界条件

当前求解器支持以下边界条件（通过配置文件设置）：

| 类型 | 说明 |
|------|------|
| `Periodic` | 周期性边界 |
| `NoFlux` | 无通量边界（$\partial c / \partial n = 0$，$\partial \mu / \partial n = 0$） |

### 数值方法

- 空间离散：有限差分，二阶中心差分格式
- 时间积分：一阶显式 Euler 格式
- 每步流程：
  1. 对 $c$ 施加边界条件（更新 ghost 单元）
  2. 计算 $\mu = f'(c) - \kappa \nabla^2 c$（稳态方程，写入 $\mu$ 的物理格点）
  3. 对 $\mu$ 施加边界条件
  4. 推进 $c^{n+1} = c^n + \Delta t \cdot M \nabla^2 \mu^n$

### 数值稳定性

显式 Euler 格式对双调和算子 $\nabla^4$ 的稳定性约束：

$$\Delta t \leq \frac{\Delta x^4}{8\,M\,\kappa}$$

对默认参数（$\Delta x = 1,\;M = 5,\;\kappa = 2$）：

$$\Delta t_{\max} = \frac{1}{8 \times 5 \times 2} = 0.0125$$

推荐使用 $\Delta t \leq 0.001$。

---

## 使用方法

### 1. 编译

在项目根目录下执行 CMake 构建：

```bash
cd /path/to/PhiX
mkdir -p build && cd build
cmake ..
make CH_2D
```

编译产物 `CH_2D` 将生成在 `applications/solvers/Cahn-Hillard_double-well/2D/` 目录下。

### 2. 准备配置文件与初始场

在工作目录下创建如下结构：

```
workdir/
├── settings.jsonc          ← 参数配置文件
└── settings/
    └── initial_field/
        ├── c.dat           ← 初始浓度场（可选，从头开始时自动生成）
        └── mu.dat          ← 初始化学势场（可选）
```

配置文件 `settings.jsonc` 示例：

```jsonc
{
    "mesh": {
        "nx": 200,   "ny": 200,    // 网格点数
        "dx": 1.0,   "dy": 1.0,   // 格点间距
        "x0": 0.0,   "y0": 0.0    // 原点坐标
    },

    "initialize": {
        "start_from": "0",         // "0" 从头开始，或填写时间步数以续算
        "dt"        : 0.001,
        "nSteps"    : 10000000
    },

    "boundary_conditions": {
        "x_min": "Periodic",
        "x_max": "Periodic",
        "y_min": "Periodic",
        "y_max": "Periodic"
    },

    "constants": {
        "M"    : 5.0,
        "kappa": 2.0,
        "rho"  : 5.0,
        "ca"   : 0.3,
        "cb"   : 0.7
    },

    "output": {
        "print_interval": 10000,    // 打印进度的步数间隔
        "write_interval": 100000,   // 写出数据的步数间隔
        "format"        : "ALL"     // "BINARY" | "DAT" | "VTK" | "ALL"
    }
}
```

### 3. 运行

```bash
./CH_2D settings.jsonc
```

从已有时间步续算（将 `start_from` 改为对应步数后执行相同命令）：

```bash
./CH_2D settings.jsonc   # settings.jsonc 中 "start_from": "100000"
```

### 4. 输出文件

程序在工作目录下生成 `output/` 文件夹，内含：

| 文件 | 说明 |
|------|------|
| `c_<step>.dat` | 浓度场文本格式 |
| `c_<step>.bin` | 浓度场二进制格式 |
| `c_<step>.vts` | 浓度场 VTK 结构化格点格式（可用 ParaView 可视化） |

### 5. 后处理

可使用 ParaView 打开 `.vts` 文件进行可视化，或使用 Python 读取 `.dat` / `.bin` 文件进行自定义分析。

---

## 参数参考

| 参数 | 推荐范围 | 说明 |
|------|----------|------|
| `M` | 1–10 | 迁移率，越大扩散越快 |
| `kappa` | 0.5–5 | 梯度能系数，控制界面厚度 |
| `rho` | 1–10 | 自由能势垒，越大相分离驱动力越强 |
| `ca` | 0–0.5 | 富 A 相平衡浓度 |
| `cb` | 0.5–1 | 富 B 相平衡浓度 |
| `dt` | ≤ $\Delta x^4/(8M\kappa)$ | 时间步长，须满足稳定性约束 |
