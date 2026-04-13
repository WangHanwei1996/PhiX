# PhiX v1.0.0 发布说明

**发布日期**：2026-04-13  
**标签**：`v1.0.0`

---

## 概述

PhiX 是一个面向相场模拟的显式有限差分 GPU 计算套件，基于 CUDA/C++17 实现。  
v1.0.0 为首个正式版本，包含完整的核心五层架构及一个 Spinodal 分解示例。

---

## 核心模块

### Mesh — 网格描述
- 纯参数容器，描述结构化正交网格的尺寸与间距，无大数组分配
- 支持 1D / 2D / 3D 均匀网格
- 支持坐标系：`CARTESIAN`、`CYLINDRICAL`、`SPHERICAL`
- 提供 `write` / `read` 持久化接口

### Field — 物理场
- 在 Mesh 网格基础上附加 ghost cell（halo）层
- 双精度，双时间层（`curr` / `prev`）
- CPU (`std::vector`) 与 GPU (`cudaMalloc`) 各持一份，提供 `toDevice` / `toHost` 同步接口
- 支持 `write` / `read` 自定义 `.field` 二进制格式

### BoundaryCondition — 边界条件
- 抽象基类 `BoundaryCondition`，统一 `applyOnCPU` / `applyOnGPU` 接口
- 内置三种边界条件：
  - **PeriodicBC**：周期边界，ghost cell 包绕到对面物理边界
  - **FixedBC**：固定值（Dirichlet）边界
  - **NoFluxBC**：零通量（Neumann）边界，一阶法向导数为零

### Equation — 方程 DSL
- 表达式构建器，支持加减拼接多个 `Term`
- 内置算子：
  - `lap(f)`：拉普拉斯算子，2 阶中心差分
  - `grad(f, axis)`：梯度分量，2 阶中心差分
  - `pw(f, fn)`：逐点函数，任意用户自定义 CUDA lambda
- 系数运算：`*`、`/`、一元 `-`
- 运行时自动调度 GPU kernel

### Solver — 求解器
- 驱动单方程显式时间推进
- 支持两种时间积分方案：
  - **EULER**：前向 Euler，一阶，每步 1 次 RHS 评估
  - **RK4**：经典四阶 Runge-Kutta，每步 4 次 RHS 评估
- 每阶段自动施加边界条件
- 提供 `step(n)` 批量推进接口

---

## 示例

### quickstart
位置：`tutorials/quickstart/`

演示 Allen-Cahn 方程在 2D 笛卡尔网格上的相场演化：

$$\frac{\partial \phi}{\partial t} = M \nabla^2 \phi + M f'(\phi)$$

其中 $f'(\phi) = \phi - \phi^3$，使用 RK4 积分。

### Spinodal Decomposition
位置：`develop/Spinodal Decomposition/`

Cahn-Hilliard 方程模拟旋节分解：

$$\frac{\partial c}{\partial t} = M \nabla^2 \mu, \quad \mu = f'(c) - \kappa \nabla^2 c$$

---

## 构建

```bash
mkdir build && cd build
cmake ..                          # 默认 sm_75 (Turing)
cmake -DPHIX_CUDA_ARCH=86 ..     # Ampere GPU
make -j$(nproc)
```

依赖：CMake ≥ 3.16、CUDA Toolkit、支持 C++17 的编译器。

---

## 已知限制

- 仅支持结构化正交网格（均匀间距）
- 显式时间积分，稳定性条件由用户负责
- 单 GPU，不支持多卡并行
- 暂不支持自适应时间步长

---

## 文件结构

```
PhiX/
├── include/          # 头文件（五层模块）
├── src/              # CUDA/C++ 实现
├── tutorials/        # 教学示例
├── develop/          # 开发中示例（Spinodal Decomposition）
├── doc/              # 模块文档
├── changelog/        # 版本迭代记录
│   └── v1.0.0/
│       └── RELEASE.md
└── CMakeLists.txt
```
