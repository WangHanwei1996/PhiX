# PhiX v1.1.0 发布说明

**发布日期**：2026-04-13  
**标签**：`v1.1.0`

---

## 概述

v1.1.0 在 v1.0.0 单标量场框架的基础上，引入了完整的**矢量场支持**——新增 `VectorField`、`VectorEquation`、`VectorSolver` 三层，与已有的五层架构无缝衔接。同时，`ScalarField` 取代原始 `Field` 成为标量场的标准实现，新增多格式 I/O 与更完善的 GPU 管理接口。

---

## 新增功能

### ScalarField — 标量场（替代 Field）

- 统一 `curr` / `prev` 双时间层存储，CPU 与 GPU 各持一份
- Ghost cell（halo）布局，支持 1D / 2D / 3D
- 三种输出格式，由 `FieldFormat` 枚举控制：
  - **BINARY**（默认，`.field`）：紧凑二进制格式，含文本头部；最小文件体积
  - **DAT**（`.dat`）：ASCII 文本，每行 `x y z value`；适合 gnuplot / matplotlib / numpy
  - **VTS**（`.vts`）：VTK XML StructuredGrid CellData；可直接用 ParaView / VisIt 可视化
- `readFromFile` 静态工厂，从文件恢复场并校验网格尺寸
- `print()` 打印字段名、尺寸及 `curr` 的 min / max / mean

### VectorField — 矢量场

- 以 SoA（Structure of Arrays）存储：N 个独立 `ScalarField` 按分量排列
- 分量命名约定：3 分量时命名为 `name_x / name_y / name_z`，其余情况为 `name_0 / name_1 / ...`
- Ghost 布局、索引与 GPU 管理全部委托给底层 `ScalarField`
- 同样支持三种输出格式（`.vfield` / `.dat` / `.vts`），VTS 输出包含矢量 CellData

### VectorEquation — 矢量方程

- 描述 `d(unknown)/dt = RHS`，其中 `unknown` 为 `VectorField`
- 内部持有 N 个 `Equation` 对象（每分量一个），接受 `VectorRHSExpr`
- 新增矢量微分算子工厂：
  - `lap(VectorField)` → `VectorRHSExpr`：对每分量施拉普拉斯
  - `grad(ScalarField)` → `VectorRHSExpr`：标量场梯度，输出矢量表达式
  - `div(VectorField)` → `RHSExpr`：矢量场散度
  - `curl(VectorField)` → `VectorRHSExpr`：旋度（3D 网格）
- 新增 `VectorRHSExpr` 表达式类型，支持与标量的乘法及分量加减

### VectorSolver — 矢量求解器

- 驱动单 `VectorEquation` 的显式时间推进
- 所有分量在每个 RK4 阶段同步评估，保障耦合矢量 PDE 的一致性
- 支持 `EULER` / `RK4` 两种时间积分方案（与标量 `Solver` 一致）
- `run(nSteps, callbackEvery, callback)` 接口，回调传入当前 `VectorSolver` 引用

---

## 改进

### BoundaryCondition
- 接口扩展，支持对 `VectorField` 各分量分别施加边界条件

### CMakeLists.txt
- 新增 `VectorEquation.cu`、`VectorSolver.cu`、`ScalarField.cu`、`VectorField.cu` 编译目标

---

## 文件结构变更

```
include/
├── field/
│   ├── ScalarField.h      ← 新增（取代 Field.h）
│   └── VectorField.h      ← 新增
├── equation/
│   └── VectorEquation.h   ← 新增
└── solver/
    └── VectorSolver.h     ← 新增
src/
├── field/
│   ├── ScalarField.cu     ← 新增
│   └── VectorField.cu     ← 新增
├── equation/
│   └── VectorEquation.cu  ← 新增
└── solver/
    └── VectorSolver.cu    ← 新增
```

---

## 已知限制

- 矢量 `curl` 算子仅在 3D 网格下有效
- 与 v1.0.0 相同的稳定性约束：显式积分，时间步长由用户负责
- 单 GPU，不支持多卡并行
