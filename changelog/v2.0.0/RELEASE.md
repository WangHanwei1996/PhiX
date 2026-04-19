# PhiX v2.0.0 发布说明

**发布日期**：2026-04-19  
**标签**：`v2.0.0`  
**作者**：Wang Hanwei &lt;wanghanweibnds2015@gmail.com&gt;

---

## 概述

v2.0.0 是一次重大版本升级，核心目标是实现 **求解器的完全配置文件驱动**。
所有物理常数、时间参数、网格参数、边界条件、输出设置均从 `.jsonc` 配置文件读取，
主程序中不再出现任何硬编码数值。

与此同时，本版本引入了 **多步 Solver API**，使 STEADY（代数辅助方程）与 TRANSIENT
（时间推进方程）可在同一个 `solver.advance()` 调用中有序执行，彻底取代了以往在主循环
中手动拆分方程求解的写法。

---

## 破坏性变更（Breaking Changes）

| 变更点 | v1.x 行为 | v2.0.0 行为 |
|--------|-----------|-------------|
| `FieldFormat` 定义位置 | 定义在 `ScalarField.h` | 迁移到独立头文件 `IO/FieldFormat.h` |
| `ScalarField::writeBinary/Dat/Vts` | private 成员函数 | 删除；统一由 `IO::writeField()` 负责 |
| `Field.cu` | 独立编译单元 | 拆并入 `ScalarField.cu` / `VectorField.cu`，删除原文件 |
| `Solver` 单方程构造签名 | 不变 | 不变（向后兼容） |

---

## 新增功能

### 1. 多步 Solver API（`include/solver/Solver.h`）

新增 `EquationType` 枚举和 `SolverStep` 结构体，支持将多个方程注册到同一个
`Solver` 对象：

```cpp
enum class EquationType { TRANSIENT, STEADY };

struct SolverStep {
    ScalarField*                    sourceField;  // 施加边界条件的场
    std::vector<BoundaryCondition*> bcs;
    Equation*                       equation;
    EquationType                    type = EquationType::TRANSIENT;
};

// 多步构造
Solver solver(
    {
        { &c,  bcs, &eqMu, EquationType::STEADY    },  // μ = f'(c) − κ∇²c
        { &mu, bcs, &eqC,  EquationType::TRANSIENT }   // dc/dt = M∇²μ
    },
    dt, TimeScheme::EULER);
```

- **STEADY**：将 RHS 直接写入未知量（代数赋值）。
- **TRANSIENT**：执行 Euler 时间积分 `unknown += dt * RHS`。
- 多步模式当前支持 Euler 时间格式；单方程 API 保持向后兼容。

---

### 2. `IO::FieldIO` 模块（`include/IO/FieldIO.h` / `src/IO/FieldIO.cpp`）

统一的场读写与重启辅助接口：

```cpp
// 写出
IO::writeField(c, "output/c_1000.field", FieldFormat::BINARY);

// 原位读入（校验网格尺寸）
IO::readField(c, "output/c_1000.field");

// 重启辅助
int step = IO::resolveStartStep(cfg["initialize"]["start_from"]);
IO::initField(c, step);  // step==0 → 读 settings/initial_field/c.field
                          // step >0 → 读 output/c_{step}.field
```

---

### 3. `IO::OutputWriter` 类（`include/IO/OutputWriter.h` / `src/IO/OutputWriter.cpp`）

配置文件驱动的输出管理器，取代主程序中散落的输出逻辑：

```cpp
IO::OutputWriter writer(cfg["output"]);
// 配置项：print_interval、write_interval、format ("BINARY"/"DAT"/"VTK"/"ALL")

for (int step = start_step; step < nSteps; ++step) {
    solver.advance();
    if (writer.shouldPrint(solver.step)) writer.printProgress(solver.step, solver.time);
    if (writer.shouldWrite(solver.step)) writer.writeFields(c, solver.step, solver.time);
}
```

---

### 4. `BCFactory`（`include/boundary/BCFactory.h` / `src/boundary/BCFactory.cpp`）

从 JSON 配置块构建边界条件集合，消除主程序中的 BC 硬编码：

```cpp
auto  bcSet = buildBCs(cfg["boundary_conditions"]);
auto& bcs   = bcSet.ptrs;
```

配置示例：

```jsonc
"boundary_conditions": {
    "x_min": "Periodic",
    "x_max": "Periodic",
    "y_min": "Periodic",
    "y_max": "Periodic"
}
```

支持 `"Periodic"` 和 `"NoFlux"`；同一轴两侧不匹配时抛出 `std::runtime_error`。

---

### 5. `IO::FieldFormat` 独立头文件（`include/IO/FieldFormat.h`）

将 `FieldFormat` 枚举从 `ScalarField.h` 中提取到独立头文件，降低模块间耦合：

```cpp
enum class FieldFormat { BINARY, DAT, VTS };
```

---

## 应用层更新

### Cahn-Hilliard Double-Well 求解器（`applications/solvers/Cahn-Hillard_double-well/2D/`）

采用上述全部新接口完成重写，主程序中 **零硬编码参数**：

- 网格、时间步、物理常数、边界条件、输出配置全部来自 `settings.jsonc`
- 重启/冷启动逻辑通过 `IO::resolveStartStep` + `IO::initField` 自动处理
- 新增文件头注释，标注作者信息

---

## 内部重构

- `Field.cu` 删除，原有实现拆分并入 `ScalarField.cu` 与 `VectorField.cu`
- `ScalarField` / `VectorField` 的私有 write 方法改为 `IO::writeField` 自由函数
- `Solver.cu` 新增 `multiStepAdvanceGPU` / `multiStepAdvanceCPU` 实现多步推进
- `etc/draft/` 和 `etc/temp/` 中的草稿文件已清理

---

## 升级指南

1. `#include "IO/FieldFormat.h"` 替换原来在 `ScalarField.h` 中对 `FieldFormat` 的依赖。
2. 将主程序中手动调用的 `phi.write(...)` 替换为 `IO::writeField(phi, path, fmt)`。
3. 拆分的双方程求解器建议迁移到多步 `Solver` 构造风格（见上方示例）。
