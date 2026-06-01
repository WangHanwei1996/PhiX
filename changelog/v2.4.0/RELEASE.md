# PhiX v2.4.0 发布说明

**发布日期**：2026-06-01  
**标签**：`v2.4.0`  
**作者**：Wang Hanwei <wanghanweibnds2015@gmail.com>

---

## 概述

v2.4.0 完成了离散格式重构的第一阶段：

- 新增 `scheme` 模块，并落地 `CD2`（二阶中心差分）
- 新增 `operators` 模块，将基础 `lap/grad` 从 `equation` 中拆出
- `Term` 与 `Equation` 接入 `ghostRequired` 需求链
- 补全 `moduleTest/scheme` 与 `moduleTest/operators`，并纳入 CTest

本版本在行为上保持与旧实现一致（默认仍为 CD2），重点是建立后续高阶格式（如 CD4）扩展所需的模块边界。

---

## 变动详情

### 1. 新增 `scheme` 模块

**新增文件**：
- `include/scheme/Scheme.h`
- `include/scheme/CentralDifference.h`

本版本提供 `scheme::CD2`：

- `ghostRequired() == 1`
- `order() == 2`
- `d1()` / `d2()` 提供一阶/二阶差分离散内核（`__host__ __device__`）

### 2. 新增 `operators` 模块并拆出基础算子

**新增文件**：
- `include/operators/Laplacian.h`
- `include/operators/Gradient.h`
- `src/operators/Laplacian.cu`
- `src/operators/Gradient.cu`

`lap/grad` 基础工厂改为由 `operators` 提供，默认分派到 `scheme::CD2`。

### 3. `equation` 模块收口

**修改文件**：
- `include/equation/Term.h`
- `include/equation/Equation.h`
- `src/equation/Equation.cu`

关键变化：

- `Term` 新增 `ghostRequired`
- `Equation` 新增 `requiredGhost()` 查询
- `Equation::setRHS` 汇总表达式的最大 ghost 需求
- 移除 `Equation.cu` 中重复的基础 `lap/grad(ScalarField)` 实现，避免与 `operators` 重复定义

### 4. 构建与测试接入

**修改文件**：
- `CMakeLists.txt`
- `test/moduleTest/CMakeLists.txt`
- `test/moduleTest/scheme/CMakeLists.txt`
- `test/moduleTest/scheme/test_scheme.cu`
- `test/moduleTest/operators/CMakeLists.txt`
- `test/moduleTest/operators/test_operators.cu`

已将 `module_scheme`、`module_operators` 纳入 CTest。

---

## 验证结果

在 `build/` 目录执行：

- `cmake --build . -j4`
- `ctest --output-on-failure`

结果：

- 构建成功
- `module_mesh`、`module_boundary`、`module_field`、`module_scheme`、`module_operators` 全部通过

---

## 兼容性说明

- 现有 `lap/grad` 调用方式保持不变（默认 CD2）
- 该版本主要是模块解耦与能力预埋，为后续高阶 stencil/ghost 扩展做准备
