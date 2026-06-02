# PhiX v2.6.1 发布说明

**发布日期**：2026-06-02
**标签**：`v2.6.1`
**作者**：Wang Hanwei <wanghanweibnds2015@gmail.com>

---

## 概述

v2.6.1 是 DSL 求值层重构的第一阶段（Stage 1），引入 `ExprTree` 表达式节点类型系统。
这是一次纯**并行添加**——旧的 `Term` / `RHSExpr` 及所有求值路径保持不变，
新树结构与旧系统共存，用于 ghost 需求推导和静态校验。

---

## 变动详情

### 新增文件

| 文件 | 说明 |
|---|---|
| `include/equation/Expr.h` | `ExprTree` 节点类型体系（ExprLeaf、ExprScale、ExprAdd、ExprMul、ExprNeg、ExprPointwise1、ExprStencil、ExprStencilBinary）；`validateGhostRequirements()` 校验器；`expr_lap/grad/iso_grad/grad_dot` 工厂声明 |
| `src/equation/Expr.cu` | `expr_lap`、`expr_grad`、`expr_iso_grad`、`expr_grad_dot` 工厂实现 |
| `test/moduleTest/equation/test_expr.cpp` | 12 项 CPU 单元测试，覆盖节点构造、ghostRequired 推导、常量折叠、validateGhostRequirements 正/负路径 |
| `test/moduleTest/equation/CMakeLists.txt` | 测试构建配置 |
| `changelog/v2.6.1/RELEASE.md` | 本文件 |

### 修改文件

| 文件 | 修改 |
|---|---|
| `CMakeLists.txt` | phix 静态库加入 `src/equation/Expr.cu` |
| `test/moduleTest/CMakeLists.txt` | 加入 `add_subdirectory(equation)` |

---

## 设计要点

### Local vs Stencil 节点分类

| 类型 | ghostRequired | isLocal |
|---|---|---|
| `ExprLeaf`、`ExprScalar` | 0 | true |
| `ExprScale`、`ExprAdd`、`ExprMul`、`ExprNeg`、`ExprPointwise1` | max(children) | true |
| `ExprStencil` (LAP/GRAD/ISO_GRAD) | stencilWidth=1 | false |
| `ExprStencilBinary` (GRAD_DOT) | stencilWidth=1 | false |

Stencil 节点不向上传播 child 的 ghost 需求——child 会在物化时满足，Stencil 本身只向
外层声明自己读取邻居所需的 halo 宽度。

### 常量折叠

`ExprTree::operator*(double s)` 检测 child 是否已是 `ExprScale`，若是则合并系数，
避免链式 `*` 产生深度嵌套的缩放节点（P5 修复）。

### 向后兼容

- 未改动 `Term.h`、`RHSExpr`、`FieldOps.inl`、`Equation.cu` 的任何求值逻辑。
- `Equation::setRHS` 尚未集成 `ExprTree`；此为 Stage 2 工作。
- 所有原有测试（7/7）继续通过。

---

## 测试结果

```
7/7 tests passed
  module_mesh      PASSED
  module_boundary  PASSED
  module_field     PASSED
  module_scheme    PASSED
  module_operators PASSED
  module_face_ops  PASSED
  module_expr      PASSED  ← 新增 (12 个断言)
```
