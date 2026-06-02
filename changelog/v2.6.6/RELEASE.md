# v2.6.6 — DSL 重构 Stage 6：VectorEquation 泛化

## 摘要

完成 DSL 重构系列（Stage 1–6）最后一步，将 `VectorEquation` 扩展为支持完整表达式求值层：
stream 感知执行、逐分量 `ExprTree` RHS 设置、BC 自动注入，以及统一的 `advanceTransient` 接口。

---

## 核心变更

### `VectorEquation` 新增接口

| 方法 | 说明 |
|------|------|
| `setRHSComponent(int c, const ExprTree&)` | 按分量设置 ExprTree RHS（内部降低为 EvalPlan） |
| `setStream(cudaStream_t)` | 将 CUDA stream 传播至所有分量 `Equation` |
| `stream() const` | 查询当前 stream（返回第 0 分量的 stream） |
| `registerBC(const ScalarField&, bcs)` | 向所有分量方程注册边界条件 |
| `advanceTransient(bcs, dt)` | 对所有分量依次执行 Forward-Euler 步进 |

---

## 测试

新增 `test/moduleTest/equation/test_vector_equation.cu`（13 个 GPU 测试）：

| # | 名称 | 说明 |
|---|------|------|
| 1 | stream 传播 | `setStream` 后各分量 stream 一致（5 个断言） |
| 2 | computeRHS 分量 | 各分量 RHS 与标量 Laplacian 参考值一致（2 个断言） |
| 3 | registerBC + ExprTree | 复合子表达式 `lap(v[0]±v[1])` 验证 BC 注入正确性（2 个断言） |
| 4 | advanceTransient | N 步积分结果与逐分量标量参考一致（2 个断言） |
| 5 | 显式 stream | 非默认流 `computeRHS` 结果与默认流一致（2 个断言） |

全量测试：**12/12 通过**。

---

## 实现说明

- `setRHSComponent` 委托给 `equations_.at(c)->setRHS(tree)`，触发带 `BcMap` 的 EvalPlan 降低。
- `advanceTransient(bcs, dt)` 调用每个分量方程自身的 `advanceTransient`，保留各分量字段引用。
- `stream()` 读取 `equations_[0]->stream()`；因 `setStream` 统一设置所有分量，保持一致性。
- BC 自动注入仅对**复合** ExprStencil 子节点生效；简单叶节点（`lap(field)`）无需注册 BC。

---

## 系列总结（v2.6.1 – v2.6.6）

| 版本 | Stage | 描述 |
|------|-------|------|
| v2.6.1 | 1 | ExprTree 节点系统 |
| v2.6.2 | 2 | EvalPlan 降低 |
| v2.6.3 | 3 | BC 自动注入（BcMap） |
| v2.6.4 | 4 | Stream 化（cudaStream_t 全面传播） |
| v2.6.5 | 5 | FusedTerm 编译期表达式模板 |
| **v2.6.6** | **6** | **VectorEquation 泛化** |

