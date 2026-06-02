# PhiX v2.6.2 发布说明

**发布日期**：2026-06-02
**标签**：`v2.6.2`
**作者**：Wang Hanwei <wanghanweibnds2015@gmail.com>

---

## 概述

v2.6.2 是 DSL 求值层重构的第二阶段（Stage 2），引入 `EvalPlan`（lowering pass）并将
`Equation::setRHS` 扩展为接受 `ExprTree`。表达式树在 `lowerExprTree()` 中被转换为
有序的 `EvalStep` 列表，执行路径与旧 `RHSExpr` 完全数值等价。

---

## 变动详情

### 新增文件

| 文件 | 说明 |
|---|---|
| `include/equation/EvalPlan.h` | `EvalStep`（执行单元，有 LOCAL/STENCIL 标签）和 `EvalPlan`（有序步骤列表）声明；`lowerExprTree()` 公共入口函数声明 |
| `src/equation/EvalPlan.cu` | lowering pass 实现：`lowerToSteps` 递归遍历，ExprLeaf→pw，ExprMul→materialise+mul_accumulate，ExprStencil{LAP/GRAD/ISO_GRAD}→Term stencil，ExprStencilBinary{GRAD_DOT}→grad_dot，ExprScalar→pw-constant；`lowerExprTree` 调用 `validateGhostRequirements` 后执行递归 |
| `test/moduleTest/equation/test_evalplan.cu` | 10 项 GPU 集成测试，覆盖 lap、grad、sum、grad_dot、coefficient、negation、two-field、Hadamard、constant-fill、ghost validation |
| `changelog/v2.6.2/RELEASE.md` | 本文件 |

### 修改文件

| 文件 | 修改 |
|---|---|
| `include/equation/Equation.h` | 增加 `setRHS(const ExprTree&)` 声明；增加 `eval_plan_` 成员（`unique_ptr<EvalPlan>`）；增加显式析构声明（破除 EvalPlan 循环包含）；`hasRHS()` 兼容两路径 |
| `src/equation/Equation.cu` | `setRHS(const ExprTree&)` 实现（调用 `lowerExprTree`）；`computeRHS`/`computeRHSCPU` 增加 `eval_plan_` 分支；增加 `Equation::~Equation() = default` |
| `CMakeLists.txt` | phix 库加入 `src/equation/EvalPlan.cu` |
| `test/moduleTest/equation/CMakeLists.txt` | 加入 `module_evalplan` 构建和 ctest 注册 |

---

## 设计要点

### ExprTree → EvalPlan lowering

```
lowerToSteps(node, coeff, layout):
  ExprScale/Neg   → accumulate coeff, recurse
  ExprAdd         → split into two sub-trees (linearity)
  ExprLeaf        → pw(f, identity, coeff)
  ExprScalar      → pw(layout, [val], 1.0)
  ExprMul         → lowerToRHSExpr(left) × lowerToRHSExpr(right) via termTimesTerm
  ExprStencil     → peel Scale/Neg wrappers → lap/grad/iso_grad(leaf, coeff)
  ExprStencilBinary → peel wrappers → grad_dot(lf, rf, coeff)
```

Stencil 节点目前只支持 plain ExprLeaf 子节点。复合 child（需要 BC injection）在
Stage 3 解决；目前抛出 `std::logic_error` 给出明确错误信息。

### 循环包含的修复

`EvalPlan.h → Equation.h → EvalPlan.h`（通过 `FieldOps.inl`）会触发 incomplete type 错误。
解决方案：`Equation.h` 仅 forward-declare `EvalPlan`，增加显式析构声明，在 `Equation.cu`
中 `= default` 定义析构（此时 `EvalPlan` 已完整）。

### 向后兼容

- 原有 `setRHS(RHSExpr)` / `setRHS(Term)` 路径保持不变。
- `setRHS(ExprTree)` 将 `rhs_expr_` 清空，`eval_plan_` 非空时走新路径。
- `hasRHS()` 两路径均返回 true。
- 所有 8 项测试通过。

---

## 测试结果

```
8/8 tests passed
  module_mesh          PASSED
  module_boundary      PASSED
  module_field         PASSED
  module_scheme        PASSED
  module_operators     PASSED
  module_face_ops      PASSED
  module_expr          PASSED
  module_evalplan      PASSED  ← 新增 (10 项 GPU 集成测试)
```
