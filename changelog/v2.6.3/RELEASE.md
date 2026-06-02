# v2.6.3 — BC Auto-Injection for Composite Stencil Expressions

## Summary

Stage 3 of the DSL expression evaluation refactor.  
`Equation::registerBC()` now enables `setRHS(ExprTree)` to automatically
apply boundary conditions when a stencil operator (∇², ∂/∂x, iso-grad) is
applied to a **composite** (non-leaf) sub-expression.

## New API

```cpp
// Register boundary conditions for a field.
// Must be called before setRHS() when using composite ExprTree stencils.
equation.registerBC(field, { &bc_x, &bc_y });

// Then set the RHS with a composite stencil expression:
equation.setRHS(expr_lap(ExprTree(c) + ExprTree(d)));
// ^ Automatically applies the registered BCs during evaluation.
```

## Changes

### `include/equation/EvalPlan.h`
- Added `BcMap` typedef: `unordered_map<const ScalarField*, vector<BoundaryCondition*>>`
- `EvalStep` now stores `vector<BoundaryCondition*> bcs`
- Added overload: `EvalPlan lowerExprTree(const ExprTree&, const BcMap&)`

### `src/equation/EvalPlan.cu`
- `lowerToSteps` takes `const BcMap&` parameter
- Composite child path: calls `lookupBcs(repField, bc_map)` and dispatches to
  `lap(RHSExpr, bcs)` / `grad(RHSExpr, axis, bcs)` / `iso_grad(RHSExpr, axis, bcs)`
- Throws `std::logic_error` when `bc_map` is empty and stencil is on a composite child

### `include/equation/Equation.h`
- Added `void registerBC(const ScalarField&, vector<BoundaryCondition*>)`
- Added private `bc_map_` member (`BcMap`)
- Added explicit `~Equation()` declaration (required for `unique_ptr<EvalPlan>` with forward declaration)

### `src/equation/Equation.cu`
- `~Equation() = default` (defined here where `EvalPlan` is fully visible)
- `registerBC()` stores into `bc_map_`
- `setRHS(const ExprTree&)` calls `lowerExprTree(tree, bc_map_)`

## Tests

`test/moduleTest/equation/test_bc_injection.cu` — 5 GPU tests:

1. `lap(c+d)` via auto-BC == `lap(c)+lap(d)` (linearity)
2. `grad(c+d, 0)` via auto-BC == `grad(c,0)+grad(d,0)`
3. `iso_grad(c+d, 0)` via auto-BC
4. Empty `bc_map` + composite stencil throws `std::logic_error`
5. `lap(c+c)` via auto-BC == `2*lap(c)`

All 9 module tests pass.
