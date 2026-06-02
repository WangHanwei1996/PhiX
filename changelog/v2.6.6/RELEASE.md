# v2.6.6 — DSL Refactor Stage 6: VectorEquation Generalization

## Overview

Completed the DSL refactor series (Stages 1–6) by generalizing `VectorEquation` to
support the full expression evaluation layer: stream-aware execution, per-component
`ExprTree` RHS, BC auto-injection, and a unified `advanceTransient` API.

## New Features

### `VectorEquation` extended API

| Method | Description |
|--------|-------------|
| `setRHSComponent(int c, const ExprTree&)` | Set per-component RHS via ExprTree (lowered to EvalPlan) |
| `setStream(cudaStream_t)` | Propagate CUDA stream to all component `Equation` objects |
| `stream() const` | Query active stream (returns component 0's stream) |
| `registerBC(const ScalarField&, bcs)` | Propagate BC registration to all component equations |
| `advanceTransient(bcs, dt)` | Forward-Euler step for all components in sequence |

## Testing

- **13 GPU tests** in `test/moduleTest/equation/test_vector_equation.cu`:
  1. Stream propagation to component equations (5 assertions)
  2. `computeRHS` components match scalar Laplacian references (2 assertions)
  3. `registerBC` + composite ExprTree: `lap(v[0]±v[1])` matches linearity identity (2 assertions)
  4. `advanceTransient` N-step integration matches per-component scalar references (2 assertions)
  5. Explicit stream `computeRHS` matches default-stream result (2 assertions)

- **12/12 ctest targets pass** (full suite).

## Implementation Notes

- `setRHSComponent` delegates to `equations_.at(c)->setRHS(tree)`, which triggers
  EvalPlan lowering with the component equation's registered `BcMap`.
- `advanceTransient(bcs, dt)` calls each component equation's own `advanceTransient`,
  preserving per-component field references.
- `stream()` reads `equations_[0]->stream()`; consistent because `setStream` sets all.
- BC auto-injection via `registerBC` only activates for **composite** ExprStencil children;
  simple leaf children (`lap(field)`) do not require registered BCs.

## Series Summary (v2.6.1 – v2.6.6)

| Version | Stage | Description |
|---------|-------|-------------|
| v2.6.1 | 1 | ExprTree node system |
| v2.6.2 | 2 | EvalPlan lowering |
| v2.6.3 | 3 | BC auto-injection via BcMap |
| v2.6.4 | 4 | Stream化 (cudaStream_t throughout) |
| v2.6.5 | 5 | FusedTerm compile-time expression templates |
| **v2.6.6** | **6** | **VectorEquation generalization** |
