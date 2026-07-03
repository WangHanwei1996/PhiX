# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What PhiX is

PhiX is an explicit-time-stepping GPU phase-field simulation suite written in C++17 + CUDA (double precision, structured orthogonal grids). The core compiles to a static library `libphix.a` (CMake target `phix`); each solver/tutorial/test links against it as a separate executable. The central design is a layered stack and an **equation DSL** that lets you write a PDE right-hand side in near-mathematical notation while the framework generates and schedules the GPU kernels.

```
Mesh → Field (ScalarField/VectorField/FaceField) → BoundaryCondition → Equation/EquationSystem → Solver
```

## Build

Requires the CUDA toolkit (nvcc), CMake ≥ 3.16, and the header-only `nlohmann/json` on the include path (provided here via a conda env — `nlohmann/json.hpp` is pulled in by `IO/ConfigFile.h`).

```bash
mkdir build && cd build
cmake .. -DPHIX_CUDA_ARCH=75      # MUST match the target GPU; default 75 (Turing)
make -j$(nproc)
```

`PHIX_CUDA_ARCH`: `75` Turing, `80`/`86` Ampere, `89` Ada, `90` Hopper.

Two CUDA flags are mandatory and already set in [CMakeLists.txt](CMakeLists.txt): `--expt-extended-lambda` and `--expt-relaxed-constexpr` (needed for the `[] __host__ __device__` lambdas in the DSL), plus `CUDA_SEPARABLE_COMPILATION ON` for device-code linking. Any new `.cu` target must also set `CUDA_SEPARABLE_COMPILATION ON`.

File-type convention: code requiring nvcc is `.cu` / `.cuh` / `.inl`; pure host code is `.cpp` / `.h`. The library deliberately mixes both.

## Tests

CTest is enabled. Module tests live under [test/moduleTest/](test/moduleTest/) (`mesh`, `boundary`, `field`, `scheme`, `operators`, `equation`); sample/integration cases under [test/sampleTest/](test/sampleTest/).

```bash
cd build
ctest --output-on-failure          # run everything
ctest -R module_fused              # run one test by name (e.g. module_expr, module_evalplan,
                                   #   module_bc_injection, module_stream, module_vector_equation)
./test/moduleTest/equation/module_fused   # or run the test binary directly
```

Each test is registered with `add_test(NAME ... COMMAND ...)` in the per-directory `CMakeLists.txt`.

## Running solver applications

Applications are config-driven and behave differently from typical CMake projects:

- **Binaries are placed next to their source**, not in `build/` — every app/tutorial sets `RUNTIME_OUTPUT_DIRECTORY = ${CMAKE_CURRENT_SOURCE_DIR}`.
- Apps are **enabled/disabled by (un)commenting `add_subdirectory(...)` lines at the bottom of the root [CMakeLists.txt](CMakeLists.txt)**. Many solvers there are intentionally commented out.
- Each app reads a **JSONC** config (JSON + `//` comments) — default path `settings/settings.jsonc`, overridable as `argv[1]` via `IO::ConfigFile::fromArgs`. Run from the directory that contains `settings/`, e.g. `./MPF_AC_DW settings/settings.jsonc`.
- Config sections are accessed positionally, e.g. `cfg["mesh"]["nx"]`, `cfg["constants"]["L"]`. Output cadence and warm-restart (`start_from` + `IO::resolveStartStep`/`initField`) are handled by `IO::OutputWriter`.
- `applications/tools/_makePhi` is a Python scaffolding tool: run `makePhi` inside a directory of `.cu` files to generate its `CMakeLists.txt` and register it in the root. Sourcing [etc/bashrc](etc/bashrc) sets `$PHIX_DIR` and puts app binaries + `makePhi` on `PATH`.

## Architecture

### Mesh ([include/mesh/](include/mesh/))
Lightweight parameter container (dimensions, spacing, coordinate system) — holds **no** large arrays. Build with `Mesh::makeUniform1D/2D/3D`. Coordinate systems: `CARTESIAN`, `CYLINDRICAL`, `SPHERICAL`. `mesh.coord(axis, i)` gives a physical cell-centre position.

### Field ([include/field/](include/field/))
`ScalarField` is the workhorse: a double-precision array padded with a `ghost` halo on every side (`storedDims[ax] = mesh.n[ax] + 2*ghost`), holding two time levels (`curr`/`prev`) with **separate CPU and GPU buffers**. `Field` is a `using` alias for `ScalarField` (legacy code). GPU memory is **explicit and lazy**: call `allocDevice()` then `uploadAllToDevice()` before solving, and `downloadCurrFromDevice()` before writing output. `index(i,j,k)` accepts physical and ghost indices. Fields are non-copyable, movable; `makeShell()` wraps an external device pointer without owning it. Also here: `VectorField`, `FaceField` (staggered/face-centred storage for finite-volume flux schemes), `FieldLayout` (index arithmetic), `GibbsSimplex` (multi-phase simplex projection, `gibbsSimplexOnGPU({&phi0,...})`), `NoiseGenerator`.

### Boundary ([include/boundary/](include/boundary/))
`BoundaryCondition` base with `PeriodicBC`, `NoFluxBC` (Neumann), `FixedBC` (Dirichlet). Constructed per `Axis::{X,Y,Z}` and `Side::{LOW,HIGH,BOTH}`. `bc.applyOnGPU(field)` refreshes ghost cells. `BCFactory` builds BCs from config. Non-uniform BCs are typically done with a hand-written kernel (see `k_apply_xbc` in MPF_AC_DW).

### Operators & Schemes ([include/operators/](include/operators/), [include/scheme/](include/scheme/))
`Laplacian`, `Gradient`, and the finite-volume face chain in `FaceOps` (`cell ──interp/faceGrad──▶ face ──facePW──▶ face ──divFace──▶ cell`). `FacePW`/`FacePWGPU` apply arbitrary pointwise functors to `FaceField`s. `scheme/` holds compile-time stencil tags (`CentralDifference`, `Isotropic`) selected by operator factories.

### Equation DSL ([include/equation/](include/equation/)) — the core abstraction, mid-refactor
Three coexisting expression layers; **all are valid and in use**:

1. **Legacy `Term` / `RHSExpr`** ([Term.h](include/equation/Term.h), [TermPW.inl](include/equation/TermPW.inl)): `lap(f,coeff)`, `grad(f,axis,coeff)`, `pw(f, functor, coeff)` build `Term`s; `+`/`-`/scalar-multiply compose them into an `RHSExpr`. Each `Term` carries a `TermLauncher` (`std::function` wrapping a GPU kernel) that captures mesh geometry at construction time.

2. **`ExprTree` / `EvalPlan`** ([Expr.h](include/equation/Expr.h), [EvalPlan.h](include/equation/EvalPlan.h), [FieldOps.inl](include/equation/FieldOps.inl)): a pure-data expression tree (no embedded launchers) that classifies nodes as **Local** (pointwise, no halo) vs **Stencil** (needs halo), infers `ghostRequired()` for validation, and constant-folds scalars. `Equation::setRHS(ExprTree)` lowers the tree into an `EvalPlan` that fuses Local subtrees into single kernels. Factories: `expr_lap`, `expr_grad`, `expr_iso_grad`, `expr_grad_dot`, `expr_pw`. `Equation::registerBC(field, bcs)` lets the lowering pass auto-apply BCs to materialised scratch buffers.

3. **`FusedTerm`** ([FusedTerm.h](include/equation/FusedTerm.h), namespace `PhiX::Fused`): compile-time-typed expression templates (`ffield`, `fpw2`, `fmul`, `fgrad_dot`, `flap`, …) evaluated by `fuse_multi_compute(...)`, which computes **multiple output fields in one kernel launch** (heavily used in `MPF_AC_DW` to build three chemical-potential fields at once).

Use the **`PHIX_FN` macro** (`= [=] __host__ __device__`) for every functor passed to `pw`/`facePW`/`fpw*` — the lambda must compile for both the CPU fallback and the GPU kernel.

### Equation / EquationSystem / Solver
- **`Equation`** ([Equation.h](include/equation/Equation.h)) wraps a non-owning `unknown` field + optional `auxFields` + a `params` map (`string→double`). `setRHS(...)` (RHSExpr/Term/ExprTree overloads), `computeRHS(rhs)` [GPU] / `computeRHSCPU(rhs)`. Two single-step modes: `advanceSteady` (`unknown = RHS`) and `advanceTransient` (`unknown += dt*RHS`). Optional per-equation CUDA stream via `setStream`.
- **`Solver`** ([solver/Solver.h](include/solver/Solver.h)) drives one equation through `TimeScheme::EULER` (1 RHS/step) or `RK4` (4 RHS/step; reuses `computeRHS` via a pointer-swap trick). Owns scratch fields; `advance()` / `run(nSteps, callbackEvery, callback)`. Non-copyable. Multi-step `Solver` chains apply equations **sequentially** (operator splitting — later equations see updated fields).
- **`EquationSystem`** ([EquationSystem.h](include/equation/EquationSystem.h)) evolves coupled equations **simultaneously**: every RHS is computed from the *same* time level before any field updates — the right choice for fully-coupled systems (multi-phase Allen-Cahn, reaction-diffusion). `add(eq, {bcs})` then `advance()`.
- **`VectorEquation` / `VectorSolver`** are the vector-field counterparts.

### IO ([include/IO/](include/IO/))
`ConfigFile` (JSONC), `FieldIO`, `OutputWriter` (config-driven write/print cadence + restart). `FieldFormat`: `BINARY` (`.field`, default), `DAT` (ASCII `x y z value`), `VTS` (VTK StructuredGrid for ParaView). IO persists physical cells only — ghost cells are not written.

## Conventions & gotchas

- Everything is in `namespace PhiX` (`PhiX::IO`, `PhiX::Fused`, `PhiX::scheme`).
- Most stencil operators require `ghost >= 1`; `setRHS` validates the tree's `ghostRequired()` against each field's `ghost` and throws `std::invalid_argument` on a mismatch.
- The docs in [doc/](doc/) come in English (`*_en.md`) and Chinese versions, but several (quickstart, equation, solver, field) still describe the **legacy `Field` + `Term`/`pw` API**. The source has since moved to `ScalarField` + the `ExprTree`/`EvalPlan`/`FusedTerm` pipeline — treat the docs as conceptual background and verify specifics against the headers.
- Release notes live in `changelog/v<X.Y.Z>/RELEASE.md` and are written in **Chinese**; each feature bump adds a new version directory (mirror this when adding features). The codebase is currently at v2.33.0.
- `develop/` holds scratch experiments (mostly git-untracked); `applications/solvers/` holds real solvers; `tutorials/quickstart/` is the canonical example (currently disabled in the root CMake pending the solver refactor).
- Generated/output artifacts — `.field`, `.dat`, `.vts`, `output/` dirs, and app binaries — are gitignored (see [.gitignore](.gitignore)).







Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

------

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.



