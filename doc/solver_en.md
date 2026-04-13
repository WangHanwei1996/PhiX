# Solver Module

`Solver` drives explicit time advancement of a single `Equation`. It manages boundary condition application, RHS computation, and time integration, while owning all GPU scratch fields internally.

---

## Time Integration Schemes

```cpp
enum class TimeScheme {
    EULER,   // Forward Euler (1st order)
    RK4      // Classical 4th-order Runge-Kutta
};
```

### Forward Euler

$$\varphi^{n+1} = \varphi^n + \Delta t \cdot f(\varphi^n)$$

- 1 RHS evaluation per step
- Strict stability constraint; suitable for quick prototyping

### Classical RK4

$$k_1 = f(\varphi^n)$$
$$k_2 = f\!\left(\varphi^n + \tfrac{\Delta t}{2} k_1\right)$$
$$k_3 = f\!\left(\varphi^n + \tfrac{\Delta t}{2} k_2\right)$$
$$k_4 = f\!\left(\varphi^n + \Delta t\, k_3\right)$$
$$\varphi^{n+1} = \varphi^n + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

- 4 RHS evaluations per step
- 4th-order accuracy, allows larger time steps
- Boundary conditions are applied before each intermediate stage

---

## Construction

```cpp
Solver solver(eq, {&bc_x, &bc_y}, dt, TimeScheme::RK4);
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `equation` | `Equation&` | Non-owning reference to the equation |
| `bcs` | `std::vector<BoundaryCondition*>` | Non-owning list of boundary conditions |
| `dt` | `double` | Time step size |
| `scheme` | `TimeScheme` | Integration scheme (default: `EULER`) |

All scratch fields (`rhs_`, `k1_`…`k4_`, `phi_tmp_`) have GPU memory allocated in the constructor. `equation.unknown` must already have device memory allocated via `allocDevice()`, otherwise `std::runtime_error` is thrown.

`Solver` is **non-copyable** (owns GPU memory through internal `Field` members).

---

## Public Members

| Member | Type | Description |
|--------|------|-------------|
| `dt` | `double` | Time step (can be changed between steps) |
| `scheme` | `TimeScheme` | Integration scheme |
| `step` | `int` | Current step counter (incremented by `advance()`) |
| `time` | `double` | Current simulation time (`time += dt` each step) |

---

## Single-Step Advancement

```cpp
solver.advance();       // GPU path (recommended)
solver.advanceCPU();    // CPU fallback (for debugging)
```

### `advance()` execution flow (GPU)

**Euler scheme:**
```
applyBCsGPU()          ← update ghost cells of unknown.d_curr
computeRHS(rhs_)       ← accumulate all Terms into rhs_.d_curr
kernel_axpy: phi += dt * rhs   ← GPU kernel
advanceTimeLevelGPU()  ← d_prev ← d_curr
step++; time += dt
```

**RK4 scheme:**
```
k1 ← f(phi)                              apply BCs on phi
k2 ← f(phi + dt/2 * k1)                 apply BCs on phi_tmp
k3 ← f(phi + dt/2 * k2)                 apply BCs on phi_tmp
k4 ← f(phi + dt   * k3)                 apply BCs on phi_tmp
phi += (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
advanceTimeLevelGPU()
step++; time += dt
```

> **RK4 pointer-swap trick**: Intermediate-stage RHS evaluation is performed by temporarily swapping `phi.d_curr` with `phi_tmp_.d_curr`. This reuses `Equation::computeRHS` without modifying any `Term` or rebuilding the expression.

---

## Multi-Step Run

```cpp
solver.run(nSteps, callbackEvery, callback);
```

| Argument | Description |
|----------|-------------|
| `nSteps` | Total number of steps to run |
| `callbackEvery` | Fire callback every N steps (0 = never) |
| `callback` | `std::function<void(const Solver&)>`, called after each qualifying step |

Callback trigger condition: `callback && callbackEvery > 0 && (step % callbackEvery == 0)`

```cpp
solver.run(10000, 500, [&](const Solver& s) {
    phi.downloadCurrFromDevice();
    phi.write("output/phi_" + std::to_string(s.step) + ".field");
    std::cout << "step=" << s.step << "  t=" << s.time << "\n";
});
```

---

## Scratch Field Overview

| Field | Purpose |
|-------|---------|
| `rhs_` | RHS accumulation target for Euler and each RK4 stage |
| `k1_`…`k4_` | RK4 stage slope vectors (allocated for both schemes, used by RK4 only) |
| `phi_tmp_` | Intermediate shifted field for RK4 stages ($\varphi + \alpha k_i$) |

All scratch fields share the same `Mesh` and `ghost` as `equation_.unknown`, with GPU memory allocated at construction time.

---

## Scheme Comparison

| | Euler | RK4 |
|-|-------|-----|
| Accuracy order | 1st | 4th |
| RHS evaluations per step | 1 | 4 |
| Stability limit (diffusion) | $\Delta t \leq \Delta x^2 / (2D)$ | ~4× larger |
| Extra GPU scratch fields | 1 | 5 |
| Recommended for | Rapid prototyping, coarse grids | Production runs, accurate results |

---

## Typical Usage Pattern

```cpp
// 1. Set up equation and boundary conditions
Equation eq(phi, "AllenCahn");
eq.setRHS(M * lap(phi) + M * pw(phi, [] __host__ __device__ (double p) {
    return p - p*p*p;
}));

PeriodicBC bc_x(Axis::X);
PeriodicBC bc_y(Axis::Y);

// 2. Construct Solver (phi must already have device memory allocated)
Solver solver(eq, {&bc_x, &bc_y}, 0.01, TimeScheme::RK4);

// 3. Run with a callback
solver.run(10000, 500, [&](const Solver& s) {
    phi.downloadCurrFromDevice();
    phi.write("output/phi_" + std::to_string(s.step) + ".field");
});

// Or step manually
for (int i = 0; i < 10000; ++i) {
    solver.advance();
    if (solver.step % 500 == 0) {
        phi.downloadCurrFromDevice();
        phi.write(...);
    }
}
```

---

## File Locations

| File | Purpose |
|------|---------|
| `include/solver/Solver.h` | Class declaration and `TimeScheme` enum |
| `src/solver/Solver.cu` | Implementation: GPU kernels, Euler/RK4 paths, `run()` (requires nvcc) |
