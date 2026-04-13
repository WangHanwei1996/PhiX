# Equation Module

The Equation module provides an expression DSL that lets users describe the right-hand side (RHS) of a time-evolution equation in a form close to mathematical notation. The framework automatically handles GPU kernel scheduling and execution.

---

## Core Concepts

| Concept | Type | Description |
|---------|------|-------------|
| Single term | `Term` | One additive contribution: operator × coefficient × source field |
| RHS expression | `RHSExpr` | Ordered accumulation of multiple `Term` objects |
| Equation | `Equation` | Holds a reference to the unknown field, a parameter map, and an `RHSExpr` |
| Launcher | `TermLauncher` | `std::function<void(double*, const double*, double)>` wrapping a GPU kernel call |

---

## Term

A `Term` represents a contribution $c \cdot \mathcal{L}[\phi]$, where $c$ is a scalar coefficient and $\mathcal{L}$ is an operator (Laplacian / Gradient / Pointwise).

### Coefficient Arithmetic

```cpp
Term t = lap(phi);

Term t2 = 2.0 * t;    // multiply coefficient by 2
Term t3 = t / 4.0;    // divide coefficient by 4
Term t4 = -t;          // negate coefficient
```

The `gpu_launcher` and `cpu_kernel` stored inside `Term` are unaffected by coefficient arithmetic; the coefficient is passed to the launcher at runtime.

---

## RHSExpr

`RHSExpr` is an ordered list of `Term`s that supports addition, subtraction, and scalar scaling:

```cpp
RHSExpr rhs = M * lap(phi) + M * pw(phi, bulkForce) - kappa * grad(phi, 0);
```

All `operator+` / `operator-` overloads return new `RHSExpr` objects; operands are unchanged.

---

## Built-in Operators

### `lap(f, coeff=1.0)` — Laplacian

$$\text{coeff} \cdot \nabla^2 f = \text{coeff} \cdot \left(\frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2} + \frac{\partial^2 f}{\partial z^2}\right)$$

- 2nd-order central finite differences, automatically summed over active axes according to `mesh.dim`
- Requires `ghost >= 1`

```cpp
Term t = lap(phi);           // coefficient defaults to 1.0
Term t = lap(phi, 0.01);     // coefficient 0.01
```

### `grad(f, axis, coeff=1.0)` — Gradient Component

$$\text{coeff} \cdot \frac{\partial f}{\partial x_\text{axis}}$$

- 2nd-order central FD: $(f_{i+1} - f_{i-1}) / (2 \Delta x)$
- `axis`: 0=x, 1=y, 2=z; out-of-range values throw `std::invalid_argument`
- Requires `ghost >= 1`

```cpp
Term t = grad(phi, 0);            // ∂φ/∂x
Term t = grad(phi, 1, -kappa);    // -κ ∂φ/∂y
```

Divergence can be composed from multiple `grad` terms:

```cpp
// ∇·u = ∂ux/∂x + ∂uy/∂y
RHSExpr div_u = grad(ux, 0) + grad(uy, 1);
```

### `pw<Functor>(f, func, coeff=1.0)` — Pointwise Transform

$$\text{coeff} \cdot \text{func}(\phi_{i,j,k})$$

Applies a user-supplied function `func` independently to each physical cell.

**Requirement**: `func` must be annotated `__host__ __device__` to support both the GPU kernel and the CPU fallback:

```cpp
// Lambda syntax (recommended)
auto t = pw(phi, [] __host__ __device__ (double p) {
    return p - p * p * p;       // Allen-Cahn bulk driving force φ - φ³
});

// Functor syntax
struct BulkForce {
    __host__ __device__ double operator()(double p) const {
        return p - p * p * p;
    }
};
auto t = pw(phi, BulkForce{});
```

> `pw` is a function template defined in `TermPW.inl` (automatically included by `Term.h`). Each distinct `Functor` type generates its own GPU kernel specialisation at compile time.

---

## Equation Class

```cpp
Equation eq(phi, "AllenCahn");
```

### Members

| Member | Type | Description |
|--------|------|-------------|
| `name` | `std::string` | Equation label (for identification only) |
| `Field to be solved` | `Field&` | Non-owning reference to the unknown field |
| `auxFields` | `std::vector<Field*>` | Other fields appearing on the RHS (non-owning, optional) |
| `params` | `std::map<std::string,double>` | Named physical constants |

### Setting the RHS

```cpp
eq.setRHS(M * lap(phi) + M * pw(phi, bulkForce));
```

`setRHS` can be called again mid-simulation to switch equations.

### Parameter Map

```cpp
eq.params["M"]     = 1.0;
eq.params["kappa"] = 0.5;

// Use in setRHS
double M     = eq.params["M"];
double kappa = eq.params["kappa"];
```

### Computing the RHS

Normally called internally by `Solver`; you do not need to call this manually:

```cpp
// GPU path (called automatically by Solver)
eq.computeRHS(rhs_field);

// CPU fallback (for debugging without a GPU)
eq.computeRHSCPU(rhs_field);
```

`computeRHS` steps:
1. `cudaMemset` zeros `rhs.d_curr`
2. Calls each `Term`'s `gpu_launcher` in order (results accumulate into `rhs.d_curr`)
3. `cudaDeviceSynchronize`

---

## Full Examples

### Allen-Cahn Equation

$$\frac{\partial \varphi}{\partial t} = M \nabla^2 \varphi + M(\varphi - \varphi^3)$$

```cpp
Equation eq(phi, "AllenCahn");
eq.params["M"] = 1.0;
double M = eq.params["M"];

eq.setRHS(
    M * lap(phi)
  + M * pw(phi, [] __host__ __device__ (double p) { return p - p*p*p; })
);
```

### Cahn-Hilliard Equation (chemical potential form)

$$\frac{\partial \varphi}{\partial t} = M \nabla^2 \mu, \quad \mu = f'(\varphi) - \kappa \nabla^2 \varphi$$

```cpp
// Compute the chemical potential field mu separately, then:
eq.setRHS(M * lap(mu));
```

---

## TermLauncher Mechanism

```cpp
using TermLauncher = std::function<void(double* rhs, const double* src, double coeff)>;
```

Each `Term` captures its mesh geometry (`nx`, `ny`, `nz`, `storedDims`, `ghost`, `d[ax]`, etc.) into lambda closures **at construction time**. At runtime, `Solver` only needs to call:

```cpp
term.gpu_launcher(rhs.d_curr, field.d_curr, term.coeff);
```

No additional context is required.

---

## File Locations

| File | Purpose |
|------|---------|
| `include/equation/Term.h` | `Term`, `RHSExpr`, `TermLauncher`, built-in operator declarations |
| `include/equation/TermPW.inl` | `pw<Functor>` template definition + GPU kernel template (included by `Term.h`) |
| `include/equation/Equation.h` | `Equation` class declaration |
| `src/equation/Equation.cu` | `lap`, `grad` implementations + `Equation` member functions (requires nvcc) |
