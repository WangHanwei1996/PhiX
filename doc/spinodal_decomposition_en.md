# Spinodal Decomposition: 2D Cahn–Hilliard Equation

This tutorial solves the **Cahn–Hilliard (CH)** equation on a 2D periodic domain to simulate spinodal decomposition — the spontaneous phase separation of a binary mixture quenched into the unstable region of its phase diagram.

---

## Problem Statement

Find the concentration field $c(\mathbf{x}, t)$ satisfying:

$$
\frac{\partial c}{\partial t} = M \nabla^2 \mu
$$

$$
\mu = 2\rho\,(c - c_a)(c - c_b)(2c - c_a - c_b) - \kappa\,\nabla^2 c
$$



where $\mu$ is the chemical potential, $f'(c) = 2\rho(c-c_a)(c-c_b)(2c-c_a-c_b)$ is the derivative of the double-well free energy density, and $-\kappa\nabla^2 c$ is the gradient energy penalty.

**Domain**: $\Omega = [0,\,200]^2$, $\Delta x = \Delta y = 1.0$

**Boundary conditions** (periodic in both directions):

$$c(0, y, t) = c(L_x, y, t), \qquad c(x, 0, t) = c(x, L_y, t)$$

**Initial condition** (small cosine perturbation around the mean concentration $c_0$):

$$
c(\mathbf{x}, 0) = c_0 + \varepsilon \Bigl[\cos(0.105x)\cos(0.11y)

\bigl(\cos(0.13x)\cos(0.087y)\bigr)^2

\cos(0.025x - 0.15y)\cos(0.07x - 0.02y)\Bigr]
$$
**Parameters**:

| Symbol | Value | Description |
|--------|-------|-------------|
| $M$ | 5.0 | Mobility |
| $\rho$ | 5.0 | Free energy barrier height |
| $c_a$ | 0.3 | Left equilibrium composition |
| $c_b$ | 0.7 | Right equilibrium composition |
| $\kappa$ | 2.0 | Gradient energy coefficient |
| $c_0$ | 0.5 | Mean concentration |
| $\varepsilon$ | 0.01 | Perturbation amplitude |
| $\Delta t$ | 0.001 | Time step |

---

## Why a Manual Time Loop?

The standard `solver.run()` interface is designed for a single PDE of the form $\partial\phi/\partial t = \text{RHS}(\phi)$.  
The Cahn–Hilliard system involves **two coupled steps per time level**:

1. Compute $\mu^n$ from $c^n$ (auxiliary calculation — $\mu$ is *not* time-integrated).
2. Advance $c^{n+1} = c^n + \Delta t\,M\nabla^2\mu^n$.

Because step 1 is not a time-integration but a field evaluation, we set up two `Equation` objects and drive the loop manually:

```
for each step:
    Apply BCs to c          ← fill ghost cells for ∇²c stencil
    eq_1.computeRHS(mu)     ← evaluate μ = f'(c) − κ∇²c into mu
    Apply BCs to mu         ← fill ghost cells for ∇²μ stencil
    solver.advance()        ← c ← c + dt · M∇²μ
```

---

## Numerical Stability

The explicit Euler scheme applied to the biharmonic operator $\nabla^4$ in the CH equation has the stability constraint:

$$\Delta t \leq \frac{\Delta x^4}{8\,M\,\kappa}$$

For the parameters above ($\Delta x = 1$, $M = 5$, $\kappa = 2$):

$$\Delta t_\text{max} = \frac{1}{8 \times 5 \times 2} = 0.0125$$

The chosen $\Delta t = 0.001$ satisfies this bound with a comfortable margin.

---

## Code Walkthrough

### 1. Mesh

```cpp
Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                200, 1.0, 0.0,   // nx, dx, x0
                                200, 1.0, 0.0);  // ny, dy, y0
```

A $200 \times 200$ uniform Cartesian grid with unit spacing.

### 2. Fields

Two fields are required: the concentration $c$ (time-integrated) and the chemical potential $\mu$ (auxiliary, overwritten every step).

```cpp
Field c(mesh, "c", /*ghost=*/1);
Field mu(mesh, "mu", /*ghost=*/1);
```

**Initialization of $c$** (cosine perturbation using `mesh.coord`):

```cpp
for (int j = 0; j < mesh.n[1]; ++j)
for (int i = 0; i < mesh.n[0]; ++i) {
    double x = mesh.coord(0, i);
    double y = mesh.coord(1, j);
    double val = c0 + eps * (
          std::cos(0.105 * x) * std::cos(0.11 * y)
        + std::pow(std::cos(0.13 * x) * std::cos(0.087 * y), 2)
        + std::cos(0.025 * x - 0.15 * y) * std::cos(0.07 * x - 0.02 * y)
    );
    c.curr[c.index(i, j)] = val;
}
c.allocDevice();
c.uploadAllToDevice();

mu.fill(0.0);
mu.allocDevice();
mu.uploadAllToDevice();
```

Both fields must be allocated **and uploaded** to the GPU before the loop starts.

### 3. Boundary Conditions

```cpp
PeriodicBC bc_x(Axis::X);
PeriodicBC bc_y(Axis::Y);
```

Periodic in both directions. The same two BC objects are applied to **both** $c$ and $\mu$ inside the loop.

### 4. Equations

**`eq_1`** evaluates the chemical potential $\mu = f'(c) - \kappa\nabla^2 c$.  
Its *target* field is `mu`; it reads from `c` via the built-in operators.

```cpp
Equation eq_1(mu, "CH_1");
eq_1.params["rho"]   = 5.0;
eq_1.params["ca"]    = 0.3;
eq_1.params["cb"]    = 0.7;
eq_1.params["kappa"] = 2.0;

const double rho   = eq_1.params["rho"];
const double ca    = eq_1.params["ca"];
const double cb    = eq_1.params["cb"];
const double kappa = eq_1.params["kappa"];

eq_1.setRHS(
    pw(c, [rho, ca, cb] __host__ __device__ (double c_val) {
        return 2.0 * rho * (c_val - ca) * (c_val - cb) * (2.0 * c_val - ca - cb);
    })
    - kappa * lap(c)
);
```

> **Note**: `pw()` applies a point-wise scalar function to the field; the lambda **must** be marked `__host__ __device__` to run on the GPU. The `lap()` operator computes $\nabla^2 c$ using 2nd-order central finite differences and is applied at the field level — it cannot be used inside a `pw` lambda.

**`eq_2`** describes the time evolution $\partial c/\partial t = M\nabla^2\mu$.  
Its target field is `c`, and it reads from `mu`.

```cpp
Equation eq_2(c, "CH_2");
eq_2.params["M"] = 5.0;
const double M = eq_2.params["M"];

eq_2.setRHS(M * lap(mu));
```

### 5. Solver and Time Loop

Only `eq_2` is passed to `Solver` because only $c$ is time-integrated.

```cpp
Solver solver(eq_2, {&bc_x, &bc_y}, dt, TimeScheme::EULER);
```

Output is scheduled at logarithmically spaced physical times (0.1, 1, 10, 100, 1000, $10^4$, $10^5$ s):

```cpp
const std::vector<double> out_times = {0.1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5};
std::set<int> out_steps;
for (double t : out_times)
    out_steps.insert(static_cast<int>(std::round(t / dt)));
```

The two-step manual loop:

```cpp
for (int s = 0; s < nSteps; ++s) {
    // Step 1: ghost cells for c, evaluate μ
    bc_x.applyOnGPU(c);
    bc_y.applyOnGPU(c);
    eq_1.computeRHS(mu);

    // Step 2: ghost cells for μ, advance c
    bc_x.applyOnGPU(mu);
    bc_y.applyOnGPU(mu);
    solver.advance();

    if (out_steps.count(solver.step)) {
        c.downloadCurrFromDevice();
        c.write("output/c_" + std::to_string(solver.step) + ".field");
    }
}
```

`solver.advance()` applies the BCs registered at construction (for $c$) **before** the RHS; the explicit BCs on $c$ at the top of the loop only serve to populate ghost cells for `eq_1`.  
The `bc_x/bc_y.applyOnGPU(mu)` calls just before `solver.advance()` fill $\mu$'s ghost cells for the $\nabla^2\mu$ stencil inside `eq_2`.

---

## Directory Structure

```
develop/Spinodal Decomposition/
├── Cahn-Hillard.cu        # solver source
├── CMakeLists.txt         # build config
├── postProcess.py         # visualization script
└── output/                # .field snapshots (written at runtime)
    ├── c_0.field          # t = 0
    ├── c_100.field        # t = 0.1
    ├── c_1000.field       # t = 1
    ├── c_10000.field      # t = 10
    ├── c_100000.field     # t = 100
    ├── c_1000000.field    # t = 1000
    ├── c_10000000.field   # t = 1e4
    └── c_100000000.field  # t = 1e5
```

---

## Build and Run

```bash
# Build (from PhiX root)
cd build
touch "../develop/Spinodal Decomposition/Cahn-Hillard.cu"   # force timestamp update
make spinodal_decomposition

# Run
cd "../develop/Spinodal Decomposition"
rm -f output/*.field
./spinodal_decomposition
```

Progress is printed every 10 000 steps:

```
Starting Cahn-Hilliard simulation (100000000 steps, dt=0.001)
  step 0  t=0  written: output/c_0.field
  [progress] step=10000  t=10
  ...
  step 100000  t=100  written: output/c_100000.field
  ...
```

---

## Post-Processing

Run from the Windows ICE environment:

```bash
conda activate ICE
python "postProcess.py"
```

`postProcess.py` reads each `.field` snapshot, plots the concentration map with the `coolwarm` colormap ($c \in [0, 1]$), and saves a PNG to `output/png/`.

---

## Expected Results

| Time | Physical behaviour |
|------|--------------------|
| $t = 0$ | Mean field $c = 0.5$, barely visible cosine modulation |
| $t \sim 1$ | Amplification of unstable modes; $c$ begins to deviate from 0.5 |
| $t \sim 10$–$100$ | Phase separation clearly visible; interconnected A-rich / B-rich domains |
| $t \sim 10^3$–$10^5$ | Coarsening: small domains merge; characteristic length $\ell \sim t^{1/3}$ |

The late-stage $t^{1/3}$ coarsening law is a classic signature of Ostwald ripening in conserved-order-parameter dynamics.
