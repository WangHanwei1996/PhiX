# PhiX Quickstart

PhiX is an explicit finite-difference GPU phase-field computation suite. Its core is built from five layered classes:

```
Mesh  →  Field  →  BoundaryCondition  →  Equation  →  Solver
```

---

## 1. Mesh

`Mesh` is a lightweight parameter container describing the dimensions and spacing of a structured orthogonal grid. It holds no large arrays.

```cpp
#include "mesh/Mesh.h"
using namespace PhiX;

// 2D uniform Cartesian grid, 256×256, spacing 0.5, origin (0, 0)
Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                256, 0.5, 0.0,   // nx, dx, x0
                                256, 0.5, 0.0);  // ny, dy, y0

mesh.print();                       // print summary to stdout
mesh.write("case/constant/mesh");   // persist to file
```

**Supported dimensions**: `makeUniform1D` / `makeUniform2D` / `makeUniform3D`  
**Supported coordinate systems**: `CARTESIAN`, `CYLINDRICAL`, `SPHERICAL`

---

## 2. Field

`Field` wraps a ghost-cell halo (one layer per side by default) around the mesh, stores two time levels (`curr` / `prev`) in double precision, and maintains independent CPU and GPU buffers.

```cpp
#include "field/Field.h"

// Create a field with 1 ghost layer per side
Field phi(mesh, "phi", /*ghost=*/1);

// CPU initialisation — zero everything
phi.fill(0.0);

// Manual initialisation using physical coordinates
for (int j = 0; j < mesh.n[1]; ++j)
for (int i = 0; i < mesh.n[0]; ++i) {
    double x = mesh.coord(0, i);
    double y = mesh.coord(1, j);
    phi.curr[phi.index(i, j)] = std::exp(-(x*x + y*y));
}

// Allocate GPU memory and upload
phi.allocDevice();
phi.uploadAllToDevice();

// File IO
phi.write("case/0/phi.field");
Field phi2 = Field::readFromFile(mesh, "case/0/phi.field");
```

**Index convention**: physical cells `i ∈ [0, nx)`, ghost cells `i ∈ [-ghost, 0)` and `[nx, nx+ghost)`, all accessed through `phi.index(i, j, k)`.

---

## 3. Boundary Conditions

Each boundary condition object acts on a specified axis and side. Different conditions can be mixed freely across directions.

```cpp
#include "boundary/PeriodicBC.h"
#include "boundary/NoFluxBC.h"
#include "boundary/FixedBC.h"

PeriodicBC bc_x(Axis::X);                     // periodic in x
NoFluxBC   bc_y(Axis::Y, Side::BOTH);         // zero-flux (Neumann) in y
FixedBC    bc_z(Axis::Z, Side::LOW, 0.0);     // Dirichlet φ = 0 on low-z side
```

| Type | Mathematical condition | Ghost cell operation |
|------|------------------------|----------------------|
| `PeriodicBC` | $\phi[-g] = \phi[N-g]$,  $\phi[N+g-1] = \phi[g-1]$ | wrap-around copy |
| `NoFluxBC` | $\partial\phi/\partial n = 0$ | ghost ← nearest physical cell |
| `FixedBC` | $\phi\big\|_\text{boundary} = \text{value}$ | ghost ← constant |

---

## 4. Equation

`Equation` describes $\partial\phi/\partial t = \text{RHS}$. The RHS is composed from built-in operators using natural arithmetic notation.

```cpp
#include "equation/Equation.h"

Equation eq(phi, "AllenCahn");

// Physical parameters
eq.params["M"]     = 1.0;   // mobility
eq.params["kappa"] = 0.5;   // interfacial energy coefficient

double M     = eq.params["M"];
double kappa = eq.params["kappa"];

// dφ/dt = M·∇²φ + M·(φ - φ³)
//
// lap(phi)            — ∇²φ, 2nd-order central FD, sums over active axes
// pw(phi, functor)    — pointwise function; functor MUST be __host__ __device__
//                       to support both the GPU kernel and the CPU fallback
eq.setRHS(
    M * lap(phi)
  + M * pw(phi, [] __host__ __device__ (double p) { return p - p*p*p; })
);
```

**Built-in operators**

| Function | Mathematical meaning | Notes |
|----------|----------------------|-------|
| `lap(f, coeff=1)` | $\text{coeff}\cdot\nabla^2 f$ | 2nd-order central FD, active axes only |
| `grad(f, axis, coeff=1)` | $\text{coeff}\cdot\partial f/\partial x_\text{axis}$ | 2nd-order central FD |
| `pw(f, func, coeff=1)` | $\text{coeff}\cdot g(f)$ | template; `func` must support GPU |

**Operator overloading**

```cpp
// These forms are all valid and composable
eq.setRHS(2.0 * lap(phi) - 0.5 * grad(phi, 0) + pw(phi, myFunc));
eq.setRHS(lap(phi) + kappa * lap(T));   // multi-field RHS
```

---

## 5. Solver

`Solver` combines an `Equation` with boundary conditions and drives time advancement.

```cpp
#include "solver/Solver.h"

// Construct: equation, BC list, time step, integration scheme
Solver solver(eq,
              {&bc_x, &bc_y},
              /*dt=*/0.01,
              TimeScheme::RK4);   // or TimeScheme::EULER

// Single step
solver.advance();

// Run a batch of steps with a callback every 100 steps
solver.run(5000, 100, [&](const Solver& s) {
    phi.downloadCurrFromDevice();
    phi.write("output/phi_" + std::to_string(s.step) + ".field");
    phi.print();   // print min/max/mean
});
```

**Time integration schemes**

| Scheme | Accuracy | RHS calls per step | Recommended for |
|--------|----------|--------------------|-----------------|
| `EULER` | $O(\Delta t)$ | 1 | Quick tests, weakly stiff problems |
| `RK4` | $O(\Delta t^4)$ | 4 | Production runs, Allen-Cahn / Cahn-Hilliard |

---

## Complete Example: 2D Allen-Cahn Equation

### Problem statement

Find $\varphi(\mathbf{x}, t)$ satisfying:

$$\frac{\partial \varphi}{\partial t} = M \nabla^2 \varphi + M \left(\varphi - \varphi^3\right), \quad \mathbf{x} \in \Omega = [0, 128]^2,\; t \in [0, 100]$$

**Boundary conditions** (periodic in both directions):

$$\varphi(0, y, t) = \varphi(L_x, y, t), \quad \frac{\partial \varphi}{\partial x}\bigg|_{x=0} = \frac{\partial \varphi}{\partial x}\bigg|_{x=L_x}$$

$$\varphi(x, 0, t) = \varphi(x, L_y, t), \quad \frac{\partial \varphi}{\partial y}\bigg|_{y=0} = \frac{\partial \varphi}{\partial y}\bigg|_{y=L_y}$$

**Initial condition** (small random perturbation around $\varphi = 0$):

$$\varphi(\mathbf{x}, 0) = 0.05\,\xi(\mathbf{x}), \quad \xi \sim \mathcal{U}(-1, 1)$$

**Parameters**: $M = 1$, $\Delta x = \Delta y = 0.5$, $\Delta t = 0.01$, RK4

This equation is the $L^2$ gradient flow of the Ginzburg-Landau free energy functional:

$$\mathcal{F}[\varphi] = \int_\Omega \left[\frac{1}{4}(\varphi^2 - 1)^2 + \frac{1}{2}|\nabla\varphi|^2\right] d\mathbf{x}$$

The double-well potential $\frac{1}{4}(\varphi^2-1)^2$ drives $\varphi$ toward $\pm 1$; the gradient term penalises sharp interfaces.

### Code

```cpp
#include "mesh/Mesh.h"
#include "field/Field.h"
#include "boundary/PeriodicBC.h"
#include "equation/Equation.h"
#include "solver/Solver.h"

#include <cstdlib>
#include <string>

int main() {
    using namespace PhiX;

    // Mesh: 256×256, dx = dy = 0.5  →  domain [0, 128]²
    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                    256, 0.5, 0.0,
                                    256, 0.5, 0.0);

    // Field: small random initial condition
    Field phi(mesh, "phi", 1);
    std::srand(42);
    for (int j = 0; j < mesh.n[1]; ++j)
    for (int i = 0; i < mesh.n[0]; ++i) {
        double r = (double)std::rand() / RAND_MAX * 2.0 - 1.0;
        phi.curr[phi.index(i, j)] = 0.05 * r;
    }
    phi.allocDevice();
    phi.uploadAllToDevice();

    // Boundary conditions: periodic in x and y
    PeriodicBC bc_x(Axis::X);
    PeriodicBC bc_y(Axis::Y);

    // Equation: dφ/dt = M∇²φ + M(φ - φ³)
    Equation eq(phi, "AllenCahn");
    eq.params["M"] = 1.0;
    double M = eq.params["M"];

    eq.setRHS(
        M * lap(phi)
      + M * pw(phi, [] __host__ __device__ (double p) { return p - p*p*p; })
    );

    // Solver: RK4, dt = 0.01
    Solver solver(eq, {&bc_x, &bc_y}, 0.01, TimeScheme::RK4);

    // Run 10000 steps (t = 0 → 100), write every 500 steps
    solver.run(10000, 500, [&](const Solver& s) {
        phi.downloadCurrFromDevice();
        phi.write("output/phi_" + std::to_string(s.step) + ".field");
    });

    return 0;
}
```

The ready-to-run version of this example lives in `tutorials/quickstart/main.cu`.

---

## Build (CMake)

The project uses CMake. The core library is compiled as a static library `phix`; each tutorial links against it separately.

**Configure and build** (run from the PhiX root directory):

```bash
mkdir build && cd build

# Set PHIX_CUDA_ARCH to match your GPU (default: 75 = Turing)
cmake .. -DPHIX_CUDA_ARCH=75

make -j$(nproc)
```

Common `PHIX_CUDA_ARCH` values:

| GPU family | Representative models | CUDA Arch |
|------------|----------------------|-----------|
| Turing | RTX 2080 | `75` |
| Ampere | RTX 3090, A100 | `86` / `80` |
| Ada Lovelace | RTX 4090 | `89` |
| Hopper | H100 | `90` |

**Run the quickstart tutorial:**

```bash
# The binary is placed next to main.cu (not inside build/)
cd tutorials/quickstart
./quickstart
```

Output `.field` files are written to `tutorials/quickstart/output/`.

**Adding a new tutorial:**

1. Create a new folder, e.g. `tutorials/my_case/`
2. Add `main.cu` and a minimal `CMakeLists.txt`:

```cmake
add_executable(my_case main.cu)
target_link_libraries(my_case PRIVATE phix)
set_target_properties(my_case PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
```

3. Append to the root `CMakeLists.txt`:

```cmake
add_subdirectory(tutorials/my_case)
```
