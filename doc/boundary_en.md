# Boundary Conditions Module

The boundary conditions module fills ghost (halo) cells before each time step so that finite-difference stencils near the boundary can access valid data. All boundary condition classes inherit from the abstract base `BoundaryCondition` and are applied through a uniform `applyOnGPU(Field&)` / `applyOnCPU(Field&)` interface.

---

## Enumerations

```cpp
enum class Axis { X = 0, Y = 1, Z = 2 };
enum class Side { LOW, HIGH, BOTH };
```

| Value | Description |
|-------|-------------|
| `Axis::X/Y/Z` | The axis the BC acts on |
| `Side::LOW` | Low-index side only ($i < 0$) |
| `Side::HIGH` | High-index side only ($i \geq n$) |
| `Side::BOTH` | Both sides simultaneously |

---

## Abstract Base Class — BoundaryCondition

```cpp
class BoundaryCondition {
public:
    Axis axis;
    Side side;

    virtual void applyOnCPU(Field& f) const = 0;
    virtual void applyOnGPU(Field& f) const = 0;
};
```

- `applyOnCPU` updates `f.curr` (the CPU-side `std::vector<double>`)
- `applyOnGPU` is a HOST function that launches a `__global__` kernel to update `f.d_curr` on the GPU
- `Solver` calls `applyOnGPU` at the start of every time step (GPU path), or `applyOnCPU` for the CPU fallback

---

## Built-in Boundary Conditions

### PeriodicBC — Periodic Boundary

Wraps ghost cells around to the opposite physical boundary, suitable for periodically repeating domains. `Side` is always `BOTH` (periodicity requires both sides simultaneously).

**Mathematics** (X axis, ghost layers $= g$):

$$f[-g,\,j,\,k] = f[n_x - g,\,j,\,k]  \qquad\text{(low ghost ← far physical end)}$$
$$f[n_x + g - 1,\,j,\,k] = f[g-1,\,j,\,k]  \qquad\text{(high ghost ← near physical start)}$$

**Usage**:

```cpp
PeriodicBC bc_x(Axis::X);   // periodic in X
PeriodicBC bc_y(Axis::Y);   // periodic in Y
```

---

### NoFluxBC — Zero-Flux (Neumann) Boundary

Sets ghost cells equal to the nearest physical boundary cell, enforcing a zero normal gradient: $\partial\phi/\partial n = 0$.

**Mathematics** (X axis, LOW side):

$$f[-1,\,j,\,k] = f[0,\,j,\,k]$$
$$f[-2,\,j,\,k] = f[0,\,j,\,k]$$

(Constant extrapolation — all ghost layers are set to the boundary cell value.)

**Usage**:

```cpp
NoFluxBC bc(Axis::X);                    // default: BOTH sides
NoFluxBC bc_lo(Axis::Y, Side::LOW);      // low side only
NoFluxBC bc_hi(Axis::Z, Side::HIGH);     // high side only
```

---

### FixedBC — Fixed Value (Dirichlet) Boundary

Sets all ghost cells on the specified side to a constant value, enforcing $\phi = \text{value}$ at the boundary.

**Mathematics** (X axis, LOW side):

$$f[-g,\,j,\,k] = \text{value}  \quad \forall g$$

(Currently implemented as constant fill; sufficient for first-order stencils. Can be upgraded to linear extrapolation for higher accuracy.)

**Usage**:

```cpp
FixedBC bc_lo(Axis::X, Side::LOW,  0.0);   // φ = 0 on low-x side
FixedBC bc_hi(Axis::X, Side::HIGH, 1.0);   // φ = 1 on high-x side
```

---

## Using Boundary Conditions with Solver

Pass boundary conditions as a raw-pointer initialiser list to `Solver`:

```cpp
PeriodicBC bc_x(Axis::X);
PeriodicBC bc_y(Axis::Y);

Solver solver(eq, {&bc_x, &bc_y}, dt, TimeScheme::RK4);
```

`Solver::advance()` applies all BCs at the start of every step (GPU path):

```
applyBCsGPU() → computeRHS() → time advance → advanceTimeLevel
```

---

## GPU Implementation Design

All built-in BCs share an axis-generic GPU kernel design using a `FaceParams` struct that abstracts the differences between the three axes:

| Field | Description |
|-------|-------------|
| `axis_stride` | Flat-index step along the BC axis (X→1, Y→sx, Z→sx·sy) |
| `n_axis` | Number of physical cells along the BC axis |
| `n_face0/1` | Thread counts for the two face dimensions (full stored extent) |
| `face_stride0/1` | Flat-index strides for the two face dimensions |

One kernel implementation therefore covers all three axes, eliminating code duplication.

---

## Writing a Custom Boundary Condition

Subclass `BoundaryCondition` and implement the two pure virtual methods:

```cpp
class MyBC : public PhiX::BoundaryCondition {
public:
    MyBC(PhiX::Axis ax) : BoundaryCondition(ax, PhiX::Side::BOTH) {}

    void applyOnCPU(PhiX::Field& f) const override {
        // Modify f.curr ghost cells (CPU vector)
    }

    void applyOnGPU(PhiX::Field& f) const override {
        // Launch a custom __global__ kernel to modify f.d_curr ghost cells
    }
};
```

---

## File Locations

| File | Purpose |
|------|---------|
| `include/boundary/BoundaryCondition.h` | Abstract base class + `Axis` / `Side` enums |
| `include/boundary/PeriodicBC.h` | Periodic BC declaration |
| `include/boundary/NoFluxBC.h` | Zero-flux BC declaration |
| `include/boundary/FixedBC.h` | Fixed-value BC declaration |
| `src/boundary/Boundary.cu` | All implementations + GPU kernels (requires nvcc) |
