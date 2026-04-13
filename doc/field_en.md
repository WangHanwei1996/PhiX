# Field

`Field` is PhiX's double-precision scalar field defined on a structured `Mesh`. Each direction is padded with `ghost` halo layers on both sides for boundary condition stencil access. CPU and GPU each hold independent storage buffers, with lazy GPU allocation and explicit synchronisation.

---

## Memory Layout

```
Physical cell indices : i ∈ [0, mesh.n[ax])
Ghost    cell indices : i ∈ [-ghost, 0)  and  [mesh.n[ax], mesh.n[ax]+ghost)

storedDims[ax] = mesh.n[ax] + 2*ghost
storedSize     = storedDims[0] × storedDims[1] × storedDims[2]
```

Row-major linear index (x fastest, z slowest):

$$\text{idx}(i,j,k) = (i+g) + s_x \cdot \bigl((j+g) + s_y \cdot (k+g)\bigr)$$

where $g$ = `ghost`, $s_x$ = `storedDims[0]`, $s_y$ = `storedDims[1]`.

---

## Data Members

| Member | Type | Description |
|--------|------|-------------|
| `name` | `std::string` | Field name (used in file IO) |
| `mesh` | `const Mesh&` | Associated mesh (reference, not owned) |
| `ghost` | `int` | Number of halo layers per side |
| `storedDims[3]` | `int[3]` | Storage dimensions including ghost layers |
| `storedSize` | `std::size_t` | Total element count including ghost layers |
| `curr` | `std::vector<double>` | CPU current time level |
| `prev` | `std::vector<double>` | CPU previous time level |
| `d_curr` | `double*` | GPU current time level (lazy, initially `nullptr`) |
| `d_prev` | `double*` | GPU previous time level (lazy, initially `nullptr`) |

---

## Construction and Lifetime

```cpp
Field phi(mesh, "phi", /*ghost=*/1);
```

- CPU buffers `curr` and `prev` are allocated and zero-initialised at construction
- GPU buffers are **not** allocated automatically; call `allocDevice()` explicitly
- `Field` is **non-copyable** (owns GPU memory) but **movable**
- `freeDevice()` is called automatically in the destructor

---

## Index Methods

```cpp
int index(int i, int j, int k) const;   // 3D
int index(int i, int j)        const;   // 2D (k=0)
int index(int i)               const;   // 1D (j=k=0)
```

Accepts both physical and ghost indices, including negative values. For example, `index(-1, j)` accesses the first ghost cell on the low-x side.

```cpp
// Setting an initial condition
for (int j = 0; j < mesh.n[1]; ++j)
for (int i = 0; i < mesh.n[0]; ++i)
    phi.curr[phi.index(i, j)] = std::sin(mesh.coord(0, i));
```

---

## Initialisation

```cpp
phi.fill(0.0);          // zero both curr and prev
phi.fillCurr(1.0);      // zero curr only
phi.fillPrev(0.0);      // zero prev only
```

---

## GPU Management

### Allocation and Deallocation

```cpp
phi.allocDevice();      // cudaMalloc d_curr and d_prev, then zero them
phi.freeDevice();       // cudaFree both; sets pointers to nullptr
phi.deviceAllocated();  // returns d_curr != nullptr
```

### CPU → GPU Upload

```cpp
phi.uploadCurrToDevice();   // curr  → d_curr
phi.uploadPrevToDevice();   // prev  → d_prev
phi.uploadAllToDevice();    // both
```

### GPU → CPU Download

```cpp
phi.downloadCurrFromDevice();   // d_curr → curr
phi.downloadPrevFromDevice();   // d_prev → prev
phi.downloadAllFromDevice();    // both
```

> `allocDevice()` must be called before any upload or download; otherwise `std::runtime_error` is thrown.

---

## Time Advancement

```cpp
phi.advanceTimeLevelCPU();   // CPU: prev ← curr  (std::copy)
phi.advanceTimeLevelGPU();   // GPU: d_prev ← d_curr  (cudaMemcpy D→D)
```

`Solver::advance()` calls the GPU path automatically at the end of each step; you do not need to call this manually.

---

## IO

### Writing to File

```cpp
phi.write("output/phi_1000.field");
```

File format: **text header + raw binary** (physical cells only; ghost cells are not persisted):

```
# PhiX Field
name    phi
nx 256  ny 256  nz 1
ghost   1
---
<nx*ny*nz doubles, row-major, x fastest>
```

### Reading from File

```cpp
Field phi = Field::readFromFile(mesh, "output/phi_1000.field", /*ghost=*/1);
```

- Physical cells are loaded into `curr`; `prev` is left unchanged
- A mismatch between the file header `nx/ny/nz` and the provided `mesh` throws `std::runtime_error`

### Reading with Python

```python
import numpy as np

def read_field(path, nx, ny, nz):
    with open(path, 'rb') as f:
        for line in f:
            if line.strip() == b'---':
                break
        data = np.frombuffer(f.read(), dtype=np.float64)
    return data.reshape((nz, ny, nx))   # z slowest, x fastest
```

### Printing a Summary

```cpp
phi.print();
```

Prints the name, stored dimensions, GPU allocation status, and min / max / mean over the physical cells of `curr`.

---

## Typical Usage Pattern

```cpp
// 1. Construct
Field phi(mesh, "phi", 1);

// 2. Set initial condition (CPU side)
for (...) phi.curr[phi.index(i, j)] = ...;

// 3. Allocate GPU memory and upload
phi.allocDevice();
phi.uploadAllToDevice();

// 4. Run the solver (Solver operates on d_curr / d_prev internally)
solver.run(nSteps, outEvery, [&](const Solver& s) {
    phi.downloadCurrFromDevice();   // download only when needed
    phi.write("output/phi_" + std::to_string(s.step) + ".field");
});
```

---

## File Locations

| File | Purpose |
|------|---------|
| `include/field/Field.h` | Class declaration and inline index methods |
| `src/field/Field.cu` | Constructor, GPU management, and IO (requires nvcc) |
