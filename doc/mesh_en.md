# Mesh

`Mesh` is PhiX's grid parameter container, describing a structured orthogonal grid. It holds only 7 scalars and no coordinate arrays, so it can be safely copied by value or embedded in a GPU struct.

---

## Data Members

| Member | Type | Description |
|--------|------|-------------|
| `dim` | `int` | Spatial dimension: 1, 2, or 3 |
| `coordSys` | `CoordSys` | Coordinate system: `CARTESIAN` / `CYLINDRICAL` / `SPHERICAL` |
| `n[3]` | `int[3]` | Cell count per direction `(nx, ny, nz)`; inactive directions are fixed to 1 |
| `d[3]` | `double[3]` | Cell spacing per direction `(dx, dy, dz)` |
| `origin[3]` | `double[3]` | Origin coordinate per direction `(x0, y0, z0)` |

> Axes at index `>= dim` are "inactive": `n[ax] == 1`, and their `d[ax]` / `origin[ax]` values play no physical role.

---

## Construction

### Direct Constructor

```cpp
Mesh m(dim, coordSys,
       nx, dx, x0,
       ny, dy, y0,
       nz, dz, z0);
```

`validate()` is called automatically at construction; invalid parameters throw `std::invalid_argument`.

### Factory Methods (recommended)

```cpp
// 1D
Mesh m = Mesh::makeUniform1D(CoordSys::CARTESIAN, 512, 0.01, 0.0);

// 2D
Mesh m = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                             256, 0.5, 0.0,   // nx, dx, x0
                             256, 0.5, 0.0);  // ny, dy, y0

// 3D
Mesh m = Mesh::makeUniform3D(CoordSys::CARTESIAN,
                             64, 1.0, 0.0,
                             64, 1.0, 0.0,
                             64, 1.0, 0.0);
```

> Factory methods delegate to the direct constructor and automatically fill inactive axes (`n=1, d=1, origin=0`).

---

## Query Methods

All query methods are `inline` and can be called on the host side (or captured into a lambda passed to a GPU kernel).

### `totalSize()`

```cpp
std::size_t totalSize() const;
```

Returns `nx * ny * nz` — the total number of physical cells.

### `coord(axis, i)`

```cpp
double coord(int axis, int i) const;
```

Returns the **cell-centre** coordinate along `axis` at index `i`:

$$x_i = \text{origin}[\text{axis}] + \left(i + 0.5\right) \cdot d[\text{axis}]$$

### `index(i, j, k)` / `index(i, j)` / `index(i)`

```cpp
int index(int i, int j, int k) const;   // 3D
int index(int i, int j)        const;   // 2D
int index(int i)               const;   // 1D
```

Row-major (x fastest, z slowest) linear index over physical cells:

$$\text{idx} = i + n_x \cdot \left(j + n_y \cdot k\right)$$

> This is the **physical-cell** index with no ghost offset. `Field::index()` is different — it includes the ghost padding.

---

## Validation

```cpp
void validate() const;       // throws std::invalid_argument on failure
bool isValid()  const noexcept;  // returns true/false without throwing
```

Rules checked:

- `dim` must be 1, 2, or 3
- Active axes (`ax < dim`): `n[ax] > 0` and `d[ax] > 0`
- Inactive axes (`ax >= dim`): `n[ax] == 1`

---

## IO

### Writing to File

```cpp
mesh.write("mesh.txt");
```

Produces a plain-text file:

```
# PhiX Mesh
dim      2
coordSys CARTESIAN
nx 256  dx 0.5  x0 0
ny 256  dy 0.5  y0 0
nz 1    dz 1    z0 0
```

### Reading from File

```cpp
Mesh m = Mesh::readFromFile("mesh.txt");
```

Line order is irrelevant. Lines starting with `#` are treated as comments. Key tokens (`nx`, `dx`, `x0`, etc.) may appear in any order on a line.

### Printing to stdout

```cpp
mesh.print();
```

Example output:

```
=== Mesh ===
  dim      : 2
  coordSys : CARTESIAN
  x: n=256  d=0.5  origin=0
  y: n=256  d=0.5  origin=0
  z: n=1  d=1  origin=0  (inactive)
  totalSize: 65536
```

---

## CoordSys Enum

```cpp
enum class CoordSys {
    CARTESIAN,    // Cartesian coordinates  (x, y, z)
    CYLINDRICAL,  // Cylindrical coordinates (r, θ, z)  — reserved
    SPHERICAL     // Spherical coordinates   (r, θ, φ)  — reserved
};
```

The current `lap` and `grad` operators assume Cartesian coordinates. `CYLINDRICAL` and `SPHERICAL` are reserved for future extension.

---

## File Locations

| File | Purpose |
|------|---------|
| `include/mesh/Mesh.h` | Class declaration and inline methods |
| `src/mesh/Mesh.cpp` | Constructor, IO, and validation (no CUDA dependency) |
