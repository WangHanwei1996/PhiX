# Mesh

`Mesh` 是 PhiX 的网格参数容器，描述结构化正交网格。它不持有任何坐标数组，仅存储 7 个标量，因此可以放心地按值传递或复制到 GPU 结构体中。

---

## 数据成员

| 成员 | 类型 | 说明 |
|------|------|------|
| `dim` | `int` | 空间维度：1、2 或 3 |
| `coordSys` | `CoordSys` | 坐标系：`CARTESIAN` / `CYLINDRICAL` / `SPHERICAL` |
| `n[3]` | `int[3]` | 各方向格点数 `(nx, ny, nz)`；未激活方向固定为 1 |
| `d[3]` | `double[3]` | 各方向格距 `(dx, dy, dz)` |
| `origin[3]` | `double[3]` | 各方向起始坐标 `(x0, y0, z0)` |

> `dim` 以上的轴视为"非激活"，`n[ax] == 1`，`d[ax]` 和 `origin[ax]` 不参与物理计算。

---

## 构造

### 直接构造

```cpp
Mesh m(dim, coordSys,
       nx, dx, x0,
       ny, dy, y0,
       nz, dz, z0);
```

构造时自动调用 `validate()`，参数非法则抛出 `std::invalid_argument`。

### 工厂方法（推荐）

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

> 工厂方法内部调用直接构造，参数自动填充非激活轴（`n=1, d=1, origin=0`）。

---

## 查询方法

所有查询方法均为 `inline`，可在 CPU 端直接调用（也可捕获进 lambda 传给 GPU）。

### `totalSize()`

```cpp
std::size_t totalSize() const;
```

返回 `nx * ny * nz`，即所有物理格点的总数。

### `coord(axis, i)`

```cpp
double coord(int axis, int i) const;
```

返回第 `axis` 轴索引为 `i` 的格点**中心**坐标：

$$x_i = \text{origin}[\text{axis}] + \left(i + 0.5\right) \cdot d[\text{axis}]$$

### `index(i, j, k)` / `index(i, j)` / `index(i)`

```cpp
int index(int i, int j, int k) const;   // 3D
int index(int i, int j)        const;   // 2D
int index(int i)               const;   // 1D
```

行主序（x 最快，z 最慢）的线性索引：

$$\text{idx} = i + n_x \cdot \left(j + n_y \cdot k\right)$$

> 注意：这是**不含 ghost 层**的物理格点索引，与 `Field::index()` 不同（后者带 ghost 偏移）。

---

## 验证

```cpp
void validate() const;   // 非法则抛出 std::invalid_argument
bool isValid()  const noexcept;  // 返回 true/false，不抛异常
```

检查规则：

- `dim` 必须为 1、2 或 3
- 激活轴：`n[ax] > 0`，`d[ax] > 0`
- 非激活轴（`ax >= dim`）：`n[ax] == 1`

---

## IO

### 写入文件

```cpp
mesh.write("mesh.txt");
```

输出纯文本格式：

```
# PhiX Mesh
dim      2
coordSys CARTESIAN
nx 256  dx 0.5  x0 0
ny 256  dy 0.5  y0 0
nz 1    dz 1    z0 0
```

### 读取文件

```cpp
Mesh m = Mesh::readFromFile("mesh.txt");
```

行顺序无关，`#` 开头为注释。字段名（`nx`, `dx`, `x0` 等）可在同一行任意排列。

### 打印到标准输出

```cpp
mesh.print();
```

示例输出：

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

## 坐标系枚举

```cpp
enum class CoordSys {
    CARTESIAN,    // 笛卡尔坐标 (x, y, z)
    CYLINDRICAL,  // 柱坐标     (r, θ, z)  — 保留，差分算子暂不区分
    SPHERICAL     // 球坐标     (r, θ, φ)  — 保留，差分算子暂不区分
};
```

当前 `lap` / `grad` 等算子均按笛卡尔坐标计算，`CYLINDRICAL` 和 `SPHERICAL` 为预留扩展槽位。

---

## 文件位置

| 文件 | 说明 |
|------|------|
| `include/mesh/Mesh.h` | 类声明、inline 方法 |
| `src/mesh/Mesh.cpp` | 构造、IO、验证实现（无 CUDA 依赖） |
