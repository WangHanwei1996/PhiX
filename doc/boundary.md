# Boundary（边界条件模块）

边界条件模块负责在每个时间步之前（或之后）更新场的虚格（ghost cells），使差分模板能正确访问边界附近的数据。所有边界条件类均继承自抽象基类 `BoundaryCondition`，通过 `applyOnGPU(Field&)` / `applyOnCPU(Field&)` 接口统一调用。

---

## 枚举类型

```cpp
enum class Axis { X = 0, Y = 1, Z = 2 };
enum class Side { LOW, HIGH, BOTH };
```

| 枚举值 | 说明 |
|--------|------|
| `Axis::X/Y/Z` | 作用轴 |
| `Side::LOW` | 仅作用于低索引侧（$i < 0$） |
| `Side::HIGH` | 仅作用于高索引侧（$i \geq n$） |
| `Side::BOTH` | 同时作用于两侧 |

---

## 抽象基类 BoundaryCondition

```cpp
class BoundaryCondition {
public:
    Axis axis;
    Side side;

    virtual void applyOnCPU(Field& f) const = 0;
    virtual void applyOnGPU(Field& f) const = 0;
};
```

- `applyOnCPU` 更新 `f.curr`（CPU 端 vector）
- `applyOnGPU` 是 HOST 函数，内部启动 `__global__` kernel，更新 `f.d_curr`（GPU 端）
- `Solver` 在每步调用 `applyOnGPU`；若使用 CPU 路径则调用 `applyOnCPU`

---

## 内置边界条件

### PeriodicBC — 周期边界

将虚格包绕到对面物理边界，适用于无限周期域。`Side` 固定为 `BOTH`（周期性必须两侧同时施加）。

**数学形式**（以 X 轴为例，ghost = $g$）：

$$f[-g,\,j,\,k] = f[n_x - g,\,j,\,k]  \qquad\text{低侧虚格}$$
$$f[n_x + g - 1,\,j,\,k] = f[g-1,\,j,\,k]  \qquad\text{高侧虚格}$$

**用法**：

```cpp
PeriodicBC bc_x(Axis::X);   // X 方向周期
PeriodicBC bc_y(Axis::Y);   // Y 方向周期
```

---

### NoFluxBC — 零通量（Neumann）边界

将虚格设为最近物理边界格点的值，使法向梯度为零：$\partial\phi/\partial n = 0$。

**数学形式**（以 X 轴 LOW 侧为例）：

$$f[-1,\,j,\,k] = f[0,\,j,\,k]$$
$$f[-2,\,j,\,k] = f[0,\,j,\,k]$$

（常数外推，所有虚格层均等于边界物理格点值）

**用法**：

```cpp
NoFluxBC bc(Axis::X);                    // 默认 BOTH
NoFluxBC bc_lo(Axis::Y, Side::LOW);      // 仅低侧
NoFluxBC bc_hi(Axis::Z, Side::HIGH);     // 仅高侧
```

---

### FixedBC — 固定值（Dirichlet）边界

将虚格设为指定常数值，强制边界处 $\phi = \text{value}$。

**数学形式**（以 X 轴 LOW 侧为例）：

$$f[-g,\,j,\,k] = \text{value}  \quad \forall g$$

（当前为常数填充；一阶模板精度足够，未来可升级为线性外推）

**用法**：

```cpp
FixedBC bc_lo(Axis::X, Side::LOW,  0.0);   // 低侧 φ = 0
FixedBC bc_hi(Axis::X, Side::HIGH, 1.0);   // 高侧 φ = 1
```

---

## 在 Solver 中使用

边界条件以裸指针列表传给 `Solver`：

```cpp
PeriodicBC bc_x(Axis::X);
PeriodicBC bc_y(Axis::Y);

Solver solver(eq, {&bc_x, &bc_y}, dt, TimeScheme::RK4);
```

`Solver::advance()` 在每步开始时调用所有 BC 的 `applyOnGPU`（GPU 路径）：

```
applyBCsGPU() → computeRHS() → 时间推进 → advanceTimeLevel
```

---

## GPU 实现原理

所有内置 BC 共用一个通用轴-泛型 GPU kernel 设计，通过 `FaceParams` 结构体抽象三个轴的差异：

| 字段 | 说明 |
|------|------|
| `axis_stride` | BC 轴方向的 flat-index 步长（X→1，Y→sx，Z→sx·sy） |
| `n_axis` | BC 轴的物理格点数 |
| `n_face0/1` | 面内两个方向的线程数（完整存储维度） |
| `face_stride0/1` | 面内方向的 flat-index 步长 |

这样一个 kernel 实现可覆盖 X/Y/Z 三个轴，减少重复代码。

---

## 自定义边界条件

继承 `BoundaryCondition` 并实现两个纯虚函数即可：

```cpp
class MyBC : public PhiX::BoundaryCondition {
public:
    MyBC(PhiX::Axis ax) : BoundaryCondition(ax, PhiX::Side::BOTH) {}

    void applyOnCPU(PhiX::Field& f) const override {
        // 修改 f.curr 的虚格
    }

    void applyOnGPU(PhiX::Field& f) const override {
        // 启动自定义 __global__ kernel，修改 f.d_curr 的虚格
    }
};
```

---

## 文件位置

| 文件 | 说明 |
|------|------|
| `include/boundary/BoundaryCondition.h` | 抽象基类 + `Axis` / `Side` 枚举 |
| `include/boundary/PeriodicBC.h` | 周期边界声明 |
| `include/boundary/NoFluxBC.h` | 零通量边界声明 |
| `include/boundary/FixedBC.h` | 固定值边界声明 |
| `src/boundary/Boundary.cu` | 全部实现 + GPU kernels（需 nvcc） |
