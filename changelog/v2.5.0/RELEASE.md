# PhiX v2.5.0 发布说明

**发布日期**：2026-06-01  
**标签**：`v2.5.0`  
**作者**：Wang Hanwei <wanghanweibnds2015@gmail.com>

---

## 概述

v2.5.0 完成了算符按 scheme 统一分派的重构：

- 新增 `scheme::Iso9`（9 点各向同性 Patra-Karttunen stencil，2D）
- `scheme::CD2` 补充 `laplacian()`/`gradient()` 整算符方法
- `operators` 模块的 `lap<Scheme>`/`grad<Scheme>` kernel 改为调用
  `Scheme::laplacian()`/`Scheme::gradient()`，实现对任意 scheme 的统一分派
- 新增运行时字符串分派 `lap(f, "Iso9")` / `grad(f, axis, "Iso9")`
- `iso_grad(ScalarField,...)` 委托给 `grad<scheme::Iso9>`，删除 `Equation.cu`
  中的冗余 `kernel_iso_grad_accumulate`

现有 5 点标准 Laplacian 与 9 点各向同性 Laplacian 统一由 scheme 参数决定，
不再是两套并存的实现。

---

## 变动详情

### 1. `scheme::CD2` 补充整算符接口

**修改文件**：`include/scheme/CentralDifference.h`

新增两个 `__host__ __device__` 方法：

```cpp
// 可分离 Laplacian：∇²f = Σ d²f/dx_a²
static double laplacian(const double* s, int c,
                        int sx, int sy, int dim,
                        double inv_dx2, double inv_dy2, double inv_dz2);

// 单轴梯度：df/dx_axis（3 点中心差分）
static double gradient(const double* s, int c, int axis,
                       int sx, int sy, int dim,
                       double inv_dx, double inv_dy, double inv_dz);
```

`d1()`/`d2()` 保持不变，向后兼容。

### 2. 新增 `scheme::Iso9`

**新增文件**：`include/scheme/Isotropic.h`

实现 9 点各向同性 Patra-Karttunen stencil（2D）：

**Laplacian（仅 2D，假设 dx == dy）：**

$$
\nabla^2 f_{i,j} \approx \frac{2}{3 \Delta x^2} \left(
  \frac{1}{2}(f_{i\pm 1,j} + f_{i,j\pm 1})
  + \frac{1}{4}(f_{i\pm 1, j\pm 1})
  - 3 f_{i,j}
\right)
$$

**梯度（以 x 方向为例）：**

$$
\frac{\partial f}{\partial x}\bigg|_{i,j} \approx
\frac{4(f_{i+1,j} - f_{i-1,j}) + (f_{i+1,j+1} - f_{i-1,j+1}) + (f_{i+1,j-1} - f_{i-1,j-1})}{12\,\Delta x}
$$

对 1D / 3D 网格或 `axis >= 2` 自动退化为 `CD2`。  
stencil 宽度仍为 1（`ghostRequired() == 1`）。

### 3. `operators` 按 scheme 统一分派

**修改文件**：
- `include/operators/Laplacian.h`
- `include/operators/Gradient.h`
- `src/operators/Laplacian.cu`
- `src/operators/Gradient.cu`

变化摘要：

| 变更 | 内容 |
|---|---|
| kernel 分派 | `kernel_lap_accumulate<Scheme>` 改为调用 `Scheme::laplacian()` |
| kernel 分派 | `kernel_grad_accumulate<Scheme>` 改为调用 `Scheme::gradient()`，签名新增 `dim`/`inv_dy`/`inv_dz` |
| 显式实例化 | `lap<CD2>` / `lap<Iso9>` / `grad<CD2>` / `grad<Iso9>` |
| 字符串分派 | `lap(f, "Iso9")` / `grad(f, axis, "Iso9")` |

### 4. `iso_grad` 委托重构

**修改文件**：`src/equation/Equation.cu`

- `iso_grad(const ScalarField&, int, double)` 改为直接 `return grad<scheme::Iso9>(...)`，
  删除原内联 9 点 kernel 实现（≈ 80 行）
- `iso_grad(Term/RHSExpr, ...)` 表达式版本改为使用新增的
  `kernel_iso9_grad_accumulate`（调用 `scheme::Iso9::gradient`），
  删除 `kernel_iso_grad_accumulate`

---

## API 对比

### 调用方式

```cpp
// 旧（仍可用，等价于 CD2）
eqC.setRHS(M * lap(mu));
iso_grad(phi, 0)

// 新：编译期 scheme 选择
lap<scheme::CD2>(mu)         // 标准 5 点
lap<scheme::Iso9>(mu)        // 9 点各向同性（2D）

// 新：运行时字符串分派（适合配置文件驱动）
lap(mu, "Iso9")
grad(phi, 0, "Iso9")
```

---

## 验证结果

```
cmake --build . -j4 && ctest --output-on-failure
100% tests passed, 0 tests failed out of 5
CH_2D builds successfully
```
