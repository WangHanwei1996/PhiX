# Equation（方程模块）

Equation 模块提供一套表达式 DSL，让用户用接近数学公式的写法描述时间演化方程的右端项（RHS），由框架自动处理 GPU kernel 的调度与执行。

---

## 核心概念

| 概念 | 对应类型 | 说明 |
|------|----------|------|
| 单项 | `Term` | 一个加性贡献：算子 × 系数 × 源场 |
| RHS 表达式 | `RHSExpr` | 多个 `Term` 的有序累加 |
| 方程 | `Equation` | 持有未知场引用、参数表和 `RHSExpr` |
| 启动器 | `TermLauncher` | `std::function<void(double*, const double*, double)>`，封装 GPU kernel 调用 |

---

## Term

`Term` 代表一项 $c \cdot \mathcal{L}[\phi]$，其中 $c$ 为系数，$\mathcal{L}$ 为算子（Laplacian / Gradient / Pointwise）。

### 系数运算

```cpp
Term t = lap(phi);

Term t2 = 2.0 * t;       // 系数 × 2
Term t3 = t / 4.0;       // 系数 ÷ 4
Term t4 = -t;             // 系数取负
```

`Term` 内部的 `gpu_launcher` 和 `cpu_kernel` 不随系数运算改变，运行时才将系数传入。

---

## RHSExpr

`RHSExpr` 是 `Term` 的有序列表，支持加减拼接和整体缩放：

```cpp
RHSExpr rhs = M * lap(phi) + M * pw(phi, bulkForce) - kappa * grad(phi, 0);
```

所有 `operator+` / `operator-` 都返回新的 `RHSExpr`，原对象不变。

---

## 内置算子

### `lap(f, coeff=1.0)` — 拉普拉斯算子

$$\text{coeff} \cdot \nabla^2 f = \text{coeff} \cdot \left(\frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2} + \frac{\partial^2 f}{\partial z^2}\right)$$

- 2 阶中心差分，自动按 `mesh.dim` 求和（1D 只有 x 项，2D 有 x+y，3D 有 x+y+z）
- 需要 ghost ≥ 1

```cpp
Term t = lap(phi);          // 系数默认 1.0
Term t = lap(phi, 0.01);    // 系数 0.01
```

### `grad(f, axis, coeff=1.0)` — 梯度分量

$$\text{coeff} \cdot \frac{\partial f}{\partial x_\text{axis}}$$

- 2 阶中心差分：$(f_{i+1} - f_{i-1}) / (2 \Delta x)$
- `axis`：0=x，1=y，2=z；超出 `mesh.dim` 时抛出 `std::invalid_argument`
- 需要 ghost ≥ 1

```cpp
Term t = grad(phi, 0);           // ∂φ/∂x
Term t = grad(phi, 1, -kappa);   // -κ ∂φ/∂y
```

散度可由多个 `grad` 组合：

```cpp
// ∇·u = ∂ux/∂x + ∂uy/∂y
RHSExpr div_u = grad(ux, 0) + grad(uy, 1);
```

### `pw<Functor>(f, func, coeff=1.0)` — 逐格点变换

$$\text{coeff} \cdot f(\phi_{i,j,k})$$

对每个物理格点独立地应用用户提供的函数 `func`。

**要求**：`func` 必须标注 `__host__ __device__`，以同时支持 GPU kernel 和 CPU fallback：

```cpp
// lambda 写法（推荐）
auto t = pw(phi, [] __host__ __device__ (double p) {
    return p - p * p * p;       // Allen-Cahn 驱动力 φ - φ³
});

// 仿函数写法
struct BulkForce {
    __host__ __device__ double operator()(double p) const {
        return p - p * p * p;
    }
};
auto t = pw(phi, BulkForce{});
```

> `pw` 是模板函数，定义在 `TermPW.inl` 中（由 `Term.h` 自动包含）。每种不同的 `Functor` 类型会生成独立的 GPU kernel 特化版本。

---

## Equation 类

```cpp
Equation eq(phi, "AllenCahn");
```

### 成员

| 成员 | 类型 | 说明 |
|------|------|------|
| `name` | `std::string` | 方程名称（仅用于标识） |
| `unknown` | `Field&` | 未知场引用（非持有） |
| `auxFields` | `std::vector<Field*>` | 其他 RHS 依赖场（非持有，可选） |
| `params` | `std::map<std::string,double>` | 命名物理参数 |

### 设置 RHS

```cpp
eq.setRHS(M * lap(phi) + M * pw(phi, bulkForce));
```

`setRHS` 可在模拟过程中再次调用以更换方程形式。

### 参数表

```cpp
eq.params["M"]     = 1.0;
eq.params["kappa"] = 0.5;

// 用在 setRHS 中
double M     = eq.params["M"];
double kappa = eq.params["kappa"];
```

### 计算 RHS

通常由 `Solver` 内部调用，不需要手动调用：

```cpp
// GPU 路径（推荐，由 Solver 自动调用）
eq.computeRHS(rhs_field);

// CPU fallback（调试用）
eq.computeRHSCPU(rhs_field);
```

`computeRHS` 流程：
1. `cudaMemset` 清零 `rhs.d_curr`
2. 顺序调用每个 `Term` 的 `gpu_launcher`（累加到 `rhs.d_curr`）
3. `cudaDeviceSynchronize`

---

## 完整示例

### Allen-Cahn 方程

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

### Cahn-Hilliard 方程（化学势形式）

$$\frac{\partial \varphi}{\partial t} = M \nabla^2 \mu, \quad \mu = f'(\varphi) - \kappa \nabla^2 \varphi$$

```cpp
// 先计算化学势场 mu（单独 Equation 或手动计算）
// 再用 lap(mu) 作为主方程 RHS
eq.setRHS(M * lap(mu));
```

---

## TermLauncher 机制

```cpp
using TermLauncher = std::function<void(double* rhs, const double* src, double coeff)>;
```

每个 `Term` 在**构造时**将网格信息（`nx, ny, nz, storedDims, ghost, d[ax]` 等）捕获进 lambda 闭包。`Solver` 在运行时只需调用 `term.gpu_launcher(rhs.d_curr, field.d_curr, term.coeff)`，无需传递任何上下文。

---

## 文件位置

| 文件 | 说明 |
|------|------|
| `include/equation/Term.h` | `Term`、`RHSExpr`、`TermLauncher`、内置算子声明 |
| `include/equation/TermPW.inl` | `pw<Functor>` 模板定义 + GPU kernel 模板（由 `Term.h` 包含） |
| `include/equation/Equation.h` | `Equation` 类声明 |
| `src/equation/Equation.cu` | `lap`、`grad` 实现 + `Equation` 成员函数（需 nvcc） |
