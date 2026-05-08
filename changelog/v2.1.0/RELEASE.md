# PhiX v2.1.0 发布说明

**发布日期**：2026-05-08  
**标签**：`v2.1.0`  
**作者**：Wang Hanwei <wanghanweibnds2015@gmail.com>

---

## 概述

v2.1.0 对框架做了一次**架构拆解**：将原本集中在 `Solver` 类中的单步推进逻辑（BC 应用、稳态赋值、时间积分）下沉到 `Equation` 类本身，使每个方程对象可以独立完成一次时间步推进，无需通过 `Solver` 聚合后再驱动。

同时 DSL 层新增了 **9 点各向同性梯度**（`iso_grad`）、**梯度点积**（`grad_dot`）、**命名 Hadamard 乘积**（`mul`）、**向量点积**（`dot`）以及**对复合 Term 的逐点操作**（`pw(Term, ...)`），进一步扩展了无辅助场写法的表达能力。

新增了 `dendrite_growth`、`glass_formation_4_phases`、`GFA/verification0` 三个应用求解器，以及 `applications/tools/test` 测试工具。

---

## 架构变动

### Solver 功能向 Equation 下沉

原本需要通过 `Solver`（或 `MultiStepSolver`）驱动的单步推进逻辑，现在由 `Equation` 自身提供两个方法：

```cpp
// 稳态赋值：unknown = RHS  （不推进时间）
void Equation::advanceSteady(const std::vector<BoundaryCondition*>& bcs,
                              ScalarField* sourceField = nullptr);

// 前向 Euler 时间积分：unknown += dt * RHS
void Equation::advanceTransient(const std::vector<BoundaryCondition*>& bcs,
                                 double dt,
                                 ScalarField* sourceField = nullptr);
```

两个方法均会在计算 RHS 前先对 `sourceField`（默认为 `unknown` 自身）应用边界条件。  
`advanceTransient` 完成后自动调用 `advanceTimeLevelGPU()` 并递增 `step` / `time`。

`Equation` 新增公有成员：

```cpp
int    step = 0;   // 由 advanceTransient 递增
double time = 0.0; // 由 advanceTransient 递增
```

**典型用法（替代 Solver）：**

```cpp
// 旧写法
Solver solver({ {&c, bcs, &eqMu, STEADY}, {&mu, bcs, &eqC, TRANSIENT} }, dt);
solver.advance();

// 新写法
eqMu.advanceSteady(bcs, &c);
eqC.advanceTransient(bcs, dt, &mu);
```

`Solver` 类**未被移除**，多步 Solver（`{SolverStep...}` 构造）仍可使用。单方程场景推荐使用新 API。

---

## 新增 DSL 功能

### 1. `iso_grad`：9 点各向同性梯度（Patra-Karttunen，2D）

```cpp
Term iso_grad(const ScalarField& f, int axis, double coeff = 1.0);
Term iso_grad(const Term&    t, int axis,
              const std::vector<BoundaryCondition*>& bcs, double coeff = 1.0);
Term iso_grad(const RHSExpr& e, int axis,
              const std::vector<BoundaryCondition*>& bcs, double coeff = 1.0);
```

采用 Patra-Karttunen 9 点各向同性格式：

$$
\frac{\partial f}{\partial x}\bigg|_{i,j} \approx
\frac{4(f_{i+1,j}-f_{i-1,j})+(f_{i+1,j+1}-f_{i-1,j+1})+(f_{i+1,j-1}-f_{i-1,j-1})}{12\Delta x}
$$

与 `div(iso_grad(...))` 组合得到 9 点各向同性 Laplacian，可显著抑制枝晶生长等问题中的网格各向异性。非 2D 网格或 `axis >= 2` 时自动退化为标准 3 点中心差分。

### 2. `grad_dot`：梯度点积

```cpp
Term grad_dot(const ScalarField& f, const ScalarField& g, double coeff = 1.0);
```

逐点计算 $\nabla f \cdot \nabla g = \sum_a \frac{\partial f}{\partial x_a} \frac{\partial g}{\partial x_a}$，使用 2 阶中心差分，支持 1D/2D/3D。典型用途：

```cpp
// GFA 类模型中的 |∇φ_i · ∇φ_j|
mul(phi_j, grad_dot(phi_i, phi_j), -eps_ij * eps_ij)
```

### 3. `mul`：命名 Hadamard 乘积

```cpp
Term mul(const ScalarField& f1, const ScalarField& f2, double coeff = 1.0);
Term mul(const Term&        t,  const ScalarField& f,  double coeff = 1.0);
Term mul(const Term&        t1, const Term&        t2, double coeff = 1.0);
Term mul(const RHSExpr&     e,  const ScalarField& f,  double coeff = 1.0);
Term mul(const Term&        t,  const RHSExpr&     e,  double coeff = 1.0);
// ...（共 9 个重载，覆盖 ScalarField / Term / RHSExpr 的全部组合）
```

函数形式与 `operator*` 等价，但可携带 `coeff` 参数，方便日后扩展其他乘积类型（dot product、矩阵乘法等）而不产生运算符歧义。

### 4. `dot`：向量点积

```cpp
RHSExpr dot(const VectorField&   a, const VectorField&   b, double coeff = 1.0);
RHSExpr dot(const VectorRHSExpr& a, const VectorField&   b, double coeff = 1.0);
RHSExpr dot(const VectorField&   a, const VectorRHSExpr& b, double coeff = 1.0);
RHSExpr dot(const VectorRHSExpr& a, const VectorRHSExpr& b, double coeff = 1.0);
```

逐点计算 $\text{coeff} \cdot \sum_c A_c \cdot B_c$，返回标量 `RHSExpr`。

### 5. `pw(Term, ...)`：对复合 Term 的逐点操作

```cpp
template<typename F> Term pw(const Term& t,                     F func, double coeff = 1.0);
template<typename F> Term pw(const Term& t1, const Term& t2,    F func, double coeff = 1.0);
template<typename F> Term pw(const Term& t1, const Term& t2,    
                              const Term& t3, F func, double coeff = 1.0);
```

先将 Term 物化到 scratch buffer，再应用用户 functor，无需单独的辅助场。

---

## 应用层迁移

以下求解器已从 `Solver` API 迁移到 `Equation::advanceSteady / advanceTransient`：

| 文件 | 变更 |
|---|---|
| `applications/solvers/Cahn-Hillard_double-well/2D/CH_double-well.cu` | 移除 `#include "solver/Solver.h"`，改用 `eqMu.advanceSteady` + `eqC.advanceTransient` |
| `applications/solvers/Cahn-Hillard+Allen-Cahn_double-well/2D/CH+AC_double-well.cu` | 同上，三方程流水线 |
| `applications/solvers/glass_formation/2D/GFA.cu` | 同上，增加注释文档头，优化代码结构 |

---

## 新增求解器与工具

| 路径 | 说明 |
|---|---|
| `applications/solvers/dendrite_growth/2D` | 枝晶生长求解器（2D，Allen-Cahn + 温度场，`iso_grad` 各向同性 Laplacian） |
| `applications/solvers/glass_formation_4_phases/2D` | GFA 四相模型（液相 + 非晶 + 两种晶相） |
| `applications/solvers/GFA/verification0` | GFA 验证算例（与参考解对比） |
| `applications/tools/test` | DSL 单元测试工具 |

以上均已注册至顶层 `CMakeLists.txt`。

---

## 其他变更

- `src/IO/FieldIO.cpp`：`writeScalarDat` 中增加 `is2D` 标志用于 2D 输出格式判断。

---

## 已验证

- `Cahn-Hillard_double-well`、`Cahn-Hillard+Allen-Cahn_double-well` 迁移后数值结果与旧版一致
- `glass_formation/2D/GFA.cu` 重构后与 v2.0.2 bit-exact
- `tutorials/quickstart` 编译运行通过
- 完整库 `cmake --build .` 无错误
