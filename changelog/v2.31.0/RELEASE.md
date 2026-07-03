# v2.31.0 — Poisson 求解（PFHub 基础设施 ③）

## 摘要

BM6 电化学需要每步解静电 Poisson 方程。两层交付：

- **`ConjugateGradient::solveOperator(L, α, σ, x, b, ...)`**：广义系统
  A = α·I − σ·L。α=1 即原半隐式 Helmholtz 形式（`solve()` 保持为其
  包装，接口不变）；**α=0、σ=1、L=D∇² 即纯 Poisson −D∇²x = b**。
  α 与 σ 一样放设备标量槽——CUDA graph 对系数变化保持可复用。
- **`PoissonSolver`**：−∇·(D∇Φ) = rhs 的开箱即用封装。纯
  Neumann/周期 BC 下算子有常数零空间：RHS 投影到零均值、解返回零
  均值（`projectNullspace = false` 供 Dirichlet 定水平的情形）。CG
  的迭代天然保持在零均值子空间（离散 Laplacian 对周期/镜像 ghost
  的行和为零），投影只需进出各一次。

```cpp
PoissonSolver poisson(mesh, 1, epsPermittivity, {&bcx, &bcy});
poisson.solve(Phi, chargeDensity);        // BM6 静电
```

## 实测

周期一致系统 **8 次迭代**恢复参考解（<1e-9）；no-flux **20 次**；
非零均值 RHS 正确投影（解零均值、算子残差 <1e-8）；解析精度
O(dx²)（1.05e-2 vs dx²=9.6e-3）。

## 测试

`module_poisson`（已注册 ctest）。全量 ctest **36/36**，FLOAT 3/3。

## 兼容性

`solve()` 原语义不变（内联转发）；设备标量槽扩到 7。
