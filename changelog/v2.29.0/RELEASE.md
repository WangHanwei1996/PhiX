# v2.29.0 — 自由能泛函评估件（PFHub 基础设施 ①）

## 摘要

PFHub 基准（BM1/2/3）的必交产物是自由能时序 F(t)。本版补上两块
零中间场、单趟归约的评估件：

- **`reduce::fieldSumPW(a[,b[,c]], fn)`**（`field/ReducePW.h`，header-only，
  需 nvcc）：对物理格逐点求 Σ fn(a[,b[,c]])，1–3 场重载，double 累加，
  ghost 永不参与——体自由能密度、任意逐点泛函一调用出积分；
- **`reduce::fieldGradSq(f)`**：Σ|∇f|²（CD2，融合 gather，无 scratch 场），
  乘 κ/2·dV 即梯度能项。

组合即 PFHub 口径的自由能：

```cpp
double F = dV * ( reduce::fieldSumPW(c, PHIX_FN (Real v) {
                      const Real d1 = v - Real(0.3), d2 = Real(0.7) - v;
                      return Real(5.0) * d1*d1 * d2*d2;   // BM1 体项
                  })
                + 0.5 * kappa * reduce::fieldGradSq(c) );
```

## 实现要点

- 与既有归约共享缓存 scratch（`reduce::detail::scratchTemp/scratchOut`
  访问器公开给 header 模板），无 per-call cudaMalloc；
- `if constexpr` 分发 1/2/3 场函子（C++17）；
- `fieldGradSq` 需 f 的 ghost 已刷新（BC 先行），1D/2D/3D 通用。

## 测试

`module_energy`（已注册 ctest）：fieldSumPW 三种重载对照 CPU（毒化
ghost，<1e-12）；fieldGradSq 对解析 ∫|∇c|²=(kx²+ky²)π² 二阶收敛
（实测 order 1.99）；BM1 风格 F 组合冒烟。全量 ctest **34/34**。

## 兼容性

纯新增。
