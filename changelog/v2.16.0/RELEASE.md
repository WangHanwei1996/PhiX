# v2.16.0 — 线性求解器层（matrix-free CG + Helmholtz/Biharmonic 算子）

## 摘要

补上框架评价（doc/claude/framework_evaluation.md）认定的第一块地基：
**线性代数层**。全程 GPU、matrix-free（永不装配矩阵，L 以 stencil kernel
形式作用），求解半隐式/隐式时间步所需的系统：

```
A x = b ,    A = I − σ·L        （σ = dt，解算器外置——自适应 dt 免费）
```

- **`LaplacianOp`**：L = D·∇²（CD2）——Allen-Cahn/扩散类刚性项；
- **`BiharmonicOp`**：L = −G·∇⁴（两趟 CD2，中间量 ∇²x 单独施加 BC）
  ——Cahn-Hilliard 四阶项，A = I + σ·G·∇⁴；
- **`ConjugateGradient`**：scratch 场一次分配循环复用；点积走 v2.9.0
  归约层（无论 PHIX_PRECISION 一律 **double 累加**）；非 SPD 情形
  （<p,Ap> ≤ 0）主动报错。
- `reduce::fieldDot(a, b)`：物理格内积，新增于归约工具箱。

实测收敛（RTX 5080，tol=1e-12）：2D 周期 Helmholtz **7 次迭代**、
1D no-flux（σD/dx² ≈ 82）**15 次**、1D 双调和 **9 次**。

---

## 核心变更

### 新增文件

| 文件 | 说明 |
|------|------|
| `include/solver/LinearSolver.h` | `LinearOperator` 抽象 + `LaplacianOp` / `BiharmonicOp` + `ConjugateGradient` |
| `src/solver/LinearSolver.cu` | 算子 kernel、CG 迭代（residual/form-A/update 融合 kernel）；已注册 `phix` 库 |
| `test/moduleTest/solver/test_linsolve.cu` | 模块测试 `module_linsolve`（新 `solver` 测试子目录） |

`include/field/Reduce.h` / `src/field/Reduce.cu`：新增 `fieldDot`。

### API

```cpp
PeriodicBC bcx(mesh.facePatch(Axis::X, Side::LOW));
LaplacianOp L(D, {&bcx, &bcy});          // L = D∇²，BC 在 apply 内刷新 ghost
ConjugateGradient cg(mesh, /*ghost=*/1); // scratch 一次分配

// 解 (I − dt·L) x = b；x 传入初猜、传出解
auto res = cg.solve(L, dt, x, b, /*relTol=*/1e-8, /*maxIter=*/500);
// res.iterations / res.relResidual / res.converged
// 默认不收敛抛异常；throwOnFail=false 则返回 converged=false

// Cahn-Hilliard 四阶隐式部分：A = I + dt·Mκ·∇⁴
BiharmonicOp B(M*kappa, {&bcC}, {&bcLap});   // bcsX 作用于 x，bcsLap 作用于 ∇²x
cg.solve(B, dt, c, b, ...);
```

### 设计边界（重要）

- **BC 线性性限制**：CG 要求 A 线性 → 算子只接受齐次 BC
  （`PeriodicBC`/`NoFluxBC`；`FixedBC` 仅 value=0，非零 Dirichlet 请自行
  提升到 b）。周期/无通量下 A = I − σL 为 SPD，CG 是正确的 Krylov 选择。
- `apply()` 会刷新（改写）输入场的 **ghost**，物理格不动——文档已注明。
- 预条件子暂缺（本类常系数 Helmholtz 条件数温和，实测个位数迭代；
  大 σ/dx² 需要时的下一步是 Chebyshev/几何多重网格，接口已为其留位：
  任何 `LinearOperator` 均可作为 CG 的算子）。

---

## 测试

`module_linsolve`（已注册 ctest）。方法论：**一致系统**——对任意参考场
x_ref 用同一算子构造 b = A·x_ref，从零初猜求解必须收回 x_ref（与 BC
类型、空间精度无关，专测求解器本身）：

1. `fieldDot` 对照 CPU（ghost 毒值不泄漏）；
2. Helmholtz 2D 周期（σD/dx²≈7）：7 迭代，误差 <1e-9；
3. Helmholtz 1D no-flux（σD/dx²≈82）：15 迭代，误差 <1e-9；
4. Biharmonic 1D 周期：9 迭代，误差 <1e-8（含中间量 BC 路径）;
5. 边界情形：b=0 → x=0；maxIter 耗尽 → throwOnFail=false 返回
   converged=false / 默认抛异常。

全量 ctest：DOUBLE **25/25**，FLOAT 构建（库含本模块）编译+smoke 通过。

---

## 兼容性

纯新增。v2.17.0 的 `SemiImplicitSolver` 将以本层为引擎。
