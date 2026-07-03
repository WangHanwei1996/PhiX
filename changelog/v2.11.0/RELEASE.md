# v2.11.0 — 空间格式库扩充：CD4 四阶中心差分 + 一阶迎风对流

## 摘要

- 新增 **`scheme::CD4`**：逐轴 5 点四阶中心差分（d1/d2/Laplacian/Gradient），
  要求 ghost ≥ 2。适用于 1D/2D/3D、非正方网格（可分离求和形式）。
- 新增 **迎风对流算子 `adv(u, f)`**：一阶 donor-cell 迎风的 u·∇f，
  按格点速度符号逐轴选单侧差分——中心差分做对流会色散振荡，
  迎风保单调（CFL < 1 稳定），代价是一阶精度（数值扩散 ~|u|·dx/2）。
  框架此前完全没有对流项能力。
- **`setRHS` 对 Term/RHSExpr 路径补上 ghost 校验**：此前
  `requiredGhost_` 只存不查，CD4 用在 ghost=1 的场上会静默越界读；
  现在与 ExprTree 路径一致，模板宽度超出场晕圈立即抛
  `std::invalid_argument`。

---

## 核心变更

### 新增/修改文件

| 文件 | 说明 |
|------|------|
| `include/scheme/CentralDifference.h` | 新增 `struct CD4`（ghostRequired=2, order=4） |
| `src/operators/Laplacian.cu` / `Gradient.cu` | 显式实例化 `lap<CD4>` / `grad<CD4>`；字符串调度接受 `"CD4"` |
| `include/operators/Advection.h` / `src/operators/Advection.cu` | 迎风对流算子（新源文件已注册 `phix` 库） |
| `src/equation/Equation.cu` | `setRHS(Term/RHSExpr)` 逐项校验 `field->ghost >= term.ghostRequired` |
| `test/moduleTest/operators/test_schemes_ext.cu` | 模块测试 `module_schemes_ext` |

### API

```cpp
// 四阶模板（场需 ghost >= 2；周期/NoFlux/Fixed BC kernel 本就按 ghost 宽度填层）
eq.setRHS(lap(f, "CD4", kappa) + grad(f, /*axis=*/0, "CD4", c0));
eq.setRHS(lap<scheme::CD4>(f, kappa));          // 模板形式等价

// 一阶迎风对流：adv(u,f) = u·∇f（数学项，符号由调用者掌握）
// ∂f/∂t + u·∇f = 0  →  eq.setRHS(adv(u, f, -1.0));
VectorField u(mesh, "u", /*nComponents=*/2);
eq.setRHS(adv(u, f, -1.0));
```

CD4 差分公式：

```
d1: ( f[i-2] − 8f[i-1] + 8f[i+1] − f[i+2] ) / (12·dx)
d2: ( −f[i-2] + 16f[i-1] − 30f[i] + 16f[i+1] − f[i+2] ) / (12·dx²)
```

范围说明：本次扩充落在 **Term 层**（`Equation::setRHS(Term/RHSExpr)` 全量
可用）；`ExprTree`（`expr_lap` 等）与 `FusedTerm` 层的格式选择尚未打通，
留待后续版本。

---

## 测试

`module_schemes_ext`（已注册 ctest）：

1. **CD4 精度**：sin(kx)cos(ky) 解析填 ghost（含角点），N=48 网格上
   CD4 的 Laplacian/梯度误差 < CD2 的 5%（且绝对误差达标）；
   严格收敛阶测量在 v2.11.1 收敛套件中；
2. **ghost 校验**：CD4 term 用于 ghost=1 场，`setRHS` 抛
   `std::invalid_argument`；
3. **迎风方向正确性**：2D 变符号速度场（ux 中途翻号、uy<0），
   GPU 结果与手算迎风参考逐格一致（<1e-13）；
4. **1D 周期阶跃输运**（Euler，CFL=0.5，100 步）：GPU 与 host 端同格式
   精确复算一致（<1e-12）、无过冲/下冲（单调性）、离散总和守恒
   （常速度 + 周期域迎风通量 telescoping，<1e-11）。

全量 ctest 17/17 通过（无回归）。

---

## 兼容性

- 既有 scheme（CD2/Iso9）与调用路径不变。
- `setRHS` 的 ghost 校验是新增的**防错行为变化**：此前"能跑但越界"的
  非法组合现在会抛异常——这是修复而非破坏。
