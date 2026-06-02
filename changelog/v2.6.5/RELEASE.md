# v2.6.5 — Stage 5: FusedTerm Stream 化 + 测试

## 摘要

为 `FusedTerm.h`（编译期表达式模板融合）补全 Stage 4 的 stream 传播，
将 `FusedRHSExpr` 内核调用改为通过 `ScratchPool::stream` 传递流，
`fuse_multi_compute` 新增显式 stream 参数；
同时提交了首次包含 FusedTerm.h 的完整版本（含 `GradDotNode`、`Pw1Node`、`Pw2Node`、
`kernel_fused_multi3`、`FusedRHSExpr`、`fuse_multi_compute`、`fuse()`）。

---

## 核心变更

### `include/equation/FusedTerm.h` *(首次提交)*

**节点类型**：
- `FieldNode{d_data}` — 读字段值 `d_data[c]`
- `LapNode{d_data}` — 二阶中心差分 Laplacian（2D/3D 自适应）
- `GradDotNode{d_f, d_g}` — `∇f · ∇g`（中心差分）
- `ScaleNode<Inner>` — `coeff * inner`
- `MulNode<Lhs,Rhs>` — element-wise 乘
- `AddNode<Lhs,Rhs>` — 加法
- `Pw1Node<Fn>` — 单场逐点函数
- `Pw2Node<Fn>` — 双场逐点函数

**操作符重载**（SFINAE 约束，不干扰 `PhiX::ScalarField`）：
`operator+`, `operator*(double)`, `operator-`

**工厂函数**：`ffield`, `flap`, `fgrad_dot`, `fmul`, `fpw`, `fpw2`

**核函数**：
- `kernel_fused_accumulate<Expr>` — 单输出，`d_rhs[c] += expr.eval(c, p)`
- `kernel_fused_multi3<E0,E1,E2>` — 三输出，每 cell 一次写

**接口**：
- `fuse_multi_compute(layout, out0,e0, out1,e1, out2,e2 [,stream])` — 三路同步核
- `FusedRHSExpr<Expr>` — 隐式转 `Term`，与 `Equation::setRHS` 无缝集成
- `fuse(expr, layout)` — 工厂，返回 `FusedRHSExpr<Expr>`

**Stream 变更（Stage 4 跟进）**：
- `FusedRHSExpr::operator Term()` 的 `gpu_launcher`：`ScratchPool&` → `pool`，
  `kernel_fused_accumulate` 加 `0, pool.stream`
- `fuse_multi_compute` 新增 `cudaStream_t stream = nullptr` 参数（默认流兼容）

---

## 使用示例

```cpp
#include "equation/FusedTerm.h"
using namespace PhiX::Fused;

// 单输出: eps2*∇²φ + (φ - φ³)
auto expr = flap(phi) * eps2
          + fpw(phi, PHIX_FN(double v){ return v - v*v*v; });
eq.setRHS(fuse(expr, phi));   // → Equation::setRHS(const Term&) 路径
eq.advanceTransient(bcs, dt);

// 三路同步 (MPF_AC_DW μ 方程)
fuse_multi_compute(phi0,
    mu0,  expr_mu0,
    mu_a, expr_mu_a,
    mu_b, expr_mu_b);
```

---

## 测试

新增 `test/moduleTest/equation/test_fused.cu`（11 个 GPU 测试）：

| # | 名称 | 说明 |
|---|------|------|
| 1 | ffield leaf | `fuse(ffield(f), f)` 结果等于 `f[c]`（误差 < 1e-14） |
| 2 | flap vs lap | 融合 Laplacian 与标准 `lap(f)` 一致（误差 < 1e-12） |
| 3 | fmul product | `fmul(ffield,ffield)` 逐点乘法正确（误差 < 1e-14） |
| 4 | fpw f^3 | `fpw(f, v^3)` 结果正确（误差 < 1e-14） |
| 5 | fpw2 f*g^2 | `fpw2(f, g, a*b^2)` 结果正确（误差 < 1e-14） |
| 6 | composite | `flap*eps2 + fpw2` 复合表达式（误差 < 1e-12） |
| 7 | fuse_multi_compute | 三路输出正确（3 个断言，误差 < 1e-14） |
| 8 | explicit stream | `fuse()` 在非默认流上结果正确（误差 < 1e-12） |
| 9 | advanceTransient | 5 步融合 flap 与标准 lap 结果一致（误差 < 1e-12） |

全量测试：**11/11 通过**。

---

## 重要修复

测试辅助函数（`fillSmooth`、`maxDiff`、CPU 参考值）使用正确的 3D 存储索引
`(i+g) + sx*((j+g) + sy*(k+g))`，与 GPU 核函数一致。
对于 2D 网格（`nz=1`）此索引比 `(i+g)+sx*(j+g)` 多 `sx*sy*g` 偏移，
2D 测试辅助函数若不修正会导致 GPU vs CPU 比较失败（写入位置不同）。

---

## 影响

- **向后兼容**：默认 `stream = nullptr`，现有代码无需修改
- **性能**：复合 RHS（如 MPF_AC_DW μ 方程 10+ 项）现可在一个核函数内完成，
  避免多个核函数的全局内存中间写
- **`fuse_multi_compute`**：三方程共用 L1/L2 缓存中的相同输入字段，减少内存带宽
