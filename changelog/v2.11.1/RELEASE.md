# v2.11.1 — 收敛阶验证套件（MMS）+ 修复 Iso9 Laplacian 权重错误

## 摘要

新增 `test/convergence/` 收敛阶回归套件：网格/步长加密序列上实测各空间
格式与时间积分器的收敛阶，断言与标称阶一致（ctest 常驻，防回归）。

**套件首次运行即抓到一个既有正确性 bug**：`scheme::Iso9` 的 9 点
Laplacian 权重错误，实测阶 ≈ 0（不一致格式）——算子收敛到 (2/3)·∇²
而非 ∇²。本版一并修复。

---

## 修复：Iso9 Laplacian（重要）

旧实现（`include/scheme/Isotropic.h`）：

```
(face/2 + corner/4 − 3·center) · 2/(3·dx²)
```

一致性检验（f = x²，∇²f = 2）给出 4/3 ≠ 2：算子整体偏小 1/3，
且**不随网格加密消失**（零阶不一致，实测 p ≈ 0.04）。

修正为标准 Mehrstellen / Patra-Karttunen 权重：

```
[4·(face sum) + (corner sum) − 20·center] / (6·dx²)
```

修正后实测阶 1.99（见下）。Iso9 **梯度**权重经验证本来就正确（p≈1.98）。

影响面：仓内当前**没有**求解器使用 `Iso9`/`iso_grad`（grep 确认），
此修复不改变任何在用算例的结果；此前若有外部代码用过 Iso9 Laplacian，
其扩散强度被系统性低估了 1/3，需重新标定。

---

## 新增：收敛阶套件

| 目标 | 内容 | 断言 |
|------|------|------|
| `conv_spatial` | sin(kx)cos(ky) 解析 ghost，N=32/64/128，L2 误差阶 | lap: CD2≈2, Iso9≈2, CD4≈4；grad: CD2≈2, Iso9≈2, CD4≈4（±0.2） |
| `conv_temporal` | dφ/dt=−φ 积分到 T=1，dt 减半序列 | EULER≈1（±0.15），RK4≈4（±0.30） |
| `conv_pde` | 1D 周期扩散 φ=sin(x)e^{−t} 端到端（PeriodicBC+CD2+RK4，dt=0.2dx² 隔离空间误差） | p≈2（±0.2） |

本版实测值（RTX 5080）：

```
lap CD2   p=1.995, 1.999      grad CD2   p=1.998, 1.999
lap Iso9  p=1.991, 1.998      grad Iso9  p=1.983, 1.996
lap CD4   p=3.985, 3.996      grad CD4   p=3.995, 3.999
EULER p≈1.00   RK4 p≈4.00     PDE(1D扩散) p≈2.00
```

根 `CMakeLists.txt` 注册 `add_subdirectory(test/convergence)`。
全量 ctest 20/20 通过。

---

## 兼容性

- Iso9 Laplacian 数值结果改变（这是修复）；Iso9 梯度、CD2/CD4 不受影响。
- 其余为纯测试基础设施新增。
