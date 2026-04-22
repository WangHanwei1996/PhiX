# PhiX v2.0.2 发布说明

**发布日期**：2026-04-22  
**标签**：`v2.0.2`  
**作者**：Wang Hanwei &lt;wanghanweibnds2015@gmail.com&gt;

---

## 概述

v2.0.2 在方程 DSL 层新增了**复合 Term 算术运算**支持，并在 GFA 求解器上完成了验证与性能调优。同时新增了 `etc/bashrc` 环境脚本，方便工作目录快速导航。

---

## 新增功能

### 1. 复合 Term DSL（懒物化方案）

涉及文件：  
- `include/equation/Term.h`  
- `include/equation/FieldOps.inl`  
- `include/equation/TermPW.inl`  
- `include/equation/Equation.h`（`ScratchPool`）  
- `src/equation/Equation.cu`  
- `include/field/ScalarField.h`（`makeShell`）  
- `src/field/ScalarField.cu`

新增运算重载：

| 表达式 | 返回类型 |
|---|---|
| `Term * Term` | `Term` |
| `Term * ScalarField` | `Term` |
| `RHSExpr * ScalarField` | `Term` |
| `Term * VectorRHSExpr` | `VectorRHSExpr` |
| `lap(Term/RHSExpr, bcs)` | `Term` |
| `grad(Term/RHSExpr, axis, bcs)` | `Term` |
| `grad(Term/RHSExpr, bcs)` | `VectorRHSExpr` |
| `div(VectorRHSExpr, bcs)` | `RHSExpr` |

中间结果通过 `ScratchPool` 在 GPU 上按需分配，所有计算留在设备端，无 host↔device 往返。

### 2. GFA 求解器重写（`applications/solvers/glass_formation/2D/GFA.cu`）

利用新 DSL 将原 472 行、11 SolverSteps、7 个辅助场的实现压缩为 235 行、6 SolverSteps、1 个辅助场（mobility `D`），数值结果与原版 bit-exact（Linf = 0）。

### 3. 环境脚本（`etc/bashrc`）

新增 `whw` 命令（Workspace Hub Waypoint）：`source etc/bashrc` 后，在任意目录输入 `whw` 即可跳转到 `$PHIX_DIR`。

---

## 开发准则：求解器的计算效率设计

设计新求解器时，遵循以下准则可显著提升 GPU 执行效率：

### 准则 1：公共子表达式提取（最重要）

**凡是在同一个 RHS 中被引用超过 1 次的复合表达式，都应物化为辅助场。**

DSL 中的惰性 Term 没有存储，每次出现都会重新启动 kernel 计算。若 `D_term` 被引用 3 次，则 `pw3(c,φ,η,…)` 执行 3 次、BC 刷新多 2 次，性能倍增。将其存为 `ScalarField D` 并用一个 STEADY 步刷新，每步只算 1 次，在本项目 600×600 网格上实测提速约 **8.5×**（25.6 s/1000步 → 3.0 s/1000步）。

### 准则 2：STEADY 步顺序严格遵循数据依赖

每个 STEADY 步结束后 framework 自动刷新目标场的 halo（ghost cells）。依赖链必须顺序排列：

```
c → mu（eq_mu）
mu → D （eqD，需要 mu 的 halo）
D  → eqC（需要 D 的 halo）
```

顺序错误不会崩溃，但会读到过期 ghost 值，产生难以察觉的数值错误。

### 准则 3：只把"需要 halo"或"被多次引用"的量做成辅助场

| 情况 | 建议 |
|---|---|
| 该量会被 `lap()` 或 `grad()` 作用 | **必须**做成场（需要 ghost 单元）|
| 该量只做逐点乘/加，且只用一次 | 直接内联进 `pw`，无需辅助场 |
| 该量被引用 ≥ 2 次 | 做成辅助场，避免重复 kernel 启动 |

### 准则 4：把同一个 `pw` 能合并的逻辑合并

多个 `pw` 串联每个都要独立 launch 一个 kernel。能合并的逐点运算应写进同一个 `pw` lambda，减少 kernel 启动次数。

### 准则 5：STEADY 前置，TRANSIENT 后置

TRANSIENT 步会修改场值，后续步读到的是更新后的值。所有辅助量的 STEADY 计算统一放在本时间步的最前面，所有物理场的 TRANSIENT 更新放在最后，避免读写竞争。

### 准则 6：减少不必要的 BC 应用

每个 SolverStep 切换目标场时会自动应用 BC。`grad(expr, axis, bcs)` DSL 路径还会额外刷一次 halo。用真实场的 `grad(D, axis)`（BC 已由 STEADY 步刷好）代替 `grad(D_term, axis, bcs)`，可节省每步的 BC kernel 启动开销。

---

## 已验证

- GFA 求解器数值回归：c / phi / eta 在 100 步全部 bit-exact（Linf = 0）
- `tutorials/quickstart` 编译运行通过
- 完整库 `cmake --build .` 无警告无错误
