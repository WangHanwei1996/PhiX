# v2.19.0 — 调度层快赢 ②：BC 批处理（一场一 launch）

## 摘要（性能）

ghost 刷新此前是**每 BC 每场一次 kernel launch**——2D 场典型 3–4 个 BC
就是 3–4 次 launch，再乘以场数、RK4 阶段数、以及隐式算子里 **CG 的每次
迭代**。launch 延迟（WSL2 上 5–15 μs）远大于 ghost 本身的工作量。

新增 `BCBatch`：把一个场的全部内建 BC 展平成设备端描述符表，
**单 kernel（grid.z = 描述符编号）一次完成**。专项微基准（RTX 5080）：

```
3 个 BC 的 ghost 刷新     顺序 launch      批处理        提速
N=256²                    0.0191 ms       0.0079 ms     2.43×
N=1024²                   0.0210 ms       0.0077 ms     2.75×
```

完整显式 Euler 步（bench_stencil，512²，2 个周期 BC）：0.053 → ~0.044
ms/iter。受益最大的是 BC 密集路径：EquationSystem 多场（GFA_FeB 类
7 场 ×2–4 BC：每步 14–28 次 launch → 7 次）与 CG 内循环
（LaplacianOp 每 apply 2→1，BiharmonicOp 4→2，乘以每步 6–24 次迭代）。

---

## 设计

- **`BCBatch`**（`include/boundary/BCBatch.h` + 实现于 `Boundary.cu`，
  与 `makePatchParams`/既有 kernel 同 TU）：`build()` 用 `dynamic_cast`
  识别三种内建 BC（Periodic/NoFlux/Fixed）展平为 `BCDesc` 表拷到
  device；未知子类进**回退列表**，批处理 kernel 后按原样逐个
  `applyOnGPU` ——第三方 BC 永不失效，只失去批处理收益。
- **批内并发安全**：各 BC 写的 ghost 带互不相交（不同轴/侧；面 patch
  不含角格），读的全是物理格——原先的顺序 launch 本来也不依赖顺序，
  单 kernel 并发语义等价（测试逐位验证）。
- **指针交换安全**：apply 时读取场**当前**的 `d_curr`，RK4 阶段的
  d_curr 临时换绑照常工作（专项测试覆盖）。

## 接入点（库内热路径全部批化）

| 位置 | launch 变化 |
|------|------------|
| `Solver`（单方程 + 多步模式 + RK4 各阶段） | 每场 nBC → 1 |
| `EquationSystem`（Euler + RK4 各阶段） | Σ(每场 nBC) → 场数 |
| `SemiImplicitSolver` 显式段 | nBC → 1 |
| `LaplacianOp`（CG 每次迭代！） | 2 → 1 |
| `BiharmonicOp`（持久化 inner/outer 算子，批表只建一次） | 4 → 2 |

用户手写循环可直接用：`BCBatch batch; batch.build(f, {&bc1,&bc2});`
每步 `batch.applyOnGPU(f);`。逐 BC 的 `applyOnGPU` 原接口不变。

---

## 测试

`module_bc_batch`（已注册 ctest）：

1. 混合 BC 组（周期 X + NoFlux Y-low + Fixed Y-high，ghost=2）批处理
   与顺序施加**逐位相等**（diff == 0.0）；
2. 未知 BC 子类正确路由到回退列表，组合结果仍逐位一致；
3. RK4 指针交换模式下批处理跟随换绑后的缓冲。

全量 ctest **28/28**；FLOAT 构建 + smoke 通过。

## 兼容性

纯新增 + 库内求解器内部替换；逐 BC 接口与行为不变。BiharmonicOp 构造
时即建 inner/outer 持久算子（此前每次 apply 临时构造，顺带修掉了这一
低效）。
