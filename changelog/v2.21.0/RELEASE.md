# v2.21.0 — CUDA Graphs：CG burst 图捕获重放

## 摘要（性能）

v2.20.0 之后半隐式的剩余成本是每次 CG 迭代 ~10 次 kernel launch 的
延迟（WSL2 尤甚）。本版把一个 `checkEvery` burst（默认 4 次迭代，
约 40 次 launch）**捕获成一张 CUDA graph，一次 launch 重放**。

## 实测（bench_semiimplicit，RTX 5080，double）

```
                      v2.19.0     v2.20.0     本版        累计提速
implicit diff 256²    3.82 ms     2.12 ms     1.29 ms     2.96×
CH split      256²    4.88 ms     3.08 ms     1.56 ms     3.13×
implicit diff 512²    3.99 ms     2.19 ms     1.48 ms     2.70×
CH split      512²    2.63 ms     1.72 ms     1.32 ms     1.99×
```

叠加 v2.17.0 的算法收益（dt 放大 50–100×），相同物理时间的 CH 演化
对比 v2.16.0 前的纯显式：**步长 ×50 且每步更便宜**。

---

## 设计要点

- **σ = dt 放在设备标量槽**，burst 内 kernel 经指针读取——逐步改 dt
  （含自适应 dt）**不需要重新捕获**（专项测试覆盖 σ 改变后解仍正确）；
- 图以 (x, b, L, burst 长度) 为键缓存，任一变化自动销毁重捕获；
  首个 burst 前的 warm 路径保证 CUB 临时缓冲/BC 批表/双调和中间场
  等全部分配完成（capture 期间禁止 cudaMalloc）；
- **流化地基**：`LinearOperator::apply`、`BCBatch::applyOnGPU`、
  `reduce::fieldDotAsync` 均增加 `cudaStream_t` 参数（默认 nullptr，
  既有调用不变）；CG 全部工作迁移到私有流（legacy 默认流的阻塞语义
  自动保证与外部 default-stream 工作的先后顺序）；
- 新增 `LinearOperator::streamSafe()`：含 fallback BC（未知 BC 子类）
  的算子不能捕获 → 自动退回 v2.20.0 非图路径，正确性不受影响；
  `ConjugateGradient::useGraph = false` 可手动关闭；
- 剩余 host 交互：每 solve 2 次初始同步（‖b‖、ρ₀）+ 每 burst 1 次
  检查点回读。

## 变更文件

| 文件 | 说明 |
|------|------|
| `include/solver/LinearSolver.h` / `src/solver/LinearSolver.cu` | 算子 stream 虚参、`streamSafe()`、CG 私有流 + 图捕获/缓存/失效、σ 设备槽 |
| `include/boundary/BCBatch.h` / `src/boundary/Boundary.cu` | `applyOnGPU(f, stream)` |
| `include/field/Reduce.h` / `src/field/Reduce.cu` | `fieldDotAsync(..., stream)` |
| `test/moduleTest/solver/test_linsolve.cu` | 新增第 6 节：图/非图路径逐点一致（<1e-12）、σ 改变复用图仍正确 |

## 测试

全量 ctest **29/29**（module_semiimplicit 的机器精度轨迹测试经由图
路径原样通过——数值语义零变化）；FLOAT 构建 3/3。

## 兼容性

公共接口向后兼容（新参数均带默认值）。自定义 `LinearOperator` 子类
需实现新的 `apply(x, y, stream)` 签名；未覆写 `streamSafe()` 时默认
不捕获（安全侧）。
