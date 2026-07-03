# v2.18.0 — 调度层快赢 ①：prev 时间层 opt-in + 去每步全设备同步

## 摘要（性能）

调度层总攻第一弹，两项改动，实测完整显式 Euler 步（bench_stencil，
RTX 5080，double）：

```
                v2.17.0 基线        本版          提速
euler step 512²   0.071 ms/iter    0.053 ms      1.34×
euler step 1024²  0.137 ms/iter    0.078 ms      1.76×
与裸 lap kernel 的开销比（1024²）：2.85× → 1.66×
FLOAT 1024²：0.109 ms → 0.072 ms（1.51×）
```

**附带收益：每个场的设备显存占用减半**（d_prev 不再默认分配）——
16 GB 卡上可解问题规模翻倍。

---

## 改动一：prev 时间层改为 opt-in（`ScalarField::trackPrev`）

**动机**：全仓审计发现 `prev`/`d_prev` 没有任何库内计算路径读取，唯一
消费者是已停用的 `SolidificationFeC_PhiX` 旧求解器——但每个场每步都在
付一次全场 D2D memcpy（2 次内存遍历），且 `allocDevice` 无条件分配双倍
显存。

**新行为（默认 `trackPrev = false`）**：

- `allocDevice()` 只分配 `d_curr`（显存减半）；
- `advanceTimeLevelGPU/CPU()` 为空操作（每步省 2 次全场遍历 × 场数，
  EquationSystem 多场情形按场数放大）；
- `uploadAll/downloadAll` 自动跳过 prev；`uploadPrev/downloadPrev`
  在未跟踪时抛异常（带修复提示）。

**需要 prev 的模型**（如用 (curr−prev)/dt 取 ∂φ/∂t）：在 `allocDevice()`
**之前**设 `field.trackPrev = true;`，即恢复 v2.17.0 及以前的完整语义
（求解器照常在步末轮转 curr→prev）。重新启用 SolidificationFeC_PhiX 时
需给 `phi_s` 加这一行。

## 改动二：去除每步 `cudaDeviceSynchronize`

所有求解器 kernel 都发射在 CUDA 默认流上，**流内天然有序**——每步一次
的全设备同步只是把 host 钉在 GPU 后面陪跑（管线排空 + WSL2 往返尤贵），
对正确性毫无贡献。移除后 host 可以提前排队后续步的 kernel，launch 延迟
被 GPU 执行时间掩盖。

- 保留条件同步：当某方程配置了**非默认流**（`setStream`）时维持原行为
  （跨流顺序是调用方契约）——`Solver`/`multiStep` 用 `syncIfStreamed`，
  `EquationSystem` 用 `maybeSync_()`，`VectorSolver` 同理；
- host 侧读取方（`download*`、`reduce::*`、自适应 dt 的归约）本就走
  阻塞式拷贝，与流自动排序，**语义不变**；
- `bench_stencil` 计时改为循环末显式 `cudaDeviceSynchronize` 再读表
  （否则只测到入队时间）。

---

## 变更文件

| 文件 | 说明 |
|------|------|
| `include/field/ScalarField.h` / `src/field/ScalarField.cu` | `trackPrev` 成员 + alloc/upload/download/轮转全链路门控 |
| `src/solver/Solver.cu` | `syncIfStreamed` 替换 3 处每步同步 |
| `src/equation/EquationSystem.cu` / `.h` | `maybeSync_()` 替换 5 处（Euler 1 + RK4 4） |
| `src/solver/VectorSolver.cu` | 2 处同理 |
| `test/benchmark/bench_stencil.cu` | 异步 advance 下的计时修正 |
| `test/moduleTest/field/test_timelevel.cu` | 模块测试 `module_timelevel` |

## 测试

`module_timelevel`：默认模式 d_prev 不分配/轮转空操作/prev 访问器抛
异常/uploadAll 跳过；opt-in 模式逐位恢复旧语义（GPU/CPU 轮转后
prev==curr）。全量 ctest **27/27**（既有 26 项全部通过——轨迹级测试
如 module_adaptive_dt 的机器精度断言原样通过，证明去同步无语义变化）；
FLOAT 构建 + smoke 通过。

## 兼容性

- **行为变化**：默认不再维护 prev 时间层。库内求解器与全部在册算例
  不受影响（无读取方）；外部代码若读 `prev`，加 `trackPrev = true`。
- 非默认流用户行为不变（条件同步保留）。
- `advance()` 返回时 GPU 可能仍在计算；随后的下载/归约会自动等待——
  仅当用 host 计时器手工测速时需自行加 `cudaDeviceSynchronize`。
