# v2.6.4 — Stage 4: Stream 化 + 削减 DeviceSynchronize

## 摘要

将方程求值层（Equation / Term / 各算子）的所有核函数发射改为支持非默认 CUDA stream，
同时移除 `computeRHS` 内部隐式的 `cudaDeviceSynchronize()`，
为多方程异步流水线执行奠定基础。

---

## 核心变更

### `include/equation/Term.h`
- `ScratchPool` 结构体新增 `cudaStream_t stream = nullptr` 公有字段。
  - 默认为 `nullptr`（CUDA 默认流），保持向后兼容。
  - 在 `computeRHS` 入口处由 Equation 写入 `pool.stream = stream_`，
    保证整次求值中所有核函数都使用同一个流。

### `include/equation/Equation.h`
- 新增私有字段 `cudaStream_t stream_ = nullptr`。
- 新增公有 API：
  - `void setStream(cudaStream_t s)` — 切换到指定流。
  - `cudaStream_t stream() const`   — 查询当前流。

### `include/equation/FieldOps.inl`
- `mulAccumulateGPU` 前向声明新增 `cudaStream_t stream` 参数（末位）。
- `termTimesField`、`termTimesTerm` 调用处传入 `pool.stream`。

### `include/equation/TermPW.inl`
- pw1/pw2/pw3 三路 `gpu_launcher` lambda：
  - 匿名 `ScratchPool&` 参数改名为 `pool`。
  - 核函数调用改为 `<<<blocks, threads, 0, pool.stream>>>`。

### `src/operators/Laplacian.cu`
- `gpu_launcher` lambda：匿名 `ScratchPool&` 改名为 `pool`，核函数加 `0, pool.stream`。

### `src/operators/Gradient.cu`
- 同 Laplacian.cu。

### `src/operators/FaceOps.cu`
- `divFace` 的 `gpu_launcher` lambda：匿名 `ScratchPool&` 改名为 `pool`，
  `kernel_div_face` 核函数加 `0, pool.stream`。

### `src/equation/Equation.cu`
- `mulAccumulateGPU` — 新增 `cudaStream_t stream` 参数，核函数使用该流。
- `detail::materialiseGPU` — `cudaMemset` → `cudaMemsetAsync(…, pool.stream)`。
- `makeStencilOnExprTerm` 内部 `gpu_op` 类型改为包含 `ScratchPool&` 参数，
  lapOnExpr / gradOnExpr / isoGradOnExpr / grad_dot 的核函数均加 `0, pool.stream`。
- `computeRHS`:
  - `cudaMemset` → `cudaMemsetAsync(…, stream_)`。
  - 添加 `scratch_pool_.stream = stream_` 在执行前同步流。
  - **移除** 两条路径（EvalPlan / RHSExpr）中的 `cudaDeviceSynchronize()`。
- `advanceTransient`:
  - 轴位更新核函数加 `0, stream_`。
  - `cudaDeviceSynchronize()` → `cudaStreamSynchronize(stream_)`。

---

## 行为说明

| 场景 | 行为 |
|------|------|
| `stream_ = nullptr`（默认） | 使用 CUDA 默认流；核函数仍按提交顺序串行执行；结果与旧版完全一致 |
| `setStream(s)` 显式流 | 全部核函数在 `s` 上排队；调用方负责在 CPU 读取前调用 `cudaStreamSynchronize(s)` |
| `computeRHS` 之后读取 GPU 数据 | 需调用方显式 sync（或使用 `advanceTransient` 的内置 StreamSynchronize） |

---

## 测试

新增 `test/moduleTest/equation/test_stream.cu`（5 个 GPU 测试）：

| # | 名称 | 说明 |
|---|------|------|
| 1 | default stream correctness | computeRHS 默认流结果与显式同步参考值一致（误差 < 1e-14） |
| 2 | explicit stream correctness | setStream() 非默认流结果与参考值一致（误差 < 1e-14） |
| 3 | advanceTransient N steps | 5 步扩散方程两个独立方程结果一致（误差 < 1e-14） |
| 4 | EvalPlan async path | EvalPlan 路径（无内部 DeviceSync）结果与参考值一致（误差 < 1e-11） |
| 5 | pw pool.stream | pw GPU 核函数经 pool.stream 传递后结果与 CPU 参考一致（误差 < 1e-14） |

全量测试：**10/10 通过**（新增共 10 个测试目标）。

---

## 影响

- **向后兼容**：默认 `stream_ = nullptr`，现有代码无需任何修改。
- **性能**：多方程系统调用 `setStream` 并行时，可在多个流上并发求 RHS。
- **为 Stage 5+ 铺路**：流并发是融合内核调度的前提。
