# v2.12.0 — 性能基础设施：计时器 / NVTX / 构建选项 / 基准套件

## 摘要

- 新增 `include/perf/Perf.h`：`WallTimer`（host 秒表）、`CudaEventTimer`
  （cudaEvent 设备区间计时）、`PHIX_NVTX_RANGE("label")`（RAII NVTX 区间，
  Nsight Systems 时间线标注，默认编译为空操作）。
- CMake：
  - **`CMAKE_BUILD_TYPE` 未指定时默认 Release**——此前默认构建 host 侧
    完全无优化（device 侧 nvcc 默认 -O3 不受影响）；
  - `PHIX_LINEINFO=ON`：nvcc `-lineinfo`（profiling 源码关联）；
  - `PHIX_ENABLE_NVTX=ON`：`PHIX_NVTX_RANGE` 编译为真实 NVTX 区间。
- 新增 `test/benchmark/bench_stencil`：标准性能基线（lap CD2/CD4 裸
  `computeRHS` 吞吐 + 完整 `Solver::advance()` Euler 步），输出
  ms/iter、Mcells/s、近似 GB/s；已注册 ctest（规模克制，<1s）。

---

## 首次基线（RTX 5080，double，2D，PHIX_CUDA_ARCH=75 经 PTX JIT）

```
lap CD2      N= 512    0.020 ms/iter   12877 Mcells/s   ~309 GB/s
lap CD4      N= 512    0.031 ms/iter    8405 Mcells/s   ~202 GB/s
euler step   N= 512    0.074 ms/iter    3565 Mcells/s   ~ 86 GB/s
lap CD2      N=1024    0.047 ms/iter   22173 Mcells/s   ~532 GB/s
lap CD4      N=1024    0.097 ms/iter   10849 Mcells/s   ~260 GB/s
euler step   N=1024    0.131 ms/iter    7979 Mcells/s   ~192 GB/s
```

**关键读数**：完整 Euler 步比裸 lap kernel 慢 2.8–3.7 倍——BC 逐面
launch、axpy、时间层 D2D memcpy、每步 `cudaDeviceSynchronize` 构成的
框架开销占大头，与 `doc/claude/framework_evaluation.md` 的推断一致，
为后续"去同步 + kernel 融合 + 指针交换"优化提供了量化基线。

---

## 核心变更

| 文件 | 说明 |
|------|------|
| `include/perf/Perf.h` | 计时器 + NVTX 区间（需 CUDA include 路径，从 `.cu` 引用） |
| `CMakeLists.txt` | Release 默认值、`PHIX_LINEINFO`、`PHIX_ENABLE_NVTX` 选项；注册 benchmark 子目录 |
| `test/moduleTest/perf/test_perf.cu` | 模块测试 `module_perf`（计时器量程、NVTX 两模式可编译） |
| `test/benchmark/bench_stencil.cu` | 基准 `bench_stencil` |

### 用法

```bash
cmake .. -DPHIX_ENABLE_NVTX=ON -DPHIX_LINEINFO=ON
nsys profile ./test/benchmark/bench_stencil     # 时间线带 bench/* 区间
```

```cpp
#include "perf/Perf.h"
{
    PHIX_NVTX_RANGE("myloop/step");
    perf::CudaEventTimer t;  t.start();
    ... kernels ...
    double ms = t.stopMs();
}
```

---

## 测试

`module_perf`：WallTimer 量程（50ms sleep 宽容界）、CudaEventTimer 对
10×64MB memset 计时非零有界、NVTX 区间两种模式编译且作用域安全。
`bench_stencil` 注册为 ctest 冒烟。全量 ctest 22/22 通过。

## 兼容性

纯新增 + 构建默认值变化（未显式指定 build type 的构建从"无优化"变为
Release——这是修复）。
