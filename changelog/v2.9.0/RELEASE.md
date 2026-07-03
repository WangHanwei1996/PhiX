# v2.9.0 — GPU 归约工具箱（`PhiX::reduce`）

## 摘要

新增设备端归约模块：在 **不下载整场** 的前提下对 `ScalarField` 的物理格
（自动排除 ghost 晕圈）做 max/min/绝对值最大/求和/平方和/L2 范数/非有限值
检测。填补框架"运行时盲飞"的短板——此前看一眼 max|φ| 都要整场过 PCIe。

典型用途：自适应时间步（对 RHS 场取 `fieldMaxAbs`）、NaN/Inf 哨兵早停、
守恒量监控（`fieldSum`）、稳态检测、在线诊断。
本模块是后续 v2.10.0 自适应时间步的地基。

---

## 核心变更

### 新增文件

| 文件 | 说明 |
|------|------|
| `include/field/Reduce.h` | 公共 API（纯 host 头，可被 `.cpp` 引用） |
| `src/field/Reduce.cu` | CUB `DeviceReduce` 实现（已注册进 `phix` 静态库） |
| `test/moduleTest/field/test_reduce.cu` | 模块测试 `module_reduce` |

### API（`namespace PhiX::reduce`）

```cpp
double fieldMax   (const ScalarField& f);   // 物理格最大值
double fieldMin   (const ScalarField& f);
double fieldMaxAbs(const ScalarField& f);   // max |value|
double fieldSum   (const ScalarField& f);   // 裸求和（×格体积 = 积分）
double fieldSumSq (const ScalarField& f);
double fieldL2    (const ScalarField& f);   // sqrt(fieldSumSq)

bool   fieldHasNonFinite(const ScalarField& f);  // 物理格含 NaN/±Inf?

void   freeScratch();   // 显式释放内部缓存的 scratch（可选）
```

### 实现要点

- **只归约物理格**：`thrust::transform_iterator` 把物理线性索引映射到
  含 ghost 的存储索引再取值，CUB 全程不触碰 ghost —— ghost 可以留脏值/
  毒值而不影响结果。
- CUB 3.2（CUDA 13 自带，`include/cccl/`），零新外部依赖。
- 内部缓存 CUB 临时缓冲（grow-only）+ 8 字节结果槽，跨调用复用，
  无每步 `cudaMalloc` 抖动；不在静态析构里 `cudaFree`（避免 context
  已销毁时报错），由 `freeScratch()` 显式释放。
- 每次调用同步返回（默认流 + `cudaMemcpy` 结果回传）——语义即
  "此刻我要这个数"。
- 场未 `allocDevice` 时抛 `std::runtime_error`。

---

## 测试

`module_reduce`（已注册 ctest）：

- 1D/2D/3D 非 2 幂尺寸（17、37×23、9×7×5），ghost=1/2；
- **全部 ghost 格填 1e300 毒值** —— 任何晕圈泄漏立刻翻转 max/sum；
- max/min/maxAbs 与 CPU 参考按位相等，sum/sumsq/L2 相对误差 ≤1e-12；
- ghost 内 NaN 不触发 `fieldHasNonFinite`，物理格 NaN/Inf 必触发；
- 未分配设备内存调用抛异常。

全量 ctest 15/15 通过（含既有 14 项，无回归）。

---

## 兼容性

纯新增模块，无既有 API 变更。
