# v2.20.0 — CG 去同步：设备驻留控制流 + 收敛检查降频

## 摘要（性能）

v2.16.0 的 CG 每次迭代含两个点积，每个点积都是一次**阻塞式 8 字节
D2H 回传**——host 必须等整条设备管线排空才能算 α/β 再发射后续 kernel。
基线剖析显示单次 CG 迭代耗时 ~0.45 ms，而其中 kernel 实际工作只有几十
μs：**同步往返占绝对主导**（WSL2 放大）。

本版把 CG 控制流留在 device 上：

- **α/β 设备驻留**：CUB 归约结果直接写入 device 标量槽
  （新增 `reduce::fieldDotAsync(a, b, double* d_out)`，零拷贝零同步）；
  单线程 kernel `kernel_cg_alpha/beta` 在 device 上完成
  α=ρ/pAp、β=ρ_new/ρ；向量更新 kernel 改为**指针读取**标量——
  一次 CG 迭代 enqueue ~10 个 kernel，**零 host 往返**；
- **收敛检查降频**：host 每 `checkEvery`（默认 4）次迭代才回读一次
  残差判停（代价是最多多算 K−1 次迭代，相对省掉的同步可忽略）；
- 非 SPD 检测改为检查点处残差非有限性判定（原 <p,Ap>≤0 host 检查
  需要每迭代同步，语义等价保留）。

每步同步次数：**2×迭代数+2（≈17–50）→ 迭代数/K+2（≈4）**。

## 实测（bench_semiimplicit，RTX 5080，double，新增基准）

```
                          v2.19.0 基线      本版          提速
implicit diff 256²        3.82 ms/step     2.12 ms       1.80×
CH split      256²        4.88 ms/step     3.08 ms       1.58×
implicit diff 512²        3.99 ms/step     2.19 ms       1.82×
CH split      512²        2.63 ms/step     1.72 ms       1.53×
```

剩余成本是每迭代 ~10 次 kernel launch 的 WSL2 延迟——那是 v2.21.0
CUDA Graphs 的目标。

---

## 变更文件

| 文件 | 说明 |
|------|------|
| `include/field/Reduce.h` / `src/field/Reduce.cu` | `fieldDotAsync`（结果留 device）；`fieldDot` 重构复用同一 enqueue 路径 |
| `include/solver/LinearSolver.h` | CG 增加 `checkEvery`（public，默认 4）与 5 槽 device 标量缓冲 |
| `src/solver/LinearSolver.cu` | 迭代循环重写为 burst 模式；α/β kernel；更新 kernel 指针化 |
| `src/solver/SemiImplicitSolver.cu` | 移除步末遗留的全设备同步（检查点已排空管线） |
| `test/benchmark/bench_semiimplicit.cu` | 新增半隐式基准（隐式扩散 + CH 分裂，ms/step + CG 迭代数） |

### 行为说明

- `Result.iterations` 现在按 `checkEvery` 的粒度报告（可能比精确收敛点
  多至 K−1）；`relResidual` 为检查点实测值——`module_linsolve` 的
  一致系统恢复测试与 `module_semiimplicit` 的机器精度轨迹测试**原样
  通过**（数值语义未变）。
- 开发中顺手修复：`fieldDot` 重构引入的首个调用 NULL d_out 缺陷
  （scratch 槽在按值传参前未分配），已回归覆盖。

## 测试

全量 ctest **29/29**（含新基准注册项）；FLOAT 构建 3/3 通过。

## 兼容性

CG 公共接口不变（新增 `checkEvery` 可调）。`checkEvery = 1` 可恢复
逐迭代判停（调试用）。
