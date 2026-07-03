# feature/core-upgrades 分支任务清单

> 建于 2026-07-03，基于 v2.8.0（master@c42efc7）。
> 依据 [framework_evaluation.md](framework_evaluation.md) 的缺失清单，按依赖顺序排列。
> 每个模块：实现 → 自带测试通过 → 版本号按改动大小升级 → changelog → git 提交。
>
> **状态（2026-07-03）：M1–M7 全部完成**，版本 v2.9.0 → v2.14.0 逐一落地，
> 与下表规划一致。额外收获：M4 收敛套件首跑即抓出 Iso9 Laplacian 权重错误
> （零阶不一致，v2.11.1 修复）；M5 基准量化了框架开销（完整 Euler 步比裸
> kernel 慢 2.8–3.7×）；M6 实测 FLOAT 提速 CD2 1.6× / CD4 3.1×。
> 详见各版 `changelog/v2.x.y/RELEASE.md`。

## 执行顺序与版本规划

| # | 模块 | 交付物 | 验收标准 | 计划版本 |
|---|------|--------|----------|----------|
| M1 | **GPU 归约工具箱** | `include/field/Reduce.h` + `src/field/Reduce.cu`：`fieldMax/Min/MaxAbs/Sum/L2/hasNaN`，仅归约物理格（跳过 ghost），CUB 实现，临时缓冲缓存复用 | 模块测试：随机场 + ghost 填污染值，GPU 结果与 CPU 参考一致到 1e-12（相对）；NaN 检出 | v2.9.0 |
| M2 | **自适应时间步** | rate-limited 控制器：Euler 路径下 dt=clamp(tol/max\|RHS\|, dtMin, dtMax)，接入 `Solver` 与 `EquationSystem`；NaN 哨兵早停 | 测试：扩散衰减问题 dt 单调增长、每步 max\|Δφ\|≤tol、终态与定小步参考解一致；NaN 注入触发异常 | v2.10.0 |
| M3 | **空间格式库扩充** | 四阶中心差分 Laplacian/Gradient（scheme tag `CentralDifference4`，ghost≥2）+ 一阶迎风对流项 `adv(u,f)`；接入 DSL 工厂与 ghost 校验 | 测试：sin 场解析解误差（CD4 显著小于 CD2）；迎风传输阶跃剖面无过冲、CFL<1 稳定 | v2.11.0 |
| M4 | **收敛阶验证（MMS）** | `test/convergence/`：空间阶（CD2/ISO≈2、CD4≈4，网格加密序列）+ 时间阶（Euler≈1、RK4≈4） | ctest 断言实测阶在标称 ±0.2 内 | v2.11.1 |
| M5 | **性能基础设施** | `include/perf/Timer.h`（wall + CUDA event 计时、NVTX 区间，`PHIX_ENABLE_NVTX` 可选）；CMake 默认 Release、`PHIX_LINEINFO` 选项；`test/benchmark/` cells/s 基准 | Timer 精度测试；benchmark 可运行并输出表格 | v2.12.0 |
| M6 | **精度策略** | `include/core/Real.h` + CMake `PHIX_PRECISION=DOUBLE\|FLOAT`；核心库 double→Real；磁盘格式保持 double（读写转换）；FLOAT 模式下应用求解器不参编 | DOUBLE 模式全量 ctest 通过（默认零行为变化）；FLOAT 模式库+模块测试编译通过、核心测试放宽容差通过 | v2.13.0 |
| M7 | **KKS 两相分配模块** | `include/material/KKS.h` + `src/material/KKS.cu`：抛物自由能等化学势闭式分配 c=h·c_s+(1−h)·c_l；设备端 `KKSView` 可嵌入 PHIX_FN/DSL；μ 场与驱动力 ΔG 辅助 kernel；示例接线（面通量 ∇·(M∇μ)） | 单元测试：分配闭式解精确；1D 平衡界面：体相 c_s/c_l → 平衡值、μ 空间均匀、总溶质守恒（面通量）、平衡界面不漂移；对照 CH 展示界面无非物理溶质富集 | v2.14.0 |

## 模块间依赖

- M2 依赖 M1（max|RHS| 归约）。
- M4 依赖 M3（CD4 的阶要验）。
- M7 依赖 M1（守恒/均匀性诊断用归约），并受益于 M4 的测试设施。
- M6 放在 M7 之前：核心库先完成 Real 化，KKS 直接以 Real 编写。
  （若 M6 风险失控，允许降级为"仅核心场/算子层 Real 化"，KKS 照常推进。）

## 工程约定

- 每模块独立提交，conventional-commit 风格（`feat(reduce): ...`）。
- changelog 按 `changelog/v<X.Y.Z>/RELEASE.md` 中文撰写，风格仿 v2.6.x/v2.7.0。
- 新 `.cu` 源文件必须显式注册进 `add_library(phix ...)`（无 GLOB）；
  新测试用 `add_test` 注册进 ctest。
- 构建环境：`build/`（Release，PHIX_CUDA_ARCH=75，RTX 5080 上经 PTX JIT 运行）。

## KKS 模块动机（背景）

现有 c 方程求解器（CH_AC_2D、GFA_binary、GFA_FeB）均为经典 CH/WBM 形式：
界面内单一浓度场同时受两相自由能作用，扩散界面加宽后界面区出现非物理的
溶质富集/贫化与虚假驱动力（interface excess / solute trapping at
numerically wide interfaces）。KKS（Kim-Kim-Suzuki, PRE 60, 7186 (1999)）
把每个格点的 c 分解为两相浓度 c_s、c_l，以等化学势
∂f_s/∂c_s = ∂f_l/∂c_l 约束闭合，使化学自由能对界面能的贡献解耦——
界面宽度可以取数值上方便的值而不引入非物理溶质传递。
首期实现抛物自由能（f_i = ½k_i(c−c_i^eq)² + b_i）的闭式分配，
接口设计预留表格自由能（FreeEnergyTable）+ 逐点 Newton 的扩展位。
