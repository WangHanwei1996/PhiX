# PhiX v2.35.0 发布说明

## PFHub Benchmark 1（旋节分解）复现：变体 1a / 1b 跑通

PFHub 六算例复现的第一例。按"求解器进 applications/solvers、
算例进 develop"的两步走流程：

### 求解器（复用 + 最小扩展）

`applications/solvers/Cahn-Hillard_double-well/2D` 的方程与 BM1 规格
完全一致（μ = 2ρ(c−c_α)(c−c_β)(2c−c_α−c_β) − κ∇²c，∂c/∂t = M∇²μ），
无需新求解器。仅加一段**可选** `pfhub` 配置节：出现时按
`energy_interval` 输出 `free_energy.csv`（time, free_energy, total_c），

- F = Σ[ρ(c−c_α)²(c_β−c)² + κ/2|∇c|²]dV 全部走 GPU 归约
  （`fieldSumPW` + `fieldGradSq`，v2.29.0 基础设施）；
- 老配置（无 `pfhub` 节）行为零变化。

### 算例（develop/pfhub/bm1_spinodal/）

- `gen_initial_field.py`：规格给定的多余弦初始场（c₀=0.5，ε=0.01），
  DAT 格式写入两个变体的 `settings/initial_field/`；
- `a_periodic/`、`b_noflux/`：200×200，dx=1，ρ_s=5，c_α=0.3，c_β=0.7，
  κ=2，M=5，显式 Euler dt=2e-3（稳定极限≈3.1e-3），50 万步到 t=1000
  （RTX 5080 约 30 s）；
- 结果 CSV 随算例入库。

### 验证（t=1000）

| 变体 | F(0) | F(1000) | F(t) 单调性 | 质量漂移 |
|------|------|---------|-------------|----------|
| 1a 周期 | 319.10 | 83.74 | 401 个采样点严格单调降 | **逐位为 0** |
| 1b 无通量 | 319.04 | 72.67 | 严格单调降 | **逐位为 0** |

F(0)≈319 与 PFHub 已上传结果的初始自由能一致。

### 待办

变体 1c（T 形域）需基于 `DomainMask`（v2.34.0）的掩码面通量 CH
求解器，列入后续。
