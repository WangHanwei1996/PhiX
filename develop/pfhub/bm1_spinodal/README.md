# PFHub Benchmark 1 — 旋节分解（Cahn-Hilliard）

规格: https://pages.nist.gov/pfhub/benchmarks/benchmark1.ipynb/

f_chem = ρ_s(c−c_α)²(c_β−c)²，ρ_s=5，c_α=0.3，c_β=0.7，κ=2，M=5；
初始场为规格给定的多余弦扰动（c₀=0.5，ε=0.01）。

**求解器**：复用 `applications/solvers/Cahn-Hillard_double-well/2D`
（方程与 BM1 完全一致），配置里加 `pfhub` 段即输出 `free_energy.csv`
（time, free_energy, total_c）。

## 运行

```bash
python3 gen_initial_field.py          # 生成两个变体的 c/mu 初始场
cd a_periodic                          # 或 b_noflux
../../../../applications/solvers/Cahn-Hillard_double-well/2D/Cahn-Hillard_double-well settings/settings.jsonc
```

200×200，dx=1，显式 Euler dt=2e-3（稳定极限≈3.1e-3），50 万步到
t=1000，RTX 5080 上约 30 s。延长 `nSteps` 并用 `start_from` 热重启可继续。

## 结果（t=1000）

| 变体 | BC | F(0) | F(1000) | F 单调降 | 质量漂移 |
|------|----|------|---------|----------|----------|
| 1a | 周期 | 319.10 | 83.74 | ✓ | 0.0 |
| 1b | 无通量 | 319.04 | 72.67 | ✓ | 0.0 |

F(0)≈319 与 PFHub 各上传结果的初始能量一致；总溶质
Σc·dV = 20100.9149909 全程逐位不变（保守格式）。

变体 1c（T 形域）需掩码化的面通量 CH 求解器（`DomainMask`），
暂未实现——见根目录 changelog 的后续计划。
