# PFHub Benchmark 6 — 电化学（CH + Poisson 静电耦合）

规格: https://pages.nist.gov/pfhub/benchmarks/benchmark6-hackathon.ipynb/

F = ∫[ρ(c−c_α)²(c_β−c)² + κ/2|∇c|² + k·c·Φ/2]dV；
μ = f′(c) + kΦ − κ∇²c，∂c/∂t = ∇·(M∇μ)，∇²Φ = −kc/ε（每步求解）；
ρ=5，c_α=0.3，c_β=0.7，κ=2，M=5，k=0.09，ε=90；
Φ 边条：左 Φ=0，右 Φ=sin(y/7)（非均匀 Dirichlet），上下 Neumann；
c 全边无通量；t_end=400。

**求解器**：`applications/solvers/CH_Poisson_electro/2D`（新写，调
v2.31.0 `PoissonSolver`——矩阵自由 CG + CUDA graph）。要点：

- **非均匀 Dirichlet 折 RHS**：CG 算子必须线性，故 CG 内用齐次 BC
  （Fixed 0 / NoFlux），sin(y/7) 的 ghost 提升解析折进右端——
  仅最右列贡献 sin(y_j/7)/dx²，时不变、启动时预计算一次。
  **注意**：PhiX `FixedBC` 是常值 ghost 填充（Dirichlet 落在 ghost
  中心而非壁面中点，边界一阶精度）——提升项必须与之匹配用 g/dx²；
  误用中点式 2g/dx² 会使近壁 Φ 高一倍（已踩坑修正）。FixedBC 升级为
  线性外插（二阶）后此处应同步改回 2g/dx²；
- Φ 以前一步解热启动；Φ 在 CH 中仅逐点出现（kΦ），μ 的 ghost 走
  NoFlux（边界零通量 → 溶质守恒），全程无需 Φ ghost 刷新；
- `pfhub` 节输出 free_energy.csv（含 kcΦ/2 静电能）及 t 末
  x=50 / y=50 截面（y|x, concentration, potential）。

## 运行

```bash
python3 gen_initial_field.py    # 规格多余弦初始场 (c1=0.04, cos(0.2x) 首项)
cd a_square
../../../../applications/solvers/CH_Poisson_electro/2D/CH_Poisson_electro settings/settings.jsonc
```

128²（dx=0.78125），dt=8e-4，50 万步到 t=400（RTX 5080 约 20 分钟，
每步一次 Poisson 解 ~92 CG 迭代）。

变体 b（矩形+半圆域）需 DomainMask 掩码化的 CH+Poisson，未实现。
