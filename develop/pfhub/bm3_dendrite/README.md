# PFHub Benchmark 3 — 枝晶生长（Karma–Rappel 薄界面模型）

规格: https://pages.nist.gov/pfhub/benchmarks/benchmark3.ipynb/

τ(n)∂φ/∂t = ∇·[W(n)²∇φ] + 各向异性交叉项 + (1−φ²)[φ−λU(1−φ²)]，
∂U/∂t = D∇²U + ½∂φ/∂t；
a(n)=1+ε₄cos4θ，W=W₀a，τ=τ₀a²，λ=Dτ₀/(0.6267W₀²)≈15.96；
D=10，Δ=0.3，ε₄=0.05，W₀=τ₀=1。

**求解器**：直接复用 `applications/solvers/dendrite_growth/2D`
（交错面通量格式，v2 版），只加了可选 `pfhub` 输出段：
time、固相分数 ∫(φ+1)/2 dV、自由能
F=∫[½W(n)²|∇φ|²−φ²/2+φ⁴/4+λUφ(1−2φ²/3+φ⁵/5)]dV、
尖端位置 tip_x（`interfacePosition` 底行 φ=0 过零 + 线性插值）。

## 运行

```bash
python3 gen_ic.py     # tanh 种子 R=8 于角点, U=-0.3 均匀
../../../applications/solvers/dendrite_growth/2D/dendrite_growth settings/settings.jsonc
```

域 960×960 W₀（四分之一枝晶，NoFlux 对称），1200²、dx=0.8，
dt=0.008（U 方程极限 0.016 的一半），18.75 万步到 t=1500，
RTX 5080 约 6 分钟。规格精度需求可改 dx=0.4（2400²）。

## 结果（t=1500）

- 尖端位置 8 → 252.9（远离 960 边界，无约束效应）；
- F(t) 301 采样点严格单调降；
- 稳态尖端速度（t>1000 线性拟合）V = 0.1414 W₀/τ₀，
  无量纲 Ṽ = V·d₀/D = 7.8e-4（d₀ = a₁W₀/λ = 0.0554）——
  Δ=0.3 低过冷下与 Karma–Rappel 可解性理论值同量级
  （dx=0.8 分辨率下的预期精度内）。
