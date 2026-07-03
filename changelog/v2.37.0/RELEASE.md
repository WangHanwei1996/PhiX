# PhiX v2.37.0 发布说明

## PFHub Benchmark 3（枝晶生长）复现跑通

### 求解器（复用 + 最小扩展）

`applications/solvers/dendrite_growth/2D`（交错面通量 Karma–Rappel
薄界面实现）与 BM3 规格完全一致，直接复用。仅加可选 `pfhub`
配置节，输出 bm3_data.csv：

- 固相分数 ∫(φ+1)/2 dV（fieldSumPW）；
- 自由能 F = ∫[½W(n)²|∇φ|² + f_bulk(φ,U)]dV——各向异性梯度能
  用 (φ_x, φ_y) 两场 pw 归约逐点算 a(n)²；
- 枝晶尖端位置：`interfacePosition`（v2.30.0 诊断模块）沿底行
  找 φ=0 过零 + 亚网格线性插值。

### 算例（develop/pfhub/bm3_dendrite）

960×960 W₀ 域（1200²，dx=0.8），dt=0.008，18.75 万步到 t=1500
（RTX 5080 约 6 分钟）；tanh 种子 R=8 于角点 + NoFlux 四分之一
对称；D=10，Δ=0.3，ε₄=0.05，λ≈15.96。

### 验证（t=1500）

- 尖端 8 → 252.9（域 960，无边界约束）；F(t) 301 采样点严格单调降；
- 稳态尖端速度 V = 0.1414 W₀/τ₀ → 无量纲 Ṽ = V·d₀/D = 7.8e-4，
  与 Δ=0.3 低过冷 Karma–Rappel 可解性理论同量级。
