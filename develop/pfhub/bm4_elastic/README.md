# PFHub Benchmark 4 — 错配析出相（CH + FFT 弹性）

规格: https://pages.nist.gov/pfhub/benchmarks/benchmark4.ipynb/

F = ∫[w·P(η) + κ/2|∇η|² + f_el]dV：
P(η) 为 BM4 给定的 10 阶双阱（阱位 η=0/1，系数 a₀..a₁₀ 硬编码于
求解器），κ=0.29 aJ/nm，w=0.1 aJ/nm³，M=5；
膨胀错配 ε*(η)=0.5%·h(η)·δ_ij，立方模量 C11=250/C12=150/C44=100
aJ/nm³（变体 a：均匀模量）。

**求解器**：`applications/solvers/CH_elastic_precipitate/2D`（新写）。
每步：ε*=ε_T·h(η) → `ElasticityFFT2D`（v2.33.0）解 ∇·σ=0（1 正 +
3 逆 cuFFT）→ μ = w·P′ − ε_T·h′(η)·(σ₁₁+σ₂₂) − κ∇²η → η += dt·M∇²μ。
σ₁₁+σ₂₂ = (C11+C12)(ε₁₁+ε₂₂−2ε*) 由应变场逐点重构，μ 组装是单个
三场 pw + lap。

## 与规格的偏差（记录在案）

- 力学与 η 均用**周期** BC（谱方法一致性）；规格为自由边界 +
  无通量。域取析出相 20 倍半径，周期镜像作用可忽略；
- 演化方程用 CH（保守，规格同款）；仅实现变体 (a)
  小圆核 + 均匀模量——非均匀模量（c,d,g,h）需 Moulinec–Suquet
  迭代，超出当前 ElasticityFFT2D 范围。

## 运行

```bash
python3 gen_initial_field.py    # r=20 nm 圆核, eta_m=0.0065, 界面 5 nm
cd a_circle
../../../../applications/solvers/CH_elastic_precipitate/2D/CH_elastic_precipitate settings/settings.jsonc
```

400×400，dx=1 nm，dt=5e-3，10 万步到 t=500。
输出 bm4_data.csv：time, F, F_el, F_grad, 析出相面积 ∫h dV, 半轴 a₁₀。
