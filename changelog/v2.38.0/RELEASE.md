# PhiX v2.38.0 发布说明

## PFHub Benchmark 4（错配析出相）复现：变体 4a 跑通

### 新求解器 `applications/solvers/CH_elastic_precipitate/2D`

首个耦合 v2.33.0 谱弹性模块的相场求解器：

- F = ∫[w·P(η) + κ/2|∇η|² + f_el]dV，P 为 BM4 规格 10 阶双阱
  （a₀..a₁₀ 硬编码，阱位 η=0/1）；
- 每步序列：ε* = ε_T·h(η)（STEADY）→ `ElasticityFFT2D::solve`
  （∇·σ=0，1 正 + 3 逆 cuFFT，全 device）→
  μ = w·P′(η) − ε_T·h′(η)·(σ₁₁+σ₂₂) − κ∇²η（单个三场 pw + lap，
  σ 迹用 (C11+C12)(ε₁₁+ε₂₂−2ε*) 逐点重构）→ η += dt·M∇²μ；
- 可选 `pfhub` 节输出 time、F 总量、弹性能、梯度能、
  析出相面积 ∫h dV、半轴 a₁₀（`interfacePosition` 亚网格插值）。
- 设备 lambda 陷阱记录：10 阶多项式系数须为**局部数组**
  （static 数组不被 [=] 捕获，设备端会解引用主机地址）。

### 算例（develop/pfhub/bm4_elastic/a_circle）

400×400 nm（dx=1），r=20 nm 圆核，η_m=0.0065，界面 5 nm；
w=0.1，κ=0.29，M=5，ε_T=0.5%，C=(250,150,100) aJ/nm³；
dt=5e-3，10 万步到 t=500（RTX 5080 约 2.5 分钟）。

已记录偏差：力学+η 均用周期 BC（谱一致性；域为析出相 20 倍），
仅覆盖均匀模量变体（非均匀需 Moulinec–Suquet 迭代，未实现）。

### 验证（t=500）

F 101 采样点严格单调降（14.41→13.95）；a₁₀ = 20.09 稳定
（r=20 为近平衡尺寸）；弹性/梯度/体积能分量全程有限且平滑。
