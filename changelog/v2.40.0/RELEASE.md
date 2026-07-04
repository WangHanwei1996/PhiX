# PhiX v2.40.0 发布说明

## PFHub Benchmark 6（电化学）复现：变体 6a 跑通——六算例收官

### 新求解器 `applications/solvers/CH_Poisson_electro/2D`

CH + Poisson 静电耦合（每步一次矩阵自由 CG 解，v2.31.0
`PoissonSolver`，CUDA graph 迭代体 + 前步热启动）：

- μ = f′(c) + kΦ − κ∇²c；∂c/∂t = ∇·(M∇μ)；∇²Φ = −kc/ε；
- **非均匀 Dirichlet（右边界 Φ=sin(y/7)）折 RHS**：CG 算子保持线性
  （齐次 FixedBC/NoFluxBC），ghost 提升解析预折进右端最右列；
- Φ 仅逐点进入 CH（kΦ），μ ghost 走 NoFlux → 溶质严格守恒；
- `pfhub` 节输出 free_energy.csv（含 kcΦ/2 静电能）与 t 末
  x=50 / y=50 截面 CSV（PFHub 交付格式）。

### 修正记录（影响精度，已修）

PhiX `FixedBC` 为**常值 ghost 填充**（Dirichlet 落于 ghost 中心，
边界一阶）：RHS 提升项须用 g/dx²；首版误用中点式 2g/dx² 导致近壁
Φ 偏高一倍，靠"外推壁值 vs sin(y/7)"复合检查抓出。FixedBC 将来
升级线性外插时此处须同步改动（README 有注记）。

### 算例（develop/pfhub/bm6_electro/a_square）

128²（dx=0.78125），dt=8e-4，50 万步到 t=400（RTX 5080 约 20 分钟，
每步 Poisson 解 ~10² 次 CG 迭代）；变体 b（矩形+半圆域）需
DomainMask 掩码化 CH+Poisson，列入后续。

### 验证（t=400）

- F(t) 201 采样点严格单调降（185.34 → 105.95）；总溶质**逐位守恒**；
- 右壁 ghost 中心外推 Φ = 0.784 vs sin(y_j/7) = 0.793（1.1%，
  线性外推离散曲率）；左壁外推 4.4e-4 ≈ 0；
- x=50 截面上 c ∈ [0.299, 0.681]——旋节分解进入双阱 (0.3/0.7)。

### PFHub 六算例总结（v2.35.0–v2.40.0）

BM1a/b（复用 CH 求解器）、BM2a（新 CH_4AC_Ostwald）、BM3（复用
dendrite_growth）、BM4a（新 CH_elastic_precipitate + 谱弹性）、
BM5a/b（新 LBM_channel）、BM6a（新 CH_Poisson_electro）全部跑通，
每例带定量验证；总索引 develop/pfhub/README.md。回归 38/38
DOUBLE + 3/3 FLOAT 全绿。
