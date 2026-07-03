# PhiX v2.36.0 发布说明

## PFHub Benchmark 2（Ostwald 熟化）复现：变体 2a 跑通

### 新求解器 `applications/solvers/CH_4AC_Ostwald/2D`

PFHub BM2 模型：双抛物线自由能 + 1 守恒 c + 4 非守恒 η
（ϱ²=2，c_α=0.3，c_β=0.7，κ_c=κ_η=3，w=1，α=5，M=L=5）。
结构仿既有 `Cahn-Hillard+Allen-Cahn_double-well`：

- h 与 ∂f/∂c 对晶粒可加 → μ 方程由 4 个 pw 项 + lap 拼装（Term 层）；
- 晶粒交叉项 Σ_{j≠i}η_j² 走辅助场 S=Ση²（每步 STEADY 重建、冻结于
  时间层 n）→ 4 个 Allen-Cahn 方程彼此同层推进，无顺序偏置；
- 可选 `pfhub` 配置节输出 free_energy.csv，交叉能用恒等式
  αΣΣ_{j≠i}η_i²η_j² = α(S²−Ση⁴) 归约（fieldSumPW ≤3 场限制内）。

### 算例（develop/pfhub/bm2_ostwald/a_periodic）

200×200 周期域，dx=1，dt=1.5e-3，66.7 万步到 t=1000（RTX 5080
约 3.5 分钟）；初始场为规格 i 依赖多余弦（ε=0.05，ε_η=0.1，ψ=1.5）。

### 验证

- F(0)=9064.4，F(t) 401 采样点严格单调降；总溶质**逐位守恒**；
- 与 MOOSE 参考（自适应 FEM+隐式，用户存档
  `develop/Ostwald_Ripening/reference/MOOSE_IA_2a.csv`）对标：
  大部分时段 **±5% 内**（t=50 处 −0.8%，t=1000 处 −4.5%；t≈600
  峰值 −9.7% 为粗化事件时序差，属实现间典型差异）。
