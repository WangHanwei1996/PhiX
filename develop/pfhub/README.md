# PFHub 基准算例复现总索引

NIST PFHub（pages.nist.gov/pfhub）六个相场基准的 PhiX 复现。
工作流：求解器在 `applications/solvers/`（能复用则复用），
算例目录只放配置 + 初始场生成脚本 + 结果 CSV。

| 算例 | 物理 | 求解器 | 复用/新写 | 验证要点 |
|------|------|--------|-----------|----------|
| [bm1_spinodal](bm1_spinodal/) 1a/1b | 旋节分解 CH | Cahn-Hillard_double-well | **复用**+能量 CSV | F(0)=319 与 PFHub 一致；质量逐位守恒 |
| [bm2_ostwald](bm2_ostwald/) 2a | CH+4×AC 熟化 | CH_4AC_Ostwald | 新写 | MOOSE 参考 ±5% |
| [bm3_dendrite](bm3_dendrite/) | Karma–Rappel 枝晶 | dendrite_growth | **复用**+尖端诊断 | Ṽ=7.8e-4 与 KR98 可解性同量级 |
| [bm4_elastic](bm4_elastic/) 4a | CH+FFT 弹性析出 | CH_elastic_precipitate | 新写（谱弹性 v2.33.0） | a₁₀=20.09 近平衡；F 单调 |
| [bm5_stokes](bm5_stokes/) 5a/5b | Stokes 通道流 | LBM_channel | 新写（LBM v2.32.0） | 泊肃叶 0.05%；dp/dx 0.2% |
| [bm6_electro](bm6_electro/) 6a | CH+Poisson 静电 | CH_Poisson_electro | 新写（CG v2.31.0） | 见算例 README |

每个算例目录的 README 记录规格来源、与规格的偏差（如有）、
运行命令与验证数据。基础设施模块（能量归约、PFHubWriter、
interfacePosition、PoissonSolver、LBM 边界、ElasticityFFT、
DomainMask）见 changelog v2.29.0–v2.34.0。

## 未覆盖（列入后续）

- BM1c/BM2c（T 形域）与 BM6b（矩形+半圆域）：需 DomainMask
  掩码化的守恒面通量 CH（模块已就绪，求解器未写）；
- BM2d/BM1d（球面）：需曲面网格，超出结构化网格范围；
- BM4 非均匀模量变体：需 Moulinec–Suquet 迭代；
- BM7（MMS 验证）：与 test/convergence 的 MMS 套件同类，未列为算例。
