# implicit_examples — 隐式/半隐式求解示例集

基于 v2.16.0 线性求解层（matrix-free CG + CUDA graph）与 v2.17.0
`SemiImplicitSolver`（IMEX）。三个自包含算例，由易到难：

| 算例 | 方程 | 拆分 | dt 相对显式极限 | 自检 |
|------|------|------|----------------|------|
| `heat_implicit` | 热传导 ∂T/∂t = D∇²T | 纯隐式（N≡0） | **200×** | 总热量守恒（0 漂移） |
| `allen_cahn_semi` | Allen-Cahn 收缩圆 | κ∇² 隐式 + 双阱显式 | 20× | 半径 vs 曲率流理论 R²=R0²−2Mκt（~2%） |
| `spinodal_ch_semi` | Cahn-Hilliard 旋节分解 | ∇⁴ 隐式 + M∇²f'(c) 显式 | **50×** | 质量守恒 ~1e-12、max\|c\|→1 |

构建：根 CMake 已注册（`develop/implicit_examples`），二进制生成在本目录。
运行：`./heat_implicit [nSteps]` 等；输出 `output/*.vts`（ParaView）。

## 参数区与拆分选择（重要）

- **CH（∇⁴ 刚性）是半隐式的主场**：dt 收益 50×+ 且无精度陷阱；
- **AC 看刚度比** dt_react/dt_diff = 2δ²/dx²：界面宽（δ≥8dx）时扩散
  主导，20× 稳且准；界面薄时反应与扩散同阶，收益天然有限；
- **Eyre 稳定化**（`LaplacianOp` 第三参 shift = M·W·s）可把 dt 推过
  反应极限，但 dt·M·W·s ≳ 1 时界面动力学被人为拖慢
  ~(1+dt·M·W·s) 倍——**稳定 ≠ 准确**，定量计算盯住 dt·M·W。
