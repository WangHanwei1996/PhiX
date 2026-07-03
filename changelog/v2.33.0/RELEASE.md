# PhiX v2.33.0 发布说明

## 新增：谱方法均匀弹性模块（PFHub 基础设施 5/6）

面向 PFHub BM4（弹性析出相）等含错配应变的相场问题，新增基于
Khachaturyan 微弹性理论的 FFT 力学平衡求解器。

### `mechanics/ElasticityFFT.h` + `src/mechanics/ElasticityFFT.cu`

- **物理模型**：均匀立方弹性体 + 膨胀型本征应变场
  `ε*_ij(x) = eStar(x)·δ_ij`（相场应用中 `eStar = ε_misfit·h(φ)`），
  在全周期 2D 元胞（平面应变）上精确求解 `∇·σ = 0`，`σ = C:(ε−ε*)`。
- **算法**：谱空间逐模式闭式解——对每个傅里叶模式 ξ 求声学张量
  `K_ik = C_ijkl ξ_j ξ_l` 的 2×2 逆，应变谱直接得出：
  `ε̂11 = s·w1ξ1`、`ε̂22 = s·w2ξ2`、`ε̂12 = ½s(w1ξ2+w2ξ1)`，
  其中 `s = (C11+C12)·ê*`、`w = K⁻¹ξ`。全程 device 计算：
  **1 次正变换 + 3 次逆变换（cuFFT）**，光滑场谱精度。
- **均值模式约定**：`zeroMeanStress = true`（默认）为自由周期元胞
  `⟨σ⟩ = 0 → ⟨ε⟩ = ⟨ε*⟩`；`false` 为刚性约束 `⟨ε⟩ = 0`。
- **弹性能密度**：可选输出 `e_el = ½(ε−ε*):C:(ε−ε*)`，配合
  `reduce::fieldSum` 直接得总弹性能（PFHub BM4 的自由能分量）。
- **精度双轨**：随 `PHIX_PRECISION` 自动切换 cuFFT `D2Z/Z2D` 与
  `R2C/C2R`。
- `ElasticParams2D::validate()` 做正定性检查（`C11>0`、`C44>0`、
  `C11>|C12|`），非法参数抛 `std::invalid_argument`。
- 构建系统：根 CMake 新增 `find_package(CUDAToolkit)` 并对 `phix`
  公开链接 `CUDA::cufft`。

### 已知限制（文档化于头文件）

- 均匀模量（非均匀模量需在其上叠加 Moulinec–Suquet 迭代）；
- 2D 平面应变；本征应变限膨胀型（各向同性错配）。

## 测试

新增 `test/moduleTest/mechanics/test_elasticity.cu`（`module_elasticity`），
五项校验：

1. 均匀本征应变 + `⟨σ⟩=0`：处处 `ε = ε*`，弹性能为 0（无应力自由元胞）；
2. **手工推导单模态锚点**：`e* = A·cos(kx)` 的闭式解
   `ε11 = (C11+C12)/C11·e*`、`ε22 = ε12 = 0`，实测最大偏差 **3.5e-18**
   （机器精度）；
3. Eshelby 内部均匀性：各向同性常数（`C11−C12 = 2C44`）下圆形夹杂
   内部总应变均匀，实测内部弥散 **0.17%**；
4. 力学平衡：重构应力场的中心差分散度
   `max|∇·σ|·dx / max|σ| = 6.8e-5`；
5. 参数正定性校验抛异常。

测试规模：DOUBLE 37/37 通过（新增 1 项），FLOAT 3/3 通过。
