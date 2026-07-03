# v2.28.0 — 各向异性补全：3D 晶粒取向旋转 + 2D Eggleston 凸化正则

## 摘要

补上 v2.27.0 遗留的两个不确定项：

### ① 3D 晶粒取向（非轴对齐晶体）

`Aniso3DParams` 新增旋转矩阵 `R[9]`（lab→crystal，默认单位阵）与
Bunge z-x-z 欧拉角接口：

```cpp
Aniso3DParams ap;  ap.eps = 0.05;
ap.setEulerZXZ(phi1, Phi, phi2);      // 材料学标准 Bunge 约定
eqPhi.setRHS(anisoDiv3D(phi, ap));
```

通量在晶系求值后旋回实验系：J_lab = W0²·a·[a·p_lab + Rᵀ·v_c]，
v_c,i = 16ε·p_c,i·(n_c,i²−S)，p_c = R·p_lab。每面代价 +2 次 3×3
矩阵乘（纯 FMA）。`validate()` 检查 R 正交性（RᵀR=I，1e-8）。

**测试（物理对称性做裁判）**：绕 z 转 90°（立方点群元素）结果与单位阵
**逐格一致**（<1e-10 相对）；转 45°（非对称操作）结果显著不同（证明
旋转真实生效）；旋转后 lab <110> 方向对应晶系 <100> → a = 1+ε 点检；
非正交 R 抛异常。

### ② 2D 强各向异性凸化（Eggleston, McFadden & Braun 2001）

ε 超过凸性极限 1/(m²−1) 时，刚度 γ+γ'' 在 γ 极大方向周围的锥内变负
（缺失取向，演化病态）。`AnisoParams.regularize = true` 启用边际稳定
延拓：锥内 γ̃(δ) = A·cos δ（γ̃+γ̃''≡0），在 δ=±θ_m 处 C¹ 匹配，
θ_m 由 tan(θ_m)·γ(θ_m) = ε·m·sin(m·θ_m) 在 host 端二分求解一次
（`anisoComputeRegularization(eps, m)` 公开可查）。锥内判据
cos(mθ') > cos(mθ_m) 复用现有递推结果，仅锥内面（界面小部分）付一次
atan2。**亚临界 ε 下该开关为严格 no-op**（触发阈值置于不可达）。

**测试**：θ_m/A 的 C⁰/C¹ 匹配条件数值验证（<1e-12）；亚临界
regularize=true 与 false **逐位一致**；ε=0.15（m=4，极限的 2.2 倍）
正则化 AC 演化 400 步无 NaN、有界。

**范围说明**：凸化目前 2D-only；3D 强各向异性保持 ε<0.3 硬界并在
validate 报错信息中注明。

## 测试

全量 ctest **33/33**，FLOAT 3/3。

## 兼容性

纯新增参数（默认值下行为不变：R=单位阵、regularize=false）。
