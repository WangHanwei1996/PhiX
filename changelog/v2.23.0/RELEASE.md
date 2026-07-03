# v2.23.0 — 各向异性模块（`operators/Anisotropy`）

## 摘要

把枝晶求解器里手写的 m 重表面能各向异性（Kobayashi 形式）提炼为库级
模块并优化：

```
a(θ) = 1 + ε·cos(m(θ−θ0)),  θ = atan2(φ_y, φ_x)
∇·J,  J = W0²[ a²∇φ + a·a'·(−φ_y, φ_x) ]
```

- **单 kernel 融合**：原实现是 ~9 次 launch + 6 个中间 FaceField +
  2 个格心梯度场的链（2×faceGrad、2×interp、2×facePW、divFace、
  梯度预备）；`anisoDiv()` 在寄存器里现算 4 个面通量并直接累加散度，
  **零中间存储**，且保持保守性（两侧格用完全相同的输入重算同一面）；
- **去超越函数**：FP64 的 atan2+sincos 在消费卡上 1/64 吞吐，第一版
  融合 kernel 反而比链慢 0.6×——改为**复数幂递推**从 (φx,φy) 代数
  求 cos(mθ)/sin(mθ)（θ0 旋转 host 预算），任意 m 通用、全程乘加；
- 实测（RTX 5080，double）：链 vs 融合 512² **0.173→0.132 ms
  （1.31×）**，1024² 0.587→0.492 ms（1.19×），外加省下 8 个场的
  显存与带宽；
- 离散**逐位对齐旧链**：内部格最大偏差 1.4e-12（测试断言）；边界面
  上旧链的 interp 用单侧值（不读 ghost）而融合版用 ghost 平均——
  这是旧链的边界近似差异，已在测试注释说明。

## API

```cpp
#include "operators/Anisotropy.h"

AnisoParams ap;               // W0、ε、m 重对称、θ0 取向
ap.W0 = W0; ap.eps = 0.05; ap.m = 4; ap.theta0 = 0.0;

eqPhi.setRHS( anisoDiv(phi, ap) + pw(phi, U, PHIX_FN (...) {...}) );

anisoFactorOnGPU(phi, aField, ap);   // 格心 a(θ)，供 τ(θ)=τ0·a² 等
aniso::factor(theta, eps, m, th0);   // device 可用的 a(θ) 助手
```

限制：2D（面内 m 重）；ghost ≥ 1；模板读对角邻居——四个域角的
角 ghost 由面 patch BC 不填（见 v2.24.0 的 BCBatch 角格填充）；
ε ≥ 1 抛异常（凸性失效需正则化）。

## 开发中修掉的坑（记录给后来者）

2D 场在 z 向同样有 ghost 填充（storedDims[2]=1+2g），k=0 物理切片
的扁平索引必须含 `sy·g` 偏移——本模块第一版漏掉后 kernel 写到错误
切片，表现为输出恒零。已修复并由 ε=0 退化测试锁死。

## 测试

`module_anisotropy`（已注册 ctest）：ε=0 严格退化为 W0²·CD2 Laplacian
（<1e-10）；与旧面通量链内部逐位一致（1.4e-12）；周期 ghost + 居中
blob 下散度全域和 ≈ 0（保守性）；GPU==CPU 双路径；a(θ) 的 m 重周期
性与参数校验。全量 ctest **30/30**，FLOAT 3/3。

## 兼容性

纯新增。dendrite_growth 求解器可择期迁移到 `anisoDiv`（旧链保持可用）。
