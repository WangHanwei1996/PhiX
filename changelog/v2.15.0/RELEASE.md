# v2.15.0 — KKS 反截留流（anti-trapping current）

## 摘要

补齐 KKS 定量化的最后一块：**Karma 反截留流**（Karma PRL 87, 115701
(2001)；Echebarria–Folch–Karma–Plapp PRE 70, 061604 (2004)）。

v2.14.0 的等化学势分配消除的是**静态**界面伪影；界面**运动**时，被数值
加宽的界面在单侧扩散（M_s ≪ M_l）下仍会产生随 W 缩放的虚假溶质截留
——界面两侧出现 ∝ V·W 的化学势跳变，固相冻结进过量溶质。反截留流在
溶质方程中加入与界面速度成正比的修正流，把该 O(W) 截留项抵消：

```
∂c/∂t = ∇·( M(φ)∇μ − J_at )
J_at  = − a · W · (c_l − c_s) · (∂φ/∂t) · ∇φ/|∇φ|      （物理流）
```

**基准实测**（1D 掠过界面，Pe = V·W/D ≈ 0.31，M_s/M_l = 1e-3）：
界面两侧 μ 跳变 **1.247 → 0.046（消除 96.3%）**，总溶质守恒保持机器
精度（~1e-16）。

---

## 系数 a 的约定推导与数值标定

经典值 1/(2√2) 属于 EFKP 约定（φ∈[−1,1]，剖面 tanh(x/(√2W))）。
换算到本模块约定（φ∈[0,1]，剖面 ½(1−tanh(x/W))，h=φ²(3−2φ)，
迁移率按 h 插值）恰好放大 2√2 倍 → **默认 a = 1.0**。

数值标定验证了这一推导：基准工况下截留比对 a 严格线性，
零点在 a ≈ 1.04：

```
a=0.353 → 剩余 72.9%   a=0.710 → 37.6%   a=1.060 → −3.8%
a=1.410 → −48.9%（过修正）
```

`a` 保持可调——更换 h(φ) 插值或迁移率插值形式后需用界面宽度收敛
研究重新校验。固相扩散不可忽略时需 Ohno–Matsuura (PRE 79, 031603
(2009)) 修正，本版未含。

---

## 核心变更

| 文件 | 说明 |
|------|------|
| `include/material/KKSAntiTrapping.h` | `KKSAntiTrappingParams{W, a, gradEps}` + `kksAddAntiTrappingGPU/CPU` |
| `src/material/KKSAntiTrapping.cu` | 面心组装 kernel（逐活动轴，法向差分 + 横向平均的 \|∇φ\|），已注册 `phix` 库 |
| `test/moduleTest/material/test_kks_antitrapping.cu` | 模块测试 `module_kks_at` |

### 用法（并入保守面通量链，divFace 之前累加）

```cpp
KKSAntiTrappingParams at;  at.W = W_interface;

faceGradGPU(mu, 0, jx);                                   // jx = ∂μ/∂x
facePWGPU(jx, jx, hFace, PHIX_FN (Real g, Real h) {       // ×M(h)
    return (h * Ms + (Real(1) - h) * Ml) * g; });
kksAddAntiTrappingGPU(model, at, c, phi, dphidt, &jx, &jy);
eqC.setRHS(divFace(jx, jy));            // dc/dt = ∇·(M∇μ − J_at)
```

- `dphidt`：传当前步 φ 方程的 RHS 场（显式 Euler 下即 ∂φ/∂t），在 φ
  更新**之前**求值；
- **面场符号约定**（重要，本模块开发中即因此修正过一次符号）：
  `divFace` 累加 `+∇·F`，链中面场存的是 M∇μ = **物理流的负值**，
  故本函数向面场累加的是 −J_at；
- c/φ/dphidt 进入前需刷新 ghost；输出为 **累加（+=）** 语义，可与任意
  已组装的通量复合；1D/2D/3D 通用。

---

## 测试

`module_kks_at`（已注册 ctest）：

1. **GPU == CPU**：2D 斜界面 + 面场预置底值（同时验证累加语义），
   逐面一致 <1e-14；
2. **1D 符号与幅值**：凝固前沿（固左液右，c_l>c_s）物理流必须指向
   液相；峰值与解析式 a·W·(c_l−c_s)·max(∂φ/∂t) 吻合 5% 内；
3. **动界面物理**（上述基准）：μ 跳变消除 >85%（实测 96.3%）、
   有/无 j_at 两组守恒均 <1e-9（实测 ~1e-16）、基线跳变足够大
   （防测试空转）。

全量 ctest：DOUBLE **24/24** 通过；FLOAT 构建（库含本模块）编译与
smoke 通过。

---

## 兼容性

纯新增，无既有 API 变更。`material/Material.h` 伞头文件已含本模块。
