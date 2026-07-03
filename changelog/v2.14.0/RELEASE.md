# v2.14.0 — KKS 两相分配模块（`material/KKS`）

## 摘要

新增 Kim-Kim-Suzuki 两相二元合金模块（PRE 60, 7186 (1999)），解决现有
c 方程求解器（CH_AC_2D / GFA_binary / GFA_FeB，均为 WBM/经典 CH 形式）
的核心缺陷：**扩散界面内单一浓度场同时感受两相自由能，数值加宽的界面
产生非物理的溶质富集与虚假化学界面能**。

KKS 把每个格点的浓度分解为两相浓度，以等化学势条件闭合：

```
c = h·c_s + (1−h)·c_l ,   ∂f_s/∂c_s = ∂f_l/∂c_l = μ
```

抛物自由能 f_i = ½k_i(c−c_i⁰)² + b_i 下分配为**闭式解**（无逐点 Newton）：

```
μ   = ( c − h·c_s⁰ − (1−h)·c_l⁰ ) / ( h/k_s + (1−h)/k_l )
c_s = c_s⁰ + μ/k_s ,   c_l = c_l⁰ + μ/k_l
```

物理验证（本版测试实测）：静止 tanh 界面 + 偏离平衡的初始 c，
以 dc/dt = ∇²μ 弛豫后——μ 空间离散度 3e-1 → **5e-13**（完全均匀），
终态 c 与 h 加权两相混合逐点一致到 **2.5e-13**，即**界面零溶质富集**；
总溶质守恒 <1e-9（相对）。

---

## 核心变更

### 新增文件

| 文件 | 说明 |
|------|------|
| `include/material/KKS.h` | `KKSView`（设备端求值器）+ `KKSParabolic`（host 参数/平衡）+ `kks::h/dh` 插值函数 + 场级分配 kernel 声明 |
| `src/material/KKS.cu` | 闭式分配 kernel（GPU/CPU 双路径，已注册 `phix` 库）；公切线平衡求解 |
| `test/moduleTest/material/test_kks.cu` | 模块测试 `module_kks` |

`material/Material.h` 伞头文件已包含 `KKS.h`。全模块以 `Real` 编写，
DOUBLE/FLOAT 两种精度构建均通过。

### API

```cpp
#include "material/Material.h"

// f_s = ½·4·(c−0.2)²,  f_l = ½·1·(c−0.7)²
KKSParabolic model(/*ks=*/4.0, /*cs0=*/0.2, /*kl=*/1.0, /*cl0=*/0.7);
KKSView v = model.view();          // trivially copyable, __host__ __device__

// —— 逐点用法（嵌入 PHIX_FN / fpw2，φ 方程驱动力项）——
eqPhi.setRHS( lap(phi, kappa*Mphi)
            + pw(phi, c, PHIX_FN (Real p, Real cc) {
                  // ΔG>0 促凝固；h'(φ)·ΔG 即 KKS 驱动力项
                  return Mphi * kks::dh(p) * v.drivingForce(cc, kks::h(p));
              }) );

// —— 场级用法（μ 场 → 溶质扩散链）——
kksPartitionOnGPU(model, c, hFrac, cs, cl, mu);   // 一次 kernel 出三个场
bcMu.applyOnGPU(mu);
eqC.setRHS(lap(mu, M));           // dc/dt = ∇·(M∇μ)；变 M(φ) 可走 face-flux 链

auto eq = model.equilibrium();    // 公切线：{mu, cs, cl}
```

驱动力定义：ΔG = f_l(c_l) − f_s(c_s) − μ(c_l−c_s)（巨势差），两相平衡时
恒为 0（任意 h）——化学能对界面能的贡献解耦，界面宽度可取数值方便值。
`KKSView::dmudc(h)` 给出 ∂μ/∂c，配 ∇·(M∇μ) 的有效扩散率 D_eff = M·∂μ/∂c
（显式 dt 稳定性估计用）。

### 与现有 GFA 求解器的衔接

GFA_FeB 现行的 c 相关 f_S 抛物线驱动力（v2.8.0）正是本模块抛物形式的
特例场景：后续可把 GFA_FeB/GFA_binary 的 μ 构造替换为
`kksPartitionOnGPU` + face-flux `∇·(M_c(φ)∇μ)`，消除现有 WBM 形式的
界面溶质伪影。表格自由能（`FreeEnergyTable`）+ 逐点 Newton 的扩展位
已在接口层预留（`KKSView` 为独立求值器，可并列实现 `KKSTableView`）。

---

## 测试

`module_kks`（已注册 ctest，DOUBLE 套件）：

1. **分配恒等式**：7 组 (c,h) 散点上等化学势（两相 μ 相等 <1e-12）、
   混合规则 c=h·cs+(1−h)·cl（<1e-12）、单相极限（h=0/1 时 cl/cs=c）、
   插值端点；
2. **平衡与驱动力**：bs=bl 时 μ_eq=0、平衡浓度=抛物线极小点；
   平衡一致剖面上任意 h 的 ΔG<1e-13；过饱和液相 ΔG>0；
3. **GPU==CPU**：场级 kernel 与 host 参考逐点一致（<1e-14）；
4. **1D 物理弛豫**（12 万步，NoFlux，M=1）：守恒 / μ 均匀化 / 零界面
   富集，实测数字见摘要。

全量 ctest：DOUBLE **23/23** 通过，FLOAT（build-float）float_smoke +
bench 通过。

---

## 兼容性

纯新增模块，无既有 API 变更。
