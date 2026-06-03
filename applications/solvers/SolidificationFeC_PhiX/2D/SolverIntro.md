# Fe-C 等轴凝固求解器分析

## 概述

本示例模拟铁碳（Fe-C）二元合金的等轴枝晶凝固过程，采用多相场方法（Multi-Phase-Field, MPF）与扩散方程耦合求解。系统以一个液态基底中的单个固相晶核为初始状态，在均匀冷却条件下追踪固相的生长与溶质再分配过程。

- **维度**：2D（$N_x = 301,\ N_z = 301$，$N_y = 0$）
- **网格间距**：$\Delta x = 1.5\ \mu\text{m}$
- **总步数**：100,000 步
- **物理场**：相场 $\phi$、成分场 $C$（C 摩尔分数）、温度场 $T$（均匀降温）

---

## 时间步循环求解流程

每一时间步的完整求解流程如下：

```
for t = tStart → nSteps:
  1. IP.Set(Phi)                              // 设置界面性质（界面能、迁移率）
  2. DF.CalculateInterfaceMobility(...)       // 计算浓度相关界面迁移率
  3. DO.CalculatePhaseFieldIncrements(...)    // 计算曲率驱动力对应的相场增量
  4. DF.GetDrivingForce(...)                  // 计算化学驱动力
  5. dG.Average(Phi, BC)                      // 对化学驱动力在界面处进行平均
  6. dG.MergePhaseFieldIncrements(...)        // 将化学驱动力合并入相场增量
  7. Phi.NormalizeIncrements(BC, dt)          // 对相场增量进行归一化
  8. DF.Solve(Phi, Cx, Tx, BC, dt)            // 求解溶质扩散方程
  9. Tx.Set(BC, Phi, dt)                      // 更新温度（均匀冷却）
 10. Phi.MergeIncrements(BC, dt)              // 将增量更新到相场
 11. （周期性）写 VTK / 原始数据 / 屏幕输出
```

---

## 各步骤方程详解

### 步骤 1：设置界面性质

对每对相 $(\alpha, \beta)$，从输入参数中根据各向异性模型（Cubic / Iso）计算界面能 $\sigma_{\alpha\beta}$ 和界面迁移率 $\mu_{\alpha\beta}$。

**立方各向异性（Cubic）**界面能（4 次对称）：

$$
\sigma_{\alpha\beta}(\hat{n}) = \sigma_0 \left[1 + \varepsilon_E \left(n_x^4 + n_y^4 + n_z^4 - \tfrac{3}{5}\right)\right]
$$

其中 $\hat{n}$ 为界面法向量，$\varepsilon_E$ 为各向异性强度参数。

---

### 步骤 2：计算浓度相关界面迁移率（Ext 模型）

当界面迁移率模型为 `Ext`（由扩散系数计算）时：

$$
\mu_{\alpha\beta}^{\text{eff}} = \frac{8\left(D_\alpha + D_\beta\right)}{m_L \cdot \eta \cdot \Delta S \cdot \Delta C_{eq}}
$$

- $D_\alpha, D_\beta$：相 $\alpha, \beta$ 中的溶质扩散系数  
- $m_L$：液相线斜率  
- $\eta$：界面宽度  
- $\Delta S = S_\beta - S_\alpha$：两相熔化熵之差  
- $\Delta C_{eq}$：平衡浓度差（液固线间距）

此有效迁移率将界面迁移与溶质扩散相耦合，保证扩散控制凝固行为。

---

### 步骤 3：计算相场增量（双障势，Double Obstacle）

对计算域中每个界面点，对所有相对 $(\alpha, \beta)$ 计算曲率贡献的相场演化方程增量：

$$
\boxed{
\left.\frac{\partial \phi_\alpha}{\partial t}\right|_{\text{curv}} =
\frac{\mu_{\alpha\beta} \cdot \text{scale}}{N}
\left[
\sigma_{\alpha\beta}\left(\nabla^2\phi_\alpha - \nabla^2\phi_\beta
+ \frac{\pi^2}{\eta^2}(\phi_\alpha - \phi_\beta)\right)
+ \sum_{\gamma \neq \alpha,\beta}
(\sigma_{\beta\gamma} - \sigma_{\alpha\gamma})
\left(\nabla^2\phi_\gamma + \frac{\pi^2}{\eta^2}\phi_\gamma\right)
\right]
}
$$

- $N$：当前网格点处相场数目（归一化因子）
- $\text{scale} = \sqrt{V_\alpha / V_\alpha^{ref} \cdot V_\beta / V_\beta^{ref}}$：体积比缩放因子（用于新生晶核的平滑处理）
- $\eta$：界面宽度（物理量，$= N_{\text{IWidth}} \cdot \Delta x$）
- $\frac{\pi^2}{\eta^2} \phi_\alpha$：双障势中的体自由能微分项

当有三相交汇时，还包含三相点能量项：

$$
\Delta \dot{\phi}_\alpha \mathrel{+}=
f_{TJ} \cdot \frac{\pi^2}{\eta^2}
(\sigma_{\alpha\beta} + \sigma_{\beta\gamma} + \sigma_{\alpha\gamma})
(\phi_\alpha \phi_\gamma - \phi_\beta \phi_\gamma)
$$

---

### 步骤 4：计算化学驱动力

基于线性相图模型（平衡分配），对每个界面点上的相对 $(\alpha, \beta)$ 计算化学驱动力：

$$
\boxed{
\Delta G_{\alpha\beta}^{chem} =
\frac{\Delta S}{2}
\left[
\left(T_s^{\alpha\beta} + m_L^{\alpha\beta}\left(C^\alpha - C_s^{\alpha\beta}\right) - T\right)
+ \left(T_s^{\beta\alpha} + m_L^{\beta\alpha}\left(C^\beta - C_s^{\beta\alpha}\right) - T\right)
\right]
}
$$

其中：
- $T_s^{\alpha\beta}$：液相线与固相线的交叉温度（`Ts_0_1`）
- $C_s^{\alpha\beta}$：交叉点浓度（`Cs_0_1`）
- $m_L^{\alpha\beta}$：液相线斜率（`ML_0_1`）
- $m_L^{\beta\alpha}$：固相线斜率（`ML_1_0`）
- $C^\alpha, C^\beta$：当前各相局部平衡浓度
- $\Delta S = S_\beta - S_\alpha$：两相间熔化熵之差（由 `EF_n` 给出）

物理意义：当局部温度低于液相线温度时，$\Delta G^{chem} > 0$，固相生长具有热力学驱动力。

---

### 步骤 5：驱动力界面平均

将化学驱动力在整个扩散界面区域内平均，以消除数值噪声并提高稳定性：

$$
\Delta G_{\alpha\beta}^{avg}(\mathbf{r}) = \frac{1}{|\Omega_{int}|} \int_{\Omega_{int}} \Delta G_{\alpha\beta}^{chem}(\mathbf{r}')\, d\mathbf{r}'
$$

只有界面内 $\phi \in [\phi_{cut}, 1-\phi_{cut}]$ 的点参与平均（`CutOff_0_1 = 0.95`）。

---

### 步骤 6：合并化学驱动力到相场增量

将化学驱动力乘以驱动力前置因子后叠加到相场增量中：

$$
\left.\frac{\partial \phi_\alpha}{\partial t}\right|_{\text{total}} =
\left.\frac{\partial \phi_\alpha}{\partial t}\right|_{\text{curv}}
+ \mu_{\alpha\beta} \cdot \frac{2\pi}{N \cdot \eta} \cdot \Delta G_{\alpha\beta}^{avg}
$$

---

### 步骤 7：归一化相场增量

对所有相场增量进行约束，确保 $\sum_\alpha \phi_\alpha = 1$ 的守恒条件始终满足，并限制单步最大变化量以保证数值稳定。

---

### 步骤 8：求解溶质扩散方程

#### 8a. 菲克扩散（体扩散）

扩散系数（Arrhenius 型）：

$$
D_\alpha(T) = D_\alpha^0 \exp\left(-\frac{Q_\alpha}{RT}\right)
$$

溶质浓度演化（有限差分格式）：

$$
\boxed{
\frac{\partial C}{\partial t} = \nabla \cdot \left(\sum_\alpha \phi_\alpha D_\alpha \nabla C_\alpha\right)
}
$$

数值离散（对每个相 $\alpha$）：

$$
\delta C_\alpha^{i,j,k} = -\sum_{\langle nb \rangle} w_{nb} \cdot \sqrt{\phi_\alpha^{i,j,k} \cdot \phi_\alpha^{nb}} \cdot
\frac{D_\alpha^{i,j,k} + D_\alpha^{nb}}{2} \cdot \left(C_\alpha^{i,j,k} - C_\alpha^{nb}\right)
$$

其中求和遍历所有 Laplacian 模板方向（$w_{nb}$ 为模板权重），仅当两端相体积分数均 $> 0$ 时参与计算。

#### 8b. 反捕获电流（Anti-trapping Current）

为消除界面有限厚度引入的人工溶质捕获效应，添加修正通量：

$$
\boxed{
j_{AT} = -\frac{\eta}{\pi} \cdot \frac{D_\alpha - D_\beta}{D_\alpha + D_\beta} \cdot \sqrt{\phi_\alpha \phi_\beta} \cdot (C^\alpha - C^\beta) \cdot \dot{\phi}_{\alpha\beta} \cdot \hat{n}
}
$$

- $\hat{n}$：界面法向量  
- $\dot{\phi}_{\alpha\beta}$：界面速度（相场对时间导数）  
- 该项对液相注入额外溶质，补偿界面移动导致的非物理浓度分布

#### 8c. 平衡浓度分配

根据线性相图，界面处各相平衡浓度：

$$
C_{eq}^\alpha = C_s^{\alpha\beta} + \frac{T - T_s^{\alpha\beta}}{m_L^{\alpha\beta}}
$$

分配系数（partition coefficient）：

$$
k = \frac{|m_L^{liq}|}{|m_L^{sol}|} = \frac{|ML_{0\_1}|}{|ML_{1\_0}|}
= \frac{7800}{22941} \approx 0.340
$$

---

### 步骤 9：更新温度

温度按均匀冷却速率更新（本例冷却速率 = 0，温度恒定为 $T_0 = 1740\ \text{K}$）：

$$
T^{n+1} = T^n + \dot{T} \cdot \Delta t
$$

---

### 步骤 10：更新相场

将归一化后的相场增量乘以时间步长，显式 Euler 积分：

$$
\phi_\alpha^{n+1}(\mathbf{r}) = \phi_\alpha^n(\mathbf{r}) + \frac{\partial \phi_\alpha}{\partial t}\bigg|_{\text{total}} \cdot \Delta t
$$

并再次强制 $0 \le \phi_\alpha \le 1$ 和 $\sum_\alpha \phi_\alpha = 1$。

---

## 时间步长限制

扩散方程的 CFL 稳定性条件：

$$
\Delta t \le \frac{(\Delta x)^2}{4 D_{max}} = \frac{(1.5\times10^{-6})^2}{4 \times 2\times10^{-8}} \approx 2.81\times10^{-5}\ \text{s}
$$

代码中由 `DF.ReportMaximumTimeStep()` 自动计算并赋值给 `RTC.dt`。

---

## 配置参数表（ProjectInput.opi）

### @RunTimeControl — 运行时控制

| 参数标识 | 说明 | 值 |
|---|---|---|
| `SimTtl` | 仿真标题 | FE-C Equiaxed Solidification |
| `LUnits` / `TUnits` / `MUnits` / `EUnits` | 单位制 | m / s / kg / J |
| `nSteps` | 总时间步数 | 100,000 |
| `FTime` | VTK 输出间隔（步） | 1,000 |
| `STime` | 屏幕输出间隔（步） | 1,000 |
| `dt` | 初始时间步长 | 1×10⁻⁵ s（由扩散 CFL 覆盖） |
| `nOMP` | OpenMP 线程数 | 4 |
| `Restrt` | 重启开关 | No |
| `tStart` | 重启起始时间步 | 0 |
| `tRstrt` | 重启文件输出间隔（步） | 10,000 |

### @Settings — 网格与物理场

| 参数标识 | 说明 | 值 |
|---|---|---|
| `Nx` / `Ny` / `Nz` | 三方向网格数 | 301 / 0（2D）/ 301 |
| `dx` | 网格间距 $\Delta x$ | 1.5×10⁻⁶ m |
| `IWidth` | 界面宽度（网格点数） $N_\eta$ | 4.5 |
| `Comp_0` / `Comp_1` | 组元名称 | C（碳）/ FE（铁） |
| `Phase_0` / `State_0` | 第 0 相名称 / 状态 | Melt / Liquid |
| `Phase_1` / `State_1` | 第 1 相名称 / 状态 | Solid / Solid |
| `RefElement_0/1` | 参考组元（各相） | FE |

> 实际界面宽度：$\eta = N_\eta \cdot \Delta x = 4.5 \times 1.5\ \mu\text{m} = 6.75\ \mu\text{m}$

### @DrivingForce — 驱动力设置

| 参数标识 | 说明 | 值 |
|---|---|---|
| `Average` | 启用驱动力界面平均 | Yes |
| `CutOff_0_1` | 驱动力截断阈值（$\phi$ 分数） | 0.95 |

### @InterfaceProperties — 界面性质

| 参数标识 | 说明 | 值 |
|---|---|---|
| `EnergyModel_0_1` | Melt-Solid 界面能模型 | Cubic（立方各向异性） |
| `Sigma_0_1` | Melt-Solid 界面能 $\sigma$ | 0.24 J/m² |
| `EpsilonE_0_1` | 界面能各向异性强度 $\varepsilon_E$ | 0.75 |
| `MobilityModel_0_1` | Melt-Solid 迁移率模型 | Cubic（立方各向异性） |
| `Mu_0_1` | Melt-Solid 界面迁移率 $\mu$ | 8×10⁻¹⁰ m⁴/(J·s) |
| `EpsilonM_0_1` | 迁移率各向异性强度 $\varepsilon_M$ | 0.35 |
| `EnergyModel_0_0` / `Sigma_0_0` | Melt-Melt 界面能（模型/值） | Iso / 0.24 J/m² |
| `EnergyModel_1_1` / `Sigma_1_1` | Solid-Solid 界面能（模型/值） | Iso / 0.24 J/m² |
| `MobilityModel_0_0` / `Mu_0_0` | Melt-Melt 迁移率 | Iso / 4.0×10⁻¹⁰ m⁴/(J·s) |
| `MobilityModel_1_1` / `Mu_1_1` | Solid-Solid 迁移率 | Iso / 4.0×10⁻¹² m⁴/(J·s) |

### @EquilibriumPartitionDiffusion — 平衡分配扩散

| 参数标识 | 说明 | 值 |
|---|---|---|
| `RefElement` | 扩散求解器参考组元 | FE |
| `Cs_0_1` | 液固线交叉点浓度 $C_s^{01}$ | 0 |
| `Ts_0_1` | 液固线交叉点温度 $T_s^{01}$ | 1809.15 K（纯铁熔点） |
| `ML_0_1` | 液相线斜率 $m_L^{liq}$ | −7800 K |
| `ML_1_0` | 固相线斜率 $m_L^{sol}$ | −22941 K |
| `DC_0` | 液相扩散系数 $D_0^0$ | 2.0×10⁻⁸ m²/s |
| `DC_1` | 固相扩散系数 $D_0^1$ | 6.0×10⁻⁹ m²/s |
| `AE_0` / `AE_1` | 扩散激活能（液/固相） | 0 J/mol（无温度依赖） |
| `EF_0` | 液相熔化熵 $S_0$ | 0 J/(m³·K) |
| `EF_1` | 固相熔化熵 $S_1$（为负，凝固放热） | −1.0×10⁶ J/(m³·K) |
| `Flag_0` / `Flag_1` | 化学计量相标志 | No / No |

> 分配系数：$k = |ML_{0\_1}| / |ML_{1\_0}| = 7800/22941 \approx 0.340$

### @Composition — 成分场初值

| 参数标识 | 说明 | 值 |
|---|---|---|
| `C0_0_C` | Melt 相 C 初始摩尔分数 | 0.0082（≈0.82 at.%） |
| `C0_1_C` | Solid 相 C 初始摩尔分数 | 0.002788（≈0.279 at.%） |
| `C0_0_FE` | Melt 相 Fe 初始摩尔分数 | 0.9918 |
| `C0_1_FE` | Solid 相 Fe 初始摩尔分数 | 0.997212 |
| `Vm_0_C` / `Vm_1_C` | C 组元摩尔体积 | 7.0922×10⁻⁶ m³/mol |
| `CMIN/CMAX_n_C` | C 浓度物理上下限 | [0, 1] |

> 初始固相浓度由分配系数确定：$C_1^C = k \cdot C_0^C \approx 0.340 \times 0.0082 \approx 0.00279$

### @BoundaryConditions — 边界条件

| 方向 | 起始端 | 末端 |
|---|---|---|
| X 轴 | NoFlux（无通量） | NoFlux |
| Y 轴 | NoFlux | NoFlux |
| Z 轴 | NoFlux（实为 X-Z 平面 2D） | NoFlux |

### @Temperature — 温度场

| 参数标识 | 说明 | 值 |
|---|---|---|
| `T0` | 初始均匀温度 | 1740.0 K |
| `DT_Dt` | 冷却速率（负为降温） | 0.0 K/s（本例恒温） |
| `DT_DRX/Y/Z` | 温度梯度（x/y/z 方向） | 0 K/m（无梯度） |
| `R0X/Y/Z` | 温度梯度参考点坐标 | (0, 0, 0) |

---

## 初始化方式

```
Phase 0（Melt）:  充满整个计算域
Phase 1（Solid）: 在域中心 (Nx/2, Ny/2, Nz/2) 植入单个晶核点（PlantGrainNucleus）
```

```cpp
int idx0 = Initializations::Single(Phi, 0, BC, OPSettings);   // Melt 充满全域
Phi.FieldsStatistics[idx0].State = AggregateStates::Liquid;

int idx1 = Phi.PlantGrainNucleus(1, Nx/2, Ny/2, Nz/2);        // Solid 晶核
Phi.FieldsStatistics[idx1].State = AggregateStates::Solid;
```

随后调用 `Cx.SetInitialMoleFractions(Phi)` 根据各相初始浓度设置成分场，`Tx.SetInitial(BC)` 设置初始均匀温度。

---

## 物理参数汇总

| 物理量 | 符号 | 值 |
|---|---|---|
| 系统尺寸 | $L$ | $301 \times 301 \times 1.5\ \mu\text{m} = 451.5 \times 451.5\ \mu\text{m}$ |
| 界面宽度 | $\eta$ | 6.75 μm |
| 液相线斜率 | $m_L$ | −7800 K/（摩尔分数） |
| 固相线斜率 | $m_S$ | −22941 K/（摩尔分数） |
| 分配系数 | $k$ | ≈ 0.340 |
| 纯铁熔点 | $T_m$ | 1809.15 K |
| 初始温度 | $T_0$ | 1740.0 K（过冷度 ≈ 69 K） |
| 界面能（Melt/Solid） | $\sigma$ | 0.24 J/m² |
| 液相扩散系数 | $D_L$ | 2.0×10⁻⁸ m²/s |
| 固相扩散系数 | $D_S$ | 6.0×10⁻⁹ m²/s |
