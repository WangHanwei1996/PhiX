# η-only Allen-Cahn 方程说明

本文档描述 `GFA_eta_only.cu` 求解的方程及其对应的自由能模型。

---

## 1. 物理背景

在 Wang & Napolitano 的相场-金属玻璃模型中，结构弛豫变量 $\eta$ 描述非晶相的有序度：

- $\eta = 0$：液态（无序）
- $\eta = 1$：玻璃态（有序）

本验证求解器固定所有晶体相场变量为

$$
\phi_0 = 1,\quad \phi_1 = \phi_2 = \phi_3 = 0,
$$

即整个计算域始终属于非晶/液态，仅 $\eta$ 参与演化。

---

## 2. 自由能密度

在上述限制条件下，局部自由能密度为

$$
f_\eta = h(\eta)\,\Delta f^{SR} + w_\eta\,\eta^2(1-\eta)^2 + \frac{\beta}{2}|\nabla\eta|^2,
$$

其中

$$
h(\eta) = \eta^3(10 - 15\eta + 6\eta^2)
$$

为 5 阶插值函数，$\Delta f^{SR}$ 为液态与玻璃态之间的结构弛豫自由能差（常数，由配置文件给定）。

**符号约定**：$\Delta f^{SR} < 0$ 表示玻璃态更稳定（$\eta = 1$ 为能量较低态）。

---

## 3. 变分导数

对总自由能 $F = \int_\Omega f_\eta\,d\Omega$ 求 $\eta$ 的变分导数：

$$
\frac{\delta F}{\delta\eta}
= h'(\eta)\,\Delta f^{SR} + 2w_\eta\,\eta(1-\eta)(1-2\eta) - \beta\nabla^2\eta,
$$

其中

$$
h'(\eta) = 30\eta^2(1-\eta)^2.
$$

---

## 4. Allen-Cahn 演化方程

$$
\frac{\partial\eta}{\partial t}
= -L_\eta\,\frac{\delta F}{\delta\eta}
= -L_\eta\!\left[30\eta^2(1-\eta)^2\Delta f^{SR} + 2w_\eta\,\eta(1-\eta)(1-2\eta)\right]
+ L_\eta\beta\,\nabla^2\eta.
$$

该方程保证自由能单调下降（梯度流）。

---

## 5. 论文参数（Verification 4 / Fig. 1 趋势复现）

| 参数 | 含义 | 单位 |
|---|---|---|
| $\Delta f^{SR}$ | 结构弛豫驱动力 | J/m³ |
| $\beta$ | $\|\nabla\eta\|^2$ 梯度能系数 | J/m |
| $w_\eta$ | 双势阱能垒高度 | J/m³ |
| $L_\eta$ | Allen-Cahn 迁移率 | m³/(J·s) |

论文给出三种情况（$\Delta f^{SR}$ 对应 4 kJ/mol 的液→玻璃驱动力）：

| Case | $\beta$ (J/m) | $w_\eta$ (J/m³) | 预期现象 |
|---|---|---|---|
| 1 | $4\times10^{-11}$ | $4\times10^{8}$ | 较大能垒，粗结构成核-长大 |
| 2 | $4\times10^{-12}$ | $4\times10^{7}$ | 能垒低一个数量级，更细密 |
| 3 | $4\times10^{-12}$ | $4\times10^{6}$ | 近零能垒，spinodal-like 均匀转变 |

界面特征长度 $\delta = \sqrt{\beta / (2w_\eta)}$（case 1/2 均约 $7.1\times10^{-10}$ m）。

---

## 6. 数值方案

- **空间离散**：有限差分，二阶中心 Laplacian（由 PhiX 框架提供）
- **时间推进**：显式 Euler，时间步 $\Delta t$ 满足稳定性条件

$$
\Delta t \le \frac{\Delta x^2}{2\,L_\eta\,\beta}
$$

- **边界条件**：四周周期边界
- **初始条件**：$\eta = 0.5 \pm 0.05\,U[-1,1]$（由 `initial_field/gen_initial_eta.py` 生成）

---

## 7. 通过标准

1. $\Delta f^{SR} < 0$ 时，最终 $\langle\eta\rangle \to 1$（玻璃态）；
2. 降低 $\beta$ 和 $w_\eta$ 时，形貌变得更细密（case 2 比 case 1 细）；
3. 无噪声时总自由能 $F(t)$ 单调不增；
4. 三种 case 的形貌对比应定性符合论文 Fig. 1 趋势。
