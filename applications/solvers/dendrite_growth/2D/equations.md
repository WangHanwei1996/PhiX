# Dendrite Growth Solver (2D) — 求解方程复原

本文档将 `dendrite_growth.cu` 中实际编码的离散流程，反推回其对应的连续偏微分方程形式，便于核对建模是否正确。

---

## 1. 变量

| 符号 | 含义 |
|---|---|
| $\phi(\mathbf{x},t)$ | 相场序参数（$0$ 液相，$1$ 固相） |
| $U(\mathbf{x},t)$ | 无量纲过冷度，$U = (T-T_m)/(L/c_p)$ |

辅助场（每步先稳态求解，再用于瞬态推进）：

$$
\phi_x,\ \phi_y,\ a,\ J_x,\ J_y,\ \dot\phi
$$

---

## 2. 物理常数

代码中 `cfg["constants"]` 提供：

$$
D,\quad \tau_0,\quad W_0,\quad \varepsilon_m,\quad m,\quad \theta_0
$$

派生量：

$$
W_0^2 = W_0 \cdot W_0,\qquad
\lambda \;=\; \frac{D\,\tau_0}{0.6267\,W_0^{2}}
$$

---

## 3. 各向异性函数

定义局部界面法向角

$$
\theta(\mathbf{x},t) \;=\; \operatorname{atan2}(\phi_y,\;\phi_x),
\qquad
\phi_x = \partial_x \phi,\ \phi_y = \partial_y \phi
$$

各向异性系数

$$
a(\theta) \;=\; 1 + \varepsilon_m \cos\!\big(m(\theta-\theta_0)\big)
$$

其角导数

$$
\frac{\partial a}{\partial \theta} \;=\; -\,\varepsilon_m\,m\,\sin\!\big(m(\theta-\theta_0)\big)
$$

---

## 4. 各向异性通量 $\mathbf{J}$

代码采用解析展开形式（已消去 $|\nabla\phi|^2$ 因子）：

$$
J_x \;=\; W_0^{2}\,a\,\Big[\,a\,\phi_x \;+\; \varepsilon_m\,m\,\sin\!\big(m(\theta-\theta_0)\big)\,\phi_y\,\Big]
$$

$$
J_y \;=\; W_0^{2}\,a\,\Big[\,a\,\phi_y \;-\; \varepsilon_m\,m\,\sin\!\big(m(\theta-\theta_0)\big)\,\phi_x\,\Big]
$$

> 注：代码中的 `sin_term = epsilon_m*m*sin(m*(theta-theta_0)) = -∂a/∂θ`，故等价于通常文献中的
> $\mathbf{J} = W_0^2\,a^2\,\nabla\phi \;-\; W_0^2\,a\,\dfrac{\partial a}{\partial\theta}\,(\nabla\phi)^{\perp}$，
> 其中 $(\nabla\phi)^{\perp} = (-\phi_y,\ \phi_x)$。

---

## 5. 相场演化方程

非线性源项（双井 + 热耦合）：

$$
\mathcal{N}(\phi, U) \;=\; \big(\phi - \lambda\,U\,(1-\phi^{2})\big)\,(1-\phi^{2})
$$

时间常数：

$$
\tau(\theta) \;=\; \tau_0\,a(\theta)^{2}
$$

代码中存储的 $\dot\phi \equiv \partial\phi/\partial t$ 满足

$$
\boxed{\;
\tau_0\,a^{2}\;\frac{\partial \phi}{\partial t}
\;=\; \mathcal{N}(\phi,U) \;+\; \frac{\partial J_x}{\partial x} \;+\; \frac{\partial J_y}{\partial y}
\;}
$$

即

$$
\tau(\theta)\,\partial_t \phi
\;=\; \big(\phi - \lambda U(1-\phi^{2})\big)(1-\phi^{2})
\;+\; \nabla\!\cdot\!\mathbf{J}
$$

随后做 Euler 推进 $\phi \leftarrow \phi + \Delta t\,\dot\phi$。

---

## 6. 温度（无量纲过冷度）演化方程

$$
\boxed{\;
\frac{\partial U}{\partial t}
\;=\; D\,\nabla^{2} U \;+\; \tfrac{1}{2}\,\frac{\partial \phi}{\partial t}
\;}
$$

代码中 $\partial\phi/\partial t$ 直接使用上一步存下的 `dphi`（同一时刻、同一 Euler 阶段）。

---

## 7. 求解流水线（与代码一一对应）

每个时间步顺序执行 8 个 sub-equation：

| # | 输入 BC | 更新场 | 表达式 | 类型 |
|---|---|---|---|---|
| 1 | $\phi$ | $\phi_x$ | $\partial_x \phi$ | STEADY |
| 2 | $\phi$ | $\phi_y$ | $\partial_y \phi$ | STEADY |
| 3 | $\phi_x$ | $a$ | $1+\varepsilon_m\cos(m(\theta-\theta_0))$ | STEADY |
| 4 | $\phi_y$ | $J_x$ | 见 §4 | STEADY |
| 5 | $J_x$ | $J_y$ | 见 §4 | STEADY |
| 6 | $J_y$ | $\dot\phi$ | $\big[\mathcal{N} + \partial_x J_x + \partial_y J_y\big]/(\tau_0 a^{2})$ | STEADY |
| 7 | $\phi$ | $\phi$ | $\phi += \Delta t\,\dot\phi$ | TRANSIENT |
| 8 | $U$ | $U$ | $U += \Delta t\,(D\nabla^2 U + \tfrac{1}{2}\dot\phi)$ | TRANSIENT |

> 说明：每个 sub-equation 的 “输入 BC” 列指 `Solver` 在求值该方程前先对哪个场施加边界条件 / 填充 halo。这样链式安排是为了：求 $a$ 时 $\phi_x$ halo 已就绪；求 $J_x$ 时 $\phi_y$ halo 已就绪；求 $\nabla\!\cdot\!\mathbf{J}$ 时 $J_x,J_y$ halo 已就绪。

---

## 8. 耦合常数定义

$$
\lambda \;=\; \frac{D\,\tau_0}{0.6267\,W_0^{2}}
$$

（来自 Karma–Rappel 薄界面渐近匹配，$0.6267 = a_1$ 常数。）

---

## 9. 汇总：连续 PDE 形式

$$
\boxed{
\begin{aligned}
\tau_0\,a(\theta)^{2}\,\partial_t \phi
&= \big(\phi - \lambda U(1-\phi^{2})\big)(1-\phi^{2}) + \nabla\!\cdot\!\mathbf{J}(\phi),\\[4pt]
\partial_t U
&= D\,\nabla^{2} U + \tfrac{1}{2}\,\partial_t \phi,\\[4pt]
\mathbf{J}(\phi)
&= W_0^{2}\,a^{2}\,\nabla\phi \;-\; W_0^{2}\,a\,\frac{\partial a}{\partial\theta}\,(\nabla\phi)^{\perp},\\[4pt]
a(\theta) &= 1+\varepsilon_m\cos\!\big(m(\theta-\theta_0)\big),\quad
\theta = \operatorname{atan2}(\partial_y\phi,\partial_x\phi).
\end{aligned}}
$$

请核对：
1. $\lambda$ 的常数 $0.6267$ 是否与所采用的 Karma 模型一致；
2. 温度方程中潜热源项系数 $1/2$ 是否符合无量纲化约定；
3. 通量 $\mathbf{J}$ 的符号约定（特别是 $(\nabla\phi)^\perp$ 的方向）是否与参考文献一致；
4. 边界条件链（§7 表）顺序是否覆盖所有需要的 halo 更新。
