# GFA Solver — 求解方程组

每个时间步按以下顺序求解 11 个方程。

---

## 符号说明

| 符号 | 含义 |
|---|---|
| $\phi$ | 晶体序参量（0：液/非晶，1：晶体） |
| $\eta$ | 非晶序参量（0：液体，1：非晶） |
| $c$ | B 的摩尔分数 |
| $\mu$ | 化学势 $= \partial f / \partial c$ |
| $D$ | 有效扩散系数 $= M_c(\phi,\eta)\cdot c(1-c)$ |
| $h(x)$ | 插值函数 $x^3(10 - 15x + 6x^2)$，$h'(x)=30x^2(1-x)^2$ |
| $g'(x)$ | 双井势导数 $2x(1-x)(1-2x)$ |
| $f_L(c,T)$ | 液相自由能密度 $= G^L(c,T)/V_m^L$ |
| $f_S(T)$ | 固相自由能密度 $= (3G_{Fe}^{bcc}+G_B^\beta+\Delta G_{Fe_3B}^f)/V_m^S$ |
| $\Delta f^{Am\to L}(T)$ | 非晶与液相自由能密度差 $= R T_g \ln(1+\alpha)\,f(T/T_g)/V_m^L$ |

---

## 步骤 1 — 化学势（STEADY）

$$
\mu = \left[1 - h(\phi)\right]\frac{1}{V_m^L}\frac{\partial G^L}{\partial c}
$$

其中

$$
\frac{\partial G^L}{\partial c}
= G_B^L(T) - G_{Fe}^L(T)
+ RT\ln\frac{c}{1-c}
+ (1-2c)\,L_{B,Fe}^L(T)
$$

---

## 步骤 2 — $x$ 方向化学势梯度（STEADY）

$$
(\nabla\mu)_x = \frac{\partial \mu}{\partial x}
$$

---

## 步骤 3 — $y$ 方向化学势梯度（STEADY）

$$
(\nabla\mu)_y = \frac{\partial \mu}{\partial y}
$$

---

## 步骤 4 — 化学势 Laplacian（STEADY）

$$
\nabla^2\mu
$$

---

## 步骤 5 — 有效扩散系数 $D$（STEADY）

$$
D = M_c(\phi,\eta)\cdot c(1-c)
$$

其中 $M_c$ 是场相关的迁移率：

$$
M_c(\phi,\eta) =
\frac{
\left[1-h(\phi)\right]\!\left[\left(1-h(\eta)\right)D_L + h(\eta)\,D_{Am}\right]
+ h(\phi)\,D_S
}{RT}
$$

$$
D_L = 2\times10^{-6}\exp\!\left(\frac{-1.11\times10^5}{RT}\right),\quad
D_S = 1.311\times10^{-6}\exp\!\left(\frac{-1.51\times10^5}{RT}\right)
$$

$D_{Am}$ 为常数（从配置文件读取）。

---

## 步骤 6 — $x$ 方向 $D$ 梯度（STEADY）

$$
(\nabla D)_x = \frac{\partial D}{\partial x}
$$

---

## 步骤 7 — $y$ 方向 $D$ 梯度（STEADY）

$$
(\nabla D)_y = \frac{\partial D}{\partial y}
$$

---

## 步骤 8 — 梯度点积（STEADY）

$$
\nabla D \cdot \nabla\mu
= (\nabla D)_x\,(\nabla\mu)_x + (\nabla D)_y\,(\nabla\mu)_y
$$

---

## 步骤 9 — 浓度方程（TRANSIENT）

变迁移率 Cahn-Hilliard 方程通过展开散度得到：

$$
\frac{\partial c}{\partial t}
= \nabla\cdot\!\left(D\,\nabla\mu\right)
= D\,\nabla^2\mu
+ \nabla D\cdot\nabla\mu
$$

其中 $D = M_c(\phi,\eta)\cdot c(1-c)$。

---

## 步骤 10 — 晶体序参量方程（TRANSIENT）

$$
\frac{\partial\phi}{\partial t}
= -M_\phi\frac{\partial f}{\partial\phi}
+ M_\phi\,\epsilon^2\,\nabla^2\phi
$$

其中

$$
M_\phi = 22.1\exp\!\left(\frac{-140\times10^3}{RT}\right)
$$

$$
\frac{\partial f}{\partial\phi}
= h'(\phi)\!\left[f_S(T) - f_L(c,T) - h(\eta)\,\Delta f^{Am\to L}(T)\right]
+ w_\phi\,g'(\phi)
+ 2\,w_{ex}\,\phi\,\eta^2
$$

---

## 步骤 11 — 非晶序参量方程（TRANSIENT）

$$
\frac{\partial\eta}{\partial t}
= -M_\eta\frac{\partial f}{\partial\eta}
+ M_\eta\,\beta^2\,\nabla^2\eta
$$

其中

$$
\frac{\partial f}{\partial\eta}
= \left[1 - h(\phi)\right]h'(\eta)\,\Delta f^{Am\to L}(T)
+ w_\eta\,g'(\eta)
+ 2\,w_{ex}\,\phi^2\,\eta
$$

---

## CALPHAD 热力学函数

### 液相 $G^L$

$$
G^L = c\,G_B^L(T) + (1-c)\,G_{Fe}^L(T)
+ RT\!\left[c\ln c + (1-c)\ln(1-c)\right]
+ c(1-c)\,L_{B,Fe}^L(T)
$$

$$
L_{B,Fe}^L = -122861 + 14.59\,T \quad \text{[J/mol]}
$$

### 固相 $G^S$（Fe₃B）

$$
G^S = 3\,G_{Fe}^{bcc}(T) + G_B^\beta(T) + \Delta G_{Fe_3B}^f(T)
$$

$$
\Delta G_{Fe_3B}^f = -77749 + 2.59\,T \quad \text{[J/mol]}
$$

### 非晶自由能差 $\Delta f^{Am\to L}$

$$
\Delta f^{Am\to L}(T)
= \frac{R\,T_g\ln(1+\alpha)}{V_m^L}\,f\!\left(\frac{T}{T_g}\right)
$$

$$
f(\tau) =
\begin{cases}
1 - \dfrac{9.9167285\times10^{-1}}{\tau}
  - 1.11737779\times10^{-1}\,\tau^3
  - 4.96612349\times10^{-3}\,\tau^9
  - 1.11737779\times10^{-3}\,\tau^{15},
& \tau \le 1 \\[6pt]
- 1.05443689\times10^{-1}\,\tau^{-5}
- 3.34741816\times10^{-3}\,\tau^{-15}
- 7.02957924\times10^{-4}\,\tau^{-25},
& \tau > 1
\end{cases}
$$
