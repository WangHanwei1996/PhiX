# 相场界面厚度的数学推导

> 对象：双井 + 梯度能形式的非守恒序参量 $\phi$（Allen–Cahn 型相场）。
> 结论：**平衡界面厚度只由梯度能系数 $\varepsilon^2$（`eps_sq`）与势垒高度 $w$（`w_phi`）决定**，
> 与迁移率、温度、体自由能（$f_L,f_S$）等都无关。
>
> 适用代码：`applications/solvers/GFA_binary`（$g(\phi)=\phi^2(1-\phi)^2$，梯度项 $\tfrac{\varepsilon^2}{2}|\nabla\phi|^2$）。

---

## 1. 模型设定

序参量 $\phi$ 的自由能泛函中，与界面结构有关的两项为

$$
\mathcal{F}[\phi]=\int_\Omega\Big[\,w\,g(\phi)+\frac{\varepsilon^2}{2}\,|\nabla\phi|^2\,\Big]\,\mathrm{d}V,
\qquad
g(\phi)=\phi^2(1-\phi)^2 .
$$

- $g(\phi)$：双井势，在 $\phi=0$（液相）与 $\phi=1$（固相/晶体）处取极小 $g=0$，在 $\phi=\tfrac12$ 处取势垒峰 $g=\tfrac{1}{16}$；
- $w\equiv$ `w_phi`：势垒高度 $[\mathrm{J/m^3}]$（量纲化时；本算例非量纲）；
- $\varepsilon^2\equiv$ `eps_sq`：梯度能系数 $[\mathrm{J/m}]$。

> **关于体驱动力的说明。** 完整 $\phi$ 方程还含体自由能差驱动项 $(f_S-f_L)\,h'(\phi)$。
> 在两相**共存平衡**的平直界面处（两相化学势/巨势相等），该项的净贡献为零，
> 不改变界面廓线，只在偏离平衡时驱动界面**移动**（属动力学，不改变厚度）。
> 因此推导平衡界面厚度时只保留 $w\,g+\tfrac{\varepsilon^2}{2}|\nabla\phi|^2$。

---

## 2. 平衡界面：Euler–Lagrange 方程

考虑沿 $x$ 方向的一维平直界面 $\phi=\phi(x)$，边界条件

$$
\phi(-\infty)=0,\qquad \phi(+\infty)=1,\qquad \phi'(\pm\infty)=0 .
$$

对泛函取变分，平衡（$\delta\mathcal F/\delta\phi=0$）给出

$$
\frac{\delta \mathcal F}{\delta\phi}=w\,g'(\phi)-\varepsilon^2\,\phi''=0
\quad\Longrightarrow\quad
\boxed{\;\varepsilon^2\,\phi''=w\,g'(\phi)\;}
$$

其中

$$
g'(\phi)=2\phi(1-\phi)(1-2\phi).
$$

（与代码 `g_prime(phi) = 2*phi*(1-phi)*(1-2*phi)` 一致。）

---

## 3. 首次积分（能量等分）

将方程两边乘以 $\phi'$：

$$
\varepsilon^2\,\phi''\phi'=w\,g'(\phi)\,\phi'
\quad\Longrightarrow\quad
\frac{\mathrm{d}}{\mathrm{d}x}\!\left[\frac{\varepsilon^2}{2}(\phi')^2\right]
=\frac{\mathrm{d}}{\mathrm{d}x}\big[\,w\,g(\phi)\,\big].
$$

积分一次：

$$
\frac{\varepsilon^2}{2}(\phi')^2=w\,g(\phi)+C .
$$

由边界条件 $x\to\pm\infty$ 时 $\phi'\to0$ 且 $g\to0$，得 $C=0$。于是

$$
\boxed{\;\frac{\varepsilon^2}{2}(\phi')^2=w\,g(\phi)\;}
\tag{首次积分}
$$

这条关系即"**梯度能密度 = 势垒能密度**"（界面处两者逐点相等，能量等分）。

取正根（$\phi$ 随 $x$ 增大），并用 $\sqrt{g(\phi)}=\phi(1-\phi)$（$\phi\in[0,1]$）：

$$
\phi'=\sqrt{\frac{2w}{\varepsilon^2}}\;\phi(1-\phi).
$$

---

## 4. 平衡廓线 $\phi(x)$

分离变量：

$$
\frac{\mathrm{d}\phi}{\phi(1-\phi)}=\sqrt{\frac{2w}{\varepsilon^2}}\;\mathrm{d}x .
$$

左边 $\displaystyle\int\frac{\mathrm{d}\phi}{\phi(1-\phi)}=\int\!\Big(\frac1\phi+\frac1{1-\phi}\Big)\mathrm{d}\phi=\ln\frac{\phi}{1-\phi}$，故

$$
\ln\frac{\phi}{1-\phi}=\sqrt{\frac{2w}{\varepsilon^2}}\,(x-x_0).
$$

定义**特征长度**

$$
\boxed{\;\lambda\equiv\sqrt{\dfrac{\varepsilon^2}{2w}}\;}
$$

解得 logistic / 双曲正切廓线：

$$
\boxed{\;\phi(x)=\dfrac{1}{1+e^{-(x-x_0)/\lambda}}
=\dfrac12\Big[\,1+\tanh\dfrac{x-x_0}{2\lambda}\,\Big]\;}
$$

$x_0$ 为界面中心（$\phi=\tfrac12$ 处）。

---

## 5. 界面厚度的几种定义

廓线的所有"宽度"度量都正比于 $\lambda$，差别只是常数因子。常用三种：

| 定义 | 表达式 | 与 $\lambda$ 关系 |
|---|---|---|
| **特征长度** $\lambda$ | $\sqrt{\varepsilon^2/(2w)}$ | $\lambda$ |
| **本项目约定** $\delta$ | $\sqrt{2\varepsilon^2/w}$ | $\delta=2\lambda$ |
| 中心切线宽度 $W_\mathrm{t}=1/\phi'_{\max}$ | $\sqrt{8\varepsilon^2/w}$ | $4\lambda=2\delta$ |
| 10%–90% 宽度 | $2\lambda\ln 9$ | $\approx4.39\lambda$ |

其中中心最大斜率 $\phi'_{\max}=\phi'(x_0)=\sqrt{\tfrac{2w}{\varepsilon^2}}\cdot\tfrac12\cdot\tfrac12=\dfrac{1}{4\lambda}$。

本项目（及 `settings.jsonc` 注释）采用

$$
\boxed{\;\delta=\sqrt{\dfrac{2\,\varepsilon^2}{w}}=\sqrt{\dfrac{2\,\texttt{eps\_sq}}{\texttt{w\_phi}}}\;}
\qquad\Longrightarrow\qquad
\delta\propto\sqrt{\dfrac{\varepsilon^2}{w}} .
$$

**关键结论：界面厚度只由 $\varepsilon^2$ 和 $w$ 这两个参数决定。**
迁移率 $M_\phi,M_c$、时间步 $\mathrm{d}t$、温度 $T$、冷却速率、体自由能表 $f_L/f_S/\partial_c f$ 等
**都不影响平衡界面厚度**（它们只影响界面演化的快慢与相稳定性）。

---

## 6. 界面能（界面张力）$\sigma$

单位面积的过剩自由能：

$$
\sigma=\int_{-\infty}^{\infty}\Big[w\,g(\phi)+\frac{\varepsilon^2}{2}(\phi')^2\Big]\mathrm{d}x .
$$

由首次积分两项相等，被积函数 $=\varepsilon^2(\phi')^2$。换元 $\mathrm{d}x=\mathrm{d}\phi/\phi'$：

$$
\sigma=\int_{-\infty}^{\infty}\varepsilon^2(\phi')^2\,\mathrm{d}x
=\int_0^1\varepsilon^2\,\phi'\,\mathrm{d}\phi
=\varepsilon^2\sqrt{\frac{2w}{\varepsilon^2}}\int_0^1\phi(1-\phi)\,\mathrm{d}\phi .
$$

由 $\displaystyle\int_0^1\phi(1-\phi)\,\mathrm{d}\phi=\tfrac16$，得

$$
\boxed{\;\sigma=\dfrac{1}{6}\sqrt{2\,w\,\varepsilon^2}=\dfrac{\sqrt{2}}{6}\sqrt{\texttt{eps\_sq}\cdot\texttt{w\_phi}}\;}
\qquad\Longrightarrow\qquad
\sigma\propto\sqrt{\varepsilon^2 w}.
$$

---

## 7. 标度关系与"设计"反解公式

把两条结果并列：

$$
\delta=\sqrt{\frac{2\varepsilon^2}{w}}\propto\sqrt{\frac{\varepsilon^2}{w}},
\qquad
\sigma=\frac16\sqrt{2w\varepsilon^2}\propto\sqrt{\varepsilon^2 w}.
$$

即 $\varepsilon^2$ 与 $w$ **同时**控制厚度 $\delta$ 与界面能 $\sigma$。二者可独立反解：

$$
\boxed{\;\varepsilon^2=3\,\sigma\,\delta\;},\qquad
\boxed{\;w=\dfrac{6\,\sigma}{\delta}\;}
$$

（验证：$\sqrt{2\varepsilon^2/w}=\sqrt{2\cdot3\sigma\delta/(6\sigma/\delta)}=\sqrt{\delta^2}=\delta$；
$\tfrac16\sqrt{2w\varepsilon^2}=\tfrac16\sqrt{2\cdot\tfrac{6\sigma}{\delta}\cdot3\sigma\delta}=\tfrac16\sqrt{36\sigma^2}=\sigma$。）

**调参指南：**

- **只改厚度、保持界面能**：固定 $\varepsilon^2 w$。令 $\varepsilon^2\to a\,\varepsilon^2,\ w\to w/a$，则 $\delta\to a\,\delta$，$\sigma$ 不变。
- **只改界面能、保持厚度**：固定 $\varepsilon^2/w$。令 $\varepsilon^2\to a\,\varepsilon^2,\ w\to a\,w$，则 $\sigma\to a\,\sigma$，$\delta$ 不变。
- **加厚界面**：增大 `eps_sq` 或减小 `w_phi`（但会改 $\sigma$，除非按上面同步缩放）。

---

## 8. 离散化与分辨率要求

上述 $\delta$ 是**连续**理论值；数值上还须用网格**解析**它。中心差分 Laplacian 要求界面至少跨

$$
\frac{\delta}{\mathrm{d}x}\gtrsim 3\sim 4\ \text{个格点},
$$

否则界面被网格"钉住"（grid pinning）、出现各向异性与廓线畸变。$\mathrm{d}x$ **不决定** $\delta$，只决定能否解析它。

> 注：显式 Euler 的稳定性由迁移率与梯度系数决定（$\mathrm{d}t\lesssim \mathrm{d}x^2/(4M_\phi\varepsilon^2)$ 等），
> 与界面厚度是**两件独立的事**。

---

## 9. 数值示例（当前 stage-3 参数）

`settings.jsonc`：`eps_sq = 3`，`w_phi = 1`，`dx = 1`（非量纲）。

$$
\lambda=\sqrt{\tfrac{3}{2}}\approx1.225,\qquad
\delta=\sqrt{\tfrac{2\cdot3}{1}}=\sqrt6\approx2.449,\qquad
\sigma=\tfrac16\sqrt{2\cdot1\cdot3}=\tfrac{\sqrt6}{6}\approx0.408 .
$$

界面跨 $\delta/\mathrm{d}x\approx2.45$ 格 —— **偏薄、接近欠解析**，建议把 $\delta$ 提到 $\gtrsim3$ 格。
例如想要 $\delta=3$ 且保持现 $\sigma\approx0.408$：由 $\varepsilon^2=3\sigma\delta,\ w=6\sigma/\delta$ 得
$\varepsilon^2=3\cdot0.408\cdot3\approx3.67$，$w=6\cdot0.408/3\approx0.816$。

---

## 10. 假设与适用范围

1. **平直、平衡界面**：忽略界面曲率（曲率引入 Gibbs–Thomson 修正，量级 $\sim\sigma\kappa$）与体驱动力 $(f_S-f_L)h'(\phi)$ 的净偏移；强驱动下廓线会略偏离 tanh，但厚度量级不变。
2. **双井取 $g=\phi^2(1-\phi)^2$**：换成别的势（如 $\phi^2(1-\phi)^2$ 的不同前因子、或 $\cos$ 型）会改变积分常数 $\tfrac16$ 与廓线形状，但 $\delta\propto\sqrt{\varepsilon^2/w}$、$\sigma\propto\sqrt{\varepsilon^2 w}$ 的标度律不变。
3. **梯度能各向同性**：$\tfrac{\varepsilon^2}{2}|\nabla\phi|^2$ 给出各向同性界面能；各向异性需让 $\varepsilon^2$ 依赖界面法向。

---

### 符号对照

| 符号 | 含义 | 代码/配置 |
|---|---|---|
| $\varepsilon^2$ | 梯度能系数 | `eps_sq` |
| $w$ | 双井势垒高度 | `w_phi` |
| $g(\phi)$ | 双井势 $\phi^2(1-\phi)^2$ | `g_prime` 为其导数 |
| $\lambda$ | 特征长度 $\sqrt{\varepsilon^2/2w}$ | — |
| $\delta$ | 界面厚度 $\sqrt{2\varepsilon^2/w}=2\lambda$ | settings 注释约定 |
| $\sigma$ | 界面能 $\tfrac16\sqrt{2w\varepsilon^2}$ | — |
