# Cahn–Hilliard 迁移率 $M_c$ 与溶质扩散系数 $D$ 的关系

> 结论：$M_c$ **不是**溶质扩散系数，而是 Cahn–Hilliard **迁移率**。
> 两者差一个"热力学因子"（自由能对成分的二阶导）：
> $$\boxed{\,D=M_c\cdot\dfrac{\partial^2 f}{\partial c^2}\,}$$
>
> 适用代码：`applications/solvers/GFA_binary`（`GFA_binary` / `GFA_evo`）。
> 相关：界面厚度见 [`../interface_thickness/interface_thickness.md`](../interface_thickness/interface_thickness.md)。

---

## 1. Cahn–Hilliard 方程

守恒序参量 $c$（成分）的演化：

$$
\frac{\partial c}{\partial t}=\nabla\!\cdot\big(M_c\,\nabla\mu\big),
\qquad
\mu=\frac{\partial f}{\partial c}.
$$

- $M_c$：**迁移率**（mobility），即配置里的 `M_c`；
- $\mu$：化学势 $=\partial f/\partial c$；本模型无 $|\nabla c|^2$ 梯度能，故 $\mu$ 纯逐点（无 $-\kappa_c\nabla^2 c$ 项）。

本算例中（两侧 $\mu$，已补 $\partial f_S/\partial c$ 项）：

$$
\mu=\frac{\partial f_L}{\partial c}\,(1-h(\phi))+\frac{\partial f_S}{\partial c}\,h(\phi).
$$

---

## 2. 退化为 Fick 定律 → 得到 $D$

把 $\mu$ 对 $c$ 线性化。由链式法则

$$
\nabla\mu=\frac{\partial\mu}{\partial c}\,\nabla c=\frac{\partial^2 f}{\partial c^2}\,\nabla c .
$$

代入 CH 方程，设 $M_c$、$\partial^2 f/\partial c^2$ 空间缓变：

$$
\frac{\partial c}{\partial t}
=\nabla\!\cdot\!\Big(M_c\,\frac{\partial^2 f}{\partial c^2}\,\nabla c\Big)
\;\approx\;
\underbrace{M_c\,\frac{\partial^2 f}{\partial c^2}}_{\displaystyle D}\;\nabla^2 c .
$$

与 Fick 第二定律 $\partial c/\partial t=D\nabla^2 c$ 对照，即得

$$
\boxed{\;D=M_c\cdot\frac{\partial^2 f}{\partial c^2}\;}
$$

这里的 $D$ 是**互扩散系数**（interdiffusion coefficient），$\dfrac{\partial^2 f}{\partial c^2}$ 称**热力学因子**（thermodynamic factor，常记 $\chi$）。

> 直观：$M_c$ 描述"原子在化学势梯度下走多快"，而真正驱动成分均匀化的是化学势梯度，
> 化学势梯度又由自由能曲率 $\partial^2 f/\partial c^2$ 把成分梯度"放大"出来。两者相乘才是扩散系数。

---

## 3. 本模型的热力学因子与各相扩散系数

体自由能 $f=f_L(c)\,[1-h(\phi)]+f_S(c)\,h(\phi)$（固定 $\phi$），故

$$
\chi(\phi)=\frac{\partial^2 f}{\partial c^2}
=\underbrace{\frac{\partial^2 f_L}{\partial c^2}}_{2\rho^2}\,(1-h)
+\underbrace{\frac{\partial^2 f_S}{\partial c^2}}_{2\rho_s^2}\,h .
$$

对 stage-3 抛物线自由能 $f_L=\rho^2(c-c_a)^2+\dots$、$f_S=\rho_s^2(c-c_s)^2+\dots$（T 项无 $c$ 依赖，不进二阶导）：

$$
D(\phi)=M_c\,\chi(\phi)=M_c\big[\,2\rho^2(1-h)+2\rho_s^2\,h\,\big].
$$

| 相 | 热力学因子 $\chi$ | 扩散系数 $D$ |
|---|---|---|
| 液相 $\phi=0$ | $2\rho^2$ | $D_L=2\rho^2 M_c$ |
| 固相 $\phi=1$ | $2\rho_s^2$ | $D_S=2\rho_s^2 M_c$ |

**注意：** 这里 $D$ 与 $c$ **无关**（抛物线自由能 → 曲率为常数）。真实 CALPHAD 自由能含熵项
$\sim RT/[c(1-c)]$，曲率在 $c\to0,1$ 处发散，那时 $D$ 会强烈依赖 $c$（见 §5）。

---

## 4. 与"晚期发散修复"的联系（为什么 $\partial f_S/\partial c$ 项重要）

若 $\mu$ 只取**单侧**（缺 $\partial f_S/\partial c\cdot h$，旧版 / 化学计量假设），则

$$
\chi_{\text{单侧}}=2\rho^2(1-h)\;\xrightarrow{\;\phi=1\;}\;0
\quad\Longrightarrow\quad
D_S=0 .
$$

固相里溶质扩散系数为零 ⇒ **退化抛物型**：$c$ 在固相被冻结，且界面交叉项无正则化 → 显式时间推进在长时间（$\sim$ 数万步）后**发散**。

补上 $\partial f_S/\partial c\cdot h$ 后：

$$
\chi=2\rho^2(1-h)+2\rho_s^2 h\;\xrightarrow{\;\phi=1\;}\;2\rho_s^2
\quad\Longrightarrow\quad
D_S=2\rho_s^2 M_c>0 ,
$$

固相扩散恢复正常、非退化 —— 这正是补该项能消除发散的本质（不仅是热力学驱动力的问题，更是**扩散算子不再退化**）。

> 代码对照：`GFA_binary` 用 `dfSdc.fetab` 查表得 $\partial f_S/\partial c$；
> `GFA_evo` 用配置 `rho_s`、`c_s` 解析算 $2\rho_s^2(c-c_s)$。两者效果一致。

---

## 5. 量纲（SI）与真实自由能

$$
[\,M_c\,]=\frac{\mathrm{m}^5}{\mathrm{J\cdot s}},\qquad
\Big[\frac{\partial^2 f}{\partial c^2}\Big]=\frac{\mathrm{J}}{\mathrm{m}^3}
\quad(f\ \text{为 J/m}^3,\ c\ \text{无量纲})
\quad\Longrightarrow\quad
[\,D\,]=\frac{\mathrm{m}^2}{\mathrm{s}}.\ \checkmark
$$

SI 的 Cu-Zr 算例取 $M_c=10^{-19}\,\mathrm{m^5/(J\cdot s)}$。该情形下 $f_L$ 含理想/正规溶液熵项，

$$
\frac{\partial^2 f_L}{\partial c^2}=\frac{RT}{V_m}\frac{1}{c(1-c)}+(\text{正规项}),
$$

在 $c\to0$ 或 $1$ 处发散，故 $D=M_c\,\partial^2 f/\partial c^2$ 在稀端急剧增大 —— 这也是显式 Euler 稳定步长在 $c\to0,1$ 处被压得很小的原因。

---

## 6. 数值示例（当前 stage-3 参数，非量纲）

`settings.jsonc`：`M_c = 5`，$\rho=\rho_s=\sqrt2$（$\rho^2=\rho_s^2=2$）。

$$
\chi=2\rho^2=4\ (\text{两相相同}),\qquad
D=M_c\,\chi=5\times4=\boxed{20}.
$$

即液相、固相扩散系数都是 $20$（非量纲），且与 $c$ 无关。

> 显式 Euler 的 CH 稳定限：$\mathrm{d}t<\dfrac{\mathrm{d}x^2}{4D}=\dfrac{1}{4\times20}=0.0125$；
> 当前 $\mathrm{d}t=0.001$，约 $12\times$ 余量。

---

## 7. 调参指南

- **想要某个目标互扩散系数 $D^\*$**：由 $D=M_c\,\chi$，取 $M_c=D^\*/\chi=D^\*/(2\rho^2)$。
  例：想要液相 $D_L=10$、当前 $\rho^2=2$ → $M_c=10/4=2.5$。
- **想让两相扩散不同**：调 $\rho_s\ne\rho$，则 $D_S/D_L=\rho_s^2/\rho^2$。
- **$M_c$ 只缩放扩散快慢，不改平衡相成分**（$c_a,c_s$ 决定平衡成分）、也不改界面厚度（那是 $\varepsilon^2,w_\phi$ 的事）。

---

### 符号对照

| 符号 | 含义 | 代码/配置 |
|---|---|---|
| $M_c$ | CH 迁移率（mobility） | `M_c` |
| $\mu$ | 化学势 $\partial f/\partial c$ | `mu`（辅助场） |
| $\chi=\partial^2 f/\partial c^2$ | 热力学因子 | 由 $f_L,f_S$ 推出 |
| $D=M_c\chi$ | 互扩散系数 | 派生量，非直接配置 |
| $\rho,\rho_s$ | $f_L,f_S$ 的 $c$-曲率平方根 | `rho`(表)、`rho_s`(GFA_evo 配置) |
| $c_a,c_s$ | $f_L,f_S$ 极小值成分 | `ca`(表)、`c_s`(GFA_evo 配置) |
