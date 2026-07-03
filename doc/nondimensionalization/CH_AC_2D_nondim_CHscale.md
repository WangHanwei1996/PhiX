# CH_AC_2D 的无量纲化（CH 扩散时间标度）

> 求解器源文件：`applications/solvers/Cahn-Hillard+Allen-Cahn_double-well/2D/CH_AC_2D.cu`
>
> 本文用 **CH 扩散为主时钟**的一套标度（$E^*=\rho^2$、$\ell=\sqrt{\kappa_c/\rho^2}$、$\tau=\ell^2/(M\rho^2)$）对该求解器做无量纲化。
> 这与同目录 [`CH_AC_nondim.md`](CH_AC_nondim.md) 互为补充——后者用 **AC 弛豫为主时钟**的标度（$E^*=w$、$\ell=\sqrt{\kappa_\eta/w}$、$\tau=1/(Lw)$），并把"改用 CH 扩散时间归一"列为待推方向（其 §9 第 1 条）；本文正是那条路线的完整展开。两套结果数学等价、互为倒数关系，区别只在突出哪个物理过程。

记 $h(\eta)=\eta^3(6\eta^2-15\eta+10)$，$h'(\eta)=30\eta^2(1-\eta)^2$；带星号 $(\cdot)^*$ 为无量纲量。

---

## 0. 从代码还原控制方程

`CH_AC_2D.cu`（第 93–123 行）的 `setRHS` 组装出的连续控制方程：

$$
\mu = 2\rho^2(c-c_a) + 2\rho^2(c_a-c_b)\,h(\eta) - \kappa_c\,\nabla^2 c
$$
$$
\frac{\partial c}{\partial t} = M\,\nabla^2\mu
$$
$$
\frac{\partial \eta}{\partial t} = -L\Big[\,30\rho^2\,\eta^2(1-\eta)^2(2c-c_a-c_b)(c_a-c_b) + 2w\,\eta(1-\eta)(1-2\eta) - \kappa_\eta\,\nabla^2\eta\,\Big]
$$

这是 split 形式的 Cahn–Hilliard（守恒浓度场 $c$）与非守恒 Allen–Cahn（序参量 $\eta$）耦合，对应自由能泛函

$$
F[c,\eta]=\int_\Omega\Big[\rho^2\big(c-c_{eq}(\eta)\big)^2 + w\,g(\eta) + \tfrac{\kappa_c}{2}|\nabla c|^2 + \tfrac{\kappa_\eta}{2}|\nabla\eta|^2\Big]\,\mathrm dV,
$$

其中 $c_{eq}(\eta)=c_a+(c_b-c_a)h(\eta)$、$g(\eta)=\eta^2(1-\eta)^2$，且 $\mu=\delta F/\delta c$、AC 右端 $=-L\,\delta F/\delta\eta$。（自由能的逐项反推见 `CH_AC_nondim.md` §1。）

---

## 1. 物理量与量纲

记能量密度量纲 $E_d \equiv E\,L^{-3} = M L^{-1}T^{-2}$（J/m³）。$c,\eta,c_a,c_b$ 均无量纲（摩尔分数 / 序参量），无量纲化**不缩放它们本身**。

| 符号 | 含义 | 量纲 |
|---|---|---|
| $c,\eta$ | 浓度 / 序参量 | $1$ |
| $c_a,c_b$ | 两相平衡浓度 | $1$ |
| $\rho^2$ | 化学自由能曲率（势阱深度系数） | $E_d = M L^{-1}T^{-2}$ |
| $w$ | 双阱势垒高度 | $E_d = M L^{-1}T^{-2}$ |
| $\kappa_c$ | $c$ 的梯度能系数 | $E_d L^2 = M L\,T^{-2}$ |
| $\kappa_\eta$ | $\eta$ 的梯度能系数 | $E_d L^2 = M L\,T^{-2}$ |
| $\mu$ | 化学势（能量密度） | $E_d$ |
| $M$ | CH 迁移率 | $L^5 E^{-1}T^{-1} = M^{-1}L^4 T$ |
| $L$ | AC 动力学系数 | $L^3 E^{-1}T^{-1} = M^{-1}L\,T$ |

---

## 2. 量纲一致性检查（逐项）

**$\mu$ 方程**（各项须为 $E_d$）：$2\rho^2(c-c_a)\to E_d$；$2\rho^2(c_a-c_b)h\to E_d$；$\kappa_c\nabla^2 c\to E_d L^2\cdot L^{-2}=E_d$。三项均 $E_d$，一致 ✓

**CH 方程**（须为 $T^{-1}$）：$\partial c/\partial t\to T^{-1}$；$M\nabla^2\mu\to (L^5E^{-1}T^{-1})(L^{-2})(E_d)=T^{-1}$。一致 ✓

**AC 方程**（须为 $T^{-1}$）：方括号内每项均 $E_d$（双阱 $w\cdot$无量纲；$\kappa_\eta\nabla^2\eta\to E_d$；化学驱动 $\rho^2\cdot$无量纲）；$L\cdot E_d=(L^3E^{-1}T^{-1})(EL^{-3})=T^{-1}$。一致 ✓

---

## 3. 特征尺度选择（CH 标度）

| 尺度 | 取值 | 理由 |
|---|---|---|
| 长度 $\ell$ | $\ell=\sqrt{\kappa_c/\rho^2}$ | $\mu$ 方程中梯度能 ↔ 体自由能平衡给出的**化学扩散界面宽度**；使 CH 梯度项 $O(1)$ |
| 能量密度 $E^*$ | $E^*=\rho^2$ | 化学势 / 体自由能的自然标度，设 $\mu=\rho^2\mu^*$ |
| 时间 $\tau$ | $\tau=\dfrac{\ell^2}{M\rho^2}=\dfrac{\kappa_c}{M\rho^4}$ | **CH 扩散时间**（守恒场弛豫尺度），使 $\partial_{t^*}c$ 系数为 1 |

无量纲变量：

$$
x=\ell\,x^*,\qquad t=\tau\,t^*,\qquad \mu=\rho^2\,\mu^*,\qquad \nabla^2=\ell^{-2}\nabla^{*2}.
$$

---

## 4. 逐方程推导

**$\mu$ 方程**：代入后除以 $\rho^2$，

$$
\mu^* = 2(c-c_a)+2(c_a-c_b)h(\eta)-\underbrace{\frac{\kappa_c}{\rho^2\ell^2}}_{=\,1}\nabla^{*2}c
\qquad(\text{因 }\ell^2=\kappa_c/\rho^2).
$$

**CH 方程**：

$$
\frac{1}{\tau}\partial_{t^*}c=\frac{M\rho^2}{\ell^2}\nabla^{*2}\mu^*
\;\Longrightarrow\;
\partial_{t^*}c=\underbrace{\frac{M\rho^2\tau}{\ell^2}}_{=\,1}\nabla^{*2}\mu^*
\qquad(\text{因 }\tau=\ell^2/(M\rho^2)).
$$

**AC 方程**：乘 $\tau$ 后从方括号提出 $\rho^2$，

$$
\partial_{t^*}\eta=-\,\underbrace{L\rho^2\tau}_{R}\Big[30\,\eta^2(1-\eta)^2(2c-c_a-c_b)(c_a-c_b)+2\underbrace{\tfrac{w}{\rho^2}}_{\chi}\eta(1-\eta)(1-2\eta)-\underbrace{\tfrac{\kappa_\eta}{\rho^2\ell^2}}_{\kappa_\eta/\kappa_c}\nabla^{*2}\eta\Big].
$$

代入 $\ell^2=\kappa_c/\rho^2$、$\tau=\kappa_c/(M\rho^4)$：

$$
R=L\rho^2\tau=\frac{L\kappa_c}{M\rho^2},\qquad \frac{\kappa_\eta}{\rho^2\ell^2}=\frac{\kappa_\eta}{\kappa_c}.
$$

---

## 5. 无量纲方程组（最终形式）

$$
\boxed{\;\mu^* = 2(c-c_a)+2(c_a-c_b)\,h(\eta)-\nabla^{*2}c\;}
$$
$$
\boxed{\;\frac{\partial c}{\partial t^*}=\nabla^{*2}\mu^*\;}
$$
$$
\boxed{\;\frac{\partial \eta}{\partial t^*}=-R\Big[30\,\eta^2(1-\eta)^2(2c-c_a-c_b)(c_a-c_b)+2\chi\,\eta(1-\eta)(1-2\eta)-\frac{\kappa_\eta}{\kappa_c}\nabla^{*2}\eta\Big]\;}
$$

**8 个有量纲常数 $(\rho^2,w,\kappa_c,\kappa_\eta,M,L)+(c_a,c_b)$ → 3 个无量纲数 + 2 个成分参数。**

---

## 6. 无量纲数汇总

这类相场模型无通用命名（不像 Reynolds / Péclet），下表按"何者之比"给出物理含义：

| 群 | 表达式 | 物理意义（何者之比） | 极限行为 |
|---|---|---|---|
| 梯度能比 | $\dfrac{\kappa_\eta}{\kappa_c}$ | $\eta$ 界面梯度惩罚 / $c$ 界面梯度惩罚 $\sim(\eta$ 界面宽 $/c$ 界面宽$)^2$ | $\gg1$：$\eta$ 界面更宽，序参量过渡控制界面厚度；$\ll1$：浓度界面更宽 |
| 阱–化学比 $\chi$ | $\dfrac{w}{\rho^2}$ | 双阱势垒 / 化学自由能曲率 | $\gg1$：双阱刚硬，$\eta$ 被钉在 0/1，界面锐、相身份主导；$\ll1$：化学驱动易翻转 $\eta$ |
| 动力学比 $R$ | $\dfrac{L\kappa_c}{M\rho^2}=L\rho^2\tau=\dfrac{\tau_{CH}}{\tau_{AC}}$ | AC 界面弛豫速率 / CH 扩散速率 | $\gg1$：$\eta$ 弛豫极快，界面始终近局部平衡 → **扩散控制的 Ostwald 熟化（LSW）**；$\ll1$：界面动力学限速 |

> 与 AC 标度版的换算：$\chi_{\text{本文}}=w/\rho^2=1/\chi_{\text{AC版}}$，梯度能比 $=\kappa_\eta/\kappa_c=1/K$，$R=L\kappa_c/(M\rho^2)$；而 `CH_AC_nondim.md` 的派生量 $R_{\text{那里}}=\kappa_\eta L/(2M\rho^2)=1/(2\chi\mathcal M)$ 与本文 $R$ 同一量纲、差一个 $\kappa_\eta/\kappa_c$ 与常数因子。

此外组合 $\dfrac{\Delta c^2}{\chi}=\dfrac{\rho^2\Delta c^2}{w}$（$\Delta c=c_b-c_a$）衡量**化学耦合驱动 / 双阱势垒**：AC 体驱动 $\sim\rho^2\Delta c^2$、势垒 $\sim w$，决定相变能否克服势垒推进界面。

---

## 7. 主导平衡分析

- **Ostwald 熟化典型区**：物理上希望界面始终局部平衡、由长程扩散限速 → $R\gg1$（$\eta$ 快、$c$ 慢）。此时 AC 方括号 $\approx0$ 给出**界面局部平衡条件**（Gibbs–Thomson 曲率–过饱和关系），整体退化为**扩散控制粗化**，回到经典 LSW $\langle r\rangle^3\sim t$。代码把 CH 时间取作 $O(1)$（$\partial_{t^*}c=\nabla^{*2}\mu^*$）正契合"扩散为主时钟"的选择。

- **梯度项不可忽略**：$\mu^*$ 中 $-\nabla^{*2}c$ 系数已归一，它设定界面宽度、防止浓度间断。若改用域尺度 $\Lambda$，该系数变成小参数 $\varepsilon^2=\kappa_c/(\rho^2\Lambda^2)\ll1$ → 进入**锐界面极限**，体相内 $\mu^*\approx2(c-c_a)+2(c_a-c_b)h$，梯度项只在 $O(\varepsilon)$ 界面层起作用（匹配渐近推导 Gibbs–Thomson 的出发点）。

- **$\chi$ 与界面刚度**：$\chi\gg1$ 双阱主导 AC，$\eta$ 几乎处处 0 或 1，化学耦合只在界面薄层起作用；$\chi\lesssim\Delta c^2$ 时化学驱动可与势垒抗衡，界面可被过饱和"推动"乃至触发形核 / 相翻转。数值上 $\chi$ 过大 → 界面过硬，需更小 $dt$。

- **显式时间步约束**：本求解器显式推进。无量纲化后稳定性由两条扩散型 CFL 决定——CH 为四阶算子 $\partial_{t^*}c\sim\nabla^{*4}c$，要求 $\Delta t^*\lesssim C\,\Delta x^{*4}$（最苛刻）；AC 为二阶，$\Delta t^*\lesssim C\,\Delta x^{*2}/(R\,\kappa_\eta/\kappa_c)$。$R$ 越大 AC 越限速，故 $R\gg1$ 虽是想要的熟化区，却对显式步长最不利。

---

## 8. 备选特征尺度

| 换什么尺度 | 得到什么数 | 何时更合适 |
|---|---|---|
| 长度取 $\ell=\sqrt{\kappa_\eta/\rho^2}$（$\eta$ 界面宽） | 梯度能比 $\to1$，CH 梯度系数变 $\kappa_c/\kappa_\eta$ | $\eta$ 过渡层是控制界面厚度的那一个时 |
| 长度取域尺寸 $\Lambda$ | $\varepsilon^2=\kappa_c/(\rho^2\Lambda^2)\ll1$ 作为小参数 | 锐界面 / 匹配渐近分析，导出 Gibbs–Thomson |
| 能量密度取 $E^*=w$（而非 $\rho^2$） | 化学曲率以 $1/\chi=\rho^2/w$ 出现；得到 `CH_AC_nondim.md` 那套 $(\chi,K,\mathcal M)$ | 双阱势垒是主导能量标度时（强分凝） |
| 时间取 AC 弛豫时间 $\tau_{AC}=1/(L\rho^2)$ | AC 系数 $\to1$，CH 系数 $\to1/R$ | 想精细解析序参量界面动力学（界面控制而非扩散控制）时 |

---

## 9. 一句话小结

在 $\ell=\sqrt{\kappa_c/\rho^2}$、$\tau=\kappa_c/(M\rho^4)$、$\mu\sim\rho^2$ 标度下，该耦合 CH+AC 系统仅由 **3 个独立无量纲数**控制——梯度能比 $\kappa_\eta/\kappa_c$、阱–化学比 $w/\rho^2$、动力学比 $L\kappa_c/(M\rho^2)$——外加浓度差 $\Delta c=c_b-c_a$。Ostwald 熟化对应动力学比 $R\gg1$（界面快弛豫、扩散限速）的极限。
