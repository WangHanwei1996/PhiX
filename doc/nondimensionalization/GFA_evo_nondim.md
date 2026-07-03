# GFA_evo 的无量纲化（Cu–Zr 二元玻璃形成：CH + 双序参量 AC）

> 求解器源文件：`applications/solvers/GFA_binary/GFA_evo.cu`
> 模型文档：`applications/solvers/GFA_binary/doc/modeling_stage6.md`
> 示例配置：`applications/solvers/GFA_binary/test/settings/settings.jsonc`
>
> 本文从代码 / stage-6 文档还原控制方程，做量纲分析，选取特征尺度，把三场耦合系统压成无量纲形式并导出全部无量纲群，最后给出量级与主导平衡分析。
>
> 与同目录的 [`CH_AC_2D_nondim_CHscale.md`](CH_AC_2D_nondim_CHscale.md)、[`CH_AC_nondim.md`](CH_AC_nondim.md) 相比，本模型多了：**查表（CALPHAD）自由能**、**Arrhenius 温度依赖迁移率**、**线性降温协议**、以及**第二个序参量 η（非晶相）与 φ–η 交叉耦合**。这些都会带来额外的无量纲数。

记号：能量密度量纲 $E_d\equiv E\,L^{-3}=M L^{-1}\Theta^0\,\mathsf{t}^{-2}$（J/m³，$\mathsf t$ 为时间、$\Theta$ 为温度）；$c,\phi,\eta$ 无量纲；带星号 $(\cdot)^*$ 为无量纲量；$h(x)=x^3(6x^2-15x+10)$、$g(x)=x^2(1-x)^2$。

---

## 0. 从代码还原控制方程

`GFA_evo.cu`（第 414–442 行）装配的连续控制方程（stage 6）：

**自由能密度**（晶体部分）
$$
f=f_L(c,T)\big[1-h(\phi)\big]+f_S(c,T)\,h(\phi)+w_\phi\,g(\phi)+\tfrac{\varepsilon^2}{2}|\nabla\phi|^2,
$$
其中 $f_L,f_S,\ \partial f_L/\partial c$ **查 CALPHAD 表**（`data/material_properties/Cu-Zr/`，单位 J/m³，已含 $G_m/V_m$）。

**化学势（辅助场，逐点、无 $|\nabla c|^2$ 能 ⇒ 无 $\nabla^2 c$ 项）**
$$
\mu=\frac{\partial f_L}{\partial c}\big[1-h(\phi)\big]+\underbrace{2\rho_s^2(c-c_s)}_{\partial f_S/\partial c}\,h(\phi)
$$
> 注：stage-6 文档把 $f_S$ 当化学计量相（无 $c$ 依赖），$\mu$ 只剩液相项；**代码**保留了可配置的固相曲率项 $\partial f_S/\partial c=2\rho_s^2(c-c_s)$（置 `rho_s=0` 即关闭）。本文按代码的完整形式处理。

**Cahn–Hilliard（守恒 $c$）**
$$
\frac{\partial c}{\partial t}=M_c\,\nabla^2\mu
$$

**Allen–Cahn（晶体序参量 $\phi$）**
$$
\frac{\partial \phi}{\partial t}=-M_\phi(T)\Big[\big(f_S-f_L-h(\eta)\,\Delta f^{A\to L}\big)h'(\phi)+w_\phi\,g'(\phi)+2w_{ex}\,\eta^2\phi-\varepsilon^2\nabla^2\phi\Big]
$$

**Allen–Cahn（非晶序参量 $\eta$）**
$$
\frac{\partial \eta}{\partial t}=-M_\eta\Big[\big(1-h(\phi)\big)h'(\eta)\,\Delta f^{A\to L}+w_\eta\,g'(\eta)+2w_{ex}\,\eta\phi^2-\beta^2\nabla^2\eta\Big]
$$

**温度相关项（空间均匀，仅随时间）**
$$
M_\phi(T)=M_\phi^{\rm pref}\exp\!\Big(-\frac{Q_\phi}{R_g T}\Big)\ \text{(Arrhenius)},\qquad
\Delta f^{A\to L}(T)=\frac{R_g T\ln(1+\alpha)}{V_m}\,f(\tau),\quad \tau=\frac{T}{T_g},
$$
$M_\eta=$ const；$f(\tau)$ 为 stage-6 分段多项式（`f_tau()`）。**线性降温协议** $T(t)=T_{\rm start}-\dot T\,t$（$\dot T=$ `cooling_rate`，钳制到表的 $T$ 范围）。

---

## 1. 物理量与量纲

| 符号 | 含义 | 量纲 |
|---|---|---|
| $c,\phi,\eta$ | 浓度 / 晶体序参量 / 非晶序参量 | $1$ |
| $c_s,\alpha$ | 固相极小成分 / 驱动力幅值 | $1$ |
| $f_L,f_S,\mu,\Delta f^{A\to L}$ | 自由能密度 / 化学势 / 非晶→液驱动 | $E_d$ |
| $w_\phi,w_\eta,w_{ex},\rho_s^2$ | 双阱势垒 / 交叉耦合 / 固相曲率 | $E_d$ |
| $\varepsilon^2,\beta^2$ | $\phi,\eta$ 梯度能系数 | $E_d L^2 = E L^{-1}$（J/m） |
| $M_c$ | CH 成分迁移率 | $L^5 E^{-1}\mathsf t^{-1}$（m⁵/(J·s)） |
| $M_\phi,M_\eta,M_\phi^{\rm pref}$ | AC 迁移率 | $L^3 E^{-1}\mathsf t^{-1}$（m³/(J·s)） |
| $T,T_g,T_{\rm start}$ | 温度 | $\Theta$ |
| $\dot T$ (`cooling_rate`) | 降温速率 | $\Theta\,\mathsf t^{-1}$ |
| $R_g$ | 气体常数 | $E\,N^{-1}\Theta^{-1}$（J/(mol·K)） |
| $V_m$ | 摩尔体积 | $L^3 N^{-1}$（m³/mol） |
| $Q_\phi$ | 激活能 | $E\,N^{-1}$（J/mol） |

---

## 2. 量纲一致性检查（逐项）

- **$\Delta f^{A\to L}=R_g T\ln(1+\alpha)f(\tau)/V_m$**：$R_g T\to E N^{-1}$，除 $V_m\,(L^3N^{-1})$ 得 $E L^{-3}=E_d$ ✓（与 $f_S-f_L$ 同量纲，可相加）。
- **$\mu$**：$\partial f_L/\partial c\to E_d$（$c$ 无量纲）；$2\rho_s^2(c-c_s)\to E_d$。均 $E_d$ ✓
- **CH**：$\partial c/\partial t\to\mathsf t^{-1}$；$M_c\nabla^2\mu\to(L^5E^{-1}\mathsf t^{-1})(L^{-2})(E_d)=\mathsf t^{-1}$ ✓
- **AC-$\phi$**：方括号每项 $E_d$（驱动 $E_d\cdot1$；$w_\phi g'\to E_d$；$w_{ex}\eta^2\phi\to E_d$；$\varepsilon^2\nabla^2\phi\to E_d L^2\cdot L^{-2}=E_d$）；$M_\phi\cdot E_d=(L^3E^{-1}\mathsf t^{-1})(E_d)=\mathsf t^{-1}$ ✓
- **AC-$\eta$**：同构，$\beta^2\nabla^2\eta\to E_d$，$M_\eta\cdot E_d=\mathsf t^{-1}$ ✓
- **Arrhenius指数** $Q_\phi/(R_g T)\to (EN^{-1})/(EN^{-1}\Theta^{-1}\cdot\Theta)=1$ 无量纲 ✓

全部一致 ✓

---

## 3. 特征尺度选择（默认：界面控制 + 化学能标度）

本模型有**两个内禀能量标度**——界面（势垒 $w_\phi$）与化学（CALPHAD 曲率 $\sim R_gT/V_m$）。让序参量方程最干净的选法：

| 尺度 | 取值 | 理由 |
|---|---|---|
| 长度 $\ell_0$ | $\ell_0=\sqrt{\varepsilon^2/w_\phi}$ | $\phi$ 场「梯度能 ↔ 双阱」平衡给出的**晶体界面宽度**（界面厚 $\delta_\phi=\sqrt2\,\ell_0$） |
| 界面能密度 $E^*$ | $E^*=w_\phi$ | 双阱垒高，AC 项的自然标度 |
| 化学能密度 $f_c^*$ | $f_c^*=\dfrac{R_g T_g}{V_m}$ | CALPHAD / 化学势的自然标度；使 $\mu$、$\Delta f^{A\to L}$ 为 $O(1)$ |
| 时间 $t_0$ | $t_0=\dfrac{1}{M_\phi^{\circ}\,w_\phi}$，$M_\phi^{\circ}\!\equiv\!M_\phi(T_g)$ | $\phi$ 的 AC 弛豫时间（用 $T_g$ 处迁移率作参考） |
| 温度 | $\theta=\dfrac{T}{T_g}$ | 约化温度（即文档 $f(\tau)$ 里的 $\tau$；为避免与时间冲突，本文记 $\theta$） |

无量纲变量：

$$
x=\ell_0 x^*,\quad t=t_0 t^*,\quad \nabla^2=\ell_0^{-2}\nabla^{*2},\quad \mu=f_c^*\,\mu^+,\quad \theta=T/T_g.
$$

把查表函数也无量纲化（均 $O(1)$）：
$$
\psi(c,\theta)\equiv\frac{\partial f_L/\partial c}{f_c^*},\qquad
\Delta g(c,\theta)\equiv\frac{f_S-f_L}{w_\phi}\ \text{(热力学驱动力)}.
$$

> **$f_c^*=R_gT_g/V_m$ 从哪来、为何与 $c$ 方程绑定。** CALPHAD 摩尔自由能 $G_m(c,T)$ 对成分的依赖主要来自**理想混合熵** $R_gT[(1-c)\ln(1-c)+c\ln c]$（外加同量级 ~kJ/mol 的参考项与过剩项）。其一阶导（化学势）$\partial G_m/\partial c\sim R_gT\ln\tfrac{c}{1-c}$、二阶导（曲率）$\partial^2 G_m/\partial c^2\sim R_gT/[c(1-c)]$ —— **都 $\sim R_gT$ 每摩尔**（同一 $R_gT$ 熵项的导数）。除以 $V_m$ 化为能量密度即 $f_c^*=R_gT/V_m$（取 $T_g$ 定参考），$\approx5.5\times10^8$ J/m³。代码里 $\mu$ 由查表的 $\partial f_L/\partial c$ 搭起、$\Delta f^{A\to L}$ 也含 $R_gT/V_m$，故它是 $\mu$ 的自然标度。
> 它**只跟 $c$ 方程绑定**，因为 CH 方程 $\partial_t c=M_c\nabla^2\mu$ 的唯一作用对象就是 $\mu$：无量纲化 $\mu=f_c^*\mu^+$ 后，乘积 $M_c f_c^*$ 的量纲恰是 m²/s —— 一个**扩散系数**。把 CH 写成 Fick 形式 $\partial_t c=\nabla\!\cdot\!(D_{\rm chem}\nabla c)+\cdots$ 即 $D_{\rm chem}=M_c\,\partial^2 f_L/\partial c^2\sim M_c f_c^*$。所以 $f_c^*$ 是把"迁移率 $M_c$"翻译成"菲克扩散系数 $D$"的桥（自由能曲率），也是为何 $c$ 方程的控制数是 $\mathcal D=M_c f_c^* t_0/\ell_0^2$ 而非 $M_c$ 单独。（严格说 $\mu$ 标度用一阶导、$D$ 用二阶导，二者差 $O(1)$ 因子 $1/[c(1-c)]$，能共用是因都源自那个 $R_gT$ 熵项。）

---

## 4. 逐方程推导

**化学势**（按 $f_c^*$ 标度，纯逐点）：
$$
\mu^+=\frac{\mu}{f_c^*}=\psi(c,\theta)\,[1-h(\phi)]+2R_s\,(c-c_s)\,h(\phi),\qquad R_s\equiv\frac{\rho_s^2}{f_c^*}.
$$

**Cahn–Hilliard**：$\tfrac1{t_0}\partial_{t^*}c=M_c\ell_0^{-2}\nabla^{*2}(f_c^*\mu^+)$，整理
$$
\boxed{\ \partial_{t^*}c=\mathcal D\,\nabla^{*2}\mu^+\ },\qquad
\mathcal D\equiv\frac{M_c f_c^*\,t_0}{\ell_0^2}=\frac{M_c f_c^*}{M_\phi^{\circ}\varepsilon^2}.
$$

**AC-$\phi$**：乘 $t_0=1/(M_\phi^\circ w_\phi)$、逐项除 $w_\phi$，并用 $\varepsilon^2\nabla^2/w_\phi=\ell_0^{-2}\ell_0^2\nabla^{*2}=\nabla^{*2}$：
$$
\boxed{\ \partial_{t^*}\phi=-A_\phi(\theta)\Big[\big(\Delta g-h(\eta)\,\delta_{AL}\big)h'(\phi)+g'(\phi)+2W_{ex}\,\eta^2\phi-\nabla^{*2}\phi\Big]\ }
$$

**AC-$\eta$**：乘 $t_0$、除 $w_\phi$，$\beta^2\nabla^2/w_\phi=(\beta^2/\varepsilon^2)\nabla^{*2}$：
$$
\boxed{\ \partial_{t^*}\eta=-\mathcal M\Big[\big(1-h(\phi)\big)h'(\eta)\,\delta_{AL}+W_\eta\,g'(\eta)+2W_{ex}\,\eta\phi^2-K_g\nabla^{*2}\eta\Big]\ }
$$

**降温协议**：$T=T_g\theta$、$t=t_0t^*$ ⇒
$$
\boxed{\ \theta(t^*)=\theta_{\rm start}-\mathcal C\,t^*\ },\qquad \mathcal C\equiv\frac{\dot T\,t_0}{T_g}=\frac{\dot T}{M_\phi^\circ w_\phi T_g}.
$$

其中出现的无量纲量：
$$
\delta_{AL}(\theta)=\frac{\Delta f^{A\to L}}{w_\phi}=\mathcal W^{-1}\,\theta\ln(1+\alpha)\,f(\theta),\quad
A_\phi(\theta)=\frac{M_\phi(T)}{M_\phi^\circ}=\exp\!\Big[-\Gamma\big(\tfrac1\theta-1\big)\Big],
$$
$$
\mathcal W=\frac{w_\phi}{f_c^*},\ \ W_{ex}=\frac{w_{ex}}{w_\phi},\ \ W_\eta=\frac{w_\eta}{w_\phi},\ \ K_g=\frac{\beta^2}{\varepsilon^2},\ \ \mathcal M=\frac{M_\eta}{M_\phi^\circ},\ \ \Gamma=\frac{Q_\phi}{R_g T_g}.
$$
（用 $R_gT/V_m=f_c^*\theta$ 推得 $\delta_{AL}=(f_c^*/w_\phi)\theta\ln(1+\alpha)f(\theta)=\mathcal W^{-1}\theta\ln(1+\alpha)f(\theta)$。）

---

## 5. 无量纲数汇总

把约 18 个有量纲常数压成下面这些无量纲群（无通用命名，按"何者之比"解读）：

| 群 | 表达式 | 物理意义（何者之比） | 极限行为 |
|---|---|---|---|
| 动力学比 $\mathcal D$ | $\dfrac{M_c f_c^*}{M_\phi^\circ\varepsilon^2}=\dfrac{t_{AC}}{t_{\rm diff}}$ | 化学互扩散速率 / $\phi$ 界面弛豫速率 | $\gg1$：扩散极快 → 界面局部平衡（KKS 型、扩散控制）；$\ll1$：溶质截留、界面控制 |
| AC 迁移率比 $\mathcal M$ | $\dfrac{M_\eta}{M_\phi^\circ}$ | $\eta$ 弛豫速率 / $\phi$ 弛豫速率 | $\mathcal M{=}0$：$\eta$ 冻结（committed config 即此）；$\gg1$：非晶序参量优先响应 |
| 梯度比 $K_g$ | $\dfrac{\beta^2}{\varepsilon^2}=(\delta_\eta/\delta_\phi)^2$ | $\eta$ 界面宽² / $\phi$ 界面宽² | $\gg1$：$\eta$ 界面远宽于 $\phi$（需更粗网格才分辨，或欠分辨发散） |
| 势垒比 $W_\eta$ | $\dfrac{w_\eta}{w_\phi}$ | $\eta$ 双阱 / $\phi$ 双阱 | 决定两序参量界面能之比 |
| 交叉耦合 $W_{ex}$ | $\dfrac{w_{ex}}{w_\phi}$ | 相斥惩罚 / 双阱垒高 | $\sim O(1)$ 才能保证 $\phi,\eta$ 互斥（不共存）；$\ll1$ 耦合可忽略 |
| 标度分离数 $\mathcal W$ | $\dfrac{w_\phi}{f_c^*}=\dfrac{w_\phi V_m}{R_g T_g}$ | 界面势垒能 / 化学自由能标度 | $\sim O(1)$：界面能与化学驱动同量级（**良好标定区**）；$\ll1$：化学项压倒界面项 |
| 固相曲率 $R_s$ | $\dfrac{\rho_s^2}{f_c^*}$ | 固相 $c$-曲率 / 化学标度 | 控制 $\mu$ 中固相项相对权重 |
| Arrhenius 数 $\Gamma$ | $\dfrac{Q_\phi}{R_g T_g}$ | 激活能 / 热能（$T_g$ 处） | $\gg1$：$M_\phi$ 对 $T$ 极敏感（降温迅速冻结结晶 → 玻璃化）；$=0$：迁移率与 $T$ 无关 |
| 降温数 $\mathcal C$ | $\dfrac{\dot T}{M_\phi^\circ w_\phi T_g}=\dfrac{t_{AC}}{t_{\rm cool}}$ | $\phi$ 弛豫时间 / 降温时间 | $\gg1$：淬火快于界面响应（易玻璃化）；$\ll1$：准等温（接近平衡结晶） |
| 幅值 $\alpha$ | $\ln(1+\alpha)$ | 非晶→液驱动幅值 | 进入 $\delta_{AL}$ |
| 约化初温 $\theta_{\rm start}$ | $T_{\rm start}/T_g$ | 初温 / 玻璃转变温度 | $>1$ 过热熔体起步 |

另有两个场依赖的无量纲驱动（查表 / 解析）：
$$
\Delta g(c,\theta)=\frac{f_S-f_L}{w_\phi}\ \text{（热力学驱动，过冷度的体现，符号随 }T\text{ 变）},\quad
\delta_{AL}(\theta)=\mathcal W^{-1}\theta\ln(1+\alpha)f(\theta).
$$
以及域上的 **Cahn 数** $\mathrm{Cn}=\ell_0/L_{\rm dom}$（界面宽 / 域尺寸，须 $\ll1$）。

**计数**：18 个有量纲常数 $\to$ 约 11 个无量纲数 $+\{\psi,\Delta g\}$ 两张约化查表 $+\{c_s,\theta_{\rm start}\}$。

---

## 5b. φ–η 竞争数（结晶 vs 玻璃化——本模型的灵魂）

§5 那些 φ–c 的比值（$\mathcal D,\mathcal W,R_s$）是必要的，但它们回答的是"成分扩散跟不跟得上界面"。真正回答 **"这团过冷液体是结晶还是成玻"** 的，是 φ（晶体）与 η（非晶）**争夺同一份液体**时的一组 φ↔η 比值。

> 这些竞争数不是新的独立自由度——它们是 §5 已有群（$A_\phi,\mathcal M,\mathcal C,\Delta g,\delta_{AL},K_g,W_\eta,W_{ex}$）**重新组合**成有物理意义的"竞争"形式。独立维数仍是 ~11；价值在于把"谁赢"显式化。

先定义两扇区各自的**体相转变速率**（量纲 $\mathsf t^{-1}$，即"驱动力 × 迁移率"）：
$$
\Gamma_\phi = M_\phi(T)\,|f_S-f_L|,\qquad \Gamma_\eta = M_\eta\,|\Delta f^{A\to L}|.
$$

| 竞争数 | 表达式 | 物理意义 | 判据 |
|---|---|---|---|
| 热力学竞争 $\mathcal R_{\rm th}$ | $\dfrac{f_S-f_L}{\Delta f^{A\to L}}=\dfrac{\Delta g}{\delta_{AL}}$ | 晶体 vs 非晶相对液体的**自由能优势之比**（纯热力学：约掉了 $w_\phi$ 及所有动力学/界面参数） | $|\mathcal R_{\rm th}|>1$ 晶体是更深的自由能阱；$<1$ 非晶可**逆转** φ 驱动（见下） |
| 动力学竞争 / **GFA 判别数** $\mathcal K$ | $\dfrac{\Gamma_\phi}{\Gamma_\eta}=\dfrac{M_\phi(T)}{M_\eta}\dfrac{|f_S-f_L|}{|\Delta f^{A\to L}|}=\dfrac{A_\phi(\theta)}{\mathcal M}\,|\mathcal R_{\rm th}|$ | 晶体生长速率 / 非晶生长速率 | $\mathcal K\gg1$ 结晶取胜；$\mathcal K\ll1$ **玻璃化** |
| 结晶 Damköhler $\mathrm{Da}_\phi$ | $\Gamma_\phi t_{\rm cool}=\dfrac{M_\phi(T)|f_S-f_L|\,T_g}{\dot T}=\dfrac{A_\phi|\Delta g|}{\mathcal C}$ | 结晶速率 / 降温速率（$t_{\rm cool}=T_g/\dot T$） | $\gg1$ 淬火中来得及结晶；$\ll1$ **冻结成玻璃** |
| 非晶 Damköhler $\mathrm{Da}_\eta$ | $\Gamma_\eta t_{\rm cool}=\dfrac{\mathcal M|\delta_{AL}|}{\mathcal C}$ | 非晶化速率 / 降温速率 | $\mathrm{Da}_\phi/\mathrm{Da}_\eta=\mathcal K$（两条 TTT 曲线之争） |
| 界面宽比 | $\dfrac{\delta_\eta}{\delta_\phi}=\sqrt{\dfrac{\beta^2/w_\eta}{\varepsilon^2/w_\phi}}=\sqrt{\dfrac{K_g}{W_\eta}}$ | η 界面宽 / φ 界面宽 | $\gg1$：η 界面更宽，网格分辨与显式稳定由它定 |
| 界面能比 | $\dfrac{\sigma_\eta}{\sigma_\phi}=\sqrt{\dfrac{\beta^2 w_\eta}{\varepsilon^2 w_\phi}}=\sqrt{K_g W_\eta}$ | η 界面能 / φ 界面能（$\sigma\sim\sqrt{\text{梯度}\times\text{垒}}$） | 决定哪个相更易成核/共格 |
| 双侧互斥 | $\Pi_{ex}^\phi=\dfrac{w_{ex}}{w_\phi}=W_{ex}$；$\ \Pi_{ex}^\eta=\dfrac{w_{ex}}{w_\eta}=\dfrac{W_{ex}}{W_\eta}$ | 交叉惩罚相对**各自**势垒 | **两个都 $\gtrsim1$** 才真互斥；只一个大则便宜的那相容忍对方 |
| 弛豫时间比 | $\dfrac{t_\eta}{t_\phi}=\dfrac{M_\phi^\circ w_\phi}{M_\eta w_\eta}=\dfrac{1}{\mathcal M W_\eta}$ | η 界面弛豫 / φ 界面弛豫（**界面动力学**，区别于上面的体相转变速率） | 设定两序参量谁先把界面铺开 |

**三处单看 $W_{ex}$ 看不出的耦合结构：**

1. **$\mathcal R_{\rm th}$ 是非晶能否"刹住"结晶的阀门。** φ 的晶体驱动 $\Delta g-h(\eta)\delta_{AL}$ 在非晶充分形成（$\eta\to1,\ h\to1$）时降为 $\Delta g-\delta_{AL}=\Delta g\,(1-\mathcal R_{\rm th}^{-1})$。于是 $\mathcal R_{\rm th}<1$ 时非晶的存在直接把 φ 驱动力**翻号**（晶体回熔，非晶屏蔽结晶）；$\mathcal R_{\rm th}>1$ 时只削弱不逆转。这是 η→φ 的单向抑制。

2. **互斥是 φ↔η 双向的。** η 方程的 $(1-h(\phi))$：晶体一出现就掐断非晶驱动；φ 方程的 $-h(\eta)\delta_{AL}$：非晶削弱晶体驱动。"赢家通吃"的强度由热力学侧 $\mathcal R_{\rm th}$ 与梯度耦合侧 $\Pi_{ex}^\phi,\Pi_{ex}^\eta$ **共同**设定，缺一不可。

3. **GFA 判据链。** $\mathcal K=\dfrac{A_\phi(\theta)}{\mathcal M}\,|\mathcal R_{\rm th}|$ 把三件事串起来：降温经 $A_\phi=e^{-\Gamma(1/\theta-1)}$ 让 $\mathcal K$ 随过冷**骤降**（大 $\Gamma$ → 迁移率冻结），$\mathcal M$ 设 η 的相对快慢，$\mathcal R_{\rm th}$ 给热力学偏好。**临界冷却速率**对应 $\mathrm{Da}_\phi\!\sim\!1$，即 $\dot T_c\sim M_\phi|f_S-f_L|T_g$——低于它结晶、高于它成玻。

---

## 6. 主导平衡分析 + 配置诊断

### 6.1 两个能量标度与 $\mathcal W$

本模型的核心张力是 **界面能标度 $w_\phi,\varepsilon^2$** 与 **化学能标度 $f_c^*=R_gT_g/V_m$** 之比 $\mathcal W=w_\phi/f_c^*$。
$$
f_c^*=\frac{R_g T_g}{V_m}=\frac{8.314\times700}{1.058\times10^{-5}}\approx5.5\times10^{8}\ \text{J/m}^3 .
$$
金属 ~1 nm 界面的物理势垒约 $w_\phi\sim\sigma/\delta\sim0.2/10^{-9}\sim2\times10^8$ J/m³，故 $\mathcal W\sim O(1)$ 才是**良好标定区**：界面能与化学驱动同量级、界面被分辨、$\delta_{AL},\Delta g,R_s$ 均 $O(1)$。

### 6.2 ⚠️ committed `settings.jsonc` 的诊断

无量纲化正好暴露了示例配置的几处问题——这也是无量纲化的诊断价值所在：

1. **`w_phi=4.41e2` 几乎肯定是 `4.41e8` 的笔误（差 $10^6$）。** 用界面宽校验：
   $$
   \delta_\phi=\sqrt{2\varepsilon^2/w_\phi},\quad \varepsilon^2=0.7\times10^{-9}\ \text{J/m}.
   $$
   - 取 $w_\phi=441$：$\delta_\phi\approx1.78\ \mu$m，$\mathrm{Cn}=\ell_0/L_{\rm dom}\approx1.26\mu\text{m}/120\text{nm}\approx10$ —— 界面比 120 nm 的域还宽 $10$ 倍，物理上不可能、完全未分辨。
   - 取 $w_\phi=4.41\times10^8$：$\ell_0=\sqrt{\varepsilon^2/w_\phi}\approx1.26$ nm，$\delta_\phi\approx1.78$ nm $\approx3$ 格（$dx{=}0.6$ nm），$\mathrm{Cn}\approx0.011\ll1$ ✓ —— 正与配置注释「$\delta\approx1.85$ nm $\approx3$ 格」吻合，且 $\mathcal W\approx0.8=O(1)$。

   → 结论：注释描述的物理意图对应 $w_\phi\approx4.4\times10^8$，但键值写成了 $4.41\times10^2$。本文后续量级均按**修正值** $w_\phi=4.41\times10^8$。

2. **这是一个「降维 / 调试」配置**，并非完整三场物理场景：
   - `M_eta=0` ⇒ $\mathcal M=0$ ⇒ **$\eta$ 被完全冻结**，实际只跑 $c$–$\phi$ 子系统；
   - `cooling_rate=0` ⇒ $\mathcal C=0$ ⇒ **等温**（无淬火）；
   - `Q_phi=0` ⇒ $\Gamma=0$ ⇒ $A_\phi\equiv1$，**迁移率与温度无关**；
   - `rho_s` 注释标「占位待调」，`w_eta=10, w_ex=100, beta_sq=1e-4` 在 $w_\phi\sim10^8$ 标度下给出 $W_\eta,W_{ex}\sim10^{-7}$、$K_g\sim10^5$（$\eta$ 界面达毫米级）——均为**未标定占位值**。要真正激活 stage-6 的 $\eta$ 物理，需把 $w_\eta,w_{ex}$ 提到与 $w_\phi$ 同量级、$\beta^2$ 与 $\varepsilon^2$ 同量级，并给 $M_\eta>0$。

3. **$M_c$ 量级偏大**：$M_c=10^{-9}$ 配 $f_c^*\sim5\times10^8$ 给有效扩散 $D\sim M_c f_c^*\sim0.5$ m²/s（液态实际 $\sim10^{-9}$ m²/s），使 $\mathcal D\ggg1$（形式上落在严格扩散控制 / 局部平衡极限）。这与配置注释里基于 $D\sim10^{-9}$ 的 CH 稳定性估计自相矛盾，宜复核 $M_c$。

### 6.3 物理主导平衡（修正标定下）

- **结晶 vs 玻璃化的竞争**（详见 §5b）集中在 GFA 判别数 $\mathcal K=\tfrac{A_\phi}{\mathcal M}|\mathcal R_{\rm th}|$ 与结晶 Damköhler $\mathrm{Da}_\phi=A_\phi|\Delta g|/\mathcal C$：$\mathrm{Da}_\phi\gg1$ 淬火中结晶取胜，$\mathrm{Da}_\phi\ll1$（大 $\Gamma$、大 $\mathcal C$ 联合压制 φ 通道）冻结成玻璃，临界冷却速率即 $\mathrm{Da}_\phi\!\sim\!1\Rightarrow\dot T_c\sim M_\phi|f_S-f_L|T_g$。非晶能否**逆转**（而非仅削弱）结晶则看 $\mathcal R_{\rm th}\lessgtr1$。
- **扩散 vs 界面**：$\mathcal D\gg1$ 时界面始终处局部平衡（KKS 极限），相变速率由 $M_\phi$ 限制；$\mathcal D\ll1$ 出现溶质截留。
- **$\phi$–$\eta$ 互斥**：仅当 $W_{ex}=w_{ex}/w_\phi\sim O(1)$，交叉项 $2W_{ex}\eta^2\phi$ 才足以阻止两序参量在同一点共存。
- **显式步长**：本求解器显式 Euler。CH 此处**无 $|\nabla c|^2$ 能 ⇒ 是二阶非线性扩散**（非四阶），稳定限 $\Delta t^*\lesssim C\Delta x^{*2}/\mathcal D$；AC 受 $\Delta t^*\lesssim C\Delta x^{*2}$ 与 $K_g$ 约束。$\mathcal D\gg1$（快扩散）是最苛刻的限制者。

---

## 7. 无量纲方程组（最终形式）

$$
\mu^+=\psi(c,\theta)[1-h(\phi)]+2R_s(c-c_s)h(\phi)
$$
$$
\partial_{t^*}c=\mathcal D\,\nabla^{*2}\mu^+
$$
$$
\partial_{t^*}\phi=-A_\phi(\theta)\big[(\Delta g-h(\eta)\delta_{AL})h'(\phi)+g'(\phi)+2W_{ex}\eta^2\phi-\nabla^{*2}\phi\big]
$$
$$
\partial_{t^*}\eta=-\mathcal M\big[(1-h(\phi))h'(\eta)\delta_{AL}+W_\eta g'(\eta)+2W_{ex}\eta\phi^2-K_g\nabla^{*2}\eta\big]
$$
$$
\theta=\theta_{\rm start}-\mathcal C\,t^*,\quad
\delta_{AL}=\mathcal W^{-1}\theta\ln(1+\alpha)f(\theta),\quad
A_\phi=e^{-\Gamma(1/\theta-1)}.
$$

控制参数：$\big(\mathcal D,\ \mathcal M,\ K_g,\ W_\eta,\ W_{ex},\ \mathcal W,\ R_s,\ \Gamma,\ \mathcal C,\ \alpha,\ \theta_{\rm start},\ c_s\big)$ + 约化查表 $\psi,\Delta g$。

---

## 8. 备选特征尺度

| 换什么尺度 | 影响 | 何时更合适 |
|---|---|---|
| 能量统一用 $E^*=f_c^*$（弃 $w_\phi$） | 双阱与梯度项带系数 $\mathcal W$；$\mu^+$ 仍 $O(1)$，序参量项变 $O(\mathcal W)$ | 化学驱动主导、界面是薄层修正时（强标度分离分析） |
| 时间用 CH 扩散时间 $t_0=\ell_0^2/(M_c f_c^*)$ | CH 系数归一，$\phi/\eta$ 项各带 $1/\mathcal D$ 型系数 | 想精细解析成分扩散（扩散控制粗化）时 |
| 长度用 $\eta$ 界面宽 $\ell_0=\sqrt{\beta^2/w_\eta}$ | $\eta$ 梯度项归一，$\phi$ 侧带 $1/K_g$ | 当 $\eta$ 界面是控制最薄结构时 |
| 长度用域尺寸 $L_{\rm dom}$ | $\mathrm{Cn}^2=\ell_0^2/L_{\rm dom}^2\ll1$ 作小参数 | 锐界面 / 匹配渐近分析 |
| Arrhenius 参考温取 $T_{\rm start}$ 而非 $T_g$ | $A_\phi=e^{-\Gamma(T_g/T_{\rm start})(1/\theta-1/\theta_{\rm start})}$，$A_\phi(\theta_{\rm start}){=}1$ | 以初始（过热）态为基准跟踪迁移率衰减时 |

---

## 9. 三场耦合的线性稳定性与色散关系

把无量纲系统（§7，省略星号）在均匀基态 $(\bar c,\bar\phi,\bar\eta)$ 上线性化，扰动 $\propto e^{i\mathbf k\cdot\mathbf x+\omega t}$，$\nabla^2\to-k^2$。温度视为**冻结参数**（准静态降温，要求 $\mathcal C\ll|\omega|$；否则 $\theta(t)$ 当慢漂移另算）。得 $\omega\hat{\mathbf u}=\mathbf J(k)\hat{\mathbf u}$，$\hat{\mathbf u}=(\hat c,\hat\phi,\hat\eta)^T$：

$$
\mathbf J(k)=\begin{pmatrix}
-\mathcal D k^2\mu_c & -\mathcal D k^2\mu_\phi & 0\\[2pt]
-A_\phi\mathcal W^{-1}\mu_\phi & -A_\phi(S_\phi+k^2) & -A_\phi C_{\phi\eta}\\[2pt]
0 & -\mathcal M C_{\phi\eta} & -\mathcal M(S_\eta+K_g k^2)
\end{pmatrix}
$$

基态系数（$h'=30\phi^2(1-\phi)^2$，$h''=60\phi(2\phi-1)(\phi-1)$，$g''=2-12\phi+12\phi^2$）：
$$
\mu_c=\psi_c(1-\bar h_\phi)+2R_s\bar h_\phi,\qquad
\mu_\phi=h'(\bar\phi)\big[2R_s(\bar c-c_s)-\psi\big],
$$
$$
S_\phi=(\Delta g-\bar h_\eta\delta_{AL})h''(\bar\phi)+g''(\bar\phi)+2W_{ex}\bar\eta^2,\quad
S_\eta=(1-\bar h_\phi)h''(\bar\eta)\delta_{AL}+W_\eta g''(\bar\eta)+2W_{ex}\bar\phi^2,
$$
$$
C_{\phi\eta}=F_{\phi\eta}=F_{\eta\phi}=-h'(\bar\phi)h'(\bar\eta)\delta_{AL}+4W_{ex}\bar\phi\bar\eta,\qquad F_{\phi c}=\mathcal W^{-1}\mu_\phi .
$$

### 9.1 不解就能读出的四件事

1. **守恒 vs 非守恒**：c 行整行 $\propto k^2$（守恒 → CH 型，$\omega\to0$ 当 $k\to0$）；φ、η 行对角在 $k=0$ 有限（非守恒 → AC 型）。梯度项 $+k^2,+K_g k^2$ **恒为稳定化（耗散）**。
2. **零元 $J_{13}=J_{31}=0$**：c 不直接耦合 η（η 不进 μ）。**c 只通过 φ 间接感受 η**；φ 均匀处 c 与 η 脱钩——耦合只活在 φ 界面上。
3. **对称/互易**（皆源自共享自由能）：$C_{\phi\eta}$ 对称（来自公共项 $-\delta_{AL}h(\phi)h(\eta)$ 与 $W_{ex}\phi^2\eta^2$）；$F_{\phi c}=\mathcal W^{-1}\mu_\phi$ 互为倒易，只差标度比 $\mathcal W^{-1}$。且 $\mu_\phi\propto h'(\bar\phi)$ ⇒ **c–φ 耦合只在界面（$0<\bar\phi<1$）活跃，体相为零**。
4. **梯度流 ⇒ 实谱 ⇒ 无振荡**。量纲系统是变分梯度流 $\partial_t u_i=-\Lambda_i\delta F/\delta u_i$，$\Lambda=\mathrm{diag}(\mathcal D k^2,A_\phi,\mathcal M)\succ0$，$\omega=-\Lambda\mathcal H$（$\mathcal H$ 对称）⇒ **$\omega(k)\in\mathbb R$**：只有静态增长/衰减，**无行波/振荡失稳**（失稳是 spinodal 型驻定花样，不是波）。

### 9.2 通道 A：均匀单相基态（$\bar\phi,\bar\eta\in\{0,1\}$）——三通道解耦

液体 $\bar\phi=\bar\eta=0$ 时 $h'(0)=h''(0)=0\Rightarrow\mu_\phi=C_{\phi\eta}=0$，矩阵对角化：
$$
\omega_c=-\mathcal D\psi_c(\bar c)k^2,\quad
\omega_\phi=-A_\phi(2+k^2),\quad
\omega_\eta=-\mathcal M(2W_\eta+K_g k^2).
$$
- $\omega_\phi(0),\omega_\eta(0)<0$：**φ、η 势垒保护、对小扰动稳定** → 结晶/非晶化是**成核控制**，非 spinodal。
- $\omega_c=-\mathcal D\psi_c k^2$：液相曲率 $\psi_c>0$（常态）→ 下坡扩散纯耗散；$\psi_c<0$（混溶隙）→ 上坡失稳。

> ⚠️ **隐患**：c 方程**无 $|\nabla c|^2$ 梯度能**（μ 纯逐点，无 $-\kappa_c\nabla^2 c$）。故 $\psi_c<0$ 时 $\omega_c=\mathcal D|\psi_c|k^2\to+\infty$（$k\to\infty$）——**随 $k$ 单调发散、无最不稳定波长、无短波截断（紫外灾变）**，与标准 CH 的 $+\kappa_c k^4$ 截断（见 [`CH_AC_nondim.md`](CH_AC_nondim.md)）相反；此时 spinodal 花样尺度只由网格决定。Cu–Zr 液相在该温区大概率凸（$\psi_c>0$），但进混溶隙则模型在 c 上病态，需补 $\kappa_c$。

### 9.3 通道 B：界面基态（$0<\bar\phi<1$）——耦合失稳与波长选择

界面处 $\mu_\phi\neq0$，c–φ 子块行列式
$$
\det{}_{c\phi}=\mathcal D A_\phi k^2\big[\mu_c(S_\phi+k^2)-\mathcal W^{-1}\mu_\phi^2\big].
$$
失稳判据 $\det_{c\phi}<0$：
$$
\boxed{\ \mu_c(S_\phi+k^2)<\mathcal W^{-1}\mu_\phi^2\ }
$$
- **耦合诱导 spinodal**：即便 $\mu_c>0$ 且 $S_\phi>0$，只要 $\mathcal W^{-1}\mu_\phi^2$ 够大即失稳（成分–有序耦合，相变伴随成分配分）；驱动 $\propto\mu_\phi^2$（两相成分差²）。
- LHS 随 $k^2$ 增 ⇒ 长波带 $k\in[0,k_{\rm marg}]$，$k_{\rm marg}^2=\mathcal W^{-1}\mu_\phi^2/\mu_c-S_\phi$。c 守恒（$\omega\to0$ 当 $k\to0$）、φ 梯度项在 $k_{\rm marg}$ 压回 ⇒ **最不稳定模在有限 $k^*\in(0,k_{\rm marg})$**。
- → **c 自身丢失的波长选择，由 φ 的梯度能 $\varepsilon^2$ 经耦合补回**：成分调制尺度 $1/k^*$ 继承自晶体界面宽 $\ell_0=\sqrt{\varepsilon^2/w_\phi}$，正是 §9.2 紫外灾变在有界面时被"治好"的机制。

φ–η 子块同理：$\det_{\phi\eta}=A_\phi\mathcal M[(S_\phi+k^2)(S_\eta+K_g k^2)-C_{\phi\eta}^2]$，$C_{\phi\eta}^2\geq0$ 分裂本征值——最不稳本征矢是 φ、η 组合，倾向由竞争数 $\mathcal K=\tfrac{A_\phi}{\mathcal M}|\mathcal R_{\rm th}|$（§5b）的赢家决定；两者皆有梯度截断，波长选择正常。

### 9.4 耗散层级 + 与降温的联系

稳定区里衰减最慢的是**守恒 c 模长波端** $\omega_c=-\mathcal D\psi_c k^2\to0$（限速步）；φ、η 模即便 $k=0$ 也以 $A_\phi S_\phi$、$\mathcal M S_\eta$ 有限衰减。**φ 通道所有增长/衰减率带前因子 $A_\phi(\theta)=e^{-\Gamma(1/\theta-1)}$**：降温时 $A_\phi\to0$，φ 色散支整体压平——**不是把不稳定模变稳，而是把增长率乘趋零的 $A_\phi$，使其在淬火时标内长不起来**（§5b 玻璃化判据的色散关系版本）。

---

## 10. 一句话小结

在 $\ell_0=\sqrt{\varepsilon^2/w_\phi}$、$E^*=w_\phi$、$f_c^*=R_gT_g/V_m$、$t_0=1/(M_\phi^\circ w_\phi)$、$\theta=T/T_g$ 标度下，GFA_evo 的三场耦合系统由约 **11 个无量纲数**控制。**模型的灵魂是 φ（晶体）与 η（非晶）争夺过冷液体的竞争**（§5b）：GFA 判别数 $\mathcal K=\tfrac{A_\phi}{\mathcal M}|\mathcal R_{\rm th}|$ 和结晶 Damköhler $\mathrm{Da}_\phi=A_\phi|\Delta g|/\mathcal C$ 决定结晶还是成玻，热力学竞争 $\mathcal R_{\rm th}=(f_S-f_L)/\Delta f^{A\to L}$ 决定非晶能否逆转结晶，双侧互斥 $\Pi_{ex}^\phi,\Pi_{ex}^\eta$ 决定两相是否真不共存。其余：$\mathcal D$ 决定扩散/界面控制，$\mathcal W$ 衡量界面能与化学驱动是否同量级。无量纲化同时暴露了 committed 配置的三处问题（$w_\phi$ 疑似差 $10^6$ 的笔误、$\eta$ 扇区被 `M_eta=0` 冻结的降维调试态、$M_c$ 量级偏大），见 §6.2。
