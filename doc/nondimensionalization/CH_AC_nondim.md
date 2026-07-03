# Cahn–Hilliard + Allen–Cahn 双井模型的无量纲化

> 求解器源文件：`applications/solvers/Cahn-Hillard+Allen-Cahn_double-well/2D/CH_AC_2D.cu`
> 配置示例：`develop/CH+AC/settings/settings.jsonc`
>
> 本文从代码里的右端项**反推自由能泛函**，做量纲分析、选取特征尺度，把方程压成无量纲形式，最后代入示例参数给出具体的无量纲数与物理解读。
> 目标：把 8 个有量纲常数 $(\rho, w, \kappa_c, \kappa_\eta, M, L, c_\alpha, c_\beta)$ 压缩为 **3 个动力学无量纲数 + 2 个成分参数**。

记号：$\nabla^2$ 为 Laplacian，$\partial_t = \partial/\partial t$，带波浪号 $\tilde{\cdot}$ 的量为无量纲量。

---

## 0. 这是一个什么模型

一个**守恒场 $c$（浓度，Cahn–Hilliard）**与**非守恒序参量 $\eta$（相标识，Allen–Cahn）**耦合的相场模型，常用于 **Ostwald 熟化 / 析出长大**。

两个场都是无量纲的：$c\in[0,1]$ 是摩尔分数，$\eta\in[0,1]$ 标识两相（$\eta=0$ 基体相、$\eta=1$ 析出相）。**无量纲化不缩放 $c,\eta$ 本身**，只缩放空间、时间和能量密度。

---

## 1. 从代码反推自由能泛函

代码里实际装配的右端项（`CH_AC_2D.cu` 第 93–123 行）：

```
mu      = 2ρ²(c−c_α) + 2ρ²(c_α−c_β)·h(η) − κ_c·∇²c
dc/dt   = M·∇²mu
dη/dt   = −L[ 30ρ² η²(1−η)²(2c−c_α−c_β)(c_α−c_β) + 2w·η(1−η)(1−2η) − κ_η·∇²η ]
```

能复现这三行的、唯一自洽的自由能泛函是

$$
F[c,\eta]=\int_\Omega\Big\{\, f(c,\eta)\;+\;\tfrac{\kappa_c}{2}\,|\nabla c|^2\;+\;\tfrac{\kappa_\eta}{2}\,|\nabla\eta|^2 \,\Big\}\,\mathrm{d}V
$$

$$
f(c,\eta)=\rho^2\big[(c-c_\alpha)^2\big(1-h(\eta)\big)+(c-c_\beta)^2\,h(\eta)\big]\;+\;w\,g(\eta)
$$

其中**插值函数**与**双井势**分别为

$$
h(\eta)=\eta^3(6\eta^2-15\eta+10),\qquad h'(\eta)=30\,\eta^2(1-\eta)^2
$$

$$
g(\eta)=\eta^2(1-\eta)^2,\qquad g'(\eta)=2\,\eta(1-\eta)(1-2\eta)
$$

$h$ 满足 $h(0)=0,\,h(1)=1,\,h'(0)=h'(1)=0$，把 $f$ 在 $\eta=0$ 时切到「以 $c_\alpha$ 为底的抛物线」、在 $\eta=1$ 时切到「以 $c_\beta$ 为底的抛物线」。$g$ 是对称双井，两个井底 $\eta=0,1$，垒高 $w/16$（位于 $\eta=1/2$）。

### 1.1 验证：变分导数对回代码

**化学势**（对 $c$ 的变分导数）：

$$
\mu=\frac{\delta F}{\delta c}=\frac{\partial f}{\partial c}-\kappa_c\nabla^2 c
$$

$$
\frac{\partial f}{\partial c}=\rho^2\big[2(c-c_\alpha)(1-h)+2(c-c_\beta)h\big]
=2\rho^2(c-c_\alpha)+2\rho^2(c_\alpha-c_\beta)\,h(\eta)
$$

→ 正是代码里的 `mu`。✓

**Allen–Cahn 驱动力**（对 $\eta$ 的变分导数）：

$$
\frac{\partial f}{\partial \eta}=\rho^2 h'(\eta)\big[(c-c_\beta)^2-(c-c_\alpha)^2\big]+w\,g'(\eta)
$$

利用 $(c-c_\beta)^2-(c-c_\alpha)^2=(2c-c_\alpha-c_\beta)(c_\alpha-c_\beta)$：

$$
\frac{\partial f}{\partial \eta}=\rho^2\,h'(\eta)\,(2c-c_\alpha-c_\beta)(c_\alpha-c_\beta)+w\,g'(\eta)
$$

代入 $h'=30\eta^2(1-\eta)^2$、$g'=2\eta(1-\eta)(1-2\eta)$ → 正是代码里 `bulk + dw`。✓

### 1.2 控制方程

$$
\boxed{\;\mu=\frac{\partial f}{\partial c}-\kappa_c\nabla^2 c\;}
$$

$$
\boxed{\;\frac{\partial c}{\partial t}=M\,\nabla^2\mu\;}\qquad\text{(Cahn–Hilliard，守恒)}
$$

$$
\boxed{\;\frac{\partial \eta}{\partial t}=-L\Big(\frac{\partial f}{\partial \eta}-\kappa_\eta\nabla^2\eta\Big)\;}\qquad\text{(Allen–Cahn，非守恒)}
$$

CH 写成「散度的散度」$\partial_t c=\nabla\!\cdot\!(M\nabla\mu)$，所以 $c$ 守恒（$\int_\Omega c\,\mathrm{d}V$ 在周期/无通量边界下不变）；AC 没有这层散度结构，所以 $\eta$ 非守恒。

---

## 2. 量纲分析

设长度尺度 $\ell_0$、时间尺度 $t_0$、能量密度尺度 $E_0$（单位 J/m³）。因 $c,\eta$ 无量纲，逐项配平：

| 量 | 量纲 | 来源 |
|---|---|---|
| $f,\ \mu,\ \rho^2,\ w$ | $E_0$ | $f$ 是能量密度；$\mu=\partial f/\partial c$ 而 $c$ 无量纲 |
| $\kappa_c,\ \kappa_\eta$ | $E_0\,\ell_0^2$ | $\tfrac{\kappa}{2}|\nabla c|^2$ 须为 $E_0$，而 $|\nabla c|^2\sim\ell_0^{-2}$ |
| $M$ | $\ell_0^2\,E_0^{-1}\,t_0^{-1}$ | $\partial_t c=M\nabla^2\mu$：$t_0^{-1}=M\,E_0\,\ell_0^{-2}$ |
| $L$ | $E_0^{-1}\,t_0^{-1}$ | $\partial_t\eta=-L(\partial f/\partial\eta)$：$t_0^{-1}=L\,E_0$ |

> 提示：这里 $\mu$ 是「单位**体积**」的化学势（J/m³），不是「单位摩尔」（J/mol）。因为该模型把 $c$ 当无量纲摩尔分数，$M$ 也相应吸收了摩尔体积。

---

## 3. 选取特征尺度

三个尺度的选法不唯一；下面这组让无量纲方程**最干净**（双井系数归一、界面结构显式）：

$$
\boxed{\;E^*=w\;}\quad\text{能量密度：双井垒高} 
$$

$$
\boxed{\;\ell=\sqrt{\frac{\kappa_\eta}{w}}\;}\quad\text{长度：}\eta\text{ 场「梯度能↔双井」平衡给出的界面长度}
$$

$$
\boxed{\;\tau=\frac{1}{L\,w}\;}\quad\text{时间：Allen–Cahn 序参量弛豫时间}
$$

无量纲变量：

$$
\tilde x=\frac{x}{\ell},\quad \tilde t=\frac{t}{\tau},\quad \tilde\mu=\frac{\mu}{w},\quad \tilde\nabla=\ell\,\nabla,\quad \tilde\nabla^2=\ell^2\,\nabla^2
$$

关键替换关系（反复用到）：

$$
\nabla^2=\frac{1}{\ell^2}\tilde\nabla^2=\frac{w}{\kappa_\eta}\tilde\nabla^2,\qquad \frac{\kappa_\eta}{w}=\ell^2
$$

---

## 4. 逐方程无量纲化

### 4.1 Allen–Cahn

从

$$
\frac{\partial \eta}{\partial t}=-L\Big[\rho^2 h'(\eta)(2c-c_\alpha-c_\beta)(c_\alpha-c_\beta)+w\,g'(\eta)-\kappa_\eta\nabla^2\eta\Big]
$$

左边 $\partial_t\eta=\tfrac1\tau\partial_{\tilde t}\eta$；整条乘 $\tau=\tfrac{1}{Lw}$，并逐项除以 $w$：

- $\dfrac{\rho^2}{w}h'(\eta)(\cdots)$ → 出现 $\dfrac{\rho^2}{w}$
- $\dfrac{w}{w}g'(\eta)=g'(\eta)$ → 系数归一
- $\dfrac{\kappa_\eta}{w}\nabla^2\eta=\ell^2\cdot\dfrac{1}{\ell^2}\tilde\nabla^2\eta=\tilde\nabla^2\eta$ → 系数归一

得

$$
\boxed{\;\frac{\partial \eta}{\partial \tilde t}=\tilde\nabla^2\eta-\chi\,h'(\eta)\,(2c-c_\alpha-c_\beta)(c_\alpha-c_\beta)-g'(\eta)\;}
$$

### 4.2 化学势

$$
\tilde\mu=\frac{\mu}{w}=\frac{1}{w}\big[2\rho^2(c-c_\alpha)+2\rho^2(c_\alpha-c_\beta)h(\eta)\big]-\frac{\kappa_c}{w}\nabla^2 c
$$

其中 $\dfrac{\kappa_c}{w}\nabla^2 c=\dfrac{\kappa_c}{w}\cdot\dfrac{w}{\kappa_\eta}\tilde\nabla^2 c=\dfrac{\kappa_c}{\kappa_\eta}\tilde\nabla^2 c$。得

$$
\boxed{\;\tilde\mu=2\chi(c-c_\alpha)+2\chi(c_\alpha-c_\beta)h(\eta)-K\,\tilde\nabla^2 c\;}
$$

### 4.3 Cahn–Hilliard

$$
\frac{\partial c}{\partial t}=M\nabla^2\mu
$$

左边 $\tfrac1\tau\partial_{\tilde t}c$；右边 $M\nabla^2(w\tilde\mu)=M\cdot\dfrac{w}{\ell^2}\tilde\nabla^2(w\tilde\mu)$……逐步：

$$
\frac{1}{\tau}\partial_{\tilde t}c=M\,\frac{1}{\ell^2}\tilde\nabla^2(w\,\tilde\mu)=\frac{Mw}{\ell^2}\tilde\nabla^2\tilde\mu
$$

$$
\partial_{\tilde t}c=\tau\,\frac{Mw}{\ell^2}\tilde\nabla^2\tilde\mu=\frac{1}{Lw}\cdot\frac{Mw}{\kappa_\eta/w}\tilde\nabla^2\tilde\mu=\frac{Mw}{L\,\kappa_\eta}\tilde\nabla^2\tilde\mu
$$

得

$$
\boxed{\;\frac{\partial c}{\partial \tilde t}=\mathcal{M}\,\tilde\nabla^2\tilde\mu\;}
$$

---

## 5. 无量纲数汇总

推导中冒出来的 **3 个核心动力学无量纲数**：

| 无量纲数 | 定义 | 物理含义 |
|---|---|---|
| $\chi$ | $\dfrac{\rho^2}{w}$ | **化学驱动力 / 双井垒高**。浓度失配把 $\eta$ 推过界面的相对强度；$\chi$ 大→化学主导。 |
| $K$ | $\dfrac{\kappa_c}{\kappa_\eta}$ | **两个梯度能系数之比**，即 $c$ 界面与 $\eta$ 界面的相对厚度。 |
| $\mathcal{M}$ | $\dfrac{Mw}{L\,\kappa_\eta}$ | **CH 与 AC 的迁移率/时间尺度之比**。$\mathcal{M}$ 小→浓度扩散比序参量弛豫慢。 |

外加**纯成分/几何参数**：平衡浓度 $c_\alpha,c_\beta$（无量纲），失配 $\Delta c_{eq}=c_\beta-c_\alpha$，以及域上的 **Cahn 数** $\mathrm{Cn}=\ell/L_{\text{domain}}$（界面厚度 / 域尺寸，须 $\ll 1$ 才处于尖锐界面极限）。

一个有用的派生量 —— 把 CH 看成有效扩散 $D_{\rm eff}=M\,\partial^2 f/\partial c^2\approx 2M\rho^2$，则「界面长度上的扩散时间 / AC 弛豫时间」之比为

$$
R=\frac{\tau_{\rm CH}}{\tau_{\rm AC}}=\frac{\ell^2/D_{\rm eff}}{\tau}=\frac{\kappa_\eta L}{2M\rho^2}=\frac{1}{2\,\chi\,\mathcal{M}}
$$

$R\sim 1$ 表示两个过程强耦合（熟化想要的区间）；$R\ll1$ 浓度扩散很快、由界面动力学限速；$R\gg1$ 反之。

---

## 6. 无量纲方程组（最终形式）

$$
\tilde\mu=2\chi(c-c_\alpha)+2\chi(c_\alpha-c_\beta)\,h(\eta)-K\,\tilde\nabla^2 c
$$

$$
\frac{\partial c}{\partial \tilde t}=\mathcal{M}\,\tilde\nabla^2\tilde\mu
$$

$$
\frac{\partial \eta}{\partial \tilde t}=\tilde\nabla^2\eta-\chi\,h'(\eta)\,(2c-c_\alpha-c_\beta)(c_\alpha-c_\beta)-g'(\eta)
$$

$$
\chi=\frac{\rho^2}{w},\qquad K=\frac{\kappa_c}{\kappa_\eta},\qquad \mathcal{M}=\frac{Mw}{L\kappa_\eta}
$$

**8 个有量纲常数 → 3 个无量纲数 $(\chi,K,\mathcal{M})$ + 2 个成分参数 $(c_\alpha,c_\beta)$。**

---

## 7. 代入示例参数

`develop/CH+AC/settings/settings.jsonc`：
$\rho=\sqrt2\Rightarrow\rho^2=2$，$w=1$，$\kappa_c=\kappa_\eta=3$，$M=L=5$，$c_\alpha=0.3$，$c_\beta=0.7$，$\mathrm{d}x=1$，$\mathrm{d}t=0.001$，$n_x=200$。

| 量 | 计算 | 值 | 解读 |
|---|---|---|---|
| $\chi$ | $\rho^2/w=2/1$ | **2** | 化学驱动力是势垒的 2 倍，偏化学主导 |
| $K$ | $\kappa_c/\kappa_\eta=3/3$ | **1** | 两界面厚度同量级 |
| $\mathcal{M}$ | $Mw/(L\kappa_\eta)=5/(5\cdot3)$ | **1/3** | CH 比 AC 稍慢 |
| $R$ | $1/(2\chi\mathcal{M})=1/(4/3)$ | **0.75** | 扩散与序参量弛豫同量级 → 强耦合，适合熟化 |
| $\ell$ | $\sqrt{\kappa_\eta/w}=\sqrt3$ | **≈1.73 dx** | 界面长度尺度 |
| 界面宽度 | $2\sqrt6\,\ell^{-1}$ 取倒数 $=2\sqrt6$ | **≈4.9 cells** | 见 §8，约 5 格，分辨充分 ✓ |
| $\mathrm{Cn}$ | $\ell/(n_x\,\mathrm{d}x)=\sqrt3/200$ | **≈0.0087** | $\ll1$，尖锐界面极限 ✓ |
| $\Delta c_{eq}$ | $c_\beta-c_\alpha=0.7-0.3$ | **0.4** | 平衡两相成分差 |
| $\mathrm{d}\tilde t$ | $\mathrm{d}t/\tau=0.001/0.2$ | **0.005** | 无量纲步长 |
| $\tilde t_{\max}$ | $n_{\text{steps}}\mathrm{d}t/\tau=1000/0.2$ | **5000** | 总无量纲演化时间 |

---

## 8. 界面厚度与分辨率（顺带练手）

把 AC 方程关掉耦合、取定常一维剖面：$\tilde\nabla^2\eta=g'(\eta)$。首次积分（界面处 $\eta'\to0$）：

$$
\tfrac12\Big(\frac{\mathrm{d}\eta}{\mathrm{d}\tilde x}\Big)^2=g(\eta)=\eta^2(1-\eta)^2
\;\Rightarrow\;
\frac{\mathrm{d}\eta}{\mathrm{d}\tilde x}=-\sqrt2\,\eta(1-\eta)
$$

解得经典 tanh 剖面

$$
\eta(\tilde x)=\tfrac12\Big[1-\tanh\!\big(\tilde x/\sqrt2\big)\Big]
$$

最大斜率在 $\eta=1/2$：$\big|\mathrm{d}\eta/\mathrm{d}\tilde x\big|_{\max}=\tfrac{1}{2\sqrt2}=\tfrac{\sqrt2}{4}$。
以 $W=1/\max|\mathrm{d}\eta/\mathrm{d}x|$ 定义物理界面宽度：

$$
W=\frac{\ell}{\sqrt2/4}=2\sqrt2\,\ell=2\sqrt2\,\sqrt3=2\sqrt6\approx4.9\ \text{cells}
$$

约 5 个网格穿过界面 → **分辨充分**（相场一般要求 $\gtrsim4\!-\!5$ 格）。若把 $\kappa_\eta$ 调小或 $w$ 调大，$\ell=\sqrt{\kappa_\eta/w}$ 变小，界面会变薄甚至欠分辨，要同步加密网格。

---

## 9. 几点可继续推的方向

1. **时间尺度选择不唯一**。本文用 AC 弛豫时间 $\tau=1/(Lw)$ 归一，所以 $\mathcal{M}$ 落在 CH 方程上。若改用 CH 扩散时间归一，$1/\mathcal{M}$ 会落到 AC 方程、双井系数变 $1/\mathcal{M}$——两者等价，看你想突出哪个过程。
2. **能量尺度也可换**。选 $E^*=\rho^2$ 则 $\chi\to1/\chi$、化学项归一、双井带系数，适合「化学主导」分析。
3. **线性稳定性 / spinodal 波长**：把 $c=\bar c+\delta\hat c\,e^{i\tilde k\tilde x+\omega\tilde t}$、$\eta=\bar\eta+\delta\hat\eta\,e^{\cdots}$ 代入无量纲方程，线化后得 $2\times2$ 色散矩阵，最不稳定波数 $\tilde k^*$ 与 $\chi,K,\mathcal{M}$ 的依赖关系可解析给出——这是验证求解器的好基准。
