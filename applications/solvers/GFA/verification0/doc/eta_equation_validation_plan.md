# 仅 \(\eta\) 方程的验证方案

本文档用于第一阶段验证 Wang 与 Napolitano 相场-金属玻璃模型中的结构弛豫变量 \(\eta\)。本阶段只验证液体/玻璃结构变量 \(\eta\) 的演化方程，不引入晶体相 \(\phi_i\) 的演化，也不验证晶化竞争、临界冷却速率或相选择问题。

---

## 1. 验证目标

第一阶段的目标是证明程序能够正确描述非晶结构从液体态到玻璃态的连续弛豫过程。

在原模型中：

- \(\eta=0\)：液体态；
- \(\eta=1\)：玻璃态；
- \(\phi_i\)：多相场变量，用于区分液体/玻璃相和不同晶体相。

因此，若只验证 \(\eta\) 方程，应固定多相场变量为：

$$
\phi_0=1,\qquad \phi_1=\phi_2=\phi_3=0.
$$

这表示整个区域始终属于非晶/液体相，不允许晶体相出现。此时 \(W_{ex}\) 项自动消失，因为没有晶体相参与。

---

## 2. 只保留 \(\eta\) 方程后的自由能

在固定 \(\phi_0=1\)、\(\phi_{1,2,3}=0\) 的条件下，局部自由能密度可以简化为：

$$
f_\eta
=
h(\eta)\Delta f^{SR}(c,T)
+
w_\eta \eta^2(1-\eta)^2
+
\frac{\beta}{2}|\nabla \eta|^2.
$$

其中：

$$
h(\eta)=\eta^3(10-15\eta+6\eta^2),
$$

$$
\Delta f^{SR}(c,T)=f_0(c,T,1)-f_0(c,T,0).
$$

这里 \(\Delta f^{SR}\) 表示液体态与玻璃态之间的结构弛豫自由能差。

需要注意的是，若 \(\Delta f^{SR}<0\)，则 \(\eta=1\) 的玻璃态相对于 \(\eta=0\) 的液体态更稳定；若 \(\Delta f^{SR}>0\)，则液体态更稳定。

---

## 3. 变分导数

总自由能为：

$$
F[\eta]
=
\int_\Omega
\left[
h(\eta)\Delta f^{SR}
+
w_\eta\eta^2(1-\eta)^2
+
\frac{\beta}{2}|\nabla\eta|^2
\right]d\Omega.
$$

由泛函导数定义：

$$
\frac{\delta F}{\delta \eta}
=
\frac{\partial f_\eta}{\partial \eta}
-
\nabla\cdot
\left(
\frac{\partial f_\eta}{\partial(\nabla\eta)}
\right).
$$

首先：

$$
h'(\eta)=30\eta^2(1-\eta)^2.
$$

双势阱项的导数为：

$$
\frac{d}{d\eta}\left[\eta^2(1-\eta)^2\right]
=
2\eta(1-\eta)(1-2\eta).
$$

梯度项给出：

$$
\nabla\cdot
\left(
\frac{\partial}{\partial(\nabla\eta)}
\frac{\beta}{2}|\nabla\eta|^2
\right)
=
\beta\nabla^2\eta.
$$

因此：

$$
\frac{\delta F}{\delta \eta}
=
30\eta^2(1-\eta)^2\Delta f^{SR}
+
2w_\eta\eta(1-\eta)(1-2\eta)
-
\beta\nabla^2\eta.
$$

---

## 4. \(\eta\) 的 Allen-Cahn 演化方程

建议采用使自由能下降的符号约定：

$$
\frac{\partial\eta}{\partial t}
=
-L_\eta\frac{\delta F}{\delta\eta}.
$$

代入上式，得到：

$$
\frac{\partial \eta}{\partial t}
=
-L_\eta
\left[
30\eta^2(1-\eta)^2\Delta f^{SR}
+
2w_\eta\eta(1-\eta)(1-2\eta)
-
\beta\nabla^2\eta
\right].
$$

若要模拟噪声诱导的玻璃转变，可加入随机扰动：

$$
\frac{\partial \eta}{\partial t}
=
-L_\eta
\left[
30\eta^2(1-\eta)^2\Delta f^{SR}
+
2w_\eta\eta(1-\eta)(1-2\eta)
-
\beta\nabla^2\eta
\right]
+
\xi.
$$

其中 \(\xi\) 是高斯噪声项。

程序中要重点检查符号。如果代码写成：

$$
\partial_t\eta=+L_\eta\frac{\delta F}{\delta\eta},
$$

则必须确认代码中定义的 \(\delta F/\delta\eta\) 是否已经取了相反数。否则自由能会上升，而不是下降。

---

## 5. 验证一：0D 均匀场测试

### 5.1 测试目的

验证自由能驱动力方向、\(h'(\eta)\)、双势阱导数以及 Allen-Cahn 符号是否正确。

### 5.2 测试设置

忽略空间项和噪声：

$$
\nabla^2\eta=0,\qquad \xi=0.
$$

方程退化为常微分方程：

$$
\frac{d\eta}{dt}
=
-L_\eta
\left[
30\eta^2(1-\eta)^2\Delta f^{SR}
+
2w_\eta\eta(1-\eta)(1-2\eta)
\right].
$$

初始条件可取：

$$
\eta(t=0)=0.01,
$$

或

$$
\eta(t=0)=0.5.
$$

### 5.3 预期结果

当温度高于玻璃转变温度时，液体态更稳定，应有：

$$
\eta\rightarrow 0.
$$

当温度低于玻璃转变温度时，玻璃态更稳定，应有：

$$
\eta\rightarrow 1.
$$

### 5.4 通过标准

- \(T>T_g\) 时，\(\eta\) 最终回到 0 附近；
- \(T<T_g\) 时，\(\eta\) 最终趋近 1；
- 演化过程中 \(\eta\) 不应长期跑出 \([0,1]\) 区间；
- 如果无噪声情况下出现反向演化，优先检查 Allen-Cahn 符号和 \(\Delta f^{SR}\) 的符号。

---

## 6. 验证二：无噪声自由能单调下降

### 6.1 测试目的

验证程序是否真正实现了梯度流：

$$
\partial_t\eta=-L_\eta\frac{\delta F}{\delta\eta}.
$$

### 6.2 理论依据

在无噪声、固定温度、周期边界或齐次 Neumann 边界条件下：

$$
\frac{dF}{dt}
=
\int_\Omega
\frac{\delta F}{\delta\eta}
\frac{\partial\eta}{\partial t}
d\Omega.
$$

代入 Allen-Cahn 方程：

$$
\frac{dF}{dt}
=
-
\int_\Omega
L_\eta
\left(
\frac{\delta F}{\delta\eta}
\right)^2
d\Omega
\le 0.
$$

因此，无噪声情况下总自由能必须单调不增。

### 6.3 数值记录量

每一步记录：

$$
F^n
=
\int_\Omega
\left[
h(\eta^n)\Delta f^{SR}
+
w_\eta(\eta^n)^2(1-\eta^n)^2
+
\frac{\beta}{2}|\nabla\eta^n|^2
\right]d\Omega.
$$

离散实现中可以写成：

$$
F^n
\approx
\sum_{\mathbf{x}}
\left[
h(\eta^n)\Delta f^{SR}
+
w_\eta(\eta^n)^2(1-\eta^n)^2
+
\frac{\beta}{2}|\nabla_h\eta^n|^2
\right]\Delta V.
$$

### 6.4 通过标准

无噪声时应满足：

$$
F^{n+1}\le F^n.
$$

在显式 Euler 离散下，如果时间步过大，可能出现轻微上升。此时应减小 \(\Delta t\)。若在很小时间步下仍然系统性上升，通常说明：

1. Allen-Cahn 方程符号写反；
2. Laplacian 项符号写反；
3. \(\Delta f^{SR}\) 的单位或符号处理错误；
4. 边界条件或梯度能计算不一致。

---

## 7. 验证三：一维界面解析解

### 7.1 测试目的

验证梯度项、Laplacian、界面宽度和边界条件是否正确。

### 7.2 测试设置

取液体和玻璃两态等能：

$$
\Delta f^{SR}=0.
$$

此时自由能为：

$$
F
=
\int
\left[
w_\eta\eta^2(1-\eta)^2
+
\frac{\beta}{2}|\nabla\eta|^2
\right]dx.
$$

平衡界面满足：

$$
\beta \eta''
=
2w_\eta\eta(1-\eta)(1-2\eta).
$$

### 7.3 解析解

该方程存在 S 型界面解析解：

$$
\eta(x)
=
\frac{1}{1+\exp(-x/\delta)},
$$

其中：

$$
\delta=
\sqrt{\frac{\beta}{2w_\eta}}.
$$

界面能解析值为：

$$
\sigma_\eta
=
\frac{\sqrt{2\beta w_\eta}}{6}.
$$

### 7.4 数值测试方法

可以初始化一个阶跃界面：

$$
\eta(x,0)=
\begin{cases}
0, & x<x_0,\\
1, & x\ge x_0.
\end{cases}
$$

然后让系统在 \(\Delta f^{SR}=0\) 下弛豫。

弛豫后检查：

1. 数值界面是否接近 S 型曲线；
2. 拟合得到的界面宽度是否与 \(\sqrt{\beta/(2w_\eta)}\) 同量级；
3. 数值界面能是否接近 \(\sqrt{2\beta w_\eta}/6\)。

### 7.5 通过标准

- 界面形状应接近 logistic/tanh 型；
- 增大 \(\beta\) 时界面变宽；
- 增大 \(w_\eta\) 时界面变窄；
- 网格加密后，界面宽度和界面能应收敛。

---

## 8. 验证四：复现结构弛豫趋势

### 8.1 测试目的

复现论文中结构弛豫测试的主要趋势，而不是像素级复现图像。

原文的 Fig. 1 展示了快速淬火通过 \(T_g\) 时，\(\eta=0\) 液体态向 \(\eta=1\) 玻璃态转变的三种情况。白色代表 \(\eta=0\)，灰色代表 \(\eta=1\)。三种情况对应不同液-玻璃转变能垒。

### 8.2 参考参数

论文 Fig. 1 给出的人工参数为：

$$
f_c(\eta=0)-f_c(\eta=1)=4\ \mathrm{kJ/mol},
$$

$$
\beta=4\times 10^{-11}\ \mathrm{J/m},
$$

$$
w_\eta=4\times 10^8\ \mathrm{J/m^3},
$$

$$
L_\eta=0.1\ \mathrm{m^3/(J\cdot s)}.
$$

其中 case 2 将 \(\beta\) 和 \(w_\eta\) 同时降低一个数量级，用来表示小能垒液体-玻璃转变。

### 8.3 三种情况的预期趋势

| 情况 | 参数特征 | 预期现象 |
|---|---|---|
| case 1 | 较大 \(w_\eta\)、\(\beta\) | 转变需要明显噪声激活，结构较粗，类似成核-长大过程 |
| case 2 | \(w_\eta\)、\(\beta\) 降低一个数量级 | 更容易转变，结构更细密 |
| case 3 | 近似无能垒 | 更接近 spinodal-like 均匀转变 |

### 8.4 建议输出图像

建议输出以下结果：

1. \(\eta\) 场二维云图；
2. \(\langle \eta \rangle(t)\) 随时间变化；
3. \(F(t)\) 随时间变化；
4. 不同参数下最终 \(\eta\) 场形貌对比。

其中平均结构变量定义为：

$$
\langle\eta\rangle(t)
=
\frac{1}{|\Omega|}
\int_\Omega \eta(\mathbf{x},t)d\Omega.
$$

若 \(\langle\eta\rangle\rightarrow 1\)，说明系统整体进入玻璃态；若 \(\langle\eta\rangle\rightarrow 0\)，说明系统保持液体态。

---

## 9. 数值实现建议

### 9.1 空间离散

若使用有限差分法，可采用中心差分 Laplacian：

一维：

$$
\nabla^2\eta_i
\approx
\frac{\eta_{i+1}-2\eta_i+\eta_{i-1}}{\Delta x^2}.
$$

二维：

$$
\nabla^2\eta_{i,j}
\approx
\frac{
\eta_{i+1,j}+\eta_{i-1,j}+\eta_{i,j+1}+\eta_{i,j-1}-4\eta_{i,j}
}{\Delta x^2}.
$$

若 \(\Delta x\ne \Delta y\)，则应写为：

$$
\nabla^2\eta_{i,j}
\approx
\frac{\eta_{i+1,j}-2\eta_{i,j}+\eta_{i-1,j}}{\Delta x^2}
+
\frac{\eta_{i,j+1}-2\eta_{i,j}+\eta_{i,j-1}}{\Delta y^2}.
$$

### 9.2 时间离散

显式 Euler 格式：

$$
\eta^{n+1}
=
\eta^n
-
\Delta t L_\eta
\left[
30(\eta^n)^2(1-\eta^n)^2\Delta f^{SR}
+
2w_\eta\eta^n(1-\eta^n)(1-2\eta^n)
-
\beta\nabla^2\eta^n
\right].
$$

加入噪声时：

$$
\eta^{n+1}
=
\eta^n
-
\Delta t L_\eta
\left[
30(\eta^n)^2(1-\eta^n)^2\Delta f^{SR}
+
2w_\eta\eta^n(1-\eta^n)(1-2\eta^n)
-
\beta\nabla^2\eta^n
\right]
+
\Delta t\,\xi^n.
$$

### 9.3 稳定性建议

显式格式下，梯度项会带来类似扩散方程的稳定性约束。粗略要求：

一维：

$$
\Delta t
\lesssim
\frac{\Delta x^2}{2L_\eta\beta}.
$$

二维：

$$
\Delta t
\lesssim
\frac{\Delta x^2}{4L_\eta\beta}.
$$

这只是梯度项主导下的估计。实际还要考虑势阱项的刚性，因此建议从更小时间步开始测试。

### 9.4 变量截断

理论上 \(\eta\) 应在 \([0,1]\) 之间。数值上可以临时使用：

$$
\eta\leftarrow \min(1,\max(0,\eta)).
$$

但验证阶段不建议一开始就强行截断，因为截断可能掩盖方程符号或时间步长错误。建议先观察无截断结果，确认方程本身正确后，再考虑是否加入截断。

---

## 10. 单位检查

这是最容易出错的地方。

论文中有时给出的是 molar free energy，单位为：

$$
\mathrm{J/mol}.
$$

而相场自由能密度通常要求：

$$
\mathrm{J/m^3}.
$$

因此需要通过摩尔体积转换：

$$
f=\frac{G_m}{V_m}.
$$

如果 \(\Delta f^{SR}\) 仍然使用 \(\mathrm{J/mol}\)，而 \(w_\eta\) 使用 \(\mathrm{J/m^3}\)，则驱动力量级会完全错误。

验证时至少检查：

$$
[\Delta f^{SR}]=\mathrm{J/m^3},
$$

$$
[w_\eta]=\mathrm{J/m^3},
$$

$$
[\beta]=\mathrm{J/m},
$$

$$
[L_\eta]=\mathrm{m^3/(J\cdot s)}.
$$

这样：

$$
L_\eta\frac{\delta F}{\delta\eta}
\sim
\frac{\mathrm{m^3}}{\mathrm{J}\cdot \mathrm{s}}
\cdot
\frac{\mathrm{J}}{\mathrm{m^3}}
=
\mathrm{s^{-1}},
$$

与 \(\partial \eta/\partial t\) 的单位一致。

---

## 11. 最小验收标准

第一阶段只验证 \(\eta\) 方程时，建议采用以下四条作为通过标准：

### 11.1 0D 测试正确

高温时：

$$
\eta\rightarrow 0.
$$

低温时：

$$
\eta\rightarrow 1.
$$

### 11.2 无噪声自由能下降

在固定温度、无噪声、周期或齐次 Neumann 边界下：

$$
F^{n+1}\le F^n.
$$

### 11.3 1D 界面正确

当 \(\Delta f^{SR}=0\) 时，平衡界面应接近：

$$
\eta(x)=\frac{1}{1+\exp(-x/\delta)},
$$

其中：

$$
\delta=\sqrt{\frac{\beta}{2w_\eta}}.
$$

### 11.4 结构弛豫趋势正确

改变 \(w_\eta\) 和 \(\beta\) 后，应能够观察到从成核-长大型转变到近似无能垒均匀转变的趋势。

---

## 12. 暂时不要验证的内容

在只验证 \(\eta\) 方程阶段，不建议验证以下内容：

1. 晶体相 \(\phi_1,\phi_2,\phi_3\) 的成核和长大；
2. Cu\(_{10}\)Zr\(_7\)、CuZr、CuZr\(_2\) 的相选择；
3. 晶化起始温度；
4. 临界冷却速率 \(R_c\)；
5. 玻璃/晶体共存形貌；
6. 浓度场 \(c\) 的扩散与偏析。

这些内容必须等 \(\phi_i\) 多相场方程和浓度方程加入后才有意义。

---

## 13. 推荐执行顺序

建议按以下顺序实现和验证：

1. 写出 \(h(\eta)\)、\(h'(\eta)\)、双势阱导数；
2. 完成 0D ODE 测试；
3. 完成无噪声自由能下降测试；
4. 加入一维 Laplacian，验证界面解析解；
5. 扩展到二维周期边界；
6. 加入噪声，观察结构弛豫图像；
7. 调整 \(w_\eta\)、\(\beta\)，复现论文 Fig. 1 的趋势。

完成以上步骤后，可以认为 \(\eta\) 方程本身已经通过第一阶段验证。之后再进入 \(\phi_i\) 多相场方程和晶化竞争模型的复现。
