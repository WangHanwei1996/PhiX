# Static triple-junction benchmark：三相 AC 方程方案

目标：复现 static triple junction 的等界面能平衡结果，即三条界面在三重点处形成

$$
120^\circ,\quad 120^\circ,\quad 120^\circ .
$$

本方案只解三相 Allen--Cahn / MPF 方程，不解浓度、温度或其他序参量。

---

## 1. 相场变量与约束

三相为

$$
\phi_0,\qquad \phi_\alpha,\qquad \phi_\beta .
$$

它们满足

$$
\phi_0+\phi_\alpha+\phi_\beta=1,
$$

并希望满足

$$
0\le \phi_i\le 1.
$$

本方案只使用 pairwise mobility 形式保持求和约束，不讨论投影修正。

---

## 2. Section 3.1 的界面能参数

采用论文 3.1 的参数关系：

$$
\gamma_{0\alpha}=\gamma_{0\beta}=\gamma_0,
$$

$$
\gamma_{\alpha\beta}=r\gamma_0.
$$

若目标是三个 \($120^\circ$\) 夹角，则取

$$
r=1,
$$

即

$$
\gamma_{0\alpha}=\gamma_{0\beta}=\gamma_{\alpha\beta}=\gamma_0.
$$

为了无量纲计算，可直接令

$$
\gamma_0=1.
$$

则

$$
\gamma_{0\alpha}=1,
\qquad
\gamma_{0\beta}=1,
\qquad
\gamma_{\alpha\beta}=1.
$$

更一般地，解析平衡角为

$$
\theta
=
2\arccos\left(\frac{\gamma_{\alpha\beta}}{2\gamma_0}\right)
=
2\arccos\left(\frac{r}{2}\right).
$$

当 \($r=1$\) 时，

$$
\theta=120^\circ.
$$

---

## 3. 自由能泛函

取最常见的 pairwise double-well 形式：

$$
F[\boldsymbol\phi]
=
\int_\Omega
\left(
 f_{grad}+f_{dw}
\right)d\Omega .
$$

其中

$$
f_{grad}
=
\sum_{i<j}
\frac{\varepsilon_{ij}^2}{2}
\left|
\phi_i\nabla\phi_j-\phi_j\nabla\phi_i
\right|^2,
$$

$$
f_{dw}
=
\sum_{i<j}
W_{ij}\phi_i^2\phi_j^2.
$$

这里

$$
i,j\in\{0,\alpha,\beta\}.
$$

三相展开为

$$
f_{grad}
=
\frac{\varepsilon_{0\alpha}^2}{2}
\left|
\phi_0\nabla\phi_\alpha-\phi_\alpha\nabla\phi_0
\right|^2
+
\frac{\varepsilon_{0\beta}^2}{2}
\left|
\phi_0\nabla\phi_\beta-\phi_\beta\nabla\phi_0
\right|^2
+
\frac{\varepsilon_{\alpha\beta}^2}{2}
\left|
\phi_\alpha\nabla\phi_\beta-\phi_\beta\nabla\phi_\alpha
\right|^2,
$$

$$
f_{dw}
=
W_{0\alpha}\phi_0^2\phi_\alpha^2
+
W_{0\beta}\phi_0^2\phi_\beta^2
+
W_{\alpha\beta}\phi_\alpha^2\phi_\beta^2.
$$

---

## 4. 用目标界面能确定 \($\varepsilon_{ij}$\) 和 \($W_{ij}$\)

对于二相平界面，若

$$
f_{ij}
=
\frac{\varepsilon_{ij}^2}{2}|\nabla\phi_i|^2
+
W_{ij}\phi_i^2(1-\phi_i)^2,
$$

则界面能为

$$
\gamma_{ij}
=
\frac{\varepsilon_{ij}\sqrt{2W_{ij}}}{6}.
$$

若用 tangent width 定义界面宽度

$$
\ell=
\frac{1}{\left.d\phi/dx\right|_{\phi=1/2}},
$$

则

$$
\ell
=
\frac{4\varepsilon_{ij}}{\sqrt{2W_{ij}}}.
$$

因此给定目标界面能 \($\gamma_{ij}$\) 和统一界面宽度 \($\ell$\)，可取

$$
\boxed{
\varepsilon_{ij}^2
=
\frac{3}{2}\gamma_{ij}\ell
}
$$

$$
\boxed{
W_{ij}
=
\frac{12\gamma_{ij}}{\ell}
}
$$

等界面能 \($\gamma_{ij}=\gamma_0$\) 时，所有 pair 使用相同的

$$
\varepsilon_{ij}^2
=
\frac{3}{2}\gamma_0\ell,
\qquad
W_{ij}
=
\frac{12\gamma_0}{\ell}.
$$

---

## 5. 化学势 / 变分导数

定义

$$
\mu_i
=
\frac{\delta F}{\delta\phi_i}.
$$

总化学势写成

$$
\mu_i=\mu_i^{grad}+\mu_i^{dw}.
$$

### 5.1 双井势贡献

$$
\boxed{
\mu_i^{dw}
=
2\phi_i\sum_{j\ne i}W_{ij}\phi_j^2
}
$$

即

$$
\mu_0^{dw}
=
2\phi_0
\left(
W_{0\alpha}\phi_\alpha^2
+
W_{0\beta}\phi_\beta^2
\right),
$$

$$
\mu_\alpha^{dw}
=
2\phi_\alpha
\left(
W_{0\alpha}\phi_0^2
+
W_{\alpha\beta}\phi_\beta^2
\right),
$$

$$
\mu_\beta^{dw}
=
2\phi_\beta
\left(
W_{0\beta}\phi_0^2
+
W_{\alpha\beta}\phi_\alpha^2
\right).
$$

### 5.2 weighted gradient 贡献

对

$$
f_{grad}
=
\sum_{i<j}
\frac{\varepsilon_{ij}^2}{2}
\left|
\phi_i\nabla\phi_j-\phi_j\nabla\phi_i
\right|^2,
$$

有

$$
\boxed{
\mu_i^{grad}
=
\sum_{j\ne i}
\varepsilon_{ij}^2
\left[
2\phi_i|\nabla\phi_j|^2
-2\phi_j(\nabla\phi_i\cdot\nabla\phi_j)
+\phi_i\phi_j\nabla^2\phi_j
-\phi_j^2\nabla^2\phi_i
\right]
}
$$

因此

$$
\mu_i
=
2\phi_i\sum_{j\ne i}W_{ij}\phi_j^2
+
\sum_{j\ne i}
\varepsilon_{ij}^2
\left[
2\phi_i|\nabla\phi_j|^2
-2\phi_j(\nabla\phi_i\cdot\nabla\phi_j)
+\phi_i\phi_j\nabla^2\phi_j
-\phi_j^2\nabla^2\phi_i
\right].
$$

---

## 6. Pairwise Allen--Cahn 控制方程

采用缩放后的 pairwise kinetic coefficient：

$$
L_{ij}^{mob}=L_{ji}^{mob}.
$$

控制方程写为

$$
\boxed{
\frac{\partial\phi_i}{\partial t}
=
-\sum_{j\ne i}L_{ij}^{mob}(\mu_i-\mu_j)
}
$$

三相分量式为

$$
\boxed{
\frac{\partial\phi_0}{\partial t}
=
-
L_{0\alpha}^{mob}(\mu_0-\mu_\alpha)
-
L_{0\beta}^{mob}(\mu_0-\mu_\beta)
}
$$

$$
\boxed{
\frac{\partial\phi_\alpha}{\partial t}
=
-
L_{0\alpha}^{mob}(\mu_\alpha-\mu_0)
-
L_{\alpha\beta}^{mob}(\mu_\alpha-\mu_\beta)
}
$$

$$
\boxed{
\frac{\partial\phi_\beta}{\partial t}
=
-
L_{0\beta}^{mob}(\mu_\beta-\mu_0)
-
L_{\alpha\beta}^{mob}(\mu_\beta-\mu_\alpha)
}
$$

该形式自动满足

$$
\frac{\partial}{\partial t}
(\phi_0+\phi_\alpha+\phi_\beta)=0.
$$

若采用未缩放的 Steinbach mobility \($M_{ij}$\)，则可令

$$
L_{ij}^{mob}=\frac{M_{ij}}{3\epsilon_s},
$$

其中 \($\epsilon_s$\) 是 Steinbach 形式中的界面宽度尺度参数。本文档后续默认使用已经缩放后的 \($L_{ij}^{mob}$\)。

---

## 7. static benchmark 初始条件

计算域取

$$
\Omega=[0,W]\times[0,H].
$$

设置一个初始水平分界高度

$$
y_b.
$$

初始条件为

$$
\phi_0(x,y,0)=
\begin{cases}
1, & y>y_b,\\
0, & y\le y_b,
\end{cases}
$$

$$
\phi_\alpha(x,y,0)=
\begin{cases}
1, & y\le y_b,\ x\le W/2,\\
0, & \text{otherwise},
\end{cases}
$$

$$
\phi_\beta(x,y,0)=
\begin{cases}
1, & y\le y_b,\ x>W/2,\\
0, & \text{otherwise}.
\end{cases}
$$

初始时有

$$
\phi_0+\phi_\alpha+\phi_\beta=1.
$$

---

## 8. static benchmark 边界条件

static case 用左右 Dirichlet 固定界面端点，上下 Neumann。

### 上下边界

$$
\nabla\phi_i\cdot\mathbf n=0,
\qquad i=0,\alpha,\beta.
$$

### 左边界

$$
\phi_0=1,\quad \phi_\alpha=0,\quad \phi_\beta=0,
\qquad y>y_b,
$$

$$
\phi_0=0,\quad \phi_\alpha=1,\quad \phi_\beta=0,
\qquad y\le y_b.
$$

### 右边界

$$
\phi_0=1,\quad \phi_\alpha=0,\quad \phi_\beta=0,
\qquad y>y_b,
$$

$$
\phi_0=0,\quad \phi_\alpha=0,\quad \phi_\beta=1,
\qquad y\le y_b.
$$

---

## 9. 解析验证目标

Section 3.1 的解析角度为

$$
\theta
=
2\arccos\left(\frac{\gamma_{\alpha\beta}}{2\gamma_0}\right).
$$

当

$$
\gamma_{\alpha\beta}=\gamma_0
$$

时，

$$
\boxed{
\theta=120^\circ
}
$$

三相角度应为

$$
\boxed{
120^\circ,\quad 120^\circ,\quad 120^\circ
}
$$

对应的几何高度满足

$$
\boxed{
h_{GB}
=
W
\frac{\gamma_{\alpha\beta}}
{2\sqrt{4\gamma_0^2-\gamma_{\alpha\beta}^2}}
}
$$

等界面能时，

$$
h_{GB}
=
\frac{W}{2\sqrt{3}}.
$$

---

## 10. 推荐最小参数组

为了先得到三个 \(120^\circ\) 角，可取

$$
\gamma_0=1,
\qquad
\gamma_{0\alpha}=\gamma_{0\beta}=\gamma_{\alpha\beta}=1.
$$

选择统一弥散界面宽度

$$
\ell=6\sim 10
$$

或按网格尺度取

$$
\ell=6\Delta x\sim 10\Delta x.
$$

然后

$$
\varepsilon_{ij}^2
=\frac{3}{2}\gamma_{ij}\ell,
\qquad
W_{ij}
=\frac{12\gamma_{ij}}{\ell}.
$$

pairwise kinetic coefficient 可先统一取

$$
L_{0\alpha}^{mob}=L_{0\beta}^{mob}=L_{\alpha\beta}^{mob}=L_0.
$$

\(L_0\) 只影响弛豫速度，不影响最终平衡角。

---

## 11. 本方案要解的方程清单

只需要解三条 AC 方程：

$$
\frac{\partial\phi_0}{\partial t}
=
-
L_{0\alpha}^{mob}(\mu_0-\mu_\alpha)
-
L_{0\beta}^{mob}(\mu_0-\mu_\beta),
$$

$$
\frac{\partial\phi_\alpha}{\partial t}
=
-
L_{0\alpha}^{mob}(\mu_\alpha-\mu_0)
-
L_{\alpha\beta}^{mob}(\mu_\alpha-\mu_\beta),
$$

$$
\frac{\partial\phi_\beta}{\partial t}
=
-
L_{0\beta}^{mob}(\mu_\beta-\mu_0)
-
L_{\alpha\beta}^{mob}(\mu_\beta-\mu_\alpha),
$$

其中

$$
\mu_i
=
2\phi_i\sum_{j\ne i}W_{ij}\phi_j^2
+
\sum_{j\ne i}
\varepsilon_{ij}^2
\left[
2\phi_i|\nabla\phi_j|^2
-2\phi_j(\nabla\phi_i\cdot\nabla\phi_j)
+\phi_i\phi_j\nabla^2\phi_j
-\phi_j^2\nabla^2\phi_i
\right].
$$

