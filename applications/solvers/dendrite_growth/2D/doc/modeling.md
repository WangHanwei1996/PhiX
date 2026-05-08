#### 1 求解J：

$$
\mathbf{J} =
\begin{pmatrix}
J_x \\
J_y
\end{pmatrix}
$$

$$
J_x = W^2 \phi_x + |\nabla \phi|^2 W \frac{\partial W}{\partial \phi_x}
$$

$$
J_y = W^2 \phi_y + |\nabla \phi|^2 W \frac{\partial W}{\partial \phi_y}
$$

其中，
$$
W(\mathbf{n}) = W_0 a(\mathbf{n})
$$

$$
a(\mathbf{n}) = 1 + \epsilon_m \cos\big(m(\theta - \theta_0)\big)
$$

$$
\tan \theta = \frac{n_y}{n_x}, \qquad
\mathbf{n} = \frac{\nabla \phi}{|\nabla \phi|}
$$

$$
\phi_x,\ \phi_y, \qquad
g = \sqrt{\phi_x^2 + \phi_y^2 + \delta^2}
$$

$$
n_x = \frac{\phi_x}{g}, \qquad
n_y = \frac{\phi_y}{g}
$$



#### 2 求解Φ：

$$
\frac{\partial \phi}{\partial t}
=
\frac{
\left[\phi - \lambda U(1 - \phi^2)\right](1 - \phi^2)
+ \nabla \cdot \mathbf{J}
}{
\tau(\mathbf{n})
}
$$

$$
\tau(\mathbf{n}) = \tau_0 [a(\mathbf{n})]^2
$$

$$
\lambda = \frac{D \tau_0}{0.6267 W_0^2}
$$



#### 3 求解U：

$$
\frac{\partial U}{\partial t}
=
D \nabla^2 U
+
\frac{1}{2}\frac{\partial \phi}{\partial t}
$$

这里，$\frac{\partial \phi}{\partial t}$是步骤2中的RHS值





#### 其它说明

$\theta_0,D,\tau_0,W_0,\varepsilon_m, m$均为常数，从配置文件读取

$\delta$为防除0设置的小量