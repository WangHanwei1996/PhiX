


$$
f(c, T, \phi)
= f_L(c,T)\,[1 - h(\phi)]
+ f_S(c,T)\,h(\phi)
+ w_\phi\,g(\phi)
+\frac{\varepsilon^2}{2}|\nabla \phi|^2
$$

$f_L(c,T)$, $ f_S(c,T)$ , $ \frac{\partial f_L(c,T)}{\partial c}$ 查表，使用data中CuZr的数据。
$$
h(\phi)
= \phi^3 \left(6\phi^2 - 15\phi + 10\right)
$$

$$
g(\phi)
= \phi^2 (1 - \phi)^2
$$


$$
\mu = \frac{\partial f_L(c,T)}{\partial c}\,[1 - h(\phi)]
$$


$$
\frac{\partial c}{\partial t}
= M_c \nabla^2\mu
$$

$$
\frac{\partial \phi}{\partial t}
= -M_{\phi} \left(
(f_S-f_L) h'(\phi)
+w_\phi g'(\phi)
-\varepsilon^2 \nabla^2 \phi
\right)
$$

