


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
(f_S-f_L - h(\eta)\Delta f^{Am\rightarrow L}) h'(\phi)
+w_\phi g'(\phi)
+2w_{ex}\eta^2 \phi
-\varepsilon^2 \nabla^2 \phi
\right)
$$

$$
\frac{\partial \eta}{\partial t} = -M_\eta 
\left(
(1 - h(\phi))h'(\eta)\Delta f^{Am\rightarrow L}
+w_\eta g'(\eta) + 2w_{ex}\eta\phi^2
-\beta^2 \nabla^2 \eta
\right)
$$



$M_\phi$使用随温度变化的形式, $M_\eta$使用常数，放在settings里配置：
$$
M_\phi = 22.1 \exp(-140\times10^3/(RT))
$$
$R=8.314$, $T_g=700K$, $\alpha=0.45$
$$
\Delta f^{Am\rightarrow L} =R T \ln(1+\alpha)\, f(\tau), \qquad \tau=T/T_g
$$

$$
f(\tau)=
\begin{cases}
\displaystyle
1 - 9.9167285\times10^{-1}\,\tau^{-1} - 1.11737779\times10^{-1}\,\tau^{3} - 4.96612349\times10^{-3}\,\tau^{9} - 1.11737779\times10^{-3}\,\tau^{15}
& \tau < 1,
\\[14pt]
\displaystyle
-1.05443689\times10^{-1}\,\tau^{-5} - 3.34741816\times10^{-3}\,\tau^{-15} - 7.02957924\times10^{-4}\,\tau^{-25}
& \tau \ge 1.
\end{cases}
$$
