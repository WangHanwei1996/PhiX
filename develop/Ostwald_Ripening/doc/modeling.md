$$
F = \int_V \left( f_{\mathrm{chem}}(c, \eta_1, \ldots, \eta_p)
+ \frac{\kappa_c}{2} |\nabla c|^2
+ \sum_{i=1}^{p} \frac{\kappa_\eta}{2} |\nabla \eta_i|^2 \right)\, dV
$$

$$
f_{\mathrm{chem}}(c, \eta_1, \ldots, \eta_p)
= f^\alpha(c)\,[1 - h(\eta_1, \ldots, \eta_p)]
+ f^\beta(c)\,h(\eta_1, \ldots, \eta_p)
+ w\,g(\eta_1, \ldots, \eta_p)
$$




$$
f^\alpha(c) = \varrho^2 (c - c_\alpha)^2
$$

$$
f^\beta(c) = \varrho^2 (c_\beta - c)^2
$$

$$
h(\eta_1, \ldots, \eta_p)
= \sum_{i=1}^{p} \eta_i^3 \left(6\eta_i^2 - 15\eta_i + 10\right)
$$

$$
g(\eta_1, \ldots, \eta_p)
= \sum_{i=1}^{p} \left[ \eta_i^2 (1 - \eta_i)^2 \right]
+ \alpha \sum_{i=1}^{p} \sum_{j \ne i}^{p} \eta_i^2 \eta_j^2
$$




$$
\frac{\partial c}{\partial t}
= \nabla \cdot \left\{
M \nabla \left(
\frac{\partial f_{\mathrm{chem}}}{\partial c}
- \kappa_c \nabla^2 c
\right)
\right\}
$$

$$
\frac{\partial \eta_i}{\partial t}
= -L \left[ \frac{\delta F}{\delta \eta_i} \right]
= -L \left(
\frac{\partial f_{\mathrm{chem}}}{\partial \eta_i}
- \kappa_\eta \nabla^2 \eta_i
\right)
$$