$$
\frac{\partial \phi}{\partial t}
=
- M_{{\phi}}
\left(
\frac{\partial f}{\partial \phi}
-
\epsilon^2 \nabla^2 \phi
\right)
$$

$$
\frac{\partial \eta}{\partial t}
=
- M_{\eta}
\left(
\frac{\partial f}{\partial \eta}
-
\beta^2 \nabla^2 \eta
\right)
$$

$$
\frac{\partial c}{\partial t}
=
\nabla \cdot
\left(
M_c\, c(1-c)\,
\nabla
\left(
\frac{\partial f}{\partial c}
\right)
\right)
$$


$$
f(\phi,\eta,c,T)
=
[1-h(\phi)]
\left[
f_L(c,T)+h(\eta)\Delta f^{Am\to L}(T)
\right]
+
h(\phi)f_S(T)
+
w_{\phi}g(\phi)
+
w_{\eta}g(\eta)
+
w_{ex}\phi^2\eta^2
$$

$$

$$

#### 1

$$
f_L(c,T) = \frac{G^L}{V_m^L}
$$

$$
G^L = G^{\mathrm{ref}} + G^{\mathrm{id}} + G^{\mathrm{ex}}
$$

$$
G_{\mathrm{ref}}^L = c G_B^L + (1 - c) G_{Fe}^L
$$

$$
G_{\mathrm{id}}^L = RT \left( c \ln c + (1 - c)\ln(1 - c) \right)
$$

$$
G_{\mathrm{ex}}^L = c(1 - c) L_{B,Fe}^L
$$

$$
G_B^L =
\begin{cases}
40723.275 + 86.843839T - 15.6641T \ln T - 6.864515 \times 10^{-3} T^2 + 0.618878 \times 10^{-6} T^3 + 370843T^{-1}, & 298.15K < T \le 500K \\

41119.703 + 82.101722T - 14.9827763T \ln T - 7.095669 \times 10^{-3} T^2 + 0.507347 \times 10^{-6} T^3 + 335484T^{-1}, & 500K < T \le 2348K \\

28842.012 + 200.94731T - 31.4T \ln T, & 2348K < T \le 6000K
\end{cases}
$$

$$
G_{Fe}^L =
\begin{cases}
13265.87 + 117.57557T - 23.5143T \ln T - 4.39752 \times 10^{-3} T^2 - 0.058927 \times 10^{-6} T^3 + 773597T^{-1} - 367.516 \times 10^{-23} T^7, & 298.15K < T \le 1811K \\

-10838.83 + 291.302T - 46T \ln T, & 1811K < T \le 6000K
\end{cases}
$$


$$
L_{B,Fe}^L = -122861 + 14.59T
$$


#### 2

$$
\Delta f^{L \to Am}(c, T) = - R T_g \ln(1 + \alpha)\, f(\tau)
$$

$$
f(\tau) =
\begin{cases}
1 - 9.9167285 \times 10^{-1}\tau^{-1}
- 1.11737779 \times 10^{-1}\tau^{3}
- 4.96612349 \times 10^{-3}\tau^{9}
- 1.11737779 \times 10^{-3}\tau^{15}, & \tau \le 1 \\

-1.05443689 \times 10^{-1}\tau^{-5}
- 3.34741816 \times 10^{-3}\tau^{-15}
- 7.02957924 \times 10^{-4}\tau^{-25}, & \tau > 1
\end{cases}
$$





#### 3

$$
f_S(T) = \frac{G^S}{V_m^S}
$$

$$
G^S=G_{Fe_3B} = 3G_{Fe}^{bcc} + G_{B}^{\beta} + \Delta G_{Fe_3B}^{f}
$$

$$
G_{Fe}^{bcc} =
\begin{cases}
1225.7 + 124.134T - 23.5143T \ln T - 4.39752 \times 10^{-3} T^2 - 0.058927 \times 10^{-6} T^3 + 773597T^{-1}, & 298.15K < T \le 1811K \\

-25383.581 + 299.31255T - 46T \ln T + 2296.03 \times 10^{28} T^{-9}, & 1811K < T \le 6000K
\end{cases}
$$

$$
G_B^{\beta} =
\begin{cases}
-7735.284 + 107.111864T - 15.6641T \ln T - 6.864515 \times 10^{-3} T^2 + 0.618878 \times 10^{-6} T^3 + 370843T^{-1}, & 298.15K < T \le 1100K \\

-16649.474 + 184.801744T - 26.6047T \ln T - 0.79809 \times 10^{-3} T^2 - 0.02556 \times 10^{-6} T^3 + 1748270T^{-1}, & 1100K < T \le 2348K \\

-36667.582 + 231.336244T - 31.5957527T \ln T - 1.59488 \times 10^{-3} T^2 + 0.134719 \times 10^{-6} T^3 + 11205883T^{-1}, & 2348K < T \le 3000K \\

-21530.653 + 222.396264T - 31.4T \ln T, & 3000K < T \le 6000K
\end{cases}
$$

$$
\Delta G_{Fe_a Si_b B_c}^{f} = -77749 + 2.59T
$$





#### 4 etc

$$
h(\phi) = \phi^3 (10 - 15\phi + 6\phi^2), \quad
h(\eta) = \eta^3 (10 - 15\eta + 6\eta^2)
$$

$$
g(\phi) = \phi^2 (1 - \phi)^2, \quad
g(\eta) = \eta^2 (1 - \eta)^2
$$

$$
M_{\phi} = 22.1 \exp\left(\frac{-140 \times 10^3}{RT}\right)\ \mathrm{m^3/(J\cdot s)}
$$

$$
M_c =
\frac{
(1 - h(\phi))\big((1 - h(\eta))D_L + h(\eta)D_{Am}\big)
+ h(\phi)D_S
}{RT}
$$

$$
D_L = 2 \times 10^{-6} \exp\left(\frac{-1.11 \times 10^5}{RT}\right)\ \mathrm{m^2/s}
$$

$$
D_S = 1.311 \times 10^{-6} \exp\left(\frac{-1.51 \times 10^5}{RT}\right)\ \mathrm{m^2/s}
$$