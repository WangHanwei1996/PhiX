The free energy is typically a functional of the **field variables** and their **gradients** :
$$
F=\int_\Omega f(q, \nabla q)d\Omega \tag{1}
$$
Introduce a small **perturbation** to the field variable, denoted as:
$$
\delta q(\mathbf{x};t)=s\psi(\mathbf{x};t) \tag{2}
$$
Define a function to describe the free energy after perturbation:
$$
G(s)=F(q+\delta q) = F(q + s\psi) \tag{3}
$$
Perform a Taylor expansion at $s=0$ :
$$
G(s)=s^0 G(0)+s^1 G'(0) +\frac{1}{2!}s^2 G''(0)+\cdots \tag{4}
$$
i.e.,
$$
F(q+\delta q) = F(q)+s \frac{dF(q+\delta q)}{ds} + O(s^2) \tag{5}
$$
The **variation of a functional** can be defined as:
$$
\left . \delta F = \frac{dF(q+\delta q)}{ds} \right |_{s=0} \tag{6}
$$
Substitute Eq. (1) into Eq. (6):
$$
\delta F = \int_\Omega \frac{df}{ds} d\Omega \tag{7}
$$
Using chain rule:
$$
\frac{df}{ds}=\frac{\partial f}{\partial q} \psi + \color{	#DC143C}{\frac{\partial f}{\partial (\nabla q)} \cdot \nabla \psi} \tag{8}
$$
Let $\mathbf{A}=\frac{\partial f}{\partial (\nabla q)}$, and according to product rule for divergence:
$$
\nabla \cdot (\psi \mathbf{A}) = \mathbf{A} \cdot \nabla \psi + \psi \nabla \cdot \mathbf{A} \tag{9}
$$
i.e.,
$$
\color{#DC143C}\frac{\partial f}{\partial (\nabla q)} \cdot \nabla \psi
\color{#000000} = \nabla \cdot \left(\psi \frac{\partial f}{\partial (\nabla q)} \right) - \psi \nabla \cdot \left(\frac{\partial f}{\partial (\nabla q)}\right) \tag{10}
$$
Substitute Eq. (8) and Eq. (10) into Eq. (7):
$$
\delta F = \int_\Omega \left(\frac{\partial f}{\partial q} \psi\right)d\Omega +
\color{#00BFFF}\int_\Omega \nabla \cdot \left(\psi \frac{\partial f}{\partial (\nabla q)} \right)d\Omega
\color{#000000}- \int_\Omega \psi \nabla \cdot \left(\frac{\partial f}{\partial (\nabla q)}\right) d \Omega \tag{11}
$$
According to Gauss Divergence Theorem:
$$
\int_\Omega \nabla \cdot \left(\psi \frac{\partial f}{\partial (\nabla q)} \right)d\Omega
= \int_{\partial \Omega} \left( \psi \frac{\partial f}{\partial (\nabla q)} \cdot \mathbf{n}\right)dS \tag{12}
$$
For Dirichlet, homogeneous Neumann and periodic boundary conditions, it could be proved that
$$
\int_{\partial \Omega} \left( \psi \frac{\partial f}{\partial (\nabla q)} \cdot \mathbf{n}\right)dS =0 \tag{13}
$$
The proof is as follows.

> [!NOTE]
>
> ### 1.  Dirichlet b.c.
>
> $$
> q\equiv q_b, \qquad when \mathbf \; \mathbf{x} \in \partial \Omega \tag{N1-1}
> $$
>
> ​	Hence,
> $$
> q+s\psi = q_b \tag{N1-2}
> $$
> ​	According to Eq. (N1-1) and Eq. (N1-2):
> $$
> \psi \equiv 0, \qquad when \mathbf \; \mathbf{x} \in \partial \Omega \tag{N1-3}
> $$
> ​	Q.E.D.
>
> ### 2.  homogeneous Neumann b.c.
>
> $$
> \frac{\partial q}{\partial n}\equiv 0, \qquad when \mathbf \; \mathbf{x} \in \partial \Omega \tag{N2-1}
> $$
>
> ​	i.e. $\nabla q \cdot \mathbf{n} = 0 \; (\mathbf{x} \in \partial \Omega)$.
>
> ​	For a common free energy density, it satisfies with:
> $$
> \frac{\partial f}{\partial (\nabla q)}=C \nabla q \tag{N2-2}
> $$
>   ​	Hence $\frac{\partial f}{\partial (\nabla q)} \cdot \mathbf{n}=0$, Q.E.D.
>
> ### 3.  periodic b.c.
>
> ​	The pair of boundaries corresponding to the $k^{th}$ direction:
> $$
> \Gamma_k^- = \left\{ \mathbf{x}_k = L_k^{min} \right\}, \quad 
> \Gamma_k^+ = \left\{ \mathbf{x}_k = L_k^{min} \right\}
> \tag{N3-1}
> $$
> ​	Thus,
> $$
> \partial \Omega = \bigcup_k \left(\Gamma_k^- \cup \Gamma_k^+\right) \tag{N3-2}
> $$
> ​	Hence,
> $$
> \int_{\partial \Omega} \left( \psi \frac{\partial f}{\partial (\nabla q)} \cdot \mathbf{n}\right)dS=
> \sum _ k \left( \int_{\Gamma_k^-} \left( \psi \frac{\partial f}{\partial (\nabla q)} \cdot \mathbf{n}_k^-\right) dS +  \int_{\Gamma_k^+} \left( \psi \frac{\partial f}{\partial (\nabla q)} \cdot \mathbf{n}_k^+\right) dS \right)
> \tag{N3-3}
> $$
> ​	Since $\mathbf{n}_k^+=\mathbf{e} _ k$, $\mathbf{n}_k^-=-\mathbf{e} _ k$ , $\int_{\partial \Omega} \left( \psi \frac{\partial f}{\partial (\nabla q)} \cdot \mathbf{n}\right)dS=0$, Q.E.D.

Under such boundary conditions, Eq. (11) is simplified as:
$$
\delta F = \int_\Omega \psi\left(\frac{\partial f}{\partial q} - \nabla \cdot \left( \frac{\partial f}{\partial (\nabla q)} \right) \right)d\Omega 
\tag{14}
$$
The functional derivative is defined as:

​	if existed $D(\mathbf{x})$ , s.t. $\delta F = \int_\Omega D(\mathbf{x}) \psi(\mathbf{x}) d \Omega$  $\forall \psi$, then $D(\mathbf{x})$ is the derivative of functional, denoted as $\frac{\delta F}{\delta q}$.

From Eq. (14):
$$
\frac{\delta F}{\delta q} = \frac{\partial f}{\partial q}- \nabla \cdot \left( \frac{\partial f}{\partial (\nabla q)} \right)
\tag{15}
$$


---

In this study of glass formation ability, we have:
$$
\frac{\delta F}{\delta \phi_i} = \frac{\partial f}{\partial \phi_i} - \nabla \cdot \left( \frac{\partial f}{\partial (\nabla \phi_i)} \right)
\tag{G1}
$$

$$
\frac{\delta F}{\delta \eta} = \frac{\partial f}{\partial \eta} - \nabla \cdot \left( \frac{\partial f}{\partial (\nabla \eta)} \right)
\tag{G2}
$$

$$
\frac{\delta F}{\delta c_k}=\frac{\partial f}{\partial c_k} \tag{G3}
$$

The original governing equation (in variational form):
$$
\frac{\partial \phi_i}{\partial t} = -L_{ij} \frac{\delta F}{\delta \phi_j}
\tag{G4}
$$

$$
\frac{\partial \eta}{\partial t} = -L_{\eta} \frac{\delta F}{\delta \eta}
\tag{G5}
$$

$$
\frac{\partial c_k}{\partial t}
=
\nabla \cdot
\left(
M_{kl} \nabla \frac{\delta F}{\delta c_l}
\right)
\tag{G6}
$$

Then, the governing equations in differential form are obtained:
$$
\frac{\partial \phi_i}{\partial t} = - L_{ij} \left(\frac{\partial f}{\partial \phi_j} - \nabla \cdot \left( \frac{\partial f}{\partial (\nabla \phi_j)} \right) \right)
\tag{G7}
$$

$$
\frac{\partial \eta}{\partial t} = -L_{\eta} \left( \frac{\partial f}{\partial \eta} - \nabla \cdot \left( \frac{\partial f}{\partial (\nabla \eta)} \right) \right)
\tag{G8}
$$

$$
\frac{\partial c_k}{\partial t}
=
\nabla \cdot
\left(
M_{kl} \nabla \frac{\partial f}{\partial c_l}
\right)
\tag{G9}
$$



The specific form of free energy density ($\phi_0$ denotes the liquid/amorphous phase and $\phi_{1,2,3}$ denote three different crystal phases):
$$
f = \phi_0\left( f_0 (c,T)+h(\eta)\Delta f^{SR} (T)\right) + \sum_{i=1}^{3} \phi_i f_i \\
+\sum_{i=0}^{3}\sum_{j=i+1}^{3} w_{ij}\phi_i^2\phi_j^2 + w_\eta \eta^2 (1-\eta)^2 + w_{ex} \eta^2 \sum_{i=1}^{3}\phi_i^2 \\
+\sum_{i=0}^{3}\sum_{j=i+1}^{3} \frac{\varepsilon_{ij}^2}{2}\left( \phi_i^2 |\nabla \phi_j|^2  - 2 \phi_i \phi_j(\nabla \phi_i \cdot \nabla \phi_j) + \phi_j^2 |\nabla \phi_i|^2 \right) + \frac{\beta}{2}|\nabla \eta|^2
\tag{G10}
$$
The partial derivatives required in the derivation:
$$
\frac{\partial f}{\partial \phi_0}
=
f_0(c,T)+h(\eta)\Delta f^{SR}(T)
+
2\phi_0\sum_{j=1}^{3}w_{0j}\phi_j^2
+
\sum_{j=1}^{3}
\varepsilon_{0j}^2
\left[
\phi_0|\nabla\phi_j|^2
-
\phi_j(\nabla\phi_0\cdot\nabla\phi_j)
\right]
\tag{G11}
$$

$$
\frac{\partial f}{\partial \phi_i}
=
f_i
+
2\phi_i\sum_{\substack{j=0\\ j\neq i}}^{3}w_{ij}\phi_j^2
+
2w_{ex}\eta^2\phi_i
+
\sum_{\substack{j=0\\ j\neq i}}^{3}
\varepsilon_{ij}^2
\left[
\phi_i|\nabla\phi_j|^2
-
\phi_j(\nabla\phi_i\cdot\nabla\phi_j)
\right],
\qquad i=1,2,3
\tag{G12}
$$

$$
\frac{\partial f}{\partial \eta}
=
30\phi_0\eta^2(1-\eta)^2\Delta f^{SR}(T)
+
2w_\eta \eta(1-\eta)(1-2\eta)
+
2w_{ex}\eta\sum_{i=1}^{3}\phi_i^2
\tag{G13}
$$

$$
\color{#DC143C}\frac{\partial f}{\partial c}
=
\phi_0\frac{\partial f_0(c,T)}{\partial c}
+
\sum_{i=1}^{3}\phi_i
\frac{\partial f_i(c,T)}{\partial c}
\tag{G14}
$$

$$
\nabla\cdot
\left(
\frac{\partial f}{\partial(\nabla \phi_i)}
\right)
=
\sum_{\substack{j=0\\ j\neq i}}^{3}
\varepsilon_{ij}^2
\left[
\phi_j^2\nabla^2\phi_i
+
\phi_j(\nabla\phi_i\cdot\nabla\phi_j)
-
\phi_i|\nabla\phi_j|^2
-
\phi_i\phi_j\nabla^2\phi_j
\right],
\qquad i=0,1,2,3

\tag{G15}
$$

$$
\nabla\cdot
\left(
\frac{\partial f}{\partial(\nabla\eta)}
\right)
=
\beta\nabla^2\eta
\tag{G16}
$$

The chemical potential models for each phase are handled as follows:

![image-20260429160258532](C:\Users\Wang\AppData\Roaming\Typora\typora-user-images\image-20260429160258532.png)

> [!NOTE]
> $$
> f_i=\frac{G_{m}^{\phi_i}}{V_{m}^{\phi_i}}
> \tag{F1}
> $$
>
> ### 1. stoichiometric compounds
>
> $$
> G_m^{\phi}
> =
> \Delta G_{Cu_p Zr_q}^{\phi}
> + p^{\circ} G_{Cu}^{fcc}
> + q^{\circ} G_{Zr}^{hcp} \\
> =
> a^{\phi} + b^{\phi} T
> + p^{\circ} G_{Cu}^{fcc}
> + q^{\circ} G_{Zr}^{hcp}
> 
> \tag{F2}
> $$
>
> $\phi_1$ : $Cu_{10}Zr_{7}$, $p=0.5882, q=0.4118$
> $$
> \begin{aligned}
> G_m^{\phi_1}
> &=
> -16133-1.905T
> +0.59\,{}^0G_{\mathrm{Cu}}^{fcc}(T)
> +0.41\,{}^0G_{\mathrm{Zr}}^{hcp}(T)
> \\[4pt]
> &=
> \begin{cases}
> -23926.88
> +126.60T
> -24.13T\ln T
> -3.36\times10^{-3}T^2
> +45300.13T^{-1}
> +7.62\times10^{-8}T^3,
> & 298\le T\le 1357.77,
> \\[6pt]
> -27332.11
> +158.06T
> -28.42T\ln T
> -1.79\times10^{-3}T^2
> +14338.11T^{-1}
> +2.15\times10^{29}T^{-9},
> & 1357.77<T\le 2128,
> \\[6pt]
> -34818.02
> +214.26T
> -35.79T\ln T
> -5.29\times10^{30}T^{-9},
> & 2128<T\le 3200.
> \end{cases}
> \end{aligned}
> 
> 
> \tag{F3}
> $$
> $\phi_2$ : $CuZr$, $p=0.5, q=0.5$
> $$
> \begin{aligned}
> G_m^{\phi_2}
> &=
> -12441.68-2.703T
> +0.5\,{}^0G_{\mathrm{Cu}}^{fcc}(T)
> +0.5\,{}^0G_{\mathrm{Zr}}^{hcp}(T)
> \\[4pt]
> &=
> \begin{cases}
> -20240.71
> +125.36T
> -24.14T\ln T
> -3.52\times 10^{-3}T^2
> +43724.50T^{-1}
> +6.46\times 10^{-8}T^3,
> & 298\le T\le 1357.77,
> \\[6pt]
> -23126.49
> +152.02T
> -27.77T\ln T
> -2.19\times 10^{-3}T^2
> +17485.50T^{-1}
> +1.82\times 10^{29}T^{-9},
> & 1357.77<T\le 2128,
> \\[6pt]
> -32255.65
> +220.56T
> -36.76T\ln T
> -6.53\times 10^{30}T^{-9},
> & 2128<T\le 3200.
> \end{cases}
> \end{aligned}
> 
> \tag{F4}
> $$
> $\phi_3$ : $CuZr_2$, $p=0.33, q=0.67$
> $$
> \begin{aligned}
> G_m^{\phi_3}
> &=
> -13840+0.654T
> +0.33\,{}^0G_{\mathrm{Cu}}^{fcc}(T)
> +0.67\,{}^0G_{\mathrm{Zr}}^{hcp}(T)
> \\[4pt]
> &=
> \begin{cases}
> -21648.74
> +127.90T
> -24.15T\ln T
> -3.81\times 10^{-3}T^2
> +40748.31T^{-1}
> +4.26\times 10^{-8}T^3,
> & 298\le T\le 1357.77,
> \\[6pt]
> -23553.36
> +145.49T
> -26.54T\ln T
> -2.93\times 10^{-3}T^2
> +23430.57T^{-1}
> +1.20\times 10^{29}T^{-9},
> & 1357.77<T\le 2128,
> \\[6pt]
> -35786.44
> +237.33T
> -38.59T\ln T
> -8.88\times 10^{30}T^{-9},
> & 2128<T\le 3200.
> \end{cases}
> \end{aligned}
> \tag{F5}
> $$
>
> ### 2. liquid phase
>
> $$
> G_m^{\phi}
> =
> {}^{\mathrm{ref}}G_m^{\phi}
> +
> {}^{\mathrm{id}}G_m^{\phi}
> +
> {}^{\mathrm{xs}}G_m^{\phi}
> \tag{F6}
> $$
>
> $$
> \begin{aligned}
> G_m^{\phi_0}
> =
> \begin{cases}
> (1-c)\Big[
> 5194.28
> +120.97T
> -24.11T\ln T
> -2.66\times10^{-3}T^2
> +52478.00T^{-1}
> +1.29\times10^{-7}T^3
> -5.85\times10^{-21}T^7
> \Big]
> \\
> \qquad
> +c\Big[
> 10320.10
> +116.57T
> -24.16T\ln T
> -4.38\times10^{-3}T^2
> +34971.00T^{-1}
> +1.63\times10^{-22}T^7
> \Big]
> \\
> \qquad
> +RT\left[(1-c)\ln(1-c)+c\ln c\right]
> +c(1-c)(-68890.00+16.20T),
> & 298\le T\le 1357.77,
> \\[12pt]
> (1-c)\Big[
> -46.55
> +173.88T
> -31.38T\ln T
> \Big]
> \\
> \qquad
> +c\Big[
> 10320.10
> +116.57T
> -24.16T\ln T
> -4.38\times10^{-3}T^2
> +34971.00T^{-1}
> +1.63\times10^{-22}T^7
> \Big]
> \\
> \qquad
> +RT\left[(1-c)\ln(1-c)+c\ln c\right]
> +c(1-c)(-68890.00+16.20T),
> & 1357.77<T\le 2128,
> \\[12pt]
> (1-c)\Big[
> -46.55
> +173.88T
> -31.38T\ln T
> \Big]
> \\
> \qquad
> +c\Big[
> -8281.26
> +253.81T
> -42.14T\ln T
> \Big]
> \\
> \qquad
> +RT\left[(1-c)\ln(1-c)+c\ln c\right]
> +c(1-c)(-68890.00+16.20T),
> & 2128<T\le 3200.
> \end{cases}
> \end{aligned}
> \tag{F7}
> $$
>
> ![Gibbs_Free_Energy_Curves](D:\IET\Projects\amorphous_PFM\rsc\Napolitano_model\Gibbs_Free_Energy_Curves.png)


$$
\Delta f^{SR}
=
- R T_g \ln(1+\alpha)\, f(\tau), \qquad \tau=T/T_g
$$



$$
f(\tau)=
\begin{cases}
\displaystyle
1-\frac{1}{A}
\left[
\frac{79}{140p}\tau^{-1}
+
\frac{474}{497}
\left(\frac{1}{p}-1\right)
\left(
\frac{\tau^3}{6}
+
\frac{\tau^9}{135}
+
\frac{\tau^{15}}{600}
\right)
\right],
& \tau < 1,
\\[14pt]
\displaystyle
-\frac{1}{A}
\left(
\frac{\tau^{-5}}{10}
+
\frac{\tau^{-15}}{315}
+
\frac{\tau^{-25}}{1500}
\right),
& \tau \ge 1.
\end{cases}
$$

|      phase      |   $V_m$ (m3/mol-atom)   |
| :-------------: | :---------------------: |
|  liquid/glass   | $1.058 \times  10^{-5}$ |
| $Cu_{10}Zr_{7}$ | $9.77 \times  10^{-6}$  |
|     $CuZr$      | $1.039 \times  10^{-5}$ |
|   $CuZr_{2}$    | $1.164 \times  10^{-5}$ |





Parameters:

| $\varepsilon_{01}^{2}=1.1\times 10^{-9}\ J/m$  |  $w_{01}=4.2\times 10^{8}\ J/m^{3}$  |  $L_{01}=0.05\ m^{3}/(s \cdot J)$  |
| :--------------------------------------------: | :----------------------------------: | :--------------------------------: |
| $\varepsilon_{02}^{2}=0.7\times 10^{-9}\ J/m$  |  $w_{02}=4.1\times 10^{8}\ J/m^{3}$  |  $L_{02}=0.08\ m^{3}/(s \cdot J)$  |
| $\varepsilon_{03}^{2}=1.2\times 10^{-9}\ J/m$  |  $w_{03}=4.7\times 10^{8}\ J/m^{3}$  |  $L_{03}=0.05\ m^{3}/(s \cdot J)$  |
| $\varepsilon_{12}^{2}=1.0\times 10^{-10}\ J/m$ |  $w_{12}=1.0\times 10^{8}\ J/m^{3}$  | $L_{12}=0.005\ m^{3}/(s \cdot J)$  |
| $\varepsilon_{13}^{2}=1.0\times 10^{-10}\ J/m$ |  $w_{13}=1.0\times 10^{8}\ J/m^{3}$  | $L_{13}=0.005\ m^{3}/(s \cdot J)$  |
| $\varepsilon_{23}^{2}=1.0\times 10^{-10}\ J/m$ |  $w_{23}=1.0\times 10^{8}\ J/m^{3}$  | $L_{23}=0.005\ m^{3}/(s \cdot J)$  |
|        $\beta=2.6\times 10^{-12}\ J/m$         | $w_{\eta}=2.5\times 10^{7}\ J/m^{3}$ | $L_{\eta}=1.36\ m^{3}/(s \cdot J)$ |
|                                                |  $w_{ex}=2.0\times 10^{9}\ J/m^{3}$  |                                    |

