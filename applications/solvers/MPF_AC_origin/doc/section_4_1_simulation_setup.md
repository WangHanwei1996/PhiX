# 4. Simulation studies

## 4.1. Simulation setup

For our benchmark problem, we start with physical values in the order of typical microstructure simulations listed in Table 2 and make them non-dimensional, as would be common practice for any application to a specific material system. Three reference quantities are applied to de-dimensionalize the problem, namely the reference time $t_{ref}=100s$, length $x_{ref}=1\ \mu m=10^{-6}\ m$ and energy $\Omega_{ref}=10^6\ J/m^3$, which yields

$$
\tilde{W}=\frac{W}{x_{ref}},\qquad
\tilde{\gamma}_{\alpha\beta}=\frac{\gamma_{\alpha\beta}}{\Omega_{ref}x_{ref}},\qquad
\tilde{M}=M\frac{\Omega_{ref}t_{ref}}{x_{ref}}.
$$

Note that the choice of parameters is somewhat arbitrary and the simulation results could be re-scaled differently by changing the reference values.

The initial conditions of the problem are identical for the two following sub-problems. They are sketched in Fig. 3(b) and can be summarized as follows: We start with initially sharp interfaces and fill $0\le x\le W$ and $80<y\le 100$ with $\phi_0=1$ and, furthermore, $0\le x\le W/2$ and $0\le y\le 80$ with $\phi_\alpha=1$ and the other half $W/2<x\le W$, $0\le y\le 80$ with $\phi_\beta=1$. The respective other phases are equal to zero. A similar validation example has been used in [42]. For all cases, we employ Neumann boundary conditions (BCs) $\nabla\phi_\alpha\cdot\vec{n}=0$, $\forall \alpha$ at the top and bottom of the domain. At the left and right domain boundary

- Dirichlet BCs are used for sub-problem (1) according to the initial setup, i.e. phase boundaries are pinned at $[0,80]$ and $[W,80]$;
- Neumann BCs with $\nabla\phi_\alpha\cdot\vec{n}=0$, $\forall \alpha$ are used for subproblem (2) which in this case reflects mirror symmetry. Alternatively, the domain length can be doubled to $2W$ in combination with periodic BCs.

**Table 2**  
Set of simulation parameters.

| Parameter | Symbol | Physical value | Simulation value |
|---|---:|---:|---:|
| Width of domain | $W$ | $100\ \mu m$ | 100 |
| Height of domain | $H$ | $[100,\ldots,400]\ \mu m$ | $[100,\ldots,400]$ |
| Spatial resolution | $\Delta x$ | $W$/cells | 1 |
| Interfacial energy | $\gamma_{\alpha0}=\gamma_{\beta0}=\gamma_0$ | $1.0\ J/m^2$ | 1.0 |
|  | $\gamma_{\alpha\beta}$ | $[0.1,\ldots,2.0]\gamma_0\ J/m^2$ | $[0.01,2.0]$ |
| Mobility | $M_{\alpha0}=M_{\beta0}=M_0$ | $10^{-14}\ m^4/(Js)$ | 1 |
|  | $M_{\alpha\beta}$ | $10^{-14}\ m^4/(Js)$ | 1 |

All simulations are conducted using codes based on finite difference stencils and an explicit Euler time-stepping. MPF simulations were performed using PACE3D [36,43] which is an in-house code. Furthermore, all the models (MOP as well as MPF) were implemented for the specific case of three phases in MATLAB together with the necessary post-processing tools to acquire the metrics presented in the following sections. The MATLAB code has been made publicly available [44].
