# GFA_4ph 实际求解的 RHS(lambda 实现形式)

> 源文件:`applications/solvers/glass_formation_4_phases/2D/GFA_4ph.cu`
> 本文档给出代码里**实际拼装并在 GPU 上计算的右端项**——把迁移率 `Lᵢⱼ` 作为乘子分配到每一项后的展开形式,而非紧凑的 `∂φᵢ/∂t = −Σⱼ Lᵢⱼ δF/δφⱼ`。
> 每步显式 Euler:`场ⁿ⁺¹ = 场ⁿ + dt · RHSⁿ`(`EquationSystem` 同步)。

记号:`φₖ² ≡ φₖ·φₖ`,`|∇φₖ|² ≡ ∇φₖ·∇φₖ`,`∇φₐ·∇φᵦ` 为点积,`∇²` 为 Laplacian,`∇·(…)` 为散度。

---

## 0. 两个系数矩阵(乘子)

**迁移率乘子 `Lmat[i][j]`**(两两迁移率的图拉普拉斯,代入 Table II 数值):

```
                 j=0      j=1      j=2      j=3
   i=0  [  +0.18   −0.05    −0.08    −0.05  ]      L00 = L01+L02+L03
   i=1  [  −0.05   +0.06    −0.005   −0.005 ]      L11 = L01+L12+L13
   i=2  [  −0.08   −0.005   +0.09    −0.005 ]      L22 = L02+L12+L23
   i=3  [  −0.05   −0.005   −0.005   +0.06  ]      L33 = L03+L13+L23
```
(`L01=0.05, L02=0.08, L03=0.05, L12=L13=L23=0.005`;单位 m³/(s·J))

**梯度能系数 `eps2[j][k] = ε²ⱼₖ`**(对称,对角 0,单位 J/m):

```
   ε01²=1.1e-9   ε02²=0.7e-9   ε03²=1.2e-9   ε12²=ε13²=ε23²=1.0e-10
```

---

## 1. φᵢ 方程(i=0..3):实际装配式

```
RHS(φᵢ) = buildCellRHS(i)  +  divFace(Gx[i], Gy[i])
```

其中(`buildCellRHS`,对**所有 j 含 j=i**求和):

$$
\texttt{buildCellRHS}(i)=\sum_{j=0}^{3}\Big[\ \underbrace{\texttt{bulk}_j(L_{ij})}_{\text{体项}}\ +\ \sum_{k\neq j}\underbrace{\texttt{gradE\_nd}(\phi_j,\phi_k,\varepsilon^2_{jk},L_{ij})}_{\text{梯度能·非散度残差}}\ \Big],\qquad L_{ij}=\texttt{Lmat}[i][j]
$$

面通量散度部分(`divFace`,由 `addPairFlux` 累加,**对所有 j 含 j=i,k≠j**):

$$
\texttt{divFace}(G_x[i],G_y[i])=\sum_{j=0}^{3}\sum_{k\neq j} L_{ij}\,\varepsilon^2_{jk}\,\nabla\!\cdot\!\big(\phi_k^2\,\nabla\phi_j-\phi_k\phi_j\,\nabla\phi_k\big)
$$

### 1.1 体项乘子 `bulk_j(Lij)`(逐字对应代码)

$$
\texttt{bulk}_0(L_{ij})=
-L_{ij}\!\left(\frac{G_{liq}(c,T)}{V_m^{liq}}+h(\eta)\,\Delta f^{SR}\right)
-2L_{ij}\,\phi_0\big(w_{01}\phi_1^2+w_{02}\phi_2^2+w_{03}\phi_3^2\big)
$$

$$
\texttt{bulk}_1(L_{ij})=
-L_{ij}\,f_1
-2L_{ij}\,\phi_1\big(w_{01}\phi_0^2+w_{12}\phi_2^2+w_{13}\phi_3^2\big)
-2L_{ij}\,w_{ex}\,\eta^2\phi_1
$$

$$
\texttt{bulk}_2(L_{ij})=
-L_{ij}\,f_2
-2L_{ij}\,\phi_2\big(w_{02}\phi_0^2+w_{12}\phi_1^2+w_{23}\phi_3^2\big)
-2L_{ij}\,w_{ex}\,\eta^2\phi_2
$$

$$
\texttt{bulk}_3(L_{ij})=
-L_{ij}\,f_3
-2L_{ij}\,\phi_3\big(w_{03}\phi_0^2+w_{13}\phi_1^2+w_{23}\phi_2^2\big)
-2L_{ij}\,w_{ex}\,\eta^2\phi_3
$$

其中 `f₁=G_phi1(T)/Vm_phi1`,`f₂=G_phi2(T)/Vm_phi2`,`f₃=G_phi3(T)/Vm_phi3`(预算标量)。

### 1.2 梯度能非散度残差乘子 `gradE_nd(φⱼ,φₖ,ε²ⱼₖ,Lij)`

$$
\texttt{gradE\_nd}=
-L_{ij}\,\varepsilon^2_{jk}\,\phi_j\,|\nabla\phi_k|^2
\;+\;L_{ij}\,\varepsilon^2_{jk}\,\phi_k\,(\nabla\phi_k\!\cdot\!\nabla\phi_j)
$$

### 1.3 面通量乘子 `addPairFlux`(交错网格,逐对 j,k)

每个 (j,k) 对在 a 轴面上的通量分量(面插值系数 + 面梯度):

$$
F_a^{(j,k)}=L_{ij}\,\varepsilon^2_{jk}\Big[(\phi_k^{\,f})^2(\partial_a\phi_j)^f-(\phi_k^{\,f})(\phi_j^{\,f})(\partial_a\phi_k)^f\Big],
\qquad \text{贡献} = \partial_a F_a^{(j,k)}
$$

### 1.4 举例:φ₂ 的体项展开(代入 `Lmat[2]=[−0.08,−0.005,+0.09,−0.005]`)

因 `bulkⱼ(Lij)` 对 `Lij` 线性,故 `bulkⱼ(Lmat[2][j]) = Lmat[2][j]·bulkⱼ(1)`:

$$
\begin{aligned}
\text{RHS}_{\text{bulk}}(\phi_2)
=\;& -0.08\,\texttt{bulk}_0(1)-0.005\,\texttt{bulk}_1(1)+0.09\,\texttt{bulk}_2(1)-0.005\,\texttt{bulk}_3(1)\\[4pt]
=\;& +0.08\!\left(\tfrac{G_{liq}}{V_m^{liq}}+h(\eta)\Delta f^{SR}\right)+0.16\,\phi_0(w_{01}\phi_1^2+w_{02}\phi_2^2+w_{03}\phi_3^2)\\
&+0.005\,f_1+0.01\,\phi_1(\dots)+0.01\,w_{ex}\eta^2\phi_1\\
&-0.09\,f_2-0.18\,\phi_2(w_{02}\phi_0^2+w_{12}\phi_1^2+w_{23}\phi_3^2)-0.18\,w_{ex}\eta^2\phi_2\\
&+0.005\,f_3+0.01\,\phi_3(\dots)+0.01\,w_{ex}\eta^2\phi_3
\end{aligned}
$$

(再加 §1.2 的梯度残差对所有 j,k 求和、§1.3 的面通量散度。)其余 φ₀/φ₁/φ₃ 同理,只换 `Lmat` 行。

---

## 2. η 方程:实际 RHS(`eqEta.setRHS`)

$$
\boxed{\;
\text{RHS}(\eta)=
-L_\eta\Big(30\,\phi_0\,\eta^2(1-\eta)^2\,\Delta f^{SR}+2\,w_\eta\,g'(\eta)\Big)
\;-\;2\,L_\eta\,w_{ex}\,\eta\,(\phi_1^2+\phi_2^2+\phi_3^2)
\;+\;L_\eta\,\beta\,\nabla^2\eta
\;}
$$

`g'(η)=2η(1−η)(1−2η)`,`L_η=1.36`,`β=2.6e-12`。每步末加高斯噪声 `η ← clamp₀₁(η + N(mean,std²))`。

---

## 3. c 方程:实际 RHS(液相驱动 Cahn-Hilliard)

辅助场 μ(`eqMu.computeRHS(mu)`,逐点):

$$
\mu=\phi_0\cdot\frac{1}{V_m^{liq}}\frac{dG_{liq}}{dc}(c,T)
$$

c 的演化(`eqC.setRHS`,刷 μ 的 ghost 后):

$$
\boxed{\;\text{RHS}(c)=M_c\,\nabla^2\mu\;}\qquad(=\nabla\!\cdot(M_c\nabla\mu),\ M_c\ \text{常数})
$$

---

## 4. 每步执行顺序(代码循环)

```
assembleAllFlux();                          // 刷 φ ghost + 组装 §1.3 的 Gx[i]/Gy[i]
applyBC(c); eqMu.computeRHS(mu); applyBC(mu) // 建 μⁿ (§3)
sys.advance();                              // φ₀–₃(§1)+ η(§2)+ c(§3) 同步显式 Euler
k_proj_simplex4(φ₀–₃);                       // 投影 {φᵢ≥0, Σφᵢ=1}
k_clamp01(η);  [k_noiseClamp(η)]            // η→[0,1] + 噪声
```

---

## 5. 乘子/系数速查

| 符号 | 含义 | 值 / 来源 |
|------|------|-----------|
| `Lmat[i][j]` | φ 迁移率乘子(图拉普拉斯) | 由 `L01..L23` 组装(§0) |
| `ε²ⱼₖ` | 梯度能乘子 | constexpr(§0) |
| `wᵢⱼ, w_η, w_ex` | 双阱/耦合乘子 | **JSONC `constants`,缺省=论文值** |
| `L_η, β` | η 迁移率 / 梯度 | constexpr(1.36 / 2.6e-12) |
| `M_c` | c 迁移率乘子 | JSONC `constants.M_c` |
| `f₁,f₂,f₃` | 晶体体能量密度 | `Gᵢ(T)/Vmᵢ`(预算) |
| `G_liq, dG_liq/dc, Δf^SR` | 液相 CALPHAD / 短程有序 | `compute_*`(随 c/T) |

> 注:`T` 静态;液相体能量对 φ 线性 ⇒ 过冷液相线性不稳定(全液相初始场会自发全域结晶,是模型特性非数值发散)。
