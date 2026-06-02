# v2.6.8 — Dendrite Growth Solver Rewrite (Staggered Face-Centred Scheme)

## Summary

Complete rewrite of `applications/solvers/dendrite_growth/2D/dendrite_growth.cu`
to eliminate the checkerboard instability present in the previous node-centred
flux scheme. The new implementation follows the NIST PFHub Benchmark 3 CPU
reference (Karma & Rappel 1998) and uses the `facePW`/`facePWGPU` operators
added in v2.6.7.

## Changes

### `applications/solvers/dendrite_growth/2D/dendrite_growth.cu`  (rewrite)

**Old scheme (broken):**
- Cell-centred `J_x`, `J_y` computed via `pw(phi_x, phi_y, a_fld, Fn)` on
  cell-centre data, then divergence taken with `iso_grad(J_x, 0) + iso_grad(J_y, 1)`.
- This couples only ±1 neighbours of the same parity, creating a
  checkerboard decoupling that degrades to unstable oscillations.
- Phi clamp + U latent-heat correction introduced to manage the instability.
- Temperature Laplacian split into two iso_grad passes (U_x, U_y).

**New scheme (staggered face-centred):**

| Step | Operation | Purpose |
|------|-----------|---------|
| A1 | `eq_phi_x_cc.advanceSteady(bcs, &phi)` | cell-centre `∂φ/∂x` (CD2) for τ and face interp |
| A2 | `eq_phi_y_cc.advanceSteady(bcs, &phi)` | cell-centre `∂φ/∂y` (CD2) |
| B1 | `faceGradGPU(phi, 0, phi_x_xf)` | `∂φ/∂x` on x-faces |
| B2 | `interpGPU(phi_y_cc, 0, phi_y_xf)` | `φ_y` interpolated to x-faces |
| B3 | `facePWGPU(jx, phi_x_xf, phi_y_xf, Fn)` | Jx = W²·φ_x + A·φ_y on x-faces |
| C1 | `faceGradGPU(phi, 1, phi_y_yf)` | `∂φ/∂y` on y-faces |
| C2 | `interpGPU(phi_x_cc, 1, phi_x_yf)` | `φ_x` interpolated to y-faces |
| C3 | `facePWGPU(jy, phi_y_yf, phi_x_yf, Fn)` | Jy = W²·φ_y − A·φ_x on y-faces |
| D  | `eq_a_cc.advanceSteady` | cell-centre a(n) for τ |
| E  | `eq_dphi.advanceSteady` | `∂φ/∂t = inv_τ·(N + divFace(jx,jy))` |
| F  | `eq_phi.advanceTransient` | `φ += dt·∂φ/∂t` |
| G  | `eq_U.advanceTransient` | `U += dt·(D·∇²U + 0.5·∂φ/∂t)` |

Key improvements:
- Conservative `divFace(jx, jy)` divergence — no checkerboard decoupling.
- Phi clamp and U latent-heat correction kernels removed.
- CD2 `grad(phi, axis, 1.0)` replaces 9-point `iso_grad` for phase-field
  cell-centre gradients (consistent with the Benchmark 3 reference).
- CD2 `lap(U, D)` replaces the two-pass `iso_grad(U_x, 0, D) + iso_grad(U_y, 1, D)`
  for temperature diffusion (simpler, fewer auxiliary fields).
- Auxiliary fields reduced: removed `phi_x`, `phi_y`, `a_fld`, `J_x`, `J_y`,
  `U_x`, `U_y`; added `phi_x_cc`, `phi_y_cc`, `a_cc`, and six `FaceField`s.

### `applications/solvers/dendrite_growth/2D/CMakeLists.txt`  (recreated)

Recreated the deleted CMakeLists.txt using the standard PhiX solver pattern.

### `CMakeLists.txt`  (root)

Uncommented `add_subdirectory("applications/solvers/dendrite_growth/2D")` so
the solver is included in the default build.

## Physics

Karma–Rappel (1998) Allen-Cahn + thermal diffusion:

```
τ(n)·∂φ/∂t = ∂/∂x[W²φ_x + A·φ_y] + ∂/∂y[W²φ_y − A·φ_x]
            + (1−φ²)·(φ − λ·U·(1−φ²))

∂U/∂t = D·∇²U + 0.5·∂φ/∂t

a(n) = 1 + ε·cos(m·(θ−θ₀)),   W = W₀·a,   τ = τ₀·a²
A    = W·W₀·ε·m·sin(m·(θ−θ₀))
λ    = D·τ₀ / (0.6267·W₀²)
```

## Commit

`d63203d` feat(solvers): rewrite dendrite_growth with staggered face-centred flux scheme
