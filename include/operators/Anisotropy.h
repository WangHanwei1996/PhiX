#pragma once

// ---------------------------------------------------------------------------
// Anisotropy.h — m-fold surface-energy anisotropy (2D Kobayashi form).
//
// Phase-field dendrite/facet models replace the isotropic gradient energy
// W0²∇φ by W(θ)∇φ with
//
//     a(θ) = 1 + ε·cos(m(θ − θ0)),   θ = atan2(φ_y, φ_x),   W = W0·a
//
// giving the anisotropic driving term (variational derivative):
//
//     ∇·J,   J = W0² [ a²·∇φ  +  a·(da/dθ)·(−φ_y, φ_x) ]
//
// anisoDiv() provides that term as a SINGLE fused kernel: for every cell the
// four face fluxes are evaluated in registers (face-normal gradient +
// averaged tangential gradient, exactly the faceGrad/interp discretisation
// of the classic chain) and their divergence is accumulated into the RHS —
// replacing the hand-built pipeline of ~8 kernel launches and 6 intermediate
// FaceFields (2× faceGrad, 2× interp, 2× facePW, cell-gradient prep, divFace)
// used by the dendrite solver.  Face values are recomputed by both adjacent
// cells (compute is free next to the saved memory traffic) and are formed
// from identical inputs, so the scheme stays CONSERVATIVE (telescoping
// fluxes).
//
// Requirements/limits:
//   • 2D only (m-fold in-plane anisotropy); ghost >= 1 with BCs applied to φ;
//   • the stencil reads DIAGONAL neighbours: at the four domain corners the
//     corner-ghost cells are used, which face-patch BCs do not fill (they are
//     zero after allocDevice).  Keep interfaces away from domain corners, or
//     see BCBatch corner filling (v2.24.0).
//   • ε = 0 reduces exactly to W0²·∇²φ (CD2).
//
// Usage (dendrite RHS):
//     AnisoParams ap;  ap.W0 = W0;  ap.eps = 0.05;  ap.m = 4;
//     eqPhi.setRHS( anisoDiv(phi, ap) + pw(phi, U, PHIX_FN (...) {...}) );
//
// Cell-centre a(θ) field (for τ(θ) = τ0·a² etc.):
//     anisoFactorOnGPU(phi, aField, ap);   // physical cells, from CD2 ∇φ
// ---------------------------------------------------------------------------

#include "core/Real.h"
#include "equation/Term.h"
#include "field/ScalarField.h"

namespace PhiX {

struct AnisoParams {
    double W0     = 1.0;   // gradient-energy prefactor (flux carries W0²)
    double eps    = 0.0;   // anisotropy strength ε
    int    m      = 4;     // fold symmetry
    double theta0 = 0.0;   // preferred growth orientation

    void validate() const;   // throws std::invalid_argument
};

namespace aniso {

// a(θ) — usable inside PHIX_FN functors and kernels.
__host__ __device__ inline Real factor(Real theta, Real eps, int m,
                                       Real theta0) {
    return Real(1) + eps * cos(Real(m) * (theta - theta0));
}

} // namespace aniso

// coeff · ∇·( W0²[a²∇φ + a·a'·(−φ_y, φ_x)] ) as a single fused Term.
Term anisoDiv(const ScalarField& phi, const AnisoParams& p,
              double coeff = 1.0);

// aOut(physical cells) = a(θ(∇φ)) from CD2 cell-centre gradients.
// aOut is pointwise data — no ghost refresh required for pw()-style use.
void anisoFactorOnGPU(const ScalarField& phi, ScalarField& aOut,
                      const AnisoParams& p);
void anisoFactorOnCPU(const ScalarField& phi, ScalarField& aOut,
                      const AnisoParams& p);

} // namespace PhiX
