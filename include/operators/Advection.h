#pragma once

// ---------------------------------------------------------------------------
// Advection.h — first-order upwind (donor-cell) advection term.
//
//   adv(u, f)  →  Term representing  coeff · (u · ∇f)
//
// Per axis, the one-sided difference is chosen by the local velocity sign:
//
//   u_x > 0 :  ∂f/∂x ≈ (f[i] - f[i-1]) / dx     (backward / donor cell)
//   u_x < 0 :  ∂f/∂x ≈ (f[i+1] - f[i]) / dx     (forward)
//
// Monotone (no new extrema) and stable for CFL = max|u|·dt/dx < 1, at the
// cost of first-order accuracy (numerical diffusion ~ |u|·dx/2).  Central
// differences applied to advection produce dispersive oscillations — use
// this term whenever a transport velocity enters the model.
//
// Sign convention: adv() is the mathematical term u·∇f.  For the advection
// equation ∂f/∂t + u·∇f = 0 write   eq.setRHS(-1.0 * adv(u, f));
//
// The velocity is read at the cell centre only (no halo needed on u);
// f requires ghost >= 1 with BCs applied as usual.
// ---------------------------------------------------------------------------

#include "equation/Term.h"
#include "field/ScalarField.h"
#include "field/VectorField.h"

namespace PhiX {

// coeff · (u · ∇f), first-order upwind.  u must have >= mesh.dim components
// on the same mesh as f.
Term adv(const VectorField& u, const ScalarField& f, double coeff = 1.0);

} // namespace PhiX
