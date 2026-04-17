#pragma once

#include "boundary/BoundaryCondition.h"

namespace PhiX {

// ---------------------------------------------------------------------------
// FixedBC  (Dirichlet)
//
// Sets all ghost cells on the specified side to a constant value.
// Enforces phi = value at the boundary via ghost-cell extrapolation:
//
//   f[-1, j, k] = 2*value - f[0, j, k]   <- linear extrapolation (order-2)
//
// Currently implemented as constant fill (f[-g]=value) — sufficient for
// first-order stencils; can be upgraded to linear extrapolation later.
// ---------------------------------------------------------------------------

class FixedBC : public BoundaryCondition {
public:
    double value;

    FixedBC(Axis axis, Side side, double value);

    using BoundaryCondition::applyOnCPU;
    using BoundaryCondition::applyOnGPU;

    void applyOnCPU(ScalarField& f) const override;
    void applyOnGPU(ScalarField& f) const override;
};

} // namespace PhiX
