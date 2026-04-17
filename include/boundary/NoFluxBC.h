#pragma once

#include "boundary/BoundaryCondition.h"

namespace PhiX {

// ---------------------------------------------------------------------------
// NoFluxBC  (zero-gradient / Neumann)
//
// Sets ghost cells equal to the nearest physical boundary cell,
// giving a zero normal gradient: d(phi)/dn = 0.
//
// Math (axis X, LOW side, ghost = g):
//   f[-1, j, k] = f[0, j, k]
//   f[-2, j, k] = f[0, j, k]   <- constant extrapolation for all layers
//
// Math (axis X, HIGH side):
//   f[nx, j, k]   = f[nx-1, j, k]
//   f[nx+1, j, k] = f[nx-1, j, k]
// ---------------------------------------------------------------------------

class NoFluxBC : public BoundaryCondition {
public:
    NoFluxBC(Axis axis, Side side = Side::BOTH);

    using BoundaryCondition::applyOnCPU;
    using BoundaryCondition::applyOnGPU;

    void applyOnCPU(ScalarField& f) const override;
    void applyOnGPU(ScalarField& f) const override;
};

} // namespace PhiX
