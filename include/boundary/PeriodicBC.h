#pragma once

#include "boundary/BoundaryCondition.h"

namespace PhiX {

// ---------------------------------------------------------------------------
// PeriodicBC
//
// Wraps ghost cells around the opposite physical boundary.
// Side is always BOTH (periodicity requires both sides simultaneously).
//
// Math (axis X, ghost = g):
//   f[-g, j, k] = f[nx-g, j, k]     (low ghost copies from far end)
//   f[nx+g-1, j, k] = f[g-1, j, k] (high ghost copies from near start)
// ---------------------------------------------------------------------------

class PeriodicBC : public BoundaryCondition {
public:
    explicit PeriodicBC(Axis axis);

    using BoundaryCondition::applyOnCPU;
    using BoundaryCondition::applyOnGPU;

    void applyOnCPU(ScalarField& f) const override;
    void applyOnGPU(ScalarField& f) const override;
};

} // namespace PhiX
