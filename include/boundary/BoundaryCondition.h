#pragma once

#include "field/Field.h"

namespace PhiX {

// ---------------------------------------------------------------------------
// Axis and Side enumerations
// ---------------------------------------------------------------------------

enum class Axis { X = 0, Y = 1, Z = 2 };
enum class Side { LOW, HIGH, BOTH };

// ---------------------------------------------------------------------------
// BoundaryCondition  --  abstract base class
//
// Concrete subclasses implement applyOnCPU / applyOnGPU.
// Both methods update f.curr (CPU) or f.d_curr (GPU).
// Call them after each time step to keep ghost cells consistent.
// ---------------------------------------------------------------------------

class BoundaryCondition {
public:
    Axis axis;
    Side side;

    BoundaryCondition(Axis axis, Side side) : axis(axis), side(side) {}
    virtual ~BoundaryCondition() = default;

    // Update ghost cells of f.curr on the CPU
    virtual void applyOnCPU(Field& f) const = 0;

    // Update ghost cells of f.d_curr on the GPU
    // (HOST function; launches __global__ kernel internally)
    virtual void applyOnGPU(Field& f) const = 0;
};

} // namespace PhiX
