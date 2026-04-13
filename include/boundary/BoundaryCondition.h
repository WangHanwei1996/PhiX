#pragma once

#include "field/ScalarField.h"
#include "field/VectorField.h"

namespace PhiX {

// ---------------------------------------------------------------------------
// Axis and Side enumerations
// ---------------------------------------------------------------------------

enum class Axis { X = 0, Y = 1, Z = 2 };
enum class Side { LOW, HIGH, BOTH };

// ---------------------------------------------------------------------------
// BoundaryCondition  --  abstract base class
//
// Concrete subclasses implement applyOnCPU / applyOnGPU for ScalarField.
// VectorField overloads have default implementations that apply the BC
// independently to every component.  Override the vector overloads in a
// subclass when components must be coupled (e.g. slip boundaries).
// ---------------------------------------------------------------------------

class BoundaryCondition {
public:
    Axis axis;
    Side side;

    BoundaryCondition(Axis axis, Side side) : axis(axis), side(side) {}
    virtual ~BoundaryCondition() = default;

    // --- ScalarField interface (must be implemented by subclasses) ----------
    virtual void applyOnCPU(ScalarField& f) const = 0;
    virtual void applyOnGPU(ScalarField& f) const = 0;

    // --- VectorField interface (default: apply to each component) -----------
    virtual void applyOnCPU(VectorField& vf) const {
        for (int c = 0; c < vf.nComponents(); ++c) applyOnCPU(vf[c]);
    }
    virtual void applyOnGPU(VectorField& vf) const {
        for (int c = 0; c < vf.nComponents(); ++c) applyOnGPU(vf[c]);
    }
};

} // namespace PhiX
