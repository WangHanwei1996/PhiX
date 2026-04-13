#pragma once

#include "equation/Equation.h"
#include "equation/Term.h"
#include "field/VectorField.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace PhiX {

// ---------------------------------------------------------------------------
// VectorEquation
//
// Describes  d(unknown)/dt = RHS  where unknown is a VectorField.
//
// Internally owns N scalar Equation objects (one per component).
// The RHS is expressed as a VectorRHSExpr where each component is an RHSExpr
// built from the usual lap / grad / pw / div / curl operators.
//
// Workflow:
//   1. Construct with the unknown VectorField.
//   2. Call setRHS(VectorRHSExpr) once to define the system.
//   3. Pass to a VectorSolver, which calls computeRHS(rhs_vf) each time step.
//
// computeRHS writes into the PHYSICAL cells of each component of rhs,
// zero-initialising first — same contract as scalar Equation::computeRHS.
// ---------------------------------------------------------------------------

class VectorEquation {
public:
    std::string    name;
    VectorField&   unknown;          // the d/dt field (non-owning ref)
    std::map<std::string, double> params;  // named physical constants

    explicit VectorEquation(VectorField& unknown,
                             const std::string& name = "");

    // -----------------------------------------------------------------------
    // Set the RHS expression.
    // expr.nComponents() must equal unknown.nComponents().
    // -----------------------------------------------------------------------
    void setRHS(const VectorRHSExpr& expr);

    // -----------------------------------------------------------------------
    // Evaluate RHS into rhs on GPU (each component's d_curr must be allocated
    // and rhs must have matching layout).
    // -----------------------------------------------------------------------
    void computeRHS(VectorField& rhs) const;

    // CPU fallback
    void computeRHSCPU(VectorField& rhs) const;

    // -----------------------------------------------------------------------
    // Access individual component equations (for advanced use)
    // -----------------------------------------------------------------------
    Equation&       componentEquation(int c);
    const Equation& componentEquation(int c) const;

    bool hasRHS() const;

private:
    std::vector<std::unique_ptr<Equation>> equations_;  // one per component
};

} // namespace PhiX
