#pragma once

#include "equation/Term.h"
#include "field/ScalarField.h"

#include <string>
#include <vector>

namespace PhiX {

// ---------------------------------------------------------------------------
// Equation
//
// Describes  d(unknown)/dt = RHS
// where RHS is an RHSExpr built from lap(), grad(), pw() terms.
//
// Workflow:
//   1. Construct with the unknown Field and any auxiliary fields / params.
//   2. Call setRHS(expr) once to define the equation.
//   3. Pass to a Solver, which calls computeRHS(rhs_field) each time step.
//
// computeRHS writes into the PHYSICAL cells of rhs (zero-initialises first),
// leaving ghost cells of rhs at zero — the Solver applies BCs before calling
// this function and is responsible for rhs memory lifecycle.
// ---------------------------------------------------------------------------

class Equation {
public:
    std::string              name;
    ScalarField&                   unknown;     // the d/dt field (non-owning ref)
    std::vector<ScalarField*>      auxFields;   // other fields used on RHS (non-owning)

    explicit Equation(ScalarField& unknown, const std::string& name = "");

    // -----------------------------------------------------------------------
    // Set the RHS expression.  Can be called again to update mid-simulation.
    // -----------------------------------------------------------------------
    void setRHS(const RHSExpr& expr);
    void setRHS(const Term& t);   // convenience: single-term RHS

    // -----------------------------------------------------------------------
    // Evaluate RHS into rhs.d_curr on GPU.
    // rhs must already have device memory allocated (rhs.allocDevice()).
    // -----------------------------------------------------------------------
    void computeRHS(ScalarField& rhs) const;

    // CPU fallback (useful for testing without a GPU)
    void computeRHSCPU(ScalarField& rhs) const;

    // -----------------------------------------------------------------------
    // Convenience: return true if RHS has been set
    // -----------------------------------------------------------------------
    bool hasRHS() const { return !rhs_expr_.terms.empty(); }

private:
    RHSExpr rhs_expr_;
};

} // namespace PhiX
