#pragma once

#include "equation/Equation.h"
#include "boundary/BoundaryCondition.h"
#include "field/Field.h"

#include <functional>
#include <string>
#include <vector>

namespace PhiX {

// ---------------------------------------------------------------------------
// TimeScheme — selectable explicit time integration strategy
// ---------------------------------------------------------------------------
enum class TimeScheme {
    EULER,   // Forward Euler:  phi += dt * rhs
    RK4      // Classical 4th-order Runge-Kutta
};

// ---------------------------------------------------------------------------
// Solver
//
// Drives explicit time advancement of one Equation.
//
// Responsibilities:
//   1. Apply boundary conditions (ghost cell update) before each RHS eval.
//   2. Call equation.computeRHS(rhs) to fill the right-hand side field.
//   3. Advance unknown field by dt using the chosen TimeScheme.
//   4. Call equation.unknown.advanceTimeLevelGPU() to shift curr -> prev.
//
// The Solver owns the internal scratch fields (rhs, k1..k4 for RK4).
// All other objects (Equation, BCs) are non-owning references/pointers.
//
// Usage:
//   Solver solver(eq, {&bc_x, &bc_y}, dt, TimeScheme::EULER);
//   solver.run(1000, [&](const Solver& s) {
//       if (s.step % 100 == 0) phi.downloadCurrFromDevice(), phi.write(...);
//   });
// ---------------------------------------------------------------------------

class Solver {
public:
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------
    Solver(Equation&                           equation,
           std::vector<BoundaryCondition*>     bcs,
           double                              dt,
           TimeScheme                          scheme = TimeScheme::EULER);

    // Non-copyable (owns device scratch memory via Field)
    Solver(const Solver&)            = delete;
    Solver& operator=(const Solver&) = delete;

    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------
    double     dt;            // time step (may be changed between steps)
    TimeScheme scheme;        // integration scheme
    int        step  = 0;     // current step counter (incremented by advance())
    double     time  = 0.0;   // current simulation time

    // -----------------------------------------------------------------------
    // Single time step (GPU path)
    // -----------------------------------------------------------------------
    void advance();

    // CPU fallback (no GPU required; useful for unit tests)
    void advanceCPU();

    // -----------------------------------------------------------------------
    // Run multiple steps
    //
    // callbackEvery: fire callback every N steps (0 = never)
    // callback:      receives a const ref to this Solver after the step
    // -----------------------------------------------------------------------
    void run(int nSteps,
             int callbackEvery = 0,
             std::function<void(const Solver&)> callback = nullptr);

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------
    const Field& unknown() const { return equation_.unknown; }
    Field&       unknown()       { return equation_.unknown; }

    const Equation&                       equation() const { return equation_; }
    const std::vector<BoundaryCondition*>& bcs()     const { return bcs_; }

private:
    Equation&                       equation_;
    std::vector<BoundaryCondition*> bcs_;

    // Scratch fields — allocated in constructor, same mesh & ghost as unknown
    Field rhs_;   // used by Euler and each RK4 stage accumulator
    // RK4 stage vectors k1..k4 (only allocated when scheme == RK4)
    Field k1_, k2_, k3_, k4_;
    // RK4 also needs a temporary copy of phi to evaluate mid-stages
    Field phi_tmp_;

    bool use_rk4_ = false;

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------
    void applyBCsGPU();
    void applyBCsCPU();

    // Euler: unknown.d_curr += dt * rhs.d_curr  (elementwise GPU kernel)
    void eulerUpdateGPU();
    void eulerUpdateCPU();

    // RK4 sub-steps
    void rk4AdvanceGPU();
    void rk4AdvanceCPU();
};

} // namespace PhiX
