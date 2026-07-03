#pragma once

// ---------------------------------------------------------------------------
// LinearSolver.h — matrix-free linear-solver layer (GPU).
//
// The layer solves systems of the form
//
//     A x = b,     A = I − σ·L
//
// where L is a linear "stiff generator" represented matrix-free by a
// LinearOperator (no matrix is ever assembled — L is applied as stencil
// kernels).  This is exactly the system a backward-Euler / semi-implicit
// step needs (σ = dt), and keeping σ OUTSIDE the operator makes adaptive
// time steps free: the same operator serves every dt.
//
// Provided operators:
//   LaplacianOp  : L = D·∇²      (CD2; Allen-Cahn / diffusion stiff part)
//   BiharmonicOp : L = −G·∇²∇²   (two CD2 passes; Cahn-Hilliard ∇⁴ part,
//                                 A = I + σ·G·∇⁴)
//
// Both refresh the ghost cells of their input with the boundary conditions
// given at construction before applying the stencil.
//
// BC RESTRICTION (linearity): CG requires A to be linear, so only
// homogeneous BCs are admissible on the operator — PeriodicBC and NoFluxBC
// qualify; FixedBC only with value 0 (lift an inhomogeneous Dirichlet value
// into b yourself).  For pure periodic/no-flux problems A = I − σL is SPD
// and CG is the right Krylov method.
//
// Solver: ConjugateGradient — owns its scratch fields (construct once,
// solve every step).  Reductions (dot products) accumulate in double
// regardless of PHIX_PRECISION.
//
// Typical semi-implicit usage (see solver/SemiImplicitSolver.h for the
// packaged integrator):
//
//     LaplacianOp L(D, {&bcx, &bcy});
//     ConjugateGradient cg(mesh, ghost);
//     // per step:  b = phi + dt*N(phi);  solve (I - dt*L) phi_new = b
//     auto res = cg.solve(L, dt, phi, b, 1e-8, 500);
// ---------------------------------------------------------------------------

#include "core/Real.h"
#include "field/ScalarField.h"
#include "boundary/BoundaryCondition.h"

#include <memory>
#include <vector>

namespace PhiX {

// ===========================================================================
// LinearOperator — matrix-free y = L(x) over physical cells.
// apply() may mutate x's GHOST cells (BC refresh); physical x is untouched.
// ===========================================================================
class LinearOperator {
public:
    virtual ~LinearOperator() = default;

    virtual void apply(ScalarField& x, ScalarField& y) = 0;

    // Halo width the operator's stencil needs on x and y.
    virtual int ghostRequired() const { return 1; }
};

// ===========================================================================
// LaplacianOp — L = D·∇² (CD2)
// ===========================================================================
class LaplacianOp : public LinearOperator {
public:
    LaplacianOp(double D, std::vector<BoundaryCondition*> bcs);

    void apply(ScalarField& x, ScalarField& y) override;

    double D() const { return D_; }

private:
    double                          D_;
    std::vector<BoundaryCondition*> bcs_;
};

// ===========================================================================
// BiharmonicOp — L = −G·∇²(∇²) so that A = I − σL = I + σ·G·∇⁴.
// bcsX   : BCs of the solution field (e.g. no-flux on c: ∂c/∂n = 0)
// bcsLap : BCs of the intermediate ∇²x (for Cahn-Hilliard the natural
//          choice is no-flux again: ∂(∇²c)/∂n = 0, i.e. zero μ-flux)
// The intermediate scratch field is allocated lazily on first apply.
// ===========================================================================
class BiharmonicOp : public LinearOperator {
public:
    BiharmonicOp(double G,
                 std::vector<BoundaryCondition*> bcsX,
                 std::vector<BoundaryCondition*> bcsLap);

    void apply(ScalarField& x, ScalarField& y) override;

    double G() const { return G_; }

private:
    double                          G_;
    std::vector<BoundaryCondition*> bcsX_, bcsLap_;
    std::unique_ptr<ScalarField>    lap_;   // scratch: ∇²x
};

// ===========================================================================
// ConjugateGradient — solves (I − σ·L) x = b, matrix-free, on the GPU.
// ===========================================================================
class ConjugateGradient {
public:
    struct Result {
        int    iterations  = 0;
        double relResidual = 0.0;   // ‖r‖₂ / ‖b‖₂ at exit
        bool   converged   = false;
    };

    // Scratch fields (r, p, Lp) are allocated once for this mesh/ghost.
    ConjugateGradient(const Mesh& mesh, int ghost);

    ConjugateGradient(const ConjugateGradient&)            = delete;
    ConjugateGradient& operator=(const ConjugateGradient&) = delete;

    // x: initial guess in, solution out (its ghosts are mutated by L).
    // Throws std::runtime_error if maxIter is exhausted without reaching
    // relTol (set throwOnFail = false to get Result.converged == false
    // instead).
    Result solve(LinearOperator& L, double sigma,
                 ScalarField& x, const ScalarField& b,
                 double relTol = 1e-8, int maxIter = 500,
                 bool throwOnFail = true);

private:
    ScalarField r_, p_, Lp_;
};

} // namespace PhiX
