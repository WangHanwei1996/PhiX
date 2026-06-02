#pragma once

// ---------------------------------------------------------------------------
// EvalPlan.h — Lowered execution plan for ExprTree (Stage 2 of DSL refactor).
//
// An EvalPlan is produced by lowerExprTree() and executed by
// Equation::computeRHS() when the equation was set via setRHS(ExprTree).
//
// Stage 2 design (Strategy A — runtime fusion):
//   The lowering pass walks the ExprTree, converts each node to one or more
//   Term-based launchers (the existing GPU kernel infrastructure), and stores
//   them as an ordered list of EvalSteps.
//
//   Key structural split vs the raw RHSExpr approach:
//     • Steps are tagged as LOCAL or STENCIL so future passes can fuse/reorder.
//     • Ghost requirements are validated at lower-time, not at run-time.
//     • BC injection point is explicit (step holds a BC list, Stage 3 fills it).
//
//   Stage 2 execution is numerically identical to the old RHSExpr loop:
//   each step's launcher is called sequentially, accumulating into rhs.d_curr.
//   Actual kernel fusion is deferred to Stage 5 (Strategy B).
//
// Future stages:
//   Stage 3: lowerExprTree gains a BcMap parameter so STENCIL steps on
//            materialized expressions can auto-inject BCs.
//   Stage 4: EvalPlan holds a cudaStream_t; execute() drops DeviceSynchronize.
//   Stage 5: LOCAL steps are fused into single templated kernels (Strategy B).
// ---------------------------------------------------------------------------

#include "equation/Term.h"
#include "equation/Expr.h"

#include <string>
#include <vector>

namespace PhiX {

// ===========================================================================
// EvalStep — one unit of work in the plan
// ===========================================================================
struct EvalStep {
    enum class Kind {
        LOCAL,    // pointwise, no halo: leaf scale / pw / mul etc.
        STENCIL,  // needs ghost cells: lap, grad, grad_dot
    };

    Kind   kind  = Kind::LOCAL;
    double coeff = 1.0;

    // GPU launcher and CPU fallback — same signature as Term::gpu_launcher.
    TermLauncher gpu_launcher;
    TermLauncher cpu_kernel;

    // Representative field (for layout queries and nullptr checks).
    const ScalarField* field = nullptr;

    // [Stage 3] BCs to apply to the scratch buffer before the stencil step.
    // Populated by lowerExprTree when a BcMap is provided.
    std::vector<BoundaryCondition*> bcs;
};

// ===========================================================================
// EvalPlan — ordered list of EvalSteps
// ===========================================================================
class EvalPlan {
public:
    EvalPlan() = default;

    std::vector<EvalStep> steps;

    bool empty() const { return steps.empty(); }

    // -----------------------------------------------------------------------
    // execute — accumulate all steps into rhs.d_curr on GPU.
    // rhs must already be zeroed and have device memory allocated.
    // -----------------------------------------------------------------------
    void execute(ScalarField& rhs, ScratchPool& pool) const;

    // -----------------------------------------------------------------------
    // executeCPU — CPU fallback (no device memory required).
    // rhs.curr must already be zeroed.
    // -----------------------------------------------------------------------
    void executeCPU(ScalarField& rhs, ScratchPool& pool) const;
};

// ===========================================================================
// lowerExprTree — convert an ExprTree into an EvalPlan
//
// Walks the tree recursively and produces an ordered list of EvalSteps backed
// by the existing Term-launcher infrastructure.
//
// Restrictions (lifted in later stages):
//   • ExprStencil nodes whose child is NOT a plain ExprLeaf require BC
//     injection (Stage 3). For now they throw std::logic_error with a
//     message directing the user to use lap(expr, bcs) from the Term API.
//   • ExprScalar (bare constant) nodes without a companion field in the same
//     subtree throw std::logic_error (constant-only RHS is nonsensical).
//
// Preconditions:
//   • validateGhostRequirements(tree) must have been called before lowering.
//     lowerExprTree calls it internally; callers need not call it separately.
// ===========================================================================
EvalPlan lowerExprTree(const ExprTree& tree);

} // namespace PhiX
