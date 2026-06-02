// ---------------------------------------------------------------------------
// EvalPlan.cu — Lowering pass: ExprTree → EvalPlan.
//
// See include/equation/EvalPlan.h for the design overview.
// ---------------------------------------------------------------------------

#include "equation/EvalPlan.h"
#include "equation/TermPW.inl"   // pw<Functor>() template definitions
#include "equation/FieldOps.inl" // detail::termTimesTerm / termTimesField

#include <stdexcept>
#include <string>
#include <vector>

namespace PhiX {

// ===========================================================================
// EvalPlan::execute / executeCPU
// ===========================================================================

void EvalPlan::execute(ScalarField& rhs, ScratchPool& pool) const {
    for (const auto& s : steps) {
        if (!s.gpu_launcher)
            throw std::runtime_error("EvalPlan::execute: a step has no GPU launcher");
        s.gpu_launcher(rhs.d_curr, s.coeff, pool);
    }
}

void EvalPlan::executeCPU(ScalarField& rhs, ScratchPool& pool) const {
    for (const auto& s : steps) {
        if (!s.cpu_kernel)
            throw std::runtime_error("EvalPlan::executeCPU: a step has no CPU kernel");
        s.cpu_kernel(rhs.curr.data(), s.coeff, pool);
    }
}

// ===========================================================================
// Internal lowering helpers
// ===========================================================================

// Convert an EvalStep.kind from a Term's type.
static EvalStep::Kind kindFromTerm(const Term& t) {
    return (t.type == TermType::LAPLACIAN || t.type == TermType::GRADIENT)
               ? EvalStep::Kind::STENCIL
               : EvalStep::Kind::LOCAL;
}

// Wrap a Term into an EvalStep.
static EvalStep stepFromTerm(Term t) {
    EvalStep s;
    s.kind         = kindFromTerm(t);
    s.coeff        = t.coeff;
    s.field        = t.field;
    s.gpu_launcher = std::move(t.gpu_launcher);
    s.cpu_kernel   = std::move(t.cpu_kernel);
    return s;
}

// Forward declarations of the two recursive helpers.
static void     lowerToSteps  (const ExprNode*, double coeff,
                                const ScalarField* layout,
                                std::vector<EvalStep>& out);
static RHSExpr  lowerToRHSExpr(const ExprNode*, double coeff,
                                const ScalarField* layout);

// ---------------------------------------------------------------------------
// lowerToRHSExpr — lower a subtree to an RHSExpr (multiple Terms).
// Used when a subtree appears as an operand inside ExprMul.
// ---------------------------------------------------------------------------
static RHSExpr lowerToRHSExpr(const ExprNode* n, double coeff,
                               const ScalarField* layout)
{
    // ExprAdd flattens directly.
    if (auto* add = dynamic_cast<const ExprAdd*>(n)) {
        RHSExpr out;
        out += lowerToRHSExpr(add->left.get(),  coeff, layout);
        out += lowerToRHSExpr(add->right.get(), coeff, layout);
        return out;
    }

    // All other nodes: lower to EvalSteps, then pack into Terms.
    std::vector<EvalStep> sub_steps;
    lowerToSteps(n, coeff, layout, sub_steps);

    RHSExpr out;
    for (auto& s : sub_steps) {
        Term t;
        t.type         = (s.kind == EvalStep::Kind::STENCIL)
                             ? TermType::LAPLACIAN : TermType::COMPOSITE;
        t.coeff        = s.coeff;
        t.field        = s.field;
        t.gpu_launcher = std::move(s.gpu_launcher);
        t.cpu_kernel   = std::move(s.cpu_kernel);
        out.terms.push_back(std::move(t));
    }
    return out;
}

// ---------------------------------------------------------------------------
// lowerToSteps — main recursive lowering function.
// Appends EvalSteps to `out`.
// `coeff` is the accumulated outer coefficient.
// `layout` is the representative ScalarField for mesh/ghost queries (may be
// nullptr; caller must guarantee it is set for nodes that need it).
// ---------------------------------------------------------------------------
static void lowerToSteps(const ExprNode* n, double coeff,
                         const ScalarField* layout,
                         std::vector<EvalStep>& out)
{
    if (!n) return;

    // ── ExprScale ────────────────────────────────────────────────────────────
    if (auto* sc = dynamic_cast<const ExprScale*>(n)) {
        // Update layout if the child provides one.
        const ScalarField* child_lay = sc->child->repField();
        lowerToSteps(sc->child.get(), coeff * sc->coeff,
                     child_lay ? child_lay : layout, out);
        return;
    }

    // ── ExprNeg ──────────────────────────────────────────────────────────────
    if (auto* neg = dynamic_cast<const ExprNeg*>(n)) {
        lowerToSteps(neg->child.get(), -coeff, layout, out);
        return;
    }

    // ── ExprAdd ──────────────────────────────────────────────────────────────
    if (auto* add = dynamic_cast<const ExprAdd*>(n)) {
        lowerToSteps(add->left.get(),  coeff, layout, out);
        lowerToSteps(add->right.get(), coeff, layout, out);
        return;
    }

    // ── ExprLeaf ─────────────────────────────────────────────────────────────
    // Produce: rhs[idx] += coeff * f[idx]
    // Implemented as pw(f, identity, coeff).
    if (auto* leaf = dynamic_cast<const ExprLeaf*>(n)) {
        Term t = pw(*leaf->field,
                    [] __host__ __device__ (double v) { return v; },
                    coeff);
        out.push_back(stepFromTerm(std::move(t)));
        return;
    }

    // ── ExprScalar ───────────────────────────────────────────────────────────
    // Produce: rhs[idx] += coeff * value   (constant added to every cell)
    // Requires a layout field for mesh dimensions.
    if (auto* cst = dynamic_cast<const ExprScalar*>(n)) {
        if (!layout)
            throw std::logic_error(
                "lowerExprTree: ExprScalar node with no layout field in context");
        double val = coeff * cst->value;
        Term t = pw(*layout,
                    [val] __host__ __device__ (double) { return val; },
                    1.0);
        out.push_back(stepFromTerm(std::move(t)));
        return;
    }

    // ── ExprMul ──────────────────────────────────────────────────────────────
    // Hadamard product: left * right.
    // Lowers each operand to an RHSExpr, then combines via termTimesTerm.
    if (auto* mul_n = dynamic_cast<const ExprMul*>(n)) {
        const ScalarField* lay = mul_n->repField();
        if (!lay) lay = layout;
        if (!lay)
            throw std::logic_error("lowerExprTree: ExprMul has no layout field");

        RHSExpr left_expr  = lowerToRHSExpr(mul_n->left.get(),  1.0, lay);
        RHSExpr right_expr = lowerToRHSExpr(mul_n->right.get(), 1.0, lay);

        Term mul_term = detail::termTimesTerm(left_expr, right_expr, *lay, coeff);
        out.push_back(stepFromTerm(std::move(mul_term)));
        return;
    }

    // ── ExprStencil ──────────────────────────────────────────────────────────
    // Stencil applied to a child sub-expression.
    // Stage 2: only plain ExprLeaf children are supported (no BC injection yet).
    if (auto* st = dynamic_cast<const ExprStencil*>(n)) {
        // Peel off any Scale/Neg wrappers around the child to extract the leaf
        // and accumulate any extra coefficient they carry.
        double leaf_coeff = coeff;
        const ExprNode* child = st->child.get();

        while (true) {
            if (auto* sc = dynamic_cast<const ExprScale*>(child)) {
                leaf_coeff *= sc->coeff;
                child = sc->child.get();
            } else if (auto* neg = dynamic_cast<const ExprNeg*>(child)) {
                leaf_coeff = -leaf_coeff;
                child = neg->child.get();
            } else {
                break;
            }
        }

        if (auto* leaf = dynamic_cast<const ExprLeaf*>(child)) {
            Term t;
            switch (st->kind) {
            case StencilKind::LAP:
                t = lap(*leaf->field, leaf_coeff);
                break;
            case StencilKind::GRAD:
                t = grad(*leaf->field, st->axis, leaf_coeff);
                break;
            case StencilKind::ISO_GRAD:
                t = iso_grad(*leaf->field, st->axis, leaf_coeff);
                break;
            default:
                throw std::logic_error(
                    "lowerExprTree: unsupported StencilKind in ExprStencil");
            }
            out.push_back(stepFromTerm(std::move(t)));
            return;
        }

        // Complex child → BC injection needed (Stage 3).
        throw std::logic_error(
            "lowerExprTree: ExprStencil applied to a composite child expression "
            "requires BC auto-injection (Stage 3). "
            "Use lap(expr, bcs) / grad(expr, axis, bcs) from the Term API instead.");
    }

    // ── ExprStencilBinary ────────────────────────────────────────────────────
    // GRAD_DOT: ∇f · ∇g.  Both children must be plain leaves.
    if (auto* sb = dynamic_cast<const ExprStencilBinary*>(n)) {
        if (sb->kind != StencilKind::GRAD_DOT)
            throw std::logic_error(
                "lowerExprTree: unsupported ExprStencilBinary kind");

        // Peel Scale/Neg wrappers.
        double leaf_coeff = coeff;
        const ExprNode* l_node = sb->left.get();
        const ExprNode* r_node = sb->right.get();

        while (true) {
            if (auto* sc = dynamic_cast<const ExprScale*>(l_node)) {
                leaf_coeff *= sc->coeff;
                l_node = sc->child.get();
            } else if (auto* neg = dynamic_cast<const ExprNeg*>(l_node)) {
                leaf_coeff = -leaf_coeff;
                l_node = neg->child.get();
            } else { break; }
        }
        while (true) {
            if (auto* sc = dynamic_cast<const ExprScale*>(r_node)) {
                leaf_coeff *= sc->coeff;
                r_node = sc->child.get();
            } else if (auto* neg = dynamic_cast<const ExprNeg*>(r_node)) {
                leaf_coeff = -leaf_coeff;
                r_node = neg->child.get();
            } else { break; }
        }

        auto* lf = dynamic_cast<const ExprLeaf*>(l_node);
        auto* rf = dynamic_cast<const ExprLeaf*>(r_node);

        if (!lf || !rf)
            throw std::logic_error(
                "lowerExprTree: GRAD_DOT requires both children to be plain "
                "ScalarField leaves (Stage 3 will lift this restriction)");

        Term t = grad_dot(*lf->field, *rf->field, leaf_coeff);
        out.push_back(stepFromTerm(std::move(t)));
        return;
    }

    // ── ExprPointwise1 ───────────────────────────────────────────────────────
    // Not yet implemented in any factory (Stage 2 scope: stencil + leaf ops).
    if (dynamic_cast<const ExprPointwise1*>(n)) {
        throw std::logic_error(
            "lowerExprTree: ExprPointwise1 lowering is not yet implemented. "
            "Use pw() from the Term API for pointwise user-defined transforms.");
    }

    throw std::logic_error(
        "lowerExprTree: unknown or unhandled ExprNode type encountered");
}

// ===========================================================================
// lowerExprTree — public entry point
// ===========================================================================
EvalPlan lowerExprTree(const ExprTree& tree) {
    validateGhostRequirements(tree);

    if (!tree.node)
        throw std::logic_error("lowerExprTree: empty ExprTree");

    const ScalarField* layout = tree.repField();

    EvalPlan plan;
    lowerToSteps(tree.node.get(), 1.0, layout, plan.steps);
    return plan;
}

} // namespace PhiX
