// ---------------------------------------------------------------------------
// FieldOps.inl — Arithmetic operator overloads for ScalarField.
//
// Enables natural DSL expressions such as:
//
//   eq.setRHS(c * eta + 2.0 * c - lap(c));
//   eq.setRHS(c * c - kappa * lap(c));
//
// Included automatically at the end of TermPW.inl after the PhiX namespace
// is closed.  Do NOT include directly.
//
// Requires nvcc (uses __host__ __device__ lambdas; --expt-extended-lambda).
// ---------------------------------------------------------------------------

#pragma once

#include "equation/Term.h"
#include "equation/Equation.h"   // ScratchPool
#include "field/ScalarField.h"

#include <cuda_runtime.h>
#include <cstring>
#include <stdexcept>

namespace PhiX {

// Forward declarations of the mul_accumulate helpers (defined in
// src/equation/Equation.cu).  Used by Term/RHSExpr multiplication overloads.
namespace detail {
void mulAccumulateGPU(double* d_rhs,
                      const double* d_s1, const double* d_s2,
                      double coeff,
                      int nx, int ny, int nz,
                      int sx, int sy, int g);
void mulAccumulateCPU(double* rhs,
                      const double* s1, const double* s2,
                      double coeff,
                      int nx, int ny, int nz,
                      int sx, int sy, int g);

void materialiseGPU(const RHSExpr&, double*, std::size_t, ScratchPool&);
void materialiseGPU(const Term&,    double*, std::size_t, ScratchPool&);
void materialiseCPU(const RHSExpr&, double*, std::size_t, ScratchPool&);
void materialiseCPU(const Term&,    double*, std::size_t, ScratchPool&);

const ScalarField* repField(const Term&);
const ScalarField* repField(const RHSExpr&);
} // namespace detail

// ===========================================================================
// Field × Field  (binary pointwise) → Term
// ===========================================================================

// f1 * f2  →  term: rhs[idx] += f1[idx] * f2[idx]
inline Term operator*(const ScalarField& f1, const ScalarField& f2) {
    return pw(f1, f2,
              [] __host__ __device__ (double a, double b) { return a * b; });
}

// f1 + f2  →  term: rhs[idx] += f1[idx] + f2[idx]
inline Term operator+(const ScalarField& f1, const ScalarField& f2) {
    return pw(f1, f2,
              [] __host__ __device__ (double a, double b) { return a + b; });
}

// f1 - f2  →  term: rhs[idx] += f1[idx] - f2[idx]
inline Term operator-(const ScalarField& f1, const ScalarField& f2) {
    return pw(f1, f2,
              [] __host__ __device__ (double a, double b) { return a - b; });
}

// ===========================================================================
// Field × scalar  (scale / offset) → Term
// ===========================================================================

// s * f  →  term: rhs[idx] += s * f[idx]
//   NOTE: This is *different* from s * Term (which multiplies coeff).
//   Here s is an additive-constant-free scale baked into pw's identity func.
//   For coefficients that multiply an existing Term, prefer:  s * lap(f)
//   This overload is useful when a ScalarField expression is the LHS of *:
//     3.0 * c               → pw(c, identity, 3.0)
//     c * M                 → pw(c, identity, M)
inline Term operator*(double s, const ScalarField& f) {
    return pw(f, [] __host__ __device__ (double v) { return v; }, s);
}
inline Term operator*(const ScalarField& f, double s) {
    return pw(f, [] __host__ __device__ (double v) { return v; }, s);
}

// f + s  →  term: rhs[idx] += f[idx] + s
inline Term operator+(const ScalarField& f, double s) {
    return pw(f, [s] __host__ __device__ (double v) { return v + s; });
}
inline Term operator+(double s, const ScalarField& f) {
    return pw(f, [s] __host__ __device__ (double v) { return v + s; });
}

// f - s  →  term: rhs[idx] += f[idx] - s
inline Term operator-(const ScalarField& f, double s) {
    return pw(f, [s] __host__ __device__ (double v) { return v - s; });
}

// s - f  →  term: rhs[idx] += s - f[idx]
inline Term operator-(double s, const ScalarField& f) {
    return pw(f, [s] __host__ __device__ (double v) { return s - v; });
}

// unary -f  →  term: rhs[idx] += -f[idx]
inline Term operator-(const ScalarField& f) {
    return pw(f, [] __host__ __device__ (double v) { return -v; });
}

// ===========================================================================
// Term × Term, Term × ScalarField, RHSExpr × ScalarField, RHSExpr × RHSExpr
//
// Implemented via lazy materialisation:
//   1. Each operand that is a Term/RHSExpr is evaluated into a scratch buffer
//      from the Equation's ScratchPool.
//   2. A pointwise multiply-accumulate kernel combines the two buffers
//      (or one buffer + one ScalarField) into the rhs.
//
// All operations preserve coeff arithmetic: result.coeff is set to 1.0 and
// each operand contributes its own coeff during materialisation.  Subsequent
// `s * (a*b)` works as usual via `Term::operator*(double)`.
//
// TODO(vector): mirror these for VectorRHSExpr × {scalar Term, scalar Field,
// component-wise Vec×Vec, dot product → scalar Term} when vector solvers
// need them.
// ===========================================================================

namespace detail {

// Build a Term that captures (lhs, rhs_field) and emits
//   rhs[idx] += coeff * materialise(lhs)[idx] * field[idx]
// `field` must already be on device when the launcher is invoked.
template<typename LhsExpr>
inline Term termTimesField(LhsExpr lhs,
                           const ScalarField& field,
                           const ScalarField& layout,
                           double coeff)
{
    Term out;
    out.type  = TermType::COMPOSITE;
    out.coeff = coeff;
    out.field = &layout;

    int nx = layout.mesh.n[0], ny = layout.mesh.n[1], nz = layout.mesh.n[2];
    int sx = layout.storedDims[0], sy = layout.storedDims[1];
    int g  = layout.ghost;
    std::size_t storedSize = layout.storedSize;
    const ScalarField* pf = &field;

    out.gpu_launcher = [lhs, pf, nx, ny, nz, sx, sy, g, storedSize]
                       (double* d_rhs, double c, ScratchPool& pool) {
        double* d_scratch = pool.acquireDevice(storedSize);
        materialiseGPU(lhs, d_scratch, storedSize, pool);
        const double* d_f = pf->d_curr;
        if (!d_f)
            throw std::runtime_error("Term*Field GPU: field not on device");
        mulAccumulateGPU(d_rhs, d_scratch, d_f, c, nx, ny, nz, sx, sy, g);
    };

    out.cpu_kernel = [lhs, pf, nx, ny, nz, sx, sy, g, storedSize]
                     (double* rhs, double c, ScratchPool& pool) {
        double* h_scratch = pool.acquireHost(storedSize);
        materialiseCPU(lhs, h_scratch, storedSize, pool);
        const double* h_f = pf->curr.data();
        mulAccumulateCPU(rhs, h_scratch, h_f, c, nx, ny, nz, sx, sy, g);
    };
    return out;
}

// Build a Term that captures (lhs, rhs) where both are expressions and emits
//   rhs[idx] += coeff * materialise(lhs)[idx] * materialise(rhs)[idx]
template<typename LhsExpr, typename RhsExpr>
inline Term termTimesTerm(LhsExpr lhs, RhsExpr rhs,
                          const ScalarField& layout,
                          double coeff)
{
    Term out;
    out.type  = TermType::COMPOSITE;
    out.coeff = coeff;
    out.field = &layout;

    int nx = layout.mesh.n[0], ny = layout.mesh.n[1], nz = layout.mesh.n[2];
    int sx = layout.storedDims[0], sy = layout.storedDims[1];
    int g  = layout.ghost;
    std::size_t storedSize = layout.storedSize;

    out.gpu_launcher = [lhs, rhs, nx, ny, nz, sx, sy, g, storedSize]
                       (double* d_rhs, double c, ScratchPool& pool) {
        double* s1 = pool.acquireDevice(storedSize);
        materialiseGPU(lhs, s1, storedSize, pool);
        double* s2 = pool.acquireDevice(storedSize);
        materialiseGPU(rhs, s2, storedSize, pool);
        mulAccumulateGPU(d_rhs, s1, s2, c, nx, ny, nz, sx, sy, g);
    };

    out.cpu_kernel = [lhs, rhs, nx, ny, nz, sx, sy, g, storedSize]
                     (double* h_rhs, double c, ScratchPool& pool) {
        double* s1 = pool.acquireHost(storedSize);
        materialiseCPU(lhs, s1, storedSize, pool);
        double* s2 = pool.acquireHost(storedSize);
        materialiseCPU(rhs, s2, storedSize, pool);
        mulAccumulateCPU(h_rhs, s1, s2, c, nx, ny, nz, sx, sy, g);
    };
    return out;
}

} // namespace detail

// --- Term × ScalarField / ScalarField × Term --------------------------------
inline Term operator*(const Term& t, const ScalarField& f) {
    const ScalarField* layout = detail::repField(t);
    return detail::termTimesField<Term>(t, f, *layout, 1.0);
}
inline Term operator*(const ScalarField& f, const Term& t) {
    return t * f;
}

// --- Term × Term ------------------------------------------------------------
inline Term operator*(const Term& a, const Term& b) {
    const ScalarField* layout = detail::repField(a);
    return detail::termTimesTerm<Term, Term>(a, b, *layout, 1.0);
}

// --- RHSExpr × ScalarField / ScalarField × RHSExpr --------------------------
inline Term operator*(const RHSExpr& e, const ScalarField& f) {
    const ScalarField* layout = detail::repField(e);
    return detail::termTimesField<RHSExpr>(e, f, *layout, 1.0);
}
inline Term operator*(const ScalarField& f, const RHSExpr& e) {
    return e * f;
}

// --- RHSExpr × Term / Term × RHSExpr ----------------------------------------
inline Term operator*(const RHSExpr& a, const Term& b) {
    const ScalarField* layout = detail::repField(a);
    return detail::termTimesTerm<RHSExpr, Term>(a, b, *layout, 1.0);
}
inline Term operator*(const Term& a, const RHSExpr& b) {
    const ScalarField* layout = detail::repField(a);
    return detail::termTimesTerm<Term, RHSExpr>(a, b, *layout, 1.0);
}

// --- RHSExpr × RHSExpr ------------------------------------------------------
inline Term operator*(const RHSExpr& a, const RHSExpr& b) {
    const ScalarField* layout = detail::repField(a);
    return detail::termTimesTerm<RHSExpr, RHSExpr>(a, b, *layout, 1.0);
}

// ===========================================================================
// VectorRHSExpr × {Term, ScalarField}  -- per-component multiplication.
// Needed so that  D_term * grad(mu, bcs)  yields a VectorRHSExpr suitable
// for div(...).
//
// TODO(vector): add VectorField overloads, dot product (Vec . Vec -> Term),
// and Vec × Vec component-wise once vector solvers need them.
// ===========================================================================

inline VectorRHSExpr operator*(const Term& t, const VectorRHSExpr& v) {
    VectorRHSExpr out(v.nComponents());
    for (int c = 0; c < v.nComponents(); ++c)
        out[c] = RHSExpr(t * RHSExpr(v[c]));   // Term * RHSExpr
    return out;
}
inline VectorRHSExpr operator*(const VectorRHSExpr& v, const Term& t) {
    return t * v;
}

inline VectorRHSExpr operator*(const ScalarField& f, const VectorRHSExpr& v) {
    VectorRHSExpr out(v.nComponents());
    for (int c = 0; c < v.nComponents(); ++c)
        out[c] = RHSExpr(f * RHSExpr(v[c]));   // Field * RHSExpr
    return out;
}
inline VectorRHSExpr operator*(const VectorRHSExpr& v, const ScalarField& f) {
    return f * v;
}

inline VectorRHSExpr operator*(const RHSExpr& e, const VectorRHSExpr& v) {
    VectorRHSExpr out(v.nComponents());
    for (int c = 0; c < v.nComponents(); ++c)
        out[c] = RHSExpr(e * RHSExpr(v[c]));   // RHSExpr * RHSExpr
    return out;
}
inline VectorRHSExpr operator*(const VectorRHSExpr& v, const RHSExpr& e) {
    return e * v;
}

} // namespace PhiX
