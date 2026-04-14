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
#include "field/ScalarField.h"

namespace PhiX {

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

} // namespace PhiX
