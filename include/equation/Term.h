#pragma once

// ---------------------------------------------------------------------------
// Term.h — Expression DSL for composing RHS of time-evolution equations.
//
// Requires nvcc compilation (CUDA device code is referenced by templates).
//
// Typical usage:
//
//   double M     = eq.params["M"];
//   double kappa = eq.params["kappa"];
//
//   eq.setRHS(M * lap(phi) + M * pw(phi, [] __device__ (double p) {
//                 return p * (1.0 - p * p);   // Allen-Cahn bulk force
//             }));
// ---------------------------------------------------------------------------

#include "field/Field.h"

#include <functional>
#include <vector>

namespace PhiX {

// ---------------------------------------------------------------------------
// TermLauncher — host-side std::function that launches (or runs) one term.
//   args: (rhs_data, src_data, effective_coeff)
//   All mesh / ghost-layout info is captured at term-construction time.
// ---------------------------------------------------------------------------
using TermLauncher = std::function<void(double* /*rhs*/,
                                        const double* /*src*/,
                                        double /*coeff*/)>;

enum class TermType { LAPLACIAN, GRADIENT, POINTWISE };

// ---------------------------------------------------------------------------
// Term — one additive contribution to an RHS expression
// ---------------------------------------------------------------------------
struct Term {
    TermType      type  = TermType::LAPLACIAN;
    double        coeff = 1.0;
    const Field*  field = nullptr;   // source field (non-owning)
    int           axis  = 0;         // for GRADIENT: 0=x, 1=y, 2=z

    TermLauncher  gpu_launcher;   // host fn that launches GPU kernel
    TermLauncher  cpu_kernel;     // pure CPU fallback

    // Coefficient arithmetic — scale coeff, reuse launchers unchanged
    Term  operator*(double s) const { Term t = *this; t.coeff *= s; return t; }
    Term  operator/(double s) const { return *this * (1.0 / s); }
    Term  operator-()         const { return *this * (-1.0); }
};

inline Term operator*(double s, const Term& t) { return t * s; }
inline Term operator/(const Term& t, double s) { return t * (1.0 / s); }

// ---------------------------------------------------------------------------
// RHSExpr — ordered sum of Terms  (rhs = sum_i term_i)
// ---------------------------------------------------------------------------
struct RHSExpr {
    std::vector<Term> terms;

    RHSExpr() = default;
    explicit RHSExpr(const Term& t) { terms.push_back(t); }

    RHSExpr& operator+=(const Term& t)    { terms.push_back(t);  return *this; }
    RHSExpr& operator-=(const Term& t)    { terms.push_back(-t); return *this; }
    RHSExpr& operator+=(const RHSExpr& e) {
        terms.insert(terms.end(), e.terms.begin(), e.terms.end());
        return *this;
    }
    RHSExpr& operator-=(const RHSExpr& e) {
        for (const auto& t : e.terms) terms.push_back(-t);
        return *this;
    }

    RHSExpr operator+(const Term& t)    const { RHSExpr r=*this; r+=t; return r; }
    RHSExpr operator-(const Term& t)    const { RHSExpr r=*this; r-=t; return r; }
    RHSExpr operator+(const RHSExpr& e) const { RHSExpr r=*this; r+=e; return r; }
    RHSExpr operator-(const RHSExpr& e) const { RHSExpr r=*this; r-=e; return r; }
    RHSExpr operator*(double s)         const {
        RHSExpr r;
        for (const auto& t : terms) r.terms.push_back(t * s);
        return r;
    }
};

// Free overloads so users can write: a + b, a - b, s * expr
inline RHSExpr operator+(const Term& a, const Term& b) { RHSExpr e(a); e += b; return e; }
inline RHSExpr operator-(const Term& a, const Term& b) { RHSExpr e(a); e -= b; return e; }
inline RHSExpr operator+(const Term& t, const RHSExpr& e) { RHSExpr r(t); r += e; return r; }
inline RHSExpr operator-(const Term& t, const RHSExpr& e) { RHSExpr r(t); r -= e; return r; }
inline RHSExpr operator*(double s, const RHSExpr& e) { return e * s; }

// ---------------------------------------------------------------------------
// Built-in differential operator factories (implemented in Equation.cu)
// ---------------------------------------------------------------------------

// coeff * nabla^2(f)   — 2nd-order central FD Laplacian summed over active axes
Term lap(const Field& f, double coeff = 1.0);

// coeff * d(f)/d(x_axis)  — 2nd-order central FD component gradient
Term grad(const Field& f, int axis, double coeff = 1.0);

// ---------------------------------------------------------------------------
// pw<Functor> — pointwise user-defined transform
//
// coeff * Functor()(phi) applied element-wise to the physical cells of f.
//
// Functor requirements:
//   double operator()(double phi) const     <- CPU path
//   __device__ double operator()(double) const  <- GPU path
//
// Example functor (double-well derivative for Allen-Cahn):
//   struct DW { __device__ double operator()(double p) const
//                { return p*(1.0-p*p); } };
//
// Or with extended lambdas (nvcc --expt-extended-lambda):
//   pw(phi, [] __device__ (double p) { return p*(1.0-p*p); })
// ---------------------------------------------------------------------------
template<typename Functor>
Term pw(const Field& f, Functor func, double coeff = 1.0);

} // namespace PhiX

// Template definitions — included here so nvcc sees them in every TU that
// includes this header.  Contains __global__ code; requires nvcc.
#include "equation/TermPW.inl"
