#pragma once

// ---------------------------------------------------------------------------
// Term.h — Expression DSL for composing RHS of time-evolution equations.
//
// Requires nvcc compilation (CUDA device code is referenced by templates).
//
// Scalar DSL usage:
//   eq.setRHS(M * lap(phi) + M * pw(phi, [] __host__ __device__ (double p) {
//       return p - p*p*p; }));
//
// Vector DSL usage:
//   veq.setRHS(nu * lap(v));            // lap(VectorField) -> VectorRHSExpr
//   veq.setRHS(grad(p));                // grad(ScalarField) -> VectorRHSExpr
//   RHSExpr divV = div(v);              // div(VectorField) -> RHSExpr
//   VectorRHSExpr curlV = curl(v);      // curl(VectorField) -> VectorRHSExpr (3D)
// ---------------------------------------------------------------------------

#include "field/ScalarField.h"
#include "field/VectorField.h"

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
    const ScalarField*  field = nullptr;   // source field (non-owning)
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
Term lap(const ScalarField& f, double coeff = 1.0);

// coeff * d(f)/d(x_axis)  — 2nd-order central FD component gradient
Term grad(const ScalarField& f, int axis, double coeff = 1.0);

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
Term pw(const ScalarField& f, Functor func, double coeff = 1.0);

// ---------------------------------------------------------------------------
// pw<Functor> — 2-field pointwise transform
//   rhs[idx] += coeff * func(f1[idx], f2[idx])
//   Functor: __host__ __device__ double operator()(double, double) const
//   Both fields must share the same mesh and ghost width.
// ---------------------------------------------------------------------------
template<typename Functor>
Term pw(const ScalarField& f1, const ScalarField& f2,
        Functor func, double coeff = 1.0);

// ---------------------------------------------------------------------------
// pw<Functor> — 3-field pointwise transform
//   rhs[idx] += coeff * func(f1[idx], f2[idx], f3[idx])
//   Functor: __host__ __device__ double operator()(double, double, double) const
//   All fields must share the same mesh and ghost width.
// ---------------------------------------------------------------------------
template<typename Functor>
Term pw(const ScalarField& f1, const ScalarField& f2, const ScalarField& f3,
        Functor func, double coeff = 1.0);

// ===========================================================================
// VectorRHSExpr — per-component RHS expression for vector equations
//
// VectorRHSExpr wraps N RHSExpr objects, one per vector component.
// Supports the same coefficient arithmetic as RHSExpr:
//   double * VectorRHSExpr, VectorRHSExpr +/- VectorRHSExpr, etc.
//
// Typical use:
//   VectorEquation veq(v, "diffusion");
//   veq.setRHS(nu * lap(v));     // lap returns VectorRHSExpr
// ===========================================================================

struct VectorRHSExpr {
    std::vector<RHSExpr> components;

    VectorRHSExpr() = default;
    explicit VectorRHSExpr(int n) : components(n) {}

    int nComponents() const { return static_cast<int>(components.size()); }

    RHSExpr&       operator[](int c)       { return components[c]; }
    const RHSExpr& operator[](int c) const { return components[c]; }

    VectorRHSExpr& operator+=(const VectorRHSExpr& o) {
        for (int c = 0; c < nComponents(); ++c) components[c] += o.components[c];
        return *this;
    }
    VectorRHSExpr& operator-=(const VectorRHSExpr& o) {
        for (int c = 0; c < nComponents(); ++c) components[c] -= o.components[c];
        return *this;
    }
    VectorRHSExpr operator+(const VectorRHSExpr& o) const {
        VectorRHSExpr r = *this; r += o; return r;
    }
    VectorRHSExpr operator-(const VectorRHSExpr& o) const {
        VectorRHSExpr r = *this; r -= o; return r;
    }
    VectorRHSExpr operator*(double s) const {
        VectorRHSExpr r(nComponents());
        for (int c = 0; c < nComponents(); ++c) r[c] = components[c] * s;
        return r;
    }
    VectorRHSExpr operator-() const { return (*this) * (-1.0); }
};

inline VectorRHSExpr operator*(double s, const VectorRHSExpr& e) { return e * s; }

// ===========================================================================
// Vector operator factories
// ===========================================================================

// lap(VectorField)  — returns VectorRHSExpr;  component c is lap(vf[c])
VectorRHSExpr lap(const VectorField& vf, double coeff = 1.0);

// grad(ScalarField)  — returns VectorRHSExpr of dimension mesh.dim;
//   component c is grad(f, c)
VectorRHSExpr grad(const ScalarField& f, double coeff = 1.0);

// div(VectorField)  — returns RHSExpr (scalar);
//   result = sum_c grad(vf[c], c)
RHSExpr div(const VectorField& vf, double coeff = 1.0);

// curl(VectorField)  — returns VectorRHSExpr (3D only, dim must be 3);
//   curl[0] = dv2/dy - dv1/dz
//   curl[1] = dv0/dz - dv2/dx
//   curl[2] = dv1/dx - dv0/dy
VectorRHSExpr curl(const VectorField& vf, double coeff = 1.0);

// pw(VectorField, Functor)  — pointwise per-component;  component c is pw(vf[c], func)
template<typename Functor>
VectorRHSExpr pw(const VectorField& vf, Functor func, double coeff = 1.0);

// pw(VectorField, ScalarField, Functor) — per-component binary op with scalar field
//   component c: rhs_c[idx] += coeff * func(vf[c][idx], sf[idx])
template<typename Functor>
VectorRHSExpr pw(const VectorField& vf, const ScalarField& sf,
                 Functor func, double coeff = 1.0);

// pw(VectorField, VectorField, Functor) — component-wise binary op
//   component c: rhs_c[idx] += coeff * func(vf1[c][idx], vf2[c][idx])
template<typename Functor>
VectorRHSExpr pw(const VectorField& vf1, const VectorField& vf2,
                 Functor func, double coeff = 1.0);

} // namespace PhiX

// Template definitions — included here so nvcc sees them in every TU that
// includes this header.  Contains __global__ code; requires nvcc.
#include "equation/TermPW.inl"
