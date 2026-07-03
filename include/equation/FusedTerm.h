#pragma once

// ---------------------------------------------------------------------------
// FusedTerm.h — Compile-time expression templates for fused CUDA kernels.
//
// Motivation:
//   The standard DSL (Term / RHSExpr) launches one CUDA kernel per term.
//   Composite terms such as  mul(mul(f, g), lap(h))  require materialising
//   intermediate results into global-memory scratch buffers — typically
//   20-30 kernel launches for a 10-term RHS like the MPF_AC_DW μ equation.
//
//   FusedTerm encodes the entire RHS expression in a C++ type, so a single
//   kernel template evaluates every term per-cell without touching global
//   memory for intermediates.
//
// Usage (in a .cu file compiled by nvcc):
//   #include "equation/FusedTerm.h"
//   using namespace PhiX::Fused;
//
//   auto rhs =
//       fpw2(phi0, phi_a, PHIX_FN(double a, double b){ return 2*W*a*b*b; }) +
//       fmul(ffield(phi0), fgrad_dot(phi_a, phi_a)) * (2.0*eps2) +
//       fmul(fmul(ffield(phi0), ffield(phi_a)), flap(phi_a)) * eps2 +
//       ...;
//
//   eq_mu0.setRHS(fuse(rhs, phi0));   // compiles to a single GPU kernel
//
// Notes:
//   • All node types are trivially copyable (safe to pass to CUDA kernels).
//   • The expression tree is embedded in the kernel parameter list.
//     Typical sizes: ~30 bytes/term × 10 terms ≈ 300 bytes — well within
//     the 4 KB CUDA kernel parameter limit.
//   • operator+ and operator*(double) are constrained via FusedNodeTag so
//     they do not interfere with PhiX's ScalarField arithmetic operators.
//   • CPU fallback for fused terms is not implemented.
//
// Requires: nvcc with --expt-extended-lambda (same as rest of PhiX).
// ---------------------------------------------------------------------------

#include "field/ScalarField.h"
#include "equation/Term.h"   // Term, TermType, ScratchPool

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace PhiX {
namespace Fused {

// ============================================================================
// Tag — used for SFINAE on operator+ and operator*(double).
// Each node declares `using fused_node_tag = FusedNodeTag;` instead of
// inheriting, so aggregate initialization still works under C++14 semantics.
// ============================================================================

struct FusedNodeTag {};

namespace detail {
    template<typename T, typename = void>
    struct is_fused_node : std::false_type {};

    template<typename T>
    struct is_fused_node<T, std::void_t<typename T::fused_node_tag>> : std::true_type {};
} // namespace detail

// ============================================================================
// StencilParams — per-call mesh/stride context passed by value to eval()
// ============================================================================

struct StencilParams {
    int    sx, sy;              // storedDims[0], storedDims[1]
    Real   inv_dx2;             // 1 / dx²
    Real   inv_dy2;             // 1 / dy²  (0 if dim < 2)
    Real   inv_dz2;             // 1 / dz²  (0 if dim < 3)
    int    dim;
};

// ============================================================================
// Leaf nodes
// ============================================================================

// FieldNode: d_data[c]  (value at stored index c)
struct FieldNode {
    using fused_node_tag = FusedNodeTag;
    const Real* d_data;

    __host__ __device__ Real eval(int c, const StencilParams&) const {
        return d_data[c];
    }
};

// LapNode: ∇²f at stored index c  (2nd-order central FD)
struct LapNode {
    using fused_node_tag = FusedNodeTag;
    const Real* d_data;

    __host__ __device__ Real eval(int c, const StencilParams& p) const {
        Real v = (d_data[c + 1] - Real(2) * d_data[c] + d_data[c - 1]) * p.inv_dx2;
        if (p.dim >= 2)
            v += (d_data[c + p.sx] - Real(2) * d_data[c] + d_data[c - p.sx]) * p.inv_dy2;
        if (p.dim >= 3) {
            int ssz = p.sx * p.sy;
            v += (d_data[c + ssz] - Real(2) * d_data[c] + d_data[c - ssz]) * p.inv_dz2;
        }
        return v;
    }
};

// GradDotNode: ∇f · ∇g at stored index c  (central FD, all active axes)
struct GradDotNode {
    using fused_node_tag = FusedNodeTag;
    const Real* d_f;
    const Real* d_g;

    __host__ __device__ Real eval(int c, const StencilParams& p) const {
        // (df/dx) · (dg/dx) = [(f[c+1]-f[c-1]) * (g[c+1]-g[c-1])] / (4 dx²)
        Real v = (d_f[c + 1] - d_f[c - 1]) * (d_g[c + 1] - d_g[c - 1])
                   * (Real(0.25) * p.inv_dx2);
        if (p.dim >= 2) {
            v += (d_f[c + p.sx] - d_f[c - p.sx]) *
                 (d_g[c + p.sx] - d_g[c - p.sx]) * (Real(0.25) * p.inv_dy2);
        }
        if (p.dim >= 3) {
            int ssz = p.sx * p.sy;
            v += (d_f[c + ssz] - d_f[c - ssz]) *
                 (d_g[c + ssz] - d_g[c - ssz]) * (Real(0.25) * p.inv_dz2);
        }
        return v;
    }
};

// ============================================================================
// Composite nodes  (templated — type encodes the expression tree structure)
// ============================================================================

// ScaleNode: coeff * inner
template<typename Inner>
struct ScaleNode {
    using fused_node_tag = FusedNodeTag;
    Inner  inner;
    Real   coeff;

    __host__ __device__ Real eval(int c, const StencilParams& p) const {
        return coeff * inner.eval(c, p);
    }
};

// MulNode: lhs * rhs  (element-wise product of two sub-expressions)
template<typename Lhs, typename Rhs>
struct MulNode {
    using fused_node_tag = FusedNodeTag;
    Lhs lhs;
    Rhs rhs;

    __host__ __device__ Real eval(int c, const StencilParams& p) const {
        return lhs.eval(c, p) * rhs.eval(c, p);
    }
};

// AddNode: lhs + rhs  (sum of two sub-expressions)
template<typename Lhs, typename Rhs>
struct AddNode {
    using fused_node_tag = FusedNodeTag;
    Lhs lhs;
    Rhs rhs;

    __host__ __device__ Real eval(int c, const StencilParams& p) const {
        return lhs.eval(c, p) + rhs.eval(c, p);
    }
};

// Pw1Node: fn(f[c])  — 1-field user functor
template<typename Fn>
struct Pw1Node {
    using fused_node_tag = FusedNodeTag;
    const Real* d_f;
    Fn fn;

    __host__ __device__ Real eval(int c, const StencilParams&) const {
        return fn(d_f[c]);
    }
};

// Pw2Node: fn(f1[c], f2[c])  — 2-field user functor
template<typename Fn>
struct Pw2Node {
    using fused_node_tag = FusedNodeTag;
    const Real* d_f1;
    const Real* d_f2;
    Fn fn;

    __host__ __device__ Real eval(int c, const StencilParams&) const {
        return fn(d_f1[c], d_f2[c]);
    }
};

// ============================================================================
// Operator overloads — constrained to FusedNodeTag types via SFINAE
// so they do not interfere with PhiX::ScalarField arithmetic operators.
// ============================================================================

// expr + expr  →  AddNode
template<typename L, typename R,
         typename = std::enable_if_t<
             detail::is_fused_node<L>::value ||
             detail::is_fused_node<R>::value>>
inline AddNode<L, R> operator+(L l, R r) { return {l, r}; }

// expr * coeff  →  ScaleNode
template<typename T,
         typename = std::enable_if_t<detail::is_fused_node<T>::value>>
inline ScaleNode<T> operator*(T t, double c) { return {t, c}; }

// coeff * expr  →  ScaleNode
template<typename T,
         typename = std::enable_if_t<detail::is_fused_node<T>::value>>
inline ScaleNode<T> operator*(double c, T t) { return {t, c}; }

// -expr  →  ScaleNode with coeff=-1
template<typename T,
         typename = std::enable_if_t<detail::is_fused_node<T>::value>>
inline ScaleNode<T> operator-(T t) { return {t, -1.0}; }

// ============================================================================
// Factory functions
// ============================================================================

inline FieldNode ffield(const ScalarField& f)    { return {f.d_curr}; }
inline LapNode   flap  (const ScalarField& f)    { return {f.d_curr}; }

inline GradDotNode fgrad_dot(const ScalarField& f, const ScalarField& g) {
    return {f.d_curr, g.d_curr};
}

template<typename L, typename R>
inline MulNode<L, R> fmul(L l, R r) { return {l, r}; }

template<typename Fn>
inline Pw1Node<Fn> fpw(const ScalarField& f, Fn fn) {
    return {f.d_curr, fn};
}

template<typename Fn>
inline Pw2Node<Fn> fpw2(const ScalarField& f1, const ScalarField& f2, Fn fn) {
    return {f1.d_curr, f2.d_curr, fn};
}

// ============================================================================
// Fused CUDA kernel — one thread per physical cell, evaluates the entire
// expression tree per-cell without intermediate global-memory writes.
// ============================================================================

template<typename Expr>
__global__ void kernel_fused_accumulate(
        Real* __restrict__ d_rhs,
        Expr expr,
        int nx, int ny, int nz,
        int sx, int sy, int g,
        Real inv_dx2, Real inv_dy2, Real inv_dz2,
        int dim)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny * nz) return;

    int i = tid % nx;
    int j = (tid / nx) % ny;
    int k = tid / (nx * ny);
    int c = (i + g) + sx * ((j + g) + sy * (k + g));

    StencilParams p{sx, sy, inv_dx2, inv_dy2, inv_dz2, dim};
    d_rhs[c] += expr.eval(c, p);
}

// ============================================================================
// 3-output fused kernel — one pass over the grid, writes three output fields.
//
// All three expressions share the same StencilParams (same mesh), so stencil
// neighbour loads for shared input fields (e.g. phi0, phi_a, phi_b in the
// MPF_AC_DW μ equations) are fetched once per warp via L1/L2 cache hits
// rather than three independent kernel launches each re-reading the same data.
// ============================================================================

template<typename E0, typename E1, typename E2>
__global__ void kernel_fused_multi3(
        Real* __restrict__ d_out0, E0 e0,
        Real* __restrict__ d_out1, E1 e1,
        Real* __restrict__ d_out2, E2 e2,
        int nx, int ny, int nz,
        int sx, int sy, int g,
        Real inv_dx2, Real inv_dy2, Real inv_dz2,
        int dim)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny * nz) return;

    int i = tid % nx;
    int j = (tid / nx) % ny;
    int k = tid / (nx * ny);
    int c = (i + g) + sx * ((j + g) + sy * (k + g));

    StencilParams p{sx, sy, inv_dx2, inv_dy2, inv_dz2, dim};

    // Evaluate all three expressions per cell before writing — compiler keeps
    // shared sub-expressions (lap(phi0), grad_dot, ...) in registers.
    d_out0[c] = e0.eval(c, p);
    d_out1[c] = e1.eval(c, p);
    d_out2[c] = e2.eval(c, p);
}

// fuse_multi_compute — launch one kernel to compute three output fields.
//
// `layout`  : any field on the same mesh (provides geometry / strides).
// `outN`    : output ScalarField; d_curr is written directly (no zero-init
//             needed — the full expression replaces the field value).
// `exprN`   : a Fused expression tree built with ffield/flap/fgrad_dot/fmul/
//             fpw2/operator+/operator*.  Build with `auto expr = ...;`
//
// Input fields referenced inside exprN must have ghost cells filled (BCs
// applied) before calling.  Typical usage in a time loop:
//
//   fuse_multi_compute(phi0,
//       mu0,  expr_mu0,
//       mu_a, expr_mu_a,
//       mu_b, expr_mu_b);
//
template<typename E0, typename E1, typename E2>
inline void fuse_multi_compute(
        const ScalarField& layout,
        ScalarField& out0, E0 expr0,
        ScalarField& out1, E1 expr1,
        ScalarField& out2, E2 expr2,
        cudaStream_t stream = nullptr)
{
    int nx  = layout.mesh.n[0], ny = layout.mesh.n[1], nz = layout.mesh.n[2];
    int sx  = layout.storedDims[0], sy = layout.storedDims[1];
    int g   = layout.ghost;
    int dim = layout.mesh.dim;

    double inv_dx2 = 1.0 / (layout.mesh.d[0] * layout.mesh.d[0]);
    double inv_dy2 = (dim >= 2) ? 1.0 / (layout.mesh.d[1] * layout.mesh.d[1]) : 0.0;
    double inv_dz2 = (dim >= 3) ? 1.0 / (layout.mesh.d[2] * layout.mesh.d[2]) : 0.0;

    int blocks = (nx * ny * nz + 255) / 256;

    kernel_fused_multi3<E0, E1, E2><<<blocks, 256, 0, stream>>>(
        out0.d_curr, expr0,
        out1.d_curr, expr1,
        out2.d_curr, expr2,
        nx, ny, nz, sx, sy, g,
        inv_dx2, inv_dy2, inv_dz2, dim);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("fuse_multi_compute: ") + cudaGetErrorString(err));
}

// ============================================================================
// FusedRHSExpr<Expr>
//
// Wraps a fused expression and converts it to a single runtime Term so it
// integrates seamlessly with Equation::setRHS(const Term&).
//
// The Term's gpu_launcher captures the expression by value (safe because all
// Fused nodes are trivially copyable).
// ============================================================================

template<typename Expr>
class FusedRHSExpr {
public:
    FusedRHSExpr(Expr expr, const ScalarField& layout)
        : expr_(expr), layout_(&layout) {}

    // Implicit conversion to Term — lets Equation::setRHS accept FusedRHSExpr
    // directly without any changes to Equation.h / Equation.cu.
    operator Term() const {
        const ScalarField* lay = layout_;
        Expr  expr = expr_;

        int    nx  = lay->mesh.n[0];
        int    ny  = lay->mesh.n[1];
        int    nz  = lay->mesh.n[2];
        int    sx  = lay->storedDims[0];
        int    sy  = lay->storedDims[1];
        int    g   = lay->ghost;
        int    dim = lay->mesh.dim;
        double inv_dx2 = 1.0 / (lay->mesh.d[0] * lay->mesh.d[0]);
        double inv_dy2 = (dim >= 2) ? 1.0 / (lay->mesh.d[1] * lay->mesh.d[1]) : 0.0;
        double inv_dz2 = (dim >= 3) ? 1.0 / (lay->mesh.d[2] * lay->mesh.d[2]) : 0.0;
        int    total   = nx * ny * nz;

        Term t;
        t.type  = TermType::COMPOSITE;
        t.field = lay;
        t.coeff = 1.0;

        // Wrap in a ScaleNode so the Term::coeff passed by computeRHS is
        // honoured correctly even if the user wraps the fused term.
        t.gpu_launcher = [expr, nx, ny, nz, sx, sy, g, inv_dx2, inv_dy2, inv_dz2, dim, total]
                         (Real* d_rhs, double coeff, ScratchPool& pool) {
            auto scaled = ScaleNode<Expr>{expr, coeff};
            int blocks  = (total + 255) / 256;
            kernel_fused_accumulate<ScaleNode<Expr>><<<blocks, 256, 0, pool.stream>>>(
                d_rhs, scaled, nx, ny, nz, sx, sy, g,
                inv_dx2, inv_dy2, inv_dz2, dim);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
                throw std::runtime_error(
                    std::string("fused kernel error: ") + cudaGetErrorString(err));
        };

        // CPU path not implemented for fused expressions.
        t.cpu_kernel = [](Real*, double, ScratchPool&) {
            throw std::runtime_error(
                "FusedTerm: CPU fallback not implemented. "
                "Use the standard DSL Terms for CPU execution.");
        };

        return t;
    }

private:
    Expr               expr_;
    const ScalarField* layout_;
};

// Factory: fuse(expr, layout_field) → FusedRHSExpr<Expr>
// `layout_field` provides mesh/stride information; any field in the equation
// that lives on the same mesh works.
template<typename Expr>
inline FusedRHSExpr<Expr> fuse(Expr expr, const ScalarField& layout) {
    return FusedRHSExpr<Expr>(expr, layout);
}

} // namespace Fused
} // namespace PhiX
