// ---------------------------------------------------------------------------
// TermPW.inl — Template definitions for pw<Functor>().
// Included automatically by Term.h.  Do NOT include directly.
// Requires nvcc (contains __global__ kernel template).
// ---------------------------------------------------------------------------

#pragma once

#include <cuda_runtime.h>
#include <stdexcept>

namespace PhiX {

// ---------------------------------------------------------------------------
// GPU kernel: rhs[idx] += coeff * Func(src[idx])
// One thread per physical cell, row-major (x fast).
// ---------------------------------------------------------------------------
template<typename Functor>
__global__ void kernel_pw_accumulate(
        double*       rhs,
        const double* src,
        Functor       func,
        double        coeff,
        int nx, int ny, int nz,
        int sx, int sy,          // storedDims[0], storedDims[1]
        int ghost)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny * nz) return;

    int i = tid % nx;
    int j = (tid / nx) % ny;
    int k = tid / (nx * ny);

    int idx = (i + ghost) + sx * ((j + ghost) + sy * (k + ghost));
    rhs[idx] += coeff * func(src[idx]);
}

// ---------------------------------------------------------------------------
// GPU kernel (2-field): rhs[idx] += coeff * Func(src1[idx], src2[idx])
// Functor signature: __device__ double operator()(double, double) const
// ---------------------------------------------------------------------------
template<typename Functor>
__global__ void kernel_pw2_accumulate(
        double*       rhs,
        const double* src1,
        const double* src2,
        Functor       func,
        double        coeff,
        int nx, int ny, int nz,
        int sx, int sy,
        int ghost)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny * nz) return;

    int i = tid % nx;
    int j = (tid / nx) % ny;
    int k = tid / (nx * ny);

    int idx = (i + ghost) + sx * ((j + ghost) + sy * (k + ghost));
    rhs[idx] += coeff * func(src1[idx], src2[idx]);
}

// ---------------------------------------------------------------------------
// GPU kernel (3-field): rhs[idx] += coeff * Func(src1[idx], src2[idx], src3[idx])
// Functor signature: __device__ double operator()(double, double, double) const
// ---------------------------------------------------------------------------
template<typename Functor>
__global__ void kernel_pw3_accumulate(
        double*       rhs,
        const double* src1,
        const double* src2,
        const double* src3,
        Functor       func,
        double        coeff,
        int nx, int ny, int nz,
        int sx, int sy,
        int ghost)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny * nz) return;

    int i = tid % nx;
    int j = (tid / nx) % ny;
    int k = tid / (nx * ny);

    int idx = (i + ghost) + sx * ((j + ghost) + sy * (k + ghost));
    rhs[idx] += coeff * func(src1[idx], src2[idx], src3[idx]);
}

// ---------------------------------------------------------------------------
// pw<Functor> definition (single field)
// ---------------------------------------------------------------------------
template<typename Functor>
Term pw(const ScalarField& f, Functor func, double coeff) {
    Term t;
    t.type  = TermType::POINTWISE;
    t.field = &f;
    t.coeff = coeff;

    // Capture mesh layout at construction time
    int nx = f.mesh.n[0], ny = f.mesh.n[1], nz = f.mesh.n[2];
    int sx = f.storedDims[0], sy = f.storedDims[1];
    int g  = f.ghost;

    // Capture pointer-to-field so RK4 d_curr swap remains effective.
    const ScalarField* pf = &f;

    // GPU launcher: host function that launches the templated kernel
    t.gpu_launcher = [func, pf, nx, ny, nz, sx, sy, g]
                     (double* d_rhs, double c, ScratchPool&) mutable {
        const double* d_src = pf->d_curr;
        if (!d_src)
            throw std::runtime_error(
                "pw GPU: source field not on device");
        int total   = nx * ny * nz;
        int threads = 256;
        int blocks  = (total + threads - 1) / threads;
        kernel_pw_accumulate<Functor><<<blocks, threads>>>(
            d_rhs, d_src, func, c, nx, ny, nz, sx, sy, g);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("pw GPU kernel error: ") + cudaGetErrorString(err));
    };

    // CPU fallback (Functor::operator() must also work on host)
    t.cpu_kernel = [func, pf, nx, ny, nz, sx, sy, g]
                   (double* rhs, double c, ScratchPool&) mutable {
        const double* src = pf->curr.data();
        for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int idx = (i+g) + sx*((j+g) + sy*(k+g));
            rhs[idx] += c * func(src[idx]);
        }
    };

    return t;
}

// ---------------------------------------------------------------------------
// pw<Functor> definition (2 fields)
//   rhs[idx] += coeff * func(f1[idx], f2[idx])
//   Functor signature: (double f1_val, double f2_val) -> double
//   Both fields must share the same mesh dimensions and ghost width.
// ---------------------------------------------------------------------------
template<typename Functor>
Term pw(const ScalarField& f1, const ScalarField& f2, Functor func, double coeff) {
    // Validate mesh compatibility
    if (f1.mesh.n[0] != f2.mesh.n[0] ||
        f1.mesh.n[1] != f2.mesh.n[1] ||
        f1.mesh.n[2] != f2.mesh.n[2] ||
        f1.ghost != f2.ghost)
        throw std::invalid_argument(
            "pw(f1, f2): fields must share the same mesh dimensions and ghost width");

    Term t;
    t.type  = TermType::POINTWISE;
    t.field = &f1;   // primary field (used by computeRHS for null-check)
    t.coeff = coeff;

    int nx = f1.mesh.n[0], ny = f1.mesh.n[1], nz = f1.mesh.n[2];
    int sx = f1.storedDims[0], sy = f1.storedDims[1];
    int g  = f1.ghost;

    // Capture pointers (non-owning); read d_curr at launch time
    const ScalarField* pf1 = &f1;
    const ScalarField* pf2 = &f2;

    t.gpu_launcher = [func, nx, ny, nz, sx, sy, g, pf1, pf2]
                     (double* d_rhs, double c, ScratchPool&) mutable {
        const double* d_src1 = pf1->d_curr;
        const double* d_src2 = pf2->d_curr;
        if (!d_src1 || !d_src2)
            throw std::runtime_error(
                "pw(f1,f2) GPU: a field not on device. "
                "Call allocDevice() and uploadToDevice() first.");
        int total   = nx * ny * nz;
        int threads = 256;
        int blocks  = (total + threads - 1) / threads;
        kernel_pw2_accumulate<Functor><<<blocks, threads>>>(
            d_rhs, d_src1, d_src2, func, c, nx, ny, nz, sx, sy, g);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("pw2 GPU kernel error: ") + cudaGetErrorString(err));
    };

    t.cpu_kernel = [func, nx, ny, nz, sx, sy, g, pf1, pf2]
                   (double* rhs, double c, ScratchPool&) mutable {
        const double* src1 = pf1->curr.data();
        const double* src2 = pf2->curr.data();
        for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int idx = (i+g) + sx*((j+g) + sy*(k+g));
            rhs[idx] += c * func(src1[idx], src2[idx]);
        }
    };

    return t;
}

// ---------------------------------------------------------------------------
// pw<Functor> definition (3 fields)
//   rhs[idx] += coeff * func(f1[idx], f2[idx], f3[idx])
//   Functor signature: (double, double, double) -> double
//   All fields must share the same mesh dimensions and ghost width.
// ---------------------------------------------------------------------------
template<typename Functor>
Term pw(const ScalarField& f1, const ScalarField& f2, const ScalarField& f3,
        Functor func, double coeff) {
    if (f1.mesh.n[0] != f2.mesh.n[0] || f1.mesh.n[0] != f3.mesh.n[0] ||
        f1.mesh.n[1] != f2.mesh.n[1] || f1.mesh.n[1] != f3.mesh.n[1] ||
        f1.mesh.n[2] != f2.mesh.n[2] || f1.mesh.n[2] != f3.mesh.n[2] ||
        f1.ghost != f2.ghost || f1.ghost != f3.ghost)
        throw std::invalid_argument(
            "pw(f1,f2,f3): all fields must share the same mesh dimensions and ghost width");

    Term t;
    t.type  = TermType::POINTWISE;
    t.field = &f1;
    t.coeff = coeff;

    int nx = f1.mesh.n[0], ny = f1.mesh.n[1], nz = f1.mesh.n[2];
    int sx = f1.storedDims[0], sy = f1.storedDims[1];
    int g  = f1.ghost;

    const ScalarField* pf1 = &f1;
    const ScalarField* pf2 = &f2;
    const ScalarField* pf3 = &f3;

    t.gpu_launcher = [func, nx, ny, nz, sx, sy, g, pf1, pf2, pf3]
                     (double* d_rhs, double c, ScratchPool&) mutable {
        const double* d_src1 = pf1->d_curr;
        const double* d_src2 = pf2->d_curr;
        const double* d_src3 = pf3->d_curr;
        if (!d_src1 || !d_src2 || !d_src3)
            throw std::runtime_error(
                "pw(f1,f2,f3) GPU: a field not on device.");
        int total   = nx * ny * nz;
        int threads = 256;
        int blocks  = (total + threads - 1) / threads;
        kernel_pw3_accumulate<Functor><<<blocks, threads>>>(
            d_rhs, d_src1, d_src2, d_src3, func, c, nx, ny, nz, sx, sy, g);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("pw3 GPU kernel error: ") + cudaGetErrorString(err));
    };

    t.cpu_kernel = [func, nx, ny, nz, sx, sy, g, pf1, pf2, pf3]
                   (double* rhs, double c, ScratchPool&) mutable {
        const double* src1 = pf1->curr.data();
        const double* src2 = pf2->curr.data();
        const double* src3 = pf3->curr.data();
        for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int idx = (i+g) + sx*((j+g) + sy*(k+g));
            rhs[idx] += c * func(src1[idx], src2[idx], src3[idx]);
        }
    };

    return t;
}

// ---------------------------------------------------------------------------
// pw(VectorField, Functor) — pointwise per-component (single scalar field)
//   Returns VectorRHSExpr; component c is pw(vf[c], func, coeff)
//   NOTE: Term.h must be included before this file (VectorRHSExpr must exist).
// ---------------------------------------------------------------------------
template<typename Functor>
VectorRHSExpr pw(const VectorField& vf, Functor func, double coeff) {
    VectorRHSExpr expr(vf.nComponents());
    for (int c = 0; c < vf.nComponents(); ++c)
        expr[c] = RHSExpr(pw(vf[c], func, coeff));
    return expr;
}

// ---------------------------------------------------------------------------
// pw(VectorField, ScalarField, Functor) — per-component binary operation
//   component c: rhs_c[idx] += coeff * func(vf[c][idx], sf[idx])
//   Functor signature: (double vf_val, double sf_val) -> double
// ---------------------------------------------------------------------------
template<typename Functor>
VectorRHSExpr pw(const VectorField& vf, const ScalarField& sf,
                 Functor func, double coeff) {
    VectorRHSExpr expr(vf.nComponents());
    for (int c = 0; c < vf.nComponents(); ++c)
        expr[c] = RHSExpr(pw(vf[c], sf, func, coeff));
    return expr;
}

// ---------------------------------------------------------------------------
// pw(VectorField, VectorField, Functor) — component-wise binary operation
//   component c: rhs_c[idx] += coeff * func(vf1[c][idx], vf2[c][idx])
//   Both VectorFields must have the same number of components.
//   Functor signature: (double v1_val, double v2_val) -> double
// ---------------------------------------------------------------------------
template<typename Functor>
VectorRHSExpr pw(const VectorField& vf1, const VectorField& vf2,
                 Functor func, double coeff) {
    if (vf1.nComponents() != vf2.nComponents())
        throw std::invalid_argument(
            "pw(vf1, vf2): VectorFields must have the same number of components");
    VectorRHSExpr expr(vf1.nComponents());
    for (int c = 0; c < vf1.nComponents(); ++c)
        expr[c] = RHSExpr(pw(vf1[c], vf2[c], func, coeff));
    return expr;
}

} // namespace PhiX

// Field arithmetic operator overloads — enables DSL syntax like:
//   eq.setRHS(c * eta - 2.0 * c + lap(c))
#include "equation/FieldOps.inl"
