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
// pw<Functor> definition
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

    // GPU launcher: host function that launches the templated kernel
    t.gpu_launcher = [func, nx, ny, nz, sx, sy, g]
                     (double* d_rhs, const double* d_src, double c) mutable {
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
    t.cpu_kernel = [func, nx, ny, nz, sx, sy, g]
                   (double* rhs, const double* src, double c) mutable {
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
// pw(VectorField, Functor) — pointwise per-component
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

} // namespace PhiX
