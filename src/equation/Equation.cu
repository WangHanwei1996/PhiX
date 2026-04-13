#include "equation/Equation.h"

#include <cuda_runtime.h>
#include <stdexcept>

namespace PhiX {

// ---------------------------------------------------------------------------
// CUDA error-checking macro
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess)                                                 \
            throw std::runtime_error(                                          \
                std::string("CUDA error in " __FILE__ " line ")               \
                + std::to_string(__LINE__) + ": "                             \
                + cudaGetErrorString(_e));                                     \
    } while (0)

// ===========================================================================
// GPU kernels for built-in differential operators
// ===========================================================================

// ---------------------------------------------------------------------------
// Laplacian accumulate:
//   rhs[i,j,k] += coeff * (d²f/dx² [+ d²f/dy²] [+ d²f/dz²])
//
// 2nd-order central FD:
//   d²f/dx² ≈ (f[i+1,j,k] - 2f[i,j,k] + f[i-1,j,k]) / dx²
//
// Ghost cells of `src` must be valid before calling (BCs applied by Solver).
// Only physical cells are written in `rhs`.
// ---------------------------------------------------------------------------
__global__ void kernel_lap_accumulate(
        double*       rhs,
        const double* src,
        double        coeff,
        int nx, int ny, int nz,
        int sx, int sy,          // storedDims[0], storedDims[1]
        int ghost, int dim,
        double inv_dx2, double inv_dy2, double inv_dz2)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny * nz) return;

    int i = tid % nx;
    int j = (tid / nx) % ny;
    int k = tid / (nx * ny);

    int is = i + ghost;
    int js = j + ghost;
    int ks = k + ghost;

    // Flat stored indices for stencil neighbours
    int c  = is       + sx * (js       + sy * ks);
    int xm = (is - 1) + sx * (js       + sy * ks);
    int xp = (is + 1) + sx * (js       + sy * ks);

    double val = (src[xp] - 2.0 * src[c] + src[xm]) * inv_dx2;

    if (dim >= 2) {
        int ym = is + sx * ((js - 1) + sy * ks);
        int yp = is + sx * ((js + 1) + sy * ks);
        val += (src[yp] - 2.0 * src[c] + src[ym]) * inv_dy2;
    }

    if (dim >= 3) {
        int zm = is + sx * (js + sy * (ks - 1));
        int zp = is + sx * (js + sy * (ks + 1));
        val += (src[zp] - 2.0 * src[c] + src[zm]) * inv_dz2;
    }

    rhs[c] += coeff * val;
}

// ---------------------------------------------------------------------------
// Gradient accumulate (one component):
//   rhs[i,j,k] += coeff * df/dx_axis
//
// 2nd-order central FD:
//   df/dx ≈ (f[i+1] - f[i-1]) / (2*dx)
// ---------------------------------------------------------------------------
__global__ void kernel_grad_accumulate(
        double*       rhs,
        const double* src,
        double        coeff,
        int nx, int ny, int nz,
        int sx, int sy,
        int ghost, int axis,
        double inv_2d)   // 1 / (2 * d[axis])
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny * nz) return;

    int i = tid % nx;
    int j = (tid / nx) % ny;
    int k = tid / (nx * ny);

    int is = i + ghost;
    int js = j + ghost;
    int ks = k + ghost;

    int c   = is + sx * (js + sy * ks);
    int fwd, bwd;

    if (axis == 0) {
        fwd = (is + 1) + sx * (js + sy * ks);
        bwd = (is - 1) + sx * (js + sy * ks);
    } else if (axis == 1) {
        fwd = is + sx * ((js + 1) + sy * ks);
        bwd = is + sx * ((js - 1) + sy * ks);
    } else {
        fwd = is + sx * (js + sy * (ks + 1));
        bwd = is + sx * (js + sy * (ks - 1));
    }

    rhs[c] += coeff * (src[fwd] - src[bwd]) * inv_2d;
}

// ===========================================================================
// Built-in operator factories (return Term with launchers set up)
// ===========================================================================

Term lap(const Field& f, double coeff) {
    Term t;
    t.type  = TermType::LAPLACIAN;
    t.field = &f;
    t.coeff = coeff;

    // Capture layout and mesh spacing at construction time
    int    nx = f.mesh.n[0], ny = f.mesh.n[1], nz = f.mesh.n[2];
    int    sx = f.storedDims[0], sy = f.storedDims[1];
    int    g  = f.ghost;
    int    dim = f.mesh.dim;
    double inv_dx2 = 1.0 / (f.mesh.d[0] * f.mesh.d[0]);
    double inv_dy2 = (dim >= 2) ? 1.0 / (f.mesh.d[1] * f.mesh.d[1]) : 0.0;
    double inv_dz2 = (dim >= 3) ? 1.0 / (f.mesh.d[2] * f.mesh.d[2]) : 0.0;

    t.gpu_launcher = [nx, ny, nz, sx, sy, g, dim, inv_dx2, inv_dy2, inv_dz2]
                     (double* d_rhs, const double* d_src, double c) {
        int total = nx * ny * nz;
        kernel_lap_accumulate<<<(total + 255) / 256, 256>>>(
            d_rhs, d_src, c,
            nx, ny, nz, sx, sy,
            g, dim, inv_dx2, inv_dy2, inv_dz2);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("lap GPU kernel error: ") + cudaGetErrorString(err));
    };

    t.cpu_kernel = [nx, ny, nz, sx, sy, g, dim, inv_dx2, inv_dy2, inv_dz2]
                   (double* rhs, const double* src, double c) {
        for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int is = i+g, js = j+g, ks = k+g;
            int ctr = is + sx*(js + sy*ks);
            double val =
                (src[(is+1)+sx*(js+sy*ks)] - 2.0*src[ctr] + src[(is-1)+sx*(js+sy*ks)]) * inv_dx2;
            if (dim >= 2)
                val += (src[is+sx*((js+1)+sy*ks)] - 2.0*src[ctr] + src[is+sx*((js-1)+sy*ks)]) * inv_dy2;
            if (dim >= 3)
                val += (src[is+sx*(js+sy*(ks+1))] - 2.0*src[ctr] + src[is+sx*(js+sy*(ks-1))]) * inv_dz2;
            rhs[ctr] += c * val;
        }
    };

    return t;
}

Term grad(const Field& f, int axis, double coeff) {
    if (axis < 0 || axis >= f.mesh.dim)
        throw std::invalid_argument("grad: axis out of range for this mesh dimension");

    Term t;
    t.type  = TermType::GRADIENT;
    t.field = &f;
    t.coeff = coeff;
    t.axis  = axis;

    int    nx = f.mesh.n[0], ny = f.mesh.n[1], nz = f.mesh.n[2];
    int    sx = f.storedDims[0], sy = f.storedDims[1];
    int    g  = f.ghost;
    double inv_2d = 0.5 / f.mesh.d[axis];

    t.gpu_launcher = [nx, ny, nz, sx, sy, g, axis, inv_2d]
                     (double* d_rhs, const double* d_src, double c) {
        int total = nx * ny * nz;
        kernel_grad_accumulate<<<(total + 255) / 256, 256>>>(
            d_rhs, d_src, c, nx, ny, nz, sx, sy, g, axis, inv_2d);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("grad GPU kernel error: ") + cudaGetErrorString(err));
    };

    t.cpu_kernel = [nx, ny, nz, sx, sy, g, axis, inv_2d]
                   (double* rhs, const double* src, double c) {
        for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int is = i+g, js = j+g, ks = k+g;
            int ctr = is + sx*(js + sy*ks);
            int fwd, bwd;
            if (axis == 0) {
                fwd = (is+1) + sx*(js + sy*ks);
                bwd = (is-1) + sx*(js + sy*ks);
            } else if (axis == 1) {
                fwd = is + sx*((js+1) + sy*ks);
                bwd = is + sx*((js-1) + sy*ks);
            } else {
                fwd = is + sx*(js + sy*(ks+1));
                bwd = is + sx*(js + sy*(ks-1));
            }
            rhs[ctr] += c * (src[fwd] - src[bwd]) * inv_2d;
        }
    };

    return t;
}

// ===========================================================================
// Equation
// ===========================================================================

Equation::Equation(Field& unknown_, const std::string& name_)
    : name(name_), unknown(unknown_) {}

void Equation::setRHS(const RHSExpr& expr) {
    rhs_expr_ = expr;
}

void Equation::setRHS(const Term& t) {
    rhs_expr_ = RHSExpr(t);
}

void Equation::computeRHS(Field& rhs) const {
    if (!rhs.deviceAllocated())
        throw std::runtime_error("Equation::computeRHS: rhs device memory not allocated");
    if (rhs_expr_.terms.empty())
        throw std::runtime_error("Equation::computeRHS: RHS not set (call setRHS first)");

    // Zero physical cells of rhs (not whole stored array — ghost stays 0)
    // For simplicity zero the whole stored array; cost is negligible.
    CUDA_CHECK(cudaMemset(rhs.d_curr, 0, rhs.storedSize * sizeof(double)));

    for (const auto& term : rhs_expr_.terms) {
        if (!term.gpu_launcher)
            throw std::runtime_error(
                "Equation::computeRHS: a Term has no GPU launcher. "
                "Did you build it with a non-CUDA path?");
        const double* src = term.field->d_curr;
        if (!src)
            throw std::runtime_error(
                "Equation::computeRHS: source Field is not on device. "
                "Call Field::allocDevice() and uploadToDevice() first.");
        term.gpu_launcher(rhs.d_curr, src, term.coeff);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Equation::computeRHSCPU(Field& rhs) const {
    if (rhs_expr_.terms.empty())
        throw std::runtime_error("Equation::computeRHSCPU: RHS not set");

    std::fill(rhs.curr.begin(), rhs.curr.end(), 0.0);

    for (const auto& term : rhs_expr_.terms) {
        if (!term.cpu_kernel)
            throw std::runtime_error(
                "Equation::computeRHSCPU: a Term has no CPU kernel.");
        term.cpu_kernel(rhs.curr.data(), term.field->curr.data(), term.coeff);
    }
}

} // namespace PhiX
