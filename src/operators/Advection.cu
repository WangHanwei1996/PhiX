#include "operators/Advection.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace PhiX {

namespace {

// ---------------------------------------------------------------------------
// Upwind directional derivative along one axis, selected by velocity sign.
// ---------------------------------------------------------------------------
__host__ __device__ inline
double upwind_d1(const Real* s, int c, int stride, double inv_d, double u) {
    return (u > 0.0) ? (s[c] - s[c - stride]) * inv_d
                     : (s[c + stride] - s[c]) * inv_d;
}

__global__ void kernel_adv_accumulate(
        Real*       rhs,
        const Real* src,
        const Real* ux,
        const Real* uy,
        const Real* uz,
        double        coeff,
        int nx, int ny, int nz,
        int sx, int sy,
        int ghost, int dim,
        double inv_dx, double inv_dy, double inv_dz)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny * nz) return;

    int i = tid % nx;
    int j = (tid / nx) % ny;
    int k = tid / (nx * ny);

    int c = (i + ghost) + sx * ((j + ghost) + sy * (k + ghost));

    double val = ux[c] * upwind_d1(src, c, 1, inv_dx, ux[c]);
    if (dim >= 2) val += uy[c] * upwind_d1(src, c, sx, inv_dy, uy[c]);
    if (dim >= 3) val += uz[c] * upwind_d1(src, c, sx * sy, inv_dz, uz[c]);

    rhs[c] += coeff * val;
}

} // namespace

Term adv(const VectorField& u, const ScalarField& f, double coeff) {
    const int dim = f.mesh.dim;
    if (u.nComponents() < dim)
        throw std::invalid_argument(
            "adv: velocity field '" + u.name + "' has "
            + std::to_string(u.nComponents()) + " components but mesh is "
            + std::to_string(dim) + "D");
    for (int c = 0; c < dim; ++c)
        if (u[c].storedSize != f.storedSize)
            throw std::invalid_argument(
                "adv: velocity component '" + u[c].name
                + "' layout differs from field '" + f.name + "'");

    Term t;
    t.type  = TermType::COMPOSITE;
    t.field = &f;
    t.coeff = coeff;
    t.ghostRequired = 1;

    int    nx = f.mesh.n[0], ny = f.mesh.n[1], nz = f.mesh.n[2];
    int    sx = f.storedDims[0], sy = f.storedDims[1];
    int    g  = f.ghost;
    double inv_dx = 1.0 / f.mesh.d[0];
    double inv_dy = (dim >= 2) ? 1.0 / f.mesh.d[1] : 0.0;
    double inv_dz = (dim >= 3) ? 1.0 / f.mesh.d[2] : 0.0;

    const ScalarField* pf  = &f;
    const ScalarField* pux = &u[0];
    const ScalarField* puy = (dim >= 2) ? &u[1] : &u[0];
    const ScalarField* puz = (dim >= 3) ? &u[2] : &u[0];

    t.gpu_launcher = [pf, pux, puy, puz, nx, ny, nz, sx, sy, g, dim,
                      inv_dx, inv_dy, inv_dz]
                     (Real* d_rhs, double c, ScratchPool& pool) {
        if (!pf->d_curr || !pux->d_curr)
            throw std::runtime_error("adv GPU: source/velocity not on device");
        int total = nx * ny * nz;
        kernel_adv_accumulate<<<(total + 255) / 256, 256, 0, pool.stream>>>(
            d_rhs, pf->d_curr, pux->d_curr, puy->d_curr, puz->d_curr, c,
            nx, ny, nz, sx, sy, g, dim, inv_dx, inv_dy, inv_dz);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("adv GPU kernel error: ") + cudaGetErrorString(err));
    };

    t.cpu_kernel = [pf, pux, puy, puz, nx, ny, nz, sx, sy, g, dim,
                    inv_dx, inv_dy, inv_dz]
                   (Real* rhs, double c, ScratchPool&) {
        const Real* src = pf->curr.data();
        const Real* vx  = pux->curr.data();
        const Real* vy  = puy->curr.data();
        const Real* vz  = puz->curr.data();
        for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int ctr = (i + g) + sx * ((j + g) + sy * (k + g));
            double val = vx[ctr] * upwind_d1(src, ctr, 1, inv_dx, vx[ctr]);
            if (dim >= 2) val += vy[ctr] * upwind_d1(src, ctr, sx, inv_dy, vy[ctr]);
            if (dim >= 3) val += vz[ctr] * upwind_d1(src, ctr, sx * sy, inv_dz, vz[ctr]);
            rhs[ctr] += c * val;
        }
    };

    return t;
}

} // namespace PhiX
