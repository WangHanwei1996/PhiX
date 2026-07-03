#include "operators/Laplacian.h"
#include "scheme/Isotropic.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace PhiX {

namespace {

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess)                                                 \
            throw std::runtime_error(                                          \
                std::string("CUDA error in " __FILE__ " line ")               \
                + std::to_string(__LINE__) + ": "                            \
                + cudaGetErrorString(_e));                                     \
    } while (0)

template<typename Scheme>
__global__ void kernel_lap_accumulate(
        Real*       rhs,
        const Real* src,
        double        coeff,
        int nx, int ny, int nz,
        int sx, int sy,
        int ghost, int dim,
        double inv_dx2,
        double inv_dy2,
        double inv_dz2)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny * nz) return;

    int i = tid % nx;
    int j = (tid / nx) % ny;
    int k = tid / (nx * ny);

    int is = i + ghost;
    int js = j + ghost;
    int ks = k + ghost;

    int c = is + sx * (js + sy * ks);
    double val = Scheme::laplacian(src, c, sx, sy, dim,
                                   inv_dx2, inv_dy2, inv_dz2);
    rhs[c] += coeff * val;
}

template<typename Scheme>
Term makeLaplacianTerm(const ScalarField& f, double coeff) {
    Term t;
    t.type  = TermType::LAPLACIAN;
    t.field = &f;
    t.coeff = coeff;
    t.ghostRequired = Scheme::ghostRequired();

    int    nx = f.mesh.n[0], ny = f.mesh.n[1], nz = f.mesh.n[2];
    int    sx = f.storedDims[0], sy = f.storedDims[1];
    int    g  = f.ghost;
    int    dim = f.mesh.dim;
    double inv_dx2 = 1.0 / (f.mesh.d[0] * f.mesh.d[0]);
    double inv_dy2 = (dim >= 2) ? 1.0 / (f.mesh.d[1] * f.mesh.d[1]) : 0.0;
    double inv_dz2 = (dim >= 3) ? 1.0 / (f.mesh.d[2] * f.mesh.d[2]) : 0.0;

    const ScalarField* pf = &f;

    t.gpu_launcher = [pf, nx, ny, nz, sx, sy, g, dim, inv_dx2, inv_dy2, inv_dz2]
                     (Real* d_rhs, double c, ScratchPool& pool) {
        const Real* d_src = pf->d_curr;
        if (!d_src)
            throw std::runtime_error("lap GPU: source field not on device");
        int total = nx * ny * nz;
        kernel_lap_accumulate<Scheme><<<(total + 255) / 256, 256, 0, pool.stream>>>(
            d_rhs, d_src, c,
            nx, ny, nz, sx, sy,
            g, dim, inv_dx2, inv_dy2, inv_dz2);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("lap GPU kernel error: ") + cudaGetErrorString(err));
    };

    t.cpu_kernel = [pf, nx, ny, nz, sx, sy, g, dim, inv_dx2, inv_dy2, inv_dz2]
                   (Real* rhs, double c, ScratchPool&) {
        const Real* src = pf->curr.data();
        for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int is = i + g, js = j + g, ks = k + g;
            int ctr = is + sx * (js + sy * ks);
            double val = Scheme::laplacian(src, ctr, sx, sy, dim,
                                           inv_dx2, inv_dy2, inv_dz2);
            rhs[ctr] += c * val;
        }
    };

    return t;
}

} // namespace

template<typename Scheme>
Term lap(const ScalarField& f, double coeff) {
    return makeLaplacianTerm<Scheme>(f, coeff);
}

template Term lap<scheme::CD2>(const ScalarField&, double);
template Term lap<scheme::Iso9>(const ScalarField&, double);
template Term lap<scheme::CD4>(const ScalarField&, double);

Term lap(const ScalarField& f, double coeff) {
    return lap<scheme::CD2>(f, coeff);
}

Term lap(const ScalarField& f, const std::string& schemeName, double coeff) {
    if (schemeName == "Iso9") return lap<scheme::Iso9>(f, coeff);
    if (schemeName == "CD4") return lap<scheme::CD4>(f, coeff);
    if (schemeName == "CD2" || schemeName.empty()) return lap<scheme::CD2>(f, coeff);
    throw std::invalid_argument(
        std::string("lap: unknown scheme '") + schemeName + "'. Supported: CD2, CD4, Iso9");
}

} // namespace PhiX
