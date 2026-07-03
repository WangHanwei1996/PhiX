#include "operators/Gradient.h"
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
__global__ void kernel_grad_accumulate(
        Real*       rhs,
        const Real* src,
        double        coeff,
        int nx, int ny, int nz,
        int sx, int sy,
        int ghost, int dim, int axis,
        double inv_dx, double inv_dy, double inv_dz)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny * nz) return;

    int i = tid % nx;
    int j = (tid / nx) % ny;
    int k = tid / (nx * ny);

    int is = i + ghost;
    int js = j + ghost;
    int ks = k + ghost;
    int c  = is + sx * (js + sy * ks);

    rhs[c] += coeff * Scheme::gradient(src, c, axis, sx, sy, dim,
                                        inv_dx, inv_dy, inv_dz);
}

template<typename Scheme>
Term makeGradientTerm(const ScalarField& f, int axis, double coeff) {
    if (axis < 0 || axis >= f.mesh.dim)
        throw std::invalid_argument("grad: axis out of range for this mesh dimension");

    Term t;
    t.type  = TermType::GRADIENT;
    t.field = &f;
    t.coeff = coeff;
    t.axis  = axis;
    t.ghostRequired = Scheme::ghostRequired();

    int    nx = f.mesh.n[0], ny = f.mesh.n[1], nz = f.mesh.n[2];
    int    sx = f.storedDims[0], sy = f.storedDims[1];
    int    g  = f.ghost;
    int    dim = f.mesh.dim;
    double inv_dx = 1.0 / f.mesh.d[0];
    double inv_dy = (dim >= 2) ? 1.0 / f.mesh.d[1] : 0.0;
    double inv_dz = (dim >= 3) ? 1.0 / f.mesh.d[2] : 0.0;

    const ScalarField* pf = &f;

    t.gpu_launcher = [pf, nx, ny, nz, sx, sy, g, dim, axis, inv_dx, inv_dy, inv_dz]
                     (Real* d_rhs, double c, ScratchPool& pool) {
        const Real* d_src = pf->d_curr;
        if (!d_src)
            throw std::runtime_error("grad GPU: source field not on device");
        int total = nx * ny * nz;
        kernel_grad_accumulate<Scheme><<<(total + 255) / 256, 256, 0, pool.stream>>>(
            d_rhs, d_src, c, nx, ny, nz, sx, sy, g, dim, axis, inv_dx, inv_dy, inv_dz);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("grad GPU kernel error: ") + cudaGetErrorString(err));
    };

    t.cpu_kernel = [pf, nx, ny, nz, sx, sy, g, dim, axis, inv_dx, inv_dy, inv_dz]
                   (Real* rhs, double c, ScratchPool&) {
        const Real* src = pf->curr.data();
        for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int is = i + g, js = j + g, ks = k + g;
            int ctr = is + sx * (js + sy * ks);
            rhs[ctr] += c * Scheme::gradient(src, ctr, axis, sx, sy, dim,
                                              inv_dx, inv_dy, inv_dz);
        }
    };

    return t;
}

} // namespace

template<typename Scheme>
Term grad(const ScalarField& f, int axis, double coeff) {
    return makeGradientTerm<Scheme>(f, axis, coeff);
}

template Term grad<scheme::CD2>(const ScalarField&, int, double);
template Term grad<scheme::Iso9>(const ScalarField&, int, double);
template Term grad<scheme::CD4>(const ScalarField&, int, double);
template Term grad<scheme::CD6>(const ScalarField&, int, double);

Term grad(const ScalarField& f, int axis, double coeff) {
    return grad<scheme::CD2>(f, axis, coeff);
}

Term grad(const ScalarField& f, int axis, const std::string& schemeName, double coeff) {
    if (schemeName == "Iso9") return grad<scheme::Iso9>(f, axis, coeff);
    if (schemeName == "CD4") return grad<scheme::CD4>(f, axis, coeff);
    if (schemeName == "CD6") return grad<scheme::CD6>(f, axis, coeff);
    if (schemeName == "CD2" || schemeName.empty()) return grad<scheme::CD2>(f, axis, coeff);
    throw std::invalid_argument(
        std::string("grad: unknown scheme '") + schemeName
        + "'. Supported: CD2, CD4, CD6, Iso9");
}

} // namespace PhiX
