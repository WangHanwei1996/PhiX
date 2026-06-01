#include "operators/Gradient.h"

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
        double*       rhs,
        const double* src,
        double        coeff,
        int nx, int ny, int nz,
        int sx, int sy,
        int ghost, int axis,
        double inv_d)
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

    int stride = (axis == 0) ? 1 : (axis == 1) ? sx : sx * sy;
    rhs[c] += coeff * Scheme::d1(src, c, stride, inv_d);
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
    double inv_d = 1.0 / f.mesh.d[axis];

    const ScalarField* pf = &f;

    t.gpu_launcher = [pf, nx, ny, nz, sx, sy, g, axis, inv_d]
                     (double* d_rhs, double c, ScratchPool&) {
        const double* d_src = pf->d_curr;
        if (!d_src)
            throw std::runtime_error("grad GPU: source field not on device");
        int total = nx * ny * nz;
        kernel_grad_accumulate<Scheme><<<(total + 255) / 256, 256>>>(
            d_rhs, d_src, c, nx, ny, nz, sx, sy, g, axis, inv_d);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("grad GPU kernel error: ") + cudaGetErrorString(err));
    };

    t.cpu_kernel = [pf, nx, ny, nz, sx, sy, g, axis, inv_d]
                   (double* rhs, double c, ScratchPool&) {
        const double* src = pf->curr.data();
        int stride = (axis == 0) ? 1 : (axis == 1) ? sx : sx * sy;
        for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int is = i + g, js = j + g, ks = k + g;
            int ctr = is + sx * (js + sy * ks);
            rhs[ctr] += c * Scheme::d1(src, ctr, stride, inv_d);
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

Term grad(const ScalarField& f, int axis, double coeff) {
    return grad<scheme::CD2>(f, axis, coeff);
}

} // namespace PhiX
