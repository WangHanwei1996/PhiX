#pragma once

// ---------------------------------------------------------------------------
// ReducePW.h — pointwise-functor sum reductions over physical cells.
//
//     double s1 = reduce::fieldSumPW(c, PHIX_FN (Real v) {
//         return Real(5.0) * (v - Real(0.3)) * (v - Real(0.3))
//              * (Real(0.7) - v) * (Real(0.7) - v);   // bulk energy density
//     });
//     double F = dV * (s1 + 0.5 * kappa * reduce::fieldGradSq(c));
//
// 1–3 field overloads; the functor is evaluated per physical cell and the
// results are ACCUMULATED IN DOUBLE regardless of PHIX_PRECISION.  Ghost
// cells are never touched.  Requires nvcc (CUB/thrust — include from .cu);
// shares the cached device scratch with field/Reduce.h.
// ---------------------------------------------------------------------------

#include "core/Real.h"
#include "field/Reduce.h"
#include "field/ScalarField.h"

#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <stdexcept>
#include <string>

namespace PhiX {
namespace reduce {

namespace pwdetail {

template<typename Fn, int NF>
struct GatherPW {
    const Real* f0;
    const Real* f1;
    const Real* f2;
    int nx, ny, sx, sy, g;
    Fn fn;

    __host__ __device__ double operator()(int p) const {
        const int i = p % nx;
        const int j = (p / nx) % ny;
        const int k = p / (nx * ny);
        const std::size_t c = (i + g)
            + static_cast<std::size_t>(sx) * ((j + g)
            + static_cast<std::size_t>(sy) * (k + g));
        if constexpr (NF == 1)
            return static_cast<double>(fn(f0[c]));
        else if constexpr (NF == 2)
            return static_cast<double>(fn(f0[c], f1[c]));
        else
            return static_cast<double>(fn(f0[c], f1[c], f2[c]));
    }
};

template<typename Gather>
inline double runSum(const Gather& gather, int n) {
    auto it = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0), gather);
    double* d_out = detail::scratchOut();
    std::size_t bytes = 0;
    cudaError_t e = cub::DeviceReduce::Sum(nullptr, bytes, it, d_out, n);
    if (e == cudaSuccess)
        e = cub::DeviceReduce::Sum(detail::scratchTemp(bytes), bytes, it,
                                   d_out, n);
    double h = 0.0;
    if (e == cudaSuccess)
        e = cudaMemcpy(&h, d_out, sizeof(double), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("reduce::fieldSumPW: ")
                                 + cudaGetErrorString(e));
    return h;
}

inline void checkPW(const ScalarField& a, const ScalarField* b,
                    const ScalarField* c) {
    if (!a.d_curr)
        throw std::runtime_error("reduce::fieldSumPW: field '" + a.name
                                 + "' has no device allocation");
    for (const ScalarField* f : {b, c}) {
        if (!f) continue;
        if (f->storedSize != a.storedSize || f->ghost != a.ghost)
            throw std::invalid_argument(
                "reduce::fieldSumPW: field '" + f->name
                + "' layout differs from '" + a.name + "'");
        if (!f->d_curr)
            throw std::runtime_error("reduce::fieldSumPW: field '" + f->name
                                     + "' has no device allocation");
    }
}

} // namespace pwdetail

template<typename Fn>
double fieldSumPW(const ScalarField& a, Fn fn) {
    pwdetail::checkPW(a, nullptr, nullptr);
    pwdetail::GatherPW<Fn, 1> gth{a.d_curr, nullptr, nullptr,
                                  a.mesh.n[0], a.mesh.n[1],
                                  a.storedDims[0], a.storedDims[1],
                                  a.ghost, fn};
    return pwdetail::runSum(gth, a.mesh.n[0] * a.mesh.n[1] * a.mesh.n[2]);
}

template<typename Fn>
double fieldSumPW(const ScalarField& a, const ScalarField& b, Fn fn) {
    pwdetail::checkPW(a, &b, nullptr);
    pwdetail::GatherPW<Fn, 2> gth{a.d_curr, b.d_curr, nullptr,
                                  a.mesh.n[0], a.mesh.n[1],
                                  a.storedDims[0], a.storedDims[1],
                                  a.ghost, fn};
    return pwdetail::runSum(gth, a.mesh.n[0] * a.mesh.n[1] * a.mesh.n[2]);
}

template<typename Fn>
double fieldSumPW(const ScalarField& a, const ScalarField& b,
                  const ScalarField& c, Fn fn) {
    pwdetail::checkPW(a, &b, &c);
    pwdetail::GatherPW<Fn, 3> gth{a.d_curr, b.d_curr, c.d_curr,
                                  a.mesh.n[0], a.mesh.n[1],
                                  a.storedDims[0], a.storedDims[1],
                                  a.ghost, fn};
    return pwdetail::runSum(gth, a.mesh.n[0] * a.mesh.n[1] * a.mesh.n[2]);
}

} // namespace reduce
} // namespace PhiX
