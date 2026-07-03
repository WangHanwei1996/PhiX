#include "field/Reduce.h"

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace PhiX {
namespace reduce {

// ---------------------------------------------------------------------------
// CUDA error-checking macro (local to this translation unit)
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

namespace {

// ---------------------------------------------------------------------------
// Gather functors — map a physical linear index p in [0, nx*ny*nz) to the
// stored (halo-padded) index and fetch/transform the value.  Used through
// thrust::transform_iterator so cub::DeviceReduce never touches ghost cells.
// ---------------------------------------------------------------------------
struct GatherBase {
    const Real* data;
    int nx, ny, sx, sy, g;

    __host__ __device__ double fetch(int p) const {
        const int i = p % nx;
        const int j = (p / nx) % ny;
        const int k = p / (nx * ny);
        const std::size_t c = (i + g)
            + static_cast<std::size_t>(sx) * ((j + g)
            + static_cast<std::size_t>(sy) * (k + g));
        return data[c];
    }
};

struct GatherValue : GatherBase {
    __host__ __device__ double operator()(int p) const { return fetch(p); }
};
struct GatherAbs : GatherBase {
    __host__ __device__ double operator()(int p) const { return fabs(fetch(p)); }
};
struct GatherSq : GatherBase {
    __host__ __device__ double operator()(int p) const {
        const double v = fetch(p);
        return v * v;
    }
};
struct GatherNonFinite : GatherBase {
    __host__ __device__ int operator()(int p) const {
        return isfinite(fetch(p)) ? 0 : 1;
    }
};
struct GatherDot : GatherBase {
    const Real* data2;
    __host__ __device__ double operator()(int p) const {
        const int i = p % nx;
        const int j = (p / nx) % ny;
        const int k = p / (nx * ny);
        const std::size_t c = (i + g)
            + static_cast<std::size_t>(sx) * ((j + g)
            + static_cast<std::size_t>(sy) * (k + g));
        return static_cast<double>(data[c]) * static_cast<double>(data2[c]);
    }
};

// ---------------------------------------------------------------------------
// Cached device scratch: CUB temp storage (grow-only) + one 8-byte result
// slot.  NOT freed in a static destructor (the CUDA context may already be
// gone at exit) — freeScratch() releases explicitly.
// ---------------------------------------------------------------------------
struct Scratch {
    void*       d_temp     = nullptr;
    std::size_t temp_bytes = 0;
    void*       d_out      = nullptr;   // 8 bytes: double or int result
};
Scratch g_scratch;

void ensureScratch(std::size_t bytes) {
    if (!g_scratch.d_out)
        CUDA_CHECK(cudaMalloc(&g_scratch.d_out, sizeof(double)));
    if (bytes > g_scratch.temp_bytes) {
        if (g_scratch.d_temp) CUDA_CHECK(cudaFree(g_scratch.d_temp));
        CUDA_CHECK(cudaMalloc(&g_scratch.d_temp, bytes));
        g_scratch.temp_bytes = bytes;
    }
}

template<typename Gather>
Gather makeGather(const ScalarField& f, const char* fn) {
    if (!f.d_curr)
        throw std::runtime_error(std::string(fn) + ": field '" + f.name
                                 + "' has no device allocation");
    Gather op{};
    op.data = f.d_curr;
    op.nx = f.mesh.n[0];
    op.ny = f.mesh.n[1];
    op.sx = f.storedDims[0];
    op.sy = f.storedDims[1];
    op.g  = f.ghost;
    return op;
}

// Run one CUB device reduction over the physical cells and return the result.
// Op is invoked as op(d_temp, bytes, iterator, d_out, n).
template<typename T, typename Gather, typename CubOp>
T runReduce(const ScalarField& f, const char* fn, CubOp cubOp) {
    const Gather gather = makeGather<Gather>(f, fn);
    const int n = f.mesh.n[0] * f.mesh.n[1] * f.mesh.n[2];

    auto it = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0), gather);

    std::size_t bytes = 0;
    CUDA_CHECK(cubOp(nullptr, bytes, it, static_cast<T*>(g_scratch.d_out), n));
    ensureScratch(bytes);
    CUDA_CHECK(cubOp(g_scratch.d_temp, bytes, it,
                     static_cast<T*>(g_scratch.d_out), n));

    T h{};
    CUDA_CHECK(cudaMemcpy(&h, g_scratch.d_out, sizeof(T),
                          cudaMemcpyDeviceToHost));
    return h;
}

// cub::DeviceReduce entry points wrapped as plain callables (the member
// templates cannot be passed directly as template-template arguments).
struct CubMax {
    template<typename It, typename T>
    cudaError_t operator()(void* t, std::size_t& b, It it, T* out, int n) const {
        return cub::DeviceReduce::Max(t, b, it, out, n);
    }
};
struct CubMin {
    template<typename It, typename T>
    cudaError_t operator()(void* t, std::size_t& b, It it, T* out, int n) const {
        return cub::DeviceReduce::Min(t, b, it, out, n);
    }
};
struct CubSum {
    template<typename It, typename T>
    cudaError_t operator()(void* t, std::size_t& b, It it, T* out, int n) const {
        return cub::DeviceReduce::Sum(t, b, it, out, n);
    }
};

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

double fieldMax(const ScalarField& f) {
    return runReduce<double, GatherValue>(f, "reduce::fieldMax", CubMax{});
}

double fieldMin(const ScalarField& f) {
    return runReduce<double, GatherValue>(f, "reduce::fieldMin", CubMin{});
}

double fieldMaxAbs(const ScalarField& f) {
    return runReduce<double, GatherAbs>(f, "reduce::fieldMaxAbs", CubMax{});
}

double fieldSum(const ScalarField& f) {
    return runReduce<double, GatherValue>(f, "reduce::fieldSum", CubSum{});
}

double fieldSumSq(const ScalarField& f) {
    return runReduce<double, GatherSq>(f, "reduce::fieldSumSq", CubSum{});
}

double fieldL2(const ScalarField& f) {
    return std::sqrt(runReduce<double, GatherSq>(f, "reduce::fieldL2", CubSum{}));
}

bool fieldHasNonFinite(const ScalarField& f) {
    return runReduce<int, GatherNonFinite>(f, "reduce::fieldHasNonFinite",
                                           CubMax{}) != 0;
}

double fieldDot(const ScalarField& a, const ScalarField& b) {
    if (a.storedSize != b.storedSize || a.ghost != b.ghost)
        throw std::invalid_argument(
            "reduce::fieldDot: fields '" + a.name + "' and '" + b.name
            + "' have different layouts");
    if (!b.d_curr)
        throw std::runtime_error("reduce::fieldDot: field '" + b.name
                                 + "' has no device allocation");
    const GatherDot base = makeGather<GatherDot>(a, "reduce::fieldDot");
    GatherDot gather = base;
    gather.data2 = b.d_curr;

    const int n = a.mesh.n[0] * a.mesh.n[1] * a.mesh.n[2];
    auto it = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0), gather);

    std::size_t bytes = 0;
    CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, bytes, it,
                                      static_cast<double*>(g_scratch.d_out), n));
    ensureScratch(bytes);
    CUDA_CHECK(cub::DeviceReduce::Sum(g_scratch.d_temp, bytes, it,
                                      static_cast<double*>(g_scratch.d_out), n));
    double h = 0.0;
    CUDA_CHECK(cudaMemcpy(&h, g_scratch.d_out, sizeof(double),
                          cudaMemcpyDeviceToHost));
    return h;
}

void freeScratch() {
    if (g_scratch.d_temp) CUDA_CHECK(cudaFree(g_scratch.d_temp));
    if (g_scratch.d_out)  CUDA_CHECK(cudaFree(g_scratch.d_out));
    g_scratch = Scratch{};
}

} // namespace reduce
} // namespace PhiX
