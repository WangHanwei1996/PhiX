#include "boundary/PeriodicBC.h"
#include "boundary/NoFluxBC.h"
#include "boundary/FixedBC.h"
#include "mesh/Mesh.h"

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
                std::string("CUDA error in " __FILE__ " line ")                \
                + std::to_string(__LINE__) + ": "                              \
                + cudaGetErrorString(_e));                                     \
    } while (0)

// ===========================================================================
// Generic patch-aware kernel design
// ===========================================================================
//
// Row-major storage: flat(is, js, ks) = is + sx*(js + sy*ks)
// where is/js/ks are stored indices (physical index + ghost offset).
//
// For each Patch (axis, side, IndexBox region) we identify:
//
//   axis (normal)         : the BC axis
//   tangential axes t0,t1 : the other two axes (t0 < t1)
//   axis_stride           : flat stride along the normal axis
//   t0_stride, t1_stride  : flat strides along the two tangential axes
//   t0_count, t1_count    : number of cells along each tangential axis,
//                           taken from region.extent(t0/t1)
//   t0_lo, t1_lo          : physical starting index along t0/t1, plus ghost
//
// Each thread handles one (s0, s1) cell within the patch's region.
// ===========================================================================

struct PatchParams {
    int axis_stride;
    int n_axis;        // physical cells along normal axis
    int ghost;
    int t0_stride;
    int t1_stride;
    int t0_count;
    int t1_count;
    int t0_lo;
    int t1_lo;
};

static PatchParams makePatchParams(const ScalarField& f, const Patch& p) {
    const int sx = f.storedDims[0];
    const int sy = f.storedDims[1];

    const int a = static_cast<int>(p.axis);
    int t0, t1;
    switch (a) {
        case 0: t0 = 1; t1 = 2; break;
        case 1: t0 = 0; t1 = 2; break;
        default: t0 = 0; t1 = 1; break;
    }

    auto axStride = [&](int axis) {
        if (axis == 0) return 1;
        if (axis == 1) return sx;
        return sx * sy;
    };

    PatchParams pp{};
    pp.axis_stride = axStride(a);
    pp.n_axis      = f.mesh.n[a];
    pp.ghost       = f.ghost;
    pp.t0_stride   = axStride(t0);
    pp.t1_stride   = axStride(t1);
    pp.t0_count    = p.region.extent(t0);
    pp.t1_count    = p.region.extent(t1);
    pp.t0_lo       = p.region.lo[t0] + f.ghost;
    pp.t1_lo       = p.region.lo[t1] + f.ghost;
    return pp;
}

// ---------------------------------------------------------------------------
// Periodic kernel (single patch handles BOTH sides of its axis)
// ---------------------------------------------------------------------------
__global__ void kernel_periodic(
        Real* data,
        int t0_count, int t1_count,
        int t0_stride, int t1_stride,
        int t0_lo,     int t1_lo,
        int axis_stride,
        int n_axis,
        int ghost)
{
    int s0 = blockIdx.x * blockDim.x + threadIdx.x;
    int s1 = blockIdx.y * blockDim.y + threadIdx.y;
    if (s0 >= t0_count || s1 >= t1_count) return;

    int face_off = (t0_lo + s0) * t0_stride + (t1_lo + s1) * t1_stride;

    for (int g = 1; g <= ghost; ++g) {
        int lo_ghost  = (ghost - g)              * axis_stride + face_off;
        int lo_source = (ghost + n_axis - g)     * axis_stride + face_off;
        int hi_ghost  = (ghost + n_axis + g - 1) * axis_stride + face_off;
        int hi_source = (ghost + g - 1)          * axis_stride + face_off;

        data[lo_ghost] = data[lo_source];
        data[hi_ghost] = data[hi_source];
    }
}

// ---------------------------------------------------------------------------
// NoFlux (zero-gradient) kernel — single side determined by `is_low`
// ---------------------------------------------------------------------------
__global__ void kernel_noflux(
        Real* data,
        int t0_count, int t1_count,
        int t0_stride, int t1_stride,
        int t0_lo,     int t1_lo,
        int axis_stride,
        int n_axis,
        int ghost,
        bool is_low)
{
    int s0 = blockIdx.x * blockDim.x + threadIdx.x;
    int s1 = blockIdx.y * blockDim.y + threadIdx.y;
    if (s0 >= t0_count || s1 >= t1_count) return;

    int face_off = (t0_lo + s0) * t0_stride + (t1_lo + s1) * t1_stride;

    if (is_low) {
        int src = ghost * axis_stride + face_off;
        for (int g = 1; g <= ghost; ++g) {
            int dst = (ghost - g) * axis_stride + face_off;
            data[dst] = data[src];
        }
    } else {
        int src = (ghost + n_axis - 1) * axis_stride + face_off;
        for (int g = 1; g <= ghost; ++g) {
            int dst = (ghost + n_axis + g - 1) * axis_stride + face_off;
            data[dst] = data[src];
        }
    }
}

// ---------------------------------------------------------------------------
// Fixed (Dirichlet) kernel — single side determined by `is_low`
// ---------------------------------------------------------------------------
__global__ void kernel_fixed(
        Real* data,
        int t0_count, int t1_count,
        int t0_stride, int t1_stride,
        int t0_lo,     int t1_lo,
        int axis_stride,
        int n_axis,
        int ghost,
        bool is_low,
        double value)
{
    int s0 = blockIdx.x * blockDim.x + threadIdx.x;
    int s1 = blockIdx.y * blockDim.y + threadIdx.y;
    if (s0 >= t0_count || s1 >= t1_count) return;

    int face_off = (t0_lo + s0) * t0_stride + (t1_lo + s1) * t1_stride;

    if (is_low) {
        for (int g = 1; g <= ghost; ++g) {
            int dst = (ghost - g) * axis_stride + face_off;
            data[dst] = value;
        }
    } else {
        for (int g = 1; g <= ghost; ++g) {
            int dst = (ghost + n_axis + g - 1) * axis_stride + face_off;
            data[dst] = value;
        }
    }
}

// ===========================================================================
// CPU helpers (mirror of device kernels)
// ===========================================================================

static void cpu_periodic(Real* data, const PatchParams& pp) {
    for (int s0 = 0; s0 < pp.t0_count; ++s0)
    for (int s1 = 0; s1 < pp.t1_count; ++s1) {
        int face_off = (pp.t0_lo + s0) * pp.t0_stride
                     + (pp.t1_lo + s1) * pp.t1_stride;
        for (int g = 1; g <= pp.ghost; ++g) {
            int lo_ghost  = (pp.ghost - g)                 * pp.axis_stride + face_off;
            int lo_source = (pp.ghost + pp.n_axis - g)     * pp.axis_stride + face_off;
            int hi_ghost  = (pp.ghost + pp.n_axis + g - 1) * pp.axis_stride + face_off;
            int hi_source = (pp.ghost + g - 1)             * pp.axis_stride + face_off;
            data[lo_ghost] = data[lo_source];
            data[hi_ghost] = data[hi_source];
        }
    }
}

static void cpu_noflux(Real* data, const PatchParams& pp, bool is_low) {
    for (int s0 = 0; s0 < pp.t0_count; ++s0)
    for (int s1 = 0; s1 < pp.t1_count; ++s1) {
        int face_off = (pp.t0_lo + s0) * pp.t0_stride
                     + (pp.t1_lo + s1) * pp.t1_stride;
        if (is_low) {
            int src = pp.ghost * pp.axis_stride + face_off;
            for (int g = 1; g <= pp.ghost; ++g) {
                int dst = (pp.ghost - g) * pp.axis_stride + face_off;
                data[dst] = data[src];
            }
        } else {
            int src = (pp.ghost + pp.n_axis - 1) * pp.axis_stride + face_off;
            for (int g = 1; g <= pp.ghost; ++g) {
                int dst = (pp.ghost + pp.n_axis + g - 1) * pp.axis_stride + face_off;
                data[dst] = data[src];
            }
        }
    }
}

static void cpu_fixed(Real* data, const PatchParams& pp,
                      bool is_low, double value) {
    for (int s0 = 0; s0 < pp.t0_count; ++s0)
    for (int s1 = 0; s1 < pp.t1_count; ++s1) {
        int face_off = (pp.t0_lo + s0) * pp.t0_stride
                     + (pp.t1_lo + s1) * pp.t1_stride;
        if (is_low) {
            for (int g = 1; g <= pp.ghost; ++g) {
                int dst = (pp.ghost - g) * pp.axis_stride + face_off;
                data[dst] = value;
            }
        } else {
            for (int g = 1; g <= pp.ghost; ++g) {
                int dst = (pp.ghost + pp.n_axis + g - 1) * pp.axis_stride + face_off;
                data[dst] = value;
            }
        }
    }
}

// ===========================================================================
// PeriodicBC
// ===========================================================================

PeriodicBC::PeriodicBC(const Patch& patch) : BoundaryCondition(patch) {}

void PeriodicBC::applyOnCPU(ScalarField& f) const {
    auto pp = makePatchParams(f, patch);
    cpu_periodic(f.curr.data(), pp);
}

void PeriodicBC::applyOnGPU(ScalarField& f) const {
    if (!f.deviceAllocated())
        throw std::runtime_error("PeriodicBC::applyOnGPU: device not allocated");

    auto pp = makePatchParams(f, patch);
    dim3 block(16, 16);
    dim3 grid((pp.t0_count + 15) / 16, (pp.t1_count + 15) / 16);

    kernel_periodic<<<grid, block>>>(
        f.d_curr,
        pp.t0_count, pp.t1_count,
        pp.t0_stride, pp.t1_stride,
        pp.t0_lo,     pp.t1_lo,
        pp.axis_stride, pp.n_axis, pp.ghost);

    CUDA_CHECK(cudaGetLastError());
}

// ===========================================================================
// NoFluxBC
// ===========================================================================

NoFluxBC::NoFluxBC(const Patch& patch) : BoundaryCondition(patch) {}

void NoFluxBC::applyOnCPU(ScalarField& f) const {
    auto pp = makePatchParams(f, patch);
    cpu_noflux(f.curr.data(), pp, patch.side == Side::LOW);
}

void NoFluxBC::applyOnGPU(ScalarField& f) const {
    if (!f.deviceAllocated())
        throw std::runtime_error("NoFluxBC::applyOnGPU: device not allocated");

    auto pp = makePatchParams(f, patch);
    dim3 block(16, 16);
    dim3 grid((pp.t0_count + 15) / 16, (pp.t1_count + 15) / 16);

    kernel_noflux<<<grid, block>>>(
        f.d_curr,
        pp.t0_count, pp.t1_count,
        pp.t0_stride, pp.t1_stride,
        pp.t0_lo,     pp.t1_lo,
        pp.axis_stride, pp.n_axis, pp.ghost,
        patch.side == Side::LOW);

    CUDA_CHECK(cudaGetLastError());
}

// ===========================================================================
// FixedBC
// ===========================================================================

FixedBC::FixedBC(const Patch& patch, double value)
    : BoundaryCondition(patch), value(value) {}

void FixedBC::applyOnCPU(ScalarField& f) const {
    auto pp = makePatchParams(f, patch);
    cpu_fixed(f.curr.data(), pp, patch.side == Side::LOW, value);
}

void FixedBC::applyOnGPU(ScalarField& f) const {
    if (!f.deviceAllocated())
        throw std::runtime_error("FixedBC::applyOnGPU: device not allocated");

    auto pp = makePatchParams(f, patch);
    dim3 block(16, 16);
    dim3 grid((pp.t0_count + 15) / 16, (pp.t1_count + 15) / 16);

    kernel_fixed<<<grid, block>>>(
        f.d_curr,
        pp.t0_count, pp.t1_count,
        pp.t0_stride, pp.t1_stride,
        pp.t0_lo,     pp.t1_lo,
        pp.axis_stride, pp.n_axis, pp.ghost,
        patch.side == Side::LOW,
        value);

    CUDA_CHECK(cudaGetLastError());
}

} // namespace PhiX
