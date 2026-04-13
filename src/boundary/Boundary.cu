#include "boundary/PeriodicBC.h"
#include "boundary/NoFluxBC.h"
#include "boundary/FixedBC.h"

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
// Generic GPU kernel design
// ===========================================================================
//
// Row-major storage: flat(is, js, ks) = is + sx*(js + sy*ks)
// where is/js/ks are stored indices (physical index + ghost offset).
//
// For each axis we identify:
//   axis_stride  : distance in flat index between adjacent stored cells
//                  X -> 1,  Y -> sx,  Z -> sx*sy
//   face threads : 2-D thread block covering the two remaining dimensions
//                  in full stored extent (including ghost of other axes).
//   face_offset  : flat-index contribution from the two thread dims.
//
// This allows one kernel per BC type to handle all three axes uniformly.
// ===========================================================================

// ---------------------------------------------------------------------------
// Helper: compute kernel launch geometry for a given axis
// ---------------------------------------------------------------------------
struct FaceParams {
    int axis_stride;   // stride along the BC axis in flat memory
    int n_axis;        // number of physical cells along BC axis
    int n_face0;       // thread count dim 0 (full stored extent)
    int n_face1;       // thread count dim 1 (full stored extent)
    int face_stride0;  // flat-index stride for thread dim 0
    int face_stride1;  // flat-index stride for thread dim 1
};

static FaceParams makeFaceParams(const ScalarField& f, Axis ax) {
    int sx = f.storedDims[0];
    int sy = f.storedDims[1];
    int sz = f.storedDims[2];
    FaceParams p{};
    switch (ax) {
        case Axis::X:
            p.axis_stride  = 1;
            p.n_axis       = f.mesh.n[0];
            p.n_face0      = sy;   // threads over j (stored)
            p.n_face1      = sz;   // threads over k (stored)
            p.face_stride0 = sx;
            p.face_stride1 = sx * sy;
            break;
        case Axis::Y:
            p.axis_stride  = sx;
            p.n_axis       = f.mesh.n[1];
            p.n_face0      = sx;   // threads over i (stored)
            p.n_face1      = sz;   // threads over k (stored)
            p.face_stride0 = 1;
            p.face_stride1 = sx * sy;
            break;
        case Axis::Z:
            p.axis_stride  = sx * sy;
            p.n_axis       = f.mesh.n[2];
            p.n_face0      = sx;   // threads over i (stored)
            p.n_face1      = sy;   // threads over j (stored)
            p.face_stride0 = 1;
            p.face_stride1 = sx;
            break;
    }
    return p;
}

// ---------------------------------------------------------------------------
// Periodic kernel
//
// Each thread handles one (face0, face1) position and fills ALL ghost layers
// along the BC axis (loop over g = 1..ghost).
//
// Low ghost:  stored_idx = ghost - g  <-- copy from stored_idx = ghost + n - g
// High ghost: stored_idx = ghost+n+g-1 <-- copy from stored_idx = ghost + g - 1
// ---------------------------------------------------------------------------
__global__ void kernel_periodic(
        double* data,
        int n_face0, int n_face1,
        int axis_stride,
        int n_axis,
        int ghost,
        int face_stride0,
        int face_stride1)
{
    int t0 = blockIdx.x * blockDim.x + threadIdx.x;
    int t1 = blockIdx.y * blockDim.y + threadIdx.y;
    if (t0 >= n_face0 || t1 >= n_face1) return;

    int face_off = t0 * face_stride0 + t1 * face_stride1;

    for (int g = 1; g <= ghost; ++g) {
        int lo_ghost  = (ghost - g)          * axis_stride + face_off;
        int lo_source = (ghost + n_axis - g) * axis_stride + face_off;
        int hi_ghost  = (ghost + n_axis + g - 1) * axis_stride + face_off;
        int hi_source = (ghost + g - 1)      * axis_stride + face_off;

        data[lo_ghost]  = data[lo_source];
        data[hi_ghost]  = data[hi_source];
    }
}

// ---------------------------------------------------------------------------
// NoFlux (zero-gradient) kernel
//
// Low ghost:  stored_idx = ghost - g  <-- copy from stored_idx = ghost (i=0)
// High ghost: stored_idx = ghost+n+g-1 <-- copy from stored_idx = ghost+n-1
// All ghost layers get the same nearest-boundary value (constant extrapolation).
// ---------------------------------------------------------------------------
__global__ void kernel_noflux(
        double* data,
        int n_face0, int n_face1,
        int axis_stride,
        int n_axis,
        int ghost,
        int face_stride0,
        int face_stride1,
        bool do_low, bool do_high)
{
    int t0 = blockIdx.x * blockDim.x + threadIdx.x;
    int t1 = blockIdx.y * blockDim.y + threadIdx.y;
    if (t0 >= n_face0 || t1 >= n_face1) return;

    int face_off = t0 * face_stride0 + t1 * face_stride1;

    if (do_low) {
        int src = ghost * axis_stride + face_off;          // physical i = 0
        for (int g = 1; g <= ghost; ++g) {
            int dst = (ghost - g) * axis_stride + face_off;
            data[dst] = data[src];
        }
    }
    if (do_high) {
        int src = (ghost + n_axis - 1) * axis_stride + face_off;  // physical i = n-1
        for (int g = 1; g <= ghost; ++g) {
            int dst = (ghost + n_axis + g - 1) * axis_stride + face_off;
            data[dst] = data[src];
        }
    }
}

// ---------------------------------------------------------------------------
// Fixed (Dirichlet) kernel
//
// Sets all ghost cells on selected side(s) to a constant value.
// (Constant fill; sufficient for first-order stencils.)
// ---------------------------------------------------------------------------
__global__ void kernel_fixed(
        double* data,
        int n_face0, int n_face1,
        int axis_stride,
        int n_axis,
        int ghost,
        int face_stride0,
        int face_stride1,
        bool do_low, bool do_high,
        double value)
{
    int t0 = blockIdx.x * blockDim.x + threadIdx.x;
    int t1 = blockIdx.y * blockDim.y + threadIdx.y;
    if (t0 >= n_face0 || t1 >= n_face1) return;

    int face_off = t0 * face_stride0 + t1 * face_stride1;

    if (do_low) {
        for (int g = 1; g <= ghost; ++g) {
            int dst = (ghost - g) * axis_stride + face_off;
            data[dst] = value;
        }
    }
    if (do_high) {
        for (int g = 1; g <= ghost; ++g) {
            int dst = (ghost + n_axis + g - 1) * axis_stride + face_off;
            data[dst] = value;
        }
    }
}

// ===========================================================================
// PeriodicBC
// ===========================================================================

PeriodicBC::PeriodicBC(Axis axis)
    : BoundaryCondition(axis, Side::BOTH) {}

void PeriodicBC::applyOnCPU(ScalarField& f) const {
    int g   = f.ghost;
    double* data = f.curr.data();

    // Loop using FaceParams logic directly on CPU
    auto fp = makeFaceParams(f, axis);
    for (int t0 = 0; t0 < fp.n_face0; ++t0)
    for (int t1 = 0; t1 < fp.n_face1; ++t1) {
        int face_off = t0 * fp.face_stride0 + t1 * fp.face_stride1;
        for (int layer = 1; layer <= g; ++layer) {
            int lo_ghost  = (g - layer)             * fp.axis_stride + face_off;
            int lo_source = (g + fp.n_axis - layer) * fp.axis_stride + face_off;
            int hi_ghost  = (g + fp.n_axis + layer - 1) * fp.axis_stride + face_off;
            int hi_source = (g + layer - 1)          * fp.axis_stride + face_off;
            data[lo_ghost]  = data[lo_source];
            data[hi_ghost]  = data[hi_source];
        }
    }
}

void PeriodicBC::applyOnGPU(ScalarField& f) const {
    if (!f.deviceAllocated())
        throw std::runtime_error("PeriodicBC::applyOnGPU: device not allocated");

    auto fp = makeFaceParams(f, axis);
    dim3 block(16, 16);
    dim3 grid((fp.n_face0 + 15) / 16, (fp.n_face1 + 15) / 16);

    kernel_periodic<<<grid, block>>>(
        f.d_curr,
        fp.n_face0, fp.n_face1,
        fp.axis_stride, fp.n_axis, f.ghost,
        fp.face_stride0, fp.face_stride1);

    CUDA_CHECK(cudaGetLastError());
}

// ===========================================================================
// NoFluxBC
// ===========================================================================

NoFluxBC::NoFluxBC(Axis axis, Side side)
    : BoundaryCondition(axis, side) {}

void NoFluxBC::applyOnCPU(ScalarField& f) const {
    bool do_low  = (side == Side::LOW  || side == Side::BOTH);
    bool do_high = (side == Side::HIGH || side == Side::BOTH);

    auto fp = makeFaceParams(f, axis);
    int  g  = f.ghost;
    double* data = f.curr.data();

    for (int t0 = 0; t0 < fp.n_face0; ++t0)
    for (int t1 = 0; t1 < fp.n_face1; ++t1) {
        int face_off = t0 * fp.face_stride0 + t1 * fp.face_stride1;

        if (do_low) {
            int src = g * fp.axis_stride + face_off;
            for (int layer = 1; layer <= g; ++layer) {
                int dst = (g - layer) * fp.axis_stride + face_off;
                data[dst] = data[src];
            }
        }
        if (do_high) {
            int src = (g + fp.n_axis - 1) * fp.axis_stride + face_off;
            for (int layer = 1; layer <= g; ++layer) {
                int dst = (g + fp.n_axis + layer - 1) * fp.axis_stride + face_off;
                data[dst] = data[src];
            }
        }
    }
}

void NoFluxBC::applyOnGPU(ScalarField& f) const {
    if (!f.deviceAllocated())
        throw std::runtime_error("NoFluxBC::applyOnGPU: device not allocated");

    bool do_low  = (side == Side::LOW  || side == Side::BOTH);
    bool do_high = (side == Side::HIGH || side == Side::BOTH);

    auto fp = makeFaceParams(f, axis);
    dim3 block(16, 16);
    dim3 grid((fp.n_face0 + 15) / 16, (fp.n_face1 + 15) / 16);

    kernel_noflux<<<grid, block>>>(
        f.d_curr,
        fp.n_face0, fp.n_face1,
        fp.axis_stride, fp.n_axis, f.ghost,
        fp.face_stride0, fp.face_stride1,
        do_low, do_high);

    CUDA_CHECK(cudaGetLastError());
}

// ===========================================================================
// FixedBC
// ===========================================================================

FixedBC::FixedBC(Axis axis, Side side, double value)
    : BoundaryCondition(axis, side), value(value) {}

void FixedBC::applyOnCPU(ScalarField& f) const {
    bool do_low  = (side == Side::LOW  || side == Side::BOTH);
    bool do_high = (side == Side::HIGH || side == Side::BOTH);

    auto fp = makeFaceParams(f, axis);
    int  g  = f.ghost;
    double* data = f.curr.data();

    for (int t0 = 0; t0 < fp.n_face0; ++t0)
    for (int t1 = 0; t1 < fp.n_face1; ++t1) {
        int face_off = t0 * fp.face_stride0 + t1 * fp.face_stride1;

        if (do_low) {
            for (int layer = 1; layer <= g; ++layer) {
                int dst = (g - layer) * fp.axis_stride + face_off;
                data[dst] = value;
            }
        }
        if (do_high) {
            for (int layer = 1; layer <= g; ++layer) {
                int dst = (g + fp.n_axis + layer - 1) * fp.axis_stride + face_off;
                data[dst] = value;
            }
        }
    }
}

void FixedBC::applyOnGPU(ScalarField& f) const {
    if (!f.deviceAllocated())
        throw std::runtime_error("FixedBC::applyOnGPU: device not allocated");

    bool do_low  = (side == Side::LOW  || side == Side::BOTH);
    bool do_high = (side == Side::HIGH || side == Side::BOTH);

    auto fp = makeFaceParams(f, axis);
    dim3 block(16, 16);
    dim3 grid((fp.n_face0 + 15) / 16, (fp.n_face1 + 15) / 16);

    kernel_fixed<<<grid, block>>>(
        f.d_curr,
        fp.n_face0, fp.n_face1,
        fp.axis_stride, fp.n_axis, f.ghost,
        fp.face_stride0, fp.face_stride1,
        do_low, do_high, value);

    CUDA_CHECK(cudaGetLastError());
}

} // namespace PhiX
