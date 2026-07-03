#pragma once

// ---------------------------------------------------------------------------
// BCBatch.h — apply ALL boundary conditions of a field in ONE kernel launch.
//
// Motivation: ghost refresh was one kernel launch per BC per field per step
// (a 2D field with 4 BCs = 4 launches, multiplied by fields and RK4 stages,
// and by CG iterations inside implicit operators).  Launch latency, not the
// ghost work itself, dominates that cost — batching collapses it to one
// launch per field.
//
// Safety: the built-in BCs write disjoint ghost bands (per axis/side; face
// patches exclude the corner cells) and read only physical cells, so they
// can run concurrently inside one kernel with no ordering hazards — the
// previous sequential launches never relied on order either.
//
// Supported subclasses: PeriodicBC, NoFluxBC, FixedBC (identified via
// dynamic_cast at build()).  Any other BoundaryCondition subclass is kept
// on a fallback list and applied with its own applyOnGPU after the batched
// kernel — correctness is never lost, only the batching benefit.
//
// The descriptor table is built once (device-resident); apply() reads the
// field's CURRENT d_curr, so pointer-swap tricks (RK4 stages) keep working.
//
//     BCBatch batch;
//     batch.build(field, {&bcx, &bcyLo, &bcyHi});
//     ... per step: batch.applyOnGPU(field); ...
// ---------------------------------------------------------------------------

#include "core/Real.h"
#include "boundary/BoundaryCondition.h"

#include <cuda_runtime.h>
#include "field/ScalarField.h"

#include <vector>

namespace PhiX {

// Flattened per-BC descriptor (POD, device-copyable)
struct BCDesc {
    int  type;                    // 0 periodic, 1 noflux, 2 fixed
    int  is_low;
    int  t0_count, t1_count;
    int  t0_stride, t1_stride;
    int  t0_lo, t1_lo;
    int  axis_stride, n_axis, ghost;
    Real value;                   // fixed only
};

class BCBatch {
public:
    BCBatch() = default;
    ~BCBatch();

    BCBatch(const BCBatch&)            = delete;
    BCBatch& operator=(const BCBatch&) = delete;

    // Flatten `bcs` (as applied to fields sharing layoutRef's layout) into
    // the device descriptor table.  May be called again to rebuild.
    // The BC objects must outlive this batch (fallback entries are used
    // by pointer).
    void build(const ScalarField& layoutRef,
               const std::vector<BoundaryCondition*>& bcs);

    bool built() const { return built_; }
    int  batchedCount()  const { return nDesc_; }
    int  fallbackCount() const { return static_cast<int>(fallback_.size()); }

    // One batched kernel launch (+ any fallback BCs).  No-op for an empty
    // batch.  f must share the layout given to build().
    // `stream`: launch stream (nullptr = default).  NOTE: fallback BCs do
    // not take a stream — callers that require full stream control (graph
    // capture) must check fallbackCount() == 0.
    void applyOnGPU(ScalarField& f, cudaStream_t stream = nullptr) const;

private:
    std::vector<BoundaryCondition*> fallback_;
    BCDesc* d_descs_ = nullptr;
    int     nDesc_   = 0;
    int     maxT0_   = 0, maxT1_ = 0;
    bool    built_   = false;
};

} // namespace PhiX
