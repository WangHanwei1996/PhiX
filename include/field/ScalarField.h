#pragma once

#include "core/Real.h"
#include "field/FieldLayout.h"
#include "mesh/Mesh.h"
#include "IO/FieldFormat.h"

#include <string>
#include <vector>
#include <cstddef>

namespace PhiX {

// ---------------------------------------------------------------------------
// ScalarField
//
// Scalar double-precision field on a structured Mesh.
// Each direction is padded with `ghost` extra cells on both sides (halo):
//
//   storedDims[ax] = mesh.n[ax] + 2*ghost
//
// Physical cell indices : i in [0, mesh.n[ax])
// Ghost    cell indices : i in [-ghost, 0)  and  [mesh.n[ax], mesh.n[ax]+ghost)
//
// Flat linear index (row-major, x fast, z slow):
//   index(i,j,k) = (i+ghost) + storedDims[0]*((j+ghost) + storedDims[1]*(k+ghost))
//
// CPU arrays are always valid after construction.
// GPU arrays are allocated lazily via allocDevice().
// ---------------------------------------------------------------------------

class ScalarField {
public:
    // -----------------------------------------------------------------------
    // Identity
    // -----------------------------------------------------------------------
    std::string name;
    FieldLayout layout;

    // -----------------------------------------------------------------------
    // Geometry (cached from Mesh at construction)
    // -----------------------------------------------------------------------
    const Mesh& mesh;
    int         ghost;
    int         storedDims[3];
    std::size_t storedSize;

    // -----------------------------------------------------------------------
    // CPU storage
    // -----------------------------------------------------------------------
    std::vector<Real> curr;   // current  time level
    std::vector<Real> prev;   // previous time level

    // -----------------------------------------------------------------------
    // GPU storage (nullptr until allocDevice() is called)
    // -----------------------------------------------------------------------
    Real* d_curr = nullptr;
    Real* d_prev = nullptr;

    // -----------------------------------------------------------------------
    // Previous-time-level tracking (opt-in since v2.18.0).
    //
    // Default OFF: allocDevice() allocates only d_curr (halving device
    // memory per field) and advanceTimeLevelGPU/CPU are no-ops (saving a
    // full-field copy per step).  Set trackPrev = true BEFORE allocDevice()
    // if your model reads the pre-step value (e.g. dphi/dt via
    // (curr − prev)/dt); the solvers then rotate curr → prev at the end of
    // every step exactly as before.
    // -----------------------------------------------------------------------
    bool trackPrev = false;

    // -----------------------------------------------------------------------
    // Construction / destruction
    // -----------------------------------------------------------------------
    explicit ScalarField(const Mesh& mesh,
                         const std::string& name,
                         int ghost = 1);

    explicit ScalarField(const FieldLayout& layout,
                         const std::string& name);

    // -----------------------------------------------------------------------
    // Shell factory — build a non-owning ScalarField wrapping an externally
    // managed device buffer (e.g. one obtained from ScratchPool::acquire).
    //
    // The returned field has only `d_curr` set (to `d_buf`); `d_prev` and
    // CPU buffers are unallocated.  The destructor will NOT cudaFree the
    // wrapped pointer.  Intended for passing scratch buffers through the
    // BoundaryCondition::applyOnGPU interface.
    // -----------------------------------------------------------------------
    static ScalarField makeShell(const Mesh& mesh, int ghost,
                                 Real* d_buf,
                                 const std::string& name = "shell");

    // Non-copyable (owns GPU memory)
    ScalarField(const ScalarField&)            = delete;
    ScalarField& operator=(const ScalarField&) = delete;

    // Movable
    ScalarField(ScalarField&& other) noexcept;
    ScalarField& operator=(ScalarField&& other) noexcept;

    ~ScalarField();

    // -----------------------------------------------------------------------
    // Inline index mapping  (physical OR ghost indices accepted)
    // -----------------------------------------------------------------------
    inline int index(int i, int j, int k) const {
        return layout.index(i, j, k);
    }
    inline int index(int i, int j) const { return index(i, j, 0); }
    inline int index(int i)        const { return index(i, 0, 0); }

    // -----------------------------------------------------------------------
    // Initialisation helpers (CPU)
    // -----------------------------------------------------------------------
    void fill(double value);
    void fillCurr(double value);
    void fillPrev(double value);

    // Initialize curr by evaluating fn(x, y, z) at every physical cell centre.
    // fn must be callable as `double fn(double x, double y, double z)`.
    // Uses mesh.coord(axis, i) for coordinates; ghost cells are left unchanged.
    template<typename Fn>
    void initialize(Fn fn) {
        const int nx = mesh.n[0], ny = mesh.n[1], nz = mesh.n[2];
        for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            double x = mesh.coord(0, i);
            double y = (mesh.dim >= 2) ? mesh.coord(1, j) : 0.0;
            double z = (mesh.dim >= 3) ? mesh.coord(2, k) : 0.0;
            curr[static_cast<std::size_t>(layout.index(i, j, k))] = fn(x, y, z);
        }
    }

    // -----------------------------------------------------------------------
    // Time-stepping
    // -----------------------------------------------------------------------
    void advanceTimeLevelCPU();   // prev <- curr  (std::copy, CPU)
    void advanceTimeLevelGPU();   // d_prev <- d_curr  (cudaMemcpy D->D)

    // -----------------------------------------------------------------------
    // GPU management
    // -----------------------------------------------------------------------
    bool deviceAllocated() const { return d_curr != nullptr; }

    void allocDevice();    // cudaMalloc d_curr and d_prev
    void freeDevice();     // cudaFree both; sets pointers to nullptr

    void uploadCurrToDevice()  const;   // CPU curr -> GPU d_curr
    void uploadPrevToDevice()  const;   // CPU prev -> GPU d_prev
    void uploadAllToDevice()   const;

    void downloadCurrFromDevice();      // GPU d_curr -> CPU curr
    void downloadPrevFromDevice();      // GPU d_prev -> CPU prev
    void downloadAllFromDevice();

    // True for fields constructed via makeShell() — destructor does NOT free
    // the wrapped device pointer.
    bool ownsDeviceMemory() const { return ownsDeviceMemory_; }

    // -----------------------------------------------------------------------
    // IO  (physical cells only; ghost cells are NOT persisted)
    //
    // BINARY (default, .field):
    //   Text header followed by raw IEEE-754 double data.
    //   Header:  "# PhiX ScalarField\n"
    //            "name    <name>\n"
    //            "nx <nx>  ny <ny>  nz <nz>\n"
    //            "ghost   <ghost>\n"
    //            "---\n"
    //   Data:    nx*ny*nz doubles, row-major (x fastest, z slowest)
    //
    // DAT (.dat):
    //   Plain ASCII, one line per cell:  x  y  z  value
    //   Coordinates are cell-centre positions computed from mesh.origin / mesh.d.
    //   Suitable for gnuplot / matplotlib / numpy.loadtxt.
    //
    // VTS (.vts):
    //   VTK XML StructuredGrid with CellData.
    //   Opens directly in ParaView, VisIt, and other VTK-based tools.
    // -----------------------------------------------------------------------
    void write(const std::string& path,
               FieldFormat fmt = FieldFormat::BINARY) const;

    // Read physical cells into curr; prev is unchanged.
    // Header nx/ny/nz are validated against mesh.
    static ScalarField readFromFile(const Mesh& mesh,
                                    const std::string& path,
                                    int ghost = 1);

    // Human-readable summary (name, dims, ghost, curr min/max/mean)
    void print() const;

private:
    // True for normal ScalarFields (allocDevice cudaMallocs both buffers).
    // False for shell wrappers built via makeShell() \u2014 destructor will not
    // cudaFree the wrapped pointer.
    bool ownsDeviceMemory_ = true;
};

// ---------------------------------------------------------------------------
// Backward-compatibility alias — existing code using `Field` still compiles.
// New code should prefer `ScalarField`.
// ---------------------------------------------------------------------------
using Field = ScalarField;

} // namespace PhiX
