#pragma once

#include "mesh/Mesh.h"

#include <string>
#include <vector>
#include <cstddef>

namespace PhiX {

// ---------------------------------------------------------------------------
// Output format selector used by ScalarField::write()
// ---------------------------------------------------------------------------
enum class FieldFormat {
    BINARY,   ///< Custom binary format (.field)  — default; smallest files
    DAT,      ///< ASCII text with x y z value columns — easy for gnuplot/matplotlib
    VTS       ///< VTK XML StructuredGrid (.vts) — opens directly in ParaView/VisIt
};

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
    std::vector<double> curr;   // current  time level
    std::vector<double> prev;   // previous time level

    // -----------------------------------------------------------------------
    // GPU storage (nullptr until allocDevice() is called)
    // -----------------------------------------------------------------------
    double* d_curr = nullptr;
    double* d_prev = nullptr;

    // -----------------------------------------------------------------------
    // Construction / destruction
    // -----------------------------------------------------------------------
    explicit ScalarField(const Mesh& mesh,
                         const std::string& name,
                         int ghost = 1);

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
        return (i + ghost)
             + storedDims[0] * ((j + ghost)
             + storedDims[1] *  (k + ghost));
    }
    inline int index(int i, int j) const { return index(i, j, 0); }
    inline int index(int i)        const { return index(i, 0, 0); }

    // -----------------------------------------------------------------------
    // Initialisation helpers (CPU)
    // -----------------------------------------------------------------------
    void fill(double value);
    void fillCurr(double value);
    void fillPrev(double value);

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
    void writeBinary(const std::string& path) const;
    void writeDat   (const std::string& path) const;
    void writeVts   (const std::string& path) const;
};

// ---------------------------------------------------------------------------
// Backward-compatibility alias — existing code using `Field` still compiles.
// New code should prefer `ScalarField`.
// ---------------------------------------------------------------------------
using Field = ScalarField;

} // namespace PhiX
