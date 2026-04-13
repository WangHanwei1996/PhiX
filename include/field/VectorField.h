#pragma once

#include "field/ScalarField.h"
#include "mesh/Mesh.h"

#include <string>
#include <vector>
#include <cstddef>

namespace PhiX {

// ---------------------------------------------------------------------------
// VectorField
//
// Vector-valued field stored as SoA (Structure of Arrays):
// N independent ScalarField objects, one per component.
//
// Component naming convention:
//   nComponents == 3  ->  name_x, name_y, name_z
//   otherwise        ->  name_0, name_1, ...
//
// Ghost layout, indexing, and GPU management are all delegated to the
// underlying ScalarField objects (see ScalarField.h for details).
//
// Output formats:
//   BINARY (.vfield) : small binary file; readable by VectorField::readFromFile
//   DAT    (.dat)    : ASCII, columns: x y z v0 v1 ...
//   VTS    (.vts)    : VTK XML StructuredGrid with vector CellData
//                      (opens directly in ParaView / VisIt)
// ---------------------------------------------------------------------------

class VectorField {
public:
    // -----------------------------------------------------------------------
    // Construction / destruction
    // -----------------------------------------------------------------------

    // ghost defaults to 1 to match ScalarField convention.
    explicit VectorField(const Mesh& mesh,
                         const std::string& name,
                         int nComponents,
                         int ghost = 1);

    // Non-copyable (each component ScalarField owns GPU memory)
    VectorField(const VectorField&)            = delete;
    VectorField& operator=(const VectorField&) = delete;

    // Movable
    VectorField(VectorField&& other) noexcept;
    VectorField& operator=(VectorField&& other) noexcept;

    ~VectorField() = default;

    // -----------------------------------------------------------------------
    // Component access
    // -----------------------------------------------------------------------
    ScalarField&       operator[](int c);
    const ScalarField& operator[](int c) const;

    int nComponents() const { return static_cast<int>(components_.size()); }

    // Convenience accessors for 3-component fields
    ScalarField& x() { return (*this)[0]; }
    ScalarField& y() { return (*this)[1]; }
    ScalarField& z() { return (*this)[2]; }
    const ScalarField& x() const { return (*this)[0]; }
    const ScalarField& y() const { return (*this)[1]; }
    const ScalarField& z() const { return (*this)[2]; }

    // -----------------------------------------------------------------------
    // Metadata
    // -----------------------------------------------------------------------
    const Mesh&  mesh;    // shared mesh (from first component)
    int          ghost;   // ghost layers
    std::string  name;    // base name (components are name_x/y/z or name_0...)

    // -----------------------------------------------------------------------
    // Initialisation helpers (CPU) — delegates to all components
    // -----------------------------------------------------------------------
    void fill(double value);
    void fillCurr(double value);
    void fillPrev(double value);

    // -----------------------------------------------------------------------
    // Time-stepping — delegates to all components
    // -----------------------------------------------------------------------
    void advanceTimeLevelCPU();
    void advanceTimeLevelGPU();

    // -----------------------------------------------------------------------
    // GPU management — delegates to all components
    // -----------------------------------------------------------------------
    bool deviceAllocated() const;   // true if ALL components are allocated

    void allocDevice();
    void freeDevice();

    void uploadCurrToDevice()  const;
    void uploadPrevToDevice()  const;
    void uploadAllToDevice()   const;

    void downloadCurrFromDevice();
    void downloadPrevFromDevice();
    void downloadAllFromDevice();

    // -----------------------------------------------------------------------
    // IO
    //
    // BINARY (.vfield):
    //   Text header:
    //     "# PhiX VectorField\n"
    //     "name         <name>\n"
    //     "nComponents  <N>\n"
    //     "nx <nx>  ny <ny>  nz <nz>\n"
    //     "ghost        <ghost>\n"
    //     "---\n"
    //   Binary data:
    //     For each component c in [0, N):
    //       nx*ny*nz doubles, row-major (x fastest, z slowest)
    //
    // DAT (.dat):
    //   One line per cell:  x  y  z  v0  v1  ...
    //
    // VTS (.vts):
    //   VTK XML StructuredGrid with vector CellData.
    //   DataArray is interleaved: for each cell (x fastest) output v0 v1 ... vN-1.
    // -----------------------------------------------------------------------
    void write(const std::string& path,
               FieldFormat fmt = FieldFormat::BINARY) const;

    static VectorField readFromFile(const Mesh& mesh,
                                    const std::string& path,
                                    int ghost = 1);

    // Human-readable summary for each component
    void print() const;

private:
    std::vector<ScalarField> components_;

    void writeBinary(const std::string& path) const;
    void writeDat   (const std::string& path) const;
    void writeVts   (const std::string& path) const;

    static std::string componentName(const std::string& baseName,
                                     int c, int nComp);
};

} // namespace PhiX
