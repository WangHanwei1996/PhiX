#pragma once

#include <string>
#include <stdexcept>

namespace PhiX {

enum class CoordSys {
    CARTESIAN,
    CYLINDRICAL,
    SPHERICAL
};

class Mesh {
public:
    // ---------------------------------------------------------------
    // Data members (all public for easy kernel-side read)
    // ---------------------------------------------------------------
    int    dim;
    CoordSys coordSys;
    int    n[3];       // nx, ny, nz  (unused directions set to 1)
    double d[3];       // dx, dy, dz
    double origin[3];  // x0, y0, z0

    // ---------------------------------------------------------------
    // Construction
    // ---------------------------------------------------------------

    // Default: uninitialized (call isValid() before use)
    Mesh();

    // Direct parameter constructor
    Mesh(int dim, CoordSys cs,
         int nx, double dx, double x0,
         int ny, double dy, double y0,
         int nz, double dz, double z0);

    // Static factory helpers (cleaner call sites for 1D/2D)
    static Mesh makeUniform1D(CoordSys cs,
                              int nx, double dx, double x0 = 0.0);

    static Mesh makeUniform2D(CoordSys cs,
                              int nx, double dx, double x0,
                              int ny, double dy, double y0);

    static Mesh makeUniform3D(CoordSys cs,
                              int nx, double dx, double x0,
                              int ny, double dy, double y0,
                              int nz, double dz, double z0);

    // ---------------------------------------------------------------
    // IO
    // ---------------------------------------------------------------
    static Mesh readFromFile(const std::string& path);
    void write(const std::string& path) const;
    void print() const;

    // ---------------------------------------------------------------
    // Queries (all inline, safe to call from CPU kernel wrappers)
    // ---------------------------------------------------------------

    // Total number of grid points
    inline std::size_t totalSize() const {
        return static_cast<std::size_t>(n[0]) * n[1] * n[2];
    }

    // Cell-centre coordinate along axis (0=x,1=y,2=z) at index i
    inline double coord(int axis, int i) const {
        return origin[axis] + (i + 0.5) * d[axis];
    }

    // Row-major (x fast, z slow) linear index: index(i,j,k)
    inline int index(int i, int j, int k) const {
        return i + n[0] * (j + n[1] * k);
    }

    // 1-D convenience
    inline int index(int i) const { return i; }

    // 2-D convenience
    inline int index(int i, int j) const { return i + n[0] * j; }

    // Validate all parameters; throws std::invalid_argument on failure
    void validate() const;

    // Returns true / false without throwing
    bool isValid() const noexcept;

private:
    // Common initialisation used by constructors
    void init(int dim_, CoordSys cs_,
              int nx, double dx, double x0,
              int ny, double dy, double y0,
              int nz, double dz, double z0);
};

// String helpers (used in IO)
std::string coordSysToString(CoordSys cs);
CoordSys    stringToCoordSys(const std::string& s);

} // namespace PhiX
