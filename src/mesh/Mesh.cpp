#include "mesh/Mesh.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cctype>

namespace PhiX {

// -----------------------------------------------------------------------
// String conversion helpers
// -----------------------------------------------------------------------

std::string coordSysToString(CoordSys cs) {
    switch (cs) {
        case CoordSys::CARTESIAN:   return "CARTESIAN";
        case CoordSys::CYLINDRICAL: return "CYLINDRICAL";
        case CoordSys::SPHERICAL:   return "SPHERICAL";
    }
    return "CARTESIAN";
}

CoordSys stringToCoordSys(const std::string& s) {
    std::string upper = s;
    std::transform(upper.begin(), upper.end(), upper.begin(),
                   [](unsigned char c){ return std::toupper(c); });
    if (upper == "CYLINDRICAL") return CoordSys::CYLINDRICAL;
    if (upper == "SPHERICAL")   return CoordSys::SPHERICAL;
    if (upper == "CARTESIAN")   return CoordSys::CARTESIAN;
    throw std::invalid_argument("Unknown CoordSys: " + s);
}

// -----------------------------------------------------------------------
// Private init
// -----------------------------------------------------------------------

void Mesh::init(int dim_, CoordSys cs_,
                int nx, double dx, double x0,
                int ny, double dy, double y0,
                int nz, double dz, double z0) {
    dim      = dim_;
    coordSys = cs_;
    n[0] = nx; d[0] = dx; origin[0] = x0;
    n[1] = ny; d[1] = dy; origin[1] = y0;
    n[2] = nz; d[2] = dz; origin[2] = z0;
}

// -----------------------------------------------------------------------
// Constructors
// -----------------------------------------------------------------------

Mesh::Mesh() {
    init(1, CoordSys::CARTESIAN,
         0, 0.0, 0.0,
         1, 1.0, 0.0,
         1, 1.0, 0.0);
}

Mesh::Mesh(int dim_, CoordSys cs_,
           int nx, double dx, double x0,
           int ny, double dy, double y0,
           int nz, double dz, double z0) {
    init(dim_, cs_, nx, dx, x0, ny, dy, y0, nz, dz, z0);
    validate();
}

// -----------------------------------------------------------------------
// Static factory methods
// -----------------------------------------------------------------------

Mesh Mesh::makeUniform1D(CoordSys cs, int nx, double dx, double x0) {
    return Mesh(1, cs, nx, dx, x0, 1, 1.0, 0.0, 1, 1.0, 0.0);
}

Mesh Mesh::makeUniform2D(CoordSys cs,
                         int nx, double dx, double x0,
                         int ny, double dy, double y0) {
    return Mesh(2, cs, nx, dx, x0, ny, dy, y0, 1, 1.0, 0.0);
}

Mesh Mesh::makeUniform3D(CoordSys cs,
                         int nx, double dx, double x0,
                         int ny, double dy, double y0,
                         int nz, double dz, double z0) {
    return Mesh(3, cs, nx, dx, x0, ny, dy, y0, nz, dz, z0);
}

// -----------------------------------------------------------------------
// Validation
// -----------------------------------------------------------------------

void Mesh::validate() const {
    if (dim < 1 || dim > 3)
        throw std::invalid_argument("Mesh: dim must be 1, 2, or 3");

    for (int ax = 0; ax < dim; ++ax) {
        if (n[ax] <= 0)
            throw std::invalid_argument("Mesh: n[" + std::to_string(ax) + "] must be > 0");
        if (d[ax] <= 0.0)
            throw std::invalid_argument("Mesh: d[" + std::to_string(ax) + "] must be > 0");
    }

    // Axes beyond dim must be degenerate (n=1)
    for (int ax = dim; ax < 3; ++ax) {
        if (n[ax] != 1)
            throw std::invalid_argument(
                "Mesh: n[" + std::to_string(ax) + "] must be 1 for unused axes");
    }
}

bool Mesh::isValid() const noexcept {
    try {
        validate();
        return true;
    } catch (...) {
        return false;
    }
}

// -----------------------------------------------------------------------
// IO: write
// -----------------------------------------------------------------------

void Mesh::write(const std::string& path) const {
    validate();

    std::ofstream ofs(path);
    if (!ofs)
        throw std::runtime_error("Mesh::write: cannot open file: " + path);

    ofs << "# PhiX Mesh\n";
    ofs << "dim      " << dim << "\n";
    ofs << "coordSys " << coordSysToString(coordSys) << "\n";
    ofs << "nx " << n[0] << "  dx " << d[0] << "  x0 " << origin[0] << "\n";
    ofs << "ny " << n[1] << "  dy " << d[1] << "  y0 " << origin[1] << "\n";
    ofs << "nz " << n[2] << "  dz " << d[2] << "  z0 " << origin[2] << "\n";
}

// -----------------------------------------------------------------------
// IO: read
// -----------------------------------------------------------------------

Mesh Mesh::readFromFile(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs)
        throw std::runtime_error("Mesh::readFromFile: cannot open file: " + path);

    int dim_  = 1;
    CoordSys cs = CoordSys::CARTESIAN;
    int    n_[3]      = {1, 1, 1};
    double d_[3]      = {1.0, 1.0, 1.0};
    double origin_[3] = {0.0, 0.0, 0.0};

    // Direction labels used in the file: nx/dx/x0, ny/dy/y0, nz/dz/z0
    const char* nKey[3]   = {"nx", "ny", "nz"};
    const char* dKey[3]   = {"dx", "dy", "dz"};
    const char* o0Key[3]  = {"x0", "y0", "z0"};

    std::string line;
    while (std::getline(ifs, line)) {
        // Strip comments
        auto pos = line.find('#');
        if (pos != std::string::npos) line = line.substr(0, pos);

        std::istringstream ss(line);
        std::string key;
        if (!(ss >> key)) continue;

        if (key == "dim") {
            ss >> dim_;
        } else if (key == "coordSys") {
            std::string val; ss >> val;
            cs = stringToCoordSys(val);
        } else {
            // Try to match nx/dx/x0 etc. (any order on the line is fine)
            // First token already read as `key`; push it back by re-scanning the line
            std::istringstream row(line);
            std::string tok;
            while (row >> tok) {
                for (int ax = 0; ax < 3; ++ax) {
                    if (tok == nKey[ax])  { row >> n_[ax];      break; }
                    if (tok == dKey[ax])  { row >> d_[ax];      break; }
                    if (tok == o0Key[ax]) { row >> origin_[ax]; break; }
                }
            }
        }
    }

    return Mesh(dim_, cs,
                n_[0], d_[0], origin_[0],
                n_[1], d_[1], origin_[1],
                n_[2], d_[2], origin_[2]);
}

// -----------------------------------------------------------------------
// print
// -----------------------------------------------------------------------

void Mesh::print() const {
    std::cout << "=== Mesh ===\n";
    std::cout << "  dim      : " << dim << "\n";
    std::cout << "  coordSys : " << coordSysToString(coordSys) << "\n";
    const char* axName[3] = {"x", "y", "z"};
    for (int ax = 0; ax < 3; ++ax) {
        std::cout << "  " << axName[ax] << ": n=" << n[ax]
                  << "  d=" << d[ax]
                  << "  origin=" << origin[ax];
        if (ax >= dim) std::cout << "  (inactive)";
        std::cout << "\n";
    }
    std::cout << "  totalSize: " << totalSize() << "\n";
}

} // namespace PhiX
