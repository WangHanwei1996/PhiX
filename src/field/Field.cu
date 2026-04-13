#include "field/Field.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>

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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::size_t computeStoredSize(const int storedDims[3]) {
    return static_cast<std::size_t>(storedDims[0])
         * storedDims[1]
         * storedDims[2];
}

// Number of physical cells (no ghost)
static std::size_t physicalSize(const Mesh& mesh) {
    return static_cast<std::size_t>(mesh.n[0])
         * mesh.n[1]
         * mesh.n[2];
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

Field::Field(const Mesh& mesh_, const std::string& name_, int ghost_)
    : name(name_), mesh(mesh_), ghost(ghost_)
{
    if (ghost_ < 0)
        throw std::invalid_argument("Field: ghost must be >= 0");

    for (int ax = 0; ax < 3; ++ax)
        storedDims[ax] = mesh.n[ax] + 2 * ghost;

    storedSize = computeStoredSize(storedDims);

    curr.assign(storedSize, 0.0);
    prev.assign(storedSize, 0.0);
}

// ---------------------------------------------------------------------------
// Move semantics
// ---------------------------------------------------------------------------

Field::Field(Field&& other) noexcept
    : name(std::move(other.name))
    , mesh(other.mesh)
    , ghost(other.ghost)
    , storedSize(other.storedSize)
    , curr(std::move(other.curr))
    , prev(std::move(other.prev))
    , d_curr(other.d_curr)
    , d_prev(other.d_prev)
{
    storedDims[0] = other.storedDims[0];
    storedDims[1] = other.storedDims[1];
    storedDims[2] = other.storedDims[2];
    other.d_curr = nullptr;
    other.d_prev = nullptr;
}

Field& Field::operator=(Field&& other) noexcept {
    if (this == &other) return *this;
    freeDevice();

    name      = std::move(other.name);
    // mesh is a const ref — cannot rebind, caller is responsible for lifetime
    ghost     = other.ghost;
    storedDims[0] = other.storedDims[0];
    storedDims[1] = other.storedDims[1];
    storedDims[2] = other.storedDims[2];
    storedSize = other.storedSize;
    curr      = std::move(other.curr);
    prev      = std::move(other.prev);
    d_curr    = other.d_curr;  other.d_curr = nullptr;
    d_prev    = other.d_prev;  other.d_prev = nullptr;
    return *this;
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------

Field::~Field() {
    freeDevice();
}

// ---------------------------------------------------------------------------
// Initialisation helpers
// ---------------------------------------------------------------------------

void Field::fill(double value) {
    fillCurr(value);
    fillPrev(value);
}

void Field::fillCurr(double value) {
    std::fill(curr.begin(), curr.end(), value);
}

void Field::fillPrev(double value) {
    std::fill(prev.begin(), prev.end(), value);
}

// ---------------------------------------------------------------------------
// Time-stepping
// ---------------------------------------------------------------------------

void Field::advanceTimeLevelCPU() {
    std::copy(curr.begin(), curr.end(), prev.begin());
}

void Field::advanceTimeLevelGPU() {
    if (!deviceAllocated())
        throw std::runtime_error("Field::advanceTimeLevelGPU: device not allocated");
    CUDA_CHECK(cudaMemcpy(d_prev, d_curr,
                          storedSize * sizeof(double),
                          cudaMemcpyDeviceToDevice));
}

// ---------------------------------------------------------------------------
// GPU management
// ---------------------------------------------------------------------------

void Field::allocDevice() {
    if (deviceAllocated()) return;
    const std::size_t bytes = storedSize * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_curr, bytes));
    CUDA_CHECK(cudaMalloc(&d_prev, bytes));
    // Initialise GPU memory to zero
    CUDA_CHECK(cudaMemset(d_curr, 0, bytes));
    CUDA_CHECK(cudaMemset(d_prev, 0, bytes));
}

void Field::freeDevice() {
    if (d_curr) { cudaFree(d_curr); d_curr = nullptr; }
    if (d_prev) { cudaFree(d_prev); d_prev = nullptr; }
}

void Field::uploadCurrToDevice() const {
    if (!deviceAllocated())
        throw std::runtime_error("Field::uploadCurrToDevice: device not allocated");
    CUDA_CHECK(cudaMemcpy(d_curr, curr.data(),
                          storedSize * sizeof(double),
                          cudaMemcpyHostToDevice));
}

void Field::uploadPrevToDevice() const {
    if (!deviceAllocated())
        throw std::runtime_error("Field::uploadPrevToDevice: device not allocated");
    CUDA_CHECK(cudaMemcpy(d_prev, prev.data(),
                          storedSize * sizeof(double),
                          cudaMemcpyHostToDevice));
}

void Field::uploadAllToDevice() const {
    uploadCurrToDevice();
    uploadPrevToDevice();
}

void Field::downloadCurrFromDevice() {
    if (!deviceAllocated())
        throw std::runtime_error("Field::downloadCurrFromDevice: device not allocated");
    CUDA_CHECK(cudaMemcpy(curr.data(), d_curr,
                          storedSize * sizeof(double),
                          cudaMemcpyDeviceToHost));
}

void Field::downloadPrevFromDevice() {
    if (!deviceAllocated())
        throw std::runtime_error("Field::downloadPrevFromDevice: device not allocated");
    CUDA_CHECK(cudaMemcpy(prev.data(), d_prev,
                          storedSize * sizeof(double),
                          cudaMemcpyDeviceToHost));
}

void Field::downloadAllFromDevice() {
    downloadCurrFromDevice();
    downloadPrevFromDevice();
}

// ---------------------------------------------------------------------------
// IO helpers: iterate physical cells in row-major order
// ---------------------------------------------------------------------------

// Convert physical (i,j,k) into the stored flat index (ghost-padded)
static inline int physIdx(const Field& f, int i, int j, int k) {
    return (i + f.ghost)
         + f.storedDims[0] * ((j + f.ghost)
         + f.storedDims[1] *  (k + f.ghost));
}

// ---------------------------------------------------------------------------
// write  (dispatcher)
// ---------------------------------------------------------------------------

void Field::write(const std::string& path, FieldFormat fmt) const {
    switch (fmt) {
        case FieldFormat::BINARY: writeBinary(path); break;
        case FieldFormat::DAT:    writeDat(path);    break;
        case FieldFormat::VTS:    writeVts(path);    break;
    }
}

// ---------------------------------------------------------------------------
// writeBinary  (original binary format)
// ---------------------------------------------------------------------------

void Field::writeBinary(const std::string& path) const {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs)
        throw std::runtime_error("Field::write: cannot open file: " + path);

    // Text header
    ofs << "# PhiX Field\n";
    ofs << "name    " << name << "\n";
    ofs << "nx " << mesh.n[0]
        << "  ny " << mesh.n[1]
        << "  nz " << mesh.n[2] << "\n";
    ofs << "ghost   " << ghost << "\n";
    ofs << "---\n";

    // Binary data: physical cells only, row-major (x fastest)
    const std::size_t phySize = physicalSize(mesh);
    std::vector<double> buf;
    buf.reserve(phySize);

    for (int k = 0; k < mesh.n[2]; ++k)
        for (int j = 0; j < mesh.n[1]; ++j)
            for (int i = 0; i < mesh.n[0]; ++i)
                buf.push_back(curr[physIdx(*this, i, j, k)]);

    ofs.write(reinterpret_cast<const char*>(buf.data()),
              static_cast<std::streamsize>(phySize * sizeof(double)));
}

// ---------------------------------------------------------------------------
// writeDat  (ASCII text, x y z value, cell-centre coordinates)
// ---------------------------------------------------------------------------

void Field::writeDat(const std::string& path) const {
    std::ofstream ofs(path);
    if (!ofs)
        throw std::runtime_error("Field::write: cannot open file: " + path);

    ofs << "# PhiX Field - DAT\n";
    ofs << "# name: " << name << "\n";
    ofs << "# nx " << mesh.n[0]
        << "  ny " << mesh.n[1]
        << "  nz " << mesh.n[2] << "\n";
    ofs << "# x y z value\n";

    ofs << std::scientific << std::setprecision(12);
    for (int k = 0; k < mesh.n[2]; ++k)
        for (int j = 0; j < mesh.n[1]; ++j)
            for (int i = 0; i < mesh.n[0]; ++i) {
                double x = mesh.origin[0] + (i + 0.5) * mesh.d[0];
                double y = mesh.origin[1] + (j + 0.5) * mesh.d[1];
                double z = mesh.origin[2] + (k + 0.5) * mesh.d[2];
                ofs << x << "  " << y << "  " << z << "  "
                    << curr[physIdx(*this, i, j, k)] << "\n";
            }
}

// ---------------------------------------------------------------------------
// writeVts  (VTK XML StructuredGrid, CellData, ASCII)
//
// VTK ordering for structured grids: x fastest, z slowest — matches Field's
// own row-major layout, so no reordering is needed.
// ---------------------------------------------------------------------------

void Field::writeVts(const std::string& path) const {
    std::ofstream ofs(path);
    if (!ofs)
        throw std::runtime_error("Field::write: cannot open file: " + path);

    const int nx = mesh.n[0];
    const int ny = mesh.n[1];
    const int nz = mesh.n[2];

    // Corner (node) extents: a grid with nx*ny*nz cells has
    // (nx+1)*(ny+1)*(nz+1) corner points.
    ofs << "<?xml version=\"1.0\"?>\n";
    ofs << "<VTKFile type=\"StructuredGrid\" version=\"0.1\""
           " byte_order=\"LittleEndian\">\n";
    ofs << "  <StructuredGrid WholeExtent=\""
        << "0 " << nx << " 0 " << ny << " 0 " << nz << "\">\n";
    ofs << "    <Piece Extent=\""
        << "0 " << nx << " 0 " << ny << " 0 " << nz << "\">\n";

    // --- corner-node coordinates ---
    ofs << "      <Points>\n";
    ofs << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\""
           " format=\"ascii\">\n";
    ofs << std::scientific << std::setprecision(12);
    for (int k = 0; k <= nz; ++k)
        for (int j = 0; j <= ny; ++j)
            for (int i = 0; i <= nx; ++i) {
                double x = mesh.origin[0] + i * mesh.d[0];
                double y = mesh.origin[1] + j * mesh.d[1];
                double z = mesh.origin[2] + k * mesh.d[2];
                ofs << "          " << x << " " << y << " " << z << "\n";
            }
    ofs << "        </DataArray>\n";
    ofs << "      </Points>\n";

    // --- cell-centred field values ---
    ofs << "      <CellData Scalars=\"" << name << "\">\n";
    ofs << "        <DataArray type=\"Float64\" Name=\"" << name
        << "\" format=\"ascii\">\n";
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                ofs << "          "
                    << curr[physIdx(*this, i, j, k)] << "\n";
    ofs << "        </DataArray>\n";
    ofs << "      </CellData>\n";

    ofs << "    </Piece>\n";
    ofs << "  </StructuredGrid>\n";
    ofs << "</VTKFile>\n";
}

// ---------------------------------------------------------------------------
// readFromFile
// ---------------------------------------------------------------------------

Field Field::readFromFile(const Mesh& mesh,
                          const std::string& path,
                          int ghost) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
        throw std::runtime_error("Field::readFromFile: cannot open file: " + path);

    // Parse text header
    std::string fieldName;
    int fnx = 0, fny = 0, fnz = 0, fghost = 0;
    bool headerDone = false;

    std::string line;
    while (std::getline(ifs, line)) {
        // Strip trailing CR
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line == "---") { headerDone = true; break; }

        std::istringstream ss(line);
        std::string key;
        if (!(ss >> key) || key[0] == '#') continue;

        if (key == "name") {
            ss >> fieldName;
        } else if (key == "ghost") {
            ss >> fghost;
        } else {
            // scan for nx/ny/nz tokens anywhere on the line
            std::istringstream row(line);
            std::string tok;
            while (row >> tok) {
                if (tok == "nx")    row >> fnx;
                else if (tok == "ny") row >> fny;
                else if (tok == "nz") row >> fnz;
            }
        }
    }

    if (!headerDone)
        throw std::runtime_error("Field::readFromFile: missing '---' header terminator in " + path);

    // Validate dimensions
    if (fnx != mesh.n[0] || fny != mesh.n[1] || fnz != mesh.n[2]) {
        throw std::runtime_error(
            "Field::readFromFile: mesh dimensions in file ("
            + std::to_string(fnx) + "x" + std::to_string(fny) + "x" + std::to_string(fnz)
            + ") do not match provided mesh ("
            + std::to_string(mesh.n[0]) + "x" + std::to_string(mesh.n[1]) + "x" + std::to_string(mesh.n[2])
            + ")");
    }

    Field f(mesh, fieldName, ghost);

    // Read binary data into physical cells of curr
    const std::size_t phySize = physicalSize(mesh);
    std::vector<double> buf(phySize);
    ifs.read(reinterpret_cast<char*>(buf.data()),
             static_cast<std::streamsize>(phySize * sizeof(double)));

    if (!ifs)
        throw std::runtime_error("Field::readFromFile: unexpected end of binary data in " + path);

    std::size_t idx = 0;
    for (int k = 0; k < mesh.n[2]; ++k)
        for (int j = 0; j < mesh.n[1]; ++j)
            for (int i = 0; i < mesh.n[0]; ++i)
                f.curr[physIdx(f, i, j, k)] = buf[idx++];

    return f;
}

// ---------------------------------------------------------------------------
// print
// ---------------------------------------------------------------------------

void Field::print() const {
    std::cout << "=== Field ===\n";
    std::cout << "  name       : " << name << "\n";
    std::cout << "  ghost      : " << ghost << "\n";
    std::cout << "  storedDims : "
              << storedDims[0] << " x "
              << storedDims[1] << " x "
              << storedDims[2] << "  (" << storedSize << " cells)\n";
    std::cout << "  device     : " << (deviceAllocated() ? "allocated" : "not allocated") << "\n";

    // Min / max / mean of curr (physical cells only)
    double vmin = std::numeric_limits<double>::max();
    double vmax = std::numeric_limits<double>::lowest();
    double vsum = 0.0;
    std::size_t count = 0;

    for (int k = 0; k < mesh.n[2]; ++k)
        for (int j = 0; j < mesh.n[1]; ++j)
            for (int i = 0; i < mesh.n[0]; ++i) {
                double v = curr[physIdx(*this, i, j, k)];
                if (v < vmin) vmin = v;
                if (v > vmax) vmax = v;
                vsum += v;
                ++count;
            }

    if (count > 0) {
        std::cout << "  curr  min  : " << vmin << "\n";
        std::cout << "  curr  max  : " << vmax << "\n";
        std::cout << "  curr  mean : " << vsum / static_cast<double>(count) << "\n";
    }
}

} // namespace PhiX
