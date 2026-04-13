#include "field/VectorField.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace PhiX {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string makeComponentName(const std::string& base, int c, int nComp) {
    if (nComp == 3) {
        switch (c) {
            case 0: return base + "_x";
            case 1: return base + "_y";
            case 2: return base + "_z";
        }
    }
    return base + "_" + std::to_string(c);
}

static std::size_t physIdxVF(const ScalarField& f, int i, int j, int k) {
    return static_cast<std::size_t>((i + f.ghost)
         + f.storedDims[0] * ((j + f.ghost)
         + f.storedDims[1] *  (k + f.ghost)));
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

std::string VectorField::componentName(const std::string& baseName,
                                        int c, int nComp) {
    return makeComponentName(baseName, c, nComp);
}

VectorField::VectorField(const Mesh& mesh_,
                          const std::string& name_,
                          int nComponents,
                          int ghost_)
    : mesh(mesh_)
    , ghost(ghost_)
    , name(name_)
{
    if (nComponents <= 0)
        throw std::invalid_argument("VectorField: nComponents must be > 0");

    components_.reserve(nComponents);
    for (int c = 0; c < nComponents; ++c) {
        components_.emplace_back(mesh_, makeComponentName(name_, c, nComponents), ghost_);
    }
}

// ---------------------------------------------------------------------------
// Move semantics
// ---------------------------------------------------------------------------

VectorField::VectorField(VectorField&& other) noexcept
    : mesh(other.mesh)
    , ghost(other.ghost)
    , name(std::move(other.name))
    , components_(std::move(other.components_))
{}

VectorField& VectorField::operator=(VectorField&& other) noexcept {
    if (this == &other) return *this;
    // mesh is const ref — cannot rebind; caller must ensure same mesh lifetime
    ghost      = other.ghost;
    name       = std::move(other.name);
    components_ = std::move(other.components_);
    return *this;
}

// ---------------------------------------------------------------------------
// Component access
// ---------------------------------------------------------------------------

ScalarField& VectorField::operator[](int c) {
    if (c < 0 || c >= static_cast<int>(components_.size()))
        throw std::out_of_range("VectorField: component index out of range");
    return components_[c];
}

const ScalarField& VectorField::operator[](int c) const {
    if (c < 0 || c >= static_cast<int>(components_.size()))
        throw std::out_of_range("VectorField: component index out of range");
    return components_[c];
}

// ---------------------------------------------------------------------------
// Initialisation helpers
// ---------------------------------------------------------------------------

void VectorField::fill(double value) {
    for (auto& comp : components_) comp.fill(value);
}

void VectorField::fillCurr(double value) {
    for (auto& comp : components_) comp.fillCurr(value);
}

void VectorField::fillPrev(double value) {
    for (auto& comp : components_) comp.fillPrev(value);
}

// ---------------------------------------------------------------------------
// Time-stepping
// ---------------------------------------------------------------------------

void VectorField::advanceTimeLevelCPU() {
    for (auto& comp : components_) comp.advanceTimeLevelCPU();
}

void VectorField::advanceTimeLevelGPU() {
    for (auto& comp : components_) comp.advanceTimeLevelGPU();
}

// ---------------------------------------------------------------------------
// GPU management
// ---------------------------------------------------------------------------

bool VectorField::deviceAllocated() const {
    for (const auto& comp : components_)
        if (!comp.deviceAllocated()) return false;
    return !components_.empty();
}

void VectorField::allocDevice() {
    for (auto& comp : components_) comp.allocDevice();
}

void VectorField::freeDevice() {
    for (auto& comp : components_) comp.freeDevice();
}

void VectorField::uploadCurrToDevice() const {
    for (const auto& comp : components_) comp.uploadCurrToDevice();
}

void VectorField::uploadPrevToDevice() const {
    for (const auto& comp : components_) comp.uploadPrevToDevice();
}

void VectorField::uploadAllToDevice() const {
    for (const auto& comp : components_) comp.uploadAllToDevice();
}

void VectorField::downloadCurrFromDevice() {
    for (auto& comp : components_) comp.downloadCurrFromDevice();
}

void VectorField::downloadPrevFromDevice() {
    for (auto& comp : components_) comp.downloadPrevFromDevice();
}

void VectorField::downloadAllFromDevice() {
    for (auto& comp : components_) comp.downloadAllFromDevice();
}

// ---------------------------------------------------------------------------
// write (dispatcher)
// ---------------------------------------------------------------------------

void VectorField::write(const std::string& path, FieldFormat fmt) const {
    switch (fmt) {
        case FieldFormat::BINARY: writeBinary(path); break;
        case FieldFormat::DAT:    writeDat(path);    break;
        case FieldFormat::VTS:    writeVts(path);    break;
    }
}

// ---------------------------------------------------------------------------
// writeBinary
// ---------------------------------------------------------------------------

void VectorField::writeBinary(const std::string& path) const {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs)
        throw std::runtime_error("VectorField::write: cannot open file: " + path);

    const int N  = nComponents();
    const int nx = mesh.n[0], ny = mesh.n[1], nz = mesh.n[2];

    ofs << "# PhiX VectorField\n";
    ofs << "name         " << name << "\n";
    ofs << "nComponents  " << N    << "\n";
    ofs << "nx " << nx << "  ny " << ny << "  nz " << nz << "\n";
    ofs << "ghost        " << ghost << "\n";
    ofs << "---\n";

    const std::size_t phySize = static_cast<std::size_t>(nx) * ny * nz;
    std::vector<double> buf(phySize);

    for (int c = 0; c < N; ++c) {
        const ScalarField& sf = components_[c];
        std::size_t idx = 0;
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i)
                    buf[idx++] = sf.curr[physIdxVF(sf, i, j, k)];
        ofs.write(reinterpret_cast<const char*>(buf.data()),
                  static_cast<std::streamsize>(phySize * sizeof(double)));
    }
}

// ---------------------------------------------------------------------------
// writeDat
// ---------------------------------------------------------------------------

void VectorField::writeDat(const std::string& path) const {
    std::ofstream ofs(path);
    if (!ofs)
        throw std::runtime_error("VectorField::write: cannot open file: " + path);

    const int N  = nComponents();
    const int nx = mesh.n[0], ny = mesh.n[1], nz = mesh.n[2];

    ofs << "# PhiX VectorField - DAT\n";
    ofs << "# name: " << name << "  nComponents: " << N << "\n";
    ofs << "# nx " << nx << "  ny " << ny << "  nz " << nz << "\n";
    ofs << "# x y z";
    for (int c = 0; c < N; ++c) ofs << " v" << c;
    ofs << "\n";

    ofs << std::scientific << std::setprecision(12);

    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                double x = mesh.origin[0] + (i + 0.5) * mesh.d[0];
                double y = mesh.origin[1] + (j + 0.5) * mesh.d[1];
                double z = mesh.origin[2] + (k + 0.5) * mesh.d[2];
                ofs << x << "  " << y << "  " << z;
                for (int c = 0; c < N; ++c) {
                    const ScalarField& sf = components_[c];
                    ofs << "  " << sf.curr[physIdxVF(sf, i, j, k)];
                }
                ofs << "\n";
            }
}

// ---------------------------------------------------------------------------
// writeVts
// ---------------------------------------------------------------------------

void VectorField::writeVts(const std::string& path) const {
    std::ofstream ofs(path);
    if (!ofs)
        throw std::runtime_error("VectorField::write: cannot open file: " + path);

    const int N  = nComponents();
    const int nx = mesh.n[0], ny = mesh.n[1], nz = mesh.n[2];

    // VTK requires vectors to have exactly 3 components; pad with zeros if N<3.
    const int vtk_nc = (N <= 3) ? 3 : N;

    ofs << "<?xml version=\"1.0\"?>\n";
    ofs << "<VTKFile type=\"StructuredGrid\" version=\"0.1\""
           " byte_order=\"LittleEndian\">\n";
    ofs << "  <StructuredGrid WholeExtent=\""
        << "0 " << nx << " 0 " << ny << " 0 " << nz << "\">\n";
    ofs << "    <Piece Extent=\""
        << "0 " << nx << " 0 " << ny << " 0 " << nz << "\">\n";

    // Corner-node coordinates
    ofs << "      <Points>\n";
    ofs << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\""
           " format=\"ascii\">\n";
    ofs << std::scientific << std::setprecision(12);
    for (int k = 0; k <= nz; ++k)
        for (int j = 0; j <= ny; ++j)
            for (int i = 0; i <= nx; ++i)
                ofs << "          "
                    << mesh.origin[0] + i * mesh.d[0] << " "
                    << mesh.origin[1] + j * mesh.d[1] << " "
                    << mesh.origin[2] + k * mesh.d[2] << "\n";
    ofs << "        </DataArray>\n";
    ofs << "      </Points>\n";

    // Vector cell data (interleaved: v0 v1 v2 per cell)
    ofs << "      <CellData Vectors=\"" << name << "\">\n";
    ofs << "        <DataArray type=\"Float64\" Name=\"" << name
        << "\" NumberOfComponents=\"" << vtk_nc << "\" format=\"ascii\">\n";

    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                ofs << "         ";
                for (int c = 0; c < N; ++c) {
                    const ScalarField& sf = components_[c];
                    ofs << " " << sf.curr[physIdxVF(sf, i, j, k)];
                }
                // Pad to vtk_nc if needed
                for (int c = N; c < vtk_nc; ++c) ofs << " 0.0";
                ofs << "\n";
            }

    ofs << "        </DataArray>\n";
    ofs << "      </CellData>\n";
    ofs << "    </Piece>\n";
    ofs << "  </StructuredGrid>\n";
    ofs << "</VTKFile>\n";
}

// ---------------------------------------------------------------------------
// readFromFile
// ---------------------------------------------------------------------------

VectorField VectorField::readFromFile(const Mesh& mesh,
                                       const std::string& path,
                                       int ghost) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
        throw std::runtime_error("VectorField::readFromFile: cannot open file: " + path);

    std::string fieldName;
    int fnComponents = 0, fnx = 0, fny = 0, fnz = 0, fghost = 0;
    bool headerDone = false;

    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line == "---") { headerDone = true; break; }

        std::istringstream ss(line);
        std::string key;
        if (!(ss >> key) || key[0] == '#') continue;

        if (key == "name")        { ss >> fieldName; }
        else if (key == "nComponents") { ss >> fnComponents; }
        else if (key == "ghost")  { ss >> fghost; }
        else {
            std::istringstream row(line);
            std::string tok;
            while (row >> tok) {
                if (tok == "nx")      row >> fnx;
                else if (tok == "ny") row >> fny;
                else if (tok == "nz") row >> fnz;
            }
        }
    }

    if (!headerDone)
        throw std::runtime_error("VectorField::readFromFile: missing '---' header in " + path);
    if (fnComponents <= 0)
        throw std::runtime_error("VectorField::readFromFile: invalid nComponents in " + path);

    if (fnx != mesh.n[0] || fny != mesh.n[1] || fnz != mesh.n[2])
        throw std::runtime_error(
            "VectorField::readFromFile: mesh mismatch in " + path);

    VectorField vf(mesh, fieldName, fnComponents, ghost);

    const std::size_t phySize = static_cast<std::size_t>(fnx) * fny * fnz;
    std::vector<double> buf(phySize);

    for (int c = 0; c < fnComponents; ++c) {
        ifs.read(reinterpret_cast<char*>(buf.data()),
                 static_cast<std::streamsize>(phySize * sizeof(double)));
        if (!ifs)
            throw std::runtime_error(
                "VectorField::readFromFile: unexpected end of data for component "
                + std::to_string(c) + " in " + path);

        ScalarField& sf = vf[c];
        std::size_t idx = 0;
        for (int k = 0; k < fnz; ++k)
            for (int j = 0; j < fny; ++j)
                for (int i = 0; i < fnx; ++i)
                    sf.curr[physIdxVF(sf, i, j, k)] = buf[idx++];
    }

    return vf;
}

// ---------------------------------------------------------------------------
// print
// ---------------------------------------------------------------------------

void VectorField::print() const {
    std::cout << "=== VectorField ===\n";
    std::cout << "  name        : " << name << "\n";
    std::cout << "  nComponents : " << nComponents() << "\n";
    std::cout << "  ghost       : " << ghost << "\n";
    for (const auto& comp : components_) {
        comp.print();
    }
}

} // namespace PhiX
