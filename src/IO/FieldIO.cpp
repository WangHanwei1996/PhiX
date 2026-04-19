#include "IO/FieldIO.h"
#include "field/ScalarField.h"
#include "field/VectorField.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace PhiX {
namespace IO {

// ---------------------------------------------------------------------------
// Anonymous-namespace helpers
// ---------------------------------------------------------------------------
namespace {

// Number of physical cells (no ghost)
std::size_t physicalSize(const Mesh& mesh) {
    return static_cast<std::size_t>(mesh.n[0])
         * mesh.n[1]
         * mesh.n[2];
}

// Convert physical (i,j,k) into the stored flat index (ghost-padded)
int physIdx(const ScalarField& f, int i, int j, int k) {
    return (i + f.ghost)
         + f.storedDims[0] * ((j + f.ghost)
         + f.storedDims[1] *  (k + f.ghost));
}

// ---------------------------------------------------------------------------
// Common header parsing helpers
// ---------------------------------------------------------------------------

struct ScalarFileHeader {
    std::string fieldName;
    int nx = 0, ny = 0, nz = 0, ghost = 0;
};

struct VectorFileHeader {
    std::string fieldName;
    int nx = 0, ny = 0, nz = 0, ghost = 0, nComponents = 0;
};

ScalarFileHeader parseScalarHeader(std::ifstream& ifs, const std::string& path) {
    ScalarFileHeader h;
    bool headerDone = false;
    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line == "---") { headerDone = true; break; }

        std::istringstream ss(line);
        std::string key;
        if (!(ss >> key) || key[0] == '#') continue;

        if      (key == "name")  { ss >> h.fieldName; }
        else if (key == "ghost") { ss >> h.ghost; }
        else {
            std::istringstream row(line);
            std::string tok;
            while (row >> tok) {
                if (tok == "nx")      row >> h.nx;
                else if (tok == "ny") row >> h.ny;
                else if (tok == "nz") row >> h.nz;
            }
        }
    }
    if (!headerDone)
        throw std::runtime_error("IO: missing '---' header terminator in " + path);
    return h;
}

VectorFileHeader parseVectorHeader(std::ifstream& ifs, const std::string& path) {
    VectorFileHeader h;
    bool headerDone = false;
    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line == "---") { headerDone = true; break; }

        std::istringstream ss(line);
        std::string key;
        if (!(ss >> key) || key[0] == '#') continue;

        if      (key == "name")        { ss >> h.fieldName; }
        else if (key == "nComponents") { ss >> h.nComponents; }
        else if (key == "ghost")       { ss >> h.ghost; }
        else {
            std::istringstream row(line);
            std::string tok;
            while (row >> tok) {
                if (tok == "nx")      row >> h.nx;
                else if (tok == "ny") row >> h.ny;
                else if (tok == "nz") row >> h.nz;
            }
        }
    }
    if (!headerDone)
        throw std::runtime_error("IO: missing '---' header terminator in " + path);
    return h;
}

// ---------------------------------------------------------------------------
// DAT header parsing helpers
// ---------------------------------------------------------------------------

// ScalarField DAT header: lines starting with '#', stops (and seeks back)
// when a non-'#' line is encountered.
// Expected comment lines (order flexible):
//   # PhiX ScalarField - DAT
//   # name: <name>
//   # nx N  ny N  nz N
//   # x y z value
struct ScalarDatHeader {
    std::string fieldName;
    int nx = 0, ny = 0, nz = 0;
};

ScalarDatHeader parseScalarDatHeader(std::ifstream& ifs, const std::string& path) {
    ScalarDatHeader h;
    std::string line;
    while (true) {
        std::streampos pos = ifs.tellg();
        if (!std::getline(ifs, line)) break;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] != '#') {
            ifs.seekg(pos);   // put back the first data line
            break;
        }
        // Strip leading '#' and optional space
        std::string rest = line.substr(1);
        std::istringstream ss(rest);
        std::string key;
        if (!(ss >> key)) continue;
        if (key == "name:") {
            ss >> h.fieldName;
        } else {
            // scan for nx/ny/nz tokens anywhere on the line
            std::istringstream row(rest);
            std::string tok;
            while (row >> tok) {
                if      (tok == "nx") row >> h.nx;
                else if (tok == "ny") row >> h.ny;
                else if (tok == "nz") row >> h.nz;
            }
        }
    }
    return h;
}

// VectorField DAT header: same pattern.
// Expected comment lines:
//   # PhiX VectorField - DAT
//   # name: <name>  nComponents: N
//   # nx N  ny N  nz N
//   # x y z v0 v1 ...
struct VectorDatHeader {
    std::string fieldName;
    int nx = 0, ny = 0, nz = 0, nComponents = 0;
};

VectorDatHeader parseVectorDatHeader(std::ifstream& ifs, const std::string& path) {
    VectorDatHeader h;
    std::string line;
    while (true) {
        std::streampos pos = ifs.tellg();
        if (!std::getline(ifs, line)) break;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] != '#') {
            ifs.seekg(pos);
            break;
        }
        std::string rest = line.substr(1);
        std::istringstream row(rest);
        std::string tok;
        while (row >> tok) {
            if      (tok == "name:")        row >> h.fieldName;
            else if (tok == "nComponents:") row >> h.nComponents;
            else if (tok == "nx")           row >> h.nx;
            else if (tok == "ny")           row >> h.ny;
            else if (tok == "nz")           row >> h.nz;
        }
    }
    return h;
}

// Detect format from first line: returns true if DAT, false if BINARY.
// Seeks back to start of file afterwards.
bool isDatFormat(std::ifstream& ifs) {
    std::string line;
    std::getline(ifs, line);
    ifs.seekg(0);
    return line.find("- DAT") != std::string::npos;
}

// ===================================================================
// ScalarField write helpers
// ===================================================================

void writeScalarBinary(const ScalarField& f, const std::string& path) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs)
        throw std::runtime_error("IO::writeField: cannot open file: " + path);

    const Mesh& mesh = f.mesh;

    // Text header
    ofs << "# PhiX ScalarField\n";
    ofs << "name    " << f.name << "\n";
    ofs << "nx " << mesh.n[0]
        << "  ny " << mesh.n[1]
        << "  nz " << mesh.n[2] << "\n";
    ofs << "ghost   " << f.ghost << "\n";
    ofs << "---\n";

    // Binary data: physical cells only, row-major (x fastest)
    const std::size_t phySize = physicalSize(mesh);
    std::vector<double> buf;
    buf.reserve(phySize);

    for (int k = 0; k < mesh.n[2]; ++k)
        for (int j = 0; j < mesh.n[1]; ++j)
            for (int i = 0; i < mesh.n[0]; ++i)
                buf.push_back(f.curr[physIdx(f, i, j, k)]);

    ofs.write(reinterpret_cast<const char*>(buf.data()),
              static_cast<std::streamsize>(phySize * sizeof(double)));
}

void writeScalarDat(const ScalarField& f, const std::string& path) {
    std::ofstream ofs(path);
    if (!ofs)
        throw std::runtime_error("IO::writeField: cannot open file: " + path);

    const Mesh& mesh = f.mesh;

    ofs << "# PhiX ScalarField - DAT\n";
    ofs << "# name: " << f.name << "\n";
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
                    << f.curr[physIdx(f, i, j, k)] << "\n";
            }
}

void writeScalarVts(const ScalarField& f, const std::string& path) {
    std::ofstream ofs(path);
    if (!ofs)
        throw std::runtime_error("IO::writeField: cannot open file: " + path);

    const Mesh& mesh = f.mesh;
    const int nx = mesh.n[0];
    const int ny = mesh.n[1];
    const int nz = mesh.n[2];

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
    ofs << "      <CellData Scalars=\"" << f.name << "\">\n";
    ofs << "        <DataArray type=\"Float64\" Name=\"" << f.name
        << "\" format=\"ascii\">\n";
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                ofs << "          "
                    << f.curr[physIdx(f, i, j, k)] << "\n";
    ofs << "        </DataArray>\n";
    ofs << "      </CellData>\n";

    ofs << "    </Piece>\n";
    ofs << "  </StructuredGrid>\n";
    ofs << "</VTKFile>\n";
}

// ===================================================================
// VectorField write helpers
// ===================================================================

void writeVectorBinary(const VectorField& vf, const std::string& path) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs)
        throw std::runtime_error("IO::writeField: cannot open file: " + path);

    const Mesh& mesh = vf.mesh;
    const int N  = vf.nComponents();
    const int nx = mesh.n[0], ny = mesh.n[1], nz = mesh.n[2];

    ofs << "# PhiX VectorField\n";
    ofs << "name         " << vf.name << "\n";
    ofs << "nComponents  " << N    << "\n";
    ofs << "nx " << nx << "  ny " << ny << "  nz " << nz << "\n";
    ofs << "ghost        " << vf.ghost << "\n";
    ofs << "---\n";

    const std::size_t phySize = static_cast<std::size_t>(nx) * ny * nz;
    std::vector<double> buf(phySize);

    for (int c = 0; c < N; ++c) {
        const ScalarField& sf = vf[c];
        std::size_t idx = 0;
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i)
                    buf[idx++] = sf.curr[physIdx(sf, i, j, k)];
        ofs.write(reinterpret_cast<const char*>(buf.data()),
                  static_cast<std::streamsize>(phySize * sizeof(double)));
    }
}

void writeVectorDat(const VectorField& vf, const std::string& path) {
    std::ofstream ofs(path);
    if (!ofs)
        throw std::runtime_error("IO::writeField: cannot open file: " + path);

    const Mesh& mesh = vf.mesh;
    const int N  = vf.nComponents();
    const int nx = mesh.n[0], ny = mesh.n[1], nz = mesh.n[2];

    ofs << "# PhiX VectorField - DAT\n";
    ofs << "# name: " << vf.name << "  nComponents: " << N << "\n";
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
                    const ScalarField& sf = vf[c];
                    ofs << "  " << sf.curr[physIdx(sf, i, j, k)];
                }
                ofs << "\n";
            }
}

void writeVectorVts(const VectorField& vf, const std::string& path) {
    std::ofstream ofs(path);
    if (!ofs)
        throw std::runtime_error("IO::writeField: cannot open file: " + path);

    const Mesh& mesh = vf.mesh;
    const int N  = vf.nComponents();
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
    ofs << "      <CellData Vectors=\"" << vf.name << "\">\n";
    ofs << "        <DataArray type=\"Float64\" Name=\"" << vf.name
        << "\" NumberOfComponents=\"" << vtk_nc << "\" format=\"ascii\">\n";

    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                ofs << "         ";
                for (int c = 0; c < N; ++c) {
                    const ScalarField& sf = vf[c];
                    ofs << " " << sf.curr[physIdx(sf, i, j, k)];
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

} // anonymous namespace

// ===========================================================================
// Public API — ScalarField
// ===========================================================================

void writeField(const ScalarField& f,
                const std::string& path,
                FieldFormat fmt) {
    switch (fmt) {
        case FieldFormat::BINARY: writeScalarBinary(f, path); break;
        case FieldFormat::DAT:    writeScalarDat(f, path);    break;
        case FieldFormat::VTS:    writeScalarVts(f, path);    break;
    }
}

ScalarField readScalarField(const Mesh& mesh,
                             const std::string& path,
                             int ghost) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
        throw std::runtime_error("IO::readScalarField: cannot open file: " + path);

    if (isDatFormat(ifs)) {
        // ---- DAT (text) format ----
        const auto h = parseScalarDatHeader(ifs, path);

        if (h.nx != mesh.n[0] || h.ny != mesh.n[1] || h.nz != mesh.n[2])
            throw std::runtime_error(
                "IO::readScalarField: mesh dimensions in file ("
                + std::to_string(h.nx) + "x" + std::to_string(h.ny) + "x" + std::to_string(h.nz)
                + ") do not match provided mesh ("
                + std::to_string(mesh.n[0]) + "x" + std::to_string(mesh.n[1]) + "x" + std::to_string(mesh.n[2])
                + ")");

        ScalarField f(mesh, h.fieldName.empty() ? "field" : h.fieldName, ghost);
        std::string line;
        for (int k = 0; k < mesh.n[2]; ++k)
        for (int j = 0; j < mesh.n[1]; ++j)
        for (int i = 0; i < mesh.n[0]; ++i) {
            if (!std::getline(ifs, line))
                throw std::runtime_error("IO::readScalarField: unexpected end of DAT data in " + path);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::istringstream ss(line);
            double x, y, z, val;
            if (!(ss >> x >> y >> z >> val))
                throw std::runtime_error("IO::readScalarField: malformed DAT line in " + path);
            f.curr[physIdx(f, i, j, k)] = val;
        }
        return f;
    } else {
        // ---- BINARY format ----
        const auto h = parseScalarHeader(ifs, path);

        if (h.nx != mesh.n[0] || h.ny != mesh.n[1] || h.nz != mesh.n[2])
            throw std::runtime_error(
                "IO::readScalarField: mesh dimensions in file ("
                + std::to_string(h.nx) + "x" + std::to_string(h.ny) + "x" + std::to_string(h.nz)
                + ") do not match provided mesh ("
                + std::to_string(mesh.n[0]) + "x" + std::to_string(mesh.n[1]) + "x" + std::to_string(mesh.n[2])
                + ")");

        ScalarField f(mesh, h.fieldName, ghost);
        const std::size_t phySize = physicalSize(mesh);
        std::vector<double> buf(phySize);
        ifs.read(reinterpret_cast<char*>(buf.data()),
                 static_cast<std::streamsize>(phySize * sizeof(double)));
        if (!ifs)
            throw std::runtime_error("IO::readScalarField: unexpected end of binary data in " + path);

        std::size_t idx = 0;
        for (int k = 0; k < mesh.n[2]; ++k)
            for (int j = 0; j < mesh.n[1]; ++j)
                for (int i = 0; i < mesh.n[0]; ++i)
                    f.curr[physIdx(f, i, j, k)] = buf[idx++];
        return f;
    }
}

void readField(ScalarField& f, const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
        throw std::runtime_error("IO::readField: cannot open file: " + path);

    if (isDatFormat(ifs)) {
        // ---- DAT (text) format ----
        const auto h = parseScalarDatHeader(ifs, path);

        if (h.nx != f.mesh.n[0] || h.ny != f.mesh.n[1] || h.nz != f.mesh.n[2]) {
            throw std::runtime_error(
                "IO::readField: file dimensions ("
                + std::to_string(h.nx) + "x" + std::to_string(h.ny) + "x" + std::to_string(h.nz)
                + ") do not match field \"" + f.name + "\" mesh ("
                + std::to_string(f.mesh.n[0]) + "x" + std::to_string(f.mesh.n[1]) + "x" + std::to_string(f.mesh.n[2])
                + ")");
        }

        std::string line;
        for (int k = 0; k < f.mesh.n[2]; ++k)
        for (int j = 0; j < f.mesh.n[1]; ++j)
        for (int i = 0; i < f.mesh.n[0]; ++i) {
            if (!std::getline(ifs, line))
                throw std::runtime_error("IO::readField: unexpected end of DAT data in " + path);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::istringstream ss(line);
            double x, y, z, val;
            if (!(ss >> x >> y >> z >> val))
                throw std::runtime_error("IO::readField: malformed DAT line in " + path);
            f.curr[physIdx(f, i, j, k)] = val;
        }
    } else {
        // ---- BINARY format ----
        const auto h = parseScalarHeader(ifs, path);

        if (h.nx != f.mesh.n[0] || h.ny != f.mesh.n[1] || h.nz != f.mesh.n[2]) {
            throw std::runtime_error(
                "IO::readField: file dimensions ("
                + std::to_string(h.nx) + "x" + std::to_string(h.ny) + "x" + std::to_string(h.nz)
                + ") do not match field \"" + f.name + "\" mesh ("
                + std::to_string(f.mesh.n[0]) + "x" + std::to_string(f.mesh.n[1]) + "x" + std::to_string(f.mesh.n[2])
                + ")");
        }

        const std::size_t phySize = physicalSize(f.mesh);
        std::vector<double> buf(phySize);
        ifs.read(reinterpret_cast<char*>(buf.data()),
                 static_cast<std::streamsize>(phySize * sizeof(double)));
        if (!ifs)
            throw std::runtime_error("IO::readField: unexpected end of binary data in " + path);

        std::size_t idx = 0;
        for (int k = 0; k < f.mesh.n[2]; ++k)
            for (int j = 0; j < f.mesh.n[1]; ++j)
                for (int i = 0; i < f.mesh.n[0]; ++i)
                    f.curr[physIdx(f, i, j, k)] = buf[idx++];
    }
}

// ===========================================================================
// Public API — VectorField
// ===========================================================================

void writeField(const VectorField& vf,
                const std::string& path,
                FieldFormat fmt) {
    switch (fmt) {
        case FieldFormat::BINARY: writeVectorBinary(vf, path); break;
        case FieldFormat::DAT:    writeVectorDat(vf, path);    break;
        case FieldFormat::VTS:    writeVectorVts(vf, path);    break;
    }
}

VectorField readVectorField(const Mesh& mesh,
                              const std::string& path,
                              int ghost) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
        throw std::runtime_error("IO::readVectorField: cannot open file: " + path);

    if (isDatFormat(ifs)) {
        // ---- DAT (text) format ----
        const auto h = parseVectorDatHeader(ifs, path);

        if (h.nComponents <= 0)
            throw std::runtime_error("IO::readVectorField: invalid nComponents in " + path);
        if (h.nx != mesh.n[0] || h.ny != mesh.n[1] || h.nz != mesh.n[2])
            throw std::runtime_error("IO::readVectorField: mesh mismatch in " + path);

        const std::string fname = h.fieldName.empty() ? "field" : h.fieldName;
        VectorField vf(mesh, fname, h.nComponents, ghost);
        const int N = h.nComponents;
        std::string line;
        for (int k = 0; k < h.nz; ++k)
        for (int j = 0; j < h.ny; ++j)
        for (int i = 0; i < h.nx; ++i) {
            if (!std::getline(ifs, line))
                throw std::runtime_error("IO::readVectorField: unexpected end of DAT data in " + path);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::istringstream ss(line);
            double x, y, z;
            if (!(ss >> x >> y >> z))
                throw std::runtime_error("IO::readVectorField: malformed DAT line in " + path);
            for (int c = 0; c < N; ++c) {
                double val;
                if (!(ss >> val))
                    throw std::runtime_error(
                        "IO::readVectorField: missing component " + std::to_string(c)
                        + " in DAT line in " + path);
                vf[c].curr[physIdx(vf[c], i, j, k)] = val;
            }
        }
        return vf;
    } else {
        // ---- BINARY format ----
        const auto h = parseVectorHeader(ifs, path);

        if (h.nComponents <= 0)
            throw std::runtime_error("IO::readVectorField: invalid nComponents in " + path);
        if (h.nx != mesh.n[0] || h.ny != mesh.n[1] || h.nz != mesh.n[2])
            throw std::runtime_error("IO::readVectorField: mesh mismatch in " + path);

        VectorField vf(mesh, h.fieldName, h.nComponents, ghost);

        const std::size_t phySize = static_cast<std::size_t>(h.nx) * h.ny * h.nz;
        std::vector<double> buf(phySize);

        for (int c = 0; c < h.nComponents; ++c) {
            ifs.read(reinterpret_cast<char*>(buf.data()),
                     static_cast<std::streamsize>(phySize * sizeof(double)));
            if (!ifs)
                throw std::runtime_error(
                    "IO::readVectorField: unexpected end of data for component "
                    + std::to_string(c) + " in " + path);

            ScalarField& sf = vf[c];
            std::size_t idx = 0;
            for (int k = 0; k < h.nz; ++k)
                for (int j = 0; j < h.ny; ++j)
                    for (int i = 0; i < h.nx; ++i)
                        sf.curr[physIdx(sf, i, j, k)] = buf[idx++];
        }
        return vf;
    }
}

void readField(VectorField& vf, const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
        throw std::runtime_error("IO::readField: cannot open file: " + path);

    if (isDatFormat(ifs)) {
        // ---- DAT (text) format ----
        const auto h = parseVectorDatHeader(ifs, path);

        if (h.nComponents != vf.nComponents())
            throw std::runtime_error(
                "IO::readField: file nComponents (" + std::to_string(h.nComponents)
                + ") does not match field \"" + vf.name + "\" nComponents ("
                + std::to_string(vf.nComponents()) + ")");

        if (h.nx != vf.mesh.n[0] || h.ny != vf.mesh.n[1] || h.nz != vf.mesh.n[2])
            throw std::runtime_error(
                "IO::readField: mesh mismatch for field \"" + vf.name + "\" in " + path);

        const int N = vf.nComponents();
        std::string line;
        for (int k = 0; k < h.nz; ++k)
        for (int j = 0; j < h.ny; ++j)
        for (int i = 0; i < h.nx; ++i) {
            if (!std::getline(ifs, line))
                throw std::runtime_error("IO::readField: unexpected end of DAT data in " + path);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::istringstream ss(line);
            double x, y, z;
            if (!(ss >> x >> y >> z))
                throw std::runtime_error("IO::readField: malformed DAT line in " + path);
            for (int c = 0; c < N; ++c) {
                double val;
                if (!(ss >> val))
                    throw std::runtime_error(
                        "IO::readField: missing component " + std::to_string(c)
                        + " in DAT line in " + path);
                vf[c].curr[physIdx(vf[c], i, j, k)] = val;
            }
        }
    } else {
        // ---- BINARY format ----
        const auto h = parseVectorHeader(ifs, path);

        if (h.nComponents != vf.nComponents())
            throw std::runtime_error(
                "IO::readField: file nComponents (" + std::to_string(h.nComponents)
                + ") does not match field \"" + vf.name + "\" nComponents ("
                + std::to_string(vf.nComponents()) + ")");

        if (h.nx != vf.mesh.n[0] || h.ny != vf.mesh.n[1] || h.nz != vf.mesh.n[2])
            throw std::runtime_error("IO::readField: mesh mismatch for field \"" + vf.name + "\" in " + path);

        const std::size_t phySize = static_cast<std::size_t>(h.nx) * h.ny * h.nz;
        std::vector<double> buf(phySize);

        for (int c = 0; c < h.nComponents; ++c) {
            ifs.read(reinterpret_cast<char*>(buf.data()),
                     static_cast<std::streamsize>(phySize * sizeof(double)));
            if (!ifs)
                throw std::runtime_error(
                    "IO::readField: unexpected end of data for component "
                    + std::to_string(c) + " in " + path);

            ScalarField& sf = vf[c];
            std::size_t idx = 0;
            for (int k = 0; k < h.nz; ++k)
                for (int j = 0; j < h.ny; ++j)
                    for (int i = 0; i < h.nx; ++i)
                        sf.curr[physIdx(sf, i, j, k)] = buf[idx++];
        }
    }
}

} // namespace IO
} // namespace PhiX

// ---------------------------------------------------------------------------
// Restart / initialization helpers
// ---------------------------------------------------------------------------
namespace PhiX {
namespace IO {

int resolveStartStep(const std::string& startFrom,
                     const std::string& refFieldName)
{
    if (startFrom == "initial_field")
        return 0;

    const int target = std::stoi(startFrom);
    const std::string prefix = refFieldName + "_";

    std::vector<int> steps;
    namespace fs = std::filesystem;
    if (fs::is_directory("output")) {
        for (const auto& entry : fs::directory_iterator("output")) {
            const fs::path p = entry.path();
            if (p.extension() != ".field") continue;
            const std::string stem = p.stem().string();
            if (stem.rfind(prefix, 0) != 0) continue;
            try {
                steps.push_back(std::stoi(stem.substr(prefix.size())));
            } catch (...) {}
        }
    }

    if (steps.empty())
        throw std::runtime_error(
            "IO::resolveStartStep: start_from=\"" + startFrom +
            "\" but no " + prefix + "*.field files found in output/");

    return *std::min_element(steps.begin(), steps.end(),
        [target](int a, int b) {
            return std::abs(a - target) < std::abs(b - target);
        });
}

void initField(ScalarField& f, int startStep)
{
    std::string path;
    if (startStep == 0) {
        path = "settings/initial_field/" + f.name + ".field";
        std::cout << "  cold start: loading " << path << "\n";
    } else {
        path = "output/" + f.name + "_" + std::to_string(startStep) + ".field";
        std::cout << "  warm start: loading " << path << "\n";
    }
    readField(f, path);
}

} // namespace IO
} // namespace PhiX
