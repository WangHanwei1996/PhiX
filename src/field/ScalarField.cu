#include "field/ScalarField.h"
#include "IO/FieldIO.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <limits>
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

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

ScalarField::ScalarField(const Mesh& mesh_, const std::string& name_, int ghost_)
    : name(name_), mesh(mesh_), ghost(ghost_)
{
    if (ghost_ < 0)
        throw std::invalid_argument("ScalarField: ghost must be >= 0");

    for (int ax = 0; ax < 3; ++ax)
        storedDims[ax] = mesh.n[ax] + 2 * ghost;

    storedSize = computeStoredSize(storedDims);

    curr.assign(storedSize, 0.0);
    prev.assign(storedSize, 0.0);
}

// ---------------------------------------------------------------------------
// Move semantics
// ---------------------------------------------------------------------------

ScalarField::ScalarField(ScalarField&& other) noexcept
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

ScalarField& ScalarField::operator=(ScalarField&& other) noexcept {
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

ScalarField::~ScalarField() {
    freeDevice();
}

// ---------------------------------------------------------------------------
// Initialisation helpers
// ---------------------------------------------------------------------------

void ScalarField::fill(double value) {
    fillCurr(value);
    fillPrev(value);
}

void ScalarField::fillCurr(double value) {
    std::fill(curr.begin(), curr.end(), value);
}

void ScalarField::fillPrev(double value) {
    std::fill(prev.begin(), prev.end(), value);
}

// ---------------------------------------------------------------------------
// Time-stepping
// ---------------------------------------------------------------------------

void ScalarField::advanceTimeLevelCPU() {
    std::copy(curr.begin(), curr.end(), prev.begin());
}

void ScalarField::advanceTimeLevelGPU() {
    if (!deviceAllocated())
        throw std::runtime_error("ScalarField::advanceTimeLevelGPU: device not allocated");
    CUDA_CHECK(cudaMemcpy(d_prev, d_curr,
                          storedSize * sizeof(double),
                          cudaMemcpyDeviceToDevice));
}

// ---------------------------------------------------------------------------
// GPU management
// ---------------------------------------------------------------------------

void ScalarField::allocDevice() {
    if (deviceAllocated()) return;
    const std::size_t bytes = storedSize * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_curr, bytes));
    CUDA_CHECK(cudaMalloc(&d_prev, bytes));
    // Initialise GPU memory to zero
    CUDA_CHECK(cudaMemset(d_curr, 0, bytes));
    CUDA_CHECK(cudaMemset(d_prev, 0, bytes));
}

void ScalarField::freeDevice() {
    if (d_curr) { cudaFree(d_curr); d_curr = nullptr; }
    if (d_prev) { cudaFree(d_prev); d_prev = nullptr; }
}

void ScalarField::uploadCurrToDevice() const {
    if (!deviceAllocated())
        throw std::runtime_error("ScalarField::uploadCurrToDevice: device not allocated");
    CUDA_CHECK(cudaMemcpy(d_curr, curr.data(),
                          storedSize * sizeof(double),
                          cudaMemcpyHostToDevice));
}

void ScalarField::uploadPrevToDevice() const {
    if (!deviceAllocated())
        throw std::runtime_error("ScalarField::uploadPrevToDevice: device not allocated");
    CUDA_CHECK(cudaMemcpy(d_prev, prev.data(),
                          storedSize * sizeof(double),
                          cudaMemcpyHostToDevice));
}

void ScalarField::uploadAllToDevice() const {
    uploadCurrToDevice();
    uploadPrevToDevice();
}

void ScalarField::downloadCurrFromDevice() {
    if (!deviceAllocated())
        throw std::runtime_error("ScalarField::downloadCurrFromDevice: device not allocated");
    CUDA_CHECK(cudaMemcpy(curr.data(), d_curr,
                          storedSize * sizeof(double),
                          cudaMemcpyDeviceToHost));
}

void ScalarField::downloadPrevFromDevice() {
    if (!deviceAllocated())
        throw std::runtime_error("ScalarField::downloadPrevFromDevice: device not allocated");
    CUDA_CHECK(cudaMemcpy(prev.data(), d_prev,
                          storedSize * sizeof(double),
                          cudaMemcpyDeviceToHost));
}

void ScalarField::downloadAllFromDevice() {
    downloadCurrFromDevice();
    downloadPrevFromDevice();
}

// ---------------------------------------------------------------------------
// IO  (delegated to IO module)
// ---------------------------------------------------------------------------

void ScalarField::write(const std::string& path, FieldFormat fmt) const {
    IO::writeField(*this, path, fmt);
}

ScalarField ScalarField::readFromFile(const Mesh& mesh,
                                      const std::string& path,
                                      int ghost) {
    return IO::readScalarField(mesh, path, ghost);
}

// ---------------------------------------------------------------------------
// IO helper for print(): physical (i,j,k) -> stored flat index
// ---------------------------------------------------------------------------
static inline int physIdx(const ScalarField& f, int i, int j, int k) {
    return (i + f.ghost)
         + f.storedDims[0] * ((j + f.ghost)
         + f.storedDims[1] *  (k + f.ghost));
}

// ---------------------------------------------------------------------------
// print
// ---------------------------------------------------------------------------

void ScalarField::print() const {
    std::cout << "=== ScalarField ===\n";
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
