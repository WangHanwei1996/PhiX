#include "material/FreeEnergyTable.h"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace PhiX {
namespace Material {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

FreeEnergyTable::FreeEnergyTable(double c_min, double c_max, int nc,
                                 double T_min, double T_max, int nT,
                                 std::vector<double> data)
    : nc_(nc), nT_(nT),
      c_min_(c_min), c_max_(c_max),
      T_min_(T_min), T_max_(T_max),
      data_(std::move(data))
{
    if (nc_ < 2 || nT_ < 2)
        throw std::invalid_argument("FreeEnergyTable: nc and nT must be >= 2");
    if (c_max_ <= c_min_)
        throw std::invalid_argument("FreeEnergyTable: c_max must be > c_min");
    if (T_max_ <= T_min_)
        throw std::invalid_argument("FreeEnergyTable: T_max must be > T_min");
    if (static_cast<int>(data_.size()) != nc_ * nT_)
        throw std::invalid_argument(
            "FreeEnergyTable: data size mismatch (expected " +
            std::to_string(nc_ * nT_) + ", got " +
            std::to_string(data_.size()) + ")");

    dc_ = (c_max_ - c_min_) / (nc_ - 1);
    dT_ = (T_max_ - T_min_) / (nT_ - 1);
}

// ---------------------------------------------------------------------------
// Move semantics / destructor
// ---------------------------------------------------------------------------

FreeEnergyTable::FreeEnergyTable(FreeEnergyTable&& o) noexcept
    : nc_(o.nc_), nT_(o.nT_),
      c_min_(o.c_min_), c_max_(o.c_max_), dc_(o.dc_),
      T_min_(o.T_min_), T_max_(o.T_max_), dT_(o.dT_),
      data_(std::move(o.data_)),
      d_data_(o.d_data_)
{
    o.d_data_ = nullptr;
}

FreeEnergyTable& FreeEnergyTable::operator=(FreeEnergyTable&& o) noexcept
{
    if (this != &o) {
        freeDevice();
        nc_    = o.nc_;    nT_    = o.nT_;
        c_min_ = o.c_min_; c_max_ = o.c_max_; dc_ = o.dc_;
        T_min_ = o.T_min_; T_max_ = o.T_max_; dT_ = o.dT_;
        data_   = std::move(o.data_);
        d_data_ = o.d_data_;
        o.d_data_ = nullptr;
    }
    return *this;
}

FreeEnergyTable::~FreeEnergyTable()
{
    freeDevice();
}

// ---------------------------------------------------------------------------
// File loader
// ---------------------------------------------------------------------------

FreeEnergyTable FreeEnergyTable::fromFile(const std::string& path)
{
    std::ifstream ifs(path);
    if (!ifs.is_open())
        throw std::runtime_error("FreeEnergyTable::fromFile: cannot open '" + path + "'");

    // Skip comment lines; find the header line (nc nT c_min c_max T_min T_max)
    int    nc = 0, nT = 0;
    double c_min = 0, c_max = 0, T_min = 0, T_max = 0;
    bool   header_found = false;

    std::string line;
    while (std::getline(ifs, line)) {
        // Strip leading whitespace
        auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) continue;
        if (line[first] == '#')         continue;    // comment

        std::istringstream ss(line);
        if (!(ss >> nc >> nT >> c_min >> c_max >> T_min >> T_max))
            throw std::runtime_error(
                "FreeEnergyTable::fromFile: invalid header in '" + path + "'");
        header_found = true;
        break;
    }

    if (!header_found)
        throw std::runtime_error(
            "FreeEnergyTable::fromFile: no header found in '" + path + "'");

    // Read nc rows × nT columns of data
    std::vector<double> data;
    data.reserve(static_cast<std::size_t>(nc) * nT);

    while (std::getline(ifs, line)) {
        auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) continue;
        if (line[first] == '#')         continue;

        std::istringstream ss(line);
        double v;
        while (ss >> v)
            data.push_back(v);
    }

    if (static_cast<int>(data.size()) != nc * nT)
        throw std::runtime_error(
            "FreeEnergyTable::fromFile: expected " + std::to_string(nc * nT) +
            " values, got " + std::to_string(data.size()) +
            " in '" + path + "'");

    return FreeEnergyTable(c_min, c_max, nc, T_min, T_max, nT, std::move(data));
}

// ---------------------------------------------------------------------------
// Host evaluation — delegate to hostView() so logic lives in one place
// ---------------------------------------------------------------------------

double FreeEnergyTable::f(double c, double T) const    { return hostView().f(c, T); }
double FreeEnergyTable::dfdc(double c, double T) const { return hostView().dfdc(c, T); }
double FreeEnergyTable::dfdT(double c, double T) const { return hostView().dfdT(c, T); }

// ---------------------------------------------------------------------------
// GPU memory management
// ---------------------------------------------------------------------------

void FreeEnergyTable::allocDevice()
{
    if (d_data_) return;   // idempotent
    std::size_t bytes = static_cast<std::size_t>(nc_) * nT_ * sizeof(double);
    cudaError_t err = cudaMalloc(&d_data_, bytes);
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("FreeEnergyTable::allocDevice cudaMalloc failed: ") +
            cudaGetErrorString(err));
}

void FreeEnergyTable::uploadToDevice()
{
    if (!d_data_)
        throw std::runtime_error(
            "FreeEnergyTable::uploadToDevice: call allocDevice() first");
    std::size_t bytes = static_cast<std::size_t>(nc_) * nT_ * sizeof(double);
    cudaError_t err = cudaMemcpy(d_data_, data_.data(), bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("FreeEnergyTable::uploadToDevice cudaMemcpy failed: ") +
            cudaGetErrorString(err));
}

void FreeEnergyTable::freeDevice()
{
    if (d_data_) {
        cudaFree(d_data_);
        d_data_ = nullptr;
    }
}

// ---------------------------------------------------------------------------
// View factories
// ---------------------------------------------------------------------------

FreeEnergyTableView FreeEnergyTable::hostView() const
{
    return FreeEnergyTableView{
        data_.data(),
        nc_, nT_,
        c_min_, dc_, c_max_,
        T_min_, dT_, T_max_
    };
}

FreeEnergyTableView FreeEnergyTable::deviceView() const
{
    if (!d_data_)
        throw std::runtime_error(
            "FreeEnergyTable::deviceView: device data not allocated/uploaded");
    return FreeEnergyTableView{
        d_data_,
        nc_, nT_,
        c_min_, dc_, c_max_,
        T_min_, dT_, T_max_
    };
}

} // namespace Material
} // namespace PhiX
