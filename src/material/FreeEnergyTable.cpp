#include "material/FreeEnergyTable.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

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
// Evaluation — bilinear interpolation
// ---------------------------------------------------------------------------

double FreeEnergyTable::f(double c, double T) const
{
    c = clamp(c, c_min_, c_max_);
    T = clamp(T, T_min_, T_max_);

    // Fractional indices
    double fc = (c - c_min_) / dc_;
    double fT = (T - T_min_) / dT_;

    // Integer floor indices, clamped so ic+1 and iT+1 stay in range
    int ic = std::min(static_cast<int>(fc), nc_ - 2);
    int iT = std::min(static_cast<int>(fT), nT_ - 2);

    // Bilinear weights
    double wc = fc - ic;   // in [0, 1]
    double wT = fT - iT;

    return (1.0 - wc) * (1.0 - wT) * at(ic,     iT    )
         + (1.0 - wc) *        wT  * at(ic,     iT + 1)
         +        wc  * (1.0 - wT) * at(ic + 1, iT    )
         +        wc  *        wT  * at(ic + 1, iT + 1);
}

// ---------------------------------------------------------------------------
// Derivatives via central finite differences on the interpolated surface
// ---------------------------------------------------------------------------

double FreeEnergyTable::dfdc(double c, double T) const
{
    // Use a step of half a grid cell; clamp so both probe points stay in range
    double h = 0.5 * dc_;
    double c_lo = clamp(c - h, c_min_, c_max_);
    double c_hi = clamp(c + h, c_min_, c_max_);
    double span = c_hi - c_lo;
    if (span == 0.0) return 0.0;
    return (f(c_hi, T) - f(c_lo, T)) / span;
}

double FreeEnergyTable::dfdT(double c, double T) const
{
    double h = 0.5 * dT_;
    double T_lo = clamp(T - h, T_min_, T_max_);
    double T_hi = clamp(T + h, T_min_, T_max_);
    double span = T_hi - T_lo;
    if (span == 0.0) return 0.0;
    return (f(c, T_hi) - f(c, T_lo)) / span;
}

} // namespace Material
} // namespace PhiX
