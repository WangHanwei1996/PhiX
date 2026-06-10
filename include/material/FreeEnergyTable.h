#pragma once

#include <string>
#include <vector>
#include <stdexcept>

namespace PhiX {
namespace Material {

// ---------------------------------------------------------------------------
// FreeEnergyTable  —  2-D look-up table  f(c, T)  with bilinear interpolation
//
// Grid layout
// -----------
//   concentration : nc  uniformly spaced points in [c_min, c_max]
//   temperature   : nT  uniformly spaced points in [T_min, T_max]
//
// Internal storage (row-major, c varies slowly):
//   data_[ic * nT_ + iT]  =  f(c_min + ic*dc, T_min + iT*dT)
//
// File format  (.fetab, plain text)
// ----------------------------------
//   Lines beginning with '#' are ignored.
//   First non-comment line:   nc  nT  c_min  c_max  T_min  T_max
//   Followed by nc rows, each containing nT whitespace-separated values.
//   Example:
//
//     # Fe-B binary alloy free energy table
//     40 100 0.0 1.0 300.0 1800.0
//     -1.23e4  -1.22e4  ...   (nT values, ic=0)
//     ...
// ---------------------------------------------------------------------------

class FreeEnergyTable {
public:
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Construct from grid parameters and a flat row-major data vector.
    /// data must have exactly nc * nT elements.
    FreeEnergyTable(double c_min, double c_max, int nc,
                    double T_min, double T_max, int nT,
                    std::vector<double> data);

    /// Load from a .fetab file (see format description above).
    static FreeEnergyTable fromFile(const std::string& path);

    // -----------------------------------------------------------------------
    // Evaluation
    // -----------------------------------------------------------------------

    /// Bilinear interpolation — returns f at (c, T).
    /// c is clamped to [c_min, c_max]; T is clamped to [T_min, T_max].
    double f(double c, double T) const;

    /// Partial derivative ∂f/∂c via central finite differences on the grid.
    double dfdc(double c, double T) const;

    /// Partial derivative ∂f/∂T via central finite differences on the grid.
    double dfdT(double c, double T) const;

    // -----------------------------------------------------------------------
    // Grid accessors
    // -----------------------------------------------------------------------
    int    nc()   const { return nc_; }
    int    nT()   const { return nT_; }
    double cMin() const { return c_min_; }
    double cMax() const { return c_max_; }
    double TMin() const { return T_min_; }
    double TMax() const { return T_max_; }
    double dc()   const { return dc_; }
    double dT()   const { return dT_; }

    /// Read-only access to the raw flat data (row-major [ic*nT+iT]).
    const std::vector<double>& data() const { return data_; }

private:
    int    nc_,  nT_;
    double c_min_, c_max_, dc_;
    double T_min_, T_max_, dT_;
    std::vector<double> data_;     // size nc_ * nT_

    // Raw table value at integer grid indices (no interpolation, no clamping).
    double at(int ic, int iT) const { return data_[ic * nT_ + iT]; }

    // Clamp helper
    static double clamp(double v, double lo, double hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }
};

} // namespace Material
} // namespace PhiX
