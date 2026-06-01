#pragma once

#include "scheme/Scheme.h"

namespace PhiX::scheme {

struct CD2 {
    static constexpr int ghostRequired() { return 1; }
    static constexpr int order() { return 2; }
    static constexpr const char* name() { return "CD2"; }

    // --- Separable per-axis primitives (stride-based) ---------------------
    __host__ __device__
    static double d1(const double* s, int c, int stride, double inv_d) {
        return (s[c + stride] - s[c - stride]) * 0.5 * inv_d;
    }

    __host__ __device__
    static double d2(const double* s, int c, int stride, double inv_d2) {
        return (s[c + stride] - 2.0 * s[c] + s[c - stride]) * inv_d2;
    }

    // --- Full-operator stencils (used by operator factories) -------------
    // Standard separable Laplacian: sum of second derivatives over axes.
    __host__ __device__
    static double laplacian(const double* s, int c,
                            int sx, int sy, int dim,
                            double inv_dx2, double inv_dy2, double inv_dz2) {
        double val = d2(s, c, 1, inv_dx2);
        if (dim >= 2) val += d2(s, c, sx, inv_dy2);
        if (dim >= 3) val += d2(s, c, sx * sy, inv_dz2);
        return val;
    }

    // Standard 3-point central gradient along `axis`.
    __host__ __device__
    static double gradient(const double* s, int c, int axis,
                           int sx, int sy, int /*dim*/,
                           double inv_dx, double inv_dy, double inv_dz) {
        int    stride = (axis == 0) ? 1 : (axis == 1) ? sx : sx * sy;
        double inv_d  = (axis == 0) ? inv_dx : (axis == 1) ? inv_dy : inv_dz;
        return d1(s, c, stride, inv_d);
    }
};

} // namespace PhiX::scheme
