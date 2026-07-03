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

// ---------------------------------------------------------------------------
// CD4 — 4th-order central differences (5-point per axis, ghost >= 2)
//
//   d1: (f[i-2] - 8f[i-1] + 8f[i+1] - f[i+2]) / (12*dx)
//   d2: (-f[i-2] + 16f[i-1] - 30f[i] + 16f[i+1] - f[i+2]) / (12*dx²)
//
// Laplacian is the separable sum of per-axis d2 (works in 1D/2D/3D on
// non-square meshes).  Requires TWO ghost layers; setRHS validates this
// against the field's ghost width and throws on mismatch.
// ---------------------------------------------------------------------------
struct CD4 {
    static constexpr int ghostRequired() { return 2; }
    static constexpr int order() { return 4; }
    static constexpr const char* name() { return "CD4"; }

    __host__ __device__
    static double d1(const double* s, int c, int stride, double inv_d) {
        return (s[c - 2*stride] - 8.0*s[c - stride]
              + 8.0*s[c + stride] - s[c + 2*stride]) * inv_d / 12.0;
    }

    __host__ __device__
    static double d2(const double* s, int c, int stride, double inv_d2) {
        return (-s[c - 2*stride] + 16.0*s[c - stride] - 30.0*s[c]
              + 16.0*s[c + stride] - s[c + 2*stride]) * inv_d2 / 12.0;
    }

    __host__ __device__
    static double laplacian(const double* s, int c,
                            int sx, int sy, int dim,
                            double inv_dx2, double inv_dy2, double inv_dz2) {
        double val = d2(s, c, 1, inv_dx2);
        if (dim >= 2) val += d2(s, c, sx, inv_dy2);
        if (dim >= 3) val += d2(s, c, sx * sy, inv_dz2);
        return val;
    }

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
