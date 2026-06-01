#pragma once

#include "scheme/Scheme.h"

namespace PhiX::scheme {

struct CD2 {
    static constexpr int ghostRequired() { return 1; }
    static constexpr int order() { return 2; }
    static constexpr const char* name() { return "CD2"; }

    __host__ __device__
    static double d1(const double* s, int c, int stride, double inv_d) {
        return (s[c + stride] - s[c - stride]) * 0.5 * inv_d;
    }

    __host__ __device__
    static double d2(const double* s, int c, int stride, double inv_d2) {
        return (s[c + stride] - 2.0 * s[c] + s[c - stride]) * inv_d2;
    }
};

} // namespace PhiX::scheme
