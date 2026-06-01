#pragma once

#include "scheme/Scheme.h"
#include "scheme/CentralDifference.h"

namespace PhiX::scheme {

// ---------------------------------------------------------------------------
// Iso9  —  9-point isotropic stencil (Patra-Karttunen, 2D only)
//
// Gradient (direction x, 2D):
//   df/dx[i,j] ≈ [4(f[i+1,j]-f[i-1,j])
//                +  (f[i+1,j+1]-f[i-1,j+1])
//                +  (f[i+1,j-1]-f[i-1,j-1])] / (12*dx)
//
// The Laplacian is NOT the direct sum of separable d² terms.
// Isotropic 9-point Laplacian (Patra-Karttunen):
//   ∇²f[i,j] ≈ [−3f[i,j]
//               + (f[i+1,j]+f[i-1,j]+f[i,j+1]+f[i,j-1]) / 2
//               + (f[i+1,j+1]+f[i-1,j+1]+f[i+1,j-1]+f[i-1,j-1]) / 4]
//              * (2 / (3*dx²))           (assumes dx==dy)
//
// For 1D or 3D meshes, both laplacian() and gradient() automatically fall
// back to standard CD2.  Stencil width is still 1 in all directions.
//
// RESTRICTION: the isotropic weights assume uniform spacing (dx == dy).
//              For non-square meshes, set the scheme explicitly to CD2.
// ---------------------------------------------------------------------------
struct Iso9 {
    static constexpr int ghostRequired() { return 1; }
    static constexpr int order() { return 2; }   // effective order on smooth solutions
    static constexpr const char* name() { return "Iso9"; }

    // Per-axis d1 fallback (CD2); used when dim != 2 or axis >= 2.
    __host__ __device__
    static double d1(const double* s, int c, int stride, double inv_d) {
        return CD2::d1(s, c, stride, inv_d);
    }

    // Per-axis d2 fallback (CD2).
    __host__ __device__
    static double d2(const double* s, int c, int stride, double inv_d2) {
        return CD2::d2(s, c, stride, inv_d2);
    }

    // -------------------------------------------------------------------------
    // Isotropic 9-point Laplacian (2D).  Falls back to CD2 for 1D/3D.
    // inv_dx2 must equal inv_dy2 (square mesh); the check is omitted for
    // performance — callers are responsible.
    // -------------------------------------------------------------------------
    __host__ __device__
    static double laplacian(const double* s, int c,
                            int sx, int /*sy*/, int dim,
                            double inv_dx2, double inv_dy2, double inv_dz2) {
        if (dim == 2) {
            // Patra-Karttunen weights: 1/2 face, 1/4 corner, inv_dx2 assumed == inv_dy2
            double center  = s[c];
            double face    = s[c + 1]      + s[c - 1]
                           + s[c + sx]     + s[c - sx];
            double corner  = s[c + 1 + sx] + s[c - 1 + sx]
                           + s[c + 1 - sx] + s[c - 1 - sx];
            // ≈ (face/2 + corner/4 - 3·center) * (2/(3·dx²))
            return (0.5 * face + 0.25 * corner - 3.0 * center)
                   * (2.0 / 3.0) * inv_dx2;
        }
        // fallback
        return CD2::laplacian(s, c, sx, /*sy will be unused in 1D*/ 1, dim,
                              inv_dx2, inv_dy2, inv_dz2);
    }

    // -------------------------------------------------------------------------
    // Isotropic 9-point gradient along `axis` (2D only).  Falls back for 1D/3D.
    // -------------------------------------------------------------------------
    __host__ __device__
    static double gradient(const double* s, int c, int axis,
                           int sx, int sy, int dim,
                           double inv_dx, double inv_dy, double inv_dz) {
        if (dim == 2 && axis <= 1) {
            if (axis == 0) {
                double inv_12dx = inv_dx / 12.0;
                int xp_jm = c + 1 - sx;
                int xp_j  = c + 1;
                int xp_jp = c + 1 + sx;
                int xm_jm = c - 1 - sx;
                int xm_j  = c - 1;
                int xm_jp = c - 1 + sx;
                return (4.0 * (s[xp_j] - s[xm_j])
                           + (s[xp_jp] - s[xm_jp])
                           + (s[xp_jm] - s[xm_jm])) * inv_12dx;
            } else {
                double inv_12dy = inv_dy / 12.0;
                int xm_yp = c - 1 + sx;
                int x_yp  = c     + sx;
                int xp_yp = c + 1 + sx;
                int xm_ym = c - 1 - sx;
                int x_ym  = c     - sx;
                int xp_ym = c + 1 - sx;
                return (4.0 * (s[x_yp] - s[x_ym])
                           + (s[xp_yp] - s[xp_ym])
                           + (s[xm_yp] - s[xm_ym])) * inv_12dy;
            }
        }
        return CD2::gradient(s, c, axis, sx, sy, dim, inv_dx, inv_dy, inv_dz);
    }
};

} // namespace PhiX::scheme
