#include "operators/Anisotropy.h"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace PhiX {

void AnisoParams::validate() const {
    if (W0 <= 0.0)
        throw std::invalid_argument("AnisoParams: W0 must be > 0");
    if (eps < 0.0)
        throw std::invalid_argument("AnisoParams: eps must be >= 0");
    if (m < 1)
        throw std::invalid_argument("AnisoParams: fold symmetry m must be >= 1");
    // |ε| < 1/(m²−1) is the convexity (no missing-orientation) limit; larger
    // values need regularisation — warn-by-throw only for clearly bad input.
    if (eps >= 1.0)
        throw std::invalid_argument("AnisoParams: eps must be < 1");
}

namespace {

// ---------------------------------------------------------------------------
// Face flux along the normal direction `nrm` given the face-local gradient
// (pn = normal component, pt = tangential component).  Kobayashi convention
// (matches the dendrite solver's facePW functors):
//   J_n = W0²·a·(a·pn + s·pt),   s = ε·m·sin(m(θ−θ0)) = −a'(θ)
// with θ = atan2(φ_y, φ_x) built from (pn, pt) in the correct (x, y) order.
// ---------------------------------------------------------------------------
// cos(m(θ−θ0)) and sin(m(θ−θ0)) WITHOUT transcendentals: rotate the
// gradient by −θ0 (host-precomputed ct0/st0), then take the m-th complex
// power of the unit direction via multiply-add recurrence.  FP64 atan2 +
// sincos run at 1/64 throughput on consumer GPUs and dominated this kernel
// — the algebraic path benchmarked >2× faster end to end.
__host__ __device__ inline
void cosSinM(Real px, Real py, Real ct0, Real st0, int m,
             Real& cosm, Real& sinm)
{
    const Real cx = ct0 * px + st0 * py;    // rotation by −θ0
    const Real cy = ct0 * py - st0 * px;
    const Real p2 = cx * cx + cy * cy;
    if (p2 <= Real(1e-150)) {               // no interface direction (margin
        cosm = Real(0);                     // against px² underflow)
        sinm = Real(0);
        return;
    }
    // Normalise FIRST: powers of the unit direction stay O(1), whereas
    // p2^m under/overflows for extreme-magnitude gradients (0·inf → NaN).
#ifdef __CUDA_ARCH__
    const Real inv = rsqrt(p2);
#else
    const Real inv = Real(1) / std::sqrt(p2);
#endif
    const Real ux = cx * inv, uy = cy * inv;
    Real zr = ux, zi = uy;                  // (ux + i·uy)^m, |z| == 1
    for (int k = 1; k < m; ++k) {
        const Real t = zr * ux - zi * uy;
        zi = zr * uy + zi * ux;
        zr = t;
    }
    cosm = zr;
    sinm = zi;
}

__host__ __device__ inline
Real fluxN(Real pn, Real pt, bool nIsX,
           Real W0sq, Real eps, int m, Real ct0, Real st0)
{
    const Real px = nIsX ? pn : pt;
    const Real py = nIsX ? pt : pn;
    Real cosm, sinm;
    cosSinM(px, py, ct0, st0, m, cosm, sinm);
    const Real a = Real(1) + eps * cosm;
    const Real s = eps * Real(m) * sinm;
    // x-face: J = W0² a (a·px + s·py);  y-face: J = W0² a (a·py − s·px)
    return nIsX ? W0sq * a * (a * pn + s * pt)
                : W0sq * a * (a * pn - s * pt);
}

// ---------------------------------------------------------------------------
// Fused divergence: rhs[c] += coeff·[ (Jx_e − Jx_w)/dx + (Jy_n − Jy_s)/dy ]
// Every face flux is built from the face-normal difference and the averaged
// tangential central differences — identical inputs from both adjacent
// cells → conservative.
// ---------------------------------------------------------------------------
__global__ void kernel_aniso_div(
        Real* rhs, const Real* f,
        Real coeff,
        int nx, int ny,
        int sx, int sy, int g,
        Real inv_dx, Real inv_dy,
        Real W0sq, Real eps, int m, Real ct0, Real st0)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny) return;
    const int i = tid % nx;
    const int j = tid / nx;
    // NOTE: 2D fields are ghost-padded in z as well — the k=0 slice sits at
    // offset sy*g, exactly like cell_idx(i, j, 0) in FaceOps.
    const int c = (i + g) + sx * ((j + g) + sy * g);

    const Real q = Real(0.25);

    // west x-face (i−½): normal grad + averaged tangential φ_y
    Real pn = (f[c] - f[c - 1]) * inv_dx;
    Real pt = q * inv_dy * (f[c - 1 + sx] - f[c - 1 - sx]
                            + f[c + sx] - f[c - sx]);
    const Real jw = fluxN(pn, pt, true, W0sq, eps, m, ct0, st0);

    // east x-face (i+½)
    pn = (f[c + 1] - f[c]) * inv_dx;
    pt = q * inv_dy * (f[c + sx] - f[c - sx]
                       + f[c + 1 + sx] - f[c + 1 - sx]);
    const Real je = fluxN(pn, pt, true, W0sq, eps, m, ct0, st0);

    // south y-face (j−½): normal grad + averaged tangential φ_x
    pn = (f[c] - f[c - sx]) * inv_dy;
    pt = q * inv_dx * (f[c - sx + 1] - f[c - sx - 1]
                       + f[c + 1] - f[c - 1]);
    const Real js = fluxN(pn, pt, false, W0sq, eps, m, ct0, st0);

    // north y-face (j+½)
    pn = (f[c + sx] - f[c]) * inv_dy;
    pt = q * inv_dx * (f[c + 1] - f[c - 1]
                       + f[c + sx + 1] - f[c + sx - 1]);
    const Real jn = fluxN(pn, pt, false, W0sq, eps, m, ct0, st0);

    rhs[c] += coeff * ((je - jw) * inv_dx + (jn - js) * inv_dy);
}

// CPU mirror of one cell (shared with the Term's cpu_kernel)
inline Real anisoDivCell(const Real* f, int c, int sx,
                         Real inv_dx, Real inv_dy,
                         Real W0sq, Real eps, int m, Real ct0, Real st0)
{
    const Real q = Real(0.25);
    Real pn = (f[c] - f[c - 1]) * inv_dx;
    Real pt = q * inv_dy * (f[c - 1 + sx] - f[c - 1 - sx]
                            + f[c + sx] - f[c - sx]);
    const Real jw = fluxN(pn, pt, true, W0sq, eps, m, ct0, st0);
    pn = (f[c + 1] - f[c]) * inv_dx;
    pt = q * inv_dy * (f[c + sx] - f[c - sx]
                       + f[c + 1 + sx] - f[c + 1 - sx]);
    const Real je = fluxN(pn, pt, true, W0sq, eps, m, ct0, st0);
    pn = (f[c] - f[c - sx]) * inv_dy;
    pt = q * inv_dx * (f[c - sx + 1] - f[c - sx - 1]
                       + f[c + 1] - f[c - 1]);
    const Real js = fluxN(pn, pt, false, W0sq, eps, m, ct0, st0);
    pn = (f[c + sx] - f[c]) * inv_dy;
    pt = q * inv_dx * (f[c + 1] - f[c - 1]
                       + f[c + sx + 1] - f[c + sx - 1]);
    const Real jn = fluxN(pn, pt, false, W0sq, eps, m, ct0, st0);
    return (je - jw) * inv_dx + (jn - js) * inv_dy;
}

__global__ void kernel_aniso_factor(
        Real* out, const Real* f,
        int nx, int ny, int sx, int sy, int g,
        Real inv_2dx, Real inv_2dy,
        Real eps, int m, Real ct0, Real st0)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny) return;
    const int i = tid % nx;
    const int j = tid / nx;
    const int c = (i + g) + sx * ((j + g) + sy * g);
    const Real px = (f[c + 1] - f[c - 1]) * inv_2dx;
    const Real py = (f[c + sx] - f[c - sx]) * inv_2dy;
    Real cosm, sinm;
    cosSinM(px, py, ct0, st0, m, cosm, sinm);
    out[c] = Real(1) + eps * cosm;
}

void checkField(const ScalarField& phi, const char* fn) {
    if (phi.mesh.dim != 2)
        throw std::invalid_argument(
            std::string(fn) + ": 2D meshes only (m-fold in-plane anisotropy)");
    if (phi.ghost < 1)
        throw std::invalid_argument(std::string(fn) + ": ghost >= 1 required");
}

} // namespace

Term anisoDiv(const ScalarField& phi, const AnisoParams& p, double coeff) {
    p.validate();
    checkField(phi, "anisoDiv");

    Term t;
    t.type  = TermType::COMPOSITE;
    t.field = &phi;
    t.coeff = coeff;
    t.ghostRequired = 1;

    const int  nx = phi.mesh.n[0], ny = phi.mesh.n[1];
    const int  sx = phi.storedDims[0], sy = phi.storedDims[1];
    const int  g  = phi.ghost;
    const Real inv_dx = static_cast<Real>(1.0 / phi.mesh.d[0]);
    const Real inv_dy = static_cast<Real>(1.0 / phi.mesh.d[1]);
    const Real W0sq   = static_cast<Real>(p.W0 * p.W0);
    const Real eps    = static_cast<Real>(p.eps);
    const Real ct0    = static_cast<Real>(std::cos(p.theta0));
    const Real st0    = static_cast<Real>(std::sin(p.theta0));
    const int  m      = p.m;

    const ScalarField* pf = &phi;

    t.gpu_launcher = [pf, nx, ny, sx, sy, g, inv_dx, inv_dy, W0sq, eps, m, ct0, st0]
                     (Real* d_rhs, double c, ScratchPool& pool) {
        if (!pf->d_curr)
            throw std::runtime_error("anisoDiv GPU: field not on device");
        const int total = nx * ny;
        kernel_aniso_div<<<(total + 255) / 256, 256, 0, pool.stream>>>(
            d_rhs, pf->d_curr, static_cast<Real>(c),
            nx, ny, sx, sy, g, inv_dx, inv_dy, W0sq, eps, m, ct0, st0);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("anisoDiv kernel error: ")
                + cudaGetErrorString(err));
    };

    t.cpu_kernel = [pf, nx, ny, sx, sy, g, inv_dx, inv_dy, W0sq, eps, m, ct0, st0]
                   (Real* rhs, double c, ScratchPool&) {
        const Real* f = pf->curr.data();
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            const int ctr = (i + g) + sx * ((j + g) + sy * g);
            rhs[ctr] += static_cast<Real>(c)
                      * anisoDivCell(f, ctr, sx, inv_dx, inv_dy,
                                     W0sq, eps, m, ct0, st0);
        }
    };

    return t;
}

void anisoFactorOnGPU(const ScalarField& phi, ScalarField& aOut,
                      const AnisoParams& p) {
    p.validate();
    checkField(phi, "anisoFactorOnGPU");
    if (aOut.storedSize != phi.storedSize)
        throw std::invalid_argument("anisoFactorOnGPU: layout mismatch");
    if (!phi.d_curr || !aOut.d_curr)
        throw std::runtime_error("anisoFactorOnGPU: fields not on device");

    const int total = phi.mesh.n[0] * phi.mesh.n[1];
    kernel_aniso_factor<<<(total + 255) / 256, 256>>>(
        aOut.d_curr, phi.d_curr,
        phi.mesh.n[0], phi.mesh.n[1], phi.storedDims[0], phi.storedDims[1],
        phi.ghost,
        static_cast<Real>(0.5 / phi.mesh.d[0]),
        static_cast<Real>(0.5 / phi.mesh.d[1]),
        static_cast<Real>(p.eps), p.m,
        static_cast<Real>(std::cos(p.theta0)),
        static_cast<Real>(std::sin(p.theta0)));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("anisoFactor kernel error: ")
                                 + cudaGetErrorString(err));
}

void anisoFactorOnCPU(const ScalarField& phi, ScalarField& aOut,
                      const AnisoParams& p) {
    p.validate();
    checkField(phi, "anisoFactorOnCPU");
    if (aOut.storedSize != phi.storedSize)
        throw std::invalid_argument("anisoFactorOnCPU: layout mismatch");

    const Real* f = phi.curr.data();
    const int nx = phi.mesh.n[0], ny = phi.mesh.n[1];
    const int sx = phi.storedDims[0], sy = phi.storedDims[1];
    const int g = phi.ghost;
    const Real i2dx = static_cast<Real>(0.5 / phi.mesh.d[0]);
    const Real i2dy = static_cast<Real>(0.5 / phi.mesh.d[1]);
    for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
        const int c = (i + g) + sx * ((j + g) + sy * g);
        const Real px = (f[c + 1] - f[c - 1]) * i2dx;
        const Real py = (f[c + sx] - f[c - sx]) * i2dy;
        Real cosm, sinm;
        cosSinM(px, py, static_cast<Real>(std::cos(p.theta0)),
                static_cast<Real>(std::sin(p.theta0)), p.m, cosm, sinm);
        aOut.curr[static_cast<std::size_t>(c)] =
            Real(1) + static_cast<Real>(p.eps) * cosm;
    }
}

// ===========================================================================
// 3D cubic anisotropy (Karma–Rappel), axis-aligned crystal axes.
// ===========================================================================

void Aniso3DParams::validate() const {
    if (W0 <= 0.0)
        throw std::invalid_argument("Aniso3DParams: W0 must be > 0");
    if (eps < 0.0 || eps >= 0.3)
        throw std::invalid_argument(
            "Aniso3DParams: eps must be in [0, 0.3) — note the convexity "
            "bound is ~1/15, larger values need regularisation");
}

namespace {

// Face flux along the normal slot `nIdx` (0=x,1=y,2=z) given the face-local
// gradient components ordered as (px, py, pz):
//   J_n = W0² a p_n [a + 16ε(n_n² − S)],  S = Σn⁴
__host__ __device__ inline
Real flux3(Real px, Real py, Real pz, int nIdx, Real W0sq, Real eps)
{
    const Real pn = (nIdx == 0) ? px : (nIdx == 1) ? py : pz;
    const Real p2 = px * px + py * py + pz * pz;
    if (p2 <= Real(1e-150)) return W0sq * pn;   // margin vs p² underflow

    // Normalised direction cosines squared FIRST — S = Σ(n_i²)² stays O(1)
    // (a raw Σp⁴/|p|⁴ under/overflows for extreme gradients → 0·inf NaN).
    const Real invp2 = Real(1) / p2;
    const Real nx2 = px * px * invp2;
    const Real ny2 = py * py * invp2;
    const Real nz2 = pz * pz * invp2;
    const Real S = nx2 * nx2 + ny2 * ny2 + nz2 * nz2;
    const Real a = Real(1) - Real(3) * eps + Real(4) * eps * S;
    const Real nn2 = (nIdx == 0) ? nx2 : (nIdx == 1) ? ny2 : nz2;
    return W0sq * a * pn * (a + Real(16) * eps * (nn2 - S));
}

// Face-local gradient at the face between cells L and R along `ax`:
// normal = 2-point difference; each tangential = averaged central diffs of
// the two adjacent cells (identical inputs from both sides → conservative).
__host__ __device__ inline
Real aniso3dDivCell(const Real* f, int c, int sx, int sz,
                    Real inv_dx, Real inv_dy, Real inv_dz,
                    Real W0sq, Real eps)
{
    const Real q = Real(0.25);
    const int st[3] = {1, sx, sz};
    const Real invd[3] = {inv_dx, inv_dy, inv_dz};

    Real div = Real(0);
    for (int ax = 0; ax < 3; ++ax) {
        const int  sn  = st[ax];
        const int  t1  = (ax == 0) ? 1 : 0;          // first tangential axis
        const int  t2  = (ax == 2) ? 1 : 2;          // second tangential axis
        const int  s1  = st[t1], s2 = st[t2];
        const Real id1 = invd[t1], id2 = invd[t2];

        Real p[2][3];                                // [west|east][x,y,z]
        for (int side = 0; side < 2; ++side) {
            const int R = c + side * sn;             // right cell of the face
            const int L = R - sn;                    // left cell
            const Real pn  = (f[R] - f[L]) * invd[ax];
            const Real pt1 = q * id1 * (f[L + s1] - f[L - s1]
                                        + f[R + s1] - f[R - s1]);
            const Real pt2 = q * id2 * (f[L + s2] - f[L - s2]
                                        + f[R + s2] - f[R - s2]);
            p[side][ax] = pn;
            p[side][t1] = pt1;
            p[side][t2] = pt2;
        }
        const Real jw = flux3(p[0][0], p[0][1], p[0][2], ax, W0sq, eps);
        const Real je = flux3(p[1][0], p[1][1], p[1][2], ax, W0sq, eps);
        div += (je - jw) * invd[ax];
    }
    return div;
}

__global__ void kernel_aniso3d_div(
        Real* rhs, const Real* f,
        Real coeff,
        int nx, int ny, int nz,
        int sx, int sy, int g,
        Real inv_dx, Real inv_dy, Real inv_dz,
        Real W0sq, Real eps)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny * nz) return;
    const int i = tid % nx;
    const int j = (tid / nx) % ny;
    const int k = tid / (nx * ny);
    const int c = (i + g) + sx * ((j + g) + sy * (k + g));
    rhs[c] += coeff * aniso3dDivCell(f, c, sx, sx * sy,
                                     inv_dx, inv_dy, inv_dz, W0sq, eps);
}

__global__ void kernel_aniso3d_factor(
        Real* out, const Real* f,
        int nx, int ny, int nz,
        int sx, int sy, int g,
        Real i2dx, Real i2dy, Real i2dz, Real eps)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny * nz) return;
    const int i = tid % nx;
    const int j = (tid / nx) % ny;
    const int k = tid / (nx * ny);
    const int c = (i + g) + sx * ((j + g) + sy * (k + g));
    const int sz = sx * sy;
    out[c] = aniso::factor3D((f[c + 1] - f[c - 1]) * i2dx,
                             (f[c + sx] - f[c - sx]) * i2dy,
                             (f[c + sz] - f[c - sz]) * i2dz, eps);
}

void checkField3D(const ScalarField& phi, const char* fn) {
    if (phi.mesh.dim != 3)
        throw std::invalid_argument(
            std::string(fn) + ": 3D meshes only (use anisoDiv for 2D)");
    if (phi.ghost < 1)
        throw std::invalid_argument(std::string(fn) + ": ghost >= 1 required");
}

} // namespace

Term anisoDiv3D(const ScalarField& phi, const Aniso3DParams& p, double coeff) {
    p.validate();
    checkField3D(phi, "anisoDiv3D");

    Term t;
    t.type  = TermType::COMPOSITE;
    t.field = &phi;
    t.coeff = coeff;
    t.ghostRequired = 1;

    const int  nx = phi.mesh.n[0], ny = phi.mesh.n[1], nz = phi.mesh.n[2];
    const int  sx = phi.storedDims[0], sy = phi.storedDims[1];
    const int  g  = phi.ghost;
    const Real idx = static_cast<Real>(1.0 / phi.mesh.d[0]);
    const Real idy = static_cast<Real>(1.0 / phi.mesh.d[1]);
    const Real idz = static_cast<Real>(1.0 / phi.mesh.d[2]);
    const Real W0sq = static_cast<Real>(p.W0 * p.W0);
    const Real eps  = static_cast<Real>(p.eps);

    const ScalarField* pf = &phi;

    t.gpu_launcher = [pf, nx, ny, nz, sx, sy, g, idx, idy, idz, W0sq, eps]
                     (Real* d_rhs, double c, ScratchPool& pool) {
        if (!pf->d_curr)
            throw std::runtime_error("anisoDiv3D GPU: field not on device");
        const int total = nx * ny * nz;
        kernel_aniso3d_div<<<(total + 255) / 256, 256, 0, pool.stream>>>(
            d_rhs, pf->d_curr, static_cast<Real>(c),
            nx, ny, nz, sx, sy, g, idx, idy, idz, W0sq, eps);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("anisoDiv3D kernel error: ")
                + cudaGetErrorString(err));
    };

    t.cpu_kernel = [pf, nx, ny, nz, sx, sy, g, idx, idy, idz, W0sq, eps]
                   (Real* rhs, double c, ScratchPool&) {
        const Real* f = pf->curr.data();
        for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            const int ctr = (i + g) + sx * ((j + g) + sy * (k + g));
            rhs[ctr] += static_cast<Real>(c)
                      * aniso3dDivCell(f, ctr, sx, sx * sy,
                                       idx, idy, idz, W0sq, eps);
        }
    };

    return t;
}

void anisoFactor3DOnGPU(const ScalarField& phi, ScalarField& aOut,
                        const Aniso3DParams& p) {
    p.validate();
    checkField3D(phi, "anisoFactor3DOnGPU");
    if (aOut.storedSize != phi.storedSize)
        throw std::invalid_argument("anisoFactor3DOnGPU: layout mismatch");
    if (!phi.d_curr || !aOut.d_curr)
        throw std::runtime_error("anisoFactor3DOnGPU: fields not on device");

    const int total = phi.mesh.n[0] * phi.mesh.n[1] * phi.mesh.n[2];
    kernel_aniso3d_factor<<<(total + 255) / 256, 256>>>(
        aOut.d_curr, phi.d_curr,
        phi.mesh.n[0], phi.mesh.n[1], phi.mesh.n[2],
        phi.storedDims[0], phi.storedDims[1], phi.ghost,
        static_cast<Real>(0.5 / phi.mesh.d[0]),
        static_cast<Real>(0.5 / phi.mesh.d[1]),
        static_cast<Real>(0.5 / phi.mesh.d[2]),
        static_cast<Real>(p.eps));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("anisoFactor3D kernel error: ")
                                 + cudaGetErrorString(err));
}

void anisoFactor3DOnCPU(const ScalarField& phi, ScalarField& aOut,
                        const Aniso3DParams& p) {
    p.validate();
    checkField3D(phi, "anisoFactor3DOnCPU");
    if (aOut.storedSize != phi.storedSize)
        throw std::invalid_argument("anisoFactor3DOnCPU: layout mismatch");

    const Real* f = phi.curr.data();
    const int nx = phi.mesh.n[0], ny = phi.mesh.n[1], nz = phi.mesh.n[2];
    const int sx = phi.storedDims[0], sy = phi.storedDims[1];
    const int g = phi.ghost, sz = sx * sy;
    const Real i2dx = static_cast<Real>(0.5 / phi.mesh.d[0]);
    const Real i2dy = static_cast<Real>(0.5 / phi.mesh.d[1]);
    const Real i2dz = static_cast<Real>(0.5 / phi.mesh.d[2]);
    for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
        const int c = (i + g) + sx * ((j + g) + sy * (k + g));
        aOut.curr[static_cast<std::size_t>(c)] =
            aniso::factor3D((f[c + 1] - f[c - 1]) * i2dx,
                            (f[c + sx] - f[c - sx]) * i2dy,
                            (f[c + sz] - f[c - sz]) * i2dz,
                            static_cast<Real>(p.eps));
    }
}

} // namespace PhiX
