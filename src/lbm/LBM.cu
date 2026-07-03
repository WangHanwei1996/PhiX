#include "lbm/LBM.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace PhiX {

// ---------------------------------------------------------------------------
// CUDA error-checking macro (local to this translation unit)
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

void LBMParams::validate() const {
    if (tau <= 0.5)
        throw std::invalid_argument("LBMParams: tau must be > 0.5 "
                                    "(nu = (tau-1/2)/3 must be positive)");
}

namespace {

// D2Q9 lattice: index, velocity (cx, cy), weight, opposite index
//   6 2 5
//   3 0 1
//   7 4 8
__device__ const int CX[9]  = { 0, 1, 0,-1, 0, 1,-1,-1, 1};
__device__ const int CY[9]  = { 0, 0, 1, 0,-1, 1, 1,-1,-1};
__device__ const int OPP[9] = { 0, 3, 4, 1, 2, 7, 8, 5, 6};

__host__ __device__ inline Real wq(int i) {
    return (i == 0) ? Real(4.0 / 9.0)
         : (i < 5)  ? Real(1.0 / 9.0)
                    : Real(1.0 / 36.0);
}

__host__ __device__ inline
Real feq(int i, Real rho, Real ux, Real uy, int cx, int cy) {
    const Real cu = Real(3) * (cx * ux + cy * uy);           // c·u / c_s²
    const Real u2 = Real(1.5) * (ux * ux + uy * uy);         // u² / (2c_s²)
    return wq(i) * rho * (Real(1) + cu + Real(0.5) * cu * cu - u2);
}

// ---------------------------------------------------------------------------
// Equilibrium initialisation
// ---------------------------------------------------------------------------
__global__ void kernel_lbm_init(Real* f, int n,
                                Real rho0, Real ux0, Real uy0)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    for (int i = 0; i < 9; ++i)
        f[i * n + idx] = feq(i, rho0, ux0, uy0, CX[i], CY[i]);
}

// ---------------------------------------------------------------------------
// BGK collision + Guo forcing (in place)
// ---------------------------------------------------------------------------
__global__ void kernel_lbm_collide(Real* f, int nx, int ny,
                                   Real invTau, Real fx, Real fy)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = nx * ny;
    if (idx >= n) return;

    Real fi[9], rho = Real(0), mx = Real(0), my = Real(0);
    for (int i = 0; i < 9; ++i) {
        fi[i] = f[i * n + idx];
        rho += fi[i];
        mx  += fi[i] * CX[i];
        my  += fi[i] * CY[i];
    }
    const Real invRho = Real(1) / rho;
    const Real ux = (mx + Real(0.5) * fx) * invRho;   // Guo velocity shift
    const Real uy = (my + Real(0.5) * fy) * invRho;

    const Real fw = Real(1) - Real(0.5) * invTau;     // (1 − 1/(2τ))
    for (int i = 0; i < 9; ++i) {
        const Real cu = Real(3) * (CX[i] * ux + CY[i] * uy);
        // Guo source: w_i [3(c−u) + 9(c·u)c] · F   (c_s² = 1/3)
        const Real Si = wq(i) * (Real(3) * ((CX[i] - ux) * fx
                                            + (CY[i] - uy) * fy)
                                 + Real(3) * cu * (CX[i] * fx + CY[i] * fy));
        f[i * n + idx] = fi[i]
            - invTau * (fi[i] - feq(i, rho, ux, uy, CX[i], CY[i]))
            + fw * Si;
    }
}

// ---------------------------------------------------------------------------
// Pull streaming with per-side periodic / halfway bounce-back.
// f_new[i](x) = f_post[i](x − c_i); pulling across a WALL side instead
// takes the opposite direction from the cell itself (halfway BB).
// ---------------------------------------------------------------------------
__global__ void kernel_lbm_stream(Real* fnew, const Real* f,
                                  int nx, int ny,
                                  int wxlo, int wxhi, int wylo, int wyhi)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = nx * ny;
    if (idx >= n) return;
    const int x = idx % nx;
    const int y = idx / nx;

    for (int i = 0; i < 9; ++i) {
        int sx = x - CX[i];
        int sy = y - CY[i];
        bool bb = false;
        if (sx < 0)        { if (wxlo) bb = true; else sx += nx; }
        else if (sx >= nx) { if (wxhi) bb = true; else sx -= nx; }
        if (sy < 0)        { if (wylo) bb = true; else sy += ny; }
        else if (sy >= ny) { if (wyhi) bb = true; else sy -= ny; }

        fnew[i * n + idx] = bb ? f[OPP[i] * n + idx]
                               : f[i * n + (sx + nx * sy)];
    }
}

// ---------------------------------------------------------------------------
// Macroscopic export into ghost-padded ScalarFields (physical cells)
// ---------------------------------------------------------------------------
__global__ void kernel_lbm_macro(const Real* f, int nx, int ny,
                                 Real fx, Real fy,
                                 Real* rho_out, Real* ux_out, Real* uy_out,
                                 int sx, int sy, int g)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = nx * ny;
    if (idx >= n) return;
    const int x = idx % nx;
    const int y = idx / nx;
    const int c = (x + g) + sx * ((y + g) + sy * g);

    Real rho = Real(0), mx = Real(0), my = Real(0);
    for (int i = 0; i < 9; ++i) {
        const Real v = f[i * n + idx];
        rho += v;
        mx  += v * CX[i];
        my  += v * CY[i];
    }
    const Real invRho = Real(1) / rho;
    if (rho_out) rho_out[c] = rho;
    if (ux_out)  ux_out[c]  = (mx + Real(0.5) * fx) * invRho;
    if (uy_out)  uy_out[c]  = (my + Real(0.5) * fy) * invRho;
}

} // namespace

// ===========================================================================
// LBM2D
// ===========================================================================

LBM2D::LBM2D(const Mesh& mesh, const LBMParams& p)
    : mesh_(mesh)
    , tau_(p.tau)
    , fx_(static_cast<Real>(p.fx))
    , fy_(static_cast<Real>(p.fy))
    , nx_(mesh.n[0])
    , ny_(mesh.n[1])
{
    p.validate();
    if (mesh.dim != 2)
        throw std::invalid_argument("LBM2D: 2D meshes only (D2Q9)");
    const std::size_t bytes =
        9u * static_cast<std::size_t>(nx_) * ny_ * sizeof(Real);
    CUDA_CHECK(cudaMalloc(&d_f_, bytes));
    CUDA_CHECK(cudaMalloc(&d_ftmp_, bytes));
}

LBM2D::~LBM2D() {
    if (d_f_)    cudaFree(d_f_);      // best-effort; no throw in dtor
    if (d_ftmp_) cudaFree(d_ftmp_);
}

void LBM2D::setWall(Axis axis, Side side) {
    if (axis == Axis::Z)
        throw std::invalid_argument("LBM2D::setWall: 2D lattice has no Z");
    walls_[static_cast<int>(axis)][side == Side::LOW ? 0 : 1] = 1;
}

void LBM2D::initialize(double rho0, double ux0, double uy0) {
    const int n = nx_ * ny_;
    kernel_lbm_init<<<(n + 255) / 256, 256>>>(
        d_f_, n, static_cast<Real>(rho0),
        static_cast<Real>(ux0), static_cast<Real>(uy0));
    CUDA_CHECK(cudaGetLastError());
    step_ = 0;
}

void LBM2D::step() {
    const int n = nx_ * ny_;
    kernel_lbm_collide<<<(n + 255) / 256, 256>>>(
        d_f_, nx_, ny_, static_cast<Real>(1.0 / tau_), fx_, fy_);
    CUDA_CHECK(cudaGetLastError());
    kernel_lbm_stream<<<(n + 255) / 256, 256>>>(
        d_ftmp_, d_f_, nx_, ny_,
        walls_[0][0], walls_[0][1], walls_[1][0], walls_[1][1]);
    CUDA_CHECK(cudaGetLastError());
    Real* t = d_f_; d_f_ = d_ftmp_; d_ftmp_ = t;   // buffer swap, no copy
    ++step_;
}

void LBM2D::run(int nSteps) {
    for (int s = 0; s < nSteps; ++s) step();
}

void LBM2D::macroscopics(ScalarField* rho, ScalarField* ux, ScalarField* uy) {
    const ScalarField* ref = rho ? rho : (ux ? ux : uy);
    if (!ref) return;
    for (const ScalarField* fld : {static_cast<const ScalarField*>(rho),
                                   static_cast<const ScalarField*>(ux),
                                   static_cast<const ScalarField*>(uy)}) {
        if (!fld) continue;
        if (fld->mesh.n[0] != nx_ || fld->mesh.n[1] != ny_)
            throw std::invalid_argument(
                "LBM2D::macroscopics: field mesh differs from lattice");
        if (!fld->d_curr)
            throw std::runtime_error(
                "LBM2D::macroscopics: field not device-allocated");
    }
    const int n = nx_ * ny_;
    kernel_lbm_macro<<<(n + 255) / 256, 256>>>(
        d_f_, nx_, ny_, fx_, fy_,
        rho ? rho->d_curr : nullptr,
        ux  ? ux->d_curr  : nullptr,
        uy  ? uy->d_curr  : nullptr,
        ref->storedDims[0], ref->storedDims[1], ref->ghost);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace PhiX
