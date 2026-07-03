#include "solver/LinearSolver.h"
#include "field/Reduce.h"
#include "scheme/CentralDifference.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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

namespace {

// ---------------------------------------------------------------------------
// y = D·∇²x over physical cells (overwrite; CD2)
// ---------------------------------------------------------------------------
__global__ void kernel_lap_apply(
        Real* y, const Real* x,
        int nx, int ny, int nz,
        int sx, int sy, int g, int dim,
        Real D, Real inv_dx2, Real inv_dy2, Real inv_dz2)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny * nz) return;
    const int i = tid % nx;
    const int j = (tid / nx) % ny;
    const int k = tid / (nx * ny);
    const int c = (i + g) + sx * ((j + g) + sy * (k + g));
    y[c] = D * scheme::CD2::laplacian(x, c, sx, sy, dim,
                                      inv_dx2, inv_dy2, inv_dz2);
}

// ---------------------------------------------------------------------------
// Elementwise CG helpers (whole stored array — scratch fields are
// zero-initialised so ghost slots stay finite)
// ---------------------------------------------------------------------------

// r = b − x + sigma·Lx        (initial residual for A = I − σL)
__global__ void kernel_residual(Real* r, const Real* b, const Real* x,
                                const Real* Lx, Real sigma, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) r[i] = b[i] - x[i] + sigma * Lx[i];
}

// Ap = p − sigma·Lp           (stored into Lp in-place)
__global__ void kernel_form_A(Real* Lp, const Real* p, Real sigma, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) Lp[i] = p[i] - sigma * Lp[i];
}

// x += alpha·p ;  r −= alpha·Ap        (alpha read from a device scalar)
__global__ void kernel_update_xr(Real* x, Real* r, const Real* p,
                                 const Real* Ap, const double* alpha, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const Real a = static_cast<Real>(*alpha);
        x[i] += a * p[i];
        r[i] -= a * Ap[i];
    }
}

// p = r + beta·p                        (beta read from a device scalar)
__global__ void kernel_update_p(Real* p, const Real* r, const double* beta,
                                int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = r[i] + static_cast<Real>(*beta) * p[i];
}

// p = r  (first search direction)
__global__ void kernel_copy_p(Real* p, const Real* r, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = r[i];
}

// Device scalar slots: s[0]=rho, s[1]=rhoNew, s[2]=pAp, s[3]=alpha, s[4]=beta
__global__ void kernel_cg_alpha(double* s)
{
    s[3] = s[0] / s[2];                 // alpha = rho / <p, Ap>
}

__global__ void kernel_cg_beta(double* s)
{
    s[4] = s[1] / s[0];                 // beta = rhoNew / rho
    s[0] = s[1];                        // rho <- rhoNew
}

ScalarField makeScratch(const Mesh& mesh, int ghost, const char* name) {
    ScalarField f(mesh, name, ghost);
    f.allocDevice();
    CUDA_CHECK(cudaMemset(f.d_curr, 0, f.storedSize * sizeof(Real)));
    return f;
}

inline int blocks(std::size_t n) { return static_cast<int>((n + 255) / 256); }

} // namespace

// ===========================================================================
// LaplacianOp
// ===========================================================================

LaplacianOp::LaplacianOp(double D, std::vector<BoundaryCondition*> bcs)
    : D_(D), bcs_(std::move(bcs)) {}

void LaplacianOp::apply(ScalarField& x, ScalarField& y) {
    if (!x.d_curr || !y.d_curr)
        throw std::runtime_error("LaplacianOp::apply: fields not on device");
    if (!bcBatch_.built()) bcBatch_.build(x, bcs_);
    bcBatch_.applyOnGPU(x);

    const Mesh& m = x.mesh;
    const int dim = m.dim;
    const Real inv_dx2 = static_cast<Real>(1.0 / (m.d[0] * m.d[0]));
    const Real inv_dy2 = (dim >= 2)
        ? static_cast<Real>(1.0 / (m.d[1] * m.d[1])) : Real(0);
    const Real inv_dz2 = (dim >= 3)
        ? static_cast<Real>(1.0 / (m.d[2] * m.d[2])) : Real(0);

    const int total = m.n[0] * m.n[1] * m.n[2];
    kernel_lap_apply<<<blocks(total), 256>>>(
        y.d_curr, x.d_curr, m.n[0], m.n[1], m.n[2],
        x.storedDims[0], x.storedDims[1], x.ghost, dim,
        static_cast<Real>(D_), inv_dx2, inv_dy2, inv_dz2);
    CUDA_CHECK(cudaGetLastError());
}

// ===========================================================================
// BiharmonicOp
// ===========================================================================

BiharmonicOp::BiharmonicOp(double G,
                           std::vector<BoundaryCondition*> bcsX,
                           std::vector<BoundaryCondition*> bcsLap)
    : G_(G)
    , bcsX_(std::move(bcsX))
    , bcsLap_(std::move(bcsLap))
    , inner_(1.0, bcsX_)     // persistent: BC batches built once, reused
    , outer_(-G, bcsLap_)
{}

void BiharmonicOp::apply(ScalarField& x, ScalarField& y) {
    if (!lap_) {
        lap_ = std::make_unique<ScalarField>(x.mesh, "_bih_lap", x.ghost);
        lap_->allocDevice();
        CUDA_CHECK(cudaMemset(lap_->d_curr, 0,
                              lap_->storedSize * sizeof(Real)));
    }
    inner_.apply(x, *lap_);     // lap = ∇²x
    outer_.apply(*lap_, y);     // y = −G·∇²(lap)
}

// ===========================================================================
// ConjugateGradient
// ===========================================================================

ConjugateGradient::ConjugateGradient(const Mesh& mesh, int ghost)
    : r_(makeScratch(mesh, ghost, "_cg_r"))
    , p_(makeScratch(mesh, ghost, "_cg_p"))
    , Lp_(makeScratch(mesh, ghost, "_cg_Lp"))
{
    CUDA_CHECK(cudaMalloc(&d_s_, 5 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_s_, 0, 5 * sizeof(double)));
}

ConjugateGradient::~ConjugateGradient() {
    if (d_s_) cudaFree(d_s_);   // best-effort; no throw in dtor
}

ConjugateGradient::Result ConjugateGradient::solve(
        LinearOperator& L, double sigma,
        ScalarField& x, const ScalarField& b,
        double relTol, int maxIter, bool throwOnFail)
{
    if (!x.d_curr || !b.d_curr)
        throw std::runtime_error("ConjugateGradient::solve: x/b not on device");
    if (x.storedSize != r_.storedSize || b.storedSize != r_.storedSize)
        throw std::invalid_argument(
            "ConjugateGradient::solve: field layout differs from scratch");
    if (x.ghost < L.ghostRequired())
        throw std::invalid_argument(
            "ConjugateGradient::solve: operator needs ghost >= "
            + std::to_string(L.ghostRequired()));

    const int  n  = static_cast<int>(r_.storedSize);
    const Real sg = static_cast<Real>(sigma);

    const double bNorm = std::sqrt(reduce::fieldDot(b, b));
    if (bNorm == 0.0) {
        // A x = 0 with SPD A → x = 0
        CUDA_CHECK(cudaMemset(x.d_curr, 0, x.storedSize * sizeof(Real)));
        return {0, 0.0, true};
    }

    // r = b − A x = b − x + σ·Lx ;  p = r
    L.apply(x, Lp_);
    kernel_residual<<<blocks(n), 256>>>(r_.d_curr, b.d_curr, x.d_curr,
                                        Lp_.d_curr, sg, n);
    CUDA_CHECK(cudaGetLastError());
    kernel_copy_p<<<blocks(n), 256>>>(p_.d_curr, r_.d_curr, n);
    CUDA_CHECK(cudaGetLastError());

    // Initial rho — one synchronous readback (also catches x0 == solution)
    double rho = reduce::fieldDot(r_, r_);
    CUDA_CHECK(cudaMemcpy(d_s_, &rho, sizeof(double),
                          cudaMemcpyHostToDevice));

    Result res;
    res.relResidual = std::sqrt(rho) / bNorm;
    if (res.relResidual <= relTol) {
        res.converged = true;
        return res;
    }

    const int cadence = (checkEvery > 0) ? checkEvery : 1;
    int it = 0;
    while (it < maxIter) {
        // --- enqueue `cadence` iterations with zero host round trips ---
        const int burst = std::min(cadence, maxIter - it);
        for (int k = 0; k < burst; ++k) {
            // Ap = p − σ·Lp   (in place in Lp_)
            L.apply(p_, Lp_);
            kernel_form_A<<<blocks(n), 256>>>(Lp_.d_curr, p_.d_curr, sg, n);
            CUDA_CHECK(cudaGetLastError());

            reduce::fieldDotAsync(p_, Lp_, d_s_ + 2);          // pAp
            kernel_cg_alpha<<<1, 1>>>(d_s_);                   // α = ρ/pAp
            kernel_update_xr<<<blocks(n), 256>>>(
                x.d_curr, r_.d_curr, p_.d_curr, Lp_.d_curr, d_s_ + 3, n);
            CUDA_CHECK(cudaGetLastError());

            reduce::fieldDotAsync(r_, r_, d_s_ + 1);           // rhoNew
            kernel_cg_beta<<<1, 1>>>(d_s_);                    // β, ρ←ρNew
            kernel_update_p<<<blocks(n), 256>>>(
                p_.d_curr, r_.d_curr, d_s_ + 4, n);
            CUDA_CHECK(cudaGetLastError());
        }
        it += burst;

        // --- convergence checkpoint: single synchronous readback ---
        CUDA_CHECK(cudaMemcpy(&rho, d_s_, sizeof(double),
                              cudaMemcpyDeviceToHost));
        if (!std::isfinite(rho))
            throw std::runtime_error(
                "ConjugateGradient::solve: residual became non-finite — "
                "operator is likely not SPD (check BCs / sign of sigma·L)");
        res.iterations  = it;
        res.relResidual = std::sqrt(rho) / bNorm;
        if (res.relResidual <= relTol) {
            res.converged = true;
            return res;
        }
    }

    if (throwOnFail)
        throw std::runtime_error(
            "ConjugateGradient::solve: no convergence after "
            + std::to_string(maxIter) + " iterations (relResidual = "
            + std::to_string(res.relResidual) + ")");
    return res;
}

} // namespace PhiX
