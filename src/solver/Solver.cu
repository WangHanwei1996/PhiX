#include "solver/Solver.h"

#include <cuda_runtime.h>
#include <stdexcept>

namespace PhiX {

// ---------------------------------------------------------------------------
// CUDA error-checking macro
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

// ===========================================================================
// GPU kernels
// ===========================================================================

// ---------------------------------------------------------------------------
// Euler update:  dst[i] += coeff * src[i]   (scale-accumulate)
// Used both for Euler (coeff = dt) and inside RK4 stage assembly.
// ---------------------------------------------------------------------------
__global__ void kernel_axpy(double* dst, const double* src,
                             double coeff, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) dst[tid] += coeff * src[tid];
}

// ---------------------------------------------------------------------------
// Copy:  dst[i] = src[i]
// ---------------------------------------------------------------------------
__global__ void kernel_copy(double* dst, const double* src, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) dst[tid] = src[tid];
}

// ---------------------------------------------------------------------------
// RK4 phi_tmp assembly:  phi_tmp = phi + coeff * k_i
// ---------------------------------------------------------------------------
__global__ void kernel_rk4_tmp(double*       phi_tmp,
                                const double* phi,
                                const double* k,
                                double coeff, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) phi_tmp[tid] = phi[tid] + coeff * k[tid];
}

// ---------------------------------------------------------------------------
// RK4 final update:  phi += (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
// ---------------------------------------------------------------------------
__global__ void kernel_rk4_update(double*       phi,
                                   const double* k1,
                                   const double* k2,
                                   const double* k3,
                                   const double* k4,
                                   double dt6, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        phi[tid] += dt6 * (k1[tid] + 2.0 * k2[tid] + 2.0 * k3[tid] + k4[tid]);
}

// ===========================================================================
// Helper: allocate a scratch Field matching unknown's layout
// ===========================================================================
static Field makeScratch(const Field& ref, const std::string& tag) {
    Field f(ref.mesh, tag, ref.ghost);
    f.allocDevice();
    return f;
}

// ===========================================================================
// Constructor
// ===========================================================================

Solver::Solver(Equation&                       equation,
               std::vector<BoundaryCondition*> bcs,
               double                          dt_,
               TimeScheme                      scheme_)
    : dt(dt_)
    , scheme(scheme_)
    , equation_(equation)
    , bcs_(std::move(bcs))
    , rhs_(makeScratch(equation.unknown, equation.unknown.name + "_rhs"))
    , k1_(makeScratch(equation.unknown, "_k1"))
    , k2_(makeScratch(equation.unknown, "_k2"))
    , k3_(makeScratch(equation.unknown, "_k3"))
    , k4_(makeScratch(equation.unknown, "_k4"))
    , phi_tmp_(makeScratch(equation.unknown, "_phi_tmp"))
    , use_rk4_(scheme_ == TimeScheme::RK4)
{
    // Ensure unknown field is on device
    if (!equation_.unknown.deviceAllocated())
        throw std::runtime_error(
            "Solver: equation.unknown must have device memory allocated "
            "before constructing the Solver.");
}

// ===========================================================================
// Boundary conditions
// ===========================================================================

void Solver::applyBCsGPU() {
    for (auto* bc : bcs_) bc->applyOnGPU(equation_.unknown);
}

void Solver::applyBCsCPU() {
    for (auto* bc : bcs_) bc->applyOnCPU(equation_.unknown);
}

// ===========================================================================
// Euler path
// ===========================================================================

void Solver::eulerUpdateGPU() {
    // unknown.d_curr += dt * rhs.d_curr
    int n = static_cast<int>(equation_.unknown.storedSize);
    kernel_axpy<<<(n + 255) / 256, 256>>>(
        equation_.unknown.d_curr, rhs_.d_curr, dt, n);
    CUDA_CHECK(cudaGetLastError());
}

void Solver::eulerUpdateCPU() {
    auto& phi  = equation_.unknown.curr;
    auto& rhs  = rhs_.curr;
    for (std::size_t i = 0; i < phi.size(); ++i)
        phi[i] += dt * rhs[i];
}

// ===========================================================================
// RK4 GPU path
//
// Classical RK4 for  dphi/dt = f(phi, t):
//   k1 = f(phi)
//   k2 = f(phi + dt/2 * k1)
//   k3 = f(phi + dt/2 * k2)
//   k4 = f(phi +  dt  * k3)
//   phi += (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
//
// Each ki is stored in ki_.d_curr.
// phi_tmp_.d_curr holds the stage-shifted phi for RHS evaluation.
// BCs are applied on phi_tmp before each RHS call.
// ===========================================================================

// Helper: copy phi into phi_tmp then apply BCs on phi_tmp
static void prepTmpGPU(Field& phi_tmp, const Field& phi,
                        std::vector<BoundaryCondition*>& bcs,
                        const double* k, double coeff)
{
    int n = static_cast<int>(phi.storedSize);
    if (k && coeff != 0.0) {
        kernel_rk4_tmp<<<(n + 255) / 256, 256>>>(
            phi_tmp.d_curr, phi.d_curr, k, coeff, n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        kernel_copy<<<(n + 255) / 256, 256>>>(phi_tmp.d_curr, phi.d_curr, n);
        CUDA_CHECK(cudaGetLastError());
    }
    for (auto* bc : bcs) bc->applyOnGPU(phi_tmp);
}

void Solver::rk4AdvanceGPU() {
    int    n   = static_cast<int>(equation_.unknown.storedSize);
    double dt2 = dt * 0.5;
    double dt6 = dt / 6.0;

    Field& phi = equation_.unknown;

    // k1 = f(phi)
    applyBCsGPU();
    equation_.computeRHS(k1_);   // k1_.d_curr = rhs evaluated at phi

    // k2 = f(phi + dt/2 * k1);
    // Temporarily swap phi_tmp as the unknown for equation evaluation
    prepTmpGPU(phi_tmp_, phi, bcs_, k1_.d_curr, dt2);
    // Save real unknown, point equation to phi_tmp_, compute, restore
    {
        Field& saved  = equation_.unknown;   // ref to real phi (same object)
        // We can't rebind the reference, so we directly call the internal
        // compute method with phi_tmp_.d_curr as the source for RHS.
        // Because Equation::computeRHS uses term.field->d_curr, we need
        // to temporarily swap the device pointer of unknown.
        double* orig_ptr = saved.d_curr;
        saved.d_curr     = phi_tmp_.d_curr;
        equation_.computeRHS(k2_);
        saved.d_curr = orig_ptr;
    }

    // k3 = f(phi + dt/2 * k2)
    prepTmpGPU(phi_tmp_, phi, bcs_, k2_.d_curr, dt2);
    {
        double* orig_ptr = phi.d_curr;
        phi.d_curr       = phi_tmp_.d_curr;
        equation_.computeRHS(k3_);
        phi.d_curr = orig_ptr;
    }

    // k4 = f(phi + dt * k3)
    prepTmpGPU(phi_tmp_, phi, bcs_, k3_.d_curr, dt);
    {
        double* orig_ptr = phi.d_curr;
        phi.d_curr       = phi_tmp_.d_curr;
        equation_.computeRHS(k4_);
        phi.d_curr = orig_ptr;
    }

    // Final update: phi += (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    kernel_rk4_update<<<(n + 255) / 256, 256>>>(
        phi.d_curr,
        k1_.d_curr, k2_.d_curr, k3_.d_curr, k4_.d_curr,
        dt6, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ===========================================================================
// RK4 CPU path
// ===========================================================================

void Solver::rk4AdvanceCPU() {
    auto&       phi  = equation_.unknown.curr;
    auto&       tmp  = phi_tmp_.curr;
    auto&       k1c  = k1_.curr;
    auto&       k2c  = k2_.curr;
    auto&       k3c  = k3_.curr;
    auto&       k4c  = k4_.curr;
    std::size_t N    = phi.size();
    double dt2 = dt * 0.5, dt6 = dt / 6.0;

    auto evalRHS = [&](std::vector<double>& ki) {
        std::swap(phi, tmp);
        equation_.computeRHSCPU(rhs_);
        std::copy(rhs_.curr.begin(), rhs_.curr.end(), ki.begin());
        std::swap(phi, tmp);
    };

    // k1
    applyBCsCPU();
    equation_.computeRHSCPU(rhs_);
    std::copy(rhs_.curr.begin(), rhs_.curr.end(), k1c.begin());

    // k2
    for (std::size_t i = 0; i < N; ++i) tmp[i] = phi[i] + dt2 * k1c[i];
    applyBCsCPU();   // BCs on tmp via swap trick below
    std::swap(phi, tmp);
    equation_.computeRHSCPU(rhs_);
    std::copy(rhs_.curr.begin(), rhs_.curr.end(), k2c.begin());
    std::swap(phi, tmp);

    // k3
    for (std::size_t i = 0; i < N; ++i) tmp[i] = phi[i] + dt2 * k2c[i];
    std::swap(phi, tmp);
    equation_.computeRHSCPU(rhs_);
    std::copy(rhs_.curr.begin(), rhs_.curr.end(), k3c.begin());
    std::swap(phi, tmp);

    // k4
    for (std::size_t i = 0; i < N; ++i) tmp[i] = phi[i] + dt * k3c[i];
    std::swap(phi, tmp);
    equation_.computeRHSCPU(rhs_);
    std::copy(rhs_.curr.begin(), rhs_.curr.end(), k4c.begin());
    std::swap(phi, tmp);

    // combine
    for (std::size_t i = 0; i < N; ++i)
        phi[i] += dt6 * (k1c[i] + 2.0*k2c[i] + 2.0*k3c[i] + k4c[i]);
}

// ===========================================================================
// Public advance / advanceCPU
// ===========================================================================

void Solver::advance() {
    if (use_rk4_) {
        rk4AdvanceGPU();
    } else {
        applyBCsGPU();
        equation_.computeRHS(rhs_);
        eulerUpdateGPU();
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    equation_.unknown.advanceTimeLevelGPU();
    ++step;
    time += dt;
}

void Solver::advanceCPU() {
    if (use_rk4_) {
        rk4AdvanceCPU();
    } else {
        applyBCsCPU();
        equation_.computeRHSCPU(rhs_);
        eulerUpdateCPU();
    }
    equation_.unknown.advanceTimeLevelCPU();
    ++step;
    time += dt;
}

// ===========================================================================
// run
// ===========================================================================

void Solver::run(int nSteps,
                 int callbackEvery,
                 std::function<void(const Solver&)> callback)
{
    for (int s = 0; s < nSteps; ++s) {
        advance();
        if (callback && callbackEvery > 0 && (step % callbackEvery == 0))
            callback(*this);
    }
}

} // namespace PhiX
