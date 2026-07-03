/***********************************************************************\
 *
 *  Glass Formation Ability — 4-Phase Cu-Zr Solver (2D)
 *
 *  Author : Wang Hanwei
 *  Email  : wanghanweibnds2015@gmail.com
 *
 *  Description
 *  -----------
 *  Coupled Allen-Cahn (φ₀–φ₃, η) + Cahn-Hilliard (c) equations for
 *  simulating glass formation ability in binary Cu-Zr alloys.
 *
 *  Phase variables
 *  ---------------
 *    φ₀  — liquid / amorphous phase  (0: absent, 1: present)
 *    φ₁  — Cu₁₀Zr₇  crystal
 *    φ₂  — CuZr      crystal
 *    φ₃  — CuZr₂     crystal
 *    η   — amorphous order parameter (0: liquid, 1: amorphous)
 *    c   — mole fraction of Zr
 *    μ   — chemical potential  μ = ∂f/∂c  (auxiliary field)
 *
 *  Governing equations  [G7–G9 in variational_equation_to_differential_equation.md]
 *  ----------------------------------------------------------------------------------
 *    ∂φᵢ/∂t = −Σⱼ Lᵢⱼ δF/δφⱼ                        [G7]  (Einstein sum over j)
 *           = −Σⱼ Lᵢⱼ (∂f/∂φⱼ − ∇·∂f/∂(∇φⱼ))
 *    ∂η/∂t  = −L_η (∂f/∂η  − β∇²η)                   [G8]
 *    ∂c/∂t  = ∇·(M_c ∇μ)                              [G9, scalar mobility]
 *
 *  L matrix (symmetric, off-diagonal only — Lᵢᵢ = 0):
 *    L₀₁=0.05  L₀₂=0.08  L₀₃=0.05
 *    L₁₂=0.005  L₁₃=0.005  L₂₃=0.005
 *
 *  Free energy density  [G10]
 *  --------------------------
 *    f = φ₀(f₀(c,T) + h(η)Δf^SR(T)) + Σᵢ φᵢ fᵢ(T)
 *      + Σᵢ<ⱼ wᵢⱼ φᵢ² φⱼ²
 *      + w_η η²(1−η)² + w_ex η² Σᵢ₌₁³ φᵢ²
 *      + Σᵢ<ⱼ (εᵢⱼ²/2)|φᵢ∇φⱼ − φⱼ∇φᵢ|²
 *      + (β/2)|∇η|²
 *
 *  CALPHAD thermodynamics
 *  ----------------------
 *    f₀  — Cu-Zr liquid   (CALPHAD, composition- and T-dependent)
 *    f₁  — Cu₁₀Zr₇   (F3 in GFA_theory doc, T-dependent only)
 *    f₂  — CuZr       (F4, T-dependent only)
 *    f₃  — CuZr₂      (F5, T-dependent only)
 *
 *  Δf^SR(T) = −RT_g ln(1+α) f(τ),  τ=T/T_g  (Inden-like short-range order model)
 *
 *  NOTE: The constraint Σᵢ φᵢ = 1 is NOT enforced by these equations
 *        (diagonal L approximation).  A renormalisation step can be added
 *        after each time step if required.
 *
 \***********************************************************************/

#include "mesh/Mesh.h"
#include "field/ScalarField.h"
#include "boundary/BCFactory.h"
#include "equation/Equation.h"
#include "equation/EquationSystem.h"  // simultaneous coupled-equation update
#include "operators/FaceOps.h"   // interp/faceGrad/facePW/divFace (face-flux scheme)
#include "IO/ConfigFile.h"
#include "IO/FieldIO.h"
#include "IO/OutputWriter.h"

#include <curand_kernel.h>   // per-step thermal noise on η (cf. verification0)

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

// ===========================================================================
// Physical constant
// ===========================================================================
static constexpr double R_gas = 8.314;   // J/(mol·K)

// ===========================================================================
// Fixed model parameters  (GFA_theory parameter table)
// ===========================================================================

// Interface gradient energy coefficients  εᵢⱼ² [J/m]
static constexpr double eps01_sq = 1.1e-9;
static constexpr double eps02_sq = 0.7e-9;
static constexpr double eps03_sq = 1.2e-9;
static constexpr double eps12_sq = 1.0e-10;
static constexpr double eps13_sq = 1.0e-10;
static constexpr double eps23_sq = 1.0e-10;
static constexpr double beta     = 2.6e-12;   // |∇η|² gradient penalty [J/m]

// Double-well barrier heights wᵢⱼ / w_η / w_ex [J/m³] are now read from the JSONC
// config (constants.w01 … constants.w_ex); see main().  Paper (Table II) values are
// used as fallback defaults when a key is absent.

// Interface pair mobilities [m³/(s·J)]
static constexpr double L01      = 0.05;
static constexpr double L02      = 0.08;
static constexpr double L03      = 0.05;
static constexpr double L12      = 0.005;
static constexpr double L13      = 0.005;
static constexpr double L23      = 0.005;
static constexpr double L_eta    = 1.36;

// ===========================================================================
// CALPHAD: Cu-Zr liquid  (c = mole fraction of Zr)  [F7]
// ===========================================================================

__host__ __device__ inline double G_Cu_liq(double T)
{
    if (T <= 1357.77) {
        double T2 = T*T, T3 = T2*T, T7 = T3*T3*T;
        return 5194.28 + 120.97*T - 24.11*T*log(T)
             - 2.66e-3*T2 + 52478.0/T + 1.29e-7*T3 - 5.85e-21*T7;
    }
    return -46.55 + 173.88*T - 31.38*T*log(T);
}

__host__ __device__ inline double G_Zr_liq(double T)
{
    if (T <= 2128.0) {
        double T2 = T*T, T7 = T2*T2*T2*T;
        return 10320.10 + 116.57*T - 24.16*T*log(T)
             - 4.38e-3*T2 + 34971.0/T + 1.63e-22*T7;
    }
    return -8281.26 + 253.81*T - 42.14*T*log(T);
}

__host__ __device__ inline double L_CuZr_liq(double T)
{
    return -68890.0 + 16.20*T;
}

__host__ __device__ inline double compute_Gliq(double c, double T)
{
    double cs = fmax(1e-12, fmin(1.0 - 1e-12, c));
    return (1.0 - cs)*G_Cu_liq(T) + cs*G_Zr_liq(T)
         + R_gas*T*((1.0 - cs)*log(1.0 - cs) + cs*log(cs))
         + cs*(1.0 - cs)*L_CuZr_liq(T);
}

// μ₀ = ∂G_liq/∂c  (= chemical potential of Zr in liquid − Cu)
__host__ __device__ inline double compute_dGliq_dc(double c, double T)
{
    double cs = fmax(1e-12, fmin(1.0 - 1e-12, c));
    return G_Zr_liq(T) - G_Cu_liq(T)
         + R_gas*T*(log(cs) - log(1.0 - cs))
         + (1.0 - 2.0*cs)*L_CuZr_liq(T);
}

// ===========================================================================
// CALPHAD: crystal phases  (stoichiometric — no c dependence)
// ===========================================================================

// φ₁ : Cu₁₀Zr₇   [F3]
__host__ __device__ inline double G_phi1(double T)
{
    if (T <= 1357.77) {
        double T2 = T*T, T3 = T2*T;
        return -23926.88 + 126.60*T - 24.13*T*log(T)
             - 3.36e-3*T2 + 45300.13/T + 7.62e-8*T3;
    }
    if (T <= 2128.0) {
        double T2 = T*T, T3 = T2*T, T9 = T3*T3*T3;
        return -27332.11 + 158.06*T - 28.42*T*log(T)
             - 1.79e-3*T2 + 14338.11/T + 2.15e29/T9;
    }
    double T3 = T*T*T, T9 = T3*T3*T3;
    return -34818.02 + 214.26*T - 35.79*T*log(T) - 5.29e30/T9;
}

// φ₂ : CuZr   [F4]
__host__ __device__ inline double G_phi2(double T)
{
    if (T <= 1357.77) {
        double T2 = T*T, T3 = T2*T;
        return -20240.71 + 125.36*T - 24.14*T*log(T)
             - 3.52e-3*T2 + 43724.50/T + 6.46e-8*T3;
    }
    if (T <= 2128.0) {
        double T2 = T*T, T3 = T2*T, T9 = T3*T3*T3;
        return -23126.49 + 152.02*T - 27.77*T*log(T)
             - 2.19e-3*T2 + 17485.50/T + 1.82e29/T9;
    }
    double T3 = T*T*T, T9 = T3*T3*T3;
    return -32255.65 + 220.56*T - 36.76*T*log(T) - 6.53e30/T9;
}

// φ₃ : CuZr₂   [F5]
__host__ __device__ inline double G_phi3(double T)
{
    if (T <= 1357.77) {
        double T2 = T*T, T3 = T2*T;
        return -21648.74 + 127.90*T - 24.15*T*log(T)
             - 3.81e-3*T2 + 40748.31/T + 4.26e-8*T3;
    }
    if (T <= 2128.0) {
        double T2 = T*T, T3 = T2*T, T9 = T3*T3*T3;
        return -23553.36 + 145.49*T - 26.54*T*log(T)
             - 2.93e-3*T2 + 23430.57/T + 1.20e29/T9;
    }
    double T3 = T*T*T, T9 = T3*T3*T3;
    return -35786.44 + 237.33*T - 38.59*T*log(T) - 8.88e30/T9;
}

// ===========================================================================
// Short-range ordering driving force  [Δf^SR = −RT_g ln(1+α) f(τ),  τ=T/T_g]
//
//   A(p) = 79/(140p) + 474/497·(1/p−1)·(1/6+1/135+1/600) + 1/10+1/315+1/1500
//
//   f(τ<1) = 1 − (1/A)·[ 79/(140p)·τ⁻¹
//                        + 474/497·(1/p−1)·(τ³/6 + τ⁹/135 + τ¹⁵/600) ]
//   f(τ≥1) = −(1/A)·( τ⁻⁵/10 + τ⁻¹⁵/315 + τ⁻²⁵/1500 )
//
//   Returns energy density Δf^SR [J/m³]  (divides by Vm).
// ===========================================================================
static inline double compute_delta_f_SR(double T, double Tg,
                                        double alpha, double p, double Vm)
{
    double A = (79.0/(140.0*p))
             + (474.0/497.0)*(1.0/p - 1.0)*(1.0/6.0 + 1.0/135.0 + 1.0/600.0)
             + (1.0/10.0 + 1.0/315.0 + 1.0/1500.0);
    double tau = T / Tg;
    double f_tau;
    if (tau < 1.0) {
        double t3  = tau*tau*tau, t9 = t3*t3*t3, t15 = t9*t3*t3;
        f_tau = 1.0 - (1.0/A) * (
            79.0/(140.0*p) / tau
            + (474.0/497.0)*(1.0/p - 1.0)*(t3/6.0 + t9/135.0 + t15/600.0)
        );
    } else {
        double inv = 1.0/tau, i5  = inv*inv*inv*inv*inv,
               i15 = i5*i5*i5, i25 = i15*i5*i5;
        f_tau = -(1.0/A)*(i5/10.0 + i15/315.0 + i25/1500.0);
    }
    return -R_gas * Tg * log(1.0 + alpha) * f_tau / Vm;
}

// ===========================================================================
// Switching / barrier functions
// ===========================================================================
__host__ __device__ inline double h_func (double x) {
    return x*x*x*(10.0 - 15.0*x + 6.0*x*x);
}
__host__ __device__ inline double h_prime(double x) {
    return 30.0*x*x*(1.0 - x)*(1.0 - x);
}
__host__ __device__ inline double g_prime(double x) {
    return 2.0*x*(1.0 - x)*(1.0 - 2.0*x);
}

// ===========================================================================
// Hard clamp of a field's physical cells to [0,1]  (cf. GFA verification0).
//
// Explicit Euler can overshoot the stiff double-well once a phase leaves
// [0,1]; the wᵢⱼφᵢ²φⱼ² / CALPHAD terms then diverge.  Projecting each phase
// back into [0,1] after every step keeps the order parameters physical and
// the integration stable.  2D kernel (z stored index = ghost).
// ===========================================================================
__global__ void k_clamp01(double* d_curr,
                          int nx, int ny, int sx, int sy, int ghost)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= nx || iy >= ny) return;
    int idx = (ix + ghost) + sx * ((iy + ghost) + sy * ghost);
    double v = d_curr[idx];
    if (v < 0.0) v = 0.0;
    if (v > 1.0) v = 1.0;
    d_curr[idx] = v;
}

// ===========================================================================
// Gibbs simplex projection of the four phase fractions  (φ₀+φ₁+φ₂+φ₃ = 1).
//
// The governing equations use the diagonal-L approximation and do NOT conserve
// Σφᵢ (see file header NOTE), so a [0,1] clamp alone lets every crystal phase
// independently saturate to 1.  Projecting each cell back onto the simplex
// {φᵢ ≥ 0, Σφᵢ = 1} after every step restores phase competition: clip negatives,
// then renormalise by the sum.  (η is NOT part of this simplex — it keeps its
// own [0,1] clamp.)
// ===========================================================================
__global__ void k_proj_simplex4(double* p0, double* p1, double* p2, double* p3,
                                int nx, int ny, int sx, int sy, int ghost)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= nx || iy >= ny) return;
    int idx = (ix + ghost) + sx * ((iy + ghost) + sy * ghost);

    double a0 = fmax(p0[idx], 0.0), a1 = fmax(p1[idx], 0.0);
    double a2 = fmax(p2[idx], 0.0), a3 = fmax(p3[idx], 0.0);
    double s  = a0 + a1 + a2 + a3;
    if (s > 1e-12) {
        double inv = 1.0 / s;
        p0[idx] = a0 * inv; p1[idx] = a1 * inv;
        p2[idx] = a2 * inv; p3[idx] = a3 * inv;
    } else {                       // degenerate cell → default to liquid
        p0[idx] = 1.0; p1[idx] = 0.0; p2[idx] = 0.0; p3[idx] = 0.0;
    }
}

// ===========================================================================
// Per-step thermal noise on η  (cf. GFA verification0).
//
// η starts at 0 and its driving term ∝ η²(1−η)² vanishes at η=0, so without a
// stochastic kick η can never leave the liquid state — no glass nucleation.
// Each step we add N(mean, std²) to η and clamp back to [0,1].  One curand
// state per physical cell, seeded once.  Only η is perturbed (the φ simplex is
// left untouched).
// ===========================================================================
__global__ void k_initStates(curandState* states,
                             unsigned long long seed, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    curand_init(seed, tid, 0, &states[tid]);
}

__global__ void k_noiseClamp(double* d_curr, curandState* states,
                             int nx, int ny, int sx, int sy, int ghost,
                             double noise_mean, double noise_std)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= nx || iy >= ny) return;

    int tid   = iy * nx + ix;
    double nz = noise_mean + noise_std * curand_normal_double(&states[tid]);

    int idx = (ix + ghost) + sx * ((iy + ghost) + sy * ghost);
    double v = d_curr[idx] + nz;
    if (v < 0.0) v = 0.0;
    if (v > 1.0) v = 1.0;
    d_curr[idx] = v;
}

// ===========================================================================
// main
// ===========================================================================
int main(int argc, char* argv[])
{
    using namespace PhiX;

    IO::ConfigFile cfg = IO::ConfigFile::fromArgs(argc, argv);

    // -----------------------------------------------------------------------
    // 1. Mesh
    // -----------------------------------------------------------------------
    const int    nx = cfg["mesh"]["nx"];
    const double dx = cfg["mesh"]["dx"];
    const double x0 = cfg["mesh"]["x0"];
    const int    ny = cfg["mesh"]["ny"];
    const double dy = cfg["mesh"]["dy"];
    const double y0 = cfg["mesh"]["y0"];

    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN, nx, dx, x0, ny, dy, y0);
    mesh.print();

    // -----------------------------------------------------------------------
    // 2. Time parameters
    // -----------------------------------------------------------------------
    const double dt     = cfg["initialize"]["dt"];
    const int    nSteps = cfg["initialize"]["nSteps"];

    // -----------------------------------------------------------------------
    // 3. Physical constants  (ε², w, L from file-scope constexpr; Vm/SR from config)
    // -----------------------------------------------------------------------
    const double T       = cfg["constants"]["T"];        // temperature [K]
    // Molar volumes [m³/mol-atom]:
    //   liquid/glass 1.058e-5,  Cu₁₀Zr₇ 9.77e-6,  CuZr 1.039e-5,  CuZr₂ 1.164e-5
    const double Vm_liq  = cfg["constants"]["Vm_liq"];   // 1.058e-5 m³/mol-atom
    const double Vm_phi1 = cfg["constants"]["Vm_phi1"];  // 9.77e-6  m³/mol-atom
    const double Vm_phi2 = cfg["constants"]["Vm_phi2"];  // 1.039e-5 m³/mol-atom
    const double Vm_phi3 = cfg["constants"]["Vm_phi3"];  // 1.164e-5 m³/mol-atom

    // Short-range ordering parameters for Δf^SR
    const double T_g   = cfg["constants"]["T_g"];    // glass transition temperature [K]
    const double alpha = cfg["constants"]["alpha"];  // scaling parameter α
    const double p_SR  = cfg["constants"]["p_SR"];   // Inden p-parameter (e.g. 0.28 FCC-like)

    // Per-step thermal noise on η (Gaussian; activates glass nucleation)
    const double noise_mean = cfg["constants"]["noise_mean"];  // per-step noise mean
    const double noise_std  = cfg["constants"]["noise_std"];   // per-step noise std dev (0 = off)
    const unsigned long long noise_seed =
        cfg["constants"].count("noise_seed")
            ? (unsigned long long)cfg["constants"]["noise_seed"]
            : 42ULL;

    // Cahn-Hilliard composition mobility (constant).  Paper Eq.2 carries NO |∇c|²
    // term ⇒ κ_c = 0, so μ = ∂f/∂c has no Laplacian part (kappa_c in the config is
    // left unused in this minimal liquid-driven form).
    const double M_c = cfg["constants"]["M_c"];   // [m⁵/(J·s)]

    // Double-well barrier heights — config-overridable (defaults = paper Table II).
    auto getConst = [&](const char* key, double def) -> double {
        return cfg["constants"].count(key) ? cfg["constants"][key].get<double>() : def;
    };
    const double w01   = getConst("w01",   4.2e8);
    const double w02   = getConst("w02",   4.1e8);
    const double w03   = getConst("w03",   4.7e8);
    const double w12   = getConst("w12",   1.0e8);
    const double w13   = getConst("w13",   1.0e8);
    const double w23   = getConst("w23",   1.0e8);
    const double w_eta = getConst("w_eta", 2.5e7);
    const double w_ex  = getConst("w_ex",  2.0e9);

    // Pre-compute T-dependent scalars
    const double f1_val = G_phi1(T) / Vm_phi1;   // free energy density of φ₁
    const double f2_val = G_phi2(T) / Vm_phi2;
    const double f3_val = G_phi3(T) / Vm_phi3;
    const double dFSR   = compute_delta_f_SR(T, T_g, alpha, p_SR, Vm_liq);

    // -----------------------------------------------------------------------
    // 4. Fields
    // -----------------------------------------------------------------------
    ScalarField phi0(mesh, "phi0", /*ghost=*/1);
    ScalarField phi1(mesh, "phi1", /*ghost=*/1);
    ScalarField phi2(mesh, "phi2", /*ghost=*/1);
    ScalarField phi3(mesh, "phi3", /*ghost=*/1);
    ScalarField eta (mesh, "eta",  /*ghost=*/1);
    ScalarField c   (mesh, "c",    /*ghost=*/1);  // Zr mole fraction — now solved by CH
    ScalarField mu  (mesh, "mu",   /*ghost=*/1);  // chemical potential μ=∂f/∂c (auxiliary)

    phi0.fill(0); phi1.fill(0); phi2.fill(0); phi3.fill(0);
    eta .fill(0);
    c   .fill(0.5);  // default composition (overwritten by initField below)
    mu  .fill(0.0);

    const std::string start_from = cfg["initialize"]["start_from"];
    const int         start_step = IO::resolveStartStep(start_from);

    IO::initField(phi0, start_step);
    IO::initField(phi1, start_step);
    IO::initField(phi2, start_step);
    IO::initField(phi3, start_step);
    IO::initField(eta,  start_step);
    IO::initField(c,    start_step);   // composition is now a solved field

    auto allocUp = [](ScalarField& f){ f.allocDevice(); f.uploadAllToDevice(); };
    allocUp(phi0); allocUp(phi1); allocUp(phi2); allocUp(phi3);
    allocUp(eta);  allocUp(c);     allocUp(mu);

    // -----------------------------------------------------------------------
    // 5. Boundary conditions
    // -----------------------------------------------------------------------
    auto  bcSet = buildBCs(mesh, cfg["boundary_conditions"]);
    auto& bcs   = bcSet.ptrs;

    // -----------------------------------------------------------------------
    // 5b. [φ₀ TEMPLATE] Staggered face fields for the gradient-energy flux.
    //
    //  For each phase m, on both axes a, we need:
    //    pA[m] = interp(φₘ, a)     (φₘ averaged onto a-faces)
    //    gA[m] = faceGrad(φₘ, a)   (∂_a φₘ on a-faces)
    //  G{x,y}0 accumulate the total φ₀ divergence flux; t1/t2 are per-pair
    //  scratch.  These mirror the staggered scheme in dendrite_growth.cu.
    // -----------------------------------------------------------------------
    std::vector<ScalarField*> phis = { &phi0, &phi1, &phi2, &phi3 };

    auto makeFaceVec = [&](int ax, const std::string& tag) {
        std::vector<FaceField> v; v.reserve(4);
        for (int m = 0; m < 4; ++m)
            v.emplace_back(mesh, ax, tag + std::to_string(m));
        return v;
    };
    std::vector<FaceField> pX = makeFaceVec(0, "pX"), gX = makeFaceVec(0, "gX");
    std::vector<FaceField> pY = makeFaceVec(1, "pY"), gY = makeFaceVec(1, "gY");
    std::vector<FaceField> Gx = makeFaceVec(0, "Gx"), Gy = makeFaceVec(1, "Gy"); // per-eq flux
    FaceField t1x(mesh, 0, "t1x"), t2x(mesh, 0, "t2x");   // x-face scratch
    FaceField t1y(mesh, 1, "t1y"), t2y(mesh, 1, "t2y");   // y-face scratch

    auto allocUpFace = [](FaceField& f){ f.fill(0.0); f.allocDevice(); f.uploadToDevice(); };
    for (auto& f : pX) allocUpFace(f);
    for (auto& f : gX) allocUpFace(f);
    for (auto& f : pY) allocUpFace(f);
    for (auto& f : gY) allocUpFace(f);
    for (auto& f : Gx) allocUpFace(f);
    for (auto& f : Gy) allocUpFace(f);
    allocUpFace(t1x); allocUpFace(t2x);
    allocUpFace(t1y); allocUpFace(t2y);

    // -----------------------------------------------------------------------
    // 6. Helpers: variational derivatives δF/δφⱼ = ∂f/∂φⱼ − ∇·(∂f/∂(∇φⱼ))
    //
    //  Each helper returns RHSExpr for ONE term  −Lᵢⱼ · δF/δφⱼ
    //  contributed to the ∂φᵢ/∂t equation.
    //
    //  dF_dphi0(Lij)  — variational derivative w.r.t. φ₀  (G11 + G15 i=0):
    //    ∂f/∂φ₀ = f₀(c,T) + h(η)Δf^SR
    //             + 2φ₀(w₀₁φ₁² + w₀₂φ₂² + w₀₃φ₃²)
    //             + Σₖ ε₀ₖ²[φ₀|∇φₖ|² − φₖ(∇φ₀·∇φₖ)]
    //    ∇·(∂f/∂(∇φ₀)) = Σₖ ε₀ₖ²[φₖ²∇²φ₀ + φₖ(∇φ₀·∇φₖ) − φ₀|∇φₖ|² − φ₀φₖ∇²φₖ]
    //    → δF/δφ₀ = f₀(c,T)+h(η)Δf^SR + 2φ₀(w₀₁φ₁²+w₀₂φ₂²+w₀₃φ₃²)
    //               + Σₖ ε₀ₖ²[2φ₀|∇φₖ|²−2φₖ(∇φ₀·∇φₖ)−φₖ²∇²φ₀+φ₀φₖ∇²φₖ]
    //
    //  dF_dphis(phi_s, fs, ws0,ws1,ws2,ws3_exc, eps_s0..3, Lij)
    //    — variational derivative w.r.t. φₛ (s=1,2,3)  (G12 + G15):
    //    ∂f/∂φₛ = fₛ + 2φₛ Σₖ≠ₛ wₛₖ φₖ² + 2w_ex η²φₛ
    //             + Σₖ≠ₛ εₛₖ²[φₛ|∇φₖ|² − φₖ(∇φₛ·∇φₖ)]
    //    ∇·(∂f/∂(∇φₛ)) = Σₖ≠ₛ εₛₖ²[φₖ²∇²φₛ + φₖ(∇φₛ·∇φₖ) − φₛ|∇φₖ|² − φₛφₖ∇²φₖ]
    //    → δF/δφₛ = fₛ + 2φₛ Σₖ≠ₛ wₛₖ φₖ² + 2w_ex η²φₛ
    //               + Σₖ≠ₛ εₛₖ²[2φₛ|∇φₖ|²−2φₖ(∇φₛ·∇φₖ)−φₖ²∇²φₛ+φₛφₖ∇²φₖ]
    //
    //  The gradient-energy contribution of one pair (φₛ, φₖ) to −Lᵢⱼ·δF/δφⱼ is:
    //    +Lij·εₛₖ²·[−2φₛ|∇φₖ|² + 2φₖ(∇φₛ·∇φₖ) + φₖ²∇²φₛ − φₛφₖ∇²φₖ]
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // Face-flux reformulation of the gradient energy (all phases φ₀–φ₃).
    //
    //  For a pair (j,k) the full variational derivative splits *exactly* as
    //    δe/δφⱼ = ε²(∇φₖ)·A + ε²∇·(φₖ A),   A = φⱼ∇φₖ − φₖ∇φⱼ
    //  so the contribution −Lij·δe/δφⱼ to ∂φᵢ/∂t becomes
    //    (non-div, cell-centred):  −Lij ε² φⱼ|∇φₖ|² + Lij ε² φₖ(∇φₖ·∇φⱼ)
    //    (divergence, face flux):  +Lij ε² ∇·(φₖ²∇φⱼ − φₖφⱼ∇φₖ)
    //  Summing the two reproduces gradE() above term-for-term in the continuum,
    //  but the *diffusive* divergence is now a consistent staggered face flux
    //  (compact 5-point coupling, no odd/even checkerboard), while only the
    //  irreducible rotational source — a genuine ∇·∇ dot product that is not a
    //  divergence — remains on cell centres.  The face part is assembled by
    //  addPairFlux() below; here is just the cell-centred residual.
    // -----------------------------------------------------------------------
    auto gradE_nd = [&](const ScalarField& phi_j, const ScalarField& phi_k,
                        double eps_jk_sq, double Lij) -> RHSExpr
    {
        return
            - mul(phi_j, grad_dot(phi_k, phi_k), Lij * eps_jk_sq)   // −Lij ε² φⱼ|∇φₖ|²
            + mul(phi_k, grad_dot(phi_k, phi_j), Lij * eps_jk_sq);  // +Lij ε² φₖ(∇φₖ·∇φⱼ)
    };

    // Bulk (potential + double-well) parts of −Lij·δF/δφⱼ for j=1,2,3,
    // i.e. dF_phiⱼ with the gradient-energy gradE() terms removed.
    auto bulk_phi1 = [&](double Lij) -> RHSExpr {
        return
            pw(phi1, PHIX_FN(double) { return -Lij * f1_val; })
            - mul(phi1,
                  w01*(phi0*phi0) + w12*(phi2*phi2) + w13*(phi3*phi3),
                  2.0 * Lij)
            - mul(eta*eta, phi1, 2.0 * Lij * w_ex);
    };
    auto bulk_phi2 = [&](double Lij) -> RHSExpr {
        return
            pw(phi2, PHIX_FN(double) { return -Lij * f2_val; })
            - mul(phi2,
                  w02*(phi0*phi0) + w12*(phi1*phi1) + w23*(phi3*phi3),
                  2.0 * Lij)
            - mul(eta*eta, phi2, 2.0 * Lij * w_ex);
    };
    auto bulk_phi3 = [&](double Lij) -> RHSExpr {
        return
            pw(phi3, PHIX_FN(double) { return -Lij * f3_val; })
            - mul(phi3,
                  w03*(phi0*phi0) + w13*(phi1*phi1) + w23*(phi2*phi2),
                  2.0 * Lij)
            - mul(eta*eta, phi3, 2.0 * Lij * w_ex);
    };

    // Bulk part of −Lij·δF/δφ₀  (φ₀ has the CALPHAD liquid + h(η)Δf^SR drive,
    // and no amorphous w_ex coupling).
    auto bulk_phi0 = [&](double Lij) -> RHSExpr {
        return
            pw(c, eta, PHIX_FN (double cv, double ev) {
                return -Lij * (compute_Gliq(cv, T) / Vm_liq + h_func(ev) * dFSR);
            })
            - mul(phi0,
                  w01*(phi1*phi1) + w02*(phi2*phi2) + w03*(phi3*phi3),
                  2.0 * Lij);
    };

    // Dispatch the four bulk helpers by phase index.
    auto bulk = [&](int j, double Lij) -> RHSExpr {
        switch (j) {
            case 0:  return bulk_phi0(Lij);
            case 1:  return bulk_phi1(Lij);
            case 2:  return bulk_phi2(Lij);
            default: return bulk_phi3(Lij);
        }
    };

    // Mobility matrix as the weighted graph LAPLACIAN  L = D − A  of the pair
    // mobilities (NOT the bare adjacency A).  This makes  ∂φᵢ/∂t = −Σⱼ Lᵢⱼ δF/δφⱼ
    // the Steinbach pairwise form  −Σⱼ≠ᵢ Mᵢⱼ(δF/δφᵢ − δF/δφⱼ):
    //   • L is positive-semidefinite ⇒ dF/dt = −gᵀLg ≤ 0  (dissipative, stable)
    //   • row sums are 0 ⇒ Σᵢ ∂φᵢ/∂t = 0  (Σφᵢ conserved)
    // The bare adjacency form (Lᵢᵢ=0, Lᵢⱼ>0) is indefinite ⇒ energy can grow ⇒
    // divergence, and gives no self-relaxation (Lᵢᵢ=0) ⇒ phases saturate.
    const double Lmat[4][4] = {
        {  L01+L02+L03, -L01,        -L02,        -L03        },
        { -L01,          L01+L12+L13,-L12,        -L13        },
        { -L02,         -L12,         L02+L12+L23,-L23        },
        { -L03,         -L13,        -L23,         L03+L13+L23 },
    };
    const double eps2[4][4] = {
        { 0.0,      eps01_sq, eps02_sq, eps03_sq },
        { eps01_sq, 0.0,      eps12_sq, eps13_sq },
        { eps02_sq, eps12_sq, 0.0,      eps23_sq },
        { eps03_sq, eps13_sq, eps23_sq, 0.0      },
    };

    // Cell-centred RHS of ∂φᵢ/∂t = −Σⱼ Lᵢⱼ δF/δφⱼ  (Laplacian Lmat ⇒ Steinbach):
    //   for every j (INCLUDING the j=i self-term), add −Lmat[i][j]·δF/δφⱼ
    //   = bulk(δF/δφⱼ) + non-divergence gradient-energy residual for every (j,k).
    // The divergence (face-flux) part is added separately as divFace(Gx[i],Gy[i]).
    auto buildCellRHS = [&](int i) -> RHSExpr {
        RHSExpr e;
        for (int j = 0; j < 4; ++j) {
            const double Lij = Lmat[i][j];
            e += bulk(j, Lij);
            for (int k = 0; k < 4; ++k) {
                if (k == j) continue;
                e += gradE_nd(*phis[j], *phis[k], eps2[j][k], Lij);
            }
        }
        return e;
    };

    // -----------------------------------------------------------------------
    // 7. Equations  (Einstein sum: ∂φᵢ/∂t = −Σⱼ Lᵢⱼ δF/δφⱼ)
    //
    //   L matrix (symmetric, Lᵢᵢ=0):
    //     j=0   j=1   j=2   j=3
    // i=0  0    L01   L02   L03
    // i=1 L01    0    L12   L13
    // i=2 L02   L12    0    L23
    // i=3 L03   L13   L23    0
    // -----------------------------------------------------------------------

    // --- 7a–7d.  ∂φᵢ/∂t = −Σⱼ≠ᵢ Lᵢⱼ δF/δφⱼ   [G7, i=0..3] ----------------
    //   Each RHS = cell-centred (bulk + non-divergence gradient residual)
    //            + divFace(Gx[i], Gy[i])  for the gradient-energy divergence.
    Equation eqPhi0(phi0, "AC_phi0");
    Equation eqPhi1(phi1, "AC_phi1");
    Equation eqPhi2(phi2, "AC_phi2");
    Equation eqPhi3(phi3, "AC_phi3");
    Equation* eqPhi[4] = { &eqPhi0, &eqPhi1, &eqPhi2, &eqPhi3 };
    for (int i = 0; i < 4; ++i)
        eqPhi[i]->setRHS( buildCellRHS(i) + divFace(Gx[i], Gy[i]) );

    // --- 7e. TRANSIENT: η  (Allen-Cahn, amorphous order parameter)  [G8] ---
    //   ∂f/∂η = 30φ₀η²(1−η)²Δf^SR + 2w_η η(1−η)(1−2η) + 2w_ex η Σᵢ₌₁³ φᵢ²
    //   ∇·(∂f/∂(∇η)) = β∇²η
    //   → ∂η/∂t = −L_η [30φ₀η²(1−η)²Δf^SR + 2w_η g'(η) + 2w_ex η(φ₁²+φ₂²+φ₃²)]
    //              + L_η β ∇²η
    Equation eqEta(eta, "AC_eta");
    eqEta.setRHS(
        pw(phi0, eta, PHIX_FN (double p0, double ev) {
            double SR = 30.0 * p0 * ev * ev * (1.0 - ev) * (1.0 - ev) * dFSR;
            double dw = 2.0 * w_eta * g_prime(ev);
            return -L_eta * (SR + dw);
        })
        - mul(eta,
              (phi1 * phi1) + (phi2 * phi2) + (phi3 * phi3),
              2.0 * L_eta * w_ex)
        + L_eta * beta * lap(eta)
    );

    // --- 7g. CH: composition c  (liquid-driven minimal form)  [G9 / paper Eq.5] ----
    //   Crystals are stoichiometric ⇒ ∂fᵢ/∂c = 0, so the ONLY composition force is the
    //   liquid's:   μ = ∂f/∂c = φ₀ ∂f₀/∂c = φ₀ · dG_liq/dc / Vm_liq.
    //   Paper Eq.2 has no |∇c|² term ⇒ κ_c = 0, so μ has no Laplacian part.
    //   Two-equation idiom: μ is an auxiliary field recomputed each step (computeRHS),
    //   then c is advanced by  ∂c/∂t = ∇·(M_c∇μ) = M_c∇²μ  (constant M_c).
    //   NOTE μ→0 inside crystals (φ₀→0) ⇒ composition is simply frozen there (no
    //   partitioning at off-stoichiometry crystals).  Acceptable for Cu₅₀Zr₅₀ where the
    //   growing B2-CuZr sits at c=0.5 = alloy composition.
    Equation eqMu(mu, "CH_mu");
    eqMu.setRHS(
        pw(c, phi0, PHIX_FN (double cv, double p0) {
            return p0 * compute_dGliq_dc(cv, T) / Vm_liq;     // [J/m³]
        })
    );
    Equation eqC(c, "CH_c");
    eqC.setRHS( M_c * lap(mu) );

    // -----------------------------------------------------------------------
    // 7f. Per-step assembly of every equation's gradient-energy face flux.
    //
    //  addPairFlux accumulates  +Lij ε² ∇·(φₖ²∇φⱼ − φₖφⱼ∇φₖ)  for one (j,k)
    //  pair into (accX, accY).  On a-faces the flux component is
    //    Lij ε² [ (φₖ_f)² (∂_aφⱼ)_f − (φₖ_f)(φⱼ_f)(∂_aφₖ)_f ]
    //  using face-interpolated φ for the coefficients and face gradients for
    //  the derivatives — a standard conservative finite-volume flux.
    // -----------------------------------------------------------------------
    auto addPairFlux = [&](FaceField& accX, FaceField& accY,
                           int j, int k, double eps_jk_sq, double Lij) {
        const double w = Lij * eps_jk_sq;
        // x-faces:  t1 = (φₖ_f)²(∂ₓφⱼ)_f ,  t2 = (φₖ_f)(φⱼ_f)(∂ₓφₖ)_f
        facePWGPU(t1x, pX[k], gX[j],
                  PHIX_FN (double pk, double gj) { return pk * pk * gj; });
        facePWGPU(t2x, pX[k], pX[j], gX[k],
                  PHIX_FN (double pk, double pj, double gk) { return pk * pj * gk; });
        facePWGPU(accX, accX, t1x, t2x,
                  PHIX_FN (double g, double a, double b) { return g + w * (a - b); });
        // y-faces
        facePWGPU(t1y, pY[k], gY[j],
                  PHIX_FN (double pk, double gj) { return pk * pk * gj; });
        facePWGPU(t2y, pY[k], pY[j], gY[k],
                  PHIX_FN (double pk, double pj, double gk) { return pk * pj * gk; });
        facePWGPU(accY, accY, t1y, t2y,
                  PHIX_FN (double g, double a, double b) { return g + w * (a - b); });
    };

    auto assembleAllFlux = [&]() {
        // 1. refresh φ ghosts so faceGrad sees valid boundary values
        for (auto* bc : bcs)
            for (auto* p : phis) bc->applyOnGPU(*p);
        // 2. interp + faceGrad of every phase onto both axes' faces (shared)
        for (int m = 0; m < 4; ++m) {
            interpGPU  (*phis[m], 0, pX[m]);  faceGradGPU(*phis[m], 0, gX[m]);
            interpGPU  (*phis[m], 1, pY[m]);  faceGradGPU(*phis[m], 1, gY[m]);
        }
        // 3. per equation i: zero its accumulators, then sum its (j,k) pairs
        for (int i = 0; i < 4; ++i) {
            facePWGPU(Gx[i], Gx[i], PHIX_FN (double) { return 0.0; });
            facePWGPU(Gy[i], Gy[i], PHIX_FN (double) { return 0.0; });
            for (int j = 0; j < 4; ++j) {   // include j=i (self-term) — Laplacian L
                const double Lij = Lmat[i][j];
                for (int k = 0; k < 4; ++k) {
                    if (k == j) continue;
                    addPairFlux(Gx[i], Gy[i], j, k, eps2[j][k], Lij);
                }
            }
        }
    };

    // -----------------------------------------------------------------------
    // 8. Coupled system — SIMULTANEOUS update (all RHS from the same time
    //    level n before any field changes; the correct scheme for fully
    //    coupled multi-phase Allen-Cahn).  η joins as a cell-centred equation.
    // -----------------------------------------------------------------------
    EquationSystem sys(dt, TimeScheme::EULER);
    sys.add(eqPhi0, bcs);
    sys.add(eqPhi1, bcs);
    sys.add(eqPhi2, bcs);
    sys.add(eqPhi3, bcs);
    sys.add(eqEta,  bcs);
    sys.add(eqC,    bcs);   // composition joins the simultaneous update (μ stays auxiliary)
    sys.step = start_step;
    sys.time = start_step * dt;

    // -----------------------------------------------------------------------
    // 9. Output & time loop
    // -----------------------------------------------------------------------
    IO::OutputWriter writer(cfg["output"]);

    if (start_step == 0) {
        writer.writeFields(phi0, 0, 0.0);
        writer.writeFields(phi1, 0, 0.0);
        writer.writeFields(phi2, 0, 0.0);
        writer.writeFields(phi3, 0, 0.0);
        writer.writeFields(eta,  0, 0.0);
        writer.writeFields(c,    0, 0.0);
        std::cout << "Starting GFA-4ph simulation ("
                  << nSteps << " steps, dt=" << dt
                  << ", T=" << T << " K, c=0.5 fixed)\n";
    } else {
        std::cout << "Resuming GFA-4ph simulation from step " << start_step
                  << " (t=" << start_step * dt << "), "
                  << nSteps - start_step << " steps remaining, dt=" << dt
                  << "\n";
    }

    writer.resetTimer();

    // [0,1] projection launch config (cf. GFA verification0): all phases share
    // the same mesh / ghost / storedDims, so one set of params clamps them all.
    const dim3 clampBlk(16, 16);
    const dim3 clampGrd((nx + 15) / 16, (ny + 15) / 16);
    const int  cg  = phi0.ghost;
    const int  csx = phi0.storedDims[0];
    const int  csy = phi0.storedDims[1];

    // curand states for the per-step η noise (one per physical cell, seeded once)
    const int    nPhys   = nx * ny;
    curandState* d_states = nullptr;
    cudaMalloc(&d_states, static_cast<std::size_t>(nPhys) * sizeof(curandState));
    k_initStates<<<(nPhys + 255) / 256, 256>>>(d_states, noise_seed, nPhys);
    cudaDeviceSynchronize();

    for (int s = start_step; s < nSteps; ++s) {
        //   c is fixed at 0.5 — CH equation not solved.
        // Assemble all gradient-energy face fluxes from the time-n φ (this also
        // refreshes every φ ghost, which the cell-centred residuals need), then
        // advance the whole coupled system simultaneously (explicit Euler).
        assembleAllFlux();
        // CH: build μ^n = φ₀ ∂f₀/∂c from the time-n (c, φ₀), then refresh μ ghosts so
        // the M_c∇²μ stencil in eqC sees valid boundaries (canonical PhiX CH idiom).
        for (auto* bc : bcs) bc->applyOnGPU(c);
        eqMu.computeRHS(mu);
        for (auto* bc : bcs) bc->applyOnGPU(mu);
        sys.advance();

        // Constrain the order parameters so the stiff double-well / CALPHAD
        // terms stay bounded (explicit Euler overshoots otherwise):
        //   φ₀..φ₃ → Gibbs simplex (φᵢ≥0, Σφᵢ=1)  — restores phase competition
        //   η      → plain [0,1] clamp            — independent order parameter
        k_proj_simplex4<<<clampGrd, clampBlk>>>(
            phi0.d_curr, phi1.d_curr, phi2.d_curr, phi3.d_curr,
            nx, ny, csx, csy, cg);
        k_clamp01<<<clampGrd, clampBlk>>>(eta.d_curr, nx, ny, csx, csy, cg);

        if (writer.shouldPrint(sys.step))
            writer.printProgress(sys.step, sys.time);

        if (writer.shouldWrite(sys.step)) {
            writer.writeFields(phi0, sys.step, sys.time);
            writer.writeFields(phi1, sys.step, sys.time);
            writer.writeFields(phi2, sys.step, sys.time);
            writer.writeFields(phi3, sys.step, sys.time);
            writer.writeFields(eta,  sys.step, sys.time);
            writer.writeFields(c,    sys.step, sys.time);
        }

        // Thermal kick on η for the next step (Gaussian + clamp).  Only η is
        // perturbed; activates glass nucleation (no-op when noise_std == 0).
        if (noise_std != 0.0)
            k_noiseClamp<<<clampGrd, clampBlk>>>(
                eta.d_curr, d_states, nx, ny, csx, csy, cg,
                noise_mean, noise_std);
    }

    cudaFree(d_states);
    std::cout << "Done.\n";
    return 0;
}
