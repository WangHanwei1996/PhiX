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
#include "IO/ConfigFile.h"
#include "IO/FieldIO.h"
#include "IO/OutputWriter.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

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

// Double-well barrier heights [J/m³]
static constexpr double w01      = 4.2e8;
static constexpr double w02      = 4.1e8;
static constexpr double w03      = 4.7e8;
static constexpr double w12      = 1.0e8;
static constexpr double w13      = 1.0e8;
static constexpr double w23      = 1.0e8;
static constexpr double w_eta    = 2.5e7;
static constexpr double w_ex     = 2.0e9;

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
    ScalarField c   (mesh, "c",    /*ghost=*/1);  // fixed at 0.5 — CH not solved yet

    phi0.fill(0); phi1.fill(0); phi2.fill(0); phi3.fill(0);
    eta .fill(0);
    c   .fill(0.5);  // fixed composition: Zr mole fraction = 0.5

    const std::string start_from = cfg["initialize"]["start_from"];
    const int         start_step = IO::resolveStartStep(start_from);

    IO::initField(phi0, start_step);
    IO::initField(phi1, start_step);
    IO::initField(phi2, start_step);
    IO::initField(phi3, start_step);
    IO::initField(eta,  start_step);
    // c is fixed at 0.5 — not loaded from file

    auto allocUp = [](ScalarField& f){ f.allocDevice(); f.uploadAllToDevice(); };
    allocUp(phi0); allocUp(phi1); allocUp(phi2); allocUp(phi3);
    allocUp(eta);  allocUp(c);

    // -----------------------------------------------------------------------
    // 5. Boundary conditions
    // -----------------------------------------------------------------------
    auto  bcSet = buildBCs(cfg["boundary_conditions"]);
    auto& bcs   = bcSet.ptrs;

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

    // Gradient-energy part of −Lij·δF/δφⱼ for one (φⱼ, φₖ) pair.
    // Returns Lij·εⱼₖ²·[−2φⱼ|∇φₖ|²+2φₖ(∇φⱼ·∇φₖ)+φₖ²∇²φⱼ−φⱼφₖ∇²φₖ]
    auto gradE = [&](const ScalarField& phi_j, const ScalarField& phi_k,
                     double eps_jk_sq, double Lij) -> RHSExpr
    {
        return
            - mul(phi_j, grad_dot(phi_k, phi_k), 2.0 * Lij * eps_jk_sq)
            + mul(phi_k, grad_dot(phi_j, phi_k), 2.0 * Lij * eps_jk_sq)
            + mul(phi_k * phi_k, lap(phi_j),     Lij * eps_jk_sq)
            - mul(phi_j * phi_k, lap(phi_k),     Lij * eps_jk_sq);
    };

    // −Lij · δF/δφ₀  (G11 + G15, i=0)
    auto dF_phi0 = [&](double Lij) -> RHSExpr {
        return
            pw(c, eta, PHIX_FN (double cv, double ev) {
                return -Lij * (compute_Gliq(cv, T) / Vm_liq + h_func(ev) * dFSR);
            })
            - mul(phi0,
                  w01*(phi1*phi1) + w02*(phi2*phi2) + w03*(phi3*phi3),
                  2.0 * Lij)
            + gradE(phi0, phi1, eps01_sq, Lij)
            + gradE(phi0, phi2, eps02_sq, Lij)
            + gradE(phi0, phi3, eps03_sq, Lij);
    };

    // −Lij · δF/δφ₁  (G12 + G15, s=1)
    auto dF_phi1 = [&](double Lij) -> RHSExpr {
        return
            pw(phi1, PHIX_FN(double) { return -Lij * f1_val; })
            - mul(phi1,
                  w01*(phi0*phi0) + w12*(phi2*phi2) + w13*(phi3*phi3),
                  2.0 * Lij)
            - mul(eta*eta, phi1, 2.0 * Lij * w_ex)
            + gradE(phi1, phi0, eps01_sq, Lij)
            + gradE(phi1, phi2, eps12_sq, Lij)
            + gradE(phi1, phi3, eps13_sq, Lij);
    };

    // −Lij · δF/δφ₂  (G12 + G15, s=2)
    auto dF_phi2 = [&](double Lij) -> RHSExpr {
        return
            pw(phi2, PHIX_FN(double) { return -Lij * f2_val; })
            - mul(phi2,
                  w02*(phi0*phi0) + w12*(phi1*phi1) + w23*(phi3*phi3),
                  2.0 * Lij)
            - mul(eta*eta, phi2, 2.0 * Lij * w_ex)
            + gradE(phi2, phi0, eps02_sq, Lij)
            + gradE(phi2, phi1, eps12_sq, Lij)
            + gradE(phi2, phi3, eps23_sq, Lij);
    };

    // −Lij · δF/δφ₃  (G12 + G15, s=3)
    auto dF_phi3 = [&](double Lij) -> RHSExpr {
        return
            pw(phi3, PHIX_FN(double) { return -Lij * f3_val; })
            - mul(phi3,
                  w03*(phi0*phi0) + w13*(phi1*phi1) + w23*(phi2*phi2),
                  2.0 * Lij)
            - mul(eta*eta, phi3, 2.0 * Lij * w_ex)
            + gradE(phi3, phi0, eps03_sq, Lij)
            + gradE(phi3, phi1, eps13_sq, Lij)
            + gradE(phi3, phi2, eps23_sq, Lij);
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

    // --- 7a. ∂φ₀/∂t = −L₀₁ δF/δφ₁ − L₀₂ δF/δφ₂ − L₀₃ δF/δφ₃  [G7, i=0] -
    Equation eqPhi0(phi0, "AC_phi0");
    eqPhi0.setRHS(
        dF_phi1(L01) + dF_phi2(L02) + dF_phi3(L03)
    );

    // --- 7b. ∂φ₁/∂t = −L₀₁ δF/δφ₀ − L₁₂ δF/δφ₂ − L₁₃ δF/δφ₃  [G7, i=1] -
    Equation eqPhi1(phi1, "AC_phi1");
    eqPhi1.setRHS(
        dF_phi0(L01) + dF_phi2(L12) + dF_phi3(L13)
    );

    // --- 7c. ∂φ₂/∂t = −L₀₂ δF/δφ₀ − L₁₂ δF/δφ₁ − L₂₃ δF/δφ₃  [G7, i=2] -
    Equation eqPhi2(phi2, "AC_phi2");
    eqPhi2.setRHS(
        dF_phi0(L02) + dF_phi1(L12) + dF_phi3(L23)
    );

    // --- 7d. ∂φ₃/∂t = −L₀₃ δF/δφ₀ − L₁₃ δF/δφ₁ − L₂₃ δF/δφ₂  [G7, i=3] -
    Equation eqPhi3(phi3, "AC_phi3");
    eqPhi3.setRHS(
        dF_phi0(L03) + dF_phi1(L13) + dF_phi2(L23)
    );

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

    // -----------------------------------------------------------------------
    // 8. Time-step counter (track via φ₀ equation)
    // -----------------------------------------------------------------------
    eqPhi0.step = start_step;
    eqPhi0.time = start_step * dt;

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

    for (int s = start_step; s < nSteps; ++s) {
        // Operator-split time advance (explicit Euler):
        //   c is fixed at 0.5 — CH equation not solved
        eqPhi0.advanceTransient (bcs, dt, &phi0);
        eqPhi1.advanceTransient (bcs, dt, &phi1);
        eqPhi2.advanceTransient (bcs, dt, &phi2);
        eqPhi3.advanceTransient (bcs, dt, &phi3);
        eqEta .advanceTransient (bcs, dt, &eta);

        if (writer.shouldPrint(eqPhi0.step))
            writer.printProgress(eqPhi0.step, eqPhi0.time);

        if (writer.shouldWrite(eqPhi0.step)) {
            writer.writeFields(phi0, eqPhi0.step, eqPhi0.time);
            writer.writeFields(phi1, eqPhi0.step, eqPhi0.time);
            writer.writeFields(phi2, eqPhi0.step, eqPhi0.time);
            writer.writeFields(phi3, eqPhi0.step, eqPhi0.time);
            writer.writeFields(eta,  eqPhi0.step, eqPhi0.time);
        }
    }

    std::cout << "Done.\n";
    return 0;
}
