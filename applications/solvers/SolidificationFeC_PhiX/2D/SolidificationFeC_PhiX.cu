/***********************************************************************\
 *
 *  SolidificationFeC_PhiX.cu
 *
 *  PhiX-native reproduction of OpenPhase examples/SolidificationFeC.
 *
 *  OpenPhase reference:
 *    Phase 0: Melt  (liquid)
 *    Phase 1: Solid
 *    Component: C in Fe
 *    EquilibriumPartitionDiffusionBinary + DoubleObstacle
 *
 *  This solver keeps the OpenPhase input parameters and maps them onto the
 *  current PhiX Equation DSL:
 *    phi_l   -- liquid phase fraction
 *    phi_s   -- solid phase fraction
 *    c       -- carbon mole fraction
 *    T       -- uniform temperature
 *
\***********************************************************************/

#include "mesh/Mesh.h"
#include "field/ScalarField.h"
#include "field/GibbsSimplex.h"
#include "boundary/BCFactory.h"
#include "equation/Equation.h"
#include "equation/EquationSystem.h"
#include "IO/ConfigFile.h"
#include "IO/FieldIO.h"
#include "IO/OutputWriter.h"

#include <cmath>
#include <iostream>
#include <string>

using namespace PhiX;

__host__ __device__ inline double clamp01(double v)
{
    return fmax(0.0, fmin(1.0, v));
}

// Double-Obstacle obstacle potential g(φ)=φ(1-φ), g'(φ)=1-2φ
__host__ __device__ inline double g_prime(double p)
{
    return 1.0 - 2.0 * p;
}

__global__ void k_init_openphase_fec(
    double* phi_l, double* phi_s, double* c, double* T,
    int nx, int ny, int sx, int sy, int g,
    double dx, double dy,
    double c_liquid, double c_solid, double T0,
    double seed_radius, double interface_width)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= nx || iy >= ny) return;

    int idx = (ix + g) + sx * ((iy + g) + sy * g);

    double x = (ix + 0.5) * dx;
    double y = (iy + 0.5) * dy;
    double cx = 0.5 * nx * dx;
    double cy = 0.5 * ny * dy;
    double d = hypot(x - cx, y - cy);
    double w = fmax(interface_width, 1.0e-30);

    // seed_radius and interface_width are already in metres (passed as iw*dx, seed_r*dx)
    // w is the tanh half-width in metres — do NOT multiply by dx again
    double ps = 0.5 * (1.0 - tanh(M_PI * (d - seed_radius) / w));
    ps = clamp01(ps);
    phi_s[idx] = ps;
    phi_l[idx] = 1.0 - ps;
    c[idx] = (1.0 - ps) * c_liquid + ps * c_solid;
    T[idx] = T0;
}

__global__ void k_fill_uniform(double* data, int n, double value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = value;
}

__global__ void k_clamp_field(double* data, int n, double lo, double hi)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    double v = data[idx];
    if (!isfinite(v)) v = lo;
    data[idx] = fmax(lo, fmin(hi, v));
}

// ---------------------------------------------------------------------------
// Anti-trapping current (Step 8b of SolverIntro)
//
//   delta_c = -div(j_AT) * dt
//   j_AT    = -(eta/pi) * (D_l - D_s)/(D_l + D_s) * sqrt(phi_l*phi_s)
//             * (c_l - c_s) * (dphi_s/dt) * n_hat
//
// n_hat = grad(phi_s) / |grad(phi_s)|
// c_l - c_s estimated from local mixture: c = phi_l*c_l + phi_s*k*c_l
//   => c_l = c / (phi_l + k*phi_s),  k = D_sol/D_liq  (proxy for partition)
//
// div(j_AT) is computed by central differences of j_AT at neighbour cells.
// ---------------------------------------------------------------------------
__device__ static double2 d_jAT_at(
    const double* __restrict__ phi_s_curr,
    const double* __restrict__ phi_s_prev,
    const double* __restrict__ c_field,
    int base_idx, int sx,
    double D_liq, double D_sol, double eta, double dt, double dx, double dy,
    double prefac, double k_part)
{
    double ps = phi_s_curr[base_idx];
    double pl = fmax(0.0, 1.0 - ps);
    double prod = pl * ps;
    if (prod <= 0.0) return {0.0, 0.0};

    double dphi_dt = (phi_s_curr[base_idx] - phi_s_prev[base_idx]) / dt;

    double denom = pl + k_part * ps;
    if (denom < 1e-15) return {0.0, 0.0};
    double c_l = c_field[base_idx] / denom;
    double c_s = k_part * c_l;
    double dc  = c_l - c_s;

    double jmag = prefac * sqrt(prod) * dc * dphi_dt;

    // grad phi_s at base_idx (central diff, x±1 and y±sx in flat indexing)
    double gx = (phi_s_curr[base_idx + 1]  - phi_s_curr[base_idx - 1])  / (2.0 * dx);
    double gy = (phi_s_curr[base_idx + sx] - phi_s_curr[base_idx - sx]) / (2.0 * dy);
    double gn = hypot(gx, gy);
    if (gn < 1e-15) return {0.0, 0.0};

    return {jmag * gx / gn, jmag * gy / gn};
}

__global__ void k_anti_trapping(
    double*       c,
    const double* phi_s_curr,
    const double* phi_s_prev,
    int nx, int ny, int sx, int sy, int g,
    double D_liq, double D_sol, double eta, double dt, double dx, double dy)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= nx || iy >= ny) return;

    int idx = (ix + g) + sx * ((iy + g) + sy * g);

    double k_part = D_sol / D_liq;
    double prefac = -(eta / M_PI) * (D_liq - D_sol) / (D_liq + D_sol);

    // j_AT at ±x and ±y neighbours
    double2 jxp = d_jAT_at(phi_s_curr, phi_s_prev, c, idx + 1,  sx, D_liq, D_sol, eta, dt, dx, dy, prefac, k_part);
    double2 jxm = d_jAT_at(phi_s_curr, phi_s_prev, c, idx - 1,  sx, D_liq, D_sol, eta, dt, dx, dy, prefac, k_part);
    double2 jyp = d_jAT_at(phi_s_curr, phi_s_prev, c, idx + sx, sx, D_liq, D_sol, eta, dt, dx, dy, prefac, k_part);
    double2 jym = d_jAT_at(phi_s_curr, phi_s_prev, c, idx - sx, sx, D_liq, D_sol, eta, dt, dx, dy, prefac, k_part);

    double div_j = (jxp.x - jxm.x) / (2.0 * dx)
                 + (jyp.y - jym.y) / (2.0 * dy);

    c[idx] -= div_j * dt;
}

int main(int argc, char* argv[])
{
    IO::ConfigFile cfg = IO::ConfigFile::fromArgs(argc, argv);

    const int    nx = cfg["mesh"]["nx"];
    const double dx = cfg["mesh"]["dx"];
    const double x0 = cfg["mesh"]["x0"];
    const int    ny = cfg["mesh"]["ny"];
    const double dy = cfg["mesh"]["dy"];
    const double y0 = cfg["mesh"]["y0"];

    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN, nx, dx, x0, ny, dy, y0);
    mesh.print();

    const double dt     = cfg["initialize"]["dt"];
    const int    nSteps = cfg["initialize"]["nSteps"];
    const double iw     = cfg["initialize"]["interface_width_cells"];
    const double seed_r = cfg["initialize"]["solid_seed_radius_cells"];

    const double c_liquid = cfg["composition"]["C0_liquid_C"];
    const double c_solid  = cfg["composition"]["C0_solid_C"];

    const double T0           = cfg["temperature"]["T0"];
    const double dTdt         = cfg["temperature"]["DT_Dt"];
    const double Ts           = cfg["partition_diffusion"]["Ts_0_1"];
    const double ML           = cfg["partition_diffusion"]["ML_0_1"];
    const double MS           = cfg["partition_diffusion"]["ML_1_0"];
    const double D_liq        = cfg["partition_diffusion"]["DC_0"];
    const double D_sol        = cfg["partition_diffusion"]["DC_1"];
    const double entropy_sol  = cfg["partition_diffusion"]["EF_1"];

    const double sigma        = cfg["interface_properties"]["Sigma_0_1"];
    const double eps_energy   = cfg["interface_properties"]["EpsilonE_0_1"];
    const double mu_interface = cfg["interface_properties"]["Mu_0_1"];
    const double eps_mobility = cfg["interface_properties"]["EpsilonM_0_1"];

    const double c_min        = cfg["composition"]["CMIN_0_C"];
    const double c_max        = cfg["composition"]["CMAX_0_C"];
    const double phase_scale  = cfg["phix_mapping"]["phase_rhs_scale"];
    const double drive_scale  = cfg["phix_mapping"]["driving_force_scale"];
    const double aniso_cap    = cfg["phix_mapping"]["anisotropy_effective_cap"];

    // OpenPhase DrivingForce: CutOff_0_1 = 0.95 means drive is applied only where
    // phi_s ∈ (1-cutoff, cutoff) = (0.05, 0.95), i.e. inside the diffuse interface.
    const double drive_cutoff = cfg["driving_force"]["CutOff_0_1"];
    const double co           = 1.0 - drive_cutoff;  // = 0.05

    const double interface_width = iw * dx;
    // MPF Double-Obstacle two-phase (N=2) parameters.
    // Reference: Steinbach 1999, Eq.(15)-(16) with N=2:
    //   dφ_s/dt = (Mu/N) * [σ(∇²φ_s - ∇²φ_l) + σπ²/η² (φ_s - φ_l)] + Mu*(2π/Nη)*ΔG
    // N=2: (Mu/2)*σ*[∇²φ_s + π²/η² (2φ_s-1)] + Mu*(π/η)*ΔG
    //
    // In PhiX Allen-Cahn form with g'(φ)=1-2φ=-(φ_s-φ_l)/1:
    //   dφ_s/dt = phase_mobility * [kappa*∇²φ_s - barrier*g'(φ_s) + drive_coupling*ΔG]
    //
    // Matching each term (phase_mobility = Mu_eff * phase_rhs_scale):
    //
    //   SolverIntro Step 3 (N=2, expand α=liquid β=solid with φ_l=1-φ_s):
    //     ∂φ_s/∂t = (μ/2)*σ*(∇²φ_s-∇²φ_l + π²/η²*(φ_s-φ_l)) + μ*(π/η)*ΔG
    //             = μ*σ*∇²φ_s  -  μ*σ*π²/(2η²)*g'(φ_s)  +  μ*(π/η)*ΔG
    //
    //   So:  kappa         = σ               [J/m²]
    //        barrier       = σ*π²/(2*η²)    [J/m⁴]
    //        drive_coupling = π/η            [1/m]
    //        phase_rhs_scale = 1  (Mu is already in SI: m⁴/(J·s))
    //
    // Ext (diffusion-controlled) effective mobility (SolverIntro Step 2):
    //   μ_eff = 8(D_l + D_s) / (|mL| · η · |ΔS| · ΔC_eq)  ~1000× < Mu_0_1
    const double dC_eq  = std::abs(c_liquid - c_solid);
    const double mu_eff = 8.0 * (D_liq + D_sol)
                          / (std::abs(ML) * interface_width * std::abs(entropy_sol) * dC_eq);
    const double mu_used = std::min(mu_eff, mu_interface);
    std::cout << "Ext mu_eff = " << mu_eff << " m^4/(J*s)  (Mu_0_1 = " << mu_interface << ")\n";
    std::cout << "Using phase_mobility = " << mu_used * phase_scale << " m^4/(J*s)\n";

    const double kappa          = sigma;                                         // σ [J/m²]
    const double barrier        = M_PI * M_PI * sigma / (2.0 * interface_width * interface_width); // σπ²/(2η²)
    const double drive_coupling = M_PI / interface_width;                        // π/η
    const double phase_mobility = mu_used * phase_scale;

    ScalarField phi_l(mesh, "phi_l", 1);
    ScalarField phi_s(mesh, "phi_s", 1);
    ScalarField c    (mesh, "c",     1);
    ScalarField D    (mesh, "D",     1);
    ScalarField T    (mesh, "T",     1);
    ScalarField drive(mesh, "drive", 1);
    ScalarField A    (mesh, "A",     1);  // a_σ(θ)² anisotropy factor for interface energy
    ScalarField B    (mesh, "B",     1);  // torque coefficient a_σ(θ)·a_σ'(θ)
    // drive_mask: 1 where phi_s ∈ (co, 1-co), 0 otherwise (matches OpenPhase CutOff_0_1)
    ScalarField mask (mesh, "mask",  1);

    phi_l.fill(1.0); phi_s.fill(0.0); c.fill(c_liquid);
    D.fill(0.0); T.fill(T0); drive.fill(0.0); A.fill(1.0); B.fill(0.0); mask.fill(0.0);

    const std::string start_from = cfg["initialize"]["start_from"];
    const int start_step = IO::resolveStartStep(start_from, "phi_l");

    auto allocUp = [](ScalarField& f){ f.allocDevice(); f.uploadAllToDevice(); };
    allocUp(phi_l); allocUp(phi_s); allocUp(c);
    allocUp(D); allocUp(T); allocUp(drive); allocUp(A); allocUp(B); allocUp(mask);

    if (start_step == 0) {
        const dim3 blk(16, 16);
        const dim3 grd((nx + blk.x - 1) / blk.x, (ny + blk.y - 1) / blk.y);
        k_init_openphase_fec<<<grd, blk>>>(
            phi_l.d_curr, phi_s.d_curr, c.d_curr, T.d_curr,
            nx, ny, phi_l.storedDims[0], phi_l.storedDims[1], phi_l.ghost,
            dx, dy, c_liquid, c_solid, T0, seed_r * dx, interface_width);
        cudaDeviceSynchronize();
        PhiX::gibbsSimplexOnGPU({&phi_l, &phi_s});
    } else {
        IO::initField(phi_l, start_step);
        IO::initField(phi_s, start_step);
        IO::initField(c,     start_step);
        phi_l.uploadAllToDevice();
        phi_s.uploadAllToDevice();
        c.uploadAllToDevice();
        double t_now = T0 + dTdt * start_step * dt;
        k_fill_uniform<<<(static_cast<int>(T.storedSize) + 255) / 256, 256>>>(
            T.d_curr, static_cast<int>(T.storedSize), t_now);
    }

    auto bcSet = buildBCs(cfg["boundary_conditions"]);
    auto& bcs = bcSet.ptrs;

    // A = a_σ(θ)²,  a_σ(θ) = 1 + eps_energy*cos(4θ)  (interface-energy anisotropy)
    Equation eqA(A, "cubic_anisotropy");
    eqA.setRHS(
        pw(grad(phi_s, 0), grad(phi_s, 1), PHIX_FN(double gx, double gy) {
            double theta = atan2(gy, gx);
            double a = 1.0 + eps_energy * cos(4.0 * theta);
            return a * a;
        })
    );

    // B = a_σ(θ)·a_σ'(θ),  a_σ'(θ) = -4·eps_energy·sin(4θ)  (torque term)
    Equation eqB(B, "torque_coeff");
    eqB.setRHS(
        pw(grad(phi_s, 0), grad(phi_s, 1), PHIX_FN(double gx, double gy) {
            double theta = atan2(gy, gx);
            double a  =  1.0 + eps_energy * cos(4.0 * theta);
            double da = -4.0 * eps_energy * sin(4.0 * theta);
            return a * da;
        })
    );

    // Driving force: liquidus undercooling only (Step 4 of SolverIntro).
    //
    // SolverIntro uses C^α (liquid composition) and C^β (solid composition) separately.
    // In a single mixture-field model we cannot split them, so we use the standard
    // thin-interface approximation: ΔG = |ΔS| * (T_liq(c) - T)
    // where T_liq(c) = Ts + ML*c is the liquidus temperature at local concentration c.
    //
    // WHY NOT the two-term formula: the T_sol term uses MS*c with the LIQUID concentration,
    // giving T_sol(c_liq) ≈ 1621 K << T = 1690 K, which makes that term NEGATIVE and
    // can flip the total driving force sign, causing the solid to melt.
    // ΔS = EF_1 - EF_0 = entropy_sol - 0 = entropy_sol < 0, so -entropy_sol > 0.
    Equation eqDrive(drive, "partition_driving_force");
    eqDrive.setRHS(
        pw(c, T, PHIX_FN(double cv, double Tv) {
            double T_liq = Ts + ML * cv;        // liquidus temperature at local c
            double undercooling = T_liq - Tv;   // >0 when below liquidus → solidification
            return drive_scale * (-entropy_sol) * undercooling;
        })
    );

    // Diffusivity: mixture rule D = Σ φ_α D_α  (Step 8a)
    Equation eqD(D, "carbon_diffusivity");
    eqD.setRHS(
        pw(phi_l, phi_s, PHIX_FN(double pl, double ps) {
            return D_liq * pl + D_sol * ps;
        })
    );

    // Driving-force mask: matches OpenPhase CutOff_0_1 = 0.95
    // Active (=1) only in the diffuse interface where phi_s ∈ (co, 1-co).
    // This replaces the old phi_s*phi_l weight (peak 0.25) which under-drove
    // solidification by ~4× compared to OpenPhase.
    Equation eqMask(mask, "drive_mask");
    eqMask.setRHS(
        pw(phi_s, phi_l, PHIX_FN(double ps, double pl) {
            return (ps > co && pl > co) ? 1.0 : 0.0;
        })
    );

    // Solute diffusion: ∂c/∂t = ∇·(D ∇c)  (Fick, Step 8a)
    Equation eqC(c, "carbon");
    eqC.setRHS(
          D * lap(c)
        + grad(D, 0) * grad(c, 0)
        + grad(D, 1) * grad(c, 1)
    );

    // g'(φ) for Double-Obstacle barrier term
    auto gp_l = pw(phi_l, PHIX_FN(double p) { return g_prime(p); });
    auto gp_s = pw(phi_s, PHIX_FN(double p) { return g_prime(p); });

    // Mobility anisotropy factor: μ(θ) = μ₀·(1 + eps_mobility·cos(4θ))²
    // Applied pointwise via mu_aniso_l/s computed from grad(phi_s)
    // Here we fold it into the phase_mobility constant (isotropic approximation for mobility)
    // and keep only energy anisotropy in A/B for the Laplacian/torque terms.
    // Full mobility anisotropy would require separate M_aniso field; deferred to future work.

    // Phase equations (MPF Double-Obstacle, Step 3 + Step 6):
    // Driving force is applied only inside the diffuse interface via `mask`
    // (= 1 where phi_s ∈ (co, 1-co), = 0 in bulk), matching OpenPhase CutOff_0_1.
    // Previously phi_s*phi_l was used (peak weight 0.25 →4× weaker than OpenPhase).
    //
    // The barrier g'(phi) for Double-Obstacle is: g'(phi) = 1 - 2*phi
    // which equals -(phi_s - phi_l). Combined with phase_mobility * barrier:
    //   phase_mobility * (-barrier * g'(phi_s)) = phase_mobility * barrier * (2*phi_s - 1)
    Equation eqPhiL(phi_l, "phi_l");
    eqPhiL.setRHS(
        phase_mobility * (
            kappa * (
                A * lap(phi_l)
                + grad(A, 0) * grad(phi_l, 0) + grad(A, 1) * grad(phi_l, 1)
                + grad(B, 1) * grad(phi_l, 0) - grad(B, 0) * grad(phi_l, 1)
            )
            - barrier * gp_l
            - drive_coupling * mask * drive
        )
    );

    Equation eqPhiS(phi_s, "phi_s");
    eqPhiS.setRHS(
        phase_mobility * (
            kappa * (
                A * lap(phi_s)
                + grad(A, 0) * grad(phi_s, 0) + grad(A, 1) * grad(phi_s, 1)
                + grad(B, 1) * grad(phi_s, 0) - grad(B, 0) * grad(phi_s, 1)
            )
            - barrier * gp_s
            + drive_coupling * mask * drive
        )
    );

    EquationSystem phaseSys(dt);
    phaseSys.add(eqPhiL, bcs);
    phaseSys.add(eqPhiS, bcs);
    phaseSys.step = start_step;
    phaseSys.time = start_step * dt;
    eqC.step  = start_step;
    eqC.time  = start_step * dt;

    IO::OutputWriter writer(cfg["output"]);
    if (start_step == 0) {
        writer.writeFields(phi_l, 0, 0.0);
        writer.writeFields(phi_s, 0, 0.0);
        writer.writeFields(c,     0, 0.0);
        writer.writeFields(T,     0, 0.0);
        std::cout << "Starting SolidificationFeC_PhiX from OpenPhase parameters ("
                  << nSteps << " steps, dt=" << dt << ")\n";
        std::cout << "  OpenPhase Mu_0_1=" << mu_interface
                  << " mapped to PhiX phase mobility=" << phase_mobility
                  << " using phase_rhs_scale=" << phase_scale << "\n";
    } else {
        std::cout << "Resuming SolidificationFeC_PhiX from step "
                  << start_step << "\n";
    }

    writer.resetTimer();
    for (int s = start_step; s < nSteps; ++s) {
        double time = s * dt;
        double t_now = T0 + dTdt * time;
        int stored_n = static_cast<int>(T.storedSize);
        k_fill_uniform<<<(stored_n + 255) / 256, 256>>>(T.d_curr, stored_n, t_now);

        eqA.advanceSteady(bcs, &phi_s);
        eqB.advanceSteady(bcs, &phi_s);
        eqMask.advanceSteady(bcs);        // mask = cutoff indicator(phi_s, phi_l)
        eqDrive.advanceSteady(bcs, &c);   // T is uniform — only c needs BCs here
        phaseSys.advance();
        k_clamp_field<<<(static_cast<int>(phi_l.storedSize) + 255) / 256, 256>>>(
            phi_l.d_curr, static_cast<int>(phi_l.storedSize), 0.0, 1.0);
        k_clamp_field<<<(static_cast<int>(phi_s.storedSize) + 255) / 256, 256>>>(
            phi_s.d_curr, static_cast<int>(phi_s.storedSize), 0.0, 1.0);
        PhiX::gibbsSimplexOnGPU({&phi_l, &phi_s});

        eqD.advanceSteady(bcs);   // D = D(phi_l, phi_s), BCs applied to D itself
        eqC.advanceTransient(bcs, dt, &c);

        // Anti-trapping current (Step 8b): applied after Fick diffusion.
        // phi_s.d_prev holds the pre-advance phi_s (set by advanceTimeLevelGPU).
        if (phi_s.d_prev != nullptr) {
            const dim3 blk_at(16, 16);
            const dim3 grd_at((nx + 15) / 16, (ny + 15) / 16);
            k_anti_trapping<<<grd_at, blk_at>>>(
                c.d_curr, phi_s.d_curr, phi_s.d_prev,
                nx, ny,
                phi_s.storedDims[0], phi_s.storedDims[1], phi_s.ghost,
                D_liq, D_sol, interface_width, dt, dx, dy);
            cudaDeviceSynchronize();
        }

        k_clamp_field<<<(static_cast<int>(c.storedSize) + 255) / 256, 256>>>(
            c.d_curr, static_cast<int>(c.storedSize), c_min, c_max);

        int step = s + 1;
        double out_time = step * dt;
        if (writer.shouldPrint(step)) {
            writer.printProgress(step, out_time);
            std::cout << "  T=" << (T0 + dTdt * out_time) << " K\n";
        }
        if (writer.shouldWrite(step)) {
            writer.writeFields(phi_l, step, out_time);
            writer.writeFields(phi_s, step, out_time);
            writer.writeFields(c,     step, out_time);
            writer.writeFields(T,     step, out_time);
        }
    }

    std::cout << "Done.\n";
    return 0;
}
