/***********************************************************************\
 *
 *  GFA verification0 — η-only Allen-Cahn solver (2D)
 *
 *  Purpose
 *  -------
 *  Verification 4 (validation plan §8): reproduce the structural
 *  relaxation morphology trend from Wang & Napolitano Fig. 1.
 *
 *  All crystalline phase-fields are fixed at  φ₀=1, φ₁=φ₂=φ₃=0,
 *  so only the glass-order parameter η evolves.
 *
 *  Governing equation
 *  -------------------
 *      ∂η/∂t = −L_η [ 30 η²(1−η)² Δf^SR  +  2 w_η η(1−η)(1−2η) ]
 *              + L_η β ∇²η
 *
 *  Δf^SR is supplied directly from the config file as a constant [J/m³].
 *  For glass-stable conditions (η=1 more stable) Δf^SR must be < 0.
 *
 *  Three paper cases (validation plan §8.2)
 *  -----------------------------------------
 *  case 1 : β=4e-11 J/m,  w_η=4e8 J/m³  (base parameters)
 *  case 2 : β=4e-12 J/m,  w_η=4e7 J/m³  (one order lower — finer structure)
 *  case 3 : w_η→0                         (near-zero barrier — spinodal-like)
 *
 *  Initialisation
 *  ---------------
 *  η = init_mean ± noise_amp * U[−1,1], clamped to [0,1].
 *  Default: init_mean=0.5, noise_amp=0.05  (perturbed midpoint of the
 *  double-well, allows both liquid and glass domains to develop).
 *
 \**********************************************************************/

#include "mesh/Mesh.h"
#include "field/ScalarField.h"
#include "boundary/BCFactory.h"
#include "equation/Equation.h"
#include "IO/ConfigFile.h"
#include "IO/FieldIO.h"
#include "IO/OutputWriter.h"

#include <curand_kernel.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

// ===========================================================================
// CUDA kernels: per-step noise injection + [0,1] clamp on d_curr
// ===========================================================================

// Initialise one curandStateXORWOW per physical cell.
__global__ void k_initStates(curandState* states,
                              unsigned long long seed, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    curand_init(seed, tid, 0, &states[tid]);
}

// Clamp each physical cell to [0,1] (no noise).
__global__ void k_clamp(double* d_curr,
                         int nx, int ny,
                         int sx, int sy, int ghost)
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

// Add Gaussian noise N(noise_mean, noise_std²) to each physical cell, then clamp.
// Physical cell (ix, iy) maps to stored index:
//   (ix + ghost) + sx * ((iy + ghost) + sy * ghost)
__global__ void k_noiseClamp(double*      d_curr,
                              curandState* states,
                              int nx, int ny,
                              int sx, int sy, int ghost,
                              double noise_mean, double noise_std)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= nx || iy >= ny) return;

    int tid = iy * nx + ix;
    // N(noise_mean, noise_std²)
    double noise = noise_mean + noise_std * curand_normal_double(&states[tid]);

    int idx = (ix + ghost) + sx * ((iy + ghost) + sy * ghost);
    double v = d_curr[idx] + noise;
    if (v < 0.0) v = 0.0;
    if (v > 1.0) v = 1.0;
    d_curr[idx] = v;
}

// ===========================================================================
// Switching / barrier functions
// ===========================================================================
__host__ __device__ inline double h_func(double x) {
    return x * x * x * (10.0 - 15.0 * x + 6.0 * x * x);
}
__host__ __device__ inline double g_prime(double x) {
    // d/dx [x²(1−x)²] = 2x(1−x)(1−2x)
    return 2.0 * x * (1.0 - x) * (1.0 - 2.0 * x);
}

// ===========================================================================
// Free energy integral (host side, uses eta.curr after download)
//
//   F = ∫ [ h(η)·ΔfSR  +  w_η η²(1−η)²  +  β/2 |∇η|² ] dV
//
// Gradient computed with 2nd-order central differences + periodic wrap.
// ===========================================================================
static double computeFreeEnergy(const PhiX::ScalarField& eta,
                                 int nx, int ny, double dx, double dy,
                                 double dFSR, double w_eta, double beta)
{
    const int g  = eta.ghost;
    const int sx = eta.storedDims[0];
    const int sy = eta.storedDims[1];

    // flat index for physical cell (i,j), k=0
    auto sidx = [&](int i, int j) -> int {
        return (i + g) + sx * ((j + g) + sy * g);
    };
    // periodic wrap
    auto wi = [nx](int i) { return ((i % nx) + nx) % nx; };
    auto wj = [ny](int j) { return ((j % ny) + ny) % ny; };

    const double dV = dx * dy;   // 2D volume element (unit depth)
    double F = 0.0;

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const double ev = eta.curr[sidx(i, j)];

            // switching function h(η) = η³(10−15η+6η²)
            const double h  = ev * ev * ev * (10.0 - 15.0 * ev + 6.0 * ev * ev);
            // double-well  w_η η²(1−η)²
            const double dw = w_eta * ev * ev * (1.0 - ev) * (1.0 - ev);

            // gradient (central differences, periodic)
            const double dedx = (eta.curr[sidx(wi(i+1), j)] -
                                  eta.curr[sidx(wi(i-1), j)]) / (2.0 * dx);
            const double dedy = (eta.curr[sidx(i, wj(j+1))] -
                                  eta.curr[sidx(i, wj(j-1))]) / (2.0 * dy);

            F += (h * dFSR + dw + 0.5 * beta * (dedx*dedx + dedy*dedy)) * dV;
        }
    }
    return F;
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
    // 3. Physical parameters (all read from config — no hardcoded values)
    // -----------------------------------------------------------------------
    const double dFSR      = cfg["constants"]["delta_f_SR"];  // Δf^SR [J/m³], < 0 for glass-stable
    const double beta      = cfg["constants"]["beta"];        // gradient energy coefficient [J/m]
    const double w_eta     = cfg["constants"]["w_eta"];       // double-well barrier [J/m³]
    const double L_eta     = cfg["constants"]["L_eta"];       // Allen-Cahn mobility [m³/(J·s)]
    const double noise_mean = cfg["constants"]["noise_mean"];  // per-step Gaussian noise mean
    const double noise_std  = cfg["constants"]["noise_std"];   // per-step Gaussian noise std dev
    const unsigned long long noise_seed =
        cfg["constants"].count("noise_seed")
            ? (unsigned long long)cfg["constants"]["noise_seed"]
            : 42ULL;

    std::cout << "=== GFA verification0 — η-only (2D) ===\n"
              << "  Δf^SR = " << dFSR << " J/m³"
              << "  (" << (dFSR < 0.0 ? "glass stable" : "liquid stable") << ")\n"
              << "  β = " << beta << " J/m,  w_η = " << w_eta
              << " J/m³,  L_η = " << L_eta << " m³/(J·s)\n"
              << "  δ = √(β/2w_η) = " << std::sqrt(beta / (2.0 * w_eta)) << " m\n"
              << "  σ = √(2βw_η)/6 = "
              << std::sqrt(2.0 * beta * w_eta) / 6.0 << " J/m²\n"
              << "  noise ~ N(" << noise_mean << ", " << noise_std << "²)"
              << "  seed = " << noise_seed << "\n";

    // -----------------------------------------------------------------------
    // 4. Field
    // -----------------------------------------------------------------------
    ScalarField eta(mesh, "eta", /*ghost=*/1);
    eta.fill(0.0);

    const std::string start_from = cfg["initialize"]["start_from"];
    // Use "eta" as ref-field name so warm restart scans for eta_*.field
    const int         start_step = IO::resolveStartStep(start_from, "eta");

    IO::initField(eta, start_step);

    eta.allocDevice();
    eta.uploadAllToDevice();

    // -----------------------------------------------------------------------
    // Allocate and initialise curand states (one per physical cell)
    // -----------------------------------------------------------------------
    const int nPhys = nx * ny;
    curandState* d_states = nullptr;
    cudaMalloc(&d_states, static_cast<std::size_t>(nPhys) * sizeof(curandState));
    {
        const int threads = 256;
        const int blocks  = (nPhys + threads - 1) / threads;
        k_initStates<<<blocks, threads>>>(d_states, noise_seed, nPhys);
        cudaDeviceSynchronize();
    }
    const dim3 blk2(16, 16);
    const dim3 grd2((nx + 15) / 16, (ny + 15) / 16);
    const int  g  = eta.ghost;
    const int  sx = eta.storedDims[0];
    const int  sy = eta.storedDims[1];

    // -----------------------------------------------------------------------
    // 5. Boundary conditions
    // -----------------------------------------------------------------------
    auto  bcSet = buildBCs(cfg["boundary_conditions"]);
    auto& bcs   = bcSet.ptrs;

    // -----------------------------------------------------------------------
    // 6. η equation  (Allen-Cahn, validation plan §4)
    //
    //   ∂η/∂t = −L_η [ 30 η²(1−η)² Δf^SR  +  2 w_η g'(η) ]  +  L_η β ∇²η
    // -----------------------------------------------------------------------
    Equation eqEta(eta, "AC_eta");
    eqEta.setRHS(
        pw(eta, PHIX_FN(double ev) {
            const double SR = 30.0 * ev * ev * (1.0 - ev) * (1.0 - ev) * dFSR;
            const double dw = 2.0 * w_eta * g_prime(ev);
            return -L_eta * (SR + dw);
        })
        + L_eta * beta * lap(eta)
    );

    eqEta.step = start_step;
    eqEta.time = start_step * dt;

    // -----------------------------------------------------------------------
    // 7. Output & time loop
    // -----------------------------------------------------------------------
    IO::OutputWriter writer(cfg["output"]);

    if (start_step == 0) {
        writer.writeFields(eta, 0, 0.0);
        std::cout << "Starting η-only simulation ("
                  << nSteps << " steps, dt=" << dt << ")\n";
    } else {
        std::cout << "Resuming η-only simulation from step " << start_step
                  << " (t=" << start_step * dt << "), "
                  << nSteps - start_step << " steps remaining, dt=" << dt << "\n";
    }

    // -----------------------------------------------------------------------
    // Free energy output file
    // -----------------------------------------------------------------------
    std::ofstream fout("output/free_energy.dat",
                       start_step == 0 ? std::ios::trunc : std::ios::app);
    if (!fout)
        std::cerr << "Warning: cannot open output/free_energy.dat\n";
    fout << std::scientific << std::setprecision(12);
    if (start_step == 0)
        fout << "# step  time[s]  free_energy[J]\n";

    // Write F at step 0 (cold start)
    if (start_step == 0) {
        const double F0 = computeFreeEnergy(eta, nx, ny, dx, dy,
                                             dFSR, w_eta, beta);
        fout << 0 << "  " << 0.0 << "  " << F0 << "\n";
        fout.flush();
    }

    writer.resetTimer();

    for (int s = start_step; s < nSteps; ++s) {
        eqEta.advanceTransient(bcs, dt, &eta);

        // 1. 截断：将 advance 后的 η 压回 [0,1]
        k_clamp<<<grd2, blk2>>>(eta.d_curr, nx, ny, sx, sy, g);

        // 2. 输出（写出截断后的干净场）
        if (writer.shouldPrint(eqEta.step))
            writer.printProgress(eqEta.step, eqEta.time);

        if (writer.shouldWrite(eqEta.step)) {
            writer.writeFields(eta, eqEta.step, eqEta.time);
            // 计算并写出自由能（writeFields 已将 d_curr 下载到 curr）
            const double F = computeFreeEnergy(eta, nx, ny, dx, dy,
                                               dFSR, w_eta, beta);
            fout << eqEta.step << "  " << eqEta.time << "  " << F << "\n";
            fout.flush();
        }

        // 3. 加噪声（为下一步提供随机扰动，并再次截断）
        k_noiseClamp<<<grd2, blk2>>>(eta.d_curr, d_states,
                                      nx, ny, sx, sy, g, noise_mean, noise_std);
    }

    cudaFree(d_states);
    std::cout << "Done.\n";
    return 0;
}
