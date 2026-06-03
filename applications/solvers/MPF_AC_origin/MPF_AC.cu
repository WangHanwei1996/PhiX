/***********************************************************************\
 *
 *  MPF_AC.cu — Multi-Phase Field Allen-Cahn solver (2D)
 *
 *  Benchmark: static triple junction with equal dihedral angles (120°)
 *
 *  Three-phase system: phi0, phi_a, phi_b   (sum constraint: phi0+phi_a+phi_b=1)
 *
 *  Free energy (obstacle potential + grad1 gradient energy):
 *    f_grad = Σ_{i<j} κ_ij ∇φ_i · ∇φ_j
 *    f_dw   = Σ_{i<j} Ω_ij φ_i φ_j
 *
 *  Functional derivative:
 *    δF/δφ_i = Σ_{j≠i} (κ_ij ∇²φ_j + Ω_ij φ_j)
 *
 *  Pairwise Allen-Cahn (N=3 projection):
 *    Δφ_i = 2·δF/δφ_i − Σ_{j≠i} δF/δφ_j
 *    φ_i^{n+1} = φ_i^n − dt·M₀·Δφ_i
 *
 *  Parameter relations:
 *    κ_ij = ℓ·γ_ij          (ℓ = diffuse interface width)
 *    Ω_ij = K·γ_ij / ℓ     (K = 16/π²  for obstacle potential)
 *
 *  Boundary conditions (static triple junction benchmark, matching MATLAB ref):
 *    Top  wall (y=H): Neumann (zero normal gradient)
 *    Bottom wall (y=0): Neumann (zero normal gradient)
 *    Left  wall (x=0): non-uniform Dirichlet — upper ny_top rows: phi0=1;
 *                       lower rows: phi_a=1
 *    Right wall (x=W): non-uniform Dirichlet — upper ny_top rows: phi0=1;
 *                       lower rows: phi_b=1
 *
 *  Reference: doc/static_triple_junction_pairwise_double_well_v2.md
 *
\***********************************************************************/

#include "mesh/Mesh.h"
#include "field/ScalarField.h"
#include "boundary/NoFluxBC.h"
#include "IO/ConfigFile.h"
#include "IO/FieldIO.h"
#include "IO/OutputWriter.h"
#include "field/GibbsSimplex.h"

#include <cmath>
#include <iostream>
#include <string>

// ===========================================================================
// Kernel: Initialize fields for static triple-junction IC (matching MATLAB ref)
//   Upper ny_top rows (iy >= ny - ny_top): phi0 = 1
//   Lower rows, left  half (ix < nx/2)  : phi_a = 1
//   Lower rows, right half (ix >= nx/2) : phi_b = 1
// ===========================================================================
__global__ void k_init_fields(
    double* d_phi0, double* d_phi_a, double* d_phi_b,
    int nx, int ny, int ny_top, int sx, int sy, int g)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= nx || iy >= ny) return;

    int idx = (ix + g) + sx * ((iy + g) + sy * g);

    if (iy >= ny - ny_top) {
        // Top region: phi0 = 1
        d_phi0[idx] = 1.0;  d_phi_a[idx] = 0.0;  d_phi_b[idx] = 0.0;
    } else if (ix < nx / 2) {
        // Bottom-left: phi_a = 1
        d_phi0[idx] = 0.0;  d_phi_a[idx] = 1.0;  d_phi_b[idx] = 0.0;
    } else {
        // Bottom-right: phi_b = 1
        d_phi0[idx] = 0.0;  d_phi_a[idx] = 0.0;  d_phi_b[idx] = 1.0;
    }
}


// ===========================================================================
// Kernel: Compute functional derivatives δF/δφ_i  (obstacle + grad1).
//
//   δF/δφ_0 = κ_0a·∇²φ_a + κ_0b·∇²φ_b + Ω_0a·φ_a + Ω_0b·φ_b
//   δF/δφ_α = κ_0a·∇²φ_0 + κ_ab·∇²φ_b + Ω_0a·φ_0 + Ω_ab·φ_b
//   δF/δφ_β = κ_0b·∇²φ_0 + κ_ab·∇²φ_a + Ω_0b·φ_0 + Ω_ab·φ_a
//
//   Results are stored in d_mu0 / d_mu_a / d_mu_b (re-used as scratch).
//   Ghost cells must be filled (BCs applied) before calling this kernel.
// ===========================================================================
__global__ void k_compute_mu(
    double* d_mu0, double* d_mu_a, double* d_mu_b,
    const double* d_phi0, const double* d_phi_a, const double* d_phi_b,
    int nx, int ny, int sx, int sy, int g,
    double invdx2, double invdy2,
    double kappa_0a, double kappa_0b, double kappa_ab,
    double Omega_0a, double Omega_0b, double Omega_ab)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= nx || iy >= ny) return;

    // Flat index for (ix+di, iy+dj)
    auto IDX = [sx, sy, g] __device__ (int i, int j) {
        return (i + g) + sx * ((j + g) + sy * g);
    };

    int c  = IDX(ix,     iy    );
    int xp = IDX(ix + 1, iy    );
    int xm = IDX(ix - 1, iy    );
    int yp = IDX(ix,     iy + 1);
    int ym = IDX(ix,     iy - 1);

    double p0 = d_phi0[c];
    double pa = d_phi_a[c];
    double pb = d_phi_b[c];

    // --- Laplacians ---
    double lap0 = (d_phi0[xp] + d_phi0[xm] - 2.0 * p0) * invdx2
                + (d_phi0[yp] + d_phi0[ym] - 2.0 * p0) * invdy2;
    double lapa = (d_phi_a[xp] + d_phi_a[xm] - 2.0 * pa) * invdx2
                + (d_phi_a[yp] + d_phi_a[ym] - 2.0 * pa) * invdy2;
    double lapb = (d_phi_b[xp] + d_phi_b[xm] - 2.0 * pb) * invdx2
                + (d_phi_b[yp] + d_phi_b[ym] - 2.0 * pb) * invdy2;

    // δF/δφ_0 = Σ_{j≠0} (κ_0j·∇²φ_j + Ω_0j·φ_j)
    d_mu0[c]  = kappa_0a * lapa + kappa_0b * lapb + Omega_0a * pa + Omega_0b * pb;

    // δF/δφ_α = Σ_{j≠α} (κ_αj·∇²φ_j + Ω_αj·φ_j)
    d_mu_a[c] = kappa_0a * lap0 + kappa_ab * lapb + Omega_0a * p0 + Omega_ab * pb;

    // δF/δφ_β = Σ_{j≠β} (κ_βj·∇²φ_j + Ω_βj·φ_j)
    d_mu_b[c] = kappa_0b * lap0 + kappa_ab * lapa + Omega_0b * p0 + Omega_ab * pa;
}

// ===========================================================================
// Kernel: Explicit Euler advance — obstacle+grad1 pairwise form (N=3)
//
//   Δφ_0 = 2·(dF0) - dFa - dFb     (dFi = δF/δφ_i from k_compute_mu)
//   Δφ_α = 2·(dFa) - dF0 - dFb
//   Δφ_β = 2·(dFb) - dF0 - dFa
//   φ_i^{n+1} = φ_i^n - dt·M0·Δφ_i     (sum of Δφ_i = 0  → constraint preserved)
// ===========================================================================
__global__ void k_advance_phi(
    double* d_phi0, double* d_phi_a, double* d_phi_b,
    const double* d_mu0, const double* d_mu_a, const double* d_mu_b,
    int nx, int ny, int sx, int sy, int g,
    double dt, double M0)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= nx || iy >= ny) return;

    int idx = (ix + g) + sx * ((iy + g) + sy * g);

    double dF0 = d_mu0[idx];
    double dFa = d_mu_a[idx];
    double dFb = d_mu_b[idx];

    d_phi0[idx] -= dt * M0 * (2.0 * dF0 - dFa  - dFb);
    d_phi_a[idx] -= dt * M0 * (2.0 * dFa - dF0  - dFb);
    d_phi_b[idx] -= dt * M0 * (2.0 * dFb - dF0  - dFa);
}

// ===========================================================================
// Kernel: Non-uniform Dirichlet BC on left (x=0) and right (x=W) ghost cells.
//
//   Upper ny_top rows (iy >= ny - ny_top):
//     Left ghost + Right ghost → phi0=1, phi_a=0, phi_b=0
//   Lower rows:
//     Left  ghost → phi0=0, phi_a=1, phi_b=0
//     Right ghost → phi0=0, phi_a=0, phi_b=1
//
//   Each thread handles one iy, setting both left and right ghost columns.
// ===========================================================================
__global__ void k_apply_xbc(
    double* d_phi0, double* d_phi_a, double* d_phi_b,
    int nx, int ny, int sx, int sy, int g, int ny_top)
{
    int iy = blockIdx.x * blockDim.x + threadIdx.x;
    if (iy >= ny) return;

    bool in_top = (iy >= ny - ny_top);

    // Left ghost: ix = -1  →  stored index ix_stored = g - 1
    int idxL = (g - 1) + sx * ((iy + g) + sy * g);
    // Right ghost: ix = nx  →  stored index ix_stored = nx + g
    int idxR = (nx + g) + sx * ((iy + g) + sy * g);

    if (in_top) {
        d_phi0[idxL] = 1.0;  d_phi_a[idxL] = 0.0;  d_phi_b[idxL] = 0.0;
        d_phi0[idxR] = 1.0;  d_phi_a[idxR] = 0.0;  d_phi_b[idxR] = 0.0;
    } else {
        d_phi0[idxL] = 0.0;  d_phi_a[idxL] = 1.0;  d_phi_b[idxL] = 0.0;  // left: phi_a
        d_phi0[idxR] = 0.0;  d_phi_a[idxR] = 0.0;  d_phi_b[idxR] = 1.0;  // right: phi_b
    }
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
    const int    ny_top = cfg["initialize"]["ny_top"];

    // -----------------------------------------------------------------------
    // 3. Physical parameters
    // -----------------------------------------------------------------------
    // Pairwise interface energies
    const double gamma_0a = cfg["constants"]["gamma_0a"];
    const double gamma_0b = cfg["constants"]["gamma_0b"];
    const double gamma_ab = cfg["constants"]["gamma_ab"];

    // Diffuse interface width (physical units, same as dx units)
    const double ell = cfg["constants"]["ell"];

    // Pairwise kinetic coefficient (M0 = mobility / (3*ell), equal for all pairs)
    const double M0 = cfg["constants"]["M0"];

    // Derived parameters (obstacle + grad1)
    const double K       = 16.0 / (M_PI * M_PI);   // obstacle potential prefactor
    const double kappa_0a = ell * gamma_0a;
    const double kappa_0b = ell * gamma_0b;
    const double kappa_ab = ell * gamma_ab;
    const double Omega_0a = K * gamma_0a / ell;
    const double Omega_0b = K * gamma_0b / ell;
    const double Omega_ab = K * gamma_ab / ell;

    // Precompute finite-difference coefficients
    const double invdx2 = 1.0 / (dx * dx);
    const double invdy2 = 1.0 / (dy * dy);

    // Analytical equilibrium angle (for verification output)
    const double cos_half = gamma_ab / (2.0 * gamma_0a);
    const double theta_deg = (std::abs(cos_half) <= 1.0)
                           ? 2.0 * std::acos(cos_half) * 180.0 / M_PI
                           : -1.0;
    // Analytical grain boundary rise above y_b
    const double h_GB = (gamma_0a > 0.0 && std::abs(gamma_ab) < 2.0 * gamma_0a)
                      ? (nx * dx) * gamma_ab
                        / (2.0 * std::sqrt(4.0 * gamma_0a * gamma_0a
                                           - gamma_ab * gamma_ab))
                      : -1.0;

    std::cout << "=== MPF_AC — Static Triple Junction Benchmark (2D) ===\n"
              << "  γ_0α=" << gamma_0a << "  γ_0β=" << gamma_0b
              << "  γ_αβ=" << gamma_ab << "  ℓ=" << ell << "\n"
              << "  κ_0α=" << kappa_0a << "  Ω_0α=" << Omega_0a << "  (K=" << K << ")\n"
              << "  κ_αβ=" << kappa_ab << "  Ω_αβ=" << Omega_ab << "\n"
              << "  M0=" << M0 << "\n"
              << "  Analytical equilibrium angle θ ≈ " << theta_deg << "°\n"
              << "  Analytical GB drop below top wall: h_GB ≈ " << h_GB << "\n"
              << "  dt=" << dt << "  nSteps=" << nSteps << "\n";

    // -----------------------------------------------------------------------
    // 4. Fields
    // -----------------------------------------------------------------------
    ScalarField phi0 (mesh, "phi0",  /*ghost=*/1);
    ScalarField phi_a(mesh, "phi_a", /*ghost=*/1);
    ScalarField phi_b(mesh, "phi_b", /*ghost=*/1);

    // Scratch fields for chemical potentials (interior cells only, no BCs)
    ScalarField mu0 (mesh, "mu0",  /*ghost=*/1);
    ScalarField mu_a(mesh, "mu_a", /*ghost=*/1);
    ScalarField mu_b(mesh, "mu_b", /*ghost=*/1);

    phi0.fill(0.0);   phi_a.fill(0.0);   phi_b.fill(0.0);
    mu0.fill(0.0);    mu_a.fill(0.0);    mu_b.fill(0.0);

    // -----------------------------------------------------------------------
    // 5. Restart / initialization
    // -----------------------------------------------------------------------
    const std::string start_from = cfg["initialize"]["start_from"];
    const int start_step = IO::resolveStartStep(start_from, "phi0");

    if (start_step == 0) {
        // Cold start: initialize programmatically
        phi0.allocDevice();   phi0.uploadAllToDevice();
        phi_a.allocDevice();  phi_a.uploadAllToDevice();
        phi_b.allocDevice();  phi_b.uploadAllToDevice();
        mu0.allocDevice();    mu_a.allocDevice();   mu_b.allocDevice();

        const dim3 blk(16, 16);
        const dim3 grd((nx + 15) / 16, (ny + 15) / 16);
        k_init_fields<<<grd, blk>>>(
            phi0.d_curr, phi_a.d_curr, phi_b.d_curr,
            nx, ny, ny_top,
            phi0.storedDims[0], phi0.storedDims[1], phi0.ghost);
        cudaDeviceSynchronize();
    } else {
        // Warm restart: read from output/
        IO::initField(phi0,  start_step);
        IO::initField(phi_a, start_step);
        IO::initField(phi_b, start_step);

        phi0.allocDevice();   phi0.uploadAllToDevice();
        phi_a.allocDevice();  phi_a.uploadAllToDevice();
        phi_b.allocDevice();  phi_b.uploadAllToDevice();
        mu0.allocDevice();    mu_a.allocDevice();   mu_b.allocDevice();
    }

    // -----------------------------------------------------------------------
    // 6. Boundary conditions (matching MATLAB reference)
    //    Top  (y=H): Neumann (zero flux)
    //    Bottom(y=0): Neumann (zero flux)
    //    Left/Right: non-uniform Dirichlet via k_apply_xbc
    //      upper ny_top rows → phi0=1; lower rows → phi_a=1 (left) / phi_b=1 (right)
    // -----------------------------------------------------------------------
    NoFluxBC bcTop(Axis::Y, Side::HIGH);
    NoFluxBC bcBot(Axis::Y, Side::LOW);

    // -----------------------------------------------------------------------
    // 7. Kernel launch config
    // -----------------------------------------------------------------------
    const int  g  = phi0.ghost;
    const int  sx = phi0.storedDims[0];
    const int  sy = phi0.storedDims[1];

    const dim3 blk2(16, 16);
    const dim3 grd2((nx + 15) / 16, (ny + 15) / 16);
    const dim3 blk1D(256);
    const dim3 grd1D((ny + 255) / 256);

    // -----------------------------------------------------------------------
    // 8. Output writer
    // -----------------------------------------------------------------------
    IO::OutputWriter writer(cfg["output"]);

    int    step = start_step;
    double time = start_step * dt;

    if (start_step == 0) {
        // Download and write initial state
        phi0.downloadCurrFromDevice();
        phi_a.downloadCurrFromDevice();
        phi_b.downloadCurrFromDevice();
        writer.writeFields(phi0,  0, 0.0);
        writer.writeFields(phi_a, 0, 0.0);
        writer.writeFields(phi_b, 0, 0.0);
        std::cout << "Starting MPF_AC simulation ("
                  << nSteps << " steps, dt=" << dt << ")\n";
    } else {
        std::cout << "Resuming MPF_AC simulation from step " << start_step
                  << " (t=" << time << "), "
                  << nSteps - start_step << " steps remaining, dt=" << dt
                  << "\n";
    }

    writer.resetTimer();

    // -----------------------------------------------------------------------
    // 9. Time loop
    // -----------------------------------------------------------------------
    for (int s = start_step; s < nSteps; ++s) {

        // (a) Apply boundary conditions
        bcTop.applyOnGPU(phi0);  bcTop.applyOnGPU(phi_a);  bcTop.applyOnGPU(phi_b);
        bcBot.applyOnGPU(phi0);  bcBot.applyOnGPU(phi_a);  bcBot.applyOnGPU(phi_b);
        k_apply_xbc<<<grd1D, blk1D>>>(
            phi0.d_curr, phi_a.d_curr, phi_b.d_curr,
            nx, ny, sx, sy, g, ny_top);

        // (c) Compute functional derivatives δF/δφ_i
        k_compute_mu<<<grd2, blk2>>>(
            mu0.d_curr,  mu_a.d_curr,  mu_b.d_curr,
            phi0.d_curr, phi_a.d_curr, phi_b.d_curr,
            nx, ny, sx, sy, g,
            invdx2, invdy2,
            kappa_0a, kappa_0b, kappa_ab,
            Omega_0a, Omega_0b, Omega_ab);

        // (d) Explicit Euler advance (pairwise, M0 uniform)
        k_advance_phi<<<grd2, blk2>>>(
            phi0.d_curr, phi_a.d_curr, phi_b.d_curr,
            mu0.d_curr,  mu_a.d_curr,  mu_b.d_curr,
            nx, ny, sx, sy, g,
            dt, M0);

        // (e) Gibbs simplex projection
        PhiX::gibbsSimplexOnGPU({&phi0, &phi_a, &phi_b});

        ++step;
        time += dt;

        // (f) Output
        if (writer.shouldPrint(step))
            writer.printProgress(step, time);

        if (writer.shouldWrite(step)) {
            writer.writeFields(phi0,  step, time);
            writer.writeFields(phi_a, step, time);
            writer.writeFields(phi_b, step, time);
        }
    }

    std::cout << "Done.\n";
    return 0;
}
