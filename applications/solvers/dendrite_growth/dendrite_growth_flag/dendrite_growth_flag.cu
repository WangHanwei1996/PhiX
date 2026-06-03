/***********************************************************************\
 *
 *  Dendrite Growth Solver (2D)  — flag-gated variant
 *
 *  Identical to the 2D staggered solver, with one addition:
 *  an OpenPhase-style interface flag mechanism that zeroes dphi/dt for
 *  bulk cells (|phi| >= 1 - eps_flag) that have no interface neighbour.
 *
 *  Flag classification per step:
 *    flag = 2  Interface : |phi| < 1 - eps_flag
 *    flag = 1  Halo      : bulk cell within 3x3 of any interface cell
 *    flag = 0  Bulk      : all others
 *
 *  dphi/dt is set to zero where flag == 0.
 *  dU/dt runs full-domain (same as unflagged solver).
 *
 *  Reference: benchmark3_staggered.ipynb, compute_rhs(), flag guard line.
 *
 \***********************************************************************/

#include "mesh/Mesh.h"
#include "field/ScalarField.h"
#include "boundary/BCFactory.h"
#include "equation/Equation.h"
#include "operators/Gradient.h"
#include "operators/Laplacian.h"
#include "operators/FaceOps.h"
#include "IO/ConfigFile.h"
#include "IO/FieldIO.h"
#include "IO/OutputWriter.h"

#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>

// ============================================================================
//  applyFlagMask  —  in-place bulk-cell zeroing of dphi
//
//  For each physical cell (i,j):
//    active = true  if |phi| < 1 - eps_flag           (interface, flag=2)
//                OR any 3x3 neighbour is interface      (halo,      flag=1)
//  Sets dphi[idx] = 0 for inactive (pure bulk) cells.
//
//  Memory layout (PhiX, 2D, ghost=1):
//    index(i,j,0) = (i+g) + sx*(j+g) + sx*sy*g
//    sx = nx+2g,  sy = ny+2g
// ============================================================================
__global__ void applyFlagMask(
    const double* __restrict__ phi_dev,
    double*       __restrict__ dphi_dev,
    int nx, int ny, int ghost,
    int sx, int sy,
    double eps_flag)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;

    const int base_k = sx * sy * ghost;
    auto idx = [&](int ii, int jj) {
        return base_k + (ii + ghost) + sx * (jj + ghost);
    };

    bool active = fabs(phi_dev[idx(i, j)]) < 1.0 - eps_flag;

    if (!active) {
        for (int di = -1; di <= 1 && !active; ++di)
            for (int dj = -1; dj <= 1 && !active; ++dj) {
                if (di == 0 && dj == 0) continue;
                int ni = i + di, nj = j + dj;
                if (ni < 0 || ni >= nx || nj < 0 || nj >= ny) continue;
                active = (fabs(phi_dev[idx(ni, nj)]) < 1.0 - eps_flag);
            }
    }

    if (!active)
        dphi_dev[idx(i, j)] = 0.0;
}

static void gateWithFlag(PhiX::ScalarField& phi,
                         PhiX::ScalarField& dphi,
                         double eps_flag)
{
    const int nx    = phi.mesh.n[0];
    const int ny    = phi.mesh.n[1];
    const int ghost = phi.ghost;
    const int sx    = phi.storedDims[0];
    const int sy    = phi.storedDims[1];

    dim3 block(16, 16);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y);

    applyFlagMask<<<grid, block>>>(
        phi.d_curr, dphi.d_curr,
        nx, ny, ghost, sx, sy, eps_flag);
}

// ============================================================================
//  DEBUG helper
// ============================================================================
static void printStats(const std::string& tag, PhiX::ScalarField& f)
{
    f.downloadAllFromDevice();
    double mn = 1e300, mx = -1e300, sum = 0.0;
    int n = 0;
    for (int k = 0; k < f.mesh.n[2]; ++k)
    for (int j = 0; j < f.mesh.n[1]; ++j)
    for (int i = 0; i < f.mesh.n[0]; ++i) {
        double v = f.curr[f.index(i, j, k)];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        sum += v; ++n;
    }
    std::cout << "  " << tag
              << ": min=" << std::scientific << std::setprecision(3) << mn
              << " max=" << mx
              << " mean=" << sum / n
              << std::defaultfloat << "\n";
}

// ============================================================================
//  main
// ============================================================================
int main(int argc, char* argv[])
{
    using namespace PhiX;

    IO::ConfigFile cfg = IO::ConfigFile::fromArgs(argc, argv);

    // === 1. Mesh =============================================================
    const int    nx = cfg["mesh"]["nx"];
    const double dx = cfg["mesh"]["dx"];
    const double x0 = cfg["mesh"]["x0"];
    const int    ny = cfg["mesh"]["ny"];
    const double dy = cfg["mesh"]["dy"];
    const double y0 = cfg["mesh"]["y0"];

    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN, nx, dx, x0, ny, dy, y0);
    mesh.print();

    // === 2. Time parameters ==================================================
    const double dt     = cfg["initialize"]["dt"];
    const int    nSteps = cfg["initialize"]["nSteps"];

    // === 3. Physical constants ===============================================
    const double D         = cfg["constants"]["D"];
    const double tau_0     = cfg["constants"]["tau_0"];
    const double W_0       = cfg["constants"]["W_0"];
    const double epsilon_m = cfg["constants"]["epsilon_m"];
    const double m_order   = cfg["constants"]["m"];
    const double theta_0   = cfg["constants"]["theta_0"];

    const double W0_sq      = W_0 * W_0;
    const double lambda_val = D * tau_0 / (0.6267 * W0_sq);

    std::cout << "  lambda = " << lambda_val << "\n";

    // eps_flag: matches notebook compute_flags(phi, eps=1e-1)
    const double eps_flag = 0.1;

    // === 4. Cell-centred scalar fields =======================================
    ScalarField phi     (mesh, "phi",      1);
    ScalarField U       (mesh, "U",        1);
    ScalarField phi_x_cc(mesh, "phi_x_cc", 1);
    ScalarField phi_y_cc(mesh, "phi_y_cc", 1);
    ScalarField a_cc    (mesh, "a_cc",     1);
    ScalarField dphi    (mesh, "dphi",     1);

    phi.fill(0);  U.fill(0);
    phi_x_cc.fill(0); phi_y_cc.fill(0); a_cc.fill(1.0); dphi.fill(0);

    // === 5. Face fields ======================================================
    FaceField phi_x_xf(mesh, 0, "phi_x_xf");
    FaceField phi_y_xf(mesh, 0, "phi_y_xf");
    FaceField jx      (mesh, 0, "jx");

    FaceField phi_y_yf(mesh, 1, "phi_y_yf");
    FaceField phi_x_yf(mesh, 1, "phi_x_yf");
    FaceField jy      (mesh, 1, "jy");

    // === 6. Initialise fields ================================================
    const std::string start_from = cfg["initialize"]["start_from"];
    const int         start_step = IO::resolveStartStep(start_from);

    IO::initField(phi, start_step);
    IO::initField(U,   start_step);

    auto allocUp = [](ScalarField& f){ f.allocDevice(); f.uploadAllToDevice(); };
    allocUp(phi); allocUp(U);
    allocUp(phi_x_cc); allocUp(phi_y_cc); allocUp(a_cc); allocUp(dphi);

    auto allocUpFace = [](FaceField& f){ f.fill(0.0); f.allocDevice(); f.uploadToDevice(); };
    allocUpFace(phi_x_xf); allocUpFace(phi_y_xf); allocUpFace(jx);
    allocUpFace(phi_y_yf); allocUpFace(phi_x_yf); allocUpFace(jy);

    // === 7. Boundary conditions ==============================================
    auto  bcSet = buildBCs(mesh, cfg["boundary_conditions"]);
    auto& bcs   = bcSet.ptrs;

    // === 8. Equations ========================================================

    Equation eq_phi_x_cc(phi_x_cc, "phi_x_cc");
    eq_phi_x_cc.setRHS(grad(phi, 0, 1.0));

    Equation eq_phi_y_cc(phi_y_cc, "phi_y_cc");
    eq_phi_y_cc.setRHS(grad(phi, 1, 1.0));

    Equation eq_a_cc(a_cc, "a_cc");
    eq_a_cc.setRHS(
        pw(phi_x_cc, phi_y_cc, PHIX_FN (double px, double py) {
            double theta = atan2(py, px);
            return 1.0 + epsilon_m * cos(m_order * (theta - theta_0));
        })
    );

    auto N_bulk = pw(phi, U, PHIX_FN (double p, double u) {
        return (1.0 - p*p) * (p - lambda_val * u * (1.0 - p*p));
    });
    auto inv_tau = pw(a_cc, PHIX_FN (double a) {
        return 1.0 / (tau_0 * a * a);
    });

    Equation eq_dphi(dphi, "dphi_dt");
    eq_dphi.setRHS(
        inv_tau * (N_bulk + divFace(jx, jy))
    );

    Equation eq_phi(phi, "AC_phi");
    eq_phi.setRHS(1.0 * dphi);

    Equation eq_U(U, "diffusion_U");
    eq_U.setRHS(lap(U, D) + 0.5 * dphi);

    // === 9. Output & time loop ===============================================
    eq_U.step = start_step;
    eq_U.time = start_step * dt;

    IO::OutputWriter writer(cfg["output"]);

    if (start_step == 0) {
        writer.writeFields(phi, 0, eq_U.time);
        writer.writeFields(U,   0, eq_U.time);
        std::cout << "Starting dendrite growth simulation (flag-gated, eps="
                  << eps_flag << ", " << nSteps << " steps, dt=" << dt << ")\n";
    } else {
        std::cout << "Resuming from step " << start_step
                  << " (t=" << start_step * dt << "), "
                  << nSteps - start_step << " steps remaining, dt=" << dt << "\n";
    }

    writer.resetTimer();
    for (int s = start_step; s < nSteps; ++s) {

        // ── A: cell-centre phi gradients ──────────────────────────────────
        eq_phi_x_cc.advanceSteady(bcs, &phi);
        eq_phi_y_cc.advanceSteady(bcs, &phi);

        // ── B: assemble Jx on x-faces ─────────────────────────────────────
        faceGradGPU(phi, 0, phi_x_xf);
        interpGPU(phi_y_cc, 0, phi_y_xf);
        facePWGPU(jx, phi_x_xf, phi_y_xf,
                  PHIX_FN (double px, double py) {
                      double theta    = atan2(py, px);
                      double a        = 1.0 + epsilon_m * cos(m_order * (theta - theta_0));
                      double sin_term = epsilon_m * m_order * sin(m_order * (theta - theta_0));
                      return W0_sq * a * (a * px + sin_term * py);
                  });

        // ── C: assemble Jy on y-faces ─────────────────────────────────────
        faceGradGPU(phi, 1, phi_y_yf);
        interpGPU(phi_x_cc, 1, phi_x_yf);
        facePWGPU(jy, phi_y_yf, phi_x_yf,
                  PHIX_FN (double py, double px) {
                      double theta    = atan2(py, px);
                      double a        = 1.0 + epsilon_m * cos(m_order * (theta - theta_0));
                      double sin_term = epsilon_m * m_order * sin(m_order * (theta - theta_0));
                      return W0_sq * a * (a * py - sin_term * px);
                  });

        // ── D: cell-centre anisotropy for tau ─────────────────────────────
        eq_a_cc.advanceSteady(bcs, nullptr);

        // ── E: dphi/dt = (N + div(J)) / tau ───────────────────────────────
        eq_dphi.advanceSteady(bcs, nullptr);

        // ── Flag gate: zero dphi for pure bulk cells (flag == 0) ──────────
        gateWithFlag(phi, dphi, eps_flag);

        // ── F: phi += dt * dphi ───────────────────────────────────────────
        eq_phi.advanceTransient(bcs, dt, &phi);

        // ── G: U += dt * (D*lap(U) + 0.5*dphi) ───────────────────────────
        eq_U.advanceTransient(bcs, dt, &U);

        // ── Output ────────────────────────────────────────────────────────
        if (writer.shouldPrint(eq_U.step)) {
            writer.printProgress(eq_U.step, eq_U.time);
            printStats("phi   ", phi);
            printStats("U     ", U);
            printStats("a_cc  ", a_cc);
            printStats("dphi  ", dphi);
        }
        if (writer.shouldWrite(eq_U.step)) {
            writer.writeFields(phi, eq_U.step, eq_U.time);
            writer.writeFields(U,   eq_U.step, eq_U.time);
        }
    }

    std::cout << "Done.\n";
    return 0;
}
