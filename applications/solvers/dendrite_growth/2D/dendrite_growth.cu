/***********************************************************************\
 *
 *  Dendrite Growth Solver (2D)
 *
 *  Author : Wang Hanwei
 *  Email  : wanghanweibnds2015@gmail.com
 *
 *  Description
 *  -----------
 *  Anisotropic Allen-Cahn (phi) + dimensionless thermal diffusion (U)
 *  for simulating solidification and dendritic growth.
 *
 *  Variables:
 *    phi  -- phase-field order parameter  (0: liquid, 1: solid)
 *    U    -- dimensionless undercooling   (U = (T - T_m) / (L/c_p))
 *
 *  Evolution equations:
 *    tau(n) * dphi/dt = (phi - lambda*U*(1-phi^2))*(1-phi^2) + div(J)
 *    dU/dt            = D * lap(U) + 0.5 * dphi/dt
 *
 *  Anisotropic capillarity:
 *    a(n)   = 1 + epsilon_m * cos(m*(theta - theta_0))
 *    W(n)   = W_0 * a(n),    tau(n) = tau_0 * a(n)^2
 *    theta  = atan2(phi_y, phi_x)
 *
 *  Anisotropic flux vector (analytic derivative, |grad phi|^2 cancelled):
 *    J_x = W_0^2 * a * [a * phi_x + epsilon_m*m*sin(m*(theta-theta_0)) * phi_y]
 *    J_y = W_0^2 * a * [a * phi_y - epsilon_m*m*sin(m*(theta-theta_0)) * phi_x]
 *
 *  Coupling constant:
 *    lambda = D * tau_0 / (0.6267 * W_0^2)
 *
 *  Auxiliary fields (STEADY per step):
 *    phi_x, phi_y -- gradient components of phi
 *    a_fld        -- anisotropy function a(n)
 *    J_x, J_y     -- anisotropic flux components
 *    dphi         -- dphi/dt (stored so dU/dt can reuse it)
 *
 *  Solver pipeline (10 steps, Euler):
 *   1. BC(phi)   -> phi_x = d(phi)/dx              [STEADY]
 *   2. BC(phi)   -> phi_y = d(phi)/dy              [STEADY]
 *   3. BC(phi_x) -> a_fld = a(n)                   [STEADY, phi_x halos filled]
 *   4. BC(phi_y) -> J_x   = J_x(phi_x,phi_y,a)    [STEADY, phi_y halos filled]
 *   5. BC(J_x)   -> J_y   = J_y(phi_x,phi_y,a)    [STEADY, J_x  halos filled]
 *   6. BC(J_y)   -> dphi  = [N+div(J)]/tau(n)      [STEADY, J_y  halos filled]
 *   7. BC(phi)   -> phi  += dt * dphi              [TRANSIENT]
 *   8. BC(U)     -> U_x   = iso_grad(U,0)          [STEADY, 9-pt isotropic d/dx]
 *   9. BC(U_x)   -> U_y   = iso_grad(U,1)          [STEADY, U ghost still valid]
 *  10. BC(U_y)   -> U    += dt*(D*iso_lap(U)+0.5*dphi) [TRANSIENT]
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
#include <iomanip>
#include <string>
#include <algorithm>

// === DEBUG helper: print min/max/mean over physical cells ===================
static void printStats(const std::string& tag, PhiX::ScalarField& f) {
    f.downloadAllFromDevice();
    double mn = 1e300, mx = -1e300, sum = 0.0;
    int n = 0;
    for (int k = 0; k < f.mesh.n[2]; ++k)
    for (int j = 0; j < f.mesh.n[1]; ++j)
    for (int i = 0; i < f.mesh.n[0]; ++i) {
        double v = f.curr[f.index(i, j, k)];
        mn = std::min(mn, v); mx = std::max(mx, v); sum += v; ++n;
    }
    std::cout << "  " << tag
              << ": min=" << std::scientific << std::setprecision(3) << mn
              << " max=" << mx
              << " mean=" << sum / n
              << std::defaultfloat << "\n";
}

// ---------------------------------------------------------------------------
// Clamp phi to [-1, 1] in-place on device after each Euler update.
// Prevents interface overshoot from triggering 1-phi^2 sign flip and
// subsequent NaN divergence.
// ---------------------------------------------------------------------------
__global__ void kernel_clamp_phi(double* data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    double v = data[idx];
    if      (v >  1.0) data[idx] =  1.0;
    else if (v < -1.0) data[idx] = -1.0;
}

// ---------------------------------------------------------------------------
// Correct U for the latent heat discarded by the phi clamp.
//
// After advance() + advanceTimeLevelGPU:
//   phi.d_curr == phi.d_prev == phi_new_unclamped
// After kernel_clamp_phi:
//   phi.d_curr  = phi_clamped
//   phi.d_prev  = phi_new_unclamped  (still holds pre-clamp value)
//
// U was updated with 0.5*dphi_unclamped = 0.5*(phi_unclamped - phi_old)/dt.
// The actual phi change is 0.5*dphi_actual = 0.5*(phi_clamped - phi_old)/dt.
// Correction (exact):
//   dU = 0.5 * (phi_clamped - phi_unclamped)
//      = 0.5 * (phi.d_curr  - phi.d_prev)
// This is nonzero only in the tiny set of cells that were clamped.
// ---------------------------------------------------------------------------
__global__ void kernel_u_latent_correction(double* u,
                                           const double* phi_curr,
                                           const double* phi_prev,
                                           int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    u[idx] += 0.5 * (phi_curr[idx] - phi_prev[idx]);
}

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

    // === 4. Fields ===========================================================
    ScalarField phi  (mesh, "phi",   1);
    ScalarField U    (mesh, "U",     1);
    // Auxiliary fields
    ScalarField phi_x  (mesh, "phi_x",   1);
    ScalarField phi_y  (mesh, "phi_y",   1);
    ScalarField a_fld  (mesh, "a_fld",   1);
    ScalarField J_x    (mesh, "J_x",     1);
    ScalarField J_y    (mesh, "J_y",     1);
    ScalarField dphi   (mesh, "dphi",    1);
    ScalarField U_x    (mesh, "U_x",     1);
    ScalarField U_y    (mesh, "U_y",     1);

    phi.fill(0);   U.fill(0);
    phi_x.fill(0); phi_y.fill(0); a_fld.fill(1.0);
    J_x.fill(0);   J_y.fill(0);   dphi.fill(0);
    U_x.fill(0);   U_y.fill(0);

    const std::string start_from = cfg["initialize"]["start_from"];
    const int         start_step = IO::resolveStartStep(start_from);

    IO::initField(phi, start_step);
    IO::initField(U,   start_step);

    auto allocUp = [](ScalarField& f){ f.allocDevice(); f.uploadAllToDevice(); };
    allocUp(phi);   allocUp(U);
    allocUp(phi_x); allocUp(phi_y); allocUp(a_fld);
    allocUp(J_x);   allocUp(J_y);   allocUp(dphi);
    allocUp(U_x);   allocUp(U_y);

    // === 5. Boundary conditions ==============================================
    auto  bcSet = buildBCs(cfg["boundary_conditions"]);
    auto& bcs   = bcSet.ptrs;

    // === 6. Equations ========================================================

    // ------ 6a. phi_x = d(phi)/dx -------------------------------------------
    Equation eq_phi_x(phi_x, "phi_x");
    eq_phi_x.setRHS(iso_grad(phi, 0));

    // ------ 6b. phi_y = d(phi)/dy -------------------------------------------
    Equation eq_phi_y(phi_y, "phi_y");
    eq_phi_y.setRHS(iso_grad(phi, 1));

    // ------ 6c. a(n) = 1 + epsilon_m * cos(m*(theta - theta_0)) -------------
    Equation eq_a(a_fld, "a_fld");
    eq_a.setRHS(
        pw(phi_x, phi_y, PHIX_FN (double px, double py) {
            double theta = atan2(py, px);
            return 1.0 + epsilon_m * cos(m_order * (theta - theta_0));
        })
    );

    // ------ 6d. J_x = W0^2 * a * [a*phi_x + da/dtheta/a * phi_y] -----------
    //   where da/dtheta = -epsilon_m*m*sin(m*(theta-theta_0))
    //   so   J_x = W0^2 * a * [a*phi_x + epsilon_m*m*sin(...)*phi_y]
    Equation eq_Jx(J_x, "J_x");
    eq_Jx.setRHS(
        pw(phi_x, phi_y, a_fld, PHIX_FN (double px, double py, double a) {
            double theta    = atan2(py, px);
            double sin_term = epsilon_m * m_order * sin(m_order * (theta - theta_0));
            return W0_sq * a * (a * px + sin_term * py);
        })
    );

    // ------ 6e. J_y = W0^2 * a * [a*phi_y - epsilon_m*m*sin(...)*phi_x] ----
    Equation eq_Jy(J_y, "J_y");
    eq_Jy.setRHS(
        pw(phi_x, phi_y, a_fld, PHIX_FN (double px, double py, double a) {
            double theta    = atan2(py, px);
            double sin_term = epsilon_m * m_order * sin(m_order * (theta - theta_0));
            return W0_sq * a * (a * py - sin_term * px);
        })
    );

    // ------ 6f. dphi/dt = [N(phi,U) + div(J)] / (tau_0 * a^2) --------------
    //   N(phi,U) = (phi - lambda*U*(1-phi^2)) * (1 - phi^2)
    //   Composite DSL:  inv_tau * RHSExpr  -> Term
    auto N_term = pw(phi, U, PHIX_FN (double p, double u) {
        return (p - lambda_val * u * (1.0 - p*p)) * (1.0 - p*p);
    });
    auto inv_tau = pw(a_fld, PHIX_FN (double a) {
        return 1.0 / (tau_0 * a * a);
    });

    Equation eq_dphi(dphi, "dphi_dt");
    eq_dphi.setRHS(
        inv_tau * (N_term + iso_grad(J_x, 0) + iso_grad(J_y, 1))
    );

    // ------ 6g. phi: Euler update with stored dphi/dt -----------------------
    Equation eq_phi(phi, "AC_phi");
    eq_phi.setRHS(1.0 * dphi);

    // ------ 6h. U_x = iso_grad(U, 0)  (9-point isotropic d/dx) -------------
    Equation eq_Ux(U_x, "U_x");
    eq_Ux.setRHS(iso_grad(U, 0));

    // ------ 6i. U_y = iso_grad(U, 1)  (9-point isotropic d/dy) -------------
    Equation eq_Uy(U_y, "U_y");
    eq_Uy.setRHS(iso_grad(U, 1));

    // ------ 6j. U: D*iso_lap(U) + 0.5*dphi/dt  (9-pt isotropic Laplacian) --
    //   iso_lap(U) = iso_grad(U_x,0) + iso_grad(U_y,1)
    Equation eq_U(U, "diffusion_U");
    eq_U.setRHS(iso_grad(U_x, 0, D) + iso_grad(U_y, 1, D) + 0.5 * dphi);

    // === 7. Time loop ========================================================
    //  10 steps per time step; clamp phi between step 7 and step 8.
    eq_U.step = start_step;
    eq_U.time = start_step * dt;

    // === 8. Output & time loop ===============================================
    IO::OutputWriter writer(cfg["output"]);

    if (start_step == 0) {
        writer.writeFields(phi, 0, eq_U.time);
        writer.writeFields(U,   0, eq_U.time);
        std::cout << "Starting dendrite growth simulation ("
                  << nSteps << " steps, dt=" << dt << ")\n";
    } else {
        std::cout << "Resuming dendrite growth simulation from step " << start_step
                  << " (t=" << start_step * dt << "), "
                  << nSteps - start_step << " steps remaining, dt=" << dt << "\n";
    }

    writer.resetTimer();
    for (int s = start_step; s < nSteps; ++s) {
        // Steps 1-6: compute intermediate (auxiliary) fields
        eq_phi_x.advanceSteady(bcs, &phi);   // 1: phi_x = iso_grad(phi,0)
        eq_phi_y.advanceSteady(bcs, &phi);   // 2: phi_y = iso_grad(phi,1)
        eq_a.advanceSteady(bcs, &phi_x);     // 3: a_fld = a(n)
        eq_Jx.advanceSteady(bcs, &phi_y);    // 4: J_x
        eq_Jy.advanceSteady(bcs, &J_x);      // 5: J_y
        eq_dphi.advanceSteady(bcs, &J_y);    // 6: dphi/dt

        // Step 7: phi += dt * dphi  (phi.d_prev = phi_unclamped after advance)
        eq_phi.advanceTransient(bcs, dt, &phi);

        // -----------------------------------------------------------------
        // Clamp phi to [-1, 1] and correct U for the removed latent heat.
        // After advanceTransient, advanceTimeLevelGPU has already run:
        //   phi.d_curr == phi.d_prev == phi_new_unclamped
        // -----------------------------------------------------------------
        {
            int n = static_cast<int>(phi.storedSize);
            // Clamp phi; d_prev retains unclamped value
            kernel_clamp_phi<<<(n + 255) / 256, 256>>>(phi.d_curr, n);

            // Steps 8-10: advance U using (unclamped dphi field as source term)
            eq_Ux.advanceSteady(bcs, &U);        // 8: U_x = iso_grad(U,0)
            eq_Uy.advanceSteady(bcs, &U_x);      // 9: U_y = iso_grad(U,1)
            eq_U.advanceTransient(bcs, dt, &U_y); // 10: U += dt*(D*iso_lap+0.5*dphi)

            // Correct U for latent heat discarded by phi clamp
            kernel_u_latent_correction<<<(n + 255) / 256, 256>>>(
                U.d_curr, phi.d_curr, phi.d_prev, n);
            // Sync d_prev to clamped value for next step
            cudaMemcpy(phi.d_prev, phi.d_curr,
                       n * sizeof(double), cudaMemcpyDeviceToDevice);
        }

        if (writer.shouldPrint(eq_U.step)) {
            writer.printProgress(eq_U.step, eq_U.time);
            // [DEBUG] field-statistic diagnostics
            printStats("phi  ", phi);
            printStats("U    ", U);
            printStats("phi_x", phi_x);
            printStats("phi_y", phi_y);
            printStats("a_fld", a_fld);
            printStats("J_x  ", J_x);
            printStats("J_y  ", J_y);
            printStats("dphi ", dphi);
        }
        if (writer.shouldWrite(eq_U.step)) {
            writer.writeFields(phi, eq_U.step, eq_U.time);
            writer.writeFields(U,   eq_U.step, eq_U.time);
        }
    }
    std::cout << "Done.\n";
    return 0;
}
