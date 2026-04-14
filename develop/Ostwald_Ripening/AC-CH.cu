/************************************************************************************************************\
        
                                            Allen-Cahn & Cahn-Hilliard

    Expression : 
    μ^{n} = 2ρ^2(c^{n} - ca)(1 - eta^{n}^3 (6eta^{n}^2 - 15eta^{n} + 10)) 
            + 2ρ^2(c^{n} - cb)(eta^{n}^3 (6eta^{n}^2 - 15eta^{n} + 10)) 
            - κ_c∇²c^{n}

    c^{n+1} = c^{n} + dt*M∇²μ^{n}

    eta^{n+1} = eta^{n} - dt*L(30 ρ^2 eta^{n}^2 (1-eta^{n})^2 (2c^{n} - ca - cb)*(ca - cb) 
                + 2w * eta * (1-eta) * (1-2eta) - κ_eta∇²eta)

    Domain     : 200 × 200,  dx = dy = 1
    BC         : Periodic in both directions
    Time Scheme: Euler explicit
    IC         : c(x,y)=c0+ϵ{cos(0.105x)cos(0.11y)
                 +[cos(0.13x)cos(0.087y)]^2
                 +cos(0.025x−0.15y)cos(0.07x−0.02y)}

                 eta(x,y) = ε_eta * {cos(0.01x-4)cos(0.017y)
                                    +cos(0.12x)cos(0.12y)
                                    +psi[cos(0.047x)+0.0415y*cos(0.032x-0.005y)]^2}^2

                 
    Parameters  : M=5, ρ=sqrt(2), ca=0.3, cb=0.7, κ_c=3, κ_eta=3, L=5, w=1, c0=0.5, ϵ=0.05, ε_eta=0.1, psi=1.5

 \*************************************************************************************************************/


#include "mesh/Mesh.h"
#include "field/ScalarField.h"
#include "boundary/PeriodicBC.h"
#include "equation/Equation.h"
#include "solver/Solver.h"

#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <set>
#include <string>
#include <vector>
#include <filesystem>

int main() {
    using namespace PhiX;

    // -----------------------------------------------------------------------
    // 1. Mesh
    // -----------------------------------------------------------------------
    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                    200, 1.0, 0.0,   // nx, dx, x0
                                    200, 1.0, 0.0);  // ny, dy, y0
    mesh.print();

    // -----------------------------------------------------------------------
    // 2. Field  —  Initial condition
    // -----------------------------------------------------------------------
    ScalarField c(mesh, "c", /*ghost=*/1);
    ScalarField mu(mesh, "mu", /*ghost=*/1);
    ScalarField eta(mesh, "eta", /*ghost=*/1);

    double c0 = 0.5;
    double eps = 0.05;

    double eps_eta = 0.1;
    double psi = 1.5;

    for (int j = 0; j < mesh.n[1]; ++j)
    for (int i = 0; i < mesh.n[0]; ++i) {
        double x = mesh.coord(0, i);
        double y = mesh.coord(1, j);

        double val_c = c0 + eps * (
            cos(0.105 * x) * cos(0.11 * y)
            + pow(cos(0.13 * x) * cos(0.087 * y), 2)
            + cos(0.025 * x - 0.15 * y) * cos(0.07 * x - 0.02 * y)
        );

        c.curr[c.index(i, j)] = val_c;


        double val_eta = eps_eta * pow(
            cos(0.01 * x - 4) * cos(0.017 * y)
            + cos(0.12 * x) * cos(0.12 * y)
            + psi * pow(cos(0.047 * x) + 0.0415 * y * cos(0.032 * x - 0.005 * y), 2), 2
        );

        eta.curr[eta.index(i, j)] = val_eta;

    }

    c.allocDevice();
    c.uploadAllToDevice();

    eta.allocDevice();
    eta.uploadAllToDevice();

    mu.fill(0.0);
    mu.allocDevice();
    mu.uploadAllToDevice();

    // -----------------------------------------------------------------------
    // 3. Boundary conditions — periodic everywhere
    // -----------------------------------------------------------------------
    PeriodicBC bc_x(Axis::X);
    PeriodicBC bc_y(Axis::Y);

    // -----------------------------------------------------------------------
    // 4. Equation  
    // -----------------------------------------------------------------------
    Equation eq_1(mu, "CH_1");
    Equation eq_2(c, "CH_2");
    Equation eq_3(eta, "AC");
  
    eq_1.params["rho"] = sqrt(2);
    eq_1.params["ca"] = 0.3;
    eq_1.params["cb"] = 0.7;
    eq_1.params["kappa_c"] = 3.0;

    eq_2.params["M"] = 5.0;

    eq_3.params["rho"] = sqrt(2);
    eq_3.params["L"] = 5.0;
    eq_3.params["w"] = 1.0;
    eq_3.params["kappa_eta"] = 3.0;

    const double rho = eq_1.params["rho"];
    const double ca = eq_1.params["ca"];
    const double cb = eq_1.params["cb"];
    const double kappa = eq_1.params["kappa_c"];
    const double M = eq_2.params["M"];
    const double L = eq_3.params["L"];
    const double w = eq_3.params["w"];
    const double kappa_eta = eq_3.params["kappa_eta"];

    eq_1.setRHS(
        pw(c, [rho, ca] __host__ __device__ (double c_val) {
            return 2.0 * rho * rho * (c_val - ca);
        })
        + pw(eta, [rho, ca, cb] __host__ __device__ (double eta_val) {
            double h = eta_val * eta_val * eta_val
                     * (6.0 * eta_val * eta_val - 15.0 * eta_val + 10.0);
            return 2.0 * rho * rho * (ca - cb) * h;
        })
        - kappa * lap(c)
    );

    eq_2.setRHS(
        M * lap(mu)
    );

        // eta^{n+1} = eta^{n} - dt*L(30 ρ^2 eta^{n}^2 (1-eta^{n})^2 (2c^{n} - ca - cb)*(ca - cb) 
        //         + 2w * eta * (1-eta) * (1-2eta) - κ_eta∇²eta)
    eq_3.setRHS(
        pw(eta, c, [L, rho, ca, cb] __host__ __device__ (double eta_val, double c_val) {
            double h_deriv = 30.0 * rho * rho * eta_val * eta_val
                           * (1.0 - eta_val) * (1.0 - eta_val)
                           * (2.0 * c_val - ca - cb) * (ca - cb);
            return -L * h_deriv;
        })
        + pw(eta, [L, w] __host__ __device__ (double eta_val) {
            return -L * 2.0 * w * eta_val * (1.0 - eta_val) * (1.0 - 2.0 * eta_val);
        })
        + L * kappa_eta * lap(eta)
    );

    // -----------------------------------------------------------------------
    // 5. Solver
    // -----------------------------------------------------------------------
    const double dt      = 0.001;
    const int    nSteps  = 100000000;   // physical time = 1e5 s

    // Output at logarithmically spaced times: 0.1, 1, 10, 100, 1000, 1e4, 1e5 s
    const std::vector<double> out_times = {0.1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5};
    // Convert to step numbers (round to nearest)
    std::set<int> out_steps;
    for (double t : out_times) {
        int step = static_cast<int>(std::round(t / dt));
        if (step <= nSteps) out_steps.insert(step);
    }

    // eq_1: μ = f'(c) - κ∇²c   (auxiliary, not time-integrated)
    // eq_2: dc/dt = M∇²μ        (time-integrated by Solver)
    // eq_3: deta/dt = -L(...)     (time-integrated by Solver)
    Solver solver2(eq_2, {&bc_x, &bc_y}, dt, TimeScheme::EULER);
    Solver solver3(eq_3, {&bc_x, &bc_y}, dt, TimeScheme::EULER);

    std::filesystem::create_directories("output");

    // Write initial state (step 0, t = 0)
    c.downloadCurrFromDevice();
    eta.downloadCurrFromDevice();
    c.write("output/c_0.field");
    eta.write("output/eta_0.field");
    std::cout << "Starting simulation ("
              << nSteps << " steps, dt=" << dt << ")\n";
    std::cout << "  step 0  t=0  written: output/c_0.field\n";

    for (int s = 0; s < nSteps; ++s) {
        // Step 1: Apply BCs to c, compute μ^n = f'(c^n) - κ∇²c^n
        bc_x.applyOnGPU(c);
        bc_y.applyOnGPU(c);
        eq_1.computeRHS(mu);   // writes physical cells of mu.d_curr

        // Step 2: Apply BCs to μ (fill ghost cells for ∇²μ stencil)
        bc_x.applyOnGPU(mu);
        bc_y.applyOnGPU(mu);

        // Step 3: Advance c^{n+1} = c^n + dt * M∇²μ^n
        solver2.advance();

        // Step 4: Apply BCs to eta, compute deta/dt, advance eta
        bc_x.applyOnGPU(eta);
        bc_y.applyOnGPU(eta);
        solver3.advance();

        if (solver2.step % 10000 == 0)
            std::cout << "  [progress] step=" << solver2.step
                      << "  t=" << solver2.time << "\n" << std::flush;

        if (out_steps.count(solver2.step)) {
            c.downloadCurrFromDevice();
            std::string path = "output/c_" + std::to_string(solver2.step) + ".field";
            c.write(path);
            std::cout << "  step " << solver2.step
                      << "  t=" << solver2.time
                      << "  written: " << path << "\n" << std::flush;
        }

        if (out_steps.count(solver3.step)) {
            eta.downloadCurrFromDevice();
            std::string path = "output/eta_" + std::to_string(solver3.step) + ".field";
            eta.write(path);
            std::cout << "  step " << solver3.step
                      << "  t=" << solver3.time
                      << "  written: " << path << "\n" << std::flush;
        }
    }

    std::cout << "Done.\n";
    return 0;
}
