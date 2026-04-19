/***********************************************************************\
        
        Spinodal Decomposition: Cahn-Hillard Equation

    c^{n+1} = c^{n} + dt * M∇²μ^{n}
    μ^{n} = 2ρ(c^{n} - ca)(c^{n} - cb)(2c^{n} - ca - cb) - κ∇²c^{n}

    Domain     : 200 × 200,  dx = dy = 1.0
    BC         : periodic in both directions
    Time Scheme: Euler explicit
    IC         : c(x,y)=c0+ϵ{cos(0.105x)cos(0.11y)
                 +[cos(0.13x)cos(0.087y)]^2
                 +cos(0.025x−0.15y)cos(0.07x−0.02y)}

    Parameters  : M=5, ρ=5, ca=0.3, cb=0.7, κ=2, c0=0.5, ϵ=0.01

 \***********************************************************************/


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

    double c0 = 0.5;
    double eps = 0.01;

    for (int j = 0; j < mesh.n[1]; ++j)
    for (int i = 0; i < mesh.n[0]; ++i) {
        double x = mesh.coord(0, i);
        double y = mesh.coord(1, j);

        double val = c0 + eps * (
              std::cos(0.105 * x) * std::cos(0.11 * y)
            + std::pow(std::cos(0.13 * x) * std::cos(0.087 * y), 2)
            + std::cos(0.025 * x - 0.15 * y) * std::cos(0.07 * x - 0.02 * y)
        );

        c.curr[c.index(i, j)] = val;
    }

    c.allocDevice();
    c.uploadAllToDevice();

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

    const double rho   = 5.0;
    const double ca    = 0.3;
    const double cb    = 0.7;
    const double kappa = 2.0;
    const double M     = 5.0;

    eq_1.setRHS(
        pw(c, [rho, ca, cb] __host__ __device__ (double c_val) {
            return 2.0 * rho * (c_val - ca) * (c_val - cb) * (2.0 * c_val - ca - cb);
        })
        - kappa * lap(c)
    );

    eq_2.setRHS(
        M * lap(mu)
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
    Solver solver(eq_2, {&bc_x, &bc_y}, dt, TimeScheme::EULER);

    std::filesystem::create_directories("output");

    // Write initial state (step 0, t = 0)
    c.downloadCurrFromDevice();
    c.write("output/c_0.field");
    std::cout << "Starting Cahn-Hilliard simulation ("
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
        solver.advance();

        if (solver.step % 10000 == 0)
            std::cout << "  [progress] step=" << solver.step
                      << "  t=" << solver.time << "\n" << std::flush;

        if (out_steps.count(solver.step)) {
            c.downloadCurrFromDevice();
            std::string path = "output/c_" + std::to_string(solver.step) + ".field";
            c.write(path);
            std::cout << "  step " << solver.step
                      << "  t=" << solver.time
                      << "  written: " << path << "\n" << std::flush;
        }
    }

    std::cout << "Done.\n";
    return 0;
}
