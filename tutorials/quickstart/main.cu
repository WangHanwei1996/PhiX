// ---------------------------------------------------------------------------
// tutorials/quickstart/main.cu
//
// 2D Allen-Cahn phase-field equation
//
//   dφ/dt = M·∇²φ + M·(φ - φ³)
//
// Domain   : 256 × 256,  dx = dy = 0.5
// BC       : periodic in both directions
// Scheme   : RK4
// IC       : small random perturbation around φ = 0
// ---------------------------------------------------------------------------

#include "mesh/Mesh.h"
#include "field/ScalarField.h"
#include "boundary/PeriodicBC.h"
#include "equation/Equation.h"
#include "solver/Solver.h"

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <string>
#include <filesystem>

int main() {
    using namespace PhiX;

    // -----------------------------------------------------------------------
    // 1. Mesh
    // -----------------------------------------------------------------------
    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                    256, 0.5, 0.0,   // nx, dx, x0
                                    256, 0.5, 0.0);  // ny, dy, y0
    mesh.print();

    // -----------------------------------------------------------------------
    // 2. Field  —  small random initial condition
    // -----------------------------------------------------------------------
    ScalarField phi(mesh, "phi", /*ghost=*/1);

    std::srand(42);
    for (int j = 0; j < mesh.n[1]; ++j)
    for (int i = 0; i < mesh.n[0]; ++i) {
        double r = (double)std::rand() / RAND_MAX * 2.0 - 1.0;
        phi.curr[phi.index(i, j)] = 0.05 * r;
    }

    phi.allocDevice();
    phi.uploadAllToDevice();

    // -----------------------------------------------------------------------
    // 3. Boundary conditions — periodic everywhere
    // -----------------------------------------------------------------------
    PeriodicBC bc_x(Axis::X);
    PeriodicBC bc_y(Axis::Y);

    // -----------------------------------------------------------------------
    // 4. Equation  —  dφ/dt = M∇²φ + M(φ - φ³)
    // -----------------------------------------------------------------------
    Equation eq(phi, "AllenCahn");
    const double M = 1.0;

    eq.setRHS(
        M * lap(phi)
      + M * pw(phi, [] __host__ __device__ (double p) { return p - p * p * p; })
    );

    // -----------------------------------------------------------------------
    // 5. Solver
    // -----------------------------------------------------------------------
    const double dt     = 0.01;
    const int    nSteps = 10000;
    const int    outEvery = 500;

    Solver solver(eq, {&bc_x, &bc_y}, dt, TimeScheme::RK4);

    std::filesystem::create_directories("output");

    std::cout << "Starting Allen-Cahn simulation ("
              << nSteps << " steps, dt=" << dt << ")\n";

    // Write initial state (step 0, t = 0)
    phi.downloadCurrFromDevice();
    phi.write("output/phi_0.field");
    std::cout << "  step 0  t=0  written: output/phi_0.field\n";

    solver.run(nSteps, outEvery, [&](const Solver& s) {
        phi.downloadCurrFromDevice();

        std::string path = "output/phi_" + std::to_string(s.step) + ".field";
        phi.write(path);

        std::cout << "  step " << s.step
                  << "  t=" << s.time
                  << "  written: " << path << "\n";
    });

    std::cout << "Done.\n";
    return 0;
}
