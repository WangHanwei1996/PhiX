/***********************************************************************\
 *
 *  Cahn-Hilliard Solver — Double-Well Free Energy (2D)
 *
 *  Author : Wang Hanwei
 *  Email  : wanghanweibnds2015@gmail.com
 *
 *  Description
 *  -----------
 *  Solves the Cahn-Hilliard equation with a double-well bulk free-energy
 *  density for spinodal decomposition:
 *
 *      dc/dt = M ∇²μ
 *      μ     = f'(c) − κ ∇²c
 *      f'(c) = 2ρ (c − ca)(c − cb)(2c − ca − cb)
 *
 \***********************************************************************/

#include "mesh/Mesh.h"
#include "field/ScalarField.h"
#include "boundary/BCFactory.h"
#include "equation/Equation.h"
#include "solver/Solver.h"
#include "IO/ConfigFile.h"
#include "IO/FieldIO.h"
#include "IO/OutputWriter.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

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

    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                    nx, dx, x0,
                                    ny, dy, y0);
    mesh.print();

    // === 2. Time parameters ==================================================
    const double dt     = cfg["initialize"]["dt"];
    const int    nSteps = cfg["initialize"]["nSteps"];

    // === 3. Fields & initialization ==========================================
    ScalarField c (mesh, "c",  /*ghost=*/1);
    ScalarField mu(mesh, "mu", /*ghost=*/1);
    c.fill(0);
    mu.fill(0);

    const std::string start_from = cfg["initialize"]["start_from"];
    const int         start_step = IO::resolveStartStep(start_from);

    IO::initField(c, start_step);
    if (start_step == 0) IO::initField(mu, start_step);

    c.allocDevice();
    c.uploadAllToDevice();
    mu.allocDevice();
    mu.uploadAllToDevice();

    // === 4. Boundary conditions ==============================================
    auto  bcSet = buildBCs(cfg["boundary_conditions"]);
    auto& bcs   = bcSet.ptrs;

    // === 5. Equations ========================================================
    const double rho   = cfg["constants"]["rho"];
    const double ca    = cfg["constants"]["ca"];
    const double cb    = cfg["constants"]["cb"];
    const double kappa = cfg["constants"]["kappa"];
    const double M     = cfg["constants"]["M"];

    // μ = f'(c) − κ ∇²c
    Equation eqMu(mu, "CH_mu");
    eqMu.setRHS(
        pw(c, PHIX_FN (double c_val) {
            return 2.0 * rho * (c_val - ca) * (c_val - cb)
                       * (2.0 * c_val - ca - cb);
        })
        - kappa * lap(c)
    );

    // dc/dt = M ∇²μ
    Equation eqC(c, "CH_c");
    eqC.setRHS(M * lap(mu));

    // === 6. Solver ===========================================================
    //  eqMu (STEADY)    — algebraic: μ = f'(c) − κ∇²c
    //  eqC  (TRANSIENT) — time-integrated: dc/dt = M∇²μ
    Solver solver(
        {
            { &c,  bcs, &eqMu, EquationType::STEADY    },
            { &mu, bcs, &eqC,  EquationType::TRANSIENT }
        },
        dt, TimeScheme::EULER);

    solver.step = start_step;
    solver.time = start_step * dt;

    // === 7. Output & time loop ===============================================
    IO::OutputWriter writer(cfg["output"]);

    if (start_step == 0) {
        writer.writeFields(c, 0, solver.time);
        std::cout << "Starting Cahn-Hilliard simulation ("
                  << nSteps << " steps, dt=" << dt << ")\n";
    } else {
        std::cout << "Resuming Cahn-Hilliard simulation from step " << start_step
                  << " (t=" << start_step * dt << "), "
                  << nSteps - start_step << " steps remaining, dt=" << dt
                  << "\n";
    }

    writer.resetTimer();

    for (int step = start_step; step < nSteps; ++step) {
        solver.advance();

        if (writer.shouldPrint(solver.step))
            writer.printProgress(solver.step, solver.time);

        if (writer.shouldWrite(solver.step))
            writer.writeFields(c, solver.step, solver.time);
    }

    std::cout << "Done.\n";
    return 0;
}
