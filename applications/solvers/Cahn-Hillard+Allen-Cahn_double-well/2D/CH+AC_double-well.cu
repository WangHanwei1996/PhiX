/***********************************************************************\
 *
 *  Cahn-Hilliard + Allen-Cahn Solver -- Double-Well Free Energy (2D)
 *
 *  Author : Wang Hanwei
 *  Email  : wanghanweibnds2015@gmail.com
 *
 *  Description
 *  -----------
 *  Coupled Cahn-Hilliard / Allen-Cahn equations for Ostwald ripening:
 *
 *      mu      = 2rho^2(c-ca) + 2rho^2(ca-cb)*h(eta) - kappa_c*lap(c)
 *      h(eta)  = eta^3*(6eta^2 - 15eta + 10)
 *      dc/dt   = M * lap(mu)
 *      deta/dt = -L[30rho^2 eta^2(1-eta)^2(2c-ca-cb)(ca-cb)
 *                   + 2w*eta*(1-eta)*(1-2eta) - kappa_eta*lap(eta)]
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
    ScalarField c  (mesh, "c",   /*ghost=*/1);
    ScalarField mu (mesh, "mu",  /*ghost=*/1);
    ScalarField eta(mesh, "eta", /*ghost=*/1);

    c.fill(0);
    mu.fill(0);
    eta.fill(0);

    const std::string start_from = cfg["initialize"]["start_from"];
    const int         start_step = IO::resolveStartStep(start_from);

    IO::initField(c,   start_step);
    IO::initField(eta, start_step);
    if (start_step == 0) IO::initField(mu, start_step);

    c.allocDevice();
    c.uploadAllToDevice();
    eta.allocDevice();
    eta.uploadAllToDevice();
    mu.allocDevice();
    mu.uploadAllToDevice();

    // === 4. Boundary conditions ==============================================
    auto  bcSet = buildBCs(cfg["boundary_conditions"]);
    auto& bcs   = bcSet.ptrs;

    // === 5. Equations ========================================================
    const double rho       = cfg["constants"]["rho"];
    const double ca        = cfg["constants"]["ca"];
    const double cb        = cfg["constants"]["cb"];
    const double kappa_c   = cfg["constants"]["kappa_c"];
    const double kappa_eta = cfg["constants"]["kappa_eta"];
    const double M         = cfg["constants"]["M"];
    const double L         = cfg["constants"]["L"];
    const double w         = cfg["constants"]["w"];

    // mu = 2*rho^2*(c-ca) + 2*rho^2*(ca-cb)*h(eta) - kappa_c*lap(c)
    Equation eqMu(mu, "CH_mu");
    eqMu.setRHS(
        pw(c, PHIX_FN (double c_val) {
            return 2.0 * rho * rho * (c_val - ca);
        })
        + pw(eta, PHIX_FN (double eta_val) {
            double h = eta_val * eta_val * eta_val
                     * (6.0 * eta_val * eta_val - 15.0 * eta_val + 10.0);
            return 2.0 * rho * rho * (ca - cb) * h;
        })
        - kappa_c * lap(c)
    );

    // dc/dt = M * lap(mu)
    Equation eqC(c, "CH_c");
    eqC.setRHS(M * lap(mu));

    // deta/dt = -L[30rho^2 eta^2(1-eta)^2(2c-ca-cb)(ca-cb)
    //              + 2w*eta*(1-eta)*(1-2eta) - kappa_eta*lap(eta)]
    Equation eqEta(eta, "AC_eta");
    eqEta.setRHS(
        pw(eta, c, PHIX_FN (double eta_val, double c_val) {
            double bulk = 30.0 * rho * rho
                        * eta_val * eta_val * (1.0 - eta_val) * (1.0 - eta_val)
                        * (2.0 * c_val - ca - cb) * (ca - cb);
            double dw   = 2.0 * w * eta_val * (1.0 - eta_val) * (1.0 - 2.0 * eta_val);
            return -L * (bulk + dw);
        })
        + L * kappa_eta * lap(eta)
    );

    // === 6. Solver ===========================================================
    //  eqMu  (STEADY)    -- mu = f'(c,eta) - kappa_c*lap(c)
    //  eqC   (TRANSIENT) -- dc/dt = M*lap(mu)
    //  eqEta (TRANSIENT) -- deta/dt = -L*(...)
    Solver solver(
        {
            { &c,   bcs, &eqMu,  EquationType::STEADY    },
            { &mu,  bcs, &eqC,   EquationType::TRANSIENT },
            { &eta, bcs, &eqEta, EquationType::TRANSIENT }
        },
        dt, TimeScheme::EULER);

    solver.step = start_step;
    solver.time = start_step * dt;

    // === 7. Output & time loop ===============================================
    IO::OutputWriter writer(cfg["output"]);

    if (start_step == 0) {
        writer.writeFields(c,   0, solver.time);
        writer.writeFields(eta, 0, solver.time);
        std::cout << "Starting CH+AC simulation ("
                  << nSteps << " steps, dt=" << dt << ")\n";
    } else {
        std::cout << "Resuming CH+AC simulation from step " << start_step
                  << " (t=" << start_step * dt << "), "
                  << nSteps - start_step << " steps remaining, dt=" << dt
                  << "\n";
    }

    writer.resetTimer();

    for (int step = start_step; step < nSteps; ++step) {
        solver.advance();

        if (writer.shouldPrint(solver.step))
            writer.printProgress(solver.step, solver.time);

        if (writer.shouldWrite(solver.step)) {
            writer.writeFields(c,   solver.step, solver.time);
            writer.writeFields(eta, solver.step, solver.time);
        }
    }

    std::cout << "Done.\n";
    return 0;
}
