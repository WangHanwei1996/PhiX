/***********************************************************************\
 *
 *  GFA_binary — Cahn-Hilliard + Allen-Cahn Solver, Double-Well (2D)
 *
 *  Author : Wang Hanwei
 *  Email  : wanghanweibnds2015@gmail.com
 *
 *  Description  [doc/modeling_stage5.md]
 *  --------------------------------------
 *  Binary Cu-Zr alloy with one non-conserved order parameter phi
 *  (phi=0: liquid, phi=1: B2-CuZr solid).  Free energy density:
 *
 *      f = f_L(c,T)[1-h(phi)] + f_S(c,T)h(phi)
 *        + w_phi*g(phi) + (eps^2/2)|grad phi|^2
 *
 *      h(phi) = phi^3(6phi^2 - 15phi + 10)      [interpolation]
 *      g(phi) = phi^2(1-phi)^2                  [double well]
 *
 *  f_L(c,T), f_S(c,T), dfL/dc(c,T) and dfS/dc(c,T) are lookup tables.
 *  Both phases are c-dependent, so mu carries BOTH contributions
 *  (liquid AND solid).  dfL/dc and dfS/dc each come from their own
 *  analytic table — NOT finite-differenced from f_L/f_S.
 *
 *  Temperature follows a linear cooling protocol (uniform in space):
 *
 *      T(t) = T_start - cooling_rate * t,   clamped to the table T-range
 *
 *  cooling_rate = 0 recovers the isothermal case.
 *
 *  Governing equations (NOTE: no |grad c|^2 energy ⇒ mu is purely
 *  pointwise, no -kappa_c*lap(c) term):
 *
 *      mu      = dfL/dc*(1-h(phi)) + dfS/dc*h(phi)
 *      dc/dt   = M_c * lap(mu)
 *      dphi/dt = -M_phi(T)[(f_S-f_L)*h'(phi) + w_phi*g'(phi) - eps^2*lap(phi)]
 *
 *  The Allen-Cahn mobility is Arrhenius-activated (stage 5):
 *
 *      M_phi(T) = M_phi_pref * exp(-Q_phi / (R_gas*T))
 *
 *  with M_phi_pref, Q_phi, R_gas read from config (doc values:
 *  22.1 m^3/(J*s), 140e3 J/mol, 8.314 J/(mol*K)), evaluated once per
 *  step from the current T(t) (uniform in space).
 *
 *  Time discretisation (canonical PhiX CH idiom, cf. GFA_4ph)
 *  ----------------------------------------------------------
 *  mu is an AUXILIARY field, rebuilt every step via eqMu.computeRHS():
 *    1. mu^n = mu(c^n, phi^n)  → refresh mu ghosts (so the M_c*lap(mu)
 *       stencil in eqC is valid)
 *    2. EquationSystem advances c and phi SIMULTANEOUSLY (explicit
 *       Euler): both RHS are evaluated from the same time level n
 *       before either field is updated.
 *
 \***********************************************************************/

#include "mesh/Mesh.h"
#include "field/ScalarField.h"
#include "boundary/BCFactory.h"
#include "equation/Equation.h"
#include "equation/EquationSystem.h"  // simultaneous coupled-equation update
#include "material/FreeEnergyTable.h"

#include "IO/ConfigFile.h"
#include "IO/FieldIO.h"
#include "IO/OutputWriter.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include <cstdio>   // [TEST-ONLY DIAGNOSTICS — DELETE BEFORE COMMIT]

// ===========================================================================
// Interpolation / double-well functions  [doc/modeling_stage5.md]
//   h(phi)  = phi^3(6phi^2 - 15phi + 10),  h' = 30phi^2(1-phi)^2
//   g(phi)  = phi^2(1-phi)^2,              g' = 2phi(1-phi)(1-2phi)
// ===========================================================================
__host__ __device__ inline double h_func (double x) {
    return x*x*x*(6.0*x*x - 15.0*x + 10.0);
}
__host__ __device__ inline double h_prime(double x) {
    return 30.0*x*x*(1.0 - x)*(1.0 - x);
}
__host__ __device__ inline double g_prime(double x) {
    return 2.0*x*(1.0 - x)*(1.0 - 2.0*x);
}

// ============================================================================
// [TEST-ONLY DIAGNOSTICS — DELETE BEFORE COMMIT] vvvvvvvvvvvvvvvvvvvvvvvvvvvvv
//
//  Domain integrals of a field over the physical cells (CPU, 2D):
//    signedInt = ∫ f dV     — net contribution (sign = driving direction)
//    absInt    = ∫ |f| dV   — activity magnitude (used for the share %,
//                             because e.g. ∫lap(phi)dV ≡ 0 on periodic BCs
//                             even when the term is locally dominant)
//  Call after downloadCurrFromDevice().
// ============================================================================
struct DiagInt { double signedInt; double absInt; };

static DiagInt diagIntegrate(const PhiX::ScalarField& f, const PhiX::Mesh& mesh)
{
    DiagInt r{0.0, 0.0};
    for (int j = 0; j < mesh.n[1]; ++j)
        for (int i = 0; i < mesh.n[0]; ++i) {
            const double v = f.curr[f.index(i, j, 0)];
            r.signedInt += v;
            r.absInt    += (v < 0.0 ? -v : v);
        }
    const double dV = mesh.d[0] * mesh.d[1];
    r.signedInt *= dV;
    r.absInt    *= dV;
    return r;
}
// [TEST-ONLY DIAGNOSTICS — DELETE BEFORE COMMIT] ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
    //  c   — conserved composition          (Cahn-Hilliard)
    //  phi — non-conserved order parameter  (Allen-Cahn)
    //  mu  — chemical potential, auxiliary: recomputed from (c,phi) every
    //        step, so it is neither read at restart nor written to output.
    ScalarField c  (mesh, "c",   /*ghost=*/1);
    ScalarField phi(mesh, "phi", /*ghost=*/1);
    ScalarField mu (mesh, "mu",  /*ghost=*/1);

    c.fill(0);
    phi.fill(0);
    mu.fill(0);

    const std::string start_from = cfg["initialize"]["start_from"];
    const int         start_step = IO::resolveStartStep(start_from);

    IO::initField(c,   start_step);
    IO::initField(phi, start_step);

    c.allocDevice();    c.uploadAllToDevice();
    phi.allocDevice();  phi.uploadAllToDevice();
    mu.allocDevice();   mu.uploadAllToDevice();

    // === 4. Boundary conditions ==============================================
    auto  bcSet = buildBCs(mesh, cfg["boundary_conditions"]);
    auto& bcs   = bcSet.ptrs;

    // === 5. Physical constants ===============================================
    const double eps_sq = cfg["constants"]["eps_sq"];  // eps^2, |grad phi|^2 coefficient
    const double M_c    = cfg["constants"]["M_c"];     // CH composition mobility
    const double w_phi  = cfg["constants"]["w_phi"];   // double-well barrier height

    // AC mobility M_phi is NOT a constant — Arrhenius form [doc/modeling_stage5.md]:
    //   M_phi(T) = M_phi_pref * exp(-Q_phi/(R_gas*T)),  evaluated per step in the loop.
    const double R_gas      = cfg["constants"]["R_gas"];       // [J/(mol·K)] gas constant
    const double M_phi_pref = cfg["constants"]["M_phi_pref"];  // [m^3/(J·s)] pre-exponential factor
    const double Q_phi      = cfg["constants"]["Q_phi"];       // [J/mol]     activation energy
    auto M_phi_of_T = [&](double T) {
        return M_phi_pref * std::exp(-Q_phi / (R_gas * T));
    };

    // Linear cooling protocol: T(t) = T_start - cooling_rate * t
    const double T_start      = cfg["constants"]["T_start"];       // initial temperature [K]
    const double cooling_rate = cfg["constants"]["cooling_rate"];  // [K / unit time], 0 = isothermal

    // === 5b. Free-energy tables (f_L, f_S, dfL/dc, dfS/dc) ===================
    //  dfL/dc and dfS/dc are each loaded as their own table and evaluated via
    //  .f() — the stored value IS the analytic derivative (not FD'd from f).
    using namespace PhiX::Material;
    FreeEnergyTable tableL  = FreeEnergyTable::fromFile(cfg["tables"]["fL"]);
    FreeEnergyTable tableS  = FreeEnergyTable::fromFile(cfg["tables"]["fS"]);
    FreeEnergyTable tableDL = FreeEnergyTable::fromFile(cfg["tables"]["dfLdc"]);
    FreeEnergyTable tableDS = FreeEnergyTable::fromFile(cfg["tables"]["dfSdc"]);
    tableL .allocDevice();  tableL .uploadToDevice();
    tableS .allocDevice();  tableS .uploadToDevice();
    tableDL.allocDevice();  tableDL.uploadToDevice();
    tableDS.allocDevice();  tableDS.uploadToDevice();
    const FreeEnergyTableView feL  = tableL .deviceView();
    const FreeEnergyTableView feS  = tableS .deviceView();
    const FreeEnergyTableView fdLc = tableDL.deviceView();
    const FreeEnergyTableView fdSc = tableDS.deviceView();

    // === 6. Equations (RHSExpr DSL) ==========================================
    //  T enters the RHS functors by value, so the two T-dependent RHSs
    //  (eqMu, eqPhi) are REBUILT each step inside the time loop with the
    //  current T(t) captured — setRHS replaces the stored expression and
    //  is a cheap host-side operation (no GPU work).  Only the
    //  T-independent eqC is configured once here.

    // --- 6a. AUXILIARY: mu = dfL/dc*(1-h) + dfS/dc*h -------------------------
    //   dfL/dc and dfS/dc looked up in their analytic-derivative tables.
    //   Both phases c-dependent ⇒ mu carries liquid AND solid terms; in the
    //   solid (phi=1) mu = dfS/dc so c is driven toward the f_S minimum (cb).
    //   Purely pointwise (no |grad c|^2 energy ⇒ no lap(c) part).
    //   Evaluated each step by eqMu.computeRHS(mu) — never time-advanced.
    //   RHS set in the loop (T-dependent).
    Equation eqMu(mu, "CH_mu");

    // --- 6b. TRANSIENT: dc/dt = M_c * lap(mu) --------------------------------
    Equation eqC(c, "CH_c");
    eqC.setRHS(M_c * lap(mu));

    // --- 6c. TRANSIENT: dphi/dt = -M_phi*((f_S-f_L)h' + w_phi*g' - eps^2*lap(phi))
    //   f_S - f_L obtained from lookup tables via FreeEnergyTableView::f.
    //   RHS set in the loop (T-dependent).
    Equation eqPhi(phi, "AC_phi");

    // ========================================================================
    // [TEST-ONLY DIAGNOSTICS — DELETE BEFORE COMMIT] vvvvvvvvvvvvvvvvvvvvvvvvv
    //
    //  phi-equation term monitor.  The three dphi/dt contributions
    //
    //      bulk = -M_phi*(f_S - f_L)*h'(phi)    (T-dependent → RHS rebuilt
    //                                            at every print in the loop)
    //      dw   = -M_phi*w_phi*g'(phi)
    //      grad = +M_phi*eps_sq*lap(phi)
    //
    //  are each evaluated into a scratch field at print time and reduced to
    //  ONE number per term by domain integration (see diagIntegrate above).
    //  Sign: >0 pushes ∫phi up (toward solid), <0 toward liquid.
    //  bulk + dw + grad = ∫(dphi/dt)dV = d/dt ∫phi dV.
    // ========================================================================
    ScalarField diagBulk(mesh, "diag_bulk", /*ghost=*/1);
    ScalarField diagDW  (mesh, "diag_dw",   /*ghost=*/1);
    ScalarField diagGrad(mesh, "diag_grad", /*ghost=*/1);
    diagBulk.fill(0);  diagBulk.allocDevice();  diagBulk.uploadAllToDevice();
    diagDW  .fill(0);  diagDW  .allocDevice();  diagDW  .uploadAllToDevice();
    diagGrad.fill(0);  diagGrad.allocDevice();  diagGrad.uploadAllToDevice();

    Equation eqDiagBulk(diagBulk, "diag_bulk");   // all three RHS set per print
    Equation eqDiagDW  (diagDW,   "diag_dw");     // (need current T and M_phi(T))
    Equation eqDiagGrad(diagGrad, "diag_grad");
    // [TEST-ONLY DIAGNOSTICS — DELETE BEFORE COMMIT] ^^^^^^^^^^^^^^^^^^^^^^^^^

    // === 7. Coupled system — SIMULTANEOUS update =============================
    //  c and phi advance together (explicit Euler): both RHS are computed
    //  from the same time level before any field changes.  mu stays outside
    //  the system as an auxiliary field.  add() registers the BCs that
    //  sys.advance() re-applies to each unknown before evaluating the RHS.
    EquationSystem sys(dt, TimeScheme::EULER);
    sys.add(eqC,   bcs);
    sys.add(eqPhi, bcs);
    sys.step = start_step;
    sys.time = start_step * dt;

    // === 8. Output & time loop ===============================================
    IO::OutputWriter writer(cfg["output"]);

    if (start_step == 0) {
        writer.writeFields(c,   0, sys.time);
        writer.writeFields(phi, 0, sys.time);
        std::cout << "Starting CH+AC simulation ("
                  << nSteps << " steps, dt=" << dt << ")\n";
    } else {
        std::cout << "Resuming CH+AC simulation from step " << start_step
                  << " (t=" << start_step * dt << "), "
                  << nSteps - start_step << " steps remaining, dt=" << dt
                  << "\n";
    }

    writer.resetTimer();

    for (int s = start_step; s < nSteps; ++s) {
        // (a) Current temperature from the linear cooling protocol,
        //     clamped to the table range (lookup would clamp anyway —
        //     this just makes the saturation explicit).
        double T = T_start - cooling_rate * sys.time;
        if (T < tableL.TMin()) T = tableL.TMin();
        if (T > tableL.TMax()) T = tableL.TMax();

        // Arrhenius AC mobility at the current temperature  [stage 5]
        const double M_phi = M_phi_of_T(T);

        // (b) Rebuild the T-dependent RHSs with T captured by value.
        eqMu.setRHS(
            pw(c, phi, PHIX_FN (double c_val, double phi_val) {
                double h      = h_func(phi_val);
                double dfL_dc = fdLc.f(c_val, T);   // analytic ∂f_L/∂c table
                double dfS_dc = fdSc.f(c_val, T);   // analytic ∂f_S/∂c table
                return dfL_dc * (1.0 - h) + dfS_dc * h;
            })
        );
        eqPhi.setRHS(
            pw(phi, c, PHIX_FN (double phi_val, double c_val) {
                double fS_fL = feS.f(c_val, T) - feL.f(c_val, T);
                double bulk  = fS_fL * h_prime(phi_val);
                double dw    = w_phi * g_prime(phi_val);
                return -M_phi * (bulk + dw);
            })
            + M_phi * eps_sq * lap(phi)
        );

        // (c) Build mu^n from the time-n (c, phi) — pointwise, so no c-ghost
        //     refresh is needed — then refresh mu ghosts for the M_c*lap(mu)
        //     stencil in eqC.
        eqMu.computeRHS(mu);
        for (auto* bc : bcs) bc->applyOnGPU(mu);

        // (d) Advance c and phi simultaneously from the frozen (c,phi,mu)^n.
        sys.advance();

        // (e) Output
        if (writer.shouldPrint(sys.step)) {
            writer.printProgress(sys.step, sys.time);
            std::cout << "             T=" << T << " K  M_phi=" << M_phi << "\n";

            // [TEST-ONLY DIAGNOSTICS — DELETE BEFORE COMMIT] vvvvvvvvvvvvvvvvv
            //  Evaluate the three phi-RHS terms at the just-updated state
            //  (and this step's T, M_phi), integrate, print value + share.
            for (auto* bc : bcs) bc->applyOnGPU(phi);   // fresh ghosts for lap(phi)
            eqDiagBulk.setRHS(
                pw(phi, c, PHIX_FN (double phi_val, double c_val) {
                    double fS_fL = feS.f(c_val, T) - feL.f(c_val, T);
                    return -M_phi * fS_fL * h_prime(phi_val);
                })
            );
            eqDiagDW.setRHS(
                pw(phi, PHIX_FN (double phi_val) {
                    return -M_phi * w_phi * g_prime(phi_val);
                })
            );
            eqDiagGrad.setRHS( M_phi * eps_sq * lap(phi) );
            eqDiagBulk.computeRHS(diagBulk);
            eqDiagDW  .computeRHS(diagDW);
            eqDiagGrad.computeRHS(diagGrad);
            diagBulk.downloadCurrFromDevice();
            diagDW  .downloadCurrFromDevice();
            diagGrad.downloadCurrFromDevice();

            const DiagInt Ib = diagIntegrate(diagBulk, mesh);
            const DiagInt Iw = diagIntegrate(diagDW,   mesh);
            const DiagInt Ig = diagIntegrate(diagGrad, mesh);
            const double  A  = Ib.absInt + Iw.absInt + Ig.absInt;
            const double  s  = (A > 0.0) ? 100.0 / A : 0.0;
            std::printf("             [diag] dphi/dt:  bulk=%+.4e (%5.1f%%)"
                        "  dw=%+.4e (%5.1f%%)  grad=%+.4e (%5.1f%%)"
                        "  total=%+.4e\n",
                        Ib.signedInt, s * Ib.absInt,
                        Iw.signedInt, s * Iw.absInt,
                        Ig.signedInt, s * Ig.absInt,
                        Ib.signedInt + Iw.signedInt + Ig.signedInt);
            // [TEST-ONLY DIAGNOSTICS — DELETE BEFORE COMMIT] ^^^^^^^^^^^^^^^^^
        }

        if (writer.shouldWrite(sys.step)) {
            writer.writeFields(c,   sys.step, sys.time);
            writer.writeFields(phi, sys.step, sys.time);
        }
    }

    std::cout << "Done.\n";
    return 0;
}
