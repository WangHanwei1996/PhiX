/***********************************************************************\
 *
 *  GFA_evo — Cahn-Hilliard + two-order-parameter Allen-Cahn (2D)
 *
 *  Author : Wang Hanwei
 *  Email  : wanghanweibnds2015@gmail.com
 *
 *  Description  [doc/modeling_stage6.md]
 *  --------------------------------------
 *  Binary Cu-Zr alloy with TWO non-conserved order parameters:
 *    phi — crystalline B2-CuZr  (phi=0 liquid, phi=1 crystal)
 *    eta — amorphous / glass    (added in stage 6)
 *
 *  Free energy density (crystalline part, same as stage 5):
 *
 *      f = f_L(c,T)[1-h(phi)] + f_S(c,T)h(phi)
 *        + w_phi*g(phi) + (eps^2/2)|grad phi|^2
 *
 *      h(x) = x^3(6x^2 - 15x + 10)              [interpolation]
 *      g(x) = x^2(1-x)^2                        [double well]
 *
 *  f_L(c,T), f_S(c,T) and dfL/dc(c,T) are REAL CALPHAD tables from
 *  data/material_properties/Cu-Zr/ (CSV).  f_S is the stoichiometric
 *  B2-CuZr phase (single c column at c=0.5) ⇒ ∂f_S/∂c does not exist
 *  and mu carries only the liquid contribution.  dfL/dc comes from its
 *  own analytic table (dfdc_L_table.csv).
 *
 *  Temperature follows a linear cooling protocol (uniform in space):
 *
 *      T(t) = T_start - cooling_rate * t,   clamped to the table T-range
 *
 *  Governing equations (stage 6 — phi gains eta-coupling, eta is new):
 *
 *      mu      = dfL/dc*(1-h(phi)) + dfS/dc*h(phi),  dfS/dc = 2*rho_s^2*(c-c_s)
 *      dc/dt   = M_c * lap(mu)
 *
 *      dphi/dt = -M_phi(T)[ (f_S - f_L - h(eta)*dfAmL)*h'(phi)
 *                           + w_phi*g'(phi) + 2*w_ex*eta^2*phi
 *                           - eps^2*lap(phi) ]
 *
 *      deta/dt = -M_eta  [ (1 - h(phi))*h'(eta)*dfAmL
 *                           + w_eta*g'(eta) + 2*w_ex*eta*phi^2
 *                           - beta^2*lap(eta) ]
 *
 *  Mobilities:
 *    M_phi(T) = M_phi_pref * exp(-Q_phi/(R_gas*T))   [Arrhenius, stage 5]
 *    M_eta    = const, from config.
 *
 *  Amorphous→liquid driving factor (depends on T only, uniform in space):
 *
 *      dfAmL = R_gas*T*ln(1+alpha)*f(tau) / Vm,   tau = T/T_g
 *
 *    with the piecewise f(tau) of stage 6 (see f_tau() below);
 *    T_g, alpha, Vm from config.  The doc form R*T*ln(1+alpha)*f(tau) is
 *    per mole [J/mol]; dividing by the molar volume Vm gives [J/m^3], so
 *    it is consistent with f_S - f_L (the tables store Gm/Vm in J/m^3).
 *    Computed once per step as a host scalar and captured by value into
 *    the phi/eta RHS functors.
 *
 *  Time discretisation (canonical PhiX CH idiom, cf. GFA_4ph)
 *  ----------------------------------------------------------
 *  mu is an AUXILIARY field, rebuilt every step via eqMu.computeRHS();
 *  then EquationSystem advances c, phi and eta SIMULTANEOUSLY (explicit
 *  Euler): every RHS is evaluated from the same time level n before any
 *  field is updated.
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
// Interpolation / double-well functions  [doc/modeling_stage6.md]
//   h(x)  = x^3(6x^2 - 15x + 10),  h' = 30x^2(1-x)^2
//   g(x)  = x^2(1-x)^2,            g' = 2x(1-x)(1-2x)
//   (used for both phi and eta)
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

// ===========================================================================
// f(tau) — amorphous→liquid temperature factor  [doc/modeling_stage6.md]
//   tau = T/T_g.  Piecewise polynomial; enters dfAmL = R*T*ln(1+alpha)*f(tau).
//   Host-only: evaluated once per step to build a scalar driving force.
// ===========================================================================
static inline double f_tau(double tau)
{
    if (tau < 1.0) {
        return 1.0
             - 9.9167285e-1  * std::pow(tau, -1.0)
             - 1.11737779e-1 * std::pow(tau,  3.0)
             - 4.96612349e-3 * std::pow(tau,  9.0)
             - 1.11737779e-3 * std::pow(tau, 15.0);
    } else {
        return - 1.05443689e-1 * std::pow(tau,  -5.0)
               - 3.34741816e-3 * std::pow(tau, -15.0)
               - 7.02957924e-4 * std::pow(tau, -25.0);
    }
}

// ============================================================================
// DIAGNOSTICS (config-gated via the "diagnostics" section) vvvvvvvvvvvvvvvvvvv
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
// DIAGNOSTICS (config-gated) ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
    //  c   — conserved composition           (Cahn-Hilliard)
    //  phi — crystalline order parameter     (Allen-Cahn)
    //  eta — amorphous order parameter       (Allen-Cahn, stage 6)
    //  mu  — chemical potential, auxiliary: recomputed from (c,phi) every
    //        step, so it is neither read at restart nor written to output.
    ScalarField c  (mesh, "c",   /*ghost=*/1);
    ScalarField phi(mesh, "phi", /*ghost=*/1);
    ScalarField eta(mesh, "eta", /*ghost=*/1);
    ScalarField mu (mesh, "mu",  /*ghost=*/1);

    c.fill(0);
    phi.fill(0);
    eta.fill(0);
    mu.fill(0);

    const std::string start_from = cfg["initialize"]["start_from"];
    const int         start_step = IO::resolveStartStep(start_from);

    IO::initField(c,   start_step);
    IO::initField(phi, start_step);
    IO::initField(eta, start_step);

    c.allocDevice();    c.uploadAllToDevice();
    phi.allocDevice();  phi.uploadAllToDevice();
    eta.allocDevice();  eta.uploadAllToDevice();
    mu.allocDevice();   mu.uploadAllToDevice();

    // === 4. Boundary conditions ==============================================
    auto  bcSet = buildBCs(mesh, cfg["boundary_conditions"]);
    auto& bcs   = bcSet.ptrs;

    // === 5. Physical constants ===============================================
    const double eps_sq  = cfg["constants"]["eps_sq"];   // eps^2,  |grad phi|^2 coefficient
    const double M_c     = cfg["constants"]["M_c"];      // CH composition mobility
    const double w_phi   = cfg["constants"]["w_phi"];    // phi double-well barrier height

    // Solid free-energy curvature & well composition for mu's solid term:
    //   ∂f_S/∂c = 2*rho_s^2*(c - c_s);  added to mu below as +∂f_S/∂c*h(phi).
    //   REQUIRED keys (use .at() so a missing key throws a clear
    //   "key 'rho_s' not found" — NO silent default).  Set rho_s=0 in the
    //   config to explicitly disable the solid term.
    const double rho_s   = cfg["constants"].at("rho_s");  // sqrt of f_S c-curvature (f_S = rho_s^2*(c-c_s)^2 + ...)
    const double c_s     = cfg["constants"].at("c_s");    // f_S minimum composition

    // Stage-6 amorphous order parameter eta + phi-eta coupling
    const double M_eta   = cfg["constants"]["M_eta"];    // [m^3/(J·s)] eta AC mobility (constant)
    const double w_eta   = cfg["constants"]["w_eta"];    // eta double-well barrier height
    const double beta_sq = cfg["constants"]["beta_sq"];  // beta^2, |grad eta|^2 coefficient
    const double w_ex    = cfg["constants"]["w_ex"];     // phi-eta cross-coupling coefficient

    // AC mobility M_phi is NOT a constant — Arrhenius form [doc/modeling_stage5.md]:
    //   M_phi(T) = M_phi_pref * exp(-Q_phi/(R_gas*T)),  evaluated per step in the loop.
    const double R_gas      = cfg["constants"]["R_gas"];       // [J/(mol·K)] gas constant
    const double M_phi_pref = cfg["constants"]["M_phi_pref"];  // [m^3/(J·s)] pre-exponential factor
    const double Q_phi      = cfg["constants"]["Q_phi"];       // [J/mol]     activation energy
    auto M_phi_of_T = [&](double T) {
        return M_phi_pref * std::exp(-Q_phi / (R_gas * T));
    };

    // Amorphous→liquid driving factor [doc/modeling_stage6.md]:
    //   dfAmL(T) = R_gas*T*ln(1+alpha)*f(tau) / Vm,  tau = T/T_g  (T only, uniform)
    //   The doc form R*T*ln(1+alpha)*f(tau) is PER MOLE [J/mol]; dividing by the
    //   molar volume Vm converts it to PER VOLUME [J/m^3] so it is consistent
    //   with f_S - f_L (the CALPHAD tables already store Gm/Vm in J/m^3).
    const double T_g   = cfg["constants"]["T_g"];    // [K] glass-transition temperature
    const double alpha = cfg["constants"]["alpha"];  // [-] driving-force amplitude
    const double Vm    = cfg["constants"]["Vm"];     // [m^3/mol] molar volume (liquid, = Vm_0 of f_L table)
    auto deltaF_AmToL = [&](double T) {
        return R_gas * T * std::log(1.0 + alpha) * f_tau(T / T_g) / Vm;
    };

    // Linear cooling protocol: T(t) = T_start - cooling_rate * t
    const double T_start      = cfg["constants"]["T_start"];       // initial temperature [K]
    const double cooling_rate = cfg["constants"]["cooling_rate"];  // [K / unit time], 0 = isothermal

    // === 5b. Free-energy tables (f_L, f_S, dfL/dc) ===========================
    //  dfL/dc is loaded as its own table and evaluated via .f() — the value
    //  stored in dfdc_L_table.csv IS the analytic derivative.
    using namespace PhiX::Material;
    FreeEnergyTable tableL  = FreeEnergyTable::fromFile(cfg["tables"]["fL"]);
    FreeEnergyTable tableS  = FreeEnergyTable::fromFile(cfg["tables"]["fS"]);
    FreeEnergyTable tableDL = FreeEnergyTable::fromFile(cfg["tables"]["dfLdc"]);
    tableL .allocDevice();  tableL .uploadToDevice();
    tableS .allocDevice();  tableS .uploadToDevice();
    tableDL.allocDevice();  tableDL.uploadToDevice();
    const FreeEnergyTableView feL  = tableL .deviceView();
    const FreeEnergyTableView feS  = tableS .deviceView();
    const FreeEnergyTableView fdLc = tableDL.deviceView();

    // === 6. Equations (RHSExpr DSL) ==========================================
    //  T enters the RHS functors by value, so the T-dependent RHSs
    //  (eqMu, eqPhi, eqEta) are REBUILT each step inside the time loop with
    //  the current T(t) / M_phi(T) / dfAmL(T) captured — setRHS replaces the
    //  stored expression and is a cheap host-side operation (no GPU work).
    //  Only the T-independent eqC is configured once here.

    // --- 6a. AUXILIARY: mu = dfL/dc*(1-h) ------------------------------------
    //   dfL/dc is looked up in the analytic-derivative table (fdLc.f).
    //   f_S is stoichiometric ⇒ no ∂f_S/∂c term  [doc/modeling_stage5.md].
    //   Purely pointwise (no |grad c|^2 energy ⇒ no lap(c) part).
    //   Evaluated each step by eqMu.computeRHS(mu) — never time-advanced.
    //   RHS set in the loop (T-dependent).
    Equation eqMu(mu, "CH_mu");

    // --- 6b. TRANSIENT: dc/dt = M_c * lap(mu) --------------------------------
    Equation eqC(c, "CH_c");
    eqC.setRHS(M_c * lap(mu));

    // --- 6c. TRANSIENT: dphi/dt (crystalline order parameter) ----------------
    //   -M_phi*[ (f_S-f_L - h(eta)*dfAmL)*h'(phi) + w_phi*g'(phi)
    //            + 2*w_ex*eta^2*phi - eps^2*lap(phi) ]
    //   f_S - f_L from lookup tables; couples to eta.  RHS set in the loop.
    Equation eqPhi(phi, "AC_phi");

    // --- 6d. TRANSIENT: deta/dt (amorphous order parameter, stage 6) ---------
    //   -M_eta*[ (1-h(phi))*h'(eta)*dfAmL + w_eta*g'(eta)
    //            + 2*w_ex*eta*phi^2 - beta^2*lap(eta) ]
    //   RHS set in the loop (dfAmL is T-dependent).
    Equation eqEta(eta, "AC_eta");

    // ========================================================================
    //  DIAGNOSTICS — per-equation RHS term-share monitor  (config-gated)
    //
    //  Enabled from the optional top-level "diagnostics" section:
    //      "diagnostics": { "diag_phi": true, "diag_eta": true }
    //  Missing section/keys default to OFF (backward compatible).
    //
    //  Each equation's RHS is split into four contributions (same structure
    //  for phi and eta), evaluated into a scratch field at print time and
    //  reduced to ONE number per term by domain integration (diagIntegrate):
    //
    //      dphi/dt: bulk  = -M_phi*(f_S - f_L - h(eta)*dfAmL)*h'(phi)
    //               dw    = -M_phi*w_phi*g'(phi)
    //               cross = -M_phi*2*w_ex*eta^2*phi
    //               grad  = +M_phi*eps_sq*lap(phi)
    //
    //      deta/dt: bulk  = -M_eta*(1 - h(phi))*h'(eta)*dfAmL
    //               dw    = -M_eta*w_eta*g'(eta)
    //               cross = -M_eta*2*w_ex*eta*phi^2
    //               grad  = +M_eta*beta_sq*lap(eta)
    //
    //  Sign: >0 pushes the field's domain integral up, <0 down.
    //  bulk + dw + cross + grad = ∫(dfield/dt)dV = d/dt ∫field dV.
    //  (All RHSs are rebuilt per print: they need current T / M_phi / dfAmL.)
    //  The four scratch fields/equations are SHARED between the phi and eta
    //  passes — each pass fully reduces+prints before the next overwrites them.
    // ========================================================================
    bool diagPhi = false, diagEta = false;
    if (cfg.has("diagnostics")) {
        diagPhi = cfg["diagnostics"].value("diag_phi", false);
        diagEta = cfg["diagnostics"].value("diag_eta", false);
    }
    const bool diagAny = diagPhi || diagEta;

    ScalarField diagBulk (mesh, "diag_bulk",  /*ghost=*/1);
    ScalarField diagDW   (mesh, "diag_dw",    /*ghost=*/1);
    ScalarField diagCross(mesh, "diag_cross", /*ghost=*/1);
    ScalarField diagGrad (mesh, "diag_grad",  /*ghost=*/1);
    diagBulk.fill(0);  diagDW.fill(0);  diagCross.fill(0);  diagGrad.fill(0);
    if (diagAny) {   // only touch the GPU when a monitor is actually enabled
        diagBulk .allocDevice();  diagBulk .uploadAllToDevice();
        diagDW   .allocDevice();  diagDW   .uploadAllToDevice();
        diagCross.allocDevice();  diagCross.uploadAllToDevice();
        diagGrad .allocDevice();  diagGrad .uploadAllToDevice();
    }

    Equation eqDiagBulk (diagBulk,  "diag_bulk");
    Equation eqDiagDW   (diagDW,    "diag_dw");
    Equation eqDiagCross(diagCross, "diag_cross");
    Equation eqDiagGrad (diagGrad,  "diag_grad");

    // Reduce the four just-computed scratch fields to value + %-share, print.
    //  'tag' labels which equation (e.g. "dphi/dt" / "deta/dt").
    auto diagReduceAndPrint = [&](const char* tag) {
        diagBulk .downloadCurrFromDevice();
        diagDW   .downloadCurrFromDevice();
        diagCross.downloadCurrFromDevice();
        diagGrad .downloadCurrFromDevice();
        const DiagInt Ib = diagIntegrate(diagBulk,  mesh);
        const DiagInt Iw = diagIntegrate(diagDW,    mesh);
        const DiagInt Ix = diagIntegrate(diagCross, mesh);
        const DiagInt Ig = diagIntegrate(diagGrad,  mesh);
        const double  A  = Ib.absInt + Iw.absInt + Ix.absInt + Ig.absInt;
        const double  sc = (A > 0.0) ? 100.0 / A : 0.0;
        std::printf("             [diag] %s:  bulk=%+.4e (%5.1f%%)"
                    "  dw=%+.4e (%5.1f%%)  cross=%+.4e (%5.1f%%)"
                    "  grad=%+.4e (%5.1f%%)  total=%+.4e\n",
                    tag,
                    Ib.signedInt, sc * Ib.absInt,
                    Iw.signedInt, sc * Iw.absInt,
                    Ix.signedInt, sc * Ix.absInt,
                    Ig.signedInt, sc * Ig.absInt,
                    Ib.signedInt + Iw.signedInt + Ix.signedInt + Ig.signedInt);
    };

    // === 7. Coupled system — SIMULTANEOUS update =============================
    //  c, phi and eta advance together (explicit Euler): every RHS is computed
    //  from the same time level before any field changes.  mu stays outside
    //  the system as an auxiliary field.  add() registers the BCs that
    //  sys.advance() re-applies to each unknown before evaluating the RHS.
    EquationSystem sys(dt, TimeScheme::EULER);
    sys.add(eqC,   bcs);
    sys.add(eqPhi, bcs);
    sys.add(eqEta, bcs);
    sys.step = start_step;
    sys.time = start_step * dt;

    // === 8. Output & time loop ===============================================
    IO::OutputWriter writer(cfg["output"]);

    if (start_step == 0) {
        writer.writeFields(c,   0, sys.time);
        writer.writeFields(phi, 0, sys.time);
        writer.writeFields(eta, 0, sys.time);
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
        // Amorphous→liquid driving factor at this T (host scalar)  [stage 6]
        const double dfAmL = deltaF_AmToL(T);

        // (b) Rebuild the T-dependent RHSs with T / M_phi / dfAmL by value.
        eqMu.setRHS(
            pw(c, phi, PHIX_FN (double c_val, double phi_val) {
                double h      = h_func(phi_val);
                double dfL_dc = fdLc.f(c_val, T);                    // analytic ∂f_L/∂c table
                double dfS_dc = 2.0 * rho_s * rho_s * (c_val - c_s); // analytic ∂f_S/∂c = 2*rho_s^2*(c-c_s)
                return dfL_dc * (1.0 - h) + dfS_dc * h;
            })
        );
        // dphi/dt — couples to eta (3-field pw: phi, eta, c)
        eqPhi.setRHS(
            pw(phi, eta, c, PHIX_FN (double phi_val, double eta_val, double c_val) {
                double fS_fL = feS.f(c_val, T) - feL.f(c_val, T);
                double bulk  = (fS_fL - h_func(eta_val) * dfAmL) * h_prime(phi_val);
                double dw    = w_phi * g_prime(phi_val);
                double cross = 2.0 * w_ex * eta_val * eta_val * phi_val;
                return -M_phi * (bulk + dw + cross);
            })
            + M_phi * eps_sq * lap(phi)
        );
        // deta/dt — amorphous order parameter (2-field pw: eta, phi)
        eqEta.setRHS(
            pw(eta, phi, PHIX_FN (double eta_val, double phi_val) {
                double bulk  = (1.0 - h_func(phi_val)) * h_prime(eta_val) * dfAmL;
                double dw    = w_eta * g_prime(eta_val);
                double cross = 2.0 * w_ex * eta_val * phi_val * phi_val;
                return -M_eta * (bulk + dw + cross);
            })
            + M_eta * beta_sq * lap(eta)
        );

        // (c) Build mu^n from the time-n (c, phi) — pointwise, so no c-ghost
        //     refresh is needed — then refresh mu ghosts for the M_c*lap(mu)
        //     stencil in eqC.
        eqMu.computeRHS(mu);
        for (auto* bc : bcs) bc->applyOnGPU(mu);

        // (d) Advance c, phi and eta simultaneously from the frozen state^n.
        sys.advance();

        // (e) Output
        if (writer.shouldPrint(sys.step)) {
            writer.printProgress(sys.step, sys.time);
            std::cout << "             T=" << T << " K  M_phi=" << M_phi << "\n";

            // DIAGNOSTICS (config-gated) — evaluate each ENABLED equation's RHS
            //  terms at the just-updated state (this step's T / M_phi / dfAmL),
            //  integrate over the domain, print value + %-share.  phi and eta
            //  reuse the same four scratch fields (phi pass prints before eta
            //  overwrites them).
            if (diagPhi) {
                for (auto* bc : bcs) bc->applyOnGPU(phi);   // fresh ghosts for lap(phi)
                eqDiagBulk.setRHS(
                    pw(phi, eta, c, PHIX_FN (double phi_val, double eta_val, double c_val) {
                        double fS_fL = feS.f(c_val, T) - feL.f(c_val, T);
                        return -M_phi * (fS_fL - h_func(eta_val) * dfAmL) * h_prime(phi_val);
                    })
                );
                eqDiagDW.setRHS(
                    pw(phi, PHIX_FN (double phi_val) {
                        return -M_phi * w_phi * g_prime(phi_val);
                    })
                );
                eqDiagCross.setRHS(
                    pw(eta, phi, PHIX_FN (double eta_val, double phi_val) {
                        return -M_phi * 2.0 * w_ex * eta_val * eta_val * phi_val;
                    })
                );
                eqDiagGrad.setRHS( M_phi * eps_sq * lap(phi) );
                eqDiagBulk .computeRHS(diagBulk);
                eqDiagDW   .computeRHS(diagDW);
                eqDiagCross.computeRHS(diagCross);
                eqDiagGrad .computeRHS(diagGrad);
                diagReduceAndPrint("dphi/dt");
            }
            if (diagEta) {
                for (auto* bc : bcs) bc->applyOnGPU(eta);   // fresh ghosts for lap(eta)
                eqDiagBulk.setRHS(
                    pw(eta, phi, PHIX_FN (double eta_val, double phi_val) {
                        return -M_eta * (1.0 - h_func(phi_val)) * h_prime(eta_val) * dfAmL;
                    })
                );
                eqDiagDW.setRHS(
                    pw(eta, PHIX_FN (double eta_val) {
                        return -M_eta * w_eta * g_prime(eta_val);
                    })
                );
                eqDiagCross.setRHS(
                    pw(eta, phi, PHIX_FN (double eta_val, double phi_val) {
                        return -M_eta * 2.0 * w_ex * eta_val * phi_val * phi_val;
                    })
                );
                eqDiagGrad.setRHS( M_eta * beta_sq * lap(eta) );
                eqDiagBulk .computeRHS(diagBulk);
                eqDiagDW   .computeRHS(diagDW);
                eqDiagCross.computeRHS(diagCross);
                eqDiagGrad .computeRHS(diagGrad);
                diagReduceAndPrint("deta/dt");
            }
        }

        if (writer.shouldWrite(sys.step)) {
            writer.writeFields(c,   sys.step, sys.time);
            writer.writeFields(phi, sys.step, sys.time);
            writer.writeFields(eta, sys.step, sys.time);
        }
    }

    std::cout << "Done.\n";
    return 0;
}
