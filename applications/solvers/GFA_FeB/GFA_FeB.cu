/***********************************************************************\
 *
 *  GFA_FeB — Cahn-Hilliard + two-order-parameter Allen-Cahn (2D)
 *
 *  Author : Wang Hanwei
 *  Email  : wanghanweibnds2015@gmail.com
 *
 *  Description  [doc/calibration_plan.md, doc/modeling_stage6.md]
 *  --------------------------------------
 *  Binary Fe-B alloy with TWO non-conserved order parameters:
 *    phi — crystalline Fe2B line compound (phi=0 liquid, phi=1 crystal)
 *    eta — amorphous / glass    (glass-vs-crystal competition)
 *
 *  Adapted from GFA_evo (Cu-Zr).  Model lineage: Wang & Napolitano,
 *  Metall. Mater. Trans. A 43 (2012) 2662 (Cu-Zr CF-MPF) -> Wu, Wang,
 *  Zeng, Liu, Comput. Mater. Sci. 108 (2015) 27 (Fe-B母本).  The kernels
 *  are material-agnostic: the Fe-B change lives in the config + CALPHAD
 *  tables, NOT in the code.  Crystal phase = stable, congruently-melting
 *  Fe2B (line compound at c_B = 1/3), chosen so the equilibrium phase-
 *  diagram calibration is well-posed (Fe3B is metastable).
 *
 *  Free energy density (crystalline part, same as stage 5):
 *
 *      f = f_L(c,T)[1-h(phi)] + f_S(c,T)h(phi)
 *        + w_phi*g(phi) + (eps^2/2)|grad phi|^2
 *
 *      h(x) = x^3(6x^2 - 15x + 10)              [interpolation]
 *      g(x) = x^2(1-x)^2                        [double well]
 *
 *  f_L(c,T), f_S(c,T) and dfL/dc(c,T) are CALPHAD tables for Fe-B (CSV;
 *  build from an Fe-B assessment — Tokunaga et al., Calphad 28 (2004)
 *  354, or Hallemans et al., Z. Metallkd. 85 (1994) 676).  The f_S table
 *  is the stoichiometric Fe2B phase (single c column at c_B = 1/3); to give
 *  it a composition dependence the solid free energy is reconstructed as
 *      f_S(c,T) = f_S^table(c_s,T) + rho_s^2*(c-c_s)^2   (dfS/dc = 2*rho_s^2*(c-c_s))
 *  and this SAME f_S(c) is used in BOTH mu AND the phi driving force, so
 *  off-stoichiometric solidification is penalised consistently (a c-independent
 *  table f_S in the phi term would crystallise every composition and the melt
 *  would fully solidify).  dfL/dc comes from its own analytic table.
 *
 *  Temperature follows a linear cooling protocol (uniform in space):
 *
 *      T(t) = T_start - cooling_rate * t,   clamped to the table T-range
 *
 *  Governing equations (stage 6 — phi gains eta-coupling, eta is new):
 *
 *      mu      = dfL/dc*(1-h(phi)) + dfS/dc*h(phi),  dfS/dc = 2*rho_s^2*(c-c_s)
 *      dc/dt   = div( M_c(phi) * grad(mu) ),   M_c(phi)=h(phi)*M_c_S+(1-h(phi))*M_c_L
 *                (conservative face-flux; degenerate/phase-dependent mobility)
 *
 *      dphi/dt = -M_phi  [ (f_S - f_L - h(eta)*dfAmL)*h'(phi)
 *                           + w_phi*g'(phi) + 2*w_ex*eta^2*phi
 *                           - eps^2*lap(phi) ]
 *
 *      deta/dt = -M_eta  [ (1 - h(phi))*h'(eta)*dfAmL
 *                           + w_eta*g'(eta) + 2*w_ex*eta*phi^2
 *                           - beta^2*lap(eta) ]
 *
 *  Mobilities:
 *    M_phi    = const, from config.  (The earlier Arrhenius form
 *               M_phi_pref*exp(-Q_phi/(R_gas*T)) was an error in the source
 *               paper — Q_phi/M_phi_pref removed; R_gas kept only for dfAmL.)
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
#include "operators/FaceOps.h"        // faceGradGPU/interpGPU/facePWGPU/divFace (variable-mobility CH)
#include "material/FreeEnergyTable.h"

#include "IO/ConfigFile.h"
#include "IO/FieldIO.h"
#include "IO/OutputWriter.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <memory>

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
    const double M_c     = cfg["constants"]["M_c"];      // CH composition mobility (legacy / fallback)
    //  Phase-dependent (degenerate) CH mobility:
    //    M_c(phi) = h(phi)*M_c_S + (1-h(phi))*M_c_L    [liquid M_c_L, solid M_c_S]
    //  so D_L=M_c_L*chi_L and D_S=M_c_S*chi_S are calibrated independently.
    //  Missing keys fall back to the single M_c (⇒ constant mobility = old behaviour).
    const double M_c_L   = cfg["constants"].value("M_c_L", M_c);  // liquid composition mobility
    const double M_c_S   = cfg["constants"].value("M_c_S", M_c);  // solid  composition mobility
    const double w_phi   = cfg["constants"]["w_phi"];    // phi double-well barrier height

    // Solid free-energy curvature & well composition for mu's solid term:
    //   ∂f_S/∂c = 2*rho_s^2*(c - c_s);  added to mu below as +∂f_S/∂c*h(phi).
    //   REQUIRED keys (use .at() so a missing key throws a clear
    //   "key 'rho_s' not found" — NO silent default).  Set rho_s=0 in the
    //   config to explicitly disable the solid term.
    const double rho_s   = cfg["constants"].at("rho_s");  // sqrt of f_S c-curvature (f_S = rho_s^2*(c-c_s)^2 + ...)
    const double c_s     = cfg["constants"].at("c_s");    // f_S minimum composition (Fe2B: c_B = 1/3 ≈ 0.3333)

    // Stage-6 amorphous order parameter eta + phi-eta coupling
    const double M_eta   = cfg["constants"]["M_eta"];    // [m^3/(J·s)] eta AC mobility (constant)
    const double w_eta   = cfg["constants"]["w_eta"];    // eta double-well barrier height
    const double beta_sq = cfg["constants"]["beta_sq"];  // beta^2, |grad eta|^2 coefficient
    const double w_ex    = cfg["constants"]["w_ex"];     // phi-eta cross-coupling coefficient

    // R_gas is kept only for dfAmL (the amorphous→liquid driving factor below).
    const double R_gas = cfg["constants"]["R_gas"];   // [J/(mol·K)] gas constant (used by dfAmL)

    // AC mobility M_phi is a CONSTANT read from config.  The earlier Arrhenius
    // form M_phi = M_phi_pref*exp(-Q_phi/(R_gas*T)) [stage-5 doc] was an error in
    // the source paper, so Q_phi/M_phi_pref are gone; the phi equation takes a
    // single constant coefficient M_phi.
    const double M_phi = cfg["constants"]["M_phi"];   // [m^3/(J·s)] AC mobility (constant)

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

    // Optional temperature-dependent CH mobilities M_c_L(T), M_c_S(T).
    //  Host-eval only (T is uniform ⇒ M_c_L/M_c_S are host scalars per step, no GPU
    //  upload).  If tables.McL / tables.McS paths are given, the mobilities FREEZE on
    //  cooling (Arrhenius D_S/D_L → M_c=D/χ; see calibration-5.0); otherwise the
    //  constant M_c_L/M_c_S from constants are used (backward-compatible).
    const std::string mcL_path = cfg["tables"].value("McL", std::string(""));
    const std::string mcS_path = cfg["tables"].value("McS", std::string(""));
    std::unique_ptr<FreeEnergyTable> mcLtab, mcStab;
    if (!mcL_path.empty()) mcLtab = std::make_unique<FreeEnergyTable>(FreeEnergyTable::fromFile(mcL_path));
    if (!mcS_path.empty()) mcStab = std::make_unique<FreeEnergyTable>(FreeEnergyTable::fromFile(mcS_path));
    if (mcLtab || mcStab)
        std::cout << "  M_c(T) from table: McL=" << (mcLtab ? "yes" : "const")
                  << "  McS=" << (mcStab ? "yes" : "const") << "\n";

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

    // --- 6b. TRANSIENT: dc/dt = ∇·(M_c(phi)∇mu) — conservative variable-mobility CH
    //   Finite-volume face flux:  J = M_c(phi_face)·∇mu on faces,  dc/dt = divFace(J).
    //   M_c(phi) = h(phi)*M_c_S + (1-h(phi))*M_c_L, kept INSIDE the divergence so solute
    //   is conserved to machine precision.  (A naive M_c(phi)*lap(mu) leaks the dropped
    //   ∇M·∇mu term at the interface — measured ~4% over 2e4 steps at 100x contrast.)
    //   With M_c_L == M_c_S it reduces EXACTLY to the constant-mobility 5-point Laplacian.
    //   (interp() gives one-sided M_c at domain boundary faces ⇒ keep the periodic
    //    boundary inside a uniform-phi region so M_c is single-valued there.)
    FaceField mu_gx(mesh, 0, "mu_gx");  FaceField phi_fx(mesh, 0, "phi_fx");  FaceField jx(mesh, 0, "jx");
    FaceField mu_gy(mesh, 1, "mu_gy");  FaceField phi_fy(mesh, 1, "phi_fy");  FaceField jy(mesh, 1, "jy");
    auto allocFace = [](FaceField& f){ f.fill(0.0); f.allocDevice(); f.uploadToDevice(); };
    allocFace(mu_gx); allocFace(phi_fx); allocFace(jx);
    allocFace(mu_gy); allocFace(phi_fy); allocFace(jy);
    Equation eqC(c, "CH_c");
    eqC.setRHS(1.0 * divFace(jx, jy));   // jx,jy rebuilt from mu^n,phi^n each step (see loop)

    // --- 6c. TRANSIENT: dphi/dt (crystalline order parameter) ----------------
    //   -M_phi*[ (f_S-f_L - h(eta)*dfAmL)*h'(phi) + w_phi*g'(phi)
    //            + 2*w_ex*eta^2*phi - eps^2*lap(phi) ]
    //   f_S = f_S^table(c_s) + rho_s^2*(c-c_s)^2  (consistent with mu's dfS/dc
    //   parabola → penalises off-stoichiometric solid); f_L from table.
    //   Couples to eta.  RHS set in the loop.
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

        // Amorphous→liquid driving factor at this T (host scalar)  [stage 6]
        const double dfAmL = deltaF_AmToL(T);
        // T-dependent CH mobilities at this T (host scalars; table if given, else constant).
        //   nc=2 table ⇒ .f() is c-independent, returns M_c(T).
        const double M_c_L_T = mcLtab ? mcLtab->f(0.5, T) : M_c_L;
        const double M_c_S_T = mcStab ? mcStab->f(0.5, T) : M_c_S;

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
                // f_S(c) consistent with mu's solid parabola: table baseline
                // (Fe2B minimum at c_s) + rho_s^2*(c-c_s)^2  ⇒ dfS/dc = 2*rho_s^2*(c-c_s).
                // Without the parabola the c-independent table f_S makes the bulk
                // driving force favour crystallisation at EVERY composition, so the
                // melt fully solidifies even where c cannot form Fe2B.  The parabola
                // raises f_S away from c_s, stopping off-stoichiometric solidification.
                double fS    = feS.f(c_val, T) + rho_s * rho_s * (c_val - c_s) * (c_val - c_s);
                double fS_fL = fS - feL.f(c_val, T);
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

        // (c) Build mu^n from the time-n (c, phi) — pointwise — then refresh mu
        //     ghosts (needed by faceGrad(mu) at the boundary faces of the flux below).
        eqMu.computeRHS(mu);
        for (auto* bc : bcs) bc->applyOnGPU(mu);

        // (c2) Variable-mobility CH flux  J = M_c(phi)·∇mu  on faces (from mu^n, phi^n),
        //      consumed by eqC's divFace(jx,jy) in sys.advance() ⇒ dc/dt = ∇·(M_c(phi)∇mu).
        auto Mc_flux = PHIX_FN (double gmu, double pf) {
            double hh = h_func(pf);
            return (hh * M_c_S_T + (1.0 - hh) * M_c_L_T) * gmu;
        };
        faceGradGPU(mu, 0, mu_gx);  interpGPU(phi, 0, phi_fx);  facePWGPU(jx, mu_gx, phi_fx, Mc_flux);
        faceGradGPU(mu, 1, mu_gy);  interpGPU(phi, 1, phi_fy);  facePWGPU(jy, mu_gy, phi_fy, Mc_flux);

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
                        double fS    = feS.f(c_val, T) + rho_s * rho_s * (c_val - c_s) * (c_val - c_s);
                        double fS_fL = fS - feL.f(c_val, T);
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
