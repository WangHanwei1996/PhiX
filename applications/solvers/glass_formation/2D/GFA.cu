/***********************************************************************\
 *
 *  Glass Formation Ability (GFA) Solver (2D)
 *
 *  Author : Wang Hanwei
 *  Email  : wanghanweibnds2015@gmail.com
 *
 *  Description
 *  -----------
 *  Coupled Cahn-Hilliard (c) and Allen-Cahn (phi, eta) equations for
 *  simulating glass formation in Fe-B binary alloys.
 *
 *  Variables:
 *    phi  -- crystalline order parameter  (0: liquid/amorphous, 1: solid)
 *    eta  -- amorphous order parameter    (0: liquid,           1: amorphous)
 *    c    -- composition (mole fraction of B)
 *    mu   -- chemical potential  mu = df/dc
 *
 *  Evolution equations:
 *    dphi/dt = -M_phi * (df/dphi - eps^2 * lap(phi))
 *    deta/dt = -M_eta * (df/deta - beta^2 * lap(eta))
 *    dc/dt   = div( M_c * c*(1-c) * grad(mu) )
 *
 *  Free energy density:
 *    f = [1-h(phi)]*[f_L(c,T) + h(eta)*Df_AmL(T)]
 *      + h(phi)*f_S(T)
 *      + w_phi*g(phi) + w_eta*g(eta) + w_ex*phi^2*eta^2
 *
 *  CALPHAD thermodynamics for the Fe-B binary system.
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

// ============================================================================
//  Constants
// ============================================================================
static constexpr double R_gas = 8.314;          // J/(mol·K)

// ============================================================================
//  CALPHAD thermodynamic functions  (Fe-B system, J/mol, T in K)
// ============================================================================

// --- G_B^L : Gibbs energy of liquid B ---
__host__ __device__ inline double G_B_liquid(double T)
{
    if (T <= 500.0) {
        return 40723.275 + 86.843839*T - 15.6641*T*log(T)
             - 6.864515e-3*T*T + 0.618878e-6*T*T*T + 370843.0/T;
    } else if (T <= 2348.0) {
        return 41119.703 + 82.101722*T - 14.9827763*T*log(T)
             - 7.095669e-3*T*T + 0.507347e-6*T*T*T + 335484.0/T;
    } else {
        return 28842.012 + 200.94731*T - 31.4*T*log(T);
    }
}

// --- G_Fe^L : Gibbs energy of liquid Fe ---
__host__ __device__ inline double G_Fe_liquid(double T)
{
    if (T <= 1811.0) {
        double T2 = T*T, T3 = T2*T, T7 = T3*T3*T;
        return 13265.87 + 117.57557*T - 23.5143*T*log(T)
             - 4.39752e-3*T2 - 0.058927e-6*T3 + 773597.0/T
             - 367.516e-23*T7;
    } else {
        return -10838.83 + 291.302*T - 46.0*T*log(T);
    }
}

// --- L_{B,Fe}^L : interaction parameter ---
__host__ __device__ inline double L_BFe_liquid(double T)
{
    return -122861.0 + 14.59*T;
}

// --- G_Fe^{bcc} : Gibbs energy of bcc Fe ---
__host__ __device__ inline double G_Fe_bcc(double T)
{
    if (T <= 1811.0) {
        return 1225.7 + 124.134*T - 23.5143*T*log(T)
             - 4.39752e-3*T*T - 0.058927e-6*T*T*T + 773597.0/T;
    } else {
        double T3 = T*T*T, T9 = T3*T3*T3;
        return -25383.581 + 299.31255*T - 46.0*T*log(T)
             + 2296.03e28 / T9;
    }
}

// --- G_B^{beta} : Gibbs energy of beta-rhombohedral B ---
__host__ __device__ inline double G_B_beta(double T)
{
    if (T <= 1100.0) {
        return -7735.284 + 107.111864*T - 15.6641*T*log(T)
             - 6.864515e-3*T*T + 0.618878e-6*T*T*T + 370843.0/T;
    } else if (T <= 2348.0) {
        return -16649.474 + 184.801744*T - 26.6047*T*log(T)
             - 0.79809e-3*T*T - 0.02556e-6*T*T*T + 1748270.0/T;
    } else if (T <= 3000.0) {
        return -36667.582 + 231.336244*T - 31.5957527*T*log(T)
             - 1.59488e-3*T*T + 0.134719e-6*T*T*T + 11205883.0/T;
    } else {
        return -21530.653 + 222.396264*T - 31.4*T*log(T);
    }
}

// --- Delta G_{Fe3B}^f : formation Gibbs energy of Fe3B ---
__host__ __device__ inline double deltaG_Fe3B_f(double T)
{
    return -77749.0 + 2.59*T;
}

// --- f(tau) : scaled undercooling function ---
__host__ __device__ inline double f_tau_func(double tau)
{
    if (tau <= 1.0) {
        double t3  = tau*tau*tau;
        double t9  = t3*t3*t3;
        double t15 = t9*t3*t3;
        return 1.0
             - 9.9167285e-1  / tau
             - 1.11737779e-1 * t3
             - 4.96612349e-3 * t9
             - 1.11737779e-3 * t15;
    } else {
        double inv  = 1.0 / tau;
        double i5   = inv*inv*inv*inv*inv;
        double i10  = i5*i5;
        double i15  = i10*i5;
        double i25  = i15*i10;
        return -1.05443689e-1 * i5
             -  3.34741816e-3 * i15
             -  7.02957924e-4 * i25;
    }
}

// ============================================================================
//  Interpolation / double-well helpers
// ============================================================================

__host__ __device__ inline double h_func(double x)
{   // h(x) = x^3 (10 - 15x + 6x^2)
    return x*x*x * (10.0 - 15.0*x + 6.0*x*x);
}

__host__ __device__ inline double h_prime(double x)
{   // h'(x) = 30 x^2 (1-x)^2
    return 30.0 * x*x * (1.0-x)*(1.0-x);
}

__host__ __device__ inline double g_prime(double x)
{   // g'(x) = 2x(1-x)(1-2x)       where g(x)=x^2(1-x)^2
    return 2.0 * x * (1.0-x) * (1.0 - 2.0*x);
}

// ============================================================================
//  Composite thermodynamic quantities
// ============================================================================

// dG^L/dc = (G_B^L - G_Fe^L) + RT ln(c/(1-c)) + (1-2c) L_{B,Fe}^L
__host__ __device__ inline double compute_dGL_dc(double c, double T)
{
    double cs = fmax(1e-12, fmin(1.0 - 1e-12, c));
    return (G_B_liquid(T) - G_Fe_liquid(T))
         + R_gas * T * log(cs / (1.0 - cs))
         + (1.0 - 2.0*cs) * L_BFe_liquid(T);
}

// G^L(c,T) = c G_B^L + (1-c) G_Fe^L + RT[c ln c + (1-c) ln(1-c)] + c(1-c) L
__host__ __device__ inline double compute_GL(double c, double T)
{
    double cs = fmax(1e-12, fmin(1.0 - 1e-12, c));
    return cs * G_B_liquid(T)
         + (1.0-cs) * G_Fe_liquid(T)
         + R_gas * T * (cs*log(cs) + (1.0-cs)*log(1.0-cs))
         + cs * (1.0-cs) * L_BFe_liquid(T);
}

// f_L(c,T) = G^L / V_m^L        [J/m^3]
__host__ __device__ inline double compute_fL(double c, double T, double VmL)
{
    return compute_GL(c, T) / VmL;
}

// f_S(T) = (3 G_Fe^bcc + G_B^beta + dG_Fe3B^f) / V_m^S   [J/m^3]
__host__ __device__ inline double compute_fS(double T, double VmS)
{
    return (3.0*G_Fe_bcc(T) + G_B_beta(T) + deltaG_Fe3B_f(T)) / VmS;
}

// Df^{Am->L}(T) = R T_g ln(1+alpha) f(T/T_g) / V_m^L     [J/m^3]
__host__ __device__ inline double compute_dfAmL(double T, double Tg,
                                                double alpha, double VmL)
{
    return R_gas * Tg * log(1.0 + alpha) * f_tau_func(T / Tg) / VmL;
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

    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                    nx, dx, x0,
                                    ny, dy, y0);
    mesh.print();

    // === 2. Time parameters ==================================================
    const double dt     = cfg["initialize"]["dt"];
    const int    nSteps = cfg["initialize"]["nSteps"];

    // === 3. Physical constants ===============================================
    const double M_eta   = cfg["constants"]["M_eta"];    // eta mobility (constant)
    const double eps_sq  = cfg["constants"]["eps_sq"];    // epsilon^2  (phi gradient coeff)
    const double beta_sq = cfg["constants"]["beta_sq"];   // beta^2     (eta gradient coeff)
    const double w_phi   = cfg["constants"]["w_phi"];     // phi double-well height
    const double w_eta   = cfg["constants"]["w_eta"];     // eta double-well height
    const double w_ex    = cfg["constants"]["w_ex"];      // cross-coupling coefficient
    const double T       = cfg["constants"]["T"];         // temperature  [K]
    const double T_g     = cfg["constants"]["T_g"];       // glass transition temperature [K]
    const double alpha   = cfg["constants"]["alpha"];     // amorphous parameter
    const double V_m_L   = cfg["constants"]["V_m_L"];    // molar volume, liquid [m^3/mol]
    const double V_m_S   = cfg["constants"]["V_m_S"];    // molar volume, solid  [m^3/mol]
    const double D_Am    = cfg["constants"]["D_Am"];      // amorphous diffusivity [m^2/s]

    // Pre-compute temperature-dependent constants (T is fixed for the run)
    const double M_phi_val = 22.1 * exp(-140.0e3 / (R_gas * T));
    const double D_L       = 2.0e-6 * exp(-1.11e5 / (R_gas * T));
    const double D_S       = 1.311e-6 * exp(-1.51e5 / (R_gas * T));
    const double fS_val    = compute_fS(T, V_m_S);
    const double dfAmL_val = compute_dfAmL(T, T_g, alpha, V_m_L);

    // === 4. Fields ===========================================================
    // Primary fields
    ScalarField c   (mesh, "c",   /*ghost=*/1);
    ScalarField phi (mesh, "phi", /*ghost=*/1);
    ScalarField eta (mesh, "eta", /*ghost=*/1);
    ScalarField mu  (mesh, "mu",  /*ghost=*/1);

    // Auxiliary fields for the variable-mobility CH equation:
    //   dc/dt = div( D grad(mu) )   where  D = M_c(phi,eta) * c*(1-c)
    //         = D * lap(mu) + grad(D) . grad(mu)
    ScalarField D_field   (mesh, "D",    /*ghost=*/1);
    ScalarField grad_D_x  (mesh, "gD_x", /*ghost=*/1);
    ScalarField grad_D_y  (mesh, "gD_y", /*ghost=*/1);
    ScalarField grad_mu_x (mesh, "gm_x", /*ghost=*/1);
    ScalarField grad_mu_y (mesh, "gm_y", /*ghost=*/1);
    ScalarField dot_gD_gm (mesh, "dgdm", /*ghost=*/1);
    ScalarField lap_mu    (mesh, "lmu",  /*ghost=*/1);

    c.fill(0); phi.fill(0); eta.fill(0); mu.fill(0);
    D_field.fill(0); grad_D_x.fill(0); grad_D_y.fill(0);
    grad_mu_x.fill(0); grad_mu_y.fill(0);
    dot_gD_gm.fill(0); lap_mu.fill(0);

    // --- Initialization from file ---
    const std::string start_from = cfg["initialize"]["start_from"];
    const int         start_step = IO::resolveStartStep(start_from);

    IO::initField(c,   start_step);
    IO::initField(phi, start_step);
    IO::initField(eta, start_step);
    if (start_step == 0) IO::initField(mu, start_step);

    // --- Device allocation & upload ---
    auto allocUp = [](ScalarField& f){ f.allocDevice(); f.uploadAllToDevice(); };
    allocUp(c); allocUp(phi); allocUp(eta); allocUp(mu);
    allocUp(D_field); allocUp(grad_D_x); allocUp(grad_D_y);
    allocUp(grad_mu_x); allocUp(grad_mu_y);
    allocUp(dot_gD_gm); allocUp(lap_mu);

    // === 5. Boundary conditions ==============================================
    auto  bcSet = buildBCs(cfg["boundary_conditions"]);
    auto& bcs   = bcSet.ptrs;

    // === 6. Equations ========================================================

    // ------ 6a. STEADY: mu = df/dc -------------------------------------------
    //   df/dc = [1 - h(phi)] * (1/V_m_L) * dG^L/dc
    Equation eq_mu(mu, "mu");
    eq_mu.setRHS(
        pw(c, phi, PHIX_FN (double c_val, double phi_val) {
            double hp  = h_func(phi_val);
            double dGL = compute_dGL_dc(c_val, T);
            return (1.0 - hp) * dGL / V_m_L;
        })
    );

    // ------ 6b. STEADY: D = M_c(phi,eta) * c*(1-c) ---------------------------
    //   M_c = [(1-h(phi))((1-h(eta))D_L + h(eta)D_Am) + h(phi)D_S] / (RT)
    Equation eq_D(D_field, "D");
    eq_D.setRHS(
        pw(c, phi, eta, PHIX_FN (double cv, double pv, double ev) {
            double hp  = h_func(pv);
            double he  = h_func(ev);
            double Mc  = ((1.0-hp)*((1.0-he)*D_L + he*D_Am) + hp*D_S)
                       / (R_gas * T);
            return Mc * cv * (1.0 - cv);
        })
    );

    // ------ 6c. STEADY: auxiliary gradient / laplacian fields -----------------
    Equation eq_gm_x(grad_mu_x, "gm_x"); eq_gm_x.setRHS(grad(mu, 0));
    Equation eq_gm_y(grad_mu_y, "gm_y"); eq_gm_y.setRHS(grad(mu, 1));
    Equation eq_gD_x(grad_D_x,  "gD_x"); eq_gD_x.setRHS(grad(D_field, 0));
    Equation eq_gD_y(grad_D_y,  "gD_y"); eq_gD_y.setRHS(grad(D_field, 1));

    // dot_gD_gm = grad(D) . grad(mu)
    Equation eq_dot(dot_gD_gm, "dgdm");
    eq_dot.setRHS(grad_D_x * grad_mu_x + grad_D_y * grad_mu_y);

    Equation eq_lmu(lap_mu, "lmu");
    eq_lmu.setRHS(lap(mu));

    // ------ 6d. TRANSIENT: c equation ----------------------------------------
    //   dc/dt = div(D * grad(mu)) = D * lap(mu) + grad(D) . grad(mu)
    Equation eqC(c, "CH_c");
    eqC.setRHS(D_field * lap_mu + 1.0 * dot_gD_gm);

    // ------ 6e. TRANSIENT: phi equation (Allen-Cahn) -------------------------
    //   dphi/dt = -M_phi * df/dphi  +  M_phi * eps^2 * lap(phi)
    //   M_phi   = 22.1 * exp(-140e3 / (R*T))   (pre-computed as M_phi_val)
    //   df/dphi = h'(phi)*[f_S - f_L(c,T) - h(eta)*Df_AmL]
    //           + w_phi g'(phi)  + 2 w_ex phi eta^2
    Equation eqPhi(phi, "AC_phi");
    eqPhi.setRHS(
        pw(phi, eta, c, PHIX_FN (double pv, double ev, double cv) {
            double fL       = compute_fL(cv, T, V_m_L);
            double hp       = h_prime(pv);
            double h_eta    = h_func(ev);
            double df_dphi  = hp * (fS_val - fL - h_eta * dfAmL_val)
                            + w_phi * g_prime(pv)
                            + 2.0 * w_ex * pv * ev * ev;
            return -M_phi_val * df_dphi;
        })
        + M_phi_val * eps_sq * lap(phi)
    );

    // ------ 6f. TRANSIENT: eta equation (Allen-Cahn) -------------------------
    //   deta/dt = -M_eta * df/deta  +  M_eta * beta^2 * lap(eta)
    //   df/deta = [1-h(phi)] h'(eta) Df_AmL  + w_eta g'(eta)  + 2 w_ex phi^2 eta
    Equation eqEta(eta, "AC_eta");
    eqEta.setRHS(
        pw(phi, eta, PHIX_FN (double pv, double ev) {
            double hphi     = h_func(pv);
            double hp_eta   = h_prime(ev);
            double df_deta  = (1.0 - hphi) * hp_eta * dfAmL_val
                            + w_eta * g_prime(ev)
                            + 2.0 * w_ex * pv * pv * ev;
            return -M_eta * df_deta;
        })
        + M_eta * beta_sq * lap(eta)
    );

    // === 7. Solver ===========================================================
    //
    //  Step order (per time step):
    //   1.  BC(c)        -> mu        = df/dc                    [STEADY]
    //   2.  BC(mu)       -> gm_x      = d(mu)/dx                [STEADY]
    //   3.  ---          -> gm_y      = d(mu)/dy                [STEADY]
    //   4.  ---          -> lmu       = lap(mu)                  [STEADY]
    //   5.  BC(phi)      -> D         = M_c(phi,eta)*c*(1-c)    [STEADY]
    //   6.  BC(D)        -> gD_x      = d(D)/dx                 [STEADY]
    //   7.  ---          -> gD_y      = d(D)/dy                 [STEADY]
    //   8.  BC(eta)      -> dgdm      = grad(D).grad(mu)        [STEADY]
    //   9.  ---          -> c        += dt * RHS_c               [TRANSIENT]
    //  10.  BC(phi)      -> phi      += dt * RHS_phi             [TRANSIENT]
    //  11.  BC(eta)      -> eta      += dt * RHS_eta             [TRANSIENT]
    //
    Solver solver(
        {
            { &c,        bcs, &eq_mu,   EquationType::STEADY    },  // 1
            { &mu,       bcs, &eq_gm_x, EquationType::STEADY    },  // 2
            { &mu,       bcs, &eq_gm_y, EquationType::STEADY    },  // 3
            { &mu,       bcs, &eq_lmu,  EquationType::STEADY    },  // 4
            { &phi,      bcs, &eq_D,    EquationType::STEADY    },  // 5
            { &D_field,  bcs, &eq_gD_x, EquationType::STEADY    },  // 6
            { &D_field,  bcs, &eq_gD_y, EquationType::STEADY    },  // 7
            { &eta,      bcs, &eq_dot,  EquationType::STEADY    },  // 8
            { &c,        bcs, &eqC,     EquationType::TRANSIENT },  // 9
            { &phi,      bcs, &eqPhi,   EquationType::TRANSIENT },  // 10
            { &eta,      bcs, &eqEta,   EquationType::TRANSIENT }   // 11
        },
        dt, TimeScheme::EULER);

    solver.step = start_step;
    solver.time = start_step * dt;

    // === 8. Output & time loop ===============================================
    IO::OutputWriter writer(cfg["output"]);

    if (start_step == 0) {
        writer.writeFields(c,   0, solver.time);
        writer.writeFields(phi, 0, solver.time);
        writer.writeFields(eta, 0, solver.time);
        std::cout << "Starting GFA simulation ("
                  << nSteps << " steps, dt=" << dt
                  << ", T=" << T << " K)\n";
    } else {
        std::cout << "Resuming GFA simulation from step " << start_step
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
            writer.writeFields(phi, solver.step, solver.time);
            writer.writeFields(eta, solver.step, solver.time);
        }
    }

    std::cout << "Done.\n";
    return 0;
}
