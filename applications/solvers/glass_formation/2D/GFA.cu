/***********************************************************************\
 *  Glass Formation Ability (GFA) Solver (2D) - composite-Term DSL
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

static constexpr double R_gas = 8.314;

__host__ __device__ inline double G_B_liquid(double T) {
    if (T <= 500.0) return 40723.275 + 86.843839*T - 15.6641*T*log(T)
        - 6.864515e-3*T*T + 0.618878e-6*T*T*T + 370843.0/T;
    if (T <= 2348.0) return 41119.703 + 82.101722*T - 14.9827763*T*log(T)
        - 7.095669e-3*T*T + 0.507347e-6*T*T*T + 335484.0/T;
    return 28842.012 + 200.94731*T - 31.4*T*log(T);
}
__host__ __device__ inline double G_Fe_liquid(double T) {
    if (T <= 1811.0) {
        double T2 = T*T, T3 = T2*T, T7 = T3*T3*T;
        return 13265.87 + 117.57557*T - 23.5143*T*log(T)
             - 4.39752e-3*T2 - 0.058927e-6*T3 + 773597.0/T - 367.516e-23*T7;
    }
    return -10838.83 + 291.302*T - 46.0*T*log(T);
}
__host__ __device__ inline double L_BFe_liquid(double T) { return -122861.0 + 14.59*T; }
__host__ __device__ inline double G_Fe_bcc(double T) {
    if (T <= 1811.0) return 1225.7 + 124.134*T - 23.5143*T*log(T)
        - 4.39752e-3*T*T - 0.058927e-6*T*T*T + 773597.0/T;
    double T3 = T*T*T, T9 = T3*T3*T3;
    return -25383.581 + 299.31255*T - 46.0*T*log(T) + 2296.03e28 / T9;
}
__host__ __device__ inline double G_B_beta(double T) {
    if (T <= 1100.0) return -7735.284 + 107.111864*T - 15.6641*T*log(T)
        - 6.864515e-3*T*T + 0.618878e-6*T*T*T + 370843.0/T;
    if (T <= 2348.0) return -16649.474 + 184.801744*T - 26.6047*T*log(T)
        - 0.79809e-3*T*T - 0.02556e-6*T*T*T + 1748270.0/T;
    if (T <= 3000.0) return -36667.582 + 231.336244*T - 31.5957527*T*log(T)
        - 1.59488e-3*T*T + 0.134719e-6*T*T*T + 11205883.0/T;
    return -21530.653 + 222.396264*T - 31.4*T*log(T);
}
__host__ __device__ inline double deltaG_Fe3B_f(double T) { return -77749.0 + 2.59*T; }
__host__ __device__ inline double f_tau_func(double tau) {
    if (tau <= 1.0) {
        double t3 = tau*tau*tau, t9 = t3*t3*t3, t15 = t9*t3*t3;
        return 1.0 - 9.9167285e-1/tau - 1.11737779e-1*t3
             - 4.96612349e-3*t9 - 1.11737779e-3*t15;
    }
    double inv = 1.0/tau, i5 = inv*inv*inv*inv*inv;
    double i10 = i5*i5, i15 = i10*i5, i25 = i15*i10;
    return -1.05443689e-1*i5 - 3.34741816e-3*i15 - 7.02957924e-4*i25;
}

__host__ __device__ inline double h_func(double x)  { return x*x*x*(10.0 - 15.0*x + 6.0*x*x); }
__host__ __device__ inline double h_prime(double x) { return 30.0*x*x*(1.0-x)*(1.0-x); }
__host__ __device__ inline double g_prime(double x) { return 2.0*x*(1.0-x)*(1.0-2.0*x); }

__host__ __device__ inline double compute_dGL_dc(double c, double T) {
    double cs = fmax(1e-12, fmin(1.0-1e-12, c));
    return (G_B_liquid(T) - G_Fe_liquid(T))
         + R_gas*T*log(cs/(1.0-cs)) + (1.0 - 2.0*cs)*L_BFe_liquid(T);
}
__host__ __device__ inline double compute_GL(double c, double T) {
    double cs = fmax(1e-12, fmin(1.0-1e-12, c));
    return cs*G_B_liquid(T) + (1.0-cs)*G_Fe_liquid(T)
         + R_gas*T*(cs*log(cs) + (1.0-cs)*log(1.0-cs))
         + cs*(1.0-cs)*L_BFe_liquid(T);
}
__host__ __device__ inline double compute_fL(double c, double T, double VmL) { return compute_GL(c, T)/VmL; }
__host__ __device__ inline double compute_fS(double T, double VmS) {
    return (3.0*G_Fe_bcc(T) + G_B_beta(T) + deltaG_Fe3B_f(T))/VmS;
}
__host__ __device__ inline double compute_dfAmL(double T, double Tg, double alpha, double VmL) {
    return R_gas*Tg*log(1.0+alpha)*f_tau_func(T/Tg)/VmL;
}

int main(int argc, char* argv[])
{
    using namespace PhiX;
    IO::ConfigFile cfg = IO::ConfigFile::fromArgs(argc, argv);

    const int    nx = cfg["mesh"]["nx"];
    const double dx = cfg["mesh"]["dx"];
    const double x0 = cfg["mesh"]["x0"];
    const int    ny = cfg["mesh"]["ny"];
    const double dy = cfg["mesh"]["dy"];
    const double y0 = cfg["mesh"]["y0"];
    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN, nx, dx, x0, ny, dy, y0);
    mesh.print();

    const double dt     = cfg["initialize"]["dt"];
    const int    nSteps = cfg["initialize"]["nSteps"];

    const double M_eta   = cfg["constants"]["M_eta"];
    const double eps_sq  = cfg["constants"]["eps_sq"];
    const double beta_sq = cfg["constants"]["beta_sq"];
    const double w_phi   = cfg["constants"]["w_phi"];
    const double w_eta   = cfg["constants"]["w_eta"];
    const double w_ex    = cfg["constants"]["w_ex"];
    const double T       = cfg["constants"]["T"];
    const double T_g     = cfg["constants"]["T_g"];
    const double alpha   = cfg["constants"]["alpha"];
    const double V_m_L   = cfg["constants"]["V_m_L"];
    const double V_m_S   = cfg["constants"]["V_m_S"];
    const double D_Am    = cfg["constants"]["D_Am"];

    const double M_phi_val = 22.1 * exp(-140.0e3 / (R_gas * T));
    const double D_L       = 2.0e-6 * exp(-1.11e5 / (R_gas * T));
    const double D_S       = 1.311e-6 * exp(-1.51e5 / (R_gas * T));
    const double fS_val    = compute_fS(T, V_m_S);
    const double dfAmL_val = compute_dfAmL(T, T_g, alpha, V_m_L);

    ScalarField c   (mesh, "c",   1);
    ScalarField phi (mesh, "phi", 1);
    ScalarField eta (mesh, "eta", 1);
    ScalarField mu  (mesh, "mu",  1);
    ScalarField D   (mesh, "D",   1);
    c.fill(0); phi.fill(0); eta.fill(0); mu.fill(0); D.fill(0);

    const std::string start_from = cfg["initialize"]["start_from"];
    const int         start_step = IO::resolveStartStep(start_from);
    IO::initField(c,   start_step);
    IO::initField(phi, start_step);
    IO::initField(eta, start_step);
    if (start_step == 0) IO::initField(mu, start_step);

    auto allocUp = [](ScalarField& f){ f.allocDevice(); f.uploadAllToDevice(); };
    allocUp(c); allocUp(phi); allocUp(eta); allocUp(mu); allocUp(D);

    auto  bcSet = buildBCs(cfg["boundary_conditions"]);
    auto& bcs   = bcSet.ptrs;

    Equation eq_mu(mu, "mu");
    eq_mu.setRHS(
        pw(c, phi, PHIX_FN (double c_val, double phi_val) {
            return (1.0 - h_func(phi_val)) * compute_dGL_dc(c_val, T) / V_m_L;
        })
    );

    Equation eqD(D, "D");
    eqD.setRHS(
        pw(c, phi, eta, PHIX_FN (double cv, double pv, double ev) {
            double hp  = h_func(pv);
            double he  = h_func(ev);
            double Mc  = ((1.0-hp)*((1.0-he)*D_L + he*D_Am) + hp*D_S) / (R_gas * T);
            return Mc * cv * (1.0 - cv);
        })
    );

    Equation eqC(c, "CH_c");
    eqC.setRHS(
          D * lap(mu)
        + grad(D, 0) * grad(mu, 0)
        + grad(D, 1) * grad(mu, 1)
    );

    Equation eqPhi(phi, "AC_phi");
    eqPhi.setRHS(
        pw(phi, eta, c, PHIX_FN (double pv, double ev, double cv) {
            double fL      = compute_fL(cv, T, V_m_L);
            double hp      = h_prime(pv);
            double h_eta   = h_func(ev);
            double df_dphi = hp*(fS_val - fL - h_eta*dfAmL_val)
                           + w_phi*g_prime(pv) + 2.0*w_ex*pv*ev*ev;
            return -M_phi_val * df_dphi;
        })
        + M_phi_val * eps_sq * lap(phi)
    );

    Equation eqEta(eta, "AC_eta");
    eqEta.setRHS(
        pw(phi, eta, PHIX_FN (double pv, double ev) {
            double hphi    = h_func(pv);
            double hp_eta  = h_prime(ev);
            double df_deta = (1.0 - hphi)*hp_eta*dfAmL_val
                           + w_eta*g_prime(ev) + 2.0*w_ex*pv*pv*ev;
            return -M_eta * df_deta;
        })
        + M_eta * beta_sq * lap(eta)
    );

    Solver solver(
        {
            { &c,    bcs, &eq_mu, EquationType::STEADY    },  // 1: compute mu
            { &mu,   bcs, &eqD,   EquationType::STEADY    },  // 2: refresh mu halos, compute D
            { &D,    bcs, &eqD,   EquationType::STEADY    },  // 3: refresh D halos (recompute D)
            { &c,    bcs, &eqC,   EquationType::TRANSIENT },  // 4: update c
            { &phi,  bcs, &eqPhi, EquationType::TRANSIENT },  // 5: update phi
            { &eta,  bcs, &eqEta, EquationType::TRANSIENT }   // 6: update eta
        },
        dt, TimeScheme::EULER);

    solver.step = start_step;
    solver.time = start_step * dt;

    IO::OutputWriter writer(cfg["output"]);

    if (start_step == 0) {
        writer.writeFields(c,   0, solver.time);
        writer.writeFields(phi, 0, solver.time);
        writer.writeFields(eta, 0, solver.time);
        std::cout << "Starting GFA simulation (" << nSteps << " steps, dt=" << dt
                  << ", T=" << T << " K)\n";
    } else {
        std::cout << "Resuming GFA simulation from step " << start_step
                  << " (t=" << start_step * dt << "), "
                  << nSteps - start_step << " steps remaining, dt=" << dt << "\n";
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