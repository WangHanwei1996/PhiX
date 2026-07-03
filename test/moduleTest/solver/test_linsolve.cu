// ---------------------------------------------------------------------------
// module_linsolve — matrix-free linear-solver layer (solver/LinearSolver.h)
//
// Consistent-system methodology: for any reference field x_ref, build
// b = A·x_ref with the SAME matrix-free operator, then solve A·x = b from a
// zero guess — CG must recover x_ref to solver tolerance regardless of BC
// type or spatial accuracy.  Covers:
//
//   1. fieldDot vs CPU reference (the CG inner product).
//   2. Helmholtz (I − σ·D∇²), 2D periodic.
//   3. Helmholtz, 1D no-flux.
//   4. Biharmonic (I + σ·G∇⁴), 1D periodic (two-pass stencil + BC on the
//      intermediate ∇²x).
//   5. Edge cases: b = 0 → x = 0; maxIter exhausted → converged == false
//      with throwOnFail = false, throws otherwise.
// ---------------------------------------------------------------------------

#include "solver/LinearSolver.h"
#include "field/ScalarField.h"
#include "field/Reduce.h"
#include "boundary/PeriodicBC.h"
#include "boundary/NoFluxBC.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

using namespace PhiX;

static void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

// b = x_ref − σ·L(x_ref)   (device apply, host combine, upload)
static void buildRhs(LinearOperator& L, double sigma,
                     ScalarField& xRef, ScalarField& b) {
    ScalarField Lx(xRef.mesh, "_Lx", xRef.ghost);
    Lx.allocDevice();
    L.apply(xRef, Lx);
    Lx.downloadCurrFromDevice();
    xRef.downloadCurrFromDevice();   // ghosts were refreshed on device
    for (std::size_t i = 0; i < b.storedSize; ++i)
        b.curr[i] = xRef.curr[i] - static_cast<Real>(sigma) * Lx.curr[i];
    if (!b.deviceAllocated()) b.allocDevice();
    b.uploadAllToDevice();
}

static double maxErrVsRef(ScalarField& x, const ScalarField& xRef) {
    x.downloadCurrFromDevice();
    double err = 0.0;
    for (int k = 0; k < x.mesh.n[2]; ++k)
    for (int j = 0; j < x.mesh.n[1]; ++j)
    for (int i = 0; i < x.mesh.n[0]; ++i) {
        const std::size_t idx = static_cast<std::size_t>(x.index(i, j, k));
        err = std::max(err, std::fabs(
            static_cast<double>(x.curr[idx]) - xRef.curr[idx]));
    }
    return err;
}

int main() {
    // =======================================================================
    // 1. fieldDot vs CPU
    // =======================================================================
    {
        Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                        19, 0.1, 0.0, 13, 0.2, 0.0);
        ScalarField a(mesh, "a", 1), b(mesh, "b", 1);
        a.fillCurr(1e300);   // poisoned ghosts must not leak into the dot
        b.fillCurr(-1e300);
        double ref = 0.0;
        for (int j = 0; j < 13; ++j)
        for (int i = 0; i < 19; ++i) {
            const double av = std::sin(0.3 * i + 0.7 * j);
            const double bv = std::cos(0.5 * i - 0.2 * j) + 0.1;
            a.curr[static_cast<std::size_t>(a.index(i, j))] = av;
            b.curr[static_cast<std::size_t>(b.index(i, j))] = bv;
            ref += av * bv;
        }
        a.allocDevice(); a.uploadAllToDevice();
        b.allocDevice(); b.uploadAllToDevice();
        const double dot = reduce::fieldDot(a, b);
        require(std::fabs(dot - ref) < 1e-12 * std::max(1.0, std::fabs(ref)),
                "fieldDot mismatch: " + std::to_string(dot) + " vs "
                + std::to_string(ref));
    }

    // =======================================================================
    // 2. Helmholtz 2D periodic
    // =======================================================================
    {
        const int    N  = 64;
        const double L0 = 2.0 * M_PI, dx = L0 / N;
        Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                        N, dx, 0.0, N, dx, 0.0);
        PeriodicBC bcx(mesh.facePatch(Axis::X, Side::LOW));
        PeriodicBC bcy(mesh.facePatch(Axis::Y, Side::LOW));
        LaplacianOp L(0.7, {&bcx, &bcy});

        ScalarField xRef(mesh, "xref", 1), b(mesh, "b", 1), x(mesh, "x", 1);
        xRef.initialize([](double xx, double yy, double) {
            return std::sin(2.0 * xx) + 0.5 * std::cos(3.0 * yy)
                 + 0.2 * std::sin(xx + 2.0 * yy);
        });
        xRef.allocDevice(); xRef.uploadAllToDevice();

        const double sigma = 0.1;                 // σ·D/dx² ≈ 7 — nontrivial
        buildRhs(L, sigma, xRef, b);

        x.fill(0.0);
        x.allocDevice(); x.uploadAllToDevice();

        ConjugateGradient cg(mesh, 1);
        auto res = cg.solve(L, sigma, x, b, 1e-12, 500);
        std::printf("  helmholtz 2D periodic: %d iters, rel res %.2e\n",
                    res.iterations, res.relResidual);
        require(res.converged, "helmholtz 2D CG did not converge");
        require(maxErrVsRef(x, xRef) < 1e-9,
                "helmholtz 2D solution differs from reference");
    }

    // =======================================================================
    // 3. Helmholtz 1D no-flux
    // =======================================================================
    {
        const int N = 128;
        Mesh mesh = Mesh::makeUniform1D(CoordSys::CARTESIAN, N, 1.0 / N);
        NoFluxBC bcLo(mesh.facePatch(Axis::X, Side::LOW));
        NoFluxBC bcHi(mesh.facePatch(Axis::X, Side::HIGH));
        LaplacianOp L(1.0, {&bcLo, &bcHi});

        ScalarField xRef(mesh, "xref", 1), b(mesh, "b", 1), x(mesh, "x", 1);
        xRef.initialize([](double xx, double, double) {
            return 0.3 + std::cos(M_PI * xx) + 0.4 * std::cos(3.0 * M_PI * xx);
        });
        xRef.allocDevice(); xRef.uploadAllToDevice();

        const double sigma = 5e-3;                // σ·D/dx² ≈ 82 — stiff
        buildRhs(L, sigma, xRef, b);

        x.fill(0.0);
        x.allocDevice(); x.uploadAllToDevice();

        ConjugateGradient cg(mesh, 1);
        auto res = cg.solve(L, sigma, x, b, 1e-12, 1000);
        std::printf("  helmholtz 1D no-flux:  %d iters, rel res %.2e\n",
                    res.iterations, res.relResidual);
        require(res.converged, "helmholtz 1D CG did not converge");
        require(maxErrVsRef(x, xRef) < 1e-9,
                "helmholtz 1D solution differs from reference");
    }

    // =======================================================================
    // 4. Biharmonic 1D periodic (Cahn-Hilliard implicit part)
    // =======================================================================
    {
        const int    N  = 128;
        const double L0 = 2.0 * M_PI, dx = L0 / N;
        Mesh mesh = Mesh::makeUniform1D(CoordSys::CARTESIAN, N, dx);
        PeriodicBC bcX(mesh.facePatch(Axis::X, Side::LOW));
        PeriodicBC bcLap(mesh.facePatch(Axis::X, Side::LOW));
        BiharmonicOp L(2e-3, {&bcX}, {&bcLap});   // A = I + σ·G·∇⁴

        ScalarField xRef(mesh, "xref", 1), b(mesh, "b", 1), x(mesh, "x", 1);
        xRef.initialize([](double xx, double, double) {
            return std::sin(xx) + 0.3 * std::sin(4.0 * xx);
        });
        xRef.allocDevice(); xRef.uploadAllToDevice();

        const double sigma = 0.05;                // σ·G·k⁴max sizeable
        buildRhs(L, sigma, xRef, b);

        x.fill(0.0);
        x.allocDevice(); x.uploadAllToDevice();

        ConjugateGradient cg(mesh, 1);
        auto res = cg.solve(L, sigma, x, b, 1e-12, 2000);
        std::printf("  biharmonic 1D:         %d iters, rel res %.2e\n",
                    res.iterations, res.relResidual);
        require(res.converged, "biharmonic CG did not converge");
        require(maxErrVsRef(x, xRef) < 1e-8,
                "biharmonic solution differs from reference");
    }

    // =======================================================================
    // 5. Edge cases
    // =======================================================================
    {
        const int N = 32;
        Mesh mesh = Mesh::makeUniform1D(CoordSys::CARTESIAN, N, 1.0 / N);
        PeriodicBC bc(mesh.facePatch(Axis::X, Side::LOW));
        LaplacianOp L(1.0, {&bc});
        ConjugateGradient cg(mesh, 1);

        ScalarField b(mesh, "b", 1), x(mesh, "x", 1);
        b.fill(0.0);
        b.allocDevice(); b.uploadAllToDevice();
        x.initialize([](double xx, double, double) { return std::sin(xx); });
        x.allocDevice(); x.uploadAllToDevice();

        auto res = cg.solve(L, 0.1, x, b, 1e-10, 100);
        require(res.converged && reduce::fieldMaxAbs(x) == 0.0,
                "b = 0 must return x = 0");

        // maxIter exhaustion
        b.initialize([](double xx, double, double) { return std::cos(xx); });
        b.uploadAllToDevice();
        x.fill(0.0); x.uploadAllToDevice();
        auto res2 = cg.solve(L, 1e3, x, b, 1e-14, 1, /*throwOnFail=*/false);
        require(!res2.converged, "1-iteration solve reported convergence");

        bool threw = false;
        x.fill(0.0); x.uploadAllToDevice();
        try { cg.solve(L, 1e3, x, b, 1e-14, 1); }
        catch (const std::runtime_error&) { threw = true; }
        require(threw, "exhausted solve did not throw with throwOnFail=true");
    }

    return 0;
}
