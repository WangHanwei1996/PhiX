// ---------------------------------------------------------------------------
// test_fieldops.cu — Functional tests for multi-field pw and FieldOps operators
//
// Covers:
//   1.  pw(f, func)              — single-field baseline              (CPU + GPU)
//   2.  pw(f1, f2, func)         — 2-field pointwise                  (CPU + GPU)
//   3.  pw(f1, f2, f3, func)     — 3-field pointwise                  (CPU + GPU)
//   4.  FieldOps: f1*f2, f1+f2, f1-f2, s*f, f*s, f+s, s+f, f-s,
//                 s-f, -f                                              (CPU + GPU)
//   5.  pw(VectorField, ScalarField, func) — per-component binary op  (CPU)
//   6.  pw(VectorField, VectorField, func) — component-wise binary op (CPU)
//
// All source fields are filled with uniform constants so expected values
// are trivially computable.
// Mesh: 4 × 4, ghost = 1, dx = dy = 1.0.
// ---------------------------------------------------------------------------

#include "mesh/Mesh.h"
#include "field/ScalarField.h"
#include "field/VectorField.h"
#include "equation/Equation.h"
#include "equation/VectorEquation.h"

#include <cmath>
#include <iostream>
#include <string>

using namespace PhiX;

// ---------------------------------------------------------------------------
// Utility: check all physical cells of f.curr equal expected
// ---------------------------------------------------------------------------
static bool checkUniform(const ScalarField& f, double expected,
                          double tol = 1e-12) {
    const Mesh& m = f.mesh;
    for (int j = 0; j < m.n[1]; ++j)
    for (int i = 0; i < m.n[0]; ++i) {
        double v = f.curr[f.index(i, j)];
        if (std::abs(v - expected) > tol) {
            std::cout << "    MISMATCH at (" << i << "," << j << "): "
                      << "got " << v << ", expected " << expected << "\n";
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Utility: print test result
// ---------------------------------------------------------------------------
static int g_passed = 0, g_total = 0;

static void report(const std::string& name, bool ok) {
    std::cout << (ok ? "  [PASS]" : "  [FAIL]") << "  " << name << "\n";
    ++g_total;
    if (ok) ++g_passed;
}

// ---------------------------------------------------------------------------
// Evaluate a single Term on CPU, write into rhs.curr, check uniform result
// ---------------------------------------------------------------------------
static void testTermCPU(const std::string& name, ScalarField& primary,
                         const Term& term, ScalarField& rhs, double expected) {
    Equation eq(primary, name);
    eq.setRHS(term);
    eq.computeRHSCPU(rhs);
    report(name + " [CPU]", checkUniform(rhs, expected));
}

// ---------------------------------------------------------------------------
// Evaluate a single Term on GPU, download, check uniform result
// ---------------------------------------------------------------------------
static void testTermGPU(const std::string& name, ScalarField& primary,
                         const Term& term, ScalarField& rhs, double expected) {
    Equation eq(primary, name);
    eq.setRHS(term);
    eq.computeRHS(rhs);
    rhs.downloadCurrFromDevice();
    report(name + " [GPU]", checkUniform(rhs, expected));
}

// ---------------------------------------------------------------------------
// Evaluate a single Term via both CPU and GPU paths
// ---------------------------------------------------------------------------
static void testTerm(const std::string& name, ScalarField& primary,
                      const Term& term, ScalarField& rhs, double expected) {
    testTermCPU(name, primary, term, rhs, expected);
    testTermGPU(name, primary, term, rhs, expected);
}

// ===========================================================================
int main() {
    // -----------------------------------------------------------------------
    // Mesh and source fields
    // -----------------------------------------------------------------------
    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                    4, 1.0, 0.0,
                                    4, 1.0, 0.0);

    ScalarField f1(mesh, "f1", 1);   // physical cells = 2.0
    ScalarField f2(mesh, "f2", 1);   // physical cells = 3.0
    ScalarField f3(mesh, "f3", 1);   // physical cells = 5.0
    ScalarField rhs(mesh, "rhs", 1); // output buffer (zeroed by computeRHS*)

    f1.fillCurr(2.0);
    f2.fillCurr(3.0);
    f3.fillCurr(5.0);

    // Allocate and upload device memory for all source fields and rhs
    f1.allocDevice();  f1.uploadAllToDevice();
    f2.allocDevice();  f2.uploadAllToDevice();
    f3.allocDevice();  f3.uploadAllToDevice();
    rhs.allocDevice();

    // -----------------------------------------------------------------------
    // 1. pw(f, func) — single-field baseline
    // -----------------------------------------------------------------------
    std::cout << "\n=== 1. pw(f, func) — single-field baseline ===\n";
    {
        auto t = pw(f1, [] __host__ __device__ (double v) { return v * v; });
        // f1 = 2.0  →  2.0^2 = 4.0
        testTerm("pw(f1, v²)", f1, t, rhs, 4.0);
    }
    {
        auto t = pw(f1, [] __host__ __device__ (double v) { return v * v; }, 3.0);
        // coeff=3, f1=2 → 3*4 = 12.0
        testTerm("pw(f1, v², coeff=3)", f1, t, rhs, 12.0);
    }

    // -----------------------------------------------------------------------
    // 2. pw(f1, f2, func) — 2-field pointwise
    // -----------------------------------------------------------------------
    std::cout << "\n=== 2. pw(f1, f2, func) — 2-field pointwise ===\n";
    {
        auto t = pw(f1, f2, [] __host__ __device__ (double a, double b) { return a * b; });
        // 2.0 * 3.0 = 6.0
        testTerm("pw(f1, f2, a*b)", f1, t, rhs, 6.0);
    }
    {
        auto t = pw(f1, f2, [] __host__ __device__ (double a, double b) { return a + b; });
        // 2.0 + 3.0 = 5.0
        testTerm("pw(f1, f2, a+b)", f1, t, rhs, 5.0);
    }
    {
        auto t = pw(f1, f2, [] __host__ __device__ (double a, double b) {
            return a * b;
        }, 2.0);
        // coeff=2, 2*3=6 → 12.0
        testTerm("pw(f1, f2, a*b, coeff=2)", f1, t, rhs, 12.0);
    }

    // -----------------------------------------------------------------------
    // 3. pw(f1, f2, f3, func) — 3-field pointwise
    // -----------------------------------------------------------------------
    std::cout << "\n=== 3. pw(f1, f2, f3, func) — 3-field pointwise ===\n";
    {
        auto t = pw(f1, f2, f3,
                    [] __host__ __device__ (double a, double b, double c) {
                        return a + b + c;
                    });
        // 2.0 + 3.0 + 5.0 = 10.0
        testTerm("pw(f1, f2, f3, a+b+c)", f1, t, rhs, 10.0);
    }
    {
        auto t = pw(f1, f2, f3,
                    [] __host__ __device__ (double a, double b, double c) {
                        return a * b * c;
                    });
        // 2.0 * 3.0 * 5.0 = 30.0
        testTerm("pw(f1, f2, f3, a*b*c)", f1, t, rhs, 30.0);
    }
    {
        auto t = pw(f1, f2, f3,
                    [] __host__ __device__ (double a, double b, double c) {
                        return a + b + c;
                    }, 0.5);
        // coeff=0.5, 10.0 → 5.0
        testTerm("pw(f1, f2, f3, a+b+c, coeff=0.5)", f1, t, rhs, 5.0);
    }

    // -----------------------------------------------------------------------
    // 4. FieldOps — ScalarField arithmetic operators
    // -----------------------------------------------------------------------
    std::cout << "\n=== 4. FieldOps — ScalarField arithmetic operators ===\n";

    // Field × Field
    testTerm("f1 * f2",   f1, f1 * f2, rhs,  6.0);  // 2*3=6
    testTerm("f1 + f2",   f1, f1 + f2, rhs,  5.0);  // 2+3=5
    testTerm("f1 - f2",   f1, f1 - f2, rhs, -1.0);  // 2-3=-1

    // Field × scalar / scalar × Field
    testTerm("3.0 * f1",  f1, 3.0 * f1, rhs,  6.0);  // 3*2=6
    testTerm("f1 * 4.0",  f1, f1 * 4.0, rhs,  8.0);  // 2*4=8

    // Field + scalar / scalar + Field
    testTerm("f1 + 1.0",  f1, f1 + 1.0, rhs,  3.0);  // 2+1=3
    testTerm("1.0 + f1",  f1, 1.0 + f1, rhs,  3.0);  // 1+2=3

    // Field - scalar / scalar - Field
    testTerm("f1 - 1.0",  f1, f1 - 1.0, rhs,  1.0);  // 2-1=1
    testTerm("10.0 - f1", f1, 10.0 - f1, rhs, 8.0);  // 10-2=8

    // Unary negation
    testTerm("-f1",        f1, -f1,       rhs, -2.0);  // -2

    // Composition: (f1 * f2) as part of a larger RHSExpr
    {
        // eq.setRHS(f1 * f2)  then += pw(f3, v->v)  →  6.0 + 5.0 = 11.0
        Equation eq(f1, "compose");
        RHSExpr expr(f1 * f2);
        expr += pw(f3, [] __host__ __device__ (double v) { return v; });
        eq.setRHS(expr);
        eq.computeRHSCPU(rhs);
        report("(f1*f2) + pw(f3, id) [CPU]", checkUniform(rhs, 11.0));

        eq.computeRHS(rhs);
        rhs.downloadCurrFromDevice();
        report("(f1*f2) + pw(f3, id) [GPU]", checkUniform(rhs, 11.0));
    }

    // -----------------------------------------------------------------------
    // 5. pw(VectorField, ScalarField, func) — per-component binary op
    // -----------------------------------------------------------------------
    std::cout << "\n=== 5. pw(VectorField, ScalarField, func) ===\n";
    {
        // vf: 2 components, each = 2.0;  f2 (scalar) = 3.0
        // func(a, b) = a * b  →  2.0 * 3.0 = 6.0 per component
        VectorField vf(mesh, "vf", 2, 1);
        VectorField vrhs(mesh, "vrhs", 2, 1);
        vf.fillCurr(2.0);

        VectorEquation veq(vf, "vf_sf_test");
        veq.setRHS(pw(vf, f2,
                      [] __host__ __device__ (double a, double b) { return a * b; }));
        veq.computeRHSCPU(vrhs);

        bool ok = checkUniform(vrhs[0], 6.0) && checkUniform(vrhs[1], 6.0);
        report("pw(vf, sf, a*b) — all components [CPU]", ok);
    }
    {
        // With coeff = 0.5  →  0.5 * 6.0 = 3.0 per component
        VectorField vf(mesh, "vf", 2, 1);
        VectorField vrhs(mesh, "vrhs", 2, 1);
        vf.fillCurr(2.0);

        VectorEquation veq(vf, "vf_sf_coeff_test");
        veq.setRHS(pw(vf, f2,
                      [] __host__ __device__ (double a, double b) { return a * b; }, 0.5));
        veq.computeRHSCPU(vrhs);

        bool ok = checkUniform(vrhs[0], 3.0) && checkUniform(vrhs[1], 3.0);
        report("pw(vf, sf, a*b, coeff=0.5) — all components [CPU]", ok);
    }

    // -----------------------------------------------------------------------
    // 6. pw(VectorField, VectorField, func) — component-wise binary op
    // -----------------------------------------------------------------------
    std::cout << "\n=== 6. pw(VectorField, VectorField, func) ===\n";
    {
        // vf1: each component = 2.0;  vf2: each component = 4.0
        // func(a, b) = a + b  →  2.0 + 4.0 = 6.0 per component
        VectorField vf1(mesh, "vf1", 2, 1);
        VectorField vf2(mesh, "vf2", 2, 1);
        VectorField vrhs(mesh, "vrhs2", 2, 1);
        vf1.fillCurr(2.0);
        vf2.fillCurr(4.0);

        VectorEquation veq(vf1, "vf_vf_test");
        veq.setRHS(pw(vf1, vf2,
                      [] __host__ __device__ (double a, double b) { return a + b; }));
        veq.computeRHSCPU(vrhs);

        bool ok = checkUniform(vrhs[0], 6.0) && checkUniform(vrhs[1], 6.0);
        report("pw(vf1, vf2, a+b) — all components [CPU]", ok);
    }
    {
        // func(a, b) = a * b  →  2.0 * 4.0 = 8.0 per component
        VectorField vf1(mesh, "vf1", 2, 1);
        VectorField vf2(mesh, "vf2", 2, 1);
        VectorField vrhs(mesh, "vrhs2", 2, 1);
        vf1.fillCurr(2.0);
        vf2.fillCurr(4.0);

        VectorEquation veq(vf1, "vf_vf_mul_test");
        veq.setRHS(pw(vf1, vf2,
                      [] __host__ __device__ (double a, double b) { return a * b; }));
        veq.computeRHSCPU(vrhs);

        bool ok = checkUniform(vrhs[0], 8.0) && checkUniform(vrhs[1], 8.0);
        report("pw(vf1, vf2, a*b) — all components [CPU]", ok);
    }

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    std::cout << "\n================================================\n";
    std::cout << "Result: " << g_passed << " / " << g_total << " tests passed\n";
    if (g_passed == g_total)
        std::cout << "ALL TESTS PASSED\n";
    else
        std::cout << "SOME TESTS FAILED\n";

    return (g_passed == g_total) ? 0 : 1;
}
