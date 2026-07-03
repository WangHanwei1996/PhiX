// ---------------------------------------------------------------------------
// heat_implicit — 全隐式热传导（SemiImplicitSolver 入门示例）
//
//   ∂T/∂t = D ∇²T          2D，无通量边界，中心高斯热斑弛豫
//
// 演示要点：
//   • 纯隐式线性步：显式方程不设 RHS（N ≡ 0），刚性算子 L = D∇² 全部
//     交给 LaplacianOp + CG；
//   • dt 取显式稳定极限的 200 倍——纯显式在这个 dt 下 10 步内爆掉，
//     后向 Euler 无条件稳定；
//   • 运行时诊断全部走设备端归约（fieldMax / fieldSum），能量（总热量）
//     在无通量壁下守恒。
//
// 运行：./heat_implicit [nSteps=200]     输出 output/T_*.vts（ParaView）
// ---------------------------------------------------------------------------

#include "field/ScalarField.h"
#include "field/Reduce.h"
#include "equation/Equation.h"
#include "solver/SemiImplicitSolver.h"
#include "boundary/NoFluxBC.h"
#include "perf/Perf.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

using namespace PhiX;

int main(int argc, char** argv) {
    const int    nSteps = (argc > 1) ? std::atoi(argv[1]) : 200;
    const int    N  = 256;
    const double L0 = 1.0, dx = L0 / N;
    const double D  = 1.0;

    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                    N, dx, 0.0, N, dx, 0.0);

    ScalarField T(mesh, "T", 1);
    T.initialize([](double x, double y, double) {
        const double r2 = (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5);
        return 300.0 + 700.0 * std::exp(-r2 / (2.0 * 0.05 * 0.05));
    });
    T.allocDevice();
    T.uploadAllToDevice();

    NoFluxBC bcXlo(mesh.facePatch(Axis::X, Side::LOW));
    NoFluxBC bcXhi(mesh.facePatch(Axis::X, Side::HIGH));
    NoFluxBC bcYlo(mesh.facePatch(Axis::Y, Side::LOW));
    NoFluxBC bcYhi(mesh.facePatch(Axis::Y, Side::HIGH));
    std::vector<BoundaryCondition*> bcs = {&bcXlo, &bcXhi, &bcYlo, &bcYhi};

    // 纯隐式：不给显式 RHS —— (I − dt·D∇²) Tⁿ⁺¹ = Tⁿ
    Equation eq(T, "heat");
    LaplacianOp L(D, bcs);

    const double dtExplicit = 0.25 * dx * dx / D;   // 2D 显式稳定极限
    const double dt = 200.0 * dtExplicit;
    std::printf("heat_implicit: N=%d  dt = %.3e (200x explicit limit %.3e)\n",
                N, dt, dtExplicit);

    SemiImplicitSolver::CGOptions cgo;
    cgo.relTol = 1e-9;
    SemiImplicitSolver semi(eq, bcs, L, dt, cgo);

    std::filesystem::create_directories("output");
    T.downloadCurrFromDevice();
    T.write("output/T_0.vts", FieldFormat::VTS);

    const double heat0 = reduce::fieldSum(T);
    perf::WallTimer wall;
    for (int s = 1; s <= nSteps; ++s) {
        semi.advance();
        if (s % 50 == 0 || s == nSteps) {
            const double tMax  = reduce::fieldMax(T);
            const double drift = (reduce::fieldSum(T) - heat0) / heat0;
            std::printf("  step %4d  t=%.4e  Tmax=%7.2f  heat drift=%+.2e"
                        "  CG %d iters\n",
                        s, semi.time, tMax, drift,
                        semi.lastSolve().iterations);
            T.downloadCurrFromDevice();
            T.write("output/T_" + std::to_string(s) + ".vts",
                    FieldFormat::VTS);
        }
    }
    cudaDeviceSynchronize();
    std::printf("done: %d steps in %.2f s (%.3f ms/step)\n",
                nSteps, wall.seconds(), wall.seconds() * 1e3 / nSteps);
    return 0;
}
