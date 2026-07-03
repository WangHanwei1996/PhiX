// ---------------------------------------------------------------------------
// spinodal_ch_semi — 半隐式 Cahn-Hilliard 旋节分解（经典线性分裂示例）
//
//   ∂c/∂t = M ∇²μ ,   μ = f'(c) − κ∇²c ,   f = ¼(c²−1)²
//
// IMEX 线性分裂：四阶刚性项 −Mκ∇⁴（BiharmonicOp）隐式，
// 化学项 M∇²f'(c) 显式（f'(c) 先算进辅助场 muE，再 lap(muE, M)）。
// dt 取显式 ∇⁴ 稳定极限的 50 倍——这正是 v2.16/17 线性求解层 +
// 半隐式积分器解锁的经典用法。
//
// 运行时诊断（全部设备端归约）：
//   • 质量 Σc 守恒（周期域 + 保守形式，应到机器精度）；
//   • max|c| → 1（相分离发育）；
//   • 每步 CG 迭代数（CUDA graph 路径）。
//
// 运行：./spinodal_ch_semi [nSteps=4000]   输出 output/c_*.vts
// ---------------------------------------------------------------------------

#include "field/ScalarField.h"
#include "field/Reduce.h"
#include "equation/Equation.h"
#include "solver/SemiImplicitSolver.h"
#include "boundary/PeriodicBC.h"
#include "operators/Laplacian.h"
#include "perf/Perf.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

using namespace PhiX;

int main(int argc, char** argv) {
    const int    nSteps = (argc > 1) ? std::atoi(argv[1]) : 4000;
    const int    N  = 256;
    const double L0 = 2.0 * M_PI, dx = L0 / N;
    const double M  = 1.0;
    const double kappa = 2e-3;                     // 界面宽度 ~ 2√κ ≈ 3.6 格

    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                    N, dx, 0.0, N, dx, 0.0);

    // 确定性"伪随机"初始扰动（多模叠加，零均值）
    ScalarField c(mesh, "c", 1), muE(mesh, "muE", 1);
    c.initialize([](double x, double y, double) {
        double v = 0.0;
        for (int k = 1; k <= 6; ++k)
            v += std::sin(k * x + 0.7 * k * k) * std::cos((7 - k) * y + 0.3 * k);
        return 0.05 * v / 6.0;
    });
    c.allocDevice();   c.uploadAllToDevice();
    muE.fill(0.0);
    muE.allocDevice(); muE.uploadAllToDevice();

    PeriodicBC bcx(mesh.facePatch(Axis::X, Side::LOW));
    PeriodicBC bcy(mesh.facePatch(Axis::Y, Side::LOW));
    std::vector<BoundaryCondition*> bcs = {&bcx, &bcy};

    // muE = f'(c) = c³ − c（每步在 advance 前重算）
    Equation eqMu(c, "mu");
    eqMu.setRHS(pw(c, PHIX_FN (Real v) { return v * v * v - v; }));

    // 显式部分：N(c) = M ∇² muE；隐式部分：L = −Mκ∇⁴
    Equation eqC(c, "c");
    eqC.setRHS(lap(muE, M));
    BiharmonicOp L(M * kappa, bcs, bcs);

    const double kmax = M_PI / dx;
    const double dtExplicit = 2.0 / (M * kappa * kmax * kmax * kmax * kmax);
    const double dt = 50.0 * dtExplicit;
    std::printf("spinodal_ch_semi: N=%d kappa=%.2e"
                "  dt=%.3e (50x explicit %.3e)\n",
                N, kappa, dt, dtExplicit);

    SemiImplicitSolver::CGOptions cgo;
    cgo.relTol  = 1e-8;
    cgo.maxIter = 2000;
    SemiImplicitSolver semi(eqC, bcs, L, dt, cgo);

    std::filesystem::create_directories("output");
    const double mass0 = reduce::fieldSum(c);

    perf::WallTimer wall;
    for (int s = 1; s <= nSteps; ++s) {
        bcx.applyOnGPU(c);
        bcy.applyOnGPU(c);
        eqMu.computeRHS(muE);          // muE = f'(cⁿ)
        bcx.applyOnGPU(muE);
        bcy.applyOnGPU(muE);
        semi.advance();                // (I + dt·Mκ∇⁴) cⁿ⁺¹ = cⁿ + dt·M∇²muE

        if (s % 500 == 0 || s == nSteps) {
            const double cMax  = reduce::fieldMaxAbs(c);
            const double drift = reduce::fieldSum(c) - mass0;
            std::printf("  step %5d  t=%.4e  max|c|=%.3f  mass drift=%+.2e"
                        "  CG %d iters\n",
                        s, semi.time, cMax, drift,
                        semi.lastSolve().iterations);
            c.downloadCurrFromDevice();
            c.write("output/c_" + std::to_string(s) + ".vts",
                    FieldFormat::VTS);
        }
    }
    cudaDeviceSynchronize();
    std::printf("done: %d steps in %.2f s (%.3f ms/step)\n",
                nSteps, wall.seconds(), wall.seconds() * 1e3 / nSteps);
    return 0;
}
