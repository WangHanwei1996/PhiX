// ---------------------------------------------------------------------------
// allen_cahn_semi — 半隐式 Allen-Cahn 曲率驱动收缩圆（定量验证示例）
//
//   ∂φ/∂t = M [ κ∇²φ − W g'(φ) ] ,   g = φ²(1−φ)² （双阱）
//
// IMEX 拆分：κ∇² 与 Eyre 线性稳定化项 −W·s·φ 隐式（LaplacianOp 带
// shift），非线性余项显式。
//
// 教学要点——参数区决定半隐式的收益：
//   dt_react/dt_diff = 4κ/(W·dx²) = 2δ²/dx²（δ 为界面宽度参数）。
//   本例取宽界面 δ = 8dx → 比值 ≈ 128：扩散刚度主导，dt 放大 20×
//   仍满足 dt·M·W·s ≪ 1（稳定化几乎不引入时滞，定量正确）。
//   反之界面越薄（δ ~ 2dx）反应与扩散刚度同阶，dt·M·W ≳ 1 时
//   稳定化 BE 的一阶时滞会把界面动力学拖慢 ~(1+dt·M·W·s) 倍——
//   稳定 ≠ 准确，调 dt 时盯住 dt·M·W。
//
// 定量校验（内置）：曲率流理论下圆半径满足 dR/dt = −M·κ/R，即
//     R(t)² = R0² − 2·M·κ·t
// 程序每隔一段用设备端归约测相场面积 A=Σφ·dx²（R=√(A/π)），
// 打印测量值与理论值的相对偏差——一个能自我检验的教学算例。
//
// 运行：./allen_cahn_semi [nSteps=2000]   输出 output/phi_*.vts
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
    const int    nSteps = (argc > 1) ? std::atoi(argv[1]) : 1000;
    const int    N  = 256;
    const double L0 = 1.0, dx = L0 / N;

    const double Mmob  = 1.0;
    const double kappa = 0.01;
    const double delta = 8.0 * dx;                  // 界面宽度参数（8 格，充分解析）
    const double W     = 2.0 * kappa / (delta * delta);
    const double R0    = 0.30;

    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                    N, dx, 0.0, N, dx, 0.0);

    // 1D 平衡剖面 φ = ½(1 − tanh((r−R0)/δ))，δ = √(2κ/W)
    ScalarField phi(mesh, "phi", 1);
    phi.initialize([&](double x, double y, double) {
        const double r = std::sqrt((x - 0.5) * (x - 0.5)
                                   + (y - 0.5) * (y - 0.5));
        return 0.5 * (1.0 - std::tanh((r - R0) / delta));
    });
    phi.allocDevice();
    phi.uploadAllToDevice();

    NoFluxBC bcXlo(mesh.facePatch(Axis::X, Side::LOW));
    NoFluxBC bcXhi(mesh.facePatch(Axis::X, Side::HIGH));
    NoFluxBC bcYlo(mesh.facePatch(Axis::Y, Side::LOW));
    NoFluxBC bcYhi(mesh.facePatch(Axis::Y, Side::HIGH));
    std::vector<BoundaryCondition*> bcs = {&bcXlo, &bcXhi, &bcYlo, &bcYhi};

    // 显式部分：N(φ) = −M·W·g'(φ)（本参数区 dt·M·W ≈ 0.16，显式反应
    // 项本身稳定，无需稳定化）。若要把 dt 推过反应极限（dt·M·W ≳ 1），
    // 用 LaplacianOp 的第三个参数加 Eyre 线性稳定化 shift = M·W·s
    // （s ≥ max g''），并把显式 RHS 换成 W(s·φ − g'(φ)) —— 注意稳定化
    // 在 dt·M·W·s ≳ 1 时会把界面动力学人为拖慢 ~(1+dt·M·W·s) 倍。
    Equation eq(phi, "ac");
    eq.setRHS(pw(phi, PHIX_FN (Real p) {
        const Real gp = Real(2) * p * (Real(1) - p) * (Real(1) - Real(2) * p);
        return -gp;
    }, Mmob * W));

    // 隐式部分：L = M·κ∇²
    LaplacianOp L(Mmob * kappa, bcs);

    const double dtExplicit = 0.25 * dx * dx / (Mmob * kappa);
    const double dt = 20.0 * dtExplicit;        // dt·M·W ≈ 0.16：稳且准
    std::printf("allen_cahn_semi: N=%d  W=%.3g kappa=%.3g delta=%.4f"
                "  dt=%.3e (20x explicit %.3e, dt·M·W=%.2f)\n",
                N, W, kappa, delta, dt, dtExplicit, dt * Mmob * W);

    SemiImplicitSolver::CGOptions cgo;
    cgo.relTol = 1e-9;
    SemiImplicitSolver semi(eq, bcs, L, dt, cgo);

    std::filesystem::create_directories("output");
    const double cellA = dx * dx;

    perf::WallTimer wall;
    for (int s = 1; s <= nSteps; ++s) {
        semi.advance();
        if (s % 200 == 0 || s == nSteps) {
            const double area = reduce::fieldSum(phi) * cellA;
            const double Rm   = std::sqrt(area / M_PI);
            const double R2th = R0 * R0 - 2.0 * Mmob * kappa * semi.time;
            const double Rth  = (R2th > 0.0) ? std::sqrt(R2th) : 0.0;
            std::printf("  step %5d  t=%.4e  R=%.4f  R_theory=%.4f"
                        "  dev=%+.2f%%  CG %d iters\n",
                        s, semi.time, Rm, Rth,
                        (Rth > 0 ? 100.0 * (Rm - Rth) / Rth : 0.0),
                        semi.lastSolve().iterations);
            phi.downloadCurrFromDevice();
            phi.write("output/phi_" + std::to_string(s) + ".vts",
                      FieldFormat::VTS);
        }
    }
    cudaDeviceSynchronize();
    std::printf("done: %d steps in %.2f s (%.3f ms/step)\n",
                nSteps, wall.seconds(), wall.seconds() * 1e3 / nSteps);
    return 0;
}
