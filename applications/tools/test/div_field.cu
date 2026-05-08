/***********************************************************************\
 *
 *  div_field — compute divergence of a vector field
 *
 *  读入 .vfield 文件中的向量场 U，用 equation DSL 计算
 *      divU = div(U) = dU_x/dx + dU_y/dy
 *  输出为 VTS 格式（可直接在 ParaView 中打开）。
 *
 *  用法:
 *    ./div_field settings/settings.jsonc
 *
 \***********************************************************************/

#include "mesh/Mesh.h"
#include "field/ScalarField.h"
#include "field/VectorField.h"
#include "boundary/BCFactory.h"
#include "equation/Equation.h"
#include "IO/ConfigFile.h"
#include "IO/FieldIO.h"
#include "IO/FieldFormat.h"

#include <iostream>
#include <string>

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

    // === 2. 读入向量场 U =====================================================
    const std::string input_path  = cfg["input"]["path"];
    const std::string output_path = cfg["output"]["path"];

    std::cout << "Reading vector field: " << input_path << "\n";
    VectorField U = IO::readVectorField(mesh, input_path, /*ghost=*/1);
    U.allocDevice();
    U.uploadAllToDevice();

    // === 3. 边界条件（为 ghost 格填充服务）==================================
    auto  bcSet = buildBCs(cfg["boundary_conditions"]);
    auto& bcs   = bcSet.ptrs;

    for (int c = 0; c < U.nComponents(); ++c)
        for (auto* bc : bcs) bc->applyOnGPU(U[c]);

    // === 4. 输出场 divU ======================================================
    ScalarField divU(mesh, "divU", /*ghost=*/1);
    divU.fill(0.0);
    divU.allocDevice();
    divU.uploadAllToDevice();

    // === 5. 方程: divU = div(U) = dU_x/dx + dU_y/dy ========================
    Equation eq(divU, "divU");
    eq.setRHS(div(U));   // div(VectorField) → RHSExpr (sum of grad per axis)

    eq.computeRHS(divU);
    divU.downloadAllFromDevice();

    // === 6. 写出 VTS =========================================================
    std::cout << "Writing divU to: " << output_path << "\n";
    IO::writeField(divU, output_path, FieldFormat::VTS);

    std::cout << "Done.\n";
    return 0;
}
