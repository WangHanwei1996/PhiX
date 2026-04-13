# PhiX Quickstart

PhiX 是一个显式有限差分 GPU 相场计算套件，核心由五个层次的类构成：

```
Mesh  →  Field  →  BoundaryCondition  →  Equation  →  Solver
```

---

## 1. 网格（Mesh）

`Mesh` 是纯参数容器，描述结构化正交网格的尺寸和间距，不持有任何大数组。

```cpp
#include "mesh/Mesh.h"
using namespace PhiX;

// 二维均匀笛卡尔网格，256×256，间距 0.5，原点 (0,0)
Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                256, 0.5, 0.0,   // nx, dx, x0
                                256, 0.5, 0.0);  // ny, dy, y0

mesh.print();   // 打印摘要
mesh.write("case/constant/mesh");   // 持久化到文件
```

**支持维度**：`makeUniform1D` / `makeUniform2D` / `makeUniform3D`  
**支持坐标系**：`CARTESIAN`、`CYLINDRICAL`、`SPHERICAL`

---

## 2. 物理场（Field）

`Field` 在 `Mesh` 给定的网格外套一层 ghost cell（halo），双精度存储两个时间层（`curr` / `prev`），CPU 和 GPU 各一份。

```cpp
#include "field/Field.h"

// 创建场，ghost 层厚度默认为 1
Field phi(mesh, "phi", /*ghost=*/1);

// CPU 初始化
phi.fill(0.0);            // curr 和 prev 全置 0

// 也可以用物理坐标手动初始化 curr
for (int j = 0; j < mesh.n[1]; ++j)
for (int i = 0; i < mesh.n[0]; ++i) {
    double x = mesh.coord(0, i);
    double y = mesh.coord(1, j);
    phi.curr[phi.index(i, j)] = std::exp(-(x*x + y*y));
}

// 分配 GPU 内存并上传
phi.allocDevice();
phi.uploadAllToDevice();

// 写文件 / 读文件
phi.write("case/0/phi.field");
Field phi2 = Field::readFromFile(mesh, "case/0/phi.field");
```

**索引规则**：物理格点 `i ∈ [0, nx)`，ghost 格点 `i ∈ [-ghost, 0)` 和 `[nx, nx+ghost)`，均通过 `phi.index(i, j, k)` 访问。

---

## 3. 边界条件（BoundaryCondition）

每个边界条件对象只作用于指定的轴和侧面，可以在不同方向上混合使用。

```cpp
#include "boundary/PeriodicBC.h"
#include "boundary/NoFluxBC.h"
#include "boundary/FixedBC.h"

PeriodicBC bc_x(Axis::X);                      // x 方向周期
NoFluxBC   bc_y(Axis::Y, Side::BOTH);          // y 方向零通量（Neumann）
FixedBC    bc_z(Axis::Z, Side::LOW, 0.0);      // z 低侧 Dirichlet = 0（预留）
```

| 类型 | 数学含义 | ghost 操作 |
|------|----------|-----------|
| `PeriodicBC` | $\phi[-g] = \phi[N-g]$，$\phi[N+g-1] = \phi[g-1]$ | 两侧互抄 |
| `NoFluxBC` | $\partial\phi/\partial n = 0$ | ghost = 最近物理格点值 |
| `FixedBC` | $\phi = \text{value}$ | ghost = 常数 |

---

## 4. 方程（Equation）

`Equation` 描述 $\partial\phi/\partial t = $ RHS，RHS 用 `lap()`、`grad()`、`pw()` 搭积木式组合。

```cpp
#include "equation/Equation.h"

Equation eq(phi, "AllenCahn");

// 物性参数
eq.params["M"]     = 1.0;    // 迁移率
eq.params["kappa"] = 0.5;    // 界面能系数

double M     = eq.params["M"];
double kappa = eq.params["kappa"];

// dφ/dt = M·∇²φ + M·(φ - φ³)
//
// lap(phi)  —— ∇²φ，2 阶中心差分，自动感知维度
// pw(phi, functor)  —— 逐格点函数，functor 必须标注 __host__ __device__
//                       以便同时支持 GPU kernel 和 CPU fallback
eq.setRHS(
    M * lap(phi)
  + M * pw(phi, [] __host__ __device__ (double p) { return p - p*p*p; })
);
```

**内置算子**

| 函数 | 含义 | 说明 |
|------|------|------|
| `lap(f, coeff=1)` | $\text{coeff}\cdot\nabla^2 f$ | 2 阶中心差分，按 dim 自动求和 |
| `grad(f, axis, coeff=1)` | $\text{coeff}\cdot\partial f/\partial x_\text{axis}$ | 2 阶中心差分 |
| `pw(f, func, coeff=1)` | $\text{coeff}\cdot g(f)$ | 模板函数，`func` 须支持 GPU |

**运算符重载**

```cpp
// 以下写法等价，可自由组合
eq.setRHS(2.0 * lap(phi) - 0.5 * grad(phi, 0) + pw(phi, f));
eq.setRHS(lap(phi) + lap(T) * kappa);   // 多场混合
```

---

## 5. 求解器（Solver）

`Solver` 组装 Equation 和边界条件，驱动时间推进。

```cpp
#include "solver/Solver.h"

// 构造：传入方程、边界条件列表、时间步长、积分方案
Solver solver(eq,
              {&bc_x, &bc_y},
              /*dt=*/0.01,
              TimeScheme::RK4);   // 或 TimeScheme::EULER

// 单步推进
solver.advance();

// 批量运行，每 100 步触发回调
solver.run(5000, 100, [&](const Solver& s) {
    // 把 GPU 数据拷回 CPU 再写文件
    phi.downloadCurrFromDevice();
    phi.write("output/phi_" + std::to_string(s.step) + ".field");
    // 也可以在这里打印收敛信息
    phi.print();
});
```

**时间积分方案**

| 方案 | 误差阶 | 每步 RHS 调用次数 | 适用场景 |
|------|--------|-------------------|---------|
| `EULER` | $O(\Delta t)$ | 1 | 快速测试、刚性弱 |
| `RK4` | $O(\Delta t^4)$ | 4 | 精度要求高、Allen-Cahn/Cahn-Hilliard |

---

## 完整示例：2D Allen-Cahn 方程

```cpp
#include "mesh/Mesh.h"
#include "field/Field.h"
#include "boundary/PeriodicBC.h"
#include "equation/Equation.h"
#include "solver/Solver.h"

#include <cmath>
#include <string>

int main() {
    using namespace PhiX;

    // 网格
    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                    256, 0.5, -64.0,
                                    256, 0.5, -64.0);

    // 场：随机初始化
    Field phi(mesh, "phi", 1);
    for (int j = 0; j < mesh.n[1]; ++j)
    for (int i = 0; i < mesh.n[0]; ++i) {
        double val = (double)rand() / RAND_MAX * 2.0 - 1.0;
        phi.curr[phi.index(i, j)] = 0.05 * val;
    }
    phi.allocDevice();
    phi.uploadAllToDevice();

    // 边界：全周期
    PeriodicBC bc_x(Axis::X);
    PeriodicBC bc_y(Axis::Y);

    // 方程：dφ/dt = M·∇²φ + M·(φ - φ³)
    Equation eq(phi, "AllenCahn");
    eq.params["M"] = 1.0;
    double M = eq.params["M"];

    eq.setRHS(
        M * lap(phi)
      + M * pw(phi, [] __host__ __device__ (double p) { return p - p*p*p; })
    );

    // 求解器
    Solver solver(eq, {&bc_x, &bc_y}, 0.01, TimeScheme::RK4);

    // 运行 10000 步，每 500 步写一次
    solver.run(10000, 500, [&](const Solver& s) {
        phi.downloadCurrFromDevice();
        phi.write("output/phi_" + std::to_string(s.step) + ".field");
    });

    return 0;
}
```

---

## 编译（CMake）

项目使用 CMake 统一管理所有 tutorial 的编译，核心库被构建为静态库 `phix`，各 tutorial 单独链接。

**配置和编译（在 PhiX 根目录下执行）：**

```bash
mkdir build && cd build

# 默认 sm_75（Turing），按实际 GPU 修改 PHIX_CUDA_ARCH
cmake .. -DPHIX_CUDA_ARCH=75

make -j$(nproc)
```

常见 GPU 的 `PHIX_CUDA_ARCH` 对照：

| GPU 系列 | 代表型号 | CUDA Arch |
|---------|---------|-----------|
| Turing  | RTX 2080 | `75` |
| Ampere  | RTX 3090, A100 | `86` / `80` |
| Ada     | RTX 4090 | `89` |
| Hopper  | H100 | `90` |

**运行 quickstart tutorial：**

```bash
# 在 build/ 目录下
./tutorials/quickstart/quickstart
```

输出的 `.field` 二进制文件写入运行目录下的 `output/` 文件夹。

**添加新 tutorial：**

1. 在 `tutorials/` 下新建文件夹，如 `tutorials/my_case/`
2. 创建 `main.cu` 和如下 `CMakeLists.txt`：

```cmake
add_executable(my_case main.cu)
target_link_libraries(my_case PRIVATE phix)
set_target_properties(my_case PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
```

3. 在根目录 `CMakeLists.txt` 末尾追加：

```cmake
add_subdirectory(tutorials/my_case)
```
