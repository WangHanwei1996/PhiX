# Plan：新增轴对称边界条件 `AxisymmetricBC`

> 状态：**待审核**（未开始改动代码）。审核通过后，本文件将被合并进 [boundary.md](boundary.md) / [boundary_en.md](boundary_en.md)，并删除本计划文件。

---

## 1. 设计目标

为 PhiX 增加一类 **轴对称 / 镜像对称边界条件**，统一覆盖以下两种典型用法：

1. **柱坐标 / 球坐标的对称轴**（$r=0$ 处）。在 `Mesh::CoordSys` 取 `CYLINDRICAL` 或 `SPHERICAL` 时，径向 X 轴的 LOW 侧通常为对称轴。
2. **直角坐标下的对称面**（mirror plane），用于半域计算节省一半网格量。

两种语义的 ghost-cell 填充规则完全一致——都是镜像反射；区别仅在于使用者的物理语境。因此在 PhiX 中我们用 **同一个类** `AxisymmetricBC` 承载，命名贴近用户最初提出的需求；如果将来需要在 API 层做语义区分，可再加一个别名 `SymmetryBC` 指向同一实现。

> **范围声明**：本 Plan 仅负责 ghost-cell 的对称填充，**不**触及 Equation 模块的度量项（如柱坐标 Laplacian 中的 $\frac{1}{r}\partial_r(r\,\partial_r\phi)$ 形式）。度量感知算子是独立工作项，不在本次改动范围内。

---

## 2. Ghost 单元数学规则

设 BC 作用轴为 $A$，物理网格数 $N_A$，ghost 层数 $G$。stored 索引与 [BoundaryCondition.h](../include/boundary/BoundaryCondition.h) 现有约定一致（`stored = physical + ghost`）。

### 2.1 ScalarField — 偶反射（even reflection）

LOW 侧（对称轴在 $i = -\frac{1}{2}$ 即第 0 个物理 cell 的下表面）：

$$
f[-g,\,j,\,k] \;=\; f[\,g-1,\,j,\,k\,], \qquad g = 1, 2, \dots, G
$$

HIGH 侧：

$$
f[N_A + g - 1,\,j,\,k] \;=\; f[N_A - g,\,j,\,k\,], \qquad g = 1, 2, \dots, G
$$

> 备注：$g=1$ 时偶反射退化为 `f[-1]=f[0]`，与 `NoFluxBC` 完全一致；但 $g\geq 2$ 时两者发散——`NoFluxBC` 是常数外推（`f[-2]=f[0]`），而 `AxisymmetricBC` 是真正的镜像（`f[-2]=f[1]`）。这使得在 ghost ≥ 2 的高阶模板下 `AxisymmetricBC` 给出二阶精度的零通量，比 `NoFluxBC` 更准确。

### 2.2 VectorField — 法向分量奇反射，切向分量偶反射

设 vector 分量 $\mathbf{v}=(v_0, v_1, v_2)$，BC 作用轴 `axis = A`，记法向分量为 $v_A$：

LOW 侧：

$$
\begin{aligned}
v_A[-g,\,j,\,k]    &= -\,v_A[\,g-1,\,j,\,k\,] \\
v_{T}[-g,\,j,\,k]  &= +\,v_{T}[\,g-1,\,j,\,k\,]   \quad \text{对所有} \;T \neq A
\end{aligned}
$$

HIGH 侧：

$$
\begin{aligned}
v_A[N_A+g-1,\,j,\,k]    &= -\,v_A[N_A-g,\,j,\,k\,] \\
v_{T}[N_A+g-1,\,j,\,k]  &= +\,v_{T}[N_A-g,\,j,\,k\,]
\end{aligned}
$$

物理含义：法向速度在对称面/对称轴处为零（$v_n|_{\text{wall}}=0$）；切向速度的法向梯度为零。

> **VectorField 的分量顺序约定**：仅当 `vf.nComponents() > int(axis)` 时，才把第 `int(axis)` 个分量视作法向。否则（例如把 VectorField 当作通用多通道标量使用）退化为对所有分量做偶反射，并在 `applyOnGPU(VectorField&)` 内部留下 `assert` / `runtime_warning` 提示。文档中显式说明：本 BC 假定 `VectorField` 分量序为 (x, y, z)，与 `Mesh` 的轴号一致。

---

## 3. 类与文件结构

### 3.1 新增文件

- `include/boundary/AxisymmetricBC.h`

```cpp
#pragma once

#include "boundary/BoundaryCondition.h"

namespace PhiX {

// ---------------------------------------------------------------------------
// AxisymmetricBC  (Symmetry / Mirror)
//
// 镜像反射型边界。Ghost 单元按下式填充：
//   - ScalarField                : 偶反射  f[-g] = f[ g-1]
//   - VectorField 法向分量 v_A   : 奇反射  v_A[-g] = -v_A[ g-1]
//   - VectorField 切向分量 v_T   : 偶反射  v_T[-g] = +v_T[ g-1]
//
// 物理用途：
//   1) 柱/球坐标的对称轴 (r = 0)，axis=Axis::X, side=Side::LOW
//   2) 直角坐标的对称面（半域计算）
//
// 注意：本 BC 仅处理 ghost；坐标系度量项需由算子层另行支持。
// ---------------------------------------------------------------------------

class AxisymmetricBC : public BoundaryCondition {
public:
    AxisymmetricBC(Axis axis, Side side = Side::LOW);

    using BoundaryCondition::applyOnCPU;
    using BoundaryCondition::applyOnGPU;

    void applyOnCPU(ScalarField& f) const override;
    void applyOnGPU(ScalarField& f) const override;

    void applyOnCPU(VectorField& vf) const override;
    void applyOnGPU(VectorField& vf) const override;
};

} // namespace PhiX
```

### 3.2 修改文件

| 文件 | 改动 |
|------|------|
| [src/boundary/Boundary.cu](../src/boundary/Boundary.cu) | 新增 `kernel_mirror`（带 `double sign` 参数）+ `AxisymmetricBC` 的 CPU/GPU 实现 |
| [src/boundary/BCFactory.cpp](../src/boundary/BCFactory.cpp) | 在 `addSide` 中注册 `"Axisymmetric"` 类型；同时不允许成对 Periodic 化 |
| [include/boundary/BCFactory.h](../include/boundary/BCFactory.h) | 更新顶部 doc-comment 的「Supported BC types」与示例 |
| [doc/boundary.md](boundary.md) / [doc/boundary_en.md](boundary_en.md) | 新增 `AxisymmetricBC` 一节，参照现有 `NoFluxBC` 排版 |

CMake 不需要改动：`Boundary.cu` 已被打包进 `phix` 库；新头文件随 `include/` 被打包安装。

---

## 4. GPU Kernel 设计

复用现有 [`FaceParams`](../src/boundary/Boundary.cu) 抽象，新增一个统一的 mirror kernel：

```cpp
__global__ void kernel_mirror(
        double* data,
        int n_face0, int n_face1,
        int axis_stride, int n_axis, int ghost,
        int face_stride0, int face_stride1,
        bool do_low, bool do_high,
        double sign)        // +1.0 (even) 或 -1.0 (odd)
{
    int t0 = blockIdx.x * blockDim.x + threadIdx.x;
    int t1 = blockIdx.y * blockDim.y + threadIdx.y;
    if (t0 >= n_face0 || t1 >= n_face1) return;

    int face_off = t0 * face_stride0 + t1 * face_stride1;

    if (do_low) {
        for (int g = 1; g <= ghost; ++g) {
            int dst = (ghost - g)       * axis_stride + face_off;
            int src = (ghost + g - 1)   * axis_stride + face_off;
            data[dst] = sign * data[src];
        }
    }
    if (do_high) {
        for (int g = 1; g <= ghost; ++g) {
            int dst = (ghost + n_axis + g - 1) * axis_stride + face_off;
            int src = (ghost + n_axis - g)     * axis_stride + face_off;
            data[dst] = sign * data[src];
        }
    }
}
```

- `AxisymmetricBC::applyOnGPU(ScalarField&)` → 单次启动，`sign = +1.0`。
- `AxisymmetricBC::applyOnGPU(VectorField&)` → 对每个分量启动一次：分量 index `c == int(axis)` 时 `sign = -1.0`，否则 `+1.0`。
- CPU 实现以同样 `sign` 走串行循环（参照现有 NoFlux/Fixed 的 CPU 路径）。

线程几何与 NoFlux/Fixed 相同：`dim3 block(16,16); grid = ceil(n_face0/16) × ceil(n_face1/16)`。

---

## 5. JSON 配置接入

`buildBCs` 现有的 JSON 字符串模式扩展一个关键字 `"Axisymmetric"`，与 `"NoFlux"` 同级（独立指定 LOW / HIGH 端）。规则：

- 不可与 `"Periodic"` 在同一轴的另一侧混用（沿用现有 Periodic 必须成对的约束）。
- 可独立放在 LOW 或 HIGH，或两侧同时使用。
- 与 `"NoFlux"` 可在同一轴的两侧分别出现（例如 `x_min: Axisymmetric, x_max: NoFlux`）。

JSON 示例（柱坐标 r-z 域，r=0 为对称轴）：

```jsonc
"boundaryConditions": {
    "x_min": "Axisymmetric",   // r = 0 对称轴
    "x_max": "NoFlux",         // r = R_max 零通量
    "y_min": "NoFlux",
    "y_max": "NoFlux"
}
```

`BCFactory.cpp` 改动伪代码：

```cpp
auto addSide = [&](const std::string& type, Side side) {
    if (type == "NoFlux") {
        set.storage.push_back(std::make_unique<NoFluxBC>(axis, side));
    } else if (type == "Axisymmetric") {
        set.storage.push_back(std::make_unique<AxisymmetricBC>(axis, side));
    } else {
        throw std::runtime_error("buildBCs: unsupported BC type \"" + type + "\"");
    }
    set.ptrs.push_back(set.storage.back().get());
};
```

> 备注：`FixedBC` 目前在 `BCFactory.cpp` 中也未注册。本 Plan **不顺带**修复，避免范围扩张；若需要可单独提一项任务。

---

## 6. 兼容性与风险

| 项目 | 评估 |
|------|------|
| ABI / 二进制兼容 | 仅新增类与符号，不改基类，安全。 |
| 既有用例 | `Periodic`/`NoFlux`/`Fixed` 的 JSON 与 C++ API 均不受影响。 |
| Solver 调用 | `Solver` 只依赖 `BoundaryCondition*` 多态接口，无需改动。 |
| ScratchPool / Composite-Term BC | `lap(Term, bcs, ...)` 等高阶 DSL 通过 `ScalarField::makeShell` 把 scratch 包成 shell 后调用 `applyOnGPU(ScalarField&)`，新 BC 自动适配。 |
| VectorField 分量顺序 | 仅在 `nComponents > int(axis)` 时翻号；否则全部偶反射并保持安全。文档显式声明假设。 |
| 坐标系正确性 | Plan 不改算子，柱/球坐标度量项缺失为既存问题，不归本任务管。文档中标注 *Future work*。 |

---

## 7. 验证计划

新 BC 上线后，做以下三组最小验证（可在 `develop/` 下添加临时测试，不进入正式 tutorial）：

1. **Scalar 偶反射一致性**：构造一个简单 1D ScalarField，手工填值，调用 `AxisymmetricBC` 后对比 ghost 与镜像源是否完全相等（CPU & GPU 双路径 download 后比较）。
2. **VectorField 奇反射一致性**：构造 2D VectorField (vx, vy)，对 `Axis::X, Side::LOW` 应用 BC，验证 `vx` ghost = `−`内部、`vy` ghost = `+`内部。
3. **数值收敛**：拿 `develop/Spinodal Decomposition/` 已有 case 改成半域 + Axisymmetric mirror，与全域 Periodic 解的左半部分逐 cell 比对，误差应在浮点级别。

---

## 8. 实施步骤（审核通过后按序执行）

1. 创建 `include/boundary/AxisymmetricBC.h`。
2. 在 `src/boundary/Boundary.cu` 末尾追加 `kernel_mirror` + `AxisymmetricBC` 实现，并在文件顶部 `#include "boundary/AxisymmetricBC.h"`。
3. 修改 `src/boundary/BCFactory.cpp`：`#include` + 在 `addSide` 中加 `"Axisymmetric"` 分支。
4. 更新 `include/boundary/BCFactory.h` 顶部 doc-comment 的「Supported BC types」列表。
5. 在 `doc/boundary.md` 与 `doc/boundary_en.md` 中插入 `AxisymmetricBC` 章节（结构镜像 `NoFluxBC` 一节）。
6. `cmake --build build -j` 编译验证。
7. 跑第 7 节的三组验证。
8. 删除本 Plan 文件。

---

## 9. 待用户确认的开放问题

1. **命名**：使用 `AxisymmetricBC`（贴近你最初的措辞）还是 `SymmetryBC`（更通用）？我倾向 `AxisymmetricBC`，并在 doc 中说明它同时承担「对称面」语义。
2. **默认 Side**：构造函数默认 `Side::LOW` 是否合适？（绝大多数轴对称用例对称轴位于 r=0 即 X 轴 LOW 侧。）
3. **JSON 关键字**：`"Axisymmetric"` 是否需要同时支持别名 `"Symmetry"` / `"Mirror"`？
4. **VectorField 法向自动翻号**是否符合你期望的默认行为？还是希望用一个显式构造参数 `bool flipNormalForVector = true` 来控制？
5. 是否需要顺手把 `FixedBC` 的 JSON 注册一并补上？（独立工作项，可不做。）

请逐条回复或直接给出修订意见，确认后我即开始按第 8 节执行。
