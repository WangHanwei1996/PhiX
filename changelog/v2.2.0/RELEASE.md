# PhiX v2.2.0 发布说明

**发布日期**：2026-06-01  
**标签**：`v2.2.0`  
**作者**：Wang Hanwei <wanghanweibnds2015@gmail.com>

---

## 概述

v2.2.0 开始了底层网格与边界系统的结构性重构。本版本将原先仅由 `(Axis, Side)` 隐式表示的边界，提升为与 `Mesh` 绑定的显式 `Patch` 拓扑；边界条件对象也从“绑轴和侧”改为“绑 patch”。

这为后续的子面边界、多 patch 拼接、按数值格式驱动 ghost 层数，以及进一步的 scheme / operators 模块拆分打下了基础。

由于这是底层破坏性重构的第一步，本版本临时关闭了应用层与教程的顶层构建入口，仅保留 `phix` 核心库和 `moduleTest` 模块测试参与构建与回归。

---

## 架构变动

### 1. `Mesh`：新增 Patch 拓扑系统

**文件**：`include/mesh/Patch.h`、`src/mesh/Patch.cpp`、`include/mesh/Mesh.h`、`src/mesh/Mesh.cpp`

新增 `Patch` / `IndexBox` / `PatchKind`：

```cpp
struct IndexBox {
    int lo[3];
    int hi[3];
};

struct Patch {
    std::string name;
    Axis        axis;
    Side        side;
    IndexBox    region;
    PatchKind   kind = PatchKind::PHYSICAL;
};
```

`Mesh` 现在在构造后自动生成默认整面 patch：

- `xmin` / `xmax`
- `ymin` / `ymax`
- `zmin` / `zmax`

并新增以下能力：

- `mesh.patch(name)`：按名称查询 patch
- `mesh.facePatches(axis, side)`：查询某个面的所有 patch
- `mesh.facePatch(axis, side)`：要求该面未拆分时的便捷查询
- `mesh.addPatch(...)` / `mesh.removePatch(...)`
- `mesh.removeFacePatches(axis, side)`：清空某个面的 patch 集
- `mesh.validatePatches()`：校验覆盖完整性与无重叠性

这允许将一个整面拆分为多个不重叠的子 patch，例如把 `xmin` 切成 `inlet` / `wall` / `outlet` 三段。

### 2. `BoundaryCondition`：从 `(Axis, Side)` 切换到 `Patch`

**文件**：`include/boundary/BoundaryCondition.h`

边界条件基类签名由：

```cpp
BoundaryCondition(Axis axis, Side side)
```

变为：

```cpp
explicit BoundaryCondition(const Patch& patch)
```

`BoundaryCondition` 现在持有：

```cpp
const Patch& patch;
```

并通过 `axis()` / `side()` 访问 patch 元数据。

### 3. `PeriodicBC` / `NoFluxBC` / `FixedBC`：构造接口更新

**文件**：`include/boundary/PeriodicBC.h`、`include/boundary/NoFluxBC.h`、`include/boundary/FixedBC.h`、`src/boundary/Boundary.cu`

三个边界条件类都改为接收 `const Patch&`：

```cpp
PeriodicBC(const Patch& patch);
NoFluxBC(const Patch& patch);
FixedBC(const Patch& patch, double value);
```

其中：

- `NoFluxBC` 与 `FixedBC` 只作用于 patch 对应的一侧
- `PeriodicBC` 绑定到某轴的低侧 patch，并同时完成该轴双侧 ghost 的周期填充

### 4. `Boundary.cu`：kernel 改为 patch-aware

旧版 `Boundary.cu` 默认按“整面”遍历 ghost 填充区域；新版改为根据 patch 的 `IndexBox region` 仅遍历指定子面区域。

内部新增 `PatchParams`，从 `Patch + ScalarField` 推导出：

- 法向 stride
- 两个切向 stride
- patch 覆盖的切向尺寸
- patch 在存储数组中的起始偏移

CPU / GPU 实现均已同步适配。

### 5. `BCFactory`：构造时显式依赖 `Mesh`

**文件**：`include/boundary/BCFactory.h`、`src/boundary/BCFactory.cpp`

`buildBCs` 签名由：

```cpp
BCSet buildBCs(const nlohmann::json& bc_config);
```

变为：

```cpp
BCSet buildBCs(const Mesh& mesh, const nlohmann::json& bc_config);
```

原因是工厂现在需要通过 `Mesh` 把 legacy 的：

- `x_min` / `x_max`
- `y_min` / `y_max`
- `z_min` / `z_max`

解析到对应的默认 patch（`xmin` / `xmax` / ...）。

注意：如果某个面已经被拆分为多个子 patch，`buildBCs()` 会因 `mesh.facePatch(...)` 无法唯一解析而抛异常；这类场景需要调用者手动为各 patch 构造 BC。

---

## 构建与测试

### 1. 顶层 CMake：暂时只保留核心库与模块测试

**文件**：`CMakeLists.txt`

本版本引入：

```cmake
enable_testing()
add_subdirectory(test/moduleTest)
```

同时临时注释掉了：

- `tutorials/quickstart`
- `applications/solvers/*`
- `develop/*`
- `applications/tools/test`

并在源文件中标注：

```cmake
# TODO(v2.2.0): re-enable after solver refactor
```

### 2. 新增 `moduleTest`

**文件**：`test/moduleTest/CMakeLists.txt`

新增两个模块测试子目录：

- `test/moduleTest/mesh`
- `test/moduleTest/boundary`

#### `mesh` 模块测试

**文件**：`test/moduleTest/mesh/test_mesh.cpp`

覆盖内容：

- 默认 6 个 patch 自动生成
- `xmin` / `xmax` 默认 region 正确
- `xmin` 拆分成三个子 patch 后可通过 `validatePatches()`
- patch 重叠时能正确抛出异常

#### `boundary` 模块测试

**文件**：`test/moduleTest/boundary/test_boundary.cu`

覆盖内容：

- `NoFluxBC` 在低侧子 patch 上只填充对应区域 ghost
- `FixedBC` 在同一侧不同子 patch 上可设置不同 ghost 值
- `PeriodicBC` 在默认整面 patch 上可同时刷新双侧 ghost

---

## 破坏性变更

本版本包含以下破坏性 API 变化：

1. `BoundaryCondition` 及其子类不再接受 `(Axis, Side)` 构造参数
2. `buildBCs()` 现在必须显式传入 `Mesh`
3. 顶层 CMake 暂时不再构建教程与应用求解器

后续你在重构求解器时，需要把：

```cpp
auto bcSet = buildBCs(cfg["boundary_conditions"]);
```

改为：

```cpp
auto bcSet = buildBCs(mesh, cfg["boundary_conditions"]);
```

以及把：

```cpp
NoFluxBC bc(Axis::Y, Side::HIGH);
```

改为：

```cpp
NoFluxBC bc(mesh.patch("ymax"));
```

或在面已拆分时改为对应子 patch 名称。

---

## 已验证

在当前顶层构建范围内，以下验证已通过：

- `cmake -S . -B build`
- `cmake --build build -j4`
- `ctest --output-on-failure`

测试结果：

- `module_mesh` 通过
- `module_boundary` 通过
