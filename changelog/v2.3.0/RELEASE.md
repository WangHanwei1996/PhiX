# PhiX v2.3.0 发布说明

**发布日期**：2026-06-01  
**标签**：`v2.3.0`  
**作者**：Wang Hanwei <wanghanweibnds2015@gmail.com>

---

## 概述

v2.3.0 继续推进底层场管理重构，引入独立的 `FieldLayout` 模块，将 ghost 宽度、存储尺寸和索引映射从 `ScalarField` / `VectorField` 内部逻辑中抽离出来，形成统一的布局描述对象。

本版本的目标是先完成**布局抽象落地**，同时保持现有调用点和公开字段语义不变，为后续按 scheme 驱动 ghost 层数以及引入更复杂 centering 形式做准备。

---

## 架构变动

### 1. 新增 `FieldLayout` 模块

**文件**：`include/field/FieldLayout.h`、`src/field/FieldLayout.cpp`

新增：

```cpp
enum class Centering {
    CELL
};

class FieldLayout {
public:
    const Mesh* mesh;
    int         ghost;
    Centering   centering;
    int         storedDims[3];
    std::size_t storedSize;

    FieldLayout(const Mesh& mesh, int ghost = 1,
                Centering centering = Centering::CELL);

    int index(int i, int j, int k) const;
};
```

当前版本只实现 `CELL` 型布局，但 `Centering` 枚举已正式进入接口层。

`FieldLayout` 负责：

- 保存 `ghost`
- 计算 `storedDims`
- 计算 `storedSize`
- 提供统一的 `index(i,j,k)` 映射

### 2. `ScalarField`：持有 `FieldLayout`

**文件**：`include/field/ScalarField.h`、`src/field/ScalarField.cu`

`ScalarField` 新增成员：

```cpp
FieldLayout layout;
```

并新增构造函数：

```cpp
ScalarField(const FieldLayout& layout, const std::string& name);
```

原构造函数仍保留：

```cpp
ScalarField(const Mesh& mesh, const std::string& name, int ghost = 1);
```

但内部已转发为：

```cpp
ScalarField(FieldLayout(mesh, ghost), name)
```

兼容性策略：

- 公开字段 `ghost`
- 公开字段 `storedDims`
- 公开字段 `storedSize`

均继续保留，并在构造时从 `layout` 镜像同步，保证现有代码不需要立即迁移。

`ScalarField::index()` 也已改为委托给 `layout.index()`。

### 3. `VectorField`：持有 `FieldLayout`

**文件**：`include/field/VectorField.h`、`src/field/VectorField.cu`

`VectorField` 同样新增：

```cpp
FieldLayout layout;
```

并新增构造函数：

```cpp
VectorField(const FieldLayout& layout,
            const std::string& name,
            int nComponents);
```

原构造函数保留，但内部转发到 `FieldLayout` 版本。

各分量 `ScalarField` 现统一使用同一个 `FieldLayout` 构造，避免不同分量重复推导相同布局信息。

---

## 测试

### 新增 `field` 模块测试

**文件**：`test/moduleTest/field/CMakeLists.txt`、`test/moduleTest/field/test_field.cpp`

覆盖内容：

- `FieldLayout` 的 `ghost` / `storedDims` / `storedSize` 推导正确
- `FieldLayout::index()` 映射正确
- `ScalarField(layout, name)` 能正确继承布局信息
- `ScalarField::index()` 与 `layout.index()` 一致
- `VectorField(layout, name, nComponents)` 能正确继承布局信息
- 三分量命名 `u_x/u_y/u_z` 保持不变

同时顶层 `moduleTest` 已注册：

- `module_mesh`
- `module_boundary`
- `module_field`

---

## 兼容性说明

本版本是**内部重构版本**，对外行为保持不变：

- 旧的 `ScalarField(mesh, name, ghost)` 仍可继续使用
- 旧的 `VectorField(mesh, name, nComponents, ghost)` 仍可继续使用
- 旧代码直接读取 `ghost` / `storedDims` / `storedSize` 仍然有效

新的推荐写法开始支持：

```cpp
FieldLayout layout(mesh, 2);
ScalarField phi(layout, "phi");
VectorField u(layout, "u", 3);
```

---

## 已验证

以下验证已通过：

- `cmake --build build -j4`
- `ctest --output-on-failure`

测试结果：

- `module_mesh` 通过
- `module_boundary` 通过
- `module_field` 通过
