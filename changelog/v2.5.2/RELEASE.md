# PhiX v2.5.2 发布说明

**发布日期**：2026-06-01  
**标签**：`v2.5.2`  
**作者**：Wang Hanwei <wanghanweibnds2015@gmail.com>

---

## 概述

v2.5.2 引入半网格点（staggered）数据存储支持：

- `Centering` 枚举扩展为 `CELL | FACE_X | FACE_Y | FACE_Z`
- `FieldLayout` 按 centering 正确计算 `storedDims`（face 方向 n+1，无 ghost）
- 新增独立 `FaceField` 结构，用于存储面心量（如通量、面梯度）

---

## 变动详情

### 1. `Centering` 枚举扩展

**修改文件**：`include/field/FieldLayout.h`

```cpp
enum class Centering { CELL, FACE_X, FACE_Y, FACE_Z };

constexpr int  faceAxis(Centering c);   // 返回 0/1/2，CELL 返回 -1
constexpr bool isFace(Centering c);
```

### 2. `FieldLayout` 面布局计算

**修改文件**：
- `include/field/FieldLayout.h`
- `src/field/FieldLayout.cpp`

| centering | 法向轴 ax 的 storedDims | 切向轴的 storedDims | index() 偏移 |
|---|---|---|---|
| `CELL` | n + 2*ghost | n + 2*ghost | +ghost |
| `FACE_X/Y/Z` | n + 1（无 ghost） | n + 2*ghost | 法向无偏移，切向 +ghost |

`index(i,j,k)` 更新为统一接口，自动按 centering 选择是否加 ghost 偏移。
去掉旧的 CELL-only 校验。

### 3. 新增 `FaceField`

**新增文件**：
- `include/field/FaceField.h`
- `src/field/FaceField.cu`

面心场结构，持有 `const Mesh&` 和 `normalAxis`：

```cpp
FaceField flux_x(mesh, 0, "flux_x");   // x 方向通量，(nx+1) × ny 个值
FaceField flux_y(mesh, 1, "flux_y");   // y 方向通量，nx × (ny+1) 个值
```

主要接口：

| 方法 | 说明 |
|---|---|
| `fill(val)` | 常数初始化 |
| `initialize(fn)` | lambda 初始化，`fn(x,y,z)->double`；法向坐标为面坐标（无 +0.5） |
| `faceCoord(i_n)` | 第 i_n 个面的法向坐标：`origin + i_n * d` |
| `allocDevice()` / `freeDevice()` | GPU 内存管理 |
| `uploadToDevice()` / `downloadFromDevice()` | CPU↔GPU 传输 |

---

## 坐标约定

```
Cell centres:  x = origin + (i + 0.5) * dx    (i = 0..n-1)
Face centres:  x = origin +  i        * dx    (i = 0..n,  n+1 faces)
```

---

## 验证结果

```
cmake --build . -j4 && ctest --output-on-failure
100% tests passed, 0 tests failed out of 5
```
