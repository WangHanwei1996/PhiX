# PhiX v2.5.1 发布说明

**发布日期**：2026-06-01  
**标签**：`v2.5.1`  
**作者**：Wang Hanwei <wanghanweibnds2015@gmail.com>

---

## 概述

v2.5.1 新增场的坐标初始化接口，支持两种形式：

1. **C++ lambda**：`field.initialize([](double x, double y, double z){ ... })`
2. **配置文件命名初始化器**：`IO::initField(f, 0, "random:0.4:0.6")`

---

## 变动详情

### 1. `ScalarField::initialize(Fn)`

**修改文件**：`include/field/ScalarField.h`

新增 header-only 模板方法：

```cpp
template<typename Fn>
void initialize(Fn fn);
// fn 签名：double fn(double x, double y, double z)
```

对每个物理格心调用 `fn(mesh.coord(0,i), mesh.coord(1,j), mesh.coord(2,k))`
并写入 `curr`。Ghost 层保持不变（由 BC 负责填充）。

### 2. `VectorField::initialize(Fn)` / `initializeComponent(comp, Fn)`

**修改文件**：`include/field/VectorField.h`

```cpp
// fn 签名：auto fn(double x, double y, double z) -> indexable[N]
template<typename Fn>
void initialize(Fn fn);

// 单分量初始化
template<typename Fn>
void initializeComponent(int comp, Fn fn);
```

### 3. `IO::initField` 命名初始化器重载

**修改文件**：
- `include/IO/FieldIO.h`
- `src/IO/FieldIO.cpp`

```cpp
void initField(ScalarField& f, int startStep, const std::string& namedInit);
```

支持的格式：

| 字符串 | 说明 |
|---|---|
| `"uniform:0.5"` | 常数 0.5 |
| `"random:0.4:0.6"` | 均匀随机 [0.4, 0.6]，以字段名为种子，结果可复现 |
| `"linear:x:0.0:1.0"` | 沿 x 方向线性插值 [0, 1] |
| `"linear:y:lo:hi"` | 沿 y 方向 |
| `"linear:z:lo:hi"` | 沿 z 方向 |

warm start（`startStep > 0`）忽略 `namedInit`，仍从 `output/` 读取。

---

## 用法示例

### Lambda 初始化

```cpp
// 二维相场：中心圆形核
ScalarField phi(mesh, "phi");
double cx = mesh.origin[0] + mesh.n[0] * mesh.d[0] * 0.5;
double cy = mesh.origin[1] + mesh.n[1] * mesh.d[1] * 0.5;
phi.initialize([cx, cy](double x, double y, double) {
    double r = std::sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy));
    return (r < 5.0) ? 1.0 : 0.0;
});
```

### 配置文件命名初始化

```cpp
// settings.jsonc 中增加 "named_init": "random:0.45:0.55"
std::string namedInit = cfg["initialize"].value("named_init", "");
IO::initField(c, startStep, namedInit);
```

---

## 验证结果

```
cmake --build . -j4 && ctest --output-on-failure
100% tests passed, 0 tests failed out of 5
```
