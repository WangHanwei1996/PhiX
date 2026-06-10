# v2.7.0 — 材料模块（`PhiX::Material`）

## 摘要

新增 `material` 模块，为 PhiX 提供可扩展的材料属性管理框架。
首期实现二元合金自由能管理：通过浓度 $c$ 和温度 $T$ 查询自由能 $f(c, T)$，
底层使用双线性插值离散数据表，支持从文件加载。

---

## 模块结构

```
include/material/
    IMaterial.h         抽象基类（可扩展接口）
    FreeEnergyTable.h   二维自由能查找表（双线性插值）
    BinaryAlloy.h       二元合金热力学模型
    Material.h          伞头文件（include 全部）

src/material/
    FreeEnergyTable.cpp
    BinaryAlloy.cpp
```

两个源文件已注册到 `phix` 静态库（`CMakeLists.txt`）。

---

## 核心 API

### `FreeEnergyTable`

均匀网格离散表：$c \in [c_\min, c_\max]$（`nc` 点），$T \in [T_\min, T_\max]$（`nT` 点）。

```cpp
// 从参数和数据向量构造
FreeEnergyTable tbl(c_min, c_max, nc, T_min, T_max, nT, data);

// 从 .fetab 文件加载
auto tbl = FreeEnergyTable::fromFile("data/FeB.fetab");

// 双线性插值求值（越界自动截断到表格范围）
double f    = tbl.f(c, T);
double dfdc = tbl.dfdc(c, T);   // ∂f/∂c，中心差分
double dfdT = tbl.dfdT(c, T);   // ∂f/∂T，中心差分
```

#### `.fetab` 文件格式

```
# 注释行（以 # 开头）可任意多行
# nc  nT  c_min  c_max  T_min  T_max
40 100 0.0 1.0 300.0 1800.0
# 接下来 nc 行，每行 nT 个空白分隔的浮点数（行对应 c，列对应 T）
-1.23e4  -1.22e4  ...
...
```

### `BinaryAlloy`

继承 `IMaterial`，封装一张 `FreeEnergyTable`。

```cpp
// 从文件构造
auto alloy = BinaryAlloy::fromFile("data/FeB.fetab", "Fe-B");

// 或从已有表格构造
BinaryAlloy alloy("Fe-B", std::move(table));

// 查询热力学性质
double f    = alloy.freeEnergy(c, T);
double dfdc = alloy.dfdc(c, T);
double dfdT = alloy.dfdT(c, T);

// 访问底层表格
const FreeEnergyTable& tbl = alloy.table();
```

### `IMaterial`

轻量抽象基类，供后续扩展：

```cpp
class IMaterial {
public:
    virtual std::string name() const = 0;
};
```

可在此基础上派生多元合金、纯元素、非晶材料等模型，并统一存入
`std::vector<std::unique_ptr<IMaterial>>` 材料库。

---

## 修改文件

| 文件 | 说明 |
|------|------|
| `include/material/IMaterial.h`      | 新增，材料抽象基类 |
| `include/material/FreeEnergyTable.h`| 新增，二维自由能查找表声明 |
| `include/material/BinaryAlloy.h`    | 新增，二元合金模型声明 |
| `include/material/Material.h`       | 新增，伞头文件 |
| `src/material/FreeEnergyTable.cpp`  | 新增，查找表实现（双线性插值、文件加载） |
| `src/material/BinaryAlloy.cpp`      | 新增，二元合金实现 |
| `CMakeLists.txt`                    | 将两个新源文件加入 `phix` 库 |
