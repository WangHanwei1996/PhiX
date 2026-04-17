# PhiX v1.2.0 发布说明

**发布日期**：2026-04-17  
**标签**：`v1.2.0`

---

## 概述

v1.2.0 引入了 **Applications 体系**：将面向最终用户的求解器从 `develop/` 中分离，统一放置在 `applications/` 目录下，并配套了：

1. **外部配置文件驱动**：求解器参数（目前为网格参数）从 `.jsonc` 文件读取，不再硬编码
2. **环境变量管理**（`etc/bashrc`）：仿照 OpenFOAM 风格，`source` 后自动将 `applications/` 下所有可执行程序加入 `PATH`
3. **边界条件虚函数警告修复**：消除了 `PeriodicBC`、`FixedBC`、`NoFluxBC` 中的编译器 warning

---

## 新增内容

### Applications 目录结构

```
applications/
└── solvers/
    └── Cahn-Hillard_double-well/
        └── 2D/
            ├── Cahn-Hillard_double-well.cu
            ├── CMakeLists.txt
            └── CH_2D              ← 可执行文件
```

首个正式求解器 **`CH_2D`**：二维 Cahn-Hilliard 双井势旋节分解，对应方程：

$$\frac{\partial c}{\partial t} = M\nabla^2\mu, \quad \mu = 2\rho(c-c_a)(c-c_b)(2c-c_a-c_b) - \kappa\nabla^2 c$$

### 外部配置文件（JSONC）

参数通过 `.jsonc` 文件传入，支持 `//` 行注释：

```jsonc
{
    "mesh": {
        // 网格点数
        "nx": 200,      // x 方向格点数
        "ny": 200,      // y 方向格点数
        "dx": 1.0,
        "dy": 1.0,
        "x0": 0.0,
        "y0": 0.0
    }
}
```

运行方式：
```bash
# 指定配置文件路径（绝对或相对均可）
CH_2D /path/to/settings.jsonc

# 默认读取当前目录下的 settings/settings.jsonc
CH_2D
```

测试用配置文件位于 `develop/CH_by_solvers/settings/settings.jsonc`。

### 环境配置脚本（`etc/bashrc`）

```bash
source /path/to/PhiX/etc/bashrc
```

- 设置 `$PHIX_DIR` 为项目根目录（自动检测，无需手动指定）
- 扫描 `$PHIX_DIR/applications/` 下所有含可执行文件的目录，逐一 prepend 到 `PATH`
- 推荐写入 `~/.bashrc` 永久生效

---

## 修复

### 边界条件虚函数覆盖警告（`include/boundary/`）

`BoundaryCondition` 基类对 `applyOnCPU` / `applyOnGPU` 各有 `ScalarField` 和 `VectorField` 两个重载，派生类只覆盖 `ScalarField` 版本时编译器会发出 warning #611-D。

**修复方式**：在 `PeriodicBC`、`FixedBC`、`NoFluxBC` 中添加 `using` 声明，将基类的 `VectorField` 重载引入派生类作用域：

```cpp
using BoundaryCondition::applyOnCPU;
using BoundaryCondition::applyOnGPU;
```

---

## 构建变更

- `CMakeLists.txt` 新增 `add_subdirectory("applications/solvers/Cahn-Hillard_double-well/2D")`
- 可执行文件名统一为 `CH_2D`（原 develop 版本的 `spinodal_decomposition` 不受影响）
