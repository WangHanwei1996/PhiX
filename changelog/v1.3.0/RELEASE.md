# PhiX v1.3.0 发布说明

**发布日期**：2026-04-17  
**标签**：`v1.3.0`

---

## 概述

v1.3.0 新增 **`IO` 模块**，将配置文件的读取与解析统一封装，为后续 `Mesh`、`Field`、`Equation` 等模块从配置文件驱动做铺垫。

---

## 新增功能

### IO 模块（`include/IO/` / `src/IO/`）

新增 `PhiX::IO::ConfigFile` 类，统一处理 JSONC 配置文件的加载与访问。

**核心接口：**

```cpp
#include "IO/ConfigFile.h"

// 从命令行参数构造（推荐用于 main 入口）
// argv[1] 为路径；未提供则使用 defaultPath；失败则打印错误并退出
IO::ConfigFile cfg = IO::ConfigFile::fromArgs(argc, argv);

// 直接按路径构造（失败抛 std::runtime_error）
IO::ConfigFile cfg("path/to/settings.jsonc");

// 访问配置项
int    nx = cfg["mesh"]["nx"];
double dt = cfg["solver"]["dt"];

// 检查键是否存在
if (cfg.has("output")) { ... }

// 获取底层 json 对象（高级用途）
const auto& j = cfg.data();
```

**JSONC 注释支持：**

正确剥离 `//` 行注释，并能识别字符串内的 `//`（如路径值）不误截断：

```jsonc
{
    "mesh": {
        "nx": 200,      // x 方向格点数
        "path": "output/data"   // 不会被误截断
    }
}
```

**`fromArgs` 错误处理：**  
文件不存在或 JSON 语法错误时，自动打印含 `Usage` 提示的错误信息并以退出码 1 退出，主程序无需额外处理。

---

## 变更

### `CH_2D` 求解器简化

使用 `IO::ConfigFile::fromArgs` 后，原先的文件打开检查、注释剥离、异常捕获全部移入库内，main 函数中配置读取缩减为：

```cpp
IO::ConfigFile cfg = IO::ConfigFile::fromArgs(argc, argv);

const int    nx = cfg["mesh"]["nx"];
const double dx = cfg["mesh"]["dx"];
// ...
```

### 构建

`src/IO/ConfigFile.cpp` 已加入 `phix` 静态库，所有链接 `phix` 的目标自动可用，无需额外修改 CMakeLists.txt。
