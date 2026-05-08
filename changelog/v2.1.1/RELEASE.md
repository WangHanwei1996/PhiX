# PhiX v2.1.1 发布说明

**发布日期**：2026-05-08  
**标签**：`v2.1.1`  
**作者**：Wang Hanwei <wanghanweibnds2015@gmail.com>

---

## 概述

v2.1.1 为 `BCFactory` 补全了 **Dirichlet（固定值）边界条件的 JSON/配置驱动支持**。`FixedBC` 此前已有完整的 CPU/GPU 实现，但无法通过配置文件构造；本版本消除了这一缺口，同时保持对已有配置文件的完全向后兼容。

---

## 变动详情

### `BCFactory`：支持 `"Fixed"` Dirichlet 边界条件

**文件**：`include/boundary/BCFactory.h`、`src/boundary/BCFactory.cpp`

`buildBCs()` 现在支持三种 BC 类型：`"Periodic"`、`"NoFlux"`、`"Fixed"`。

每个边界条目可采用两种写法：

**字符串写法**（值默认为 `0.0`）：
```jsonc
"y_max": "Fixed"
```

**对象写法**（推荐，可指定任意值）：
```jsonc
"y_min": {"type": "Fixed", "value": 0.0},
"y_max": {"type": "Fixed", "value": 1.0}
```

两种写法可与 `"Periodic"`、`"NoFlux"` 混用。已有配置文件无需任何修改。

#### 实现细节

- 新增内部 helper `getBCType(const nlohmann::json&)`，统一处理字符串和对象两种入口形式。
- `addAxisBCs()` 参数由 `const std::string&` 改为 `const nlohmann::json&`，在同一 lambda 内完成类型分发。
- `buildBCs()` 各轴读取直接传递 JSON 值，不再预先 `.get<std::string>()`。

---

## 升级指南

无破坏性变更，直接升级即可。如需在配置中使用 Dirichlet 边界，按上述对象写法添加相应字段。
