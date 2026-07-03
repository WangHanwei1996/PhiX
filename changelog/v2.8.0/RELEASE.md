# v2.8.0 — GFA 二元/Fe-B 求解器、GFA_4ph 面通量迁移、IO 与材料模块增强

## 摘要

框架快照版本：合入近期在玻璃形成（GFA）方向的全部求解器与配套改动。

- 新增 **GFA_binary** 求解器（Cu-Zr 二元合金分阶段模型，stage 1–6 建模文档齐全）
  与 **GFA_FeB** 求解器（Fe-B 标定方向：c 相关 f_S 抛物线驱动力 + 退化迁移率
  M_c(φ) 保守面通量）。
- **GFA_4ph** 完成梯度能 → 交错面通量（face-flux）迁移，改用 `EquationSystem`
  同步推进，加入 Gibbs 单纯形投影与 η 每步热噪声。
- **MPF_AC_DW** 迁移到成对双阱 + grad2 加权梯度能形式，适配新 `Patch` BC API。
- IO：VTS 坐标缩放 `coord_scale`（nm 级网格 ParaView 可视）、2D 薄板厚度修正、
  时间输出改科学计数法且不再污染全局 `std::cout` 状态。
- 材料：`FreeEnergyTable` 支持 nc==1 退化 c 轴（化学计量相，f 仅依赖 T）。
- 新增 `data/material_properties/`（Fe-B、Cu-Zr 自由能与迁移率表）与多篇文档
  （框架评价、界面厚度、无量纲化、溶质扩散率）；`CLAUDE.md` 入库。

---

## 核心变更

### 新增求解器

| 目录 | 说明 |
|------|------|
| `applications/solvers/GFA_binary/` | Cu-Zr 二元 GFA 分阶段模型（GFA_binary + GFA_evo 两个演化版本）；`doc/modeling_stage1..6.md` 记录建模路线；`test/tables/*.fetab` 自由能表 |
| `applications/solvers/GFA_FeB/` | Fe-B 玻璃形成求解器：液相/Fe2B 自由能（Poletti-Battezzati CALPHAD 2013），c 相关 f_S 抛物线 φ 驱动力，M_c(φ)=h·McS+(1−h)·McL 保守面通量退化迁移率 |

根 `CMakeLists.txt` 注册以上两个求解器，并恢复
`Cahn-Hillard+Allen-Cahn_double-well/2D`（重写为 `CH_AC_2D.cu`，旧
`CH+AC_double-well.cu` 移除）。

### GFA_4ph（glass_formation_4_phases/2D）

- 梯度能项从格点 Laplacian 改为交错面通量链
  `interp/faceGrad → facePW → divFace`（与 MPF_AC_DW 同构）。
- 四相 φ 用 `EquationSystem` 同一时间层同步推进；每步 `k_proj_simplex4`
  Gibbs 单纯形投影维持 Σφ=1（爆算根因修复）。
- η 方程加入每步高斯热噪声（`noise_mean/noise_std/noise_seed` 配置项，
  curand 每格独立状态）。

### IO

| 文件 | 说明 |
|------|------|
| `include/IO/FieldIO.h` / `src/IO/FieldIO.cpp` | `writeField(..., double coordScale=1.0)`：仅缩放 VTS 节点坐标（如 1e9 → nm）；2D 时第三维厚度取 dx，避免 ParaView 中薄板比例失衡 |
| `include/IO/OutputWriter.h` / `src/IO/OutputWriter.cpp` | 配置项 `output.coord_scale`（默认 1.0）；`t=` 输出改科学计数法，经局部 `ostringstream` 格式化，不再泄漏 `std::fixed` 等粘性标志到全局 `std::cout` |

### 材料

`FreeEnergyTable` 解析流：nc==1 时复制单行并把 c 范围扩展到 [0,1]，
双线性插值对任意 c 精确返回该 c 无关值 —— 支持化学计量相（如 Fe2B）表格。

### 数据与文档

- `data/material_properties/{Fe-B,Cu-Zr}/`：f_L/f_S/dfdc_L/d2fdc_L/M_c 表
  （CSV）及生成脚本。
- `doc/claude/framework_evaluation.md`：框架与成熟 FD 套件的差距评估、
  优化优先级、半隐式路线建议。
- `doc/interface_thickness/`、`doc/nondimensionalization/`、
  `doc/solute_diffusivity/`、`doc/time_step/`：建模笔记。
- `.gitignore`：显式列出新求解器二进制与散落日志/备份。

---

## 兼容性

- `IO::writeField` 新增参数带默认值，既有调用不受影响。
- `NoFluxBC(Axis, Side)` 旧构造在 MPF_AC_DW 中已替换为
  `NoFluxBC(mesh.facePatch(Axis, Side))`（新 Patch API）。
- 无库级 API 破坏性变更。
