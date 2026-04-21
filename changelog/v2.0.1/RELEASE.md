# PhiX v2.0.1 发布说明

**发布日期**：2026-04-21  
**标签**：`v2.0.1`  
**作者**：Wang Hanwei &lt;wanghanweibnds2015@gmail.com&gt;

---

## 概述

v2.0.1 在 v2.0.0 框架基础上新增了两个完整的应用求解器，并对项目结构进行了整理。

---

## 新增应用

### 1. Cahn-Hilliard + Allen-Cahn 双阱耦合求解器（2D）

路径：`applications/solvers/Cahn-Hillard+Allen-Cahn_double-well/2D/`

用于模拟 Ostwald 熟化过程的 CH+AC 耦合体系：

$$
\mu = 2\rho^2(c - c_a) + 2\rho^2(c_a - c_b)h(\eta) - \kappa_c\nabla^2 c
$$

$$
\frac{\partial c}{\partial t} = M\nabla^2\mu
$$

$$
\frac{\partial\eta}{\partial t} = -L\left[30\rho^2\eta^2(1-\eta)^2(2c-c_a-c_b)(c_a-c_b) + 2w\eta(1-\eta)(1-2\eta) - \kappa_\eta\nabla^2\eta\right]
$$

使用多步 Solver API（μ STEADY → c TRANSIENT → η TRANSIENT），完全配置文件驱动。

---

### 2. 玻璃形成能力（GFA）求解器（2D）

路径：`applications/solvers/glass_formation/2D/`

用于模拟 Fe-B 二元合金中的玻璃形成过程，耦合了 Cahn-Hilliard 成分场与两个 Allen-Cahn 序参量（晶体 φ、非晶 η）：

$$
\frac{\partial\phi}{\partial t} = -M_\phi\frac{\partial f}{\partial\phi} + M_\phi\epsilon^2\nabla^2\phi
$$

$$
\frac{\partial\eta}{\partial t} = -M_\eta\frac{\partial f}{\partial\eta} + M_\eta\beta^2\nabla^2\eta
$$

$$
\frac{\partial c}{\partial t} = \nabla\cdot\!\left(D(\phi,\eta)\,\nabla\mu\right)
$$

热力学采用 **CALPHAD** 方法处理 Fe-B 体系（液相、Fe₃B 固相、非晶相）。  
每时间步包含 11 个求解步骤（8 个 STEADY 辅助场 + 3 个 TRANSIENT 演化方程）。  
详细方程见 `doc/equations.md`。

---

## 其他变更

- `CMakeLists.txt`：将两个新应用纳入构建系统
- `applications/solvers/Cahn-Hillard_double-well/2D/README.md`：补充 CH 双阱求解器使用说明
- `develop/CH+AC/`：新增 CH+AC 开发工作区（配置文件与初始场）
- `develop/FeB_ex1/`：新增 Fe-B 合金算例（初始场生成脚本与配置文件）
- `.gitignore`：新增对编译产物（`CH_AC_2D`、`GFA_2D`）、Windows Zone.Identifier 元数据文件、`run.log` 及视频文件（`*.mp4`）的忽略规则

---

## 兼容性

本版本与 v2.0.0 **完全向后兼容**，未对任何公共 API 做出变更。
