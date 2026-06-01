# PhiX v2.6.0 发布说明

**发布日期**：2026-06-01  
**标签**：`v2.6.0`  
**作者**：Wang Hanwei <wanghanweibnds2015@gmail.com>

---

## 概述

v2.6.0 新增两项内容：

1. **`makePhi` 开发工具**：在任意目录下自动完成 CMakeLists.txt 生成、构建系统注册和编译，一条命令即可将 `.cu` 源文件变成可全局调用的可执行求解器；`makePhi clean` 完整反向撤销所有操作。

2. **`diffusion_1D` 示例算例**：1-D 瞬态扩散，并附带 `diffusion_compare` 对比算例，直观展示 staggered 面心通量格式与 collocated 二步中心差分格式在棋盘模态（Nyquist 频率）上的行为差异。

---

## 变动详情

### 新增文件

| 文件 | 说明 |
|---|---|
| `applications/tools/_makePhi` | makePhi Python 内部实现 |
| `test/sampleTest/diffusion_1D/diffusion_1D.cu` | 1-D 扩散示例（面心通量形式） |
| `test/sampleTest/diffusion_1D/diffusion_compare.cu` | staggered vs. collocated 对比求解器 |
| `test/sampleTest/diffusion_1D/case/settings/settings.jsonc` | diffusion_1D 算例配置 |
| `test/sampleTest/diffusion_1D/case_compare/settings/settings.jsonc` | diffusion_compare 算例配置 |
| `changelog/v2.6.0/RELEASE.md` | 本文件 |

### 修改文件

| 文件 | 修改 |
|---|---|
| `etc/bashrc` | PATH 扫描范围从 `applications/` 扩展至整个项目（排除 `build/`）；新增 `makePhi` shell 函数，编译完自动 re-source |
| `test/sampleTest/CMakeLists.txt` | 添加 `diffusion_1D` 子目录 |
| `test/sampleTest/diffusion_1D/CMakeLists.txt` | 添加 `diffusion_compare` target |
| `CMakeLists.txt` | 移除误入的 `add_subdirectory(test/sampleTest/diffusion_1D)` |

---

## `makePhi` 工具说明

### 用法

```bash
# 在含 .cu 文件的目录下运行
makePhi           # 生成 + 注册 + 编译
makePhi clean     # 删除可执行文件 + 删除 CMakeLists.txt + 注销注册
```

> `makePhi` 是 `etc/bashrc` 中定义的 shell 函数，需先 `source etc/bashrc`。

### 工作流程（generate）

1. 扫描 CWD 下所有 `.cu` 文件
2. 生成带 `# [makePhi generated]` 标记的 `CMakeLists.txt`，所有 target 输出到 CWD
3. 检测是否有祖先目录已在根 `CMakeLists.txt` 中注册（防止 CMake 二进制目录冲突）；若无则追加 `add_subdirectory`
4. 执行 `cmake <root>`（在 `build/`）让新条目生效
5. 执行 `make -j<N> <targets>` 编译
6. 自动 re-source `etc/bashrc`，新可执行文件立即进入 PATH

### 工作流程（clean）

1. 从生成的 `CMakeLists.txt` 读取所有 target 名
2. 递归搜索 CWD 树，删除所有同名可执行文件（含子目录如 `case/`）
3. 删除 `CMakeLists.txt`
4. 从根 `CMakeLists.txt` 移除 `add_subdirectory` 行
5. 静默重跑 `cmake` 同步构建系统

### 安全机制

- 拒绝覆盖非 makePhi 生成的 `CMakeLists.txt`（需手动删除）
- 祖先路径检测：若 `test/sampleTest` 已注册，在 `test/sampleTest/diffusion_1D` 下运行时跳过根注册，直接编译

---

## `diffusion_1D` 示例说明

### 物理问题

$$\frac{\partial u}{\partial t} = D \frac{\partial^2 u}{\partial x^2}$$

初始条件：$u_0(x_i) = \sin(2\pi x_i / L) + 0.25\,(-1)^i$（低频正弦 + Nyquist 棋盘模态）

### `diffusion_1D`（staggered 面心通量）

两步离散：

$$
\text{flux}_{i+\frac{1}{2}} = \frac{u_{i+1} - u_i}{\Delta x}, \qquad
\text{rhs}_i = \frac{\text{flux}_{i+\frac{1}{2}} - \text{flux}_{i-\frac{1}{2}}}{\Delta x}
$$

等价于标准三点 Laplacian，棋盘模态特征值为 $4/\Delta x^2$，被正常衰减。

### `diffusion_compare`（staggered vs. collocated 对比）

| 格式 | 离散 | 棋盘模态特征值 | 结果 |
|---|---|---|---|
| **Staggered** | `faceGrad + divFace` → $(u_{i+1}-2u_i+u_{i-1})/\Delta x^2$ | $4/\Delta x^2$ | 正常衰减 ✓ |
| **Collocated** | `grad(u)` → `grad(g)` → $(u_{i+2}-2u_i+u_{i-2})/(4\Delta x^2)$ | $0$ | 零空间，不衰减 ✗ |

数值验证（$t=20$，$D=0.5$，$\Delta x=1$）：staggered 棋盘幅值 $0.25 \to 0.024$，collocated 保持 $0.25$。

---

## 升级说明

无破坏性变更，直接拉取即可。若要使用 `makePhi`：

```bash
source $PHIX_DIR/etc/bashrc
```

建议将该行加入 `~/.bashrc`。
