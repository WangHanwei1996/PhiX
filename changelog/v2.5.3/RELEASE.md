# PhiX v2.5.3 发布说明

**发布日期**：2026-06-01  
**标签**：`v2.5.3`  
**作者**：Wang Hanwei <wanghanweibnds2015@gmail.com>

---

## 概述

v2.5.3 引入面心通量算符（face-centred flux operators），完成 FaceField 生态的功能闭环：

- `interp`：cell-centre → face-centre 线性插值
- `faceGrad`：面心梯度（直接有限差分，无需插值）
- `divFace`：从面心通量构造保守散度 `Term`，可直接用于 `Equation::setRHS`

---

## 变动详情

### 新增文件

| 文件 | 说明 |
|---|---|
| `include/operators/FaceOps.h` | 三个算符的声明（CPU + GPU 版本） |
| `src/operators/FaceOps.cu` | 实现（CPU 循环 + CUDA 核函数 + divFace Term 工厂） |
| `test/moduleTest/operators/test_face_ops.cu` | 6 个单元测试 |

### 修改文件

| 文件 | 修改 |
|---|---|
| `CMakeLists.txt` | 添加 `src/operators/FaceOps.cu` 到 phix 源文件 |
| `test/moduleTest/operators/CMakeLists.txt` | 添加 `module_face_ops` 测试目标 |

---

## API 说明

### `interp(cell, axis, face)` — CPU / `interpGPU(...)` — GPU

从 cell-centre 场插值到面心场。

```cpp
FaceField flux_x(mesh, 0, "flux_x");
interp(phi, 0, flux_x);   // 0 = x 轴
```

边界面（法向坐标为 0 或 n）夹持到最近 cell 值；内部面取相邻两格均值。

### `faceGrad(cell, axis, face)` — CPU / `faceGradGPU(...)` — GPU

面心梯度：`g_face[i] = (cell[i] - cell[i-1]) / dx`。

边界面（i=0）使用 ghost cell（需提前通过 BC 填充）。

```cpp
FaceField gx(mesh, 0, "gx");
faceGrad(phi, 0, gx);
```

### `divFace(fx, fy, fz?, coeff?)` — 返回 `Term`

从预先计算的面心通量构造保守有限体积散度，返回可加入 RHS 的 `Term`：

```
rhs += coeff * (ΔFx/Δx + ΔFy/Δy + ΔFz/Δz)
```

支持 1D/2D/3D，未使用的分量传 `nullptr`；提供便捷重载。

```cpp
// 2D CH 扩散: rhs = div(M * grad(mu))
interp(M, 0, Mx);   faceGrad(mu, 0, flux_x);   // flux_x *= Mx (需 FaceField 乘法，v2.5.4)
eq.setRHS(divFace(flux_x, flux_y));
```

---

## 索引约定（法向轴）

| centering | 法向轴 | 存储大小 | 法向索引偏移 |
|---|---|---|---|
| FACE_X | 0 | `(nx+1) × (ny+2g) × (nz+2g)` | 无 ghost |
| FACE_Y | 1 | `(nx+2g) × (ny+1) × (nz+2g)` | 无 ghost |
| FACE_Z | 2 | `(nx+2g) × (ny+2g) × (nz+1)` | 无 ghost |

`face.index(i, j, k)` 封装了上述偏移逻辑，在 CPU/GPU 核函数中均可使用等价内联函数 `face_idx()`。

---

## 测试覆盖

6 个 CPU 路径单元测试（`module_face_ops`）：

| 测试 | 验证点 |
|---|---|
| `test_interp_1d` | 内部面平均、边界面夹持 |
| `test_face_grad_1d` | 线性场梯度 = 1.0，含 ghost cell 路径 |
| `test_div_face_zero_1d` | 常数通量 → 散度 = 0 |
| `test_div_face_linear_1d` | 线性通量 → 散度 = 1.0 |
| `test_div_face_2d` | 2D 双轴散度 = 2.0 |
| `test_interp_y_2d` | y 方向插值含边界夹持 |

```
ctest --output-on-failure
100% tests passed, 0 tests failed out of 6
```
