# v2.6.7 — 面心逐点算子 `facePW` / `facePWGPU`

## 摘要

补全有限体积面算子链最后一环，新增 `facePW` / `facePWGPU` 系列模板函数，
支持在 `FaceField` 上做任意非线性逐元素变换（1/2/3 场重载，CPU + GPU 双路径）。

面算子链现已闭合：

```
cell  ──interp/faceGrad──▶  face  ──facePW──▶  face  ──divFace──▶  cell
```

---

## 核心变更

### 新增文件

| 文件 | 说明 |
|------|------|
| `include/operators/FacePW.h` | 函数声明 + 末尾 `#include "FacePW.inl"` |
| `include/operators/FacePW.inl` | 模板核心：GPU 核函数 + CPU/GPU 实现（需 nvcc） |

### 修改文件

| 文件 | 说明 |
|------|------|
| `include/operators/FaceOps.h` | 末尾追加 `#include "operators/FacePW.h"`，单头便利引入 |

### API

```cpp
// CPU 路径（host functor）
template<typename Fn> void facePW(FaceField& out, const FaceField& a, Fn fn);
template<typename Fn> void facePW(FaceField& out, const FaceField& a,
                                  const FaceField& b, Fn fn);
template<typename Fn> void facePW(FaceField& out,
                                  const FaceField& a, const FaceField& b,
                                  const FaceField& c, Fn fn);

// GPU 路径（__host__ __device__ lambda，推荐配合 PHIX_FN 宏）
template<typename Fn> void facePWGPU(FaceField& out, const FaceField& a, Fn fn);
template<typename Fn> void facePWGPU(FaceField& out, const FaceField& a,
                                     const FaceField& b, Fn fn);
template<typename Fn> void facePWGPU(FaceField& out,
                                     const FaceField& a, const FaceField& b,
                                     const FaceField& c, Fn fn);
```

**使用示例**（各向异性枝晶 x 面通量）：

```cpp
facePWGPU(jx, phi_x_xf, phi_y_xf,
    PHIX_FN (double px, double py) {
        double theta    = atan2(py, px);
        double a        = 1.0 + eps * cos(m * (theta - theta0));
        double sin_term = eps * m * sin(m * (theta - theta0));
        return W0sq * a * (a * px + sin_term * py);
    });
```

### 设计要点

- **纯模板头文件**：无需新增 `.cu` 源文件，不改动 `add_library`，零编译时间开销（未使用则不实例化）。
- **索引与 `FaceOps.cu` 一致**：内联 `facepw_idx` 与 `FaceOps.cu` 的 `face_idx` 等价，法向轴无 ghost 偏移，切向轴有 ghost 偏移。
- **运行期安全校验**：比对 `normalAxis` 与 `storedDims`，参数不一致立即抛出 `std::invalid_argument`。
- **对称于 `TermPW.inl`**：命名约定、宏 `PHIX_FN`、1/2/3 场重载结构全部对齐，降低学习成本。

---

## 测试

新增 `test/moduleTest/operators/test_face_pw.cu`（8 个测试）：

| # | 名称 | 覆盖路径 |
|---|------|---------|
| 1 | cpu 1-field 1D | CPU 路径，1 场，`out = 2*a` |
| 2 | cpu 2-field 1D | CPU 路径，2 场，`out = a + b` |
| 3 | cpu 3-field 1D | CPU 路径，3 场，`out = a*b + c` |
| 4 | gpu 1-field 1D | GPU 路径，1 场，CPU/GPU 结果一致 |
| 5 | gpu 2-field 1D | GPU 路径，2 场，CPU/GPU 结果一致 |
| 6 | gpu 3-field 1D | GPU 路径，3 场，CPU/GPU 结果一致 |
| 7 | cpu 2D y-face 2-field | axis=1 索引映射验证 |
| 8 | anisotropic flux x-face | CPU+GPU，枝晶各向异性通量，解析值对比（ε=0.05，m=4） |

全量测试：**13/13 通过**（含所有既有模块测试，零回归）。

---
