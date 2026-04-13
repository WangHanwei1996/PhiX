# Field

`Field` 是 PhiX 的双精度标量场，定义在结构化 `Mesh` 上。每个方向两侧各填充 `ghost` 层虚格（halo），用于边界条件和差分模板的访问。CPU 和 GPU 各持有独立的存储缓冲区，支持懒分配和按需同步。

---

## 内存布局

```
物理格点索引 : i ∈ [0, mesh.n[ax])
虚格索引     : i ∈ [-ghost, 0)  和  [mesh.n[ax], mesh.n[ax]+ghost)

storedDims[ax] = mesh.n[ax] + 2*ghost
storedSize     = storedDims[0] × storedDims[1] × storedDims[2]
```

行主序线性索引（x 最快，z 最慢）：

$$\text{idx}(i,j,k) = (i+g) + s_x \cdot \bigl((j+g) + s_y \cdot (k+g)\bigr)$$

其中 $g$ = `ghost`，$s_x$ = `storedDims[0]`，$s_y$ = `storedDims[1]`。

---

## 数据成员

| 成员 | 类型 | 说明 |
|------|------|------|
| `name` | `std::string` | 场的名称（写入文件时用） |
| `mesh` | `const Mesh&` | 所属网格（引用，不持有） |
| `ghost` | `int` | 每侧虚格层数 |
| `storedDims[3]` | `int[3]` | 含虚格的存储维度 |
| `storedSize` | `std::size_t` | 总存储元素数（含虚格） |
| `curr` | `std::vector<double>` | CPU 当前时间层 |
| `prev` | `std::vector<double>` | CPU 前一时间层 |
| `d_curr` | `double*` | GPU 当前时间层（懒分配，初始为 `nullptr`） |
| `d_prev` | `double*` | GPU 前一时间层（懒分配，初始为 `nullptr`） |

---

## 构造与生命周期

```cpp
Field phi(mesh, "phi", /*ghost=*/1);
```

- CPU 缓冲区 `curr` 和 `prev` 在构造时分配并初始化为 0
- GPU 缓冲区**不**自动分配，需显式调用 `allocDevice()`
- `Field` **不可复制**（持有 GPU 内存所有权），但**可移动**
- 析构时自动调用 `freeDevice()`

---

## 索引方法

```cpp
int index(int i, int j, int k) const;   // 3D
int index(int i, int j)        const;   // 2D（k=0）
int index(int i)               const;   // 1D（j=k=0）
```

接受**物理区**和**虚格区**的负/越界索引，例如 `index(-1, j)` 访问 x 低侧第一层虚格。

```cpp
// 写入初始条件示例
for (int j = 0; j < mesh.n[1]; ++j)
for (int i = 0; i < mesh.n[0]; ++i)
    phi.curr[phi.index(i, j)] = std::sin(mesh.coord(0, i));
```

---

## 初始化

```cpp
phi.fill(0.0);          // curr 和 prev 全部置零
phi.fillCurr(1.0);      // 只置 curr
phi.fillPrev(0.0);      // 只置 prev
```

---

## GPU 管理

### 分配与释放

```cpp
phi.allocDevice();      // cudaMalloc d_curr 和 d_prev，并清零
phi.freeDevice();       // cudaFree，指针置 nullptr
phi.deviceAllocated();  // 返回 d_curr != nullptr
```

### CPU → GPU 上传

```cpp
phi.uploadCurrToDevice();   // curr  → d_curr
phi.uploadPrevToDevice();   // prev  → d_prev
phi.uploadAllToDevice();    // 两者都上传
```

### GPU → CPU 下载

```cpp
phi.downloadCurrFromDevice();   // d_curr → curr
phi.downloadPrevFromDevice();   // d_prev → prev
phi.downloadAllFromDevice();    // 两者都下载
```

> 上传/下载前必须先调用 `allocDevice()`，否则抛出 `std::runtime_error`。

---

## 时间推进

```cpp
phi.advanceTimeLevelCPU();   // CPU: prev ← curr（std::copy）
phi.advanceTimeLevelGPU();   // GPU: d_prev ← d_curr（cudaMemcpy D→D）
```

`Solver::advance()` 在每步结束时自动调用 GPU 路径，无需手动调用。

---

## IO

### 写文件

```cpp
phi.write("output/phi_1000.field");
```

文件格式：**文本头 + 原始二进制**（仅物理格点，不含虚格）

```
# PhiX Field
name    phi
nx 256  ny 256  nz 1
ghost   1
---
<nx*ny*nz 个 double，行主序，x 最快>
```

### 读文件

```cpp
Field phi = Field::readFromFile(mesh, "output/phi_1000.field", /*ghost=*/1);
```

- 读入物理格点到 `curr`，`prev` 不变
- 文件头的 `nx/ny/nz` 与 `mesh` 不一致时抛出 `std::runtime_error`

### 用 Python 读取

```python
import numpy as np

def read_field(path, nx, ny, nz):
    with open(path, 'rb') as f:
        for line in f:
            if line.strip() == b'---':
                break
        data = np.frombuffer(f.read(), dtype=np.float64)
    return data.reshape((nz, ny, nx))   # z 最慢，x 最快
```

### 打印摘要

```cpp
phi.print();
```

输出名称、存储维度、GPU 分配状态，以及 `curr` 物理格点的 min / max / mean。

---

## 典型使用流程

```cpp
// 1. 构造
Field phi(mesh, "phi", 1);

// 2. 设置初始条件（CPU）
for (...) phi.curr[phi.index(i, j)] = ...;

// 3. 分配 GPU 并上传
phi.allocDevice();
phi.uploadAllToDevice();

// 4. 求解（Solver 内部操作 d_curr / d_prev）
solver.run(nSteps, outEvery, [&](const Solver& s) {
    phi.downloadCurrFromDevice();   // 需要数据时再下载
    phi.write("output/phi_" + std::to_string(s.step) + ".field");
});
```

---

## 文件位置

| 文件 | 说明 |
|------|------|
| `include/field/Field.h` | 类声明、inline 索引方法 |
| `src/field/Field.cu` | 构造、GPU 管理、IO 实现（需要 nvcc） |
