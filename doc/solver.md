# Solver（求解器模块）

`Solver` 驱动单个方程的显式时间推进，管理边界条件施加、RHS 计算和时间积分方案，并持有所有 GPU scratch field。

---

## 时间积分方案

```cpp
enum class TimeScheme {
    EULER,   // 前向 Euler（一阶）
    RK4      // 经典四阶 Runge-Kutta
};
```

### Forward Euler

$$\varphi^{n+1} = \varphi^n + \Delta t \cdot f(\varphi^n)$$

- 每步 1 次 RHS 评估
- 稳定性条件严格，适合快速验证

### 经典 RK4

$$k_1 = f(\varphi^n)$$
$$k_2 = f\!\left(\varphi^n + \tfrac{\Delta t}{2} k_1\right)$$
$$k_3 = f\!\left(\varphi^n + \tfrac{\Delta t}{2} k_2\right)$$
$$k_4 = f\!\left(\varphi^n + \Delta t\, k_3\right)$$
$$\varphi^{n+1} = \varphi^n + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

- 每步 4 次 RHS 评估
- 4 阶精度，允许更大时间步长
- 每个中间阶段均施加边界条件

---

## 构造

```cpp
Solver solver(eq, {&bc_x, &bc_y}, dt, TimeScheme::RK4);
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `equation` | `Equation&` | 方程引用（非持有） |
| `bcs` | `std::vector<BoundaryCondition*>` | 边界条件列表（非持有） |
| `dt` | `double` | 时间步长 |
| `scheme` | `TimeScheme` | 积分方案，默认 `EULER` |

构造时自动为所有 scratch field（`rhs_`, `k1_`…`k4_`, `phi_tmp_`）分配 GPU 内存。要求 `equation.unknown` 已调用 `allocDevice()`，否则抛出 `std::runtime_error`。

`Solver` **不可复制**（内部持有 GPU 内存）。

---

## 公开成员

| 成员 | 类型 | 说明 |
|------|------|------|
| `dt` | `double` | 时间步长（可在步间修改） |
| `scheme` | `TimeScheme` | 积分方案 |
| `step` | `int` | 当前步数计数器（`advance()` 递增） |
| `time` | `double` | 当前模拟时间（`time += dt`） |

---

## 单步推进

```cpp
solver.advance();       // GPU 路径（推荐）
solver.advanceCPU();    // CPU fallback（调试用）
```

### `advance()` 执行流程（GPU）

**Euler 方案：**
```
applyBCsGPU()          ← 更新未知场 d_curr 的虚格
computeRHS(rhs_)       ← 在 rhs_.d_curr 上累加所有项
kernel_axpy: phi += dt * rhs   ← GPU kernel
advanceTimeLevelGPU()  ← d_prev ← d_curr
step++; time += dt
```

**RK4 方案：**
```
k1 ← f(phi)                           施加 BC
k2 ← f(phi + dt/2 * k1)              施加 BC（在 phi_tmp 上）
k3 ← f(phi + dt/2 * k2)              施加 BC（在 phi_tmp 上）
k4 ← f(phi + dt   * k3)              施加 BC（在 phi_tmp 上）
phi += (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
advanceTimeLevelGPU()
step++; time += dt
```

> **RK4 指针交换技巧**：中间阶段的 RHS 评估通过临时交换 `phi.d_curr` 与 `phi_tmp_.d_curr` 实现，无需修改 `Equation` 中的 `Term`。

---

## 多步运行

```cpp
solver.run(nSteps, callbackEvery, callback);
```

| 参数 | 说明 |
|------|------|
| `nSteps` | 总步数 |
| `callbackEvery` | 每隔多少步触发一次回调（0 = 不触发） |
| `callback` | `std::function<void(const Solver&)>`，在步结束后调用 |

回调触发条件：`callback && callbackEvery > 0 && (step % callbackEvery == 0)`

```cpp
solver.run(10000, 500, [&](const Solver& s) {
    phi.downloadCurrFromDevice();
    phi.write("output/phi_" + std::to_string(s.step) + ".field");
    std::cout << "step=" << s.step << "  t=" << s.time << "\n";
});
```

---

## scratch field 一览

| 字段 | 用途 |
|------|------|
| `rhs_` | Euler / RK4 各阶段 RHS 累加目标 |
| `k1_`…`k4_` | RK4 四个阶段斜率（仅 RK4 方案使用） |
| `phi_tmp_` | RK4 中间阶段的临时场（$\varphi + \alpha k_i$） |

它们与 `equation_.unknown` 具有相同的 `Mesh` 和 `ghost`，构造时自动分配 GPU 内存。

---

## 方案对比

| | Euler | RK4 |
|-|-------|-----|
| 精度阶 | 1 | 4 |
| 每步 RHS 评估次数 | 1 | 4 |
| 稳定区间（扩散方程） | $\Delta t \leq \frac{\Delta x^2}{2D}$ | 约大 4× |
| 内存开销（额外 GPU 场） | 1 | 5 |
| 适用场景 | 快速验证、粗网格 | 精确模拟、生产运算 |

---

## 典型使用流程

```cpp
// 1. 构建方程和边界条件
Equation eq(phi, "AllenCahn");
eq.setRHS(M * lap(phi) + M * pw(phi, [] __host__ __device__ (double p) {
    return p - p*p*p;
}));

PeriodicBC bc_x(Axis::X);
PeriodicBC bc_y(Axis::Y);

// 2. 构建 Solver（phi 必须已 allocDevice()）
Solver solver(eq, {&bc_x, &bc_y}, 0.01, TimeScheme::RK4);

// 3. 运行
solver.run(10000, 500, [&](const Solver& s) {
    phi.downloadCurrFromDevice();
    phi.write("output/phi_" + std::to_string(s.step) + ".field");
});

// 或者手动逐步控制
for (int i = 0; i < 10000; ++i) {
    solver.advance();
    if (solver.step % 500 == 0) {
        phi.downloadCurrFromDevice();
        phi.write(...);
    }
}
```

---

## 文件位置

| 文件 | 说明 |
|------|------|
| `include/solver/Solver.h` | 类声明、`TimeScheme` 枚举 |
| `src/solver/Solver.cu` | 实现：GPU kernels、Euler/RK4 路径、`run()`（需 nvcc） |
