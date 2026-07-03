# v2.10.0 — 自适应时间步（rate-limited Euler）+ NaN 哨兵

## 摘要

新增 `AdaptiveDt` 控制器：显式 Euler 下每步的场变化恰为 |Δφ| = dt·|RHS|，
因此 RHS 算完后即可闭式解出"保证每格变化 ≤ tol"的最大 dt：

```
dt* = safety · tol / max|RHS|        （max|RHS| 来自 v2.9.0 设备端归约）
dt  = clamp(min(dt*, dt·grow), dtMin, dtMax)
```

**无需拒绝/重算循环**——RHS 每步只算一次，更新直接用调整后的 dt。
缩小即时生效（安全方向），增长受 `grow` 因子限制防振荡。
配套 NaN/Inf 哨兵：每 `nanCheckEvery` 步在设备端扫描未知场，
发散立即抛 `std::runtime_error`（带步数/时刻），不再输出垃圾数据。

对刚度随时间衰减的相场问题（形核初期驱动力大 → 后期粗化平缓），
dt 自动从稳定性约束值增长到 dtMax，等效于免费提速。

---

## 核心变更

### 新增文件

| 文件 | 说明 |
|------|------|
| `include/solver/AdaptiveDt.h` | 控制器（header-only）：选项 + `propose()` + `validate()` |
| `test/moduleTest/equation/test_adaptive_dt.cu` | 模块测试 `module_adaptive_dt` |

### API

```cpp
AdaptiveDt opts;
opts.tol   = 1e-3;    // 每步允许的最大 |Δφ|（必填）
opts.dtMin = 1e-12;   // dt 硬下界（必填）
opts.dtMax = 1e-6;    // dt 硬上界（必填）
opts.grow  = 1.2;     // 每步最大增长因子（默认 1.2）
opts.safety = 0.9;    // 实际瞄准 tol 的比例（默认 0.9）
opts.nanCheckEvery = 100;   // 每 100 步查一次 NaN/Inf（0 = 关）

solver.enableAdaptiveDt(opts);   // Solver 单方程 Euler
sys.enableAdaptiveDt(opts);      // EquationSystem Euler（全部方程取 max|RHS|）

// 每步之后：solver.dt / sys.dt = 本步实际使用的 dt
//           adaptiveDt().lastMaxRate = 本步控制性 max|RHS|
```

### 接入点

| 文件 | 说明 |
|------|------|
| `include/solver/Solver.h` / `src/solver/Solver.cu` | `enableAdaptiveDt`；Euler 分支在 `computeRHS` 后、更新前调 `reduce::fieldMaxAbs(rhs)` 调整 dt；advance 末尾 NaN 哨兵；CPU 回退路径同逻辑（host 端求 max） |
| `include/equation/EquationSystem.h` / `src/equation/EquationSystem.cu` | 同上；控制速率取 **所有方程 RHS 的最大值**（最刚的方程决定 dt） |

限制（明确抛异常）：RK4 不支持（需嵌入式误差估计）；Solver 多步模式
（operator splitting）不支持（前面的方程更新后才知道后面的 RHS）。

---

## 测试

`module_adaptive_dt`（已注册 ctest）。线性衰减问题下 Euler 严格满足
φ_{n+1} = φ_n·(1−λ·dt_n)，逐项记录自适应 dt 即得到**机器精度的全轨迹参考**：

1. Solver 单方程（λ=10）：dt 从 1e-6 增长、始终在 [dtMin,dtMax] 内、
   每步 dt·max|RHS| ≤ tol、300 步轨迹与记录 dt 乘积一致（rel 1e-11）、
   time == Σdt；
2. EquationSystem 双方程（λ=5 vs 50）：控制速率逐步等于两方程理论
   max|RHS|（rel 1e-11，证明"最刚方程控盘"），两场轨迹各自吻合；
3. NaN 哨兵：爆炸 RHS（1e150·φ）数步内触发 `std::runtime_error`；
4. RK4 上 enable、非法选项（tol=0）抛 `std::invalid_argument`。

全量 ctest 16/16 通过（无回归）。

---

## 兼容性

默认 `enabled == false`，未调用 `enableAdaptiveDt` 的既有代码零行为变化、
零开销。依赖 v2.9.0 的 `PhiX::reduce`。
