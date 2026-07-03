# v2.17.0 — 半隐式/隐式时间积分（`SemiImplicitSolver`，IMEX）

## 摘要

在 v2.16.0 线性求解器层之上落地一阶 IMEX 积分器，**解除相场计算的
显式稳定性天花板**（框架评价文档中收益排序第一的算法层升级）：

```
dφ/dt = L·φ + N(φ)
(I − dt·L)·φⁿ⁺¹ = φⁿ + dt·N(φⁿ)     （L 后向 Euler，N 前向 Euler）
```

- **Allen-Cahn / 扩散**：L = D∇²（`LaplacianOp`）→ dt 不再 ∝ dx²；
- **Cahn-Hilliard**：L = −Mκ∇⁴（`BiharmonicOp`）+ 显式 N = M∇²f'(c)
  （经典线性分裂）→ dt 不再 ∝ dx⁴。

σ = dt 在算子外，逐步改 dt 零成本（与 v2.10.0 自适应控制器天然兼容）。

**实测**（RTX 5080，模块测试数字）：

- 全隐式扩散在 **100× 显式极限** 步长下，与离散 BE 精确解逐点一致到
  **9.5e-17**（机器精度；CG 每步 6 次迭代）；同 dt 下前向 Euler 10 步
  内爆掉（对照演示）；
- IMEX 反应-扩散耦合与离散精确因子一致到 3.7e-22；
- BE 时间收敛阶实测 0.985 / 0.993；
- CH 线性分裂在 **50× 显式 ∇⁴ 极限** 下 3355 步：稳定、旋节分解正常
  发育（max|c|=0.887）、质量漂移 **5.5e-15**、CG 每步约 24 次迭代。

---

## 核心变更

### 新增文件

| 文件 | 说明 |
|------|------|
| `include/solver/SemiImplicitSolver.h` | `SemiImplicitSolver` + `SemiImplicitCGOptions` |
| `src/solver/SemiImplicitSolver.cu` | advance/run 实现（已注册 `phix` 库） |
| `test/moduleTest/solver/test_semiimplicit.cu` | 模块测试 `module_semiimplicit` |

### API

```cpp
// —— Allen-Cahn 型：刚性扩散隐式，非线性驱动力显式 ——
Equation eqPhi(phi, "phi");
eqPhi.setRHS(pw(phi, PHIX_FN (Real p) {        // 只放显式部分！
    return -M * W * dg(p) + M * dh(p) * dG;    // 刚性 κ∇² 项交给 L
}));
LaplacianOp L(M * kappa, {&bc});
SemiImplicitSolver semi(eqPhi, {&bc}, L, dt);
semi.run(nSteps);
semi.lastSolve();          // 上一步 CG 迭代数/残差诊断
semi.dt = newDt;           // 逐步可变，算子无需重建

// —— Cahn-Hilliard 型：∇⁴ 隐式，M∇²f'(c) 显式（muE 辅助场模式）——
Equation eqMu(c, "mu");  eqMu.setRHS(pw(c, PHIX_FN (Real v) { return v*v*v - v; }));
Equation eqC (c, "c");   eqC.setRHS(lap(muE, M));
BiharmonicOp Lc(M * kappa, {&bcC}, {&bcLap});
SemiImplicitSolver semi(eqC, {&bcC}, Lc, dt);
loop { eqMu.computeRHS(muE); bcMu.applyOnGPU(muE); semi.advance(); }
```

- 显式方程未 `setRHS` 时按 N ≡ 0 处理（纯隐式线性步）；
- BC 需齐次（继承 v2.16.0 线性层限制：Periodic / NoFlux / FixedBC(0)）；
- 精度为时间一阶（BE/FE 对）；超大 dt 的 CH 建议按惯例在 L 中加线性
  稳定化项（Eyre 分裂），接口天然支持（任何 `LinearOperator` 皆可）。

---

## 测试

`module_semiimplicit`（已注册 ctest），方法论亮点：前两项用**离散**
特征值构造逐步放大因子，参考解精确到机器精度——积分器全链路
（BC→显式 RHS→b 组装→CG→时间层）任何一处出错都无处遁形：

1. 全隐式扩散 @100×dt_explicit vs 离散 BE 精确解（<1e-8 断言，实测
   9.5e-17）+ 前向 Euler 同 dt 爆掉对照；
2. IMEX 耦合 vs 离散精确因子 (1−λdt)/(1+dt|λ_h|)；
3. BE 时间阶 ≈1（±0.15）；
4. CH 线性分裂 @50×：无 NaN、max|c|∈[0.8,1.3]（分解发育）、
   质量守恒 <1e-9。

全量 ctest：DOUBLE **26/26**，FLOAT 构建编译+smoke 通过。

---

## 兼容性

纯新增。既有显式 `Solver`/`EquationSystem` 不受影响；半隐式与显式
求解器可在同一算例中混用（如 φ 显式 + c 半隐式的算子分裂）。
