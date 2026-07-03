# v2.13.0 — 精度策略：`PhiX::Real` 参数化（DOUBLE / FLOAT）

## 摘要

核心数值栈从硬编码 `double` 改为 `PhiX::Real`（`include/core/Real.h`），
配置期切换：

```bash
cmake .. -DPHIX_PRECISION=DOUBLE     # 默认，Real = double，行为与既往完全一致
cmake .. -DPHIX_PRECISION=FLOAT     # Real = float
```

动机：消费级 GPU 的 FP64 吞吐仅为 FP32 的 1/32–1/64（本机 RTX 5080 为
1/64），且 double 场占双倍带宽——对带宽受限的 stencil 代码，FLOAT 模式
是免费的大幅提速（见下方基准）。

---

## 精度边界设计

| 层 | 精度 | 说明 |
|----|------|------|
| 场存储 / 全部 GPU kernel | **Real** | ScalarField/VectorField/FaceField、BC、scheme（CD2/CD4/Iso9 内部已 Real 字面量化，无隐式 FP64 提升）、算子、Equation/Solver/EquationSystem kernel、FusedTerm eval 链、Gibbs 投影、噪声 |
| host 控制标量（dt、time、系数、网格几何） | double | kernel 启动边界处收窄为 Real |
| **归约（Reduce.h）** | **始终 double 累加/返回** | float 场求和/L2 不丢精度 |
| **磁盘格式（.field 二进制）** | **始终 double** | FieldIO 经 `vector<double>` 中转缓冲读写，两种构建的文件互换兼容 |
| FreeEnergyTable 查表 | double | 查表受访存限制；如需 float 表后续再做 |
| 噪声（curand） | double 生成 → Real 存储 | 分布质量优先 |

FLOAT 构建下：应用求解器与 1e-12 容差的严格测试套件**不参编**
（host 侧 double 代码 + double 标定容差），由 `test/floatSmoke` 以
float 量级容差覆盖核心路径；benchmark 两种模式均参编。

---

## 基准对比（RTX 5080，2D，同机同参）

```
                    DOUBLE (v2.12.0 基线)      FLOAT (本版)        提速
lap CD2  N=1024     22173 Mcells/s             35400 Mcells/s      1.6×
lap CD4  N=1024     10849 Mcells/s             33230 Mcells/s      3.1×
euler 步 N=1024      7979 Mcells/s              9636 Mcells/s      1.2×
```

- CD4 提速最大（double 下算力占比高，float 解除 FP64 惩罚）；
- 完整 Euler 步提速有限——瓶颈在框架开销（launch/同步/D2D 拷贝），
  与 v2.12.0 基线结论一致，是下一阶段调度层优化的对象。

---

## 主要变更文件

- 新增 `include/core/Real.h`；`CMakeLists.txt` 增加 `PHIX_PRECISION`
  选项与 FLOAT 模式的目标门控。
- 场/边界/算子/方程/求解器全链路 `double*` → `Real*`：
  `ScalarField/VectorField/FaceField/GibbsSimplex/NoiseGenerator/Reduce`、
  `Boundary.cu`、`scheme/*`、`Laplacian/Gradient/Advection/FaceOps`、
  `Term.h`（`ScratchPool`、`TermLauncher`）、`TermPW/FieldOps/FacePW.inl`、
  `FusedTerm.h`（`StencilParams`、全部 eval 节点）、
  `Equation/EvalPlan/Expr/EquationSystem/VectorEquation`、
  `Solver/VectorSolver`。
- 新增 `test/floatSmoke/test_float_smoke.cu`（float_smoke）。
- `bench_stencil` 报告实际 Real 类型与按 `sizeof(Real)` 计的带宽。

## 编写约定（重要）

设备端热路径代码请用 `Real` 形参与 `Real(…)` 字面量——一个裸的 double
字面量（如 `2.0 * x`）会把整个表达式静默提升为 FP64。`PHIX_FN` 用户
函子建议以 `Real` 为参数类型（double 形参可编译，但 float 构建下该函子
内部走 FP64）。

---

## 测试

- DOUBLE：全量 ctest **22/22 通过**（默认模式行为零变化）；
- FLOAT（`build-float/`）：库 + 全部测试目标编译通过；`float_smoke`
  通过（Real 尺寸断言、归约对照 CPU、1D 周期扩散对照解析解、
  自适应 dt + NaN 哨兵）；`bench_stencil` 通过。
