# PhiX 表达式 DSL 求值层重构设计

> 状态：设计稿（待实现）
> 范围：表达式 DSL 的**求值层**（execution layer），不改表达层语法风格
> 约束：可以大改，不要求向后兼容

## 目标

1. **自动融合 pointwise 子树** —— 减少 kernel 启动与中间结果访存
2. **统一两套 DSL** —— 主 DSL 默认产出融合表达式，废弃手写 `FusedTerm` 语法
3. **去除 `lap(expr, bcs)` 的手动传 BC**
4. **降低同步 / stream 化**

---

## 一、核心问题诊断（重构动机）

| 问题 | 现状 | 根因 |
|---|---|---|
| P1 每项一个 kernel | `Equation::computeRHS` 顺序遍历 terms，逐个调用 `gpu_launcher` | 表达式树在运行时用 `std::function` 解释执行 |
| P2 中间结果落全局内存 | 复合项经 `materialiseGPU` 写 scratch 再读回 | pointwise 与 stencil 未区分，一律物化 |
| P3 两套 DSL 割裂 | `pw/mul` vs `fpw2/fmul/ffield` | 融合是独立类型系统（`FusedTerm.h`），与主 DSL 零共享 |
| P4 手动传 BC | `lap(expr, bcs)` / `grad(expr, axis, bcs)` | 表达式不持 ghost，类型不跟踪 halo 需求 |
| P5 标量乘两条路径 | `s*Field`→生成 identity-pw；`s*Term`→改 coeff | `Term` 概念过载 |
| P6 过度同步 | 每次全数组 memset（含 ghost）+ `cudaDeviceSynchronize` | 无 stream 模型 |

相关源码位置：
- `include/equation/Term.h` —— `Term` / `RHSExpr` / 算子工厂
- `include/equation/FieldOps.inl` —— 运算符重载、materialise、mulAccumulate
- `include/equation/FusedTerm.h` —— 独立的编译期融合 DSL（待并入主 DSL）
- `src/equation/Equation.cu` —— `Equation::computeRHS` / `computeRHSCPU`

---

## 二、关键设计洞察：以「是否需要 halo」划分边界

这是整个重构的支点。把表达式节点分成两类：

- **Local（pointwise）节点**：输出 cell 只依赖同 cell 输入。
  包括 `pw`、`+`、`-`、`*`、标量缩放、`field` 叶子。
  → **可无限融合，永不物化。**
- **Stencil 节点**：输出 cell 依赖邻居 halo。
  包括 `lap`、`grad`、`iso_grad`、`grad_dot`、`div`、`curl`。
  → **是融合的边界**：它的输入子树必须先求值到一个带 ghost 的缓冲并刷新 BC，才能取邻居。

**融合规则**：从 RHS 表达式树的根往下，连续的 Local 节点合并进同一个 kernel；
遇到 Stencil 节点则切断，其操作数子树递归求值为一个「物化场」
（materialized field，带 ghost + 已应用 BC），该场作为 Stencil kernel 的输入。

这一条规则同时解决 P1、P2、P3、P4：

- P1/P2：Local 子树融合 → 一个 kernel、零中间全局内存。
- P4：物化场是真实带 ghost 的场，BC 由求值器在物化点自动应用（见第四节）。
- P3：主 DSL 自带这套规则，无需独立 `FusedTerm`。

---

## 三、新类型体系

### 3.1 用 `Expr`（表达式节点）替代当前 `Term`

定义表达式节点的代数数据类型（C++ 用 variant 或带 tag 的节点 + `shared_ptr` 子树）：

```
Expr =
  | Leaf(const ScalarField*)                       // 叶：一个场
  | Scalar(double)                                  // 叶：常量（用于折叠）
  | PointwiseN(functor, children: Expr[1..N])       // Local：用户/内建 functor
  | Stencil(kind, axis?, child: Expr)               // 需要 halo: LAP/GRAD/ISO_GRAD/DIV/...
  | StencilBinary(kind, a: Expr, b: Expr)           // grad_dot 等
```

要点：

- 节点是**纯数据描述**，不携带 `std::function`。launcher 在 lowering 阶段才生成。
- `PointwiseN` 的 functor 仍用 extended-lambda 捕获；多个相邻 Pointwise 合并时 functor 组合。
- **常量折叠**：`Scalar * Pointwise`、`Scalar + Scalar` 在构建期化简，
  消除 P5 —— 标量乘统一为「乘进 Pointwise 的系数」，不再生成独立 identity-pw 节点。

### 3.2 类型层面跟踪 halo 需求（解决 P4）

给 `Expr` 增加可推导属性 `ghostRequired()`：

- Leaf / Pointwise → 子树 max
- Stencil(LAP/GRAD, child) → `child` 物化后归零，但 Stencil 本身向**外层**
  要求 `stencilWidth`（CD2=1，iso=1，更高阶更大）

`Equation::setRHS` 时遍历树，自动算出 unknown / 各源场所需最小 ghost，
并校验各场实际 ghost 足够，不足则报错。
**用户不再手填 ghost 关系，也不再传 bcs 给算子。**

### 3.3 Lowering：把 `Expr` 树编译成 kernel 计划（解决 P1/P2/P3）

引入 lowering pass：`Expr` → `EvalPlan`（有序 step 序列）。

```
EvalPlan = list of Step
Step =
  | MaterializeStep(targetBuf, subExpr, bcs)   // 求值一个 Local 子树到 buf，应用 BC
  | FusedPointwiseStep(targetRhs, fusedKernel) // 一个融合 kernel，accumulate 进 rhs
```

算法（后序遍历）：

1. 遇到 Stencil 节点：其 child 是需要 halo 的输入 → 生成 `MaterializeStep`
   （child 子树融合成单 kernel 写入 scratch 场），随后 stencil 读该 scratch 场邻居。
2. 连续 Local 节点：收集成一个 `FusedPointwiseStep`，生成**一个** kernel，
   把整棵 Local 子树在寄存器里算完再写 rhs。

两种实现策略（二选一，权衡见第八节）：

- **策略 A —— 运行时解释 + 融合（改动小）**：保留 `std::function` 但**只在融合边界生成**，
  Local 子树用「组合 functor」运行时拼接。kernel 数 ≈ stencil 节点数 + 1（而非 term 数）。
  仍有类型擦除开销。
- **策略 B —— 编译期模板融合（性能最优）**：`Expr` 用表达式模板（CRTP/variadic），
  整棵 Local 子树 lower 成一个 `__global__` 模板实例，零 `std::function`、全内联。
  本质是把 `FusedTerm.h` 能力变成主 DSL 的**默认后端**，用户语法不变。

**推荐**：B 作为 GPU 默认路径，A 作为 CPU fallback 与运行时动态表达式的退路。

---

## 四、BC 自动应用模型（解决 P4 的运行时部分）

当前 `lap(expr, bcs)` 把 BC 传递责任丢给用户。新模型：

- `Equation` / `Solver` 持有「unknown 与各源场 → BCSet」映射
  （构造时登记；`Solver` 多步模式天然有 `sourceField + bcs`）。
- Lowering 生成 `MaterializeStep` 时，从该映射查出物化场对应 BC 自动注入。
- 物化场是「某具名源场的派生」时继承其 BC；纯中间表达式默认用 unknown 的 BC，
  或要求显式标注（见第八节风险 3）。

目标接口：

```cpp
Equation eqC(c, "CH_c");
eqC.setRHS(M * lap(mu));     // 不再需要 bcs 参数
// Equation 内部知道 mu 的 BC（由 Solver step 注册）
```

- `lap(mu)`（直接对场求 stencil）：mu 自带 ghost、BC 在 step 已声明 → 完全自动。
- `lap(expr)`（对表达式求 stencil）：物化后用 expr 根场的 BC。

---

## 五、Stream 化与同步削减（解决 P6）

- `Equation` 持有 `cudaStream_t`（可由 Solver 注入共享 stream）。
- 所有 kernel、memset 提交到该 stream，**移除每次 `computeRHS` 末尾的 `cudaDeviceSynchronize`**。
- 仅在以下时机同步：结果被 CPU 读取（IO/writeFields）、跨 stream 依赖、
  step 结束 `d_prev ← d_curr` 前。
- memset 优化：只清物理区，或「首项用 assign 而非 accumulate」，
  省掉全数组（含 ghost）memset。多步流水线中各方程排进同一 stream 顺序执行，省去中间 barrier。

---

## 六、向量路径泛型化（次要，可后做）

`VectorRHSExpr` 不再手工镜像标量逻辑：把 `Expr` 参数化为分量数 `N`，
标量是 `N=1` 特例。`lap/grad/div/curl/pw` 在 `Expr<N>` 上统一定义，
消除 `TODO(vector)`。优先级最低，可独立 PR。

---

## 七、实施阶段划分（建议执行顺序）

| 阶段 | 内容 | 可独立验证 |
|---|---|---|
| 0 | 确认 nvcc/GPU 可用，跑通现有一个求解器作数值基线 | 是 |
| 1 | 引入 `Expr` 节点类型 + 常量折叠 + `ghostRequired` 推导，**先不改求值**（旧 launcher 仍跑），新树并行存在仅做校验 | 是 |
| 2 | 实现 lowering + 融合 Local 子树（策略 A，运行时融合），替换 `computeRHS` 主循环；对照基线验证数值一致 | 是 |
| 3 | BC 自动注入：`Equation`/`Solver` 登记源场 BC，去掉 `lap(expr,bcs)` 显式参数 | 是 |
| 4 | Stream 化 + 移除强制同步，压测吞吐 | 是 |
| 5 | （可选，性能）策略 B 编译期模板融合，作为 GPU 默认后端；`FusedTerm.h` 降级为内部实现并从公开 API 移除 | 是 |
| 6 | （可选）向量路径泛型化 | 是 |

每阶段必须用阶段 0 的数值基线回归，保证重构不改变物理结果。

---

## 八、关键风险与权衡（实现者必读）

1. **策略 B 的编译期成本**：模板融合会拉长 nvcc 编译时间、增大二进制。
   若某方程项数极多（如 MPF 的 μ），单 kernel 寄存器压力可能溢出导致 occupancy 下降。
   → 需设「融合上限」，超过则自动切分多个融合 kernel。
2. **CPU fallback**：策略 B 的模板路径必须保留一条解释执行的 CPU 路径
   （复用现有 `cpu_kernel` 思路），否则无 GPU 单测失效。
3. **BC 自动推导的歧义**：纯中间表达式没有"天然 BC"。
   设计上**必须强制**——要么继承根场 BC，要么要求用户在 `Equation` 上显式声明该中间量 BC，
   **不能静默用零 BC**，否则是比现状更隐蔽的正确性陷阱。
4. **融合边界判定**：`grad_dot`、`div(VectorRHSExpr)` 等是 stencil-binary，
   lowering 要正确识别为切断点，别误并进 Local 子树。

---

## 九、保留不动的部分（避免过度重构）

- 表达层 DSL 风格（`M * lap(mu)`、`c*eta + 2*c - lap(c)`、`pw(c, PHIX_FN(...))`）—— 可读性是最大优点，**保留**。
- `pw` + extended-lambda 的 pointwise 自定义函数 —— 保留。
- `ScratchPool` 的缓冲复用思想 —— 保留（lowering 的物化场仍从池中取）。
- CPU/GPU 双路径 —— 保留（CPU 路径用于无 GPU 单测）。
