# PhiX v1.1.1 发布说明

**发布日期**：2026-04-14  
**标签**：`v1.1.1`

---

## 概述

v1.1.1 是在 v1.1.0 基础上的功能增强版本，核心改进集中在两个方面：

1. **`pw` 扩展为多场版本**：支持同时对 2 或 3 个标量场做逐点运算，大幅简化多场耦合方程的 RHS 表达
2. **`ScalarField` 算术运算符重载**（`FieldOps.inl`）：允许在 `setRHS` 中直接书写自然数学表达式

此外，新增 Ostwald 熟化示例与场运算测试工程，进一步验证耦合相场系统的正确性。

---

## 新增功能

### pw 多场逐点算子（`Term.h` / `TermPW.inl`）

在原有单场 `pw(f, func)` 的基础上，新增：

| 重载签名 | GPU Kernel | 说明 |
|---|---|---|
| `pw(f1, f2, func)` | `kernel_pw2_accumulate` | 2 场逐点：`rhs[i] += c * func(f1[i], f2[i])` |
| `pw(f1, f2, f3, func)` | `kernel_pw3_accumulate` | 3 场逐点：`rhs[i] += c * func(f1[i], f2[i], f3[i])` |

Functor 要求标注 `__host__ __device__`，支持 `--expt-extended-lambda` lambda 语法。

示例（Allen-Cahn + Cahn-Hilliard 耦合驱动力）：
```cuda
// dF/dc 依赖 c 和 eta
auto dFdc = pw(c, eta,
    [] __host__ __device__ (double c_, double e_) {
        return 2.0 * rho2 * (c_ - c_alpha) * (1.0 - h(e_))
             + 2.0 * rho2 * (c_beta - c_)  * h(e_);
    });
```

### ScalarField 算术运算符（`FieldOps.inl`）

新增对 `ScalarField` 的 `operator*`、`operator+`、`operator-` 重载，直接返回 `Term`，可插入 `RHSExpr`：

```cuda
// v1.1.0 写法（仍有效）
eq.setRHS(M * lap(c) + pw(c, [] __host__ __device__ (double v) { return v - v*v*v; }));

// v1.1.1 新写法（更自然）
eq.setRHS(M * lap(c) + c - c * c * c);
```

支持的运算：

| 表达式 | 语义 |
|---|---|
| `f1 * f2` | `rhs[i] += f1[i] * f2[i]` |
| `f1 + f2` | `rhs[i] += f1[i] + f2[i]` |
| `f1 - f2` | `rhs[i] += f1[i] - f2[i]` |
| `s * f` / `f * s` | `rhs[i] += s * f[i]` |
| `f + s` / `s + f` | `rhs[i] += f[i] + s` |
| `f - s` / `s - f` | `rhs[i] += f[i] - s` |
| `-f` | `rhs[i] += -f[i]` |

---

## 新增示例

### Ostwald 熟化（`develop/Ostwald_Ripening/`）

多相场 AC-CH 耦合系统，模拟析出相的竞争粗化过程。

**控制方程**（Allen-Cahn + Cahn-Hilliard 耦合）：

$$\frac{\partial \eta_i}{\partial t} = -L \left( \frac{\partial f_{\mathrm{chem}}}{\partial \eta_i} - \kappa_\eta \nabla^2 \eta_i \right)$$

$$\frac{\partial c}{\partial t} = \nabla \cdot \left\{ M \nabla \left( \frac{\partial f_{\mathrm{chem}}}{\partial c} - \kappa_c \nabla^2 c \right) \right\}$$

其中自由能密度：

$$f_{\mathrm{chem}} = f^\alpha(c)\,[1 - h(\boldsymbol{\eta})] + f^\beta(c)\,h(\boldsymbol{\eta}) + w\,g(\boldsymbol{\eta})$$

- 相场插值函数 $h(\boldsymbol{\eta}) = \sum_i \eta_i^3(6\eta_i^2 - 15\eta_i + 10)$
- 障碍势 $g(\boldsymbol{\eta}) = \sum_i \eta_i^2(1-\eta_i)^2 + \alpha \sum_{i \neq j} \eta_i^2 \eta_j^2$
- 建模文档：`develop/Ostwald_Ripening/doc/modeling.md`

### 场运算测试（`develop/fieldOperation/`）

针对 `pw`（单场、2 场、3 场）与 `FieldOps.inl` 算术运算符的单元测试工程，验证 GPU / CPU 路径结果一致性。

---

## 文件结构变更

```
include/
└── equation/
    ├── Term.h          ← 新增 pw(f1,f2,func) / pw(f1,f2,f3,func) 声明
    ├── TermPW.inl      ← 新增 kernel_pw2_accumulate / kernel_pw3_accumulate 及对应模板定义
    └── FieldOps.inl    ← 新增，ScalarField 算术运算符重载
develop/
├── fieldOperation/
│   ├── CMakeLists.txt  ← 新增
│   └── test_fieldops.cu← 新增
└── Ostwald_Ripening/
    ├── CMakeLists.txt  ← 新增
    ├── AC-CH.cu        ← 新增
    └── doc/
        └── modeling.md ← 新增
```

---

## 已知限制

- 继承 v1.1.0 全部已知限制
- `FieldOps.inl` 中 `f + s` 等运算会将常数烘焙进 `pw` lambda，不单独生成 constant-offset kernel
