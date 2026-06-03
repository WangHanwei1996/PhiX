# GFA_4ph 测试案例：Cu₅₀Zr₅₀ 等温晶化

## 测试目的

验证求解器能否复现 **Wang & Napolitano (2012)** 论文所描述的晶体生长物理。

测试案例为**等温晶化**：在 T = 1100 K（低于 CuZr B2 熔点 ~1208 K）下，
从液相中植入一个超临界 CuZr 晶核（半径 15 nm），观察其生长。

---

## 与论文的对照

### 已复现的物理

| 模型组件 | 来源 | 状态 |
|---------|------|------|
| CALPHAD 自由能（G_liq, G_Cu10Zr7, G_CuZr, G_CuZr2）| 论文 Eq.[1]–[7] | ✅ |
| Inden 短程有序项 Δf^SR | 论文 Eq.[8] | ✅ |
| 梯度能 Σᵢ<ⱼ (ε²ᵢⱼ/2)\|φᵢ∇φⱼ − φⱼ∇φᵢ\|² | 论文 Eq.[2] | ✅ |
| 双势阱障碍 wᵢⱼ φᵢ²φⱼ² | 论文 Eq.[2] | ✅ |
| 4 相 pairwise Allen-Cahn + Gibbs 单纯形投影 | 论文 Eq.[3] | ✅ |
| η 结构弛豫 Allen-Cahn | 论文 Eq.[4] | ✅ |
| 模型参数（Table II）| ε²ᵢⱼ, wᵢⱼ, Lᵢⱼ, β | ✅ 全部硬编码 |

### 当前局限性（论文 Fig.5 完整复现所需）

| 功能 | 说明 |
|------|------|
| ❌ 成分场 CH 方程 | c 固定为 0.5，论文求解 Cahn-Hilliard 动态成分 |
| ❌ 温度冷却调度 | T 恒定，论文计算 10²–10⁵ K/s 竞争冷却结果 |
| ❌ η 热噪声 | 无 Gaussian 扰动，论文需噪声激活自发玻璃转变 |

**结论**：当前求解器可验证**等温晶化**的热力学与动力学正确性，  
但无法重现论文 Fig.5 中的冷却速率竞争（晶化 vs. 玻璃化）主要结论。

---

## 物理预期

- **T = 1100 K**，过冷度 ΔT ≈ 108 K，CuZr 晶核应持续生长
- CuZr（φ₂）相热力学稳定（CALPHAD G_CuZr < G_liq at c=0.5, T=1100K）
- Cu₁₀Zr₇（φ₁）和 CuZr₂（φ₃）不应从 c=0.5 的液相生长
- η 维持在 0 附近（T >> Tg = 700K，液相态）

---

## 网格与参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 网格 | 256×256 | |
| dx = dy | 6×10⁻¹⁰ m | 0.6 nm ≈ δ₀₂/3，分辨界面 |
| 域尺寸 | 154 nm × 154 nm | |
| dt | 5×10⁻¹⁰ s | 显式 Euler 稳定限 ~9×10⁻¹⁰ s |
| nSteps | 200,000 | 总模拟时长 100 μs |
| 晶核半径 | 15 nm（25 cells）| 超临界，r_c ~ 5 nm |
| 界面宽度 | δ₀₂ ≈ 1.85 nm（3 cells）| √(2ε₀₂²/w₀₂) |

---

## 运行步骤

```bash
# 1. 进入 test/ 目录
cd applications/solvers/glass_formation_4_phases/2D/test

# 2. 生成初始场（创建 settings/initial_field/*.field）
python3 gen_initial_field.py

# 3. 创建输出目录
mkdir -p output

# 4. 运行求解器（从 test/ 目录执行，路径视编译位置而定）
../../../../build/applications/solvers/glass_formation_4_phases/2D/GFA_4ph
```

> **注意**：求解器使用相对路径读取 `settings/settings.jsonc` 和  
> `settings/initial_field/*.field`，必须从 `test/` 目录运行。

---

## 文件结构

```
test/
├── README.md                    ← 本文件
├── gen_initial_field.py         ← 生成初始场（运行一次）
├── settings/
│   ├── settings.jsonc           ← 求解器配置
│   └── initial_field/           ← 由 gen_initial_field.py 创建
│       ├── phi0.field
│       ├── phi1.field
│       ├── phi2.field
│       ├── phi3.field
│       └── eta.field
└── output/                      ← 求解器输出（运行后自动填充）
    ├── phi0_10000.field
    ├── phi2_10000.field
    └── ...
```
