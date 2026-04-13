# simulate.cpp — 求解问题说明

## 物理背景

本求解器实现的是 **ChiMaD Phase-Field Benchmark Problem 2**：
多相（4个晶粒/取向）析出相变的相场模拟，即一个浓度场与多个有序参数场的耦合演化问题。

体系描述：在一个均匀固溶体基体中，通过相分离同时析出 4 个 β 相晶粒（具有不同取向/变体），基体相为 α 相。

---

## 控制方程

### Cahn-Hilliard 方程（浓度场 $c$）

$$
\frac{\partial c}{\partial t} = \nabla \cdot \left( M_c \nabla \mu \right)
$$

其中化学势：

$$
\mu = \frac{\partial f}{\partial c} - \kappa_c \nabla^2 c
$$

### Allen-Cahn 方程（有序参数场 $\eta_i$，$i=1,2,3,4$）

$$
\frac{\partial \eta_i}{\partial t} = -M_p \left( \frac{\partial f}{\partial \eta_i} - \kappa_p \nabla^2 \eta_i \right)
$$

---

## 自由能密度

$$
f = \rho^2 \left[ (c - c_\alpha)^2 (1-h) + (c - c_\beta)^2 h \right]
+ \omega \sum_i \left[ \eta_i^2(1-\eta_i)^2 + \alpha \sum_{j \neq i} \eta_i^2 \eta_j^2 \right]
$$

其中插值函数：

$$
h(\{\eta_i\}) = \sum_i \eta_i^3(6\eta_i^2 - 15\eta_i + 10)
$$

- 第一项为化学自由能，描述 α 相（$c_\alpha=0.3$）与 β 相（$c_\beta=0.7$）的两相平衡。
- 第二项为结构自由能，$\eta_i^2(1-\eta_i)^2$ 为双阱势，$\alpha \sum_{j\neq i}\eta_i^2\eta_j^2$ 惩罚不同晶粒的重叠。

---

## 数值参数

| 参数 | 值 |
|------|-----|
| 网格 | $200 \times 200$ |
| 物理域尺寸 | $200 \times 200$ |
| 时间步长 $dt$ | $10^{-3}$ |
| 总模拟时间 | $10^5$ |
| 边界条件 | 周期性（默认）或无通量 |

---

## 输出文件

| 文件 | 内容 |
|------|------|
| `Fields.<t>.dat` | 各时刻的 $x,y,c,\eta_1,\eta_2,\eta_3,\eta_4$ 空间分布 |
| `Energy.dat` | 总自由能随时间的演化 |
| `Fields.<t>.dat.png` | 由 `plot.py` 生成的可视化图像 |

---

## 运行方式

```bash
# 编译
g++ -O2 -o simulate simulate.cpp -lm

# 运行（周期性边界条件）
./simulate

# 运行（无通量边界条件）
./simulate 0
```

## 后处理

```bash
python plot.py
```
