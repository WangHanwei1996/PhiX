# v2.25.0 — LBM 模块（D2Q9 BGK，`lbm/LBM.h`）

## 摘要

新增格子玻尔兹曼模块，为后续流动耦合相场（凝固熔体对流等）提供
**显式速度求解**能力：

- **D2Q9 + BGK** 单松弛碰撞，c_s² = 1/3，ν = (τ−½)/3（格子单位）;
- **Guo 体力**（2002，二阶一致：宏观速度含 F/2 修正、源项带
  1−1/(2τ) 权重）；
- 碰撞（原地）+ **pull 流动**（双缓冲指针交换，零拷贝），分布函数
  全程驻留 device；
- 逐侧边界：**周期**（默认）或**半程反弹无滑移壁**；
- `macroscopics(rho, ux, uy)` 把宏观量写进 PhiX `ScalarField` 物理格
  ——与相场求解器的耦合接口（速度场直接喂给 `adv(u, f)` 等）。

## 实测验证（module_lbm）

| 检验 | 结果 |
|------|------|
| Poiseuille 稳态剖面 vs 解析抛物线（半程 BB 壁位于外排半格外） | 最大相对偏差 **1.4e-4** |
| 黏度动力学：启动瞬态中线亏量按最慢模式 ν(π/H)² 衰减 | 实测速率 vs 解析 **−2.9%** |
| 质量守恒（碰撞+流动+反弹，5000 步，带体力） | 相对漂移 **4e-13** |
| 周期方向平移不变性 | <1e-12 |
| τ ≤ 0.5 参数校验 | 抛 `std::invalid_argument` |

## API

```cpp
#include "lbm/LBM.h"

LBMParams p;  p.tau = 0.9;  p.fx = 1e-6;      // 格子单位
LBM2D lbm(mesh, p);                            // 格子尺寸取 mesh.n
lbm.setWall(Axis::Y, Side::LOW);
lbm.setWall(Axis::Y, Side::HIGH);              // 未标记的侧 = 周期
lbm.initialize(1.0);                           // 均匀平衡态
lbm.run(20000);
lbm.macroscopics(&rho, &ux, &uy);              // 任一指针可为 null
double nu = lbm.latticeViscosity();
```

**单位约定**：模块内 dx = dt = 1（格子单位）；物理标定（选 dx_phys、
dt_phys，换算 ν 与力）由调用方完成——文档已注明。

## 范围与扩展位

首版刻意最小：D2Q9 / BGK / 常数体力 / 均匀平衡初始化。预留的下一步
（接口不变即可扩展）：逐格初速度场、空间变化力（相场耦合的界面力）、
MRT 碰撞、D3Q19。`Real` 全程参数化，FLOAT 构建同样编译。

## 测试

全量 ctest **32/32**（新增 `module_lbm`），FLOAT 3/3。

## 兼容性

纯新增模块（`src/lbm/LBM.cu` 已注册 `phix` 库）。
