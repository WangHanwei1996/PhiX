# PFHub Benchmark 5 — Stokes 通道流（LBM D2Q9）

规格: https://pages.nist.gov/pfhub/benchmarks/benchmark5-hackathon.ipynb/

−μ∇²u + ∇p = ρg，∇·u=0；ρ=100 kg/m³，μ=1 Pa·s，g=(0,−0.001)；
通道 30×6；入口抛物线 u_x(0,y)=0.009(1−((y−3)/3)²)；
变体 b：椭圆障碍圆心 (7,2.5)，半轴 rx=1、ry=1.5；压力参考 p(30,6)=0。

**求解器**：`applications/solvers/LBM_channel/2D`（新写，调 v2.32.0
LBM 模块：BGK+Guo、半程反弹壁面、Zou-He 速度入口/压力出口、障碍掩码）。
物理↔格子单位换算由配置推导（τ=3ν·dt/dx²+½=0.8，u_lat≤0.009）。
稳态判据：每 5000 步采样 max|Δu|<1e-8·max|u|。

**重力处理**（记录在案）：恒定体力在不可压缩流中被静水压
p_h=ρg·(r−r_ref) 精确平衡、速度场不变；而 LBM 均匀密度 Zou-He 出口
与静水密度梯度不相容（直接加 Guo 重力会驱动 ~7 倍伪环流）。
故 LBM 内 g=0 解动力压，输出端解析叠加 p_h（`hydrostatic` 配置节）。

## 运行

```bash
cd a_channel   # 或 b_ellipse
../../../../applications/solvers/LBM_channel/2D/LBM_channel settings/settings.jsonc
```

300×60 格子（dx=0.1），LBM 是 Navier-Stokes 求解器，本例 Re≈5.4
处于弱惯性 Stokes 区。输出 cut_x7.csv / cut_y5.csv
（y|x, velocity_x, velocity_y, pressure）；稳态收敛后另写
`output/{ux,uy,p}_<step>.vts` 场快照（物理单位、压力含静水叠加，
供 ParaView）。

## 验证（变体 a 对解析泊肃叶解）

| 量 | PhiX-LBM | 理论 | 偏差 |
|----|----------|------|------|
| max u_x | 0.009017 | 0.009 | 0.19% |
| u_x(y) 剖面 | — | 抛物线 | max 0.05% |
| dp/dx | −0.002004 | −0.002 | 0.2% |
| dp/dy | −0.10000 | −0.1（静水） | 精确 |
| p(30,5) | 0.1000 | 0.1 | 精确 |

变体 b：max u_x = 0.0218 m/s（顶部间隙加速，通量守恒一致），
6 万步收敛。
