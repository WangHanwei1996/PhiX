# PFHub Benchmark 2 — Ostwald 熟化（CH + 4×AC）

规格: https://pages.nist.gov/pfhub/benchmarks/benchmark2.ipynb/
（Jokisaari et al., Comp. Mater. Sci. 126 (2017)）

双抛物线自由能 f_chem = f_α(c)[1−h] + f_β(c)h + w·g，
ϱ²=2，c_α=0.3，c_β=0.7，κ_c=κ_η=3，w=1，α=5（晶粒重叠罚项），
M=5，L=5；1 个守恒 c 场 + 4 个非守恒 η 场。

**求解器**：`applications/solvers/CH_4AC_Ostwald/2D`（新写，结构仿
`Cahn-Hillard+Allen-Cahn_double-well`）。要点：

- h、∂f/∂c 对晶粒可加 → μ 方程用 4 个 pw 项拼装；
- 交叉项 Σ_{j≠i}η_j² 经辅助场 S=Ση²（每步 STEADY 重建，冻结在
  时间层 n）→ 4 个 AC 方程彼此**同层**推进；
- `pfhub` 配置节输出 free_energy.csv（含 wα(S²−Ση⁴) 交叉能）。

## 运行

```bash
python3 gen_initial_field.py     # 规格多余弦初始场（i 依赖系数，i=1..4）
cd a_periodic
../../../../applications/solvers/CH_4AC_Ostwald/2D/CH_4AC_Ostwald settings/settings.jsonc
```

200×200，dx=1，dt=1.5e-3（CH 稳定极限≈2.1e-3），66.7 万步到 t≈1000，
RTX 5080 约 4 分钟。

## 结果与 MOOSE 参考对比（t=1000）

F(0)=9064.4，F(t) 401 个采样点严格单调降，总溶质
Σc·dV=20504.5749543 全程逐位不变。与
`develop/Ostwald_Ripening/reference/MOOSE_IA_2a.csv`
（自适应 FEM + 隐式积分）的 TotalEnergy 对标：

| t | PhiX | MOOSE | 相对差 |
|---|------|-------|--------|
| 5 | 2698 | 2787 | −3.2% |
| 50 | 1031 | 1039 | −0.8% |
| 100 | 935 | 916 | +2.0% |
| 400 | 736 | 770 | −4.4% |
| 600 | 672 | 745 | −9.7% |
| 1000 | 618 | 647 | −4.5% |

大部分时段 ±5% 内；t≈600 的峰值偏差来自粗化事件（小粒子消失）
在两种离散间的时序差——BM2 各实现横向比较中的典型现象。
