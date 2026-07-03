#!/usr/bin/env python3
"""
生成 GFA_binary 测试案例的初始场文件。

物理设置（参考 develop/CH+AC / develop/Ostwald_Ripening 的 PFHub 型初始条件）
--------------------------------------------------------------------------
域: 200 × 200 网格, dx = dy = 0.6 nm（stage 4 真实尺度）, x0 = y0 = 0.0，周期边界

初始条件（余弦扰动，确定性、无随机数）。扰动公式中的频率是按
"坐标以格距为单位"（PFHub 200×200, dx=1）设计的，因此在物理网格上
用无量纲坐标 x̃ = x/dx, ỹ = y/dy 求值，保持空间图样与原算例一致:

  c(x̃,ỹ)   = c0 + eps_c * { cos(0.105x̃)cos(0.11ỹ)
                          + [cos(0.13x̃)cos(0.087ỹ)]^2
                          + cos(0.025x̃-0.15ỹ)cos(0.07x̃-0.02ỹ) }

  phi(x̃,ỹ) = eps_phi * { cos(0.01x̃-4)cos(0.017ỹ)
                       + cos(0.12x̃)cos(0.12ỹ)
                       + psi*[cos(0.047x̃+0.0415ỹ)cos(0.032x̃-0.005ỹ)]^2 }^2

  参数: c0=0.5, eps_c=0.05, eps_phi=0.1, psi=1.5
  （c ∈ [0.4, 0.65] 左右，处于 Cu-Zr f_L 表的 c∈[0.001,0.999] 安全区内）

  eta(x,y) = 0 附近的小正扰动（非晶序参量，stage 6 / GFA_evo 用）:
    每格点取 [ETA_FLOOR, ETA_AMP] 区间的均匀随机数，固定种子保证可复现。
    全部严格 > 0（满足 stage 6 要求），幅值 ~0.1，均值 ~0.05。
    —— 注意：eta 是 c/phi 之外唯一使用随机数的场。

mu 是辅助场（求解器每步从 (c,phi) 重建），不需要初始场文件。

用法（在 test/ 目录下运行）
  python3 gen_initial_field.py

输出: settings/initial_field/{c,phi,eta}.field  (DAT 格式)
"""

import os
import math
import random

# ── 网格参数（必须与 settings/settings.jsonc 一致）──────────────────────────
NX, NY = 200, 200
NZ     = 1
DX     = 1.8e-9    # [m]  1.8 nm（须与 settings.jsonc 一致）
DY     = 1.8e-9    # [m]
X0     = 0.0
Y0     = 0.0

# ── 初始条件参数 ────────────────────────────────────────────────────────────
C0      = 0.5    # 平均成分
EPS_C   = 0.05   # c 扰动幅值
EPS_PHI = 0.1    # phi 扰动幅值
PSI     = 1.5    # phi 扰动交叉项权重

# eta（非晶序参量）初始模式：
#   "zero"  —— 全场 eta = 0（基线/对照：eta=0 是不动点，h'(0)=g'(0)=0、
#              cross=2·w_ex·η·φ²=0、∇²η=0 ⇒ eta 恒为 0，φ 退化回纯晶体 stage-5 行为）
#   "noise" —— 0 附近的小正扰动，全部 > 0（stage 6 形核种子，见下方参数）
ETA_INIT  = "zero"
ETA_AMP   = 0.1      # eta 扰动幅值（上界）
ETA_FLOOR = 1.0e-4   # eta 下界（>0，且在 %.6f 下不会四舍五入成 0）
ETA_SEED  = 20260615 # 随机种子，保证可复现
random.seed(ETA_SEED)

# ── 导出路径 ────────────────────────────────────────────────────────────────
# 脚本位于 test/ 目录，输出到 test/settings/initial_field/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, "settings", "initial_field")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 字段名称（须与求解器中 ScalarField 的 name 一致）────────────────────────
FIELDS = ("c", "phi", "eta")

# PhiX DAT 格式头
HEADER_TMPL = (
    "# PhiX ScalarField - DAT\n"
    "# name: {name}\n"
    f"# nx {NX}  ny {NY}  nz {NZ}\n"
    "# x y z value\n"
)

# ── 打开所有文件并写入头部 ───────────────────────────────────────────────────
handles = {
    n: open(os.path.join(OUT_DIR, f"{n}.field"), "w")
    for n in FIELDS
}
for n, fh in handles.items():
    fh.write(HEADER_TMPL.format(name=n))

print(f"写入 {len(FIELDS)} 个字段（{NX}×{NY} = {NX*NY} 格点/场）...", flush=True)

# ── 主写入循环（PhiX 读取顺序: k 外层, j 中层, i 内层；坐标为格心）───────────
for k in range(NZ):
    zv = 0.0
    for j in range(NY):
        yv = Y0 + (j + 0.5) * DY
        yt = yv / DY            # ỹ：无量纲坐标（格距单位），扰动公式用
        for i in range(NX):
            xv    = X0 + (i + 0.5) * DX
            xt    = xv / DX     # x̃
            coord = f"{xv:.12e}  {yv:.12e}  {zv:.12e}"

            c_val = C0 + EPS_C * (
                math.cos(0.105 * xt) * math.cos(0.11 * yt)
                + (math.cos(0.13 * xt) * math.cos(0.087 * yt)) ** 2
                + math.cos(0.025 * xt - 0.15 * yt)
                  * math.cos(0.07 * xt - 0.02 * yt)
            )

            phi_val = EPS_PHI * (
                math.cos(0.01 * xt - 4.0) * math.cos(0.017 * yt)
                + math.cos(0.12 * xt) * math.cos(0.12 * yt)
                + PSI * (math.cos(0.047 * xt + 0.0415 * yt)
                         * math.cos(0.032 * xt - 0.005 * yt)) ** 2
            ) ** 2

            # eta：基线全 0，或 0 附近的小正扰动（严格 > 0）
            eta_val = 0.0 if ETA_INIT == "zero" else random.uniform(ETA_FLOOR, ETA_AMP)

            handles["c"  ].write(f"{coord}  {c_val:.6f}\n")
            handles["phi"].write(f"{coord}  {phi_val:.6f}\n")
            handles["eta"].write(f"{coord}  {eta_val:.6f}\n")

for fh in handles.values():
    fh.close()

# ── 摘要 ──────────────────────────────────────────────────────────────────
print("完成。生成文件：")
for n in FIELDS:
    path = os.path.join(OUT_DIR, f"{n}.field")
    size = os.path.getsize(path) / 1024
    print(f"  {path}  ({size:.0f} kB)")
print()
print("域参数:")
print(f"  尺寸: {NX*DX*1e9:.1f} nm × {NY*DY*1e9:.1f} nm")
print(f"  格点: {NX} × {NY}")
print()
print("下一步：在 test/ 目录下运行求解器")
print("  mkdir -p output")
print("  ../GFA_binary settings/settings.jsonc")
