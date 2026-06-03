#!/usr/bin/env python3
"""
生成 GFA_4ph 测试案例的初始场文件。

物理设置
--------
域: 256 × 256 网格, dx = dy = 6e-10 m (0.6 nm), x0 = y0 = 0.0
  → 域尺寸: 153.6 nm × 153.6 nm

初始条件:
  · 液相背景（φ₀=1，其余=0）
  · 域中心一个 CuZr B2 晶核（φ₂=1），半径 15 nm（超临界核，r_c ~ 5 nm）
  · η=0 全域（纯液相，无非晶有序化）
  · c=0.5 全域均匀组成场（Zr 摩尔分数）

晶核临界半径估算（T=1100 K）
  γ₀₂ ≈ √(2w₀₂ε₀₂²)/3 = √(2×4.1e8×0.7e-9)/3 ≈ 0.25 J/m²
  ΔGᵥ ≈ ΔHₘ·(ΔT/Tₘ)/Vₘ ≈ 12000×(108/1208)/1.039e-5 ≈ 1.03e8 J/m³
  r_c ≈ 2γ/ΔGᵥ ≈ 4.9 nm  →  使用 R=15 nm 的超临界核

用法（在 test/ 目录下运行）
  python3 gen_initial_field.py

输出: settings/initial_field/{phi0,phi1,phi2,phi3,eta,c}.field  (DAT 格式)
"""

import os
import math

# ── 网格参数（必须与 settings/settings.jsonc 一致）──────────────────────────
NX, NY = 256, 256
NZ     = 1
DX     = 6.0e-10   # [m]  0.6 nm
DY     = 6.0e-10   # [m]
X0     = 0.0
Y0     = 0.0

# ── 晶核参数 ────────────────────────────────────────────────────────────────
R_NUCLEUS   = 15.0e-9   # [m]  CuZr (B2) 晶核半径，超临界（r_c ~ 5 nm）
W_INTERFACE = 3.0 * DX  # [m]  tanh 界面半宽（≈ δ₀₂ = √(2ε₀₂²/w₀₂) ≈ 1.85 nm ≈ 3 cells）

# ── 导出路径 ────────────────────────────────────────────────────────────────
# 脚本位于 test/ 目录，输出到 test/settings/initial_field/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, "settings", "initial_field")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 几何 ────────────────────────────────────────────────────────────────────
cx   = X0 + NX * 0.5 * DX   # 域中心 x
cy   = Y0 + NY * 0.5 * DY   # 域中心 y

# ── 字段名称 ────────────────────────────────────────────────────────────────
FIELDS = ("phi0", "phi1", "phi2", "phi3", "eta", "c")

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

# ── 主写入循环（PhiX 读取顺序: k 外层, j 中层, i 内层）───────────────────────
for k in range(NZ):
    zv = 0.0
    for j in range(NY):
        yv  = Y0 + (j + 0.5) * DY
        dy2 = (yv - cy) ** 2
        for i in range(NX):
            xv    = X0 + (i + 0.5) * DX
            r     = math.sqrt((xv - cx) ** 2 + dy2)
            coord = f"{xv:.12e}  {yv:.12e}  {zv:.12e}"

            # 平滑 tanh 界面（宽度 W_INTERFACE ≈ δ₀₂ ≈ 3 cells）
            # phi2=1 在晶核内，phi2=0 在液相中，过渡宽度 ~3 格
            phi2_val = 0.5 * (1.0 - math.tanh(2.0 * (r - R_NUCLEUS) / W_INTERFACE))
            phi0_val = 1.0 - phi2_val

            handles["phi0"].write(f"{coord}  {phi0_val:.6f}\n")
            handles["phi1"].write(f"{coord}  0.000000\n")
            handles["phi2"].write(f"{coord}  {phi2_val:.6f}\n")
            handles["phi3"].write(f"{coord}  0.000000\n")
            handles["eta" ].write(f"{coord}  0.000000\n")
            handles["c"   ].write(f"{coord}  0.500000\n")

for fh in handles.values():
    fh.close()

# ── 摘要 ──────────────────────────────────────────────────────────────────
print("完成。生成文件：")
for n in FIELDS:
    path = os.path.join(OUT_DIR, f"{n}.field")
    size = os.path.getsize(path) / 1024
    print(f"  {path}  ({size:.0f} kB)")
print()
print(f"晶核参数:")
print(f"  相: CuZr B2 (phi2)")
print(f"  半径: {R_NUCLEUS*1e9:.1f} nm = {R_NUCLEUS/DX:.1f} cells")
print(f"  中心: ({cx*1e9:.1f}, {cy*1e9:.1f}) nm")
print(f"域参数:")
print(f"  尺寸: {NX*DX*1e9:.1f} nm × {NY*DY*1e9:.1f} nm")
print(f"  格点: {NX} × {NY}")
print()
print("下一步：在 test/ 目录下运行求解器")
print("  mkdir -p output")
print("  ../../../../build/applications/solvers/glass_formation_4_phases/2D/GFA_4ph")
