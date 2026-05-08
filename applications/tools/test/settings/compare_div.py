"""
compare_div.py — 读取 U.vfield，用 Python 数值差分计算散度，输出 VTS

散度公式（2 阶中心差分，与 PhiX div(U) 一致）：
    divU[i,j] = (Ux[i+1,j] - Ux[i-1,j]) / (2*dx)
              + (Uy[i,j+1] - Uy[i,j-1]) / (2*dy)

边界（NoFlux）处理：ghost 格用镜像填充（Neumann 零梯度），与 C++ NoFlux BC 一致。

运行：
    python3 settings/compare_div.py
"""

import struct
import math
import numpy as np
import os

# ---- 参数（与 settings.jsonc 一致）---------------------------------------
nx, ny = 10, 10
dx, dy = 1.0, 1.0
x0, y0 = 0.0, 0.0
BC = "NoFlux"          # "Periodic" 或 "NoFlux"

input_path  = "input/U.vfield"
output_path = "output/divU_py.vts"

# ---- 读取 .vfield ----------------------------------------------------------
def read_vfield(path):
    with open(path, "rb") as f:
        meta = {}
        while True:
            line = f.readline().decode("ascii").strip()
            if line == "---":
                break
            if line.startswith("nComponents"):
                meta["nComp"] = int(line.split()[1])
            elif line.startswith("nx"):
                p = line.split()
                meta["nx"], meta["ny"], meta["nz"] = int(p[1]), int(p[3]), int(p[5])
            elif line.startswith("ghost"):
                meta["ghost"] = int(line.split()[1])

        _nx, _ny, _nz = meta["nx"], meta["ny"], meta["nz"]
        nComp = meta["nComp"]
        n = _nx * _ny * _nz
        components = []
        for _ in range(nComp):
            raw = f.read(n * 8)
            # shape: (nz, ny, nx) — x fastest in storage, so reshape accordingly
            arr = np.frombuffer(raw, dtype=np.float64).reshape(_nz, _ny, _nx)
            components.append(arr[0])   # 取 2D 切片 (ny, nx)
    return components, meta

comps, meta = read_vfield(input_path)
Ux = comps[0].copy()   # (ny, nx)
Uy = comps[1].copy()

print(f"Read {input_path}: nx={meta['nx']} ny={meta['ny']} nComp={meta['nComp']}")

# ---- 添加 ghost 层（1 格）--------------------------------------------------
def add_ghost(arr, bc):
    """在 arr (ny, nx) 四周各加 1 圈 ghost，返回 (ny+2, nx+2)"""
    ny_, nx_ = arr.shape
    g = np.zeros((ny_ + 2, nx_ + 2))
    g[1:-1, 1:-1] = arr
    if bc == "Periodic":
        g[1:-1,  0] = arr[:, -1]   # x_min ← x_max
        g[1:-1, -1] = arr[:,  0]   # x_max ← x_min
        g[0,  1:-1] = arr[-1, :]   # y_min ← y_max
        g[-1, 1:-1] = arr[ 0, :]   # y_max ← y_min
        g[0,  0]  = arr[-1, -1]; g[0,  -1] = arr[-1, 0]
        g[-1, 0]  = arr[ 0, -1]; g[-1, -1] = arr[ 0, 0]
    else:  # NoFlux: zero-gradient (copy nearest interior cell)
        g[1:-1,  0] = arr[:,  0]   # x_min
        g[1:-1, -1] = arr[:, -1]   # x_max
        g[0,  1:-1] = arr[ 0, :]   # y_min
        g[-1, 1:-1] = arr[-1, :]   # y_max
        g[0,   0] = arr[ 0,  0]; g[0,  -1] = arr[ 0, -1]
        g[-1,  0] = arr[-1,  0]; g[-1, -1] = arr[-1, -1]
    return g

Ux_g = add_ghost(Ux, BC)   # (ny+2, nx+2), index: [j+1, i+1] = physical (i,j)
Uy_g = add_ghost(Uy, BC)

# ---- 2 阶中心差分散度 -------------------------------------------------------
# 对 ghost 数组: physical cell (i,j) → indices [j+1, i+1]
divU = np.zeros((ny, nx))
for j in range(ny):
    for i in range(nx):
        dUx_dx = (Ux_g[j+1, i+2] - Ux_g[j+1, i]) / (2.0 * dx)
        dUy_dy = (Uy_g[j+2, i+1] - Uy_g[j,   i+1]) / (2.0 * dy)
        divU[j, i] = dUx_dx + dUy_dy

# ---- 理论散度对比 -----------------------------------------------------------
xs = np.array([(x0 + (i + 0.5) * dx) for i in range(nx)])
ys = np.array([(y0 + (j + 0.5) * dy) for j in range(ny)])
X, Y = np.meshgrid(xs, ys)   # (ny, nx)
divU_exact = np.cos(X) * np.sin(Y) + np.sin(X) * np.cos(Y)  # = sin(x+y)

err = np.abs(divU - divU_exact)
print(f"Max error vs exact: {err.max():.4e}")
print(f"divU  range: [{divU.min():.4f}, {divU.max():.4f}]")
print(f"exact range: [{divU_exact.min():.4f}, {divU_exact.max():.4f}]")

# ---- 写 VTS（格式与 PhiX scalar VTS 完全一致）-------------------------------
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w") as f:
    nz = 1
    f.write('<?xml version="1.0"?>\n')
    f.write('<VTKFile type="StructuredGrid" version="0.1" byte_order="LittleEndian">\n')
    f.write(f'  <StructuredGrid WholeExtent="0 {nx} 0 {ny} 0 {nz}">\n')
    f.write(f'    <Piece Extent="0 {nx} 0 {ny} 0 {nz}">\n')

    # 角点坐标
    f.write('      <Points>\n')
    f.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                x = x0 + i * dx
                y = y0 + j * dy
                z = 0.0
                f.write(f'          {x:.12e} {y:.12e} {z:.12e}\n')
    f.write('        </DataArray>\n')
    f.write('      </Points>\n')

    # 标量 divU
    f.write('      <CellData Scalars="divU">\n')
    f.write('        <DataArray type="Float64" Name="divU" format="ascii">\n')
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                f.write(f'          {divU[j, i]:.12e}\n')
    f.write('        </DataArray>\n')
    f.write('      </CellData>\n')

    f.write('    </Piece>\n')
    f.write('  </StructuredGrid>\n')
    f.write('</VTKFile>\n')

print(f"Written {output_path}")
