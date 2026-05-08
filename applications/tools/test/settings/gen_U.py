"""
gen_U.py — generate input/U.vfield for div_field test

U_x(i,j) = sin(x) * sin(y)
U_y(i,j) = sin(x) * sin(y)

where x = (i + 0.5) * dx,  y = (j + 0.5) * dy

Run from the test/ directory:
    python3 settings/gen_U.py
"""

import struct
import math
import os

# ---- mesh (must match settings.jsonc) ----
nx, ny = 10, 10
dx, dy = 1.0, 1.0
x0, y0 = 0.0, 0.0
ghost   = 1
name    = "U"
nComp   = 2

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

# ---- cell-centre coordinates ----
xs = [(x0 + (i + 0.5) * dx) for i in range(nx)]
ys = [(y0 + (j + 0.5) * dy) for j in range(ny)]

# ---- build physical data: shape (nComp, ny, nx), x-fastest ----
def field_val(i, j):
    return math.sin(xs[i]) * math.sin(ys[j])

# x-fastest order: k=0, j in [0,ny), i in [0,nx)
data = []
for c in range(nComp):          # both components identical
    for j in range(ny):
        for i in range(nx):
            data.append(field_val(i, j))

# ---- write binary .vfield ----
path = "input/U.vfield"
with open(path, "wb") as f:
    header = (
        f"# PhiX VectorField\n"
        f"name         {name}\n"
        f"nComponents  {nComp}\n"
        f"nx {nx}  ny {ny}  nz 1\n"
        f"ghost        {ghost}\n"
        f"---\n"
    )
    f.write(header.encode("ascii"))
    f.write(struct.pack(f"{len(data)}d", *data))

print(f"Written {path}  ({nx}x{ny}, {nComp} components)")
print("Sample values (i=0..4, j=0):")
for i in range(min(5, nx)):
    v = field_val(i, 0)
    print(f"  U[0/1][i={i},j=0]  x={xs[i]:.3f} y={ys[0]:.3f}  val={v:.6f}")
