#!/usr/bin/env python3
"""
Generate initial field files for FeB_ex1.

Domain: 600 x 600 grid, dx = dy = 3.0e-4 m, x0 = y0 = 0.
- Inside a circle of radius 2 (cell units) centered at the domain center:
    c   = 0.33 + small random perturbation  (amplitude 0.01)
    phi = 1.0
    eta = 0.0
- Outside the circle:
    c   = 0.25
    phi = 0.0
    eta = 1.0
- mu  = 0.0  (everywhere, used only as initial guess)

Output: initial_field/{c,phi,eta,mu}.field  (DAT format, no external deps)
"""

import os
import random

# ── Mesh parameters ──────────────────────────────────────────────────────────
NX, NY = 600, 600
NZ     = 1
DX     = 3.0e-4   # [m]
DY     = 3.0e-4   # [m]
X0     = 0.0
Y0     = 0.0

RADIUS_CELLS = 2.0   # circle radius in cell units
NOISE_AMP    = 0.01  # c perturbation amplitude inside circle

# ── Derived geometry ─────────────────────────────────────────────────────────
cx     = X0 + NX * 0.5 * DX   # domain-centre x
cy     = Y0 + NY * 0.5 * DY   # domain-centre y
r_phys = RADIUS_CELLS * DX    # circle radius [m]
r_sq   = r_phys * r_phys

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "initial_field")
os.makedirs(OUT_DIR, exist_ok=True)

rng = random.Random(42)

# ── Build and write all four fields in a single pass ─────────────────────────
paths   = {n: open(os.path.join(OUT_DIR, f"{n}.field"), "w")
           for n in ("c", "phi", "eta", "mu")}

HEADER = ("# PhiX ScalarField - DAT\n"
          "# name: {name}\n"
          f"# nx {NX}  ny {NY}  nz {NZ}\n"
          "# x y z value\n")

for n, fh in paths.items():
    fh.write(HEADER.format(name=n))

print(f"Writing 4 fields ({NX}x{NY} = {NX*NY} cells each) ...", flush=True)

# PhiX readField loop order: k outer, j middle, i inner
for k in range(NZ):
    zv = 0.0
    for j in range(NY):
        yv = Y0 + (j + 0.5) * DY
        dy2 = (yv - cy) ** 2
        for i in range(NX):
            xv = X0 + (i + 0.5) * DX
            dist_sq = (xv - cx) ** 2 + dy2
            coord = f"{xv:.12e}  {yv:.12e}  {zv:.12e}"
            if dist_sq <= r_sq:
                noise = rng.uniform(-NOISE_AMP, NOISE_AMP)
                paths["c"]  .write(f"{coord}  {0.33 + noise:.12e}\n")
                paths["phi"].write(f"{coord}  {1.0:.12e}\n")
                paths["eta"].write(f"{coord}  {0.0:.12e}\n")
            else:
                paths["c"]  .write(f"{coord}  {0.25:.12e}\n")
                paths["phi"].write(f"{coord}  {0.0:.12e}\n")
                paths["eta"].write(f"{coord}  {1.0:.12e}\n")
            paths["mu"].write(f"{coord}  {0.0:.12e}\n")

for fh in paths.values():
    fh.close()

print("Done. Files written to:", OUT_DIR)
