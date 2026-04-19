"""
develop/Spinodal Decomposition/computeFreeEnergy.py

Compute the total Cahn-Hilliard free energy F at every saved snapshot and
write a two-column DAT file  (step  F).

    F = ∫_V [ f_chem(c) + (κ/2)|∇c|² ] dV
    f_chem(c) = ρ_s (c - c_α)² (c_β - c)²

Parameters (matching Cahn-Hillard.cu):
    ρ_s  = 5,  c_α = 0.3,  c_β = 0.7,  κ = 2
    dx = dy = 1.0  (200 × 200 uniform grid)

Usage:
    python computeFreeEnergy.py                   # all snapshots -> output/free_energy.dat
    python computeFreeEnergy.py --step 5000       # single step
    python computeFreeEnergy.py --input-dir output --output output/free_energy.dat
"""

import argparse
import glob
import os
import re

import numpy as np


# ---------------------------------------------------------------------------
# Physical parameters  (keep in sync with Cahn-Hillard.cu)
# ---------------------------------------------------------------------------
RHO_S   = 5.0
C_ALPHA = 0.3
C_BETA  = 0.7
KAPPA   = 2.0
DX      = 1.0
DY      = 1.0


# ---------------------------------------------------------------------------
# Field reader  (same logic as postProcess.py)
# ---------------------------------------------------------------------------

def read_field(path):
    """Return (data_2d, meta).  data_2d has shape (ny, nx)."""
    with open(path, "rb") as f:
        meta = {}
        for raw in f:
            line = raw.decode("utf-8", errors="replace").rstrip()
            if line == "---":
                break
            if line.startswith("#") or not line.strip():
                continue
            tokens = line.split()
            # Scan every token on the line so that a line like
            # "nx 200  ny 200  nz 1" is fully captured.
            for i, tok in enumerate(tokens):
                if tok == "name" and i + 1 < len(tokens):
                    meta["name"] = tokens[i + 1]
                elif tok in ("nx", "ny", "nz", "ghost") and i + 1 < len(tokens):
                    meta[tok] = int(tokens[i + 1])

        nx = meta.get("nx", 1)
        ny = meta.get("ny", 1)
        nz = meta.get("nz", 1)

        raw_bytes = f.read()
        data = np.frombuffer(raw_bytes, dtype=np.float64).reshape((nz, ny, nx))

    return np.squeeze(data), meta   # (ny, nx) for 2-D


# ---------------------------------------------------------------------------
# Free-energy calculation
# ---------------------------------------------------------------------------

def free_energy(c2d, dx=DX, dy=DY,
                rho_s=RHO_S, c_alpha=C_ALPHA, c_beta=C_BETA, kappa=KAPPA):
    """
    Compute total free energy F on a 2-D periodic domain.

    Gradient is evaluated with 2nd-order central differences (periodic wrap).
    """
    # Chemical free-energy density
    f_chem = rho_s * (c2d - c_alpha)**2 * (c_beta - c2d)**2

    # Gradient via periodic central differences
    # np.roll wraps around => implements periodic BC
    dcdx = (np.roll(c2d, -1, axis=1) - np.roll(c2d, 1, axis=1)) / (2.0 * dx)
    dcdy = (np.roll(c2d, -1, axis=0) - np.roll(c2d, 1, axis=0)) / (2.0 * dy)

    grad_sq = dcdx**2 + dcdy**2

    f_total = f_chem + 0.5 * kappa * grad_sq   # local energy density

    # Integrate: F = sum(f * dV),  dV = dx * dy (2-D: dz = 1)
    F = np.sum(f_total) * dx * dy

    return F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step_from_path(path):
    m = re.search(r"_(\d+)\.field$", path)
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute Cahn-Hilliard free energy from PhiX .field files")
    parser.add_argument("--step", type=int, default=None,
                        help="Process a single step number (default: all)")
    parser.add_argument("--input-dir", default="output",
                        help="Directory containing .field files (default: output)")
    parser.add_argument("--output", default="output/free_energy.dat",
                        help="Output DAT file (default: output/free_energy.dat)")
    args = parser.parse_args()

    if args.step is not None:
        pattern = os.path.join(args.input_dir, f"*_{args.step}.field")
    else:
        pattern = os.path.join(args.input_dir, "*.field")

    files = sorted(glob.glob(pattern), key=_step_from_path)

    if not files:
        print(f"No .field files found matching: {pattern}")
        return

    print(f"Found {len(files)} snapshot(s). Computing free energy...")

    results = []
    for path in files:
        step = _step_from_path(path)
        c2d, _ = read_field(path)
        F = free_energy(c2d)
        results.append((step, F))
        print(f"  step {step:>10d}   F = {F:.6e}")

    # Write DAT file
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as fout:
        fout.write("# step  F\n")
        for step, F in results:
            fout.write(f"{step:10d}  {F:.10e}\n")

    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
