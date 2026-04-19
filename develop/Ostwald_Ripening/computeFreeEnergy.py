"""
develop/Ostwald_Ripening/computeFreeEnergy.py

Compute the total free energy F at every saved snapshot and
write a two-column DAT file  (step  F).

    F = ∫_V [ f_chem(c,η) + (κ_c/2)|∇c|² + (κ_η/2)|∇η|² ] dV

    f_chem(c,η) = f^α(c)[1 - h(η)] + f^β(c) h(η) + w g(η)

    f^α(c) = ρ²(c - c_α)²
    f^β(c) = ρ²(c_β - c)²
    h(η)   = η³(6η² - 15η + 10)
    g(η)   = η²(1 - η)²

Parameters (matching AC-CH.cu):
    ρ = √2,  c_α = 0.3,  c_β = 0.7,
    κ_c = 3,  κ_η = 3,  w = 1
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
# Physical parameters  (keep in sync with AC-CH.cu)
# ---------------------------------------------------------------------------
RHO     = np.sqrt(2.0)
C_ALPHA = 0.3
C_BETA  = 0.7
KAPPA_C = 3.0
KAPPA_ETA = 3.0
W       = 1.0
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

def free_energy(c2d, eta2d, dx=DX, dy=DY,
                rho=RHO, c_alpha=C_ALPHA, c_beta=C_BETA,
                kappa_c=KAPPA_C, kappa_eta=KAPPA_ETA, w=W):
    """
    Compute total free energy F on a 2-D periodic domain.

    Gradient is evaluated with 2nd-order central differences (periodic wrap).
    """
    # Interpolation functions
    h = eta2d**3 * (6.0 * eta2d**2 - 15.0 * eta2d + 10.0)
    g = eta2d**2 * (1.0 - eta2d)**2

    # Parabolic free energies
    f_alpha = rho**2 * (c2d - c_alpha)**2
    f_beta  = rho**2 * (c_beta - c2d)**2

    # Chemical free-energy density
    f_chem = f_alpha * (1.0 - h) + f_beta * h + w * g

    # Gradient of c via periodic central differences
    dcdx = (np.roll(c2d, -1, axis=1) - np.roll(c2d, 1, axis=1)) / (2.0 * dx)
    dcdy = (np.roll(c2d, -1, axis=0) - np.roll(c2d, 1, axis=0)) / (2.0 * dy)
    grad_c_sq = dcdx**2 + dcdy**2

    # Gradient of eta via periodic central differences
    detadx = (np.roll(eta2d, -1, axis=1) - np.roll(eta2d, 1, axis=1)) / (2.0 * dx)
    detady = (np.roll(eta2d, -1, axis=0) - np.roll(eta2d, 1, axis=0)) / (2.0 * dy)
    grad_eta_sq = detadx**2 + detady**2

    # Total local energy density
    f_total = f_chem + 0.5 * kappa_c * grad_c_sq + 0.5 * kappa_eta * grad_eta_sq

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
        description="Compute Allen-Cahn/Cahn-Hilliard free energy from PhiX .field files")
    parser.add_argument("--step", type=int, default=None,
                        help="Process a single step number (default: all)")
    parser.add_argument("--input-dir", default="output",
                        help="Directory containing .field files (default: output)")
    parser.add_argument("--output", default="output/free_energy.dat",
                        help="Output DAT file (default: output/free_energy.dat)")
    args = parser.parse_args()

    if args.step is not None:
        c_pattern   = os.path.join(args.input_dir, f"c_{args.step}.field")
        eta_pattern = os.path.join(args.input_dir, f"eta_{args.step}.field")
    else:
        c_pattern   = os.path.join(args.input_dir, "c_*.field")
        eta_pattern = os.path.join(args.input_dir, "eta_*.field")

    c_files   = sorted(glob.glob(c_pattern),   key=_step_from_path)
    eta_files = sorted(glob.glob(eta_pattern), key=_step_from_path)

    # Build step -> path maps and find common steps
    c_map   = {_step_from_path(p): p for p in c_files}
    eta_map = {_step_from_path(p): p for p in eta_files}
    common_steps = sorted(set(c_map.keys()) & set(eta_map.keys()))

    if not common_steps:
        print(f"No matching c/eta .field pairs found in: {args.input_dir}")
        return

    print(f"Found {len(common_steps)} snapshot(s). Computing free energy...")

    results = []
    for step in common_steps:
        c2d, _   = read_field(c_map[step])
        eta2d, _ = read_field(eta_map[step])
        F = free_energy(c2d, eta2d)
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
