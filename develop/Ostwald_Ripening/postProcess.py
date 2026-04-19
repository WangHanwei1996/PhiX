"""
develop/Ostwald_Ripening/postProcess.py

Read PhiX .field snapshots from output/ and plot concentration c and
order-parameter eta side by side.

Usage:
    python postProcess.py              # plot all matched pairs, save to output/png/
    python postProcess.py --show       # also display interactively
    python postProcess.py --step 5000  # plot a single step
"""

import argparse
import glob
import os
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def read_field(path):
    """Read a PhiX .field file. Returns (data, meta) where data is ndarray
    shaped (nz, ny, nx) and meta is a dict with name/nx/ny/nz/ghost."""
    with open(path, "rb") as f:
        meta = {}
        for raw in f:
            line = raw.decode("utf-8", errors="replace").rstrip()
            if line == "---":
                break
            if line.startswith("#") or not line.strip():
                continue
            key, *rest = line.split()
            if key == "name":
                meta["name"] = rest[0] if rest else ""
            elif key == "ghost":
                meta["ghost"] = int(rest[0])
            else:
                tokens = line.split()
                for i, tok in enumerate(tokens):
                    if tok in ("nx", "ny", "nz") and i + 1 < len(tokens):
                        meta[tok] = int(tokens[i + 1])

        nx = meta.get("nx", 1)
        ny = meta.get("ny", 1)
        nz = meta.get("nz", 1)

        raw_bytes = f.read()
        data = np.frombuffer(raw_bytes, dtype=np.float64)

    data = data.reshape((nz, ny, nx))
    return data, meta


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_snapshot(c_path, eta_path, out_dir, show=False):
    c_data,   c_meta   = read_field(c_path)
    eta_data, eta_meta = read_field(eta_path)

    c_arr   = np.squeeze(c_data)    # (ny, nx)
    eta_arr = np.squeeze(eta_data)  # (ny, nx)

    step = _step_from_path(c_path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # ---- c field ----
    if step == 0:
        vmin_c, vmax_c = c_arr.min(), c_arr.max()
        c_range = f"adaptive [{vmin_c:.4f}, {vmax_c:.4f}]"
    else:
        vmin_c, vmax_c = 0.0, 1.0
        c_range = "fixed [0, 1]"

    im0 = axes[0].imshow(c_arr, origin="lower", cmap="coolwarm",
                         vmin=vmin_c, vmax=vmax_c, interpolation="nearest")
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    # cbar0.set_label(f"c")
    cbar0.locator = ticker.MaxNLocator(nbins=5)
    cbar0.update_ticks()
    axes[0].set_title("Concentration  c")
    axes[0].set_xlabel("x  [cells]")
    axes[0].set_ylabel("y  [cells]")

    # ---- eta field ----
    if step == 0:
        vmin_e, vmax_e = eta_arr.min(), eta_arr.max()
        e_range = f"adaptive [{vmin_e:.4f}, {vmax_e:.4f}]"
    else:
        vmin_e, vmax_e = 0.0, 1.0
        e_range = "fixed [0, 1]"

    im1 = axes[1].imshow(eta_arr, origin="lower", cmap="coolwarm",
                         vmin=vmin_e, vmax=vmax_e, interpolation="nearest")
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    # cbar1.set_label(f"eta")
    cbar1.locator = ticker.MaxNLocator(nbins=5)
    cbar1.update_ticks()
    axes[1].set_title("Order parameter  η")
    axes[1].set_xlabel("x  [cells]")
    axes[1].set_ylabel("y  [cells]")

    fig.suptitle(f"Allen-Cahn / Cahn-Hilliard  |  step = {step}", fontsize=13)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"snapshot_{step:08d}.png")
    fig.savefig(save_path, dpi=150)
    print(f"  saved: {save_path}")

    if show:
        plt.show()
    plt.close(fig)


def _step_from_path(path):
    """Extract the step number from e.g. output/c_5000.field -> 5000."""
    m = re.search(r"_(\d+)\.field$", path)
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Post-process Allen-Cahn / Cahn-Hilliard PhiX .field files")
    parser.add_argument("--step", type=int, default=None,
                        help="Plot a single step number (default: all)")
    parser.add_argument("--input-dir", default="output",
                        help="Directory containing .field files (default: output)")
    parser.add_argument("--output-dir", default="output/png",
                        help="Directory to write PNG files (default: output/png)")
    parser.add_argument("--show", action="store_true",
                        help="Display each figure interactively")
    args = parser.parse_args()

    if args.step is not None:
        c_files = sorted(glob.glob(
            os.path.join(args.input_dir, f"c_{args.step}.field")))
    else:
        c_files = sorted(
            glob.glob(os.path.join(args.input_dir, "c_*.field")),
            key=_step_from_path)

    if not c_files:
        print(f"No c_*.field files found in: {args.input_dir}")
        return

    print(f"Found {len(c_files)} c snapshot(s). Plotting...")
    missing = 0
    for c_path in c_files:
        step = _step_from_path(c_path)
        eta_path = os.path.join(args.input_dir, f"eta_{step}.field")
        if not os.path.exists(eta_path):
            print(f"  [skip] step {step}: eta file not found ({eta_path})")
            missing += 1
            continue
        plot_snapshot(c_path, eta_path, args.output_dir, show=args.show)

    if missing:
        print(f"  ({missing} step(s) skipped due to missing eta file)")
    print("Done.")


if __name__ == "__main__":
    main()
