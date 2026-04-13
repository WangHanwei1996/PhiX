"""
tutorials/quickstart/postProcess.py

Read PhiX .field snapshots from output/ and plot them.

Usage:
    python postProcess.py              # plot all snapshots, save to output/png/
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
                # scan for nx/ny/nz tokens
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

def plot_snapshot(path, out_dir, show=False):
    data, meta = read_field(path)

    # For 2-D fields squeeze the z dimension
    arr = np.squeeze(data)          # (ny, nx) when nz==1

    step = _step_from_path(path)
    name = meta.get("name", "phi")

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(arr, origin="lower", cmap="RdBu_r",
                   vmin=-1.0, vmax=1.0,
                   interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(name)
    cbar.locator = ticker.MaxNLocator(nbins=5)
    cbar.update_ticks()

    ax.set_title(f"{name}   step = {step}")
    ax.set_xlabel("x  [cells]")
    ax.set_ylabel("y  [cells]")
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"{name}_{step:07d}.png")
    fig.savefig(save_path, dpi=150)
    print(f"  saved: {save_path}")

    if show:
        plt.show()
    plt.close(fig)


def _step_from_path(path):
    """Extract the step number from e.g. output/phi_5000.field -> 5000."""
    m = re.search(r"_(\d+)\.field$", path)
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Post-process PhiX .field files")
    parser.add_argument("--step", type=int, default=None,
                        help="Plot a single step number (default: all)")
    parser.add_argument("--input-dir", default="output",
                        help="Directory containing .field files (default: output)")
    parser.add_argument("--output-dir", default="output/png",
                        help="Directory to write PNG files (default: output/png)")
    parser.add_argument("--show", action="store_true",
                        help="Display each figure interactively")
    args = parser.parse_args()

    # Collect files
    if args.step is not None:
        pattern = os.path.join(args.input_dir, f"*_{args.step}.field")
    else:
        pattern = os.path.join(args.input_dir, "*.field")

    files = sorted(glob.glob(pattern), key=_step_from_path)

    if not files:
        print(f"No .field files found matching: {pattern}")
        return

    print(f"Found {len(files)} snapshot(s). Plotting...")
    for path in files:
        plot_snapshot(path, args.output_dir, show=args.show)

    print("Done.")


if __name__ == "__main__":
    main()
