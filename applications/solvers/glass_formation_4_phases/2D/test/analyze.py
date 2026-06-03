import numpy as np

def read_field(fname):
    # .dat format: "# x y z value" header, then rows "x y z value"
    # Only read the last (4th) column = value
    vals = []
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            vals.append(float(parts[-1]))  # last column = value
    return np.array(vals)

fname_template = 'output/{field}_{step}.dat'

for step in list(range(0, 26)):
    try:
        phi0 = read_field(f'output/phi0_{step}.dat')
        phi1 = read_field(f'output/phi1_{step}.dat')
        phi2 = read_field(f'output/phi2_{step}.dat')
        phi3 = read_field(f'output/phi3_{step}.dat')
        c    = read_field(f'output/c_{step}.dat')
        s = phi0 + phi1 + phi2 + phi3
        nan_any = np.isnan(phi0).any() or np.isnan(phi1).any() or np.isnan(phi2).any() or np.isnan(phi3).any() or np.isnan(c).any()
        print(f"step={step}: phi0=[{phi0.min():.4f},{phi0.max():.4f}]  phi1=[{phi1.min():.4f},{phi1.max():.4f}]  phi2=[{phi2.min():.4f},{phi2.max():.4f}]  phi3=[{phi3.min():.4f},{phi3.max():.4f}]  c=[{c.min():.4f},{c.max():.4f}]  sum=[{s.min():.6f},{s.max():.6f}]  NaN={nan_any}")
    except Exception as e:
        print(f"step={step}: ERROR: {e}")
