import struct, os, sys

def read_vti_scalar(path):
    with open(path, 'rb') as f:
        data = f.read()
    # find raw AppendedData
    idx = data.find(b'<AppendedData encoding="raw">')
    if idx >= 0:
        raw = data[idx + len(b'<AppendedData encoding="raw">'):]
    else:
        idx = data.find(b'_')
        raw = data[idx:]
    underscore = raw.find(b'_')
    raw = raw[underscore + 1:]
    n_bytes = struct.unpack_from('<I', raw, 0)[0]
    n = n_bytes // 8
    vals = struct.unpack_from('<' + 'd' * n, raw, 4)
    return vals

base = '/home/whw/PhiX/applications/solvers/SolidificationFeC_PhiX/2D/test/output'

for step in [0, 10, 20, 50, 100]:
    for field in ['phi_s', 'c']:
        path = f'{base}/{field}_{step}.vti'
        if not os.path.exists(path):
            continue
        try:
            v = read_vti_scalar(path)
            lo, hi = min(v), max(v)
            nz = sum(1 for x in v if x > 0.01)
            frac_mid = sum(1 for x in v if 0.01 < x < 0.99) / len(v)
            print(f'{field} step {step:3d}: min={lo:.5f} max={hi:.5f} nz={nz} mid_frac={frac_mid:.4f}')
        except Exception as e:
            print(f'{field} step {step}: ERROR {e}')
