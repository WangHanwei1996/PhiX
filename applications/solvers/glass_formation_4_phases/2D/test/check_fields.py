#!/usr/bin/env python3
import array, os
out = "output"
step = 10000
for name in ['phi0','phi1','phi2','phi3','eta','c']:
    path = f"{out}/{name}_{step}.field"
    if not os.path.exists(path):
        print(f"{name}: file not found")
        continue
    raw = open(path, "rb").read()
    n = len(raw) // 8
    a = array.array("d")
    a.frombytes(raw)
    mn, mx, s = min(a), max(a), sum(a)
    print(f"{name:5s}: n={n:6d}  min={mn: .4f}  max={mx: .4f}  mean={s/n: .6f}")
