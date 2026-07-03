#!/usr/bin/env python3
# Plot dfAmL(T) exactly as GFA_evo.cu computes it.
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- config values (test/settings/settings.jsonc) ---
R_gas = 8.314
alpha = 0.45
T_g   = 700.0
Vm    = 1.0580e-5

def f_tau(tau):
    tau = np.asarray(tau, dtype=float)
    lo = (1.0
          - 9.9167285e-1  * tau**(-1.0)
          - 1.11737779e-1 * tau**( 3.0)
          - 4.96612349e-3 * tau**( 9.0)
          - 1.11737779e-3 * tau**(15.0))
    hi = (- 1.05443689e-1 * tau**( -5.0)
          - 3.34741816e-3 * tau**(-15.0)
          - 7.02957924e-4 * tau**(-25.0))
    return np.where(tau < 1.0, lo, hi)

def dfAmL(T):
    return R_gas * T * np.log(1.0 + alpha) * f_tau(T / T_g) / Vm

T = np.linspace(400.0, 2000.0, 1600)
y = dfAmL(T)
ft = f_tau(T / T_g)

# reference points
T_tab_lo, T_tab_hi = 500.0, 2000.0   # table clamp range
T_start = 1100.0

fig, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

ax[0].plot(T, y, lw=2, color="C0")
ax[0].axhline(0, color="k", lw=0.8, ls=":")
ax[0].axvline(T_g, color="C3", lw=1.2, ls="--", label=f"T_g={T_g:.0f} K (tau=1, piecewise switch)")
ax[0].axvline(T_start, color="C2", lw=1.2, ls="--", label=f"T_start={T_start:.0f} K")
ax[0].axvspan(T_tab_lo, T_tab_hi, color="grey", alpha=0.08, label="table T-range [500,2000] (solver clamp)")
ax[0].set_ylabel(r"dfAmL  [J/m$^3$]")
ax[0].set_title(r"dfAmL(T) = $R\,T\,\ln(1+\alpha)\,f(\tau)/V_m$,   $\tau=T/T_g$")
ax[0].legend(fontsize=8, loc="best")
ax[0].grid(alpha=0.3)

ax[1].plot(T, ft, lw=2, color="C1")
ax[1].axhline(0, color="k", lw=0.8, ls=":")
ax[1].axvline(T_g, color="C3", lw=1.2, ls="--")
ax[1].set_xlabel("T  [K]")
ax[1].set_ylabel(r"$f(\tau)$  [-]")
ax[1].set_title(r"dimensionless $f(\tau)$")
ax[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig("/tmp/dfAmL.png", dpi=110)

# print a small table
print(f"{'T[K]':>7} {'tau':>7} {'f(tau)':>12} {'dfAmL[J/m3]':>15}")
for Ti in (400,500,600,694,700,800,900,1000,1100,1300,1521,1700,2000):
    tau = Ti/T_g
    print(f"{Ti:7.0f} {tau:7.3f} {float(f_tau(tau)):12.4e} {float(dfAmL(Ti)):15.4e}")
# zero crossing of f
from numpy import sign
idx = np.where(np.diff(sign(ft)))[0]
for i in idx:
    print(f"f(tau) sign change near T = {0.5*(T[i]+T[i+1]):.1f} K")
print("saved /tmp/dfAmL.png")
