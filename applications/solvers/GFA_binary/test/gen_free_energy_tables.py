#!/usr/bin/env python3
"""
gen_free_energy_tables.py
生成 GFA_binary 使用的自由能查找表（.fetab 格式）

模型（stage 3：c 二次式 + 线性 T 依赖，测试用，见 doc/modeling_stage3.md）：
    f_L(c,T) = rho^2 * (c - ca)^2 + s_L * (T - T_m)     液相
    f_S(c,T) = rho^2 * (c - cb)^2 + s_S * (T - T_m)     固相

两相均随 T 线性变化；s_S > s_L，因此同一 c 下：
    f_S - f_L 含 (s_S - s_L)*(T - T_m) 项
    T > T_m → 固相被抬高 → 液相占优
    T < T_m → 固相被压低 → 固相占优
    T = T_m (= 配置中的 T_ref=1000 K) → T 项消失，退化为 stage 2 结果（对照基准）

线性函数在双线性插值下是精确的，nT 取多少都无误差；
这里取 nT=16 以顺带检验 T 方向的插值路径。

输出
----
    tables/fL.fetab
    tables/fS.fetab
"""

import os
import math

# ---------------------------------------------------------------------------
# 参数（与 test/settings/settings.jsonc 保持一致）
# ---------------------------------------------------------------------------
rho = math.sqrt(2.0)   # 1.4142135623730951
ca  = 0.3              # f_L 极小值（液相平衡浓度）
cb  = 0.7              # f_S 极小值（固相平衡浓度）

# T 依赖（线性测试式）
T_M = 1000.0           # [K] 两相 T 项的零点（“熔点”）；等于配置中的 T_ref
S_L = 2.0e-4           # f_L 的 T 斜率：T 跨 ±500 K 时贡献 ∓/±0.1
S_S = 6.0e-4           # f_S 的 T 斜率：差值斜率 s_S-s_L=4e-4 → ±500 K 时 ΔG=∓/±0.2

# 表格网格
NC    = 200            # c 方向节点数（足够细，覆盖 [0,1] 中的二次曲线）
NT    = 16             # T 方向节点数（线性式本身 nT=2 即精确，多取以测试插值）
C_MIN = 0.0
C_MAX = 1.0
T_MIN = 300.0          # 温度范围（K）
T_MAX = 1800.0

# ---------------------------------------------------------------------------
# 自由能函数
# ---------------------------------------------------------------------------
def fL(c, T):
    return rho**2 * (c - ca)**2 + S_L * (T - T_M)

def fS(c, T):
    return rho**2 * (c - cb)**2 + S_S * (T - T_M)

def dfLdc(c, T):
    # 解析 ∂f_L/∂c = 2*rho^2*(c - ca)（与 T 无关）；solver mu 的液相项 dfLdc*(1-h) 查此表
    return 2.0 * rho**2 * (c - ca)

def dfSdc(c, T):
    # 解析 ∂f_S/∂c = 2*rho^2*(c - cb)（与 T 无关）；solver mu 的固相项 dfSdc*h 查此表
    return 2.0 * rho**2 * (c - cb)

# ---------------------------------------------------------------------------
# 生成表格
# ---------------------------------------------------------------------------
def generate_table(func, c_min, c_max, nc, T_min, T_max, nT):
    """返回 nc×nT 的 row-major 列表（c 慢变，T 快变）"""
    data = []
    for ic in range(nc):
        c = c_min + ic * (c_max - c_min) / (nc - 1)
        row = []
        for iT in range(nT):
            T = T_min + iT * (T_max - T_min) / (nT - 1)
            row.append(func(c, T))
        data.append(row)
    return data

def write_fetab(path, header_comment, nc, nT, c_min, c_max, T_min, T_max, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(header_comment)
        f.write(f"# nc  nT  c_min  c_max   T_min   T_max\n")
        f.write(f"  {nc}  {nT}  {c_min}  {c_max}  {T_min}  {T_max}\n")
        for row in data:
            f.write("  " + "  ".join(f"{v: .10e}" for v in row) + "\n")
    print(f"Written: {path}  ({nc} × {nT} = {nc*nT} points)")

# ---------------------------------------------------------------------------
# 输出目录
# ---------------------------------------------------------------------------
out_dir = os.path.join(os.path.dirname(__file__), "tables")

# --- f_L ---
data_L = generate_table(fL, C_MIN, C_MAX, NC, T_MIN, T_MAX, NT)
write_fetab(
    os.path.join(out_dir, "fL.fetab"),
    f"""\
# ============================================================
# GFA_binary — 液相自由能表 f_L(c, T)   [stage 3: 线性 T 依赖测试式]
#
# 模型: f_L(c,T) = rho^2 * (c - ca)^2 + s_L * (T - T_m)
#   rho = sqrt(2) = {rho:.16f}
#   ca  = {ca}
#   s_L = {S_L}
#   T_m = {T_M} K
#
# nc={NC}  c in [{C_MIN}, {C_MAX}]
# nT={NT}  T in [{T_MIN}, {T_MAX}] K
# ============================================================
#
""",
    NC, NT, C_MIN, C_MAX, T_MIN, T_MAX, data_L
)

# --- f_S ---
data_S = generate_table(fS, C_MIN, C_MAX, NC, T_MIN, T_MAX, NT)
write_fetab(
    os.path.join(out_dir, "fS.fetab"),
    f"""\
# ============================================================
# GFA_binary — 固相自由能表 f_S(c, T)   [stage 3: 线性 T 依赖测试式]
#
# 模型: f_S(c,T) = rho^2 * (c - cb)^2 + s_S * (T - T_m)
#   rho = sqrt(2) = {rho:.16f}
#   cb  = {cb}
#   s_S = {S_S}
#   T_m = {T_M} K
#
# nc={NC}  c in [{C_MIN}, {C_MAX}]
# nT={NT}  T in [{T_MIN}, {T_MAX}] K
# ============================================================
#
""",
    NC, NT, C_MIN, C_MAX, T_MIN, T_MAX, data_S
)

# --- dfL/dc（解析 ∂f_L/∂c，供 stage-5 求解器 mu = dfLdc*(1-h) 查表）---
data_dLc = generate_table(dfLdc, C_MIN, C_MAX, NC, T_MIN, T_MAX, NT)
write_fetab(
    os.path.join(out_dir, "dfLdc.fetab"),
    f"""\
# ============================================================
# GFA_binary — 解析 ∂f_L/∂c 表   [stage 3: f_L 的 c 偏导]
#
# 模型: dfL/dc(c,T) = 2*rho^2*(c - ca) = {2.0*rho**2:.10f} * (c - {ca})
#   （与 T 无关 → nT 列全相同，双线性插值精确还原）
#   rho = sqrt(2) = {rho:.16f}
#   ca  = {ca}
#
# nc={NC}  c in [{C_MIN}, {C_MAX}]
# nT={NT}  T in [{T_MIN}, {T_MAX}] K
# ============================================================
#
""",
    NC, NT, C_MIN, C_MAX, T_MIN, T_MAX, data_dLc
)

# --- dfS/dc（解析 ∂f_S/∂c，供求解器 mu 的固相项 dfSdc*h 查表）---
data_dSc = generate_table(dfSdc, C_MIN, C_MAX, NC, T_MIN, T_MAX, NT)
write_fetab(
    os.path.join(out_dir, "dfSdc.fetab"),
    f"""\
# ============================================================
# GFA_binary — 解析 ∂f_S/∂c 表   [stage 3: f_S 的 c 偏导]
#
# 模型: dfS/dc(c,T) = 2*rho^2*(c - cb) = {2.0*rho**2:.10f} * (c - {cb})
#   （与 T 无关 → nT 列全相同，双线性插值精确还原）
#   rho = sqrt(2) = {rho:.16f}
#   cb  = {cb}
#
# nc={NC}  c in [{C_MIN}, {C_MAX}]
# nT={NT}  T in [{T_MIN}, {T_MAX}] K
# ============================================================
#
""",
    NC, NT, C_MIN, C_MAX, T_MIN, T_MAX, data_dSc
)

# ---------------------------------------------------------------------------
# 简单正确性检查
# ---------------------------------------------------------------------------
print()
print("正确性检查（表格节点值 vs 解析值）：")

# c 方向：T=T_MIN（首列）处
print(f"  f_L(c=0.0, T={T_MIN:.0f}) = {data_L[0][0]:.10e}  (ref: {fL(0.0, T_MIN):.10e})")
print(f"  f_S(c=1.0, T={T_MIN:.0f}) = {data_S[-1][0]:.10e}  (ref: {fS(1.0, T_MIN):.10e})")

# T 方向：固定 c=C_MIN（首行），检验线性 T 依赖
print()
print(f"T 依赖检查（c={C_MIN}，f_S-f_L 应随 T 线性变化，斜率 {S_S-S_L:.1e}/K）：")
for iT in (0, NT // 2, NT - 1):
    T = T_MIN + iT * (T_MAX - T_MIN) / (NT - 1)
    diff = data_S[0][iT] - data_L[0][iT]
    ref  = fS(C_MIN, T) - fL(C_MIN, T)
    print(f"  T={T:7.1f} K:  f_S-f_L = {diff:+.6f}  (ref: {ref:+.6f})")
print()
print("Done.")
