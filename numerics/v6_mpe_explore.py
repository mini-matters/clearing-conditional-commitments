"""
v6 exploration: find the CONTESTED regime where the endogenous-attrition masking advantage is
large and robust (if it exists). The main v6_mpe.py calibration sits in the easy regime
(pi_succ ~ 1.0 in both regimes), where there is no room for masking to matter. Here we search
(xi, p_opp, T, theta, V1/kappa) for cells with pi_masked strictly interior AND a positive advantage,
and map the advantage surface. Imports the SAME exactly-solved MPE from v6_mpe.py.
"""

import numpy as np
from math import ceil
from v6_mpe import solve_masked, solve_revealed

N, kappa, Lp, rho = 10, 1.0, 3.0, 0.05


def adv_cell(N, theta, V1r, xi, p_opp, T):
    K = max(1, ceil((1 - theta) * N))
    V1 = V1r * kappa
    sm = solve_masked(N, K, V1, xi, kappa, Lp, T, rho, p_opp)
    sr = solve_revealed(N, K, V1, xi, kappa, Lp, T, rho, p_opp)
    ni = int(np.sum((sr["W"] > 1e-6) & (sr["W"] < 1 - 1e-6)))
    return sm["pi_succ"], sr["pi_succ"], sm["pi_succ"] - sr["pi_succ"], ni


print(
    "=== v6 exploration: search for a CONTESTED regime with a robust masking advantage ==="
)
print(f"N={N} kappa={kappa} Lp={Lp} rho={rho}\n")

# 1. Grid search for large advantage with contested masked completion
cells = []
for xi in [0.5, 0.65, 0.8, 0.9, 0.95]:
    for p_opp in [0.2, 0.3, 0.4, 0.6]:
        for T in [3, 4, 5, 6, 8]:
            for theta in [0.15, 0.25, 0.35, 0.5]:
                for V1r in [1.5, 2.0, 2.5, 3.0, 4.0]:
                    pm, pr, a, ni = adv_cell(N, theta, V1r, xi, p_opp, T)
                    cells.append((a, pm, pr, ni, xi, p_opp, T, theta, V1r))

cells.sort(reverse=True)
print("--- TOP 15 cells by masking advantage (pi_masked - pi_revealed) ---")
print(
    f"{'adv':>7} {'pi_m':>6} {'pi_r':>6} {'#WoA':>5} {'xi':>5} {'popp':>5} {'T':>3} {'theta':>6} {'V1/k':>5}"
)
for a, pm, pr, ni, xi, p_opp, T, theta, V1r in cells[:15]:
    print(
        f"{a:+7.3f} {pm:6.3f} {pr:6.3f} {ni:5d} {xi:5.2f} {p_opp:5.2f} {T:3d} {theta:6.2f} {V1r:5.1f}"
    )

# contested = masked completion strictly interior
contested = [c for c in cells if 0.15 < c[1] < 0.97 and c[0] > 0.03]
print(
    f"\n--- cells that are CONTESTED (0.15<pi_m<0.97) AND advantage>0.03: {len(contested)} of {len(cells)} ---"
)
for a, pm, pr, ni, xi, p_opp, T, theta, V1r in sorted(contested, reverse=True)[:12]:
    print(
        f"{a:+7.3f} {pm:6.3f} {pr:6.3f} {ni:5d} {xi:5.2f} {p_opp:5.2f} {T:3d} {theta:6.2f} {V1r:5.1f}"
    )

# 2. If a good contested cell exists, map advantage vs T at that (xi,p_opp,theta,V1r)
if contested:
    best = sorted(contested, reverse=True)[0]
    _, _, _, _, xi, p_opp, T0, theta, V1r = best
    print(
        f"\n--- advantage vs HORIZON T at the best contested cell "
        f"(xi={xi}, p_opp={p_opp}, theta={theta}, V1/k={V1r}) ---"
    )
    print(f"{'T':>3} {'pi_m':>7} {'pi_r':>7} {'adv':>8} {'#WoA':>5}")
    for T in [2, 3, 4, 5, 6, 8, 10, 14]:
        pm, pr, a, ni = adv_cell(N, theta, V1r, xi, p_opp, T)
        print(f"{T:3d} {pm:7.4f} {pr:7.4f} {a:+8.4f} {ni:5d}")

    print(
        f"\n--- advantage vs xi (non-excludability) at (p_opp={p_opp}, theta={theta}, V1/k={V1r}, T={T0}) ---"
    )
    print(f"{'xi':>5} {'pi_m':>7} {'pi_r':>7} {'adv':>8} {'#WoA':>5}")
    for xi_v in [0.0, 0.3, 0.5, 0.65, 0.8, 0.9, 0.95, 0.99]:
        pm, pr, a, ni = adv_cell(N, theta, V1r, xi_v, p_opp, T0)
        print(f"{xi_v:5.2f} {pm:7.4f} {pr:7.4f} {a:+8.4f} {ni:5d}")

# 3. Headline: max advantage over the WHOLE grid, and whether it is contested or a deadline corner
amax, pm, pr, ni, xi, p_opp, T, theta, V1r = cells[0]
print("\n=== HEADLINE ===")
print(
    f"max masking advantage over grid = {amax:+.4f} at "
    f"xi={xi}, p_opp={p_opp}, T={T}, theta={theta}, V1/k={V1r} (pi_m={pm:.3f}, pi_r={pr:.3f}, #WoA={ni})"
)
n_big = sum(1 for c in cells if c[0] > 0.05)
n_contested_big = len(contested)
print(
    f"cells with advantage>0.05: {n_big}/{len(cells)}; contested+advantage>0.03: {n_contested_big}"
)
print(
    "interpretation: if the big-advantage cells are all deadline corners (T tiny / pi_m extreme),"
)
print(
    "the endogenized attrition has DEFLATED v5's exogenous-dial advantage to a thin boundary effect."
)
