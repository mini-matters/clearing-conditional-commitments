"""Finer 2-D scan at the C2 counterexample params: where is the true argmax?"""

import numpy as np

import v9_dynamic_two_channel as v9

P = dict(
    N=8,
    K=4,
    p_opp=0.29,
    V=4.84,
    kappa=2.00,
    xi=0.958,
    pi_b=0.32,
    L=9.625,
    Lp=0.408,
    rho=0.0126,
)


def Wdec(kt, kp, T):
    return v9.G_and_s_decoupled(
        kt,
        kp,
        P["N"],
        P["K"],
        P["pi_b"],
        P["V"],
        P["L"],
        P["xi"],
        P["kappa"],
        P["Lp"],
        T,
        P["rho"],
        P["p_opp"],
    )


for T in (2, 10):
    print(f"--- T={T}: 11-pt k_piv=0 line (the hunter's likely grid) ---")
    vals = [(kt, Wdec(float(kt), 0.0, T)) for kt in np.linspace(0, 1, 11)]
    for kt, w in vals:
        print(f"   k_type={kt:.1f}  W={w:.4f}")
    print(f"   11-pt line max = {max(w for _, w in vals):.4f}")

print("--- T=10: 21x21 grid over [0,1]^2 ---")
best, arg = -1e18, None
for kt in np.linspace(0, 1, 21):
    for kp in np.linspace(0, 1, 21):
        w = Wdec(float(kt), float(kp), 10)
        if w > best:
            best, arg = w, (float(kt), float(kp))
print(f"   21x21 argmax = {arg}  W = {best:.4f}")

print("--- T=10: refine near argmax (step 0.0125) ---")
kt0, kp0 = arg
best2, arg2 = -1e18, None
for kt in np.arange(max(0, kt0 - 0.05), min(1, kt0 + 0.05) + 1e-9, 0.0125):
    for kp in np.arange(max(0, kp0 - 0.05), min(1, kp0 + 0.05) + 1e-9, 0.0125):
        w = Wdec(float(kt), float(kp), 10)
        if w > best2:
            best2, arg2 = w, (round(float(kt), 4), round(float(kp), 4))
print(f"   refined argmax = {arg2}  W = {best2:.4f}")

print("--- T=10: fine k_piv=0 line, 41 pts, for the true line max ---")
vals = [(float(kt), Wdec(float(kt), 0.0, 10)) for kt in np.linspace(0, 1, 41)]
ktb, wb = max(vals, key=lambda t: t[1])
print(f"   41-pt line max = {wb:.4f} at k_type={ktb:.3f}")
print("--- compare: best over k_piv>0 column at that k_type ---")
for kp in (0.05, 0.1, 0.15, 0.2, 0.3):
    print(f"   W({ktb:.3f},{kp})={Wdec(ktb, kp, 10):.4f}")
