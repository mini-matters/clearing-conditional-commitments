"""Re-verify (1) the C2 counterexample, (2) anchor T=10 decoupled corner,
(3) A4 arithmetic at the anchor, (4) R2 slope/threshold numbers. Throwaway check script."""

from math import comb

import numpy as np

import v9_dynamic_two_channel as v9
import v9_verify_s11 as vs11


def betas(N, K, p):
    pmf = [comb(N - 1, j) * p**j * (1 - p) ** (N - 1 - j) for j in range(N)]
    b1 = pmf[K - 1]
    bplus = sum(pmf[K:])
    bminus = sum(pmf[: K - 1])
    return b1, bplus, bminus


def check_axioms(N, K, p_opp, V, kappa, xi, pi_b, L, Lp, rho, T):
    E = (1 - pi_b) * V - pi_b * L
    eps = xi * V - (V - kappa)
    b1, bp, bm = betas(N, K, p_opp)
    a4_lhs = (V - kappa) * b1
    a4_rhs = eps * bp + Lp * bm
    print(f"  E={E:.4f}  A1 (0<=E<kappa={kappa}): {0 <= E < kappa}")
    print(f"  eps={eps:.5f}  A2 (eps>0): {eps > 0}")
    print(f"  A3 (K-1={K - 1} >= p_opp*N={p_opp * N:.3f}): {K - 1 >= p_opp * N}")
    print(
        f"  A4: LHS=(V-k)*b1={a4_lhs:.4f}  RHS=eps*b+ + Lp*b- ="
        f" {eps:.5f}*{bp:.4f}+{Lp}*{bm:.4f}={a4_rhs:.4f}  holds: {a4_lhs > a4_rhs}"
    )
    a5_bound = min(Lp, kappa - E)
    print(
        f"  A5: rho*(T-1)={rho * (T - 1):.4f} < min(Lp,kappa-E)={a5_bound:.4f}:"
        f" {rho * (T - 1) < a5_bound}"
    )
    return b1, bp, bm


def Gs_dec(kt, kp, N, K, pi_b, V, L, xi, kappa, Lp, T, rho, p_opp):
    """G, s, W decoupled via the ground-truth solver."""
    free_ride = xi * V > (V - kappa)
    pcg, pcg_nc = v9.type_perceived(kt, pi_b, V - kappa, xi * V, V, L, xi, kappa)
    G = v9.solve_unified(N, K, pcg, pcg_nc, kappa, Lp, T, rho, p_opp, kp, free_ride)[
        "pi_clear"
    ]
    pcb, pcb_nc = v9.type_perceived(kt, pi_b, -L - kappa, xi * (-L), V, L, xi, kappa)
    s = (
        1
        - v9.solve_unified(N, K, pcb, pcb_nc, kappa, Lp, T, rho, p_opp, kp, free_ride)[
            "pi_clear"
        ]
    )
    W = (1 - pi_b) * G * V - pi_b * (1 - s) * L
    return G, s, W


print("=" * 90)
print("1. C2 COUNTEREXAMPLE  (N,K,p_opp,V,kappa,xi,pi_b,L,Lp,rho)")
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
print(f"   params: {P}")
check_axioms(
    P["N"],
    P["K"],
    P["p_opp"],
    P["V"],
    P["kappa"],
    P["xi"],
    P["pi_b"],
    P["L"],
    P["Lp"],
    P["rho"],
    T=10,
)

for T in (2, 5, 10):
    G10, s10, W10 = Gs_dec(
        1.0,
        0.0,
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
    G12, s12, W12 = Gs_dec(
        1.0,
        0.2,
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
    # full k_piv = 0 line
    line = []
    for kt in np.linspace(0, 1, 21):
        _, sL, wL = Gs_dec(
            float(kt),
            0.0,
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
        line.append((float(kt), wL, sL))
    kbest, wbest, _ = max(line, key=lambda t: t[1])
    w09 = [w for kt, w, _ in line if abs(kt - 0.9) < 1e-9][0]
    print(
        f"  T={T:2d}: W(1,0.2)={W12:.4f} (G={G12:.4f},s={s12:.4f})"
        f"   W(1,0)={W10:.4f} (G={G10:.4f},s={s10:.4f})"
        f"   max k_piv=0 line: W={wbest:.4f} at k_type={kbest:.2f}"
        f"   W(0.9,0)={w09:.4f}"
    )
    print(
        f"        C2 violated (W(1,0.2) > max-line): {W12 > wbest};"
        f"  (1,0) not line-argmax: {wbest > W10 + 1e-9}"
    )

# cross-check vs independent reimplementation
fr_gate = 1.0 if P["xi"] * P["V"] > (P["V"] - P["kappa"]) else 0.0


def fr_rule(kp, piv_w):
    return kp * (1 - piv_w) * fr_gate


for kt, kp in [(1.0, 0.0), (1.0, 0.2)]:
    Wv, Gv, sv = vs11.W_general(
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
        10,
        P["rho"],
        P["p_opp"],
        lambda k: k,
        fr_rule,
    )
    print(
        f"  v9_verify_s11.W_general T=10 ({kt},{kp}): W={Wv:.5f} G={Gv:.5f} s={sv:.5f}"
    )

print("=" * 90)
print("2. ANCHOR CALIBRATION T=10: decoupled corner re-verify (best_2d 11x11)")
A = dict(
    N=10, K=7, kappa=1.0, Lp=3.0, rho=0.05, p_opp=0.6, V=3.0, L=3.0, T=10, pi_b=0.2
)
for xi in (0.667, 0.70, 0.75):
    (kt, kp), wstar = v9.best_2d(
        A["N"],
        A["K"],
        A["pi_b"],
        A["V"],
        A["L"],
        xi,
        A["kappa"],
        A["Lp"],
        A["T"],
        A["rho"],
        A["p_opp"],
        grid=11,
    )
    print(f"  xi={xi:.3f}: argmax=({kt:.2f},{kp:.2f})  W*={wstar:+.4f}")

print("=" * 90)
print("3. A4 AT THE ANCHOR (xi=0.667, Lp=3):")
check_axioms(
    A["N"],
    A["K"],
    A["p_opp"],
    A["V"],
    A["kappa"],
    0.667,
    A["pi_b"],
    A["L"],
    A["Lp"],
    A["rho"],
    A["T"],
)
print("   non-vacuity instance (pi_b=0.5, Lp=0.5):")
check_axioms(10, 7, 0.6, 3, 1, 0.667, 0.5, 3, 0.5, 0.0, 1)

print("=" * 90)
print("4. R2 SLOPE / kbar  (N,K,p,V,kappa,xi,pi_b,L,Lp)=(10,7,0.6,3,1,0.667,0.5,3,0.5)")
b1, bp, bm = betas(10, 7, 0.6)
# D1(k) = Pc(k)*b1 - (Pn(k)-Pc(k))*b+ - Lp*b-,  Pc=-1+3k, Pn=2.001k
xi, V, kappa, Lp = 0.667, 3.0, 1.0, 0.5
E = 0.0
slope = (
    (V - kappa - (E - kappa)) * b1 + (1 - xi) * V * bp - ((xi * E) - (E - kappa)) * 0
)  # recompute directly below
D1_0 = (E - kappa) * b1 - (xi * E - (E - kappa)) * bp - Lp * bm
D1_1 = (V - kappa) * b1 - (xi * V - (V - kappa)) * bp - Lp * bm
slope = D1_1 - D1_0
print(f"  b1={b1:.6f} b+={bp:.6f} b-={bm:.6f}")
print(
    f"  D1(0)={D1_0:.6f}  D1(1)={D1_1:.6f}  slope={slope:.6f}  kbar={-D1_0 / slope:.6f}"
)

print("   R2 T=1 sanity: G on diagonal, decoupled argmax")
for k in (0.75, 0.76, 0.9, 1.0):
    G, s, W = Gs_dec(k, k, 10, 7, 0.5, 3, 3, 0.667, 1, 0.5, 1, 0.0, 0.6)
    print(f"   k={k}: G={G:.3e} s={s:.4f} W={W:.5f}")
(kt, kp), wstar = v9.best_2d(10, 7, 0.5, 3, 3, 0.667, 1, 0.5, 1, 0.0, 0.6, grid=11)
print(f"   decoupled 11x11 T=1: argmax=({kt:.2f},{kp:.2f}) W*={wstar:.5f}")
Fp = sum(comb(10, j) * 0.6**j * 0.4 ** (10 - j) for j in range(7, 11))
print(f"   (1-pi_b)*V*F(p_opp) = {0.5 * 3 * Fp:.5f}")
