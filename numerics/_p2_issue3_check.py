"""Re-derive P2 R3 anchor: diagonal stage margins vc-vw at theta=0.30 (K=7),
under (a) the script's 51-point quadrature rule and (b) the exact Laplacian integral.
Exact q_j(n) = (1/(n p)) P(Binom(n,p) >= j+1)  [Lemma 1 / Beta-Binomial identity].
"""

from math import ceil

import numpy as np
from v9_selection_lambdafree import binom_pmf, laplacian_vals, propagate


def q_exact(n, p):
    # q_j for j=0..n-1: (1/(n p)) * P(Binom(n,p) >= j+1)
    pmf = binom_pmf(n, p)
    tail = np.cumsum(pmf[::-1])[::-1]  # tail[k] = P(X >= k)
    return np.array([tail[j + 1] / (n * p) for j in range(n)])


def laplacian_vals_exact(m, h, n, V1, xi, kappa, Lp, rho, p_opp, K, Pc, Vu):
    q = q_exact(n, p_opp)
    vc, vw = -rho, -rho
    for j in range(n):
        wj = q[j]
        mpc = m + 1 + j
        if mpc >= K:
            vc += wj * (V1 - kappa)
        else:
            pc = Pc[mpc, h - 1]
            vc += wj * ((V1 - kappa) * pc + (-Lp) * (1 - pc))
        mp = m + j
        if mp >= K:
            vw += wj * (xi * V1)
        elif h - 1 == 0:
            vw += wj * 0.0
        else:
            vw += wj * Vu[mp, h - 1]
    return vc, vw


def solve(N, K, V1, xi, kappa, Lp, T, rho, p_opp, exact):
    Vu = np.zeros((N + 1, T + 1))
    Pc = np.zeros((N + 1, T + 1))
    A = np.zeros((N + 1, T + 1))
    M = np.full((N + 1, T + 1), np.nan)  # stage margin vc-vw
    for m in range(N + 1):
        for h in range(T + 1):
            if m >= K:
                Pc[m, h] = 1.0
    for h in range(1, T + 1):
        for m in range(N):
            if m >= K:
                continue
            n = N - m
            if exact:
                vc, vw = laplacian_vals_exact(
                    m, h, n, V1, xi, kappa, Lp, rho, p_opp, K, Pc, Vu
                )
            else:
                vc, vw = laplacian_vals(
                    m, h, n, V1, xi, kappa, Lp, rho, p_opp, K, Pc, Vu
                )
            a = 1.0 if vc > vw else 0.0
            A[m, h] = a
            M[m, h] = vc - vw
            propagate(m, h, n, a, V1, xi, kappa, Lp, rho, p_opp, K, Pc, Vu)
    return A, M, Pc


def main():
    N, kappa, Lp, rho, xi, p_opp, V1, T = 10, 1.0, 3.0, 0.05, 0.8, 0.6, 3.0, 10
    for theta in [0.30]:
        K = max(1, ceil(round((1 - theta) * N, 6)))
        Aq, Mq, Pcq = solve(N, K, V1, xi, kappa, Lp, T, rho, p_opp, exact=False)
        Ae, Me, Pce = solve(N, K, V1, xi, kappa, Lp, T, rho, p_opp, exact=True)
        print(f"theta={theta} K={K}")
        print("  diagonal rungs g=1..K  (state m=K-g, h=g): margin vc-vw")
        print("   g   m  h   A_quad  margin_quad   A_exact  margin_exact")
        for g in range(1, K + 1):
            m, h = K - g, g
            print(
                f"   {g}  {m:2d} {h:2d}   {Aq[m, h]:.0f}    {Mq[m, h]:+.6f}     "
                f"{Ae[m, h]:.0f}     {Me[m, h]:+.6f}"
            )
        print(f"  pi_masked quad={Pcq[0, T]:.6f} exact={Pce[0, T]:.6f}")
        # any rung margin near 0.24?
        dia = [Mq[K - g, g] for g in range(1, K + 1)]
        print("  quad diagonal margins:", [f"{x:+.4f}" for x in dia])
        die = [Me[K - g, g] for g in range(1, K + 1)]
        print("  exact diagonal margins:", [f"{x:+.4f}" for x in die])
    # also theta=0.20 pi for issue 2 cross-check
    for theta in [0.20]:
        K = max(1, ceil(round((1 - theta) * N, 6)))
        Aq, Mq, Pcq = solve(N, K, V1, xi, kappa, Lp, T, rho, p_opp, exact=False)
        Ae, Me, Pce = solve(N, K, V1, xi, kappa, Lp, T, rho, p_opp, exact=True)
        diff = [
            (m, h, Aq[m, h], Ae[m, h], Mq[m, h], Me[m, h])
            for m in range(K)
            for h in range(1, T + 1)
            if Aq[m, h] != Ae[m, h]
        ]
        print(
            f"theta={theta} K={K}: pi quad={Pcq[0, T]:.6f} exact={Pce[0, T]:.6f}; flips={diff}"
        )


if __name__ == "__main__":
    main()
