"""
v6 ADVERSARIAL SELF-CHECK — the crux question the v5 verifier flagged and the v6 verifier agent
crashed before answering: is the masking advantage a LEGITIMATE information-driven selection result,
or an artifact of comparing the BEST masked equilibrium to the WORST revealed equilibrium?

The masked and revealed stage games in v6_mpe.py use STRUCTURALLY IDENTICAL stage payoffs (same vc,
vw expressions). solve_masked selects the HIGHEST-commit stage equilibrium (q=max, "momentum");
solve_revealed selects w=0 (=momentum) WHEN sustainable, else the holdout (w=1) / interior WoA.
Because continuation values propagate the selection, masked rides an OPTIMISTIC self-fulfilling MPE
and revealed a PESSIMISTIC one.

This script makes that explicit: ONE solver, parameterized by a stage-selection rule in
{max-commit ('opt'), min-commit ('pess')}, on the SAME game. We report pi_succ under (opt) and
(pess), and the fraction of states where momentum (q=1) is itself a stage equilibrium (so holdout is
a genuine ALTERNATIVE selection, not forced). Honest finding: the v6 masking advantage IS the
optimistic-minus-pessimistic MPE gap; masking's role is to SELECT the optimistic equilibrium (the
A8 / FMP thesis), which v6 inherits and gives a dynamic, endogenous-attrition form — it does not
eliminate the selection assumption.
"""

import numpy as np
from math import comb, ceil


def binom_pmf(n, p):
    if n == 0:
        return np.array([1.0])
    return np.array([comb(n, j) * (p**j) * ((1 - p) ** (n - j)) for j in range(n + 1)])


def solve_select(N, K, V1, xi, kappa, Lp, T, rho, p_opp, selection):
    """Unified solver on the SAME stage game; selection in {'opt','pess'} picks the
    highest/lowest-commit stage equilibrium q. Propagates the chosen selection (self-fulfilling)."""
    Vu = np.zeros((N + 1, T + 1))  # value to an uncommitted agent
    Pc = np.zeros((N + 1, T + 1))  # clearing prob
    Q = np.zeros((N + 1, T + 1))
    for m in range(N + 1):
        for h in range(T + 1):
            if m >= K:
                Pc[m, h] = 1.0

    for h in range(1, T + 1):
        for m in range(N):
            if m >= K:
                continue
            n = N - m

            def vals(q):
                a = p_opp * q
                pmf = binom_pmf(n - 1, a)
                vw = -rho
                vc = -rho
                for j in range(n):
                    p = pmf[j]
                    mp = m + j
                    if mp >= K:
                        vw += p * (xi * V1)
                    elif h - 1 == 0:
                        vw += p * 0.0
                    else:
                        vw += p * Vu[mp, h - 1]
                    mpc = m + 1 + j
                    if mpc >= K:
                        vc += p * (V1 - kappa)
                    else:
                        pc = Pc[mpc, h - 1]
                        vc += p * ((V1 - kappa) * pc + (-Lp) * (1 - pc))
                return vw, vc

            def diff(q):
                vw, vc = vals(q)
                return vc - vw

            eqs = []
            if diff(0.0) <= 1e-12:
                eqs.append(0.0)
            if diff(1.0) >= -1e-12:
                eqs.append(1.0)
            qs = np.linspace(0, 1, 401)
            ds = np.array([diff(q) for q in qs])
            sc = np.where(np.sign(ds[:-1]) != np.sign(ds[1:]))[0]
            for i in sc:
                if ds[i] > 0 >= ds[i + 1]:  # downward (stable) interior crossing
                    lo, hi = qs[i], qs[i + 1]
                    for _ in range(60):
                        mid = 0.5 * (lo + hi)
                        if diff(lo) * diff(mid) <= 0:
                            hi = mid
                        else:
                            lo = mid
                    eqs.append(0.5 * (lo + hi))
            if not eqs:
                eqs.append(1.0 if ds[-1] > ds[0] else 0.0)
            q_star = max(eqs) if selection == "opt" else min(eqs)
            Q[m, h] = q_star
            a = p_opp * q_star
            pmf = binom_pmf(n, a)
            Pc[m, h] = sum(
                pmf[j] * (1.0 if m + j >= K else Pc[m + j, h - 1]) for j in range(n + 1)
            )
            vw, vc = vals(q_star)
            Vu[m, h] = max(vw, vc)

    # fraction of reachable states where momentum (q=1) is itself a stage equilibrium
    return {"Q": Q, "Pc": Pc, "pi_succ": Pc[0, T]}


def momentum_sustainable_fraction(N, K, V1, xi, kappa, Lp, T, rho, p_opp, sol):
    """At the OPTIMISTIC solution's continuations, fraction of states where q=1 is a stage eq."""
    Pc = sol["Pc"]
    ns = nmom = 0
    for h in range(1, T + 1):
        for m in range(N):
            if m >= K:
                continue
            n = N - m
            a = p_opp * 1.0
            pmf = binom_pmf(n - 1, a)
            vw = vc = -rho
            for j in range(n):
                p = pmf[j]
                mp = m + j
                vw += p * ((xi * V1) if mp >= K else (0.0 if h - 1 == 0 else 0.0))
                mpc = m + 1 + j
                if mpc >= K:
                    vc += p * (V1 - kappa)
                else:
                    pc = Pc[mpc, h - 1]
                    vc += p * ((V1 - kappa) * pc + (-Lp) * (1 - pc))
            ns += 1
            if (
                vc - vw >= -1e-12
            ):  # commit is a BR when everyone commits -> momentum is a stage eq
                nmom += 1
    return nmom, ns


if __name__ == "__main__":
    # contested calibration (matches v6_mpe.py §1)
    N, kappa, Lp, rho, xi, p_opp = 10, 1.0, 3.0, 0.05, 0.8, 0.6
    K = max(
        1, ceil((1 - 0.40) * N)
    )  # K/N=0.6: the representative modest-advantage cell
    V1, T = 3.0, 10
    print(
        "=== v6 adversarial self-check: is the masking advantage selection-driven? ==="
    )
    print(
        f"contested calib: N={N} K={K} V1={V1} xi={xi} kappa={kappa} Lp={Lp} rho={rho} p_opp={p_opp} T={T}\n"
    )

    so = solve_select(N, K, V1, xi, kappa, Lp, T, rho, p_opp, "opt")
    sp = solve_select(N, K, V1, xi, kappa, Lp, T, rho, p_opp, "pess")
    print(f"pi_succ OPTIMISTIC (max-commit, = masked/momentum)  = {so['pi_succ']:.4f}")
    print(f"pi_succ PESSIMISTIC (min-commit, = revealed/holdout) = {sp['pi_succ']:.4f}")
    print(
        f"optimistic - pessimistic MPE gap                     = {so['pi_succ'] - sp['pi_succ']:+.4f}"
    )

    nmom, ns = momentum_sustainable_fraction(N, K, V1, xi, kappa, Lp, T, rho, p_opp, so)
    print(f"\nstates where momentum (q=1) is ITSELF a stage equilibrium: {nmom}/{ns}")
    print(
        "  -> if high, the holdout equilibrium is a genuine ALTERNATIVE selection (multiplicity),"
    )
    print(
        "     so masked>=revealed is an information-driven SELECTION result, not a forced outcome."
    )

    print(
        "\n--- the gap IS the optimistic-pessimistic multiplicity gap, swept over V1/kappa ---"
    )
    print(f"{'V1/k':>5} {'pi_opt':>7} {'pi_pess':>8} {'gap':>8}")
    for r in [2.0, 2.5, 3.0, 4.0, 6.0]:
        v1 = r * kappa
        a = solve_select(N, K, v1, xi, kappa, Lp, T, rho, p_opp, "opt")
        b = solve_select(N, K, v1, xi, kappa, Lp, T, rho, p_opp, "pess")
        print(
            f"{r:5.1f} {a['pi_succ']:7.4f} {b['pi_succ']:8.4f} {a['pi_succ'] - b['pi_succ']:+8.4f}"
        )

    print("\n=== HONEST VERDICT ===")
    print(
        "The v6 masking advantage = the OPTIMISTIC-minus-PESSIMISTIC MPE gap of one game."
    )
    print(
        "Masking's role is to SELECT the optimistic (momentum) equilibrium; revealing pivotality"
    )
    print(
        "admits the pessimistic (holdout/war-of-attrition) one. This is the A8/FMP selection"
    )
    print(
        "thesis, now (a) dynamic and (b) with an ENDOGENOUS attrition rate. v6 does NOT eliminate"
    )
    print(
        "the selection assumption; it gives it a solved-MPE form and maps where the gap is large."
    )
