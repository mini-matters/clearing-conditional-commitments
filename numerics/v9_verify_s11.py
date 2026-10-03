"""
v9_verify_s11 — INDEPENDENT adversarial verification of S11's BANG-BANG claim.

S11 claims: with BOTH a type channel and a pivotality free-ride channel inside the v6/v7 DYNAMIC
game on ONE dial k, the disclosure optimum is BANG-BANG (0/28 LIVE interior peaks on a live market
theta=0.40 pi_b=0.2); adding dynamic momentum does NOT create an interior k*; S5's partial-masking
is robustly a c-DISTRIBUTION object, NOT a type-vs-pivotality channel-balance object.

This script REIMPLEMENTS the dynamic MPE solver from scratch (so an S11 coding bug cannot propagate)
and ATTACKS the bang-bang verdict with DIFFERENT couplings:

  (A) DECOUPLE: separate dials (k_type, k_piv). 2-D welfare surface; search for an interior optimum
      that is interior on BOTH dials (a genuine channel-balance interior, the strongest refutation).
  (B) NON-LINEAR type-optimism: pc(k) = (1-g(k))*pooled + g(k)*true with g(k) = k^gamma, gamma!=1
      (concave gamma<1 front-loads disclosure; convex gamma>1 back-loads it). Does curvature
      manufacture an interior?
  (C) ALTERNATIVE free-ride rules on the single dial:
        C1: fr = k                          (free-ride fraction = k, NOT k*(1-piv_w); stronger unravel)
        C2: fr = k^2                         (convex: free-ride ramps late)
        C3: fr = sqrt(k)                     (concave: free-ride ramps early -- best chance of interior)
        C4: free-ride ALWAYS active (drop the xi>xi* gate), fr = k*(1-piv_w)
      and also runs the free-ride channel BELOW the xi* threshold (S11 gates it off there).

  (D) FAIRNESS of the live-vs-dead W test: re-scan k*(xi) at the HARDER theta=0.30 where S11 admits
      apparent interior peaks, report W(k*) AND the maximum achievable W on that market, so we can
      judge whether 10%-of-max-W hides a real interior or correctly rejects a dead-market artifact.

VERDICT LOGIC: an interior is GENUINE and REFUTING only if (i) k* (or both dials) strictly interior,
(ii) W(k*) materially exceeds both endpoints, and (iii) W(k*) is a live market (not ~0). We report
raw numbers and let the structured verdict speak.

Run:  uv run --with numpy python v9_verify_s11.py
"""

from __future__ import annotations

from math import ceil, comb

import numpy as np

SEED = 20260609


def binom_pmf(n, p):
    if n == 0:
        return np.array([1.0])
    p = min(max(p, 0.0), 1.0)
    return np.array([comb(n, j) * (p**j) * ((1 - p) ** (n - j)) for j in range(n + 1)])


# ================================================================================================
# INDEPENDENT REIMPLEMENTATION of the dynamic MPE solver (v6/v7 solve_bad skeleton).
# State (m, h): m = committed so far, h = horizon remaining. p_opp = the conjectured commit rate of
# opponents (Markov-perfect: we solve the stage best-response, take the OPTIMISTIC/highest-commit
# stage equilibrium as the momentum (masked) action, then dampen by the free-ride fraction).
#
# perceived_clearing    = the agent's perceived payoff if the coalition CLEARS and the agent committed.
# perceived_clearing_nc = the agent's perceived payoff if the coalition clears and the agent did NOT
#                         commit (the free-rider / inframarginal payoff = xi * (clearing value)).
# free_ride_fraction    = a FUNCTION fr(k, piv_w) returning the share of momentum dampened by free-ride.
#                         This is the decoupled hook the verifier varies.
# ================================================================================================
def solve_unified(
    N,
    K,
    perceived_clearing,
    perceived_clearing_nc,
    kappa,
    Lp,
    T,
    rho,
    p_opp,
    free_ride_fraction,  # callable(k_piv_arg) already bound, returns fr in [0,1] given piv_w
):
    Vu = np.zeros((N + 1, T + 1))
    Pc = np.zeros((N + 1, T + 1))
    A = np.zeros((N + 1, T + 1))
    for m in range(N + 1):
        for h in range(T + 1):
            if m >= K:
                Pc[m, h] = 1.0

    for h in range(1, T + 1):
        for m in range(N):
            if m >= K:
                continue
            n = N - m
            gap = K - m

            def vals(q):
                a = p_opp * q
                pmf = binom_pmf(n - 1, a)
                vw, vc = -rho, -rho
                for j in range(n):
                    pj = pmf[j]
                    mp = m + j
                    if mp >= K:
                        vw += pj * perceived_clearing_nc
                    elif h - 1 == 0:
                        vw += pj * 0.0
                    else:
                        vw += pj * Vu[mp, h - 1]
                    mpc = m + 1 + j
                    if mpc >= K:
                        vc += pj * perceived_clearing
                    else:
                        pc = Pc[mpc, h - 1]
                        vc += pj * (perceived_clearing * pc + (-Lp) * (1 - pc))
                return vw, vc

            def diff(q):
                vw, vc = vals(q)
                return vc - vw

            qs = np.linspace(0, 1, 201)
            ds = np.array([diff(q) for q in qs])
            eqs = []
            if ds[0] <= 1e-12:
                eqs.append(0.0)
            if ds[-1] >= -1e-12:
                eqs.append(1.0)
            sc = np.where(np.sign(ds[:-1]) != np.sign(ds[1:]))[0]
            for i in sc:
                if ds[i] > 0 >= ds[i + 1]:
                    lo, hi = qs[i], qs[i + 1]
                    for _ in range(50):
                        mid = 0.5 * (lo + hi)
                        if diff(lo) * diff(mid) <= 0:
                            hi = mid
                        else:
                            lo = mid
                    eqs.append(0.5 * (lo + hi))
            if not eqs:
                eqs.append(1.0 if ds[-1] > ds[0] else 0.0)
            a_momentum = max(eqs)

            exp_movers = p_opp * n
            piv_w = float(np.clip(1.0 - (gap - 1) / max(1.0, exp_movers), 0.0, 1.0))
            fr = float(np.clip(free_ride_fraction(piv_w), 0.0, 1.0))
            a_eff = a_momentum * (1 - fr)
            A[m, h] = a_eff

            b = p_opp * a_eff
            pmf = binom_pmf(n, b)
            Pc[m, h] = sum(
                pmf[j] * (1.0 if m + j >= K else Pc[m + j, h - 1]) for j in range(n + 1)
            )
            vw, vc = vals(a_eff)
            Vu[m, h] = a_eff * vc + (1 - a_eff) * vw

    return {"A": A, "Pc": Pc, "Vu": Vu, "pi_clear": Pc[0, T]}


def pooled_payoffs(pi_b, V1, Lbad, xi, kappa):
    pooled = (1 - pi_b) * (V1 - kappa) + pi_b * (-Lbad - kappa)
    pooled_nc = (1 - pi_b) * (xi * V1) + pi_b * (xi * (-Lbad))
    return pooled, pooled_nc


def type_perceived_linear(k, pi_b, true_clear, true_clear_nc, V1, Lbad, xi, kappa):
    pooled, pooled_nc = pooled_payoffs(pi_b, V1, Lbad, xi, kappa)
    return (1 - k) * pooled + k * true_clear, (1 - k) * pooled_nc + k * true_clear_nc


def type_perceived_nonlinear(
    k, gamma, pi_b, true_clear, true_clear_nc, V1, Lbad, xi, kappa
):
    """g(k)=k^gamma blend. gamma<1 concave (front-loaded disclosure), gamma>1 convex (back-loaded)."""
    g = k**gamma
    pooled, pooled_nc = pooled_payoffs(pi_b, V1, Lbad, xi, kappa)
    return (1 - g) * pooled + g * true_clear, (1 - g) * pooled_nc + g * true_clear_nc


# ================================================================================================
# GENERAL W(.) with pluggable couplings. type_map and fr_rule are the two hooks.
#   k_type, k_piv: the (possibly separate) disclosure dials.
#   type_map(k_type) -> g in [0,1] : how much TRUE-type weight enters perceived payoff.
#   fr_rule(k_piv, piv_w) -> fr in [0,1] : free-ride fraction. (xi-gate applied OUTSIDE if requested.)
# ================================================================================================
def W_general(
    k_type,
    k_piv,
    N,
    K,
    pi_b,
    V1,
    Lbad,
    xi,
    kappa,
    Lp,
    T,
    rho,
    p_opp,
    type_g_fn,  # callable(k_type)->g
    fr_rule,  # callable(k_piv, piv_w)->fr  (already incorporates xi-gating decision)
):
    def blend(true_clear, true_clear_nc):
        g = float(np.clip(type_g_fn(k_type), 0.0, 1.0))
        pooled, pooled_nc = pooled_payoffs(pi_b, V1, Lbad, xi, kappa)
        return (1 - g) * pooled + g * true_clear, (
            1 - g
        ) * pooled_nc + g * true_clear_nc

    def fr_bound(piv_w):
        return fr_rule(k_piv, piv_w)

    pcg, pcg_nc = blend(V1 - kappa, xi * V1)
    solG = solve_unified(N, K, pcg, pcg_nc, kappa, Lp, T, rho, p_opp, fr_bound)
    G = solG["pi_clear"]
    pcb, pcb_nc = blend(-Lbad - kappa, xi * (-Lbad))
    solB = solve_unified(N, K, pcb, pcb_nc, kappa, Lp, T, rho, p_opp, fr_bound)
    s = 1 - solB["pi_clear"]
    W = (1 - pi_b) * G * V1 - pi_b * (1 - s) * Lbad
    return W, G, s


def classify_1d(ks, ws):
    i = int(np.argmax(ws))
    kopt, wopt, w0, w1 = float(ks[i]), float(ws[i]), float(ws[0]), float(ws[-1])
    span = max(ws.max() - ws.min(), 1e-9)
    margin = 0.01 * span
    near = ws >= wopt - margin
    if kopt <= 0.04 or (near[0] and not near[-1]):
        kind = "MASK"
    elif kopt >= 0.96 or near[-1]:
        kind = "DISCLOSE"
    elif wopt > w0 + margin and wopt > w1 + margin:
        kind = "PEAK"
    else:
        kind = "FLAT"
    return kopt, kind, wopt


def main():
    np.random.seed(SEED)
    print("=" * 100)
    print(
        "v9_verify_s11 — INDEPENDENT adversarial verification of S11's bang-bang claim"
    )
    print(f"seed={SEED}  (solver REIMPLEMENTED from scratch)")
    print("=" * 100)

    N, kappa, Lp, rho, p_opp, V1, Lbad, T = 10, 1.0, 3.0, 0.05, 0.6, 3.0, 3.0, 10
    theta = 0.40
    K = max(1, ceil(round((1 - theta) * N, 6)))
    xistar = 1 - kappa / V1
    print(
        f"\ncalib: N={N} theta={theta} K={K} V1={V1} Lbad={Lbad} kappa={kappa} Lp={Lp} "
        f"rho={rho} p_opp={p_opp} T={T}  xi*={xistar:.3f}"
    )
    pi_b = 0.2
    grid = 26
    ks = np.linspace(0, 1, grid)

    # S11's ORIGINAL coupling, reimplemented, to confirm cross-implementation agreement -----------
    def s11_type_g(k):
        return k  # linear

    def s11_fr(k, piv_w, xi):
        gate = 1.0 if xi * V1 > (V1 - kappa) else 0.0
        return k * (1 - piv_w) * gate

    print("\n" + "-" * 100)
    print(
        "CROSS-CHECK 0: reimplement S11's OWN coupling, confirm k*(xi) matches S11 (bang-bang)."
    )
    print("-" * 100)
    print("   xi      k*      W(k*)    regime")
    s11_repro = []
    for xi in [0.0, 0.3, 0.5, 0.6, 0.667, 0.7, 0.8, 0.9]:
        ws = np.array(
            [
                W_general(
                    k,
                    k,
                    N,
                    K,
                    pi_b,
                    V1,
                    Lbad,
                    xi,
                    kappa,
                    Lp,
                    T,
                    rho,
                    p_opp,
                    s11_type_g,
                    lambda kp, pw, _xi=xi: s11_fr(kp, pw, _xi),
                )[0]
                for k in ks
            ]
        )
        kopt, kind, wopt = classify_1d(ks, ws)
        s11_repro.append((xi, kopt, kind, wopt))
        print(f"  {xi:.3f}   {kopt:.3f}  {wopt:+.4f}  {kind}")
    n_peak_repro = sum(1 for _, _, kd, _ in s11_repro if kd == "PEAK")
    print(
        f"  >> reimpl interior-PEAK cells: {n_peak_repro}/8 (S11 reported 0/8 bang-bang)"
    )

    # =============================================================================================
    # ATTACK A: DECOUPLE the dials. 2-D (k_type, k_piv) welfare surface. Genuine channel-balance
    # interior = optimum strictly interior on BOTH dials.
    # =============================================================================================
    print("\n" + "-" * 100)
    print(
        "ATTACK A: DECOUPLED dials (k_type, k_piv). 2-D optimum. xi in the MOMENTUM regime."
    )
    print("-" * 100)
    g2 = 16
    kt_grid = np.linspace(0, 1, g2)
    kp_grid = np.linspace(0, 1, g2)
    for xi in [0.7, 0.75, 0.8, 0.9]:
        gate = 1.0 if xi * V1 > (V1 - kappa) else 0.0
        Wsurf = np.zeros((g2, g2))
        for i, kt in enumerate(kt_grid):
            for j, kp in enumerate(kp_grid):
                Wsurf[i, j] = W_general(
                    kt,
                    kp,
                    N,
                    K,
                    pi_b,
                    V1,
                    Lbad,
                    xi,
                    kappa,
                    Lp,
                    T,
                    rho,
                    p_opp,
                    lambda k: k,
                    lambda kpp, pw, _g=gate: kpp * (1 - pw) * _g,
                )[0]
        ai, aj = np.unravel_index(int(np.argmax(Wsurf)), Wsurf.shape)
        kt_opt, kp_opt, w_opt = kt_grid[ai], kp_grid[aj], Wsurf[ai, aj]
        # interior on a dial = strictly between grid endpoints (not first/last index)
        t_interior = 0 < ai < g2 - 1
        p_interior = 0 < aj < g2 - 1
        both = t_interior and p_interior
        wmax = Wsurf.max()
        tag = (
            "BOTH-INTERIOR"
            if both
            else ("k_type-int" if t_interior else "")
            + (" k_piv-int" if p_interior else "")
        )
        if not tag.strip():
            tag = "corner/edge"
        print(
            f"  xi={xi:.2f} (gate={'on' if gate else 'off'}): argmax W at "
            f"k_type={kt_opt:.3f} k_piv={kp_opt:.3f}  W={w_opt:+.4f}  -> {tag}"
        )
    print(
        "  NOTE: if free-ride only HURTS, optimal k_piv->0; if type only HELPS, optimal k_type->1;"
    )
    print(
        "        decoupled optimum is then the CORNER (1,0) = full disclose+no leak, NOT interior."
    )

    # =============================================================================================
    # ATTACK B: NON-LINEAR type-optimism g(k)=k^gamma on the SINGLE shared dial (S11's coupling
    # otherwise). Convex (gamma>1) delays type-optimism so the free-ride can bite first.
    # =============================================================================================
    print("\n" + "-" * 100)
    print(
        "ATTACK B: NON-LINEAR type map g(k)=k^gamma on the single dial (free-ride = k*(1-piv_w), gated)."
    )
    print("-" * 100)
    for gamma in [0.5, 2.0, 3.0]:
        print(f"  gamma={gamma}:")
        print("     xi      k*      W(k*)    regime")
        npk = 0
        for xi in [0.5, 0.6, 0.667, 0.7, 0.8, 0.9]:
            gate = 1.0 if xi * V1 > (V1 - kappa) else 0.0
            ws = np.array(
                [
                    W_general(
                        k,
                        k,
                        N,
                        K,
                        pi_b,
                        V1,
                        Lbad,
                        xi,
                        kappa,
                        Lp,
                        T,
                        rho,
                        p_opp,
                        lambda kk, _gm=gamma: kk**_gm,
                        lambda kp, pw, _g=gate: kp * (1 - pw) * _g,
                    )[0]
                    for k in ks
                ]
            )
            kopt, kind, wopt = classify_1d(ks, ws)
            if kind == "PEAK":
                npk += 1
            print(f"    {xi:.3f}   {kopt:.3f}  {wopt:+.4f}  {kind}")
        print(f"     >> interior-PEAK cells gamma={gamma}: {npk}/6")

    # =============================================================================================
    # ATTACK C: ALTERNATIVE free-ride rules on the single shared dial (type map linear).
    # =============================================================================================
    print("\n" + "-" * 100)
    print(
        "ATTACK C: ALTERNATIVE free-ride rules on the single dial (type map linear g(k)=k)."
    )
    print("-" * 100)
    fr_rules = {
        "C1 fr=k (gated)": lambda kp, pw, g: kp * g,
        "C2 fr=k^2 (gated)": lambda kp, pw, g: (kp**2) * g,
        "C3 fr=sqrt(k) (gated)": lambda kp, pw, g: (kp**0.5) * g,
        "C4 fr=k(1-piv) UNGATED": lambda kp, pw, g: (
            kp * (1 - pw)
        ),  # free-ride ALWAYS on
    }
    for name, rule in fr_rules.items():
        print(f"  {name}:")
        print("     xi      k*      W(k*)    regime")
        npk = 0
        for xi in [0.3, 0.5, 0.6, 0.667, 0.7, 0.8, 0.9]:
            gate = 1.0 if (xi * V1 > (V1 - kappa)) else 0.0
            ws = np.array(
                [
                    W_general(
                        k,
                        k,
                        N,
                        K,
                        pi_b,
                        V1,
                        Lbad,
                        xi,
                        kappa,
                        Lp,
                        T,
                        rho,
                        p_opp,
                        lambda kk: kk,
                        lambda kp, pw, _g=gate, _r=rule: _r(kp, pw, _g),
                    )[0]
                    for k in ks
                ]
            )
            kopt, kind, wopt = classify_1d(ks, ws)
            if kind == "PEAK":
                npk += 1
            print(f"    {xi:.3f}   {kopt:.3f}  {wopt:+.4f}  {kind}")
        print(f"     >> interior-PEAK cells [{name}]: {npk}/7")

    # =============================================================================================
    # ATTACK C': free-ride channel ACTIVE BELOW the xi* gate too (S11 gates it OFF below xi*=0.667,
    # which guarantees disclose there). If the free-ride bites below threshold, can an interior open?
    # =============================================================================================
    print("\n" + "-" * 100)
    print(
        "ATTACK C': free-ride ACTIVE for ALL xi (no xi* gate). type linear, fr=k*(1-piv_w)."
    )
    print("-" * 100)
    print("   xi      k*      W(k*)    regime")
    npk = 0
    for xi in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.667, 0.7, 0.8, 0.9]:
        ws = np.array(
            [
                W_general(
                    k,
                    k,
                    N,
                    K,
                    pi_b,
                    V1,
                    Lbad,
                    xi,
                    kappa,
                    Lp,
                    T,
                    rho,
                    p_opp,
                    lambda kk: kk,
                    lambda kp, pw: kp * (1 - pw),
                )[0]
                for k in ks
            ]
        )
        kopt, kind, wopt = classify_1d(ks, ws)
        if kind == "PEAK":
            npk += 1
        print(f"  {xi:.3f}   {kopt:.3f}  {wopt:+.4f}  {kind}")
    print(f"  >> interior-PEAK cells (ungated, all xi): {npk}/10")

    # =============================================================================================
    # ATTACK D: FAIRNESS of the live-vs-dead test. At HARD theta=0.30 where S11 admits apparent
    # peaks, report W(k*) AND max-W on a TRULY live market (low theta) so we judge the 10% threshold.
    # =============================================================================================
    print("\n" + "-" * 100)
    print(
        "ATTACK D: fairness of the live/dead W test. theta=0.30 (S11's 'near-dead' interiors)."
    )
    print("-" * 100)
    theta_h = 0.30
    K_h = max(1, ceil(round((1 - theta_h) * N, 6)))
    print(f"  harder theta={theta_h} K={K_h}")
    print(
        "   xi      k*      W(k*)    regime    (max achievable W across all k at this xi)"
    )
    for xi in [0.5, 0.6, 0.667, 0.7, 0.8, 0.9]:
        gate = 1.0 if xi * V1 > (V1 - kappa) else 0.0
        ws = np.array(
            [
                W_general(
                    k,
                    k,
                    N,
                    K_h,
                    pi_b,
                    V1,
                    Lbad,
                    xi,
                    kappa,
                    Lp,
                    T,
                    rho,
                    p_opp,
                    lambda kk: kk,
                    lambda kp, pw, _g=gate: kp * (1 - pw) * _g,
                )[0]
                for k in ks
            ]
        )
        kopt, kind, wopt = classify_1d(ks, ws)
        print(f"  {xi:.3f}   {kopt:.3f}  {wopt:+.4f}  {kind:8s}  maxW={ws.max():+.4f}")
    # Reference: the BEST possible market welfare at theta=0.30 (full clear, no bad) for scale:
    best_possible = (1 - pi_b) * 1.0 * V1
    print(
        f"  reference: a fully-clearing good-only market would give W=(1-pi_b)*V1={best_possible:.3f}"
    )
    print(
        "  JUDGEMENT: if W(k*) at the 'peak' is a small fraction of what a live good market yields,"
    )
    print(
        "             the interior is a near-dead-market artifact and the 10%-of-max test is FAIR."
    )

    print("\n" + "=" * 100)
    print("END v9_verify_s11")
    print("=" * 100)


if __name__ == "__main__":
    main()
