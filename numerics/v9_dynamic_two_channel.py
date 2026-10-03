"""
v9 S11 — the DYNAMIC two-channel synthesis: does W(k) host a GENUINE INTERIOR k* at intermediate
xi when BOTH channels run inside the v6/v7 DYNAMIC (momentum) game? Closes v8 §7 #3.

================================================================================================
THE GAP (v8 §7 #3 / S9's static limitation)
================================================================================================
v8 S9's STATIC two-channel synthesis found k*(xi) BANG-BANG (0/40 interior peaks) -- BUT its high-xi
side was structurally EMPTY: a static coordination game cannot ACTIVATE a non-excludable good
(everyone free-rides, W~0), so the high-xi masking benefit -- the v6 DYNAMIC MOMENTUM channel -- was
simply absent from the static model. v8 flagged that the "0 interior peaks" was therefore "no
interior peak IN THE STATIC MODEL," not a derived knife-edge.

v9 S11 puts BOTH channels inside ONE v6/v7 DYNAMIC MPE on ONE dial k, so the high-xi momentum channel
is present, and asks the real question: does an INTERIOR k* emerge dynamically (re-deriving S5's
partial masking from the two channels), or is it bang-bang even with momentum?

ONE DIAL k controls BOTH informational channels (the v8 S9 justification: aggregate-state
granularity leaks both type and decisiveness):
  TYPE channel (perceived-payoff optimism): at disclosure k the agent's perceived clearing payoff is
    type-weighted -- pc(k) = (1-k)*pooled + k*true. k=0: POOLED (can't tell good from bad -> both ride
    momentum on the prior-mix); k=1: TRUE (good perceives +V1 -> momentum; bad perceives -Lbad ->
    decline -> screened). [helps screen bad as k^; helps good momentum as k^]
  PIVOTALITY channel (free-ride): at disclosure k a fraction k of movers see their decisiveness and
    FREE-RIDE when inframarginal (xi>1-kappa/V1); fraction (1-k) are masked and push (momentum).
    [HURTS good as k^ via unraveling]
So for a GOOD coalition the two channels OPPOSE on k (type-optimism helps momentum; pivotality
free-ride unravels). For a BAD coalition the type channel screens (k^ -> decline). W(k) trades the
good-coalition tradeoff against the bad-screening benefit.

  G(k) = Pr(GOOD coalition clears | unified dynamic solve at k)
  s(k) = 1 - Pr(BAD coalition clears | unified dynamic solve at k)
  W(k) = (1-pi_b)*G(k)*V1 - pi_b*(1-s(k))*Lbad.

WHAT THIS DELIVERS vs INSTANTIATES (set before running):
  DELIVERS: a fully DYNAMIC test of whether the two channels on one dial produce an interior k* --
    the high-xi momentum channel S9 could not host. Either an interior k* (re-deriving S5 dynamically)
    or bang-bang WITH momentum (a stronger version of S9's finding).
  INSTANTIATES: the one-dial coupling (type & pivotality precisions move together = k); the type-
    optimism interpolation pc(k)=(1-k)pooled+k*true; the pivotality free-ride fraction = k. These are
    reduced-form couplings of the v6/v7 + S10/S12 mechanisms, honestly a unified-but-reduced model.

NUMERICS-FIRST, FIXED SEED. Built on the v7 solve_bad skeleton. Dev log at bottom.
================================================================================================
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
# UNIFIED DYNAMIC SOLVER. v7 solve_bad skeleton (state (m,h), p_opp, backward induction), with ONE
# dial k driving BOTH the type-optimism of the perceived clearing payoff AND the pivotality free-ride
# fraction. perceived_clearing/_nc are the TYPE-weighted payoffs (computed by the caller). The
# pivotality channel enters as: a fraction k of movers, when INFRAMARGINAL, free-ride (commit prob 0);
# fraction (1-k) push (momentum, opt selection). The effective commit propensity blends these.
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
    k,
    free_ride_pays,
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

            # MOMENTUM (masked) action: highest-commit stage equilibrium (opt selection, v6/v7).
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

            # PIVOTALITY free-ride: a fraction k of movers, when INFRAMARGINAL, decline. The expected
            # commit propensity is a_momentum scaled by the chance the mover is NOT a free-riding
            # inframarginal. pivotality weight (fraction marginal/decisive) ~ v7 construction.
            exp_movers = p_opp * n
            piv_w = float(np.clip(1.0 - (gap - 1) / max(1.0, exp_movers), 0.0, 1.0))
            # fraction free-riding = k (disclosure) * (1-piv_w) (chance inframarginal) * 1[free-ride pays]
            fr = k * (1 - piv_w) * (1.0 if free_ride_pays else 0.0)
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


def type_perceived(k, pi_b, true_clear, true_clear_nc, V1, Lbad, xi, kappa):
    """Type-weighted perceived payoff: (1-k)*pooled + k*true. pooled = prior-mix (can't tell type)."""
    pooled = (1 - pi_b) * (V1 - kappa) + pi_b * (-Lbad - kappa)
    pooled_nc = (1 - pi_b) * (xi * V1) + pi_b * (xi * (-Lbad))
    return (1 - k) * pooled + k * true_clear, (1 - k) * pooled_nc + k * true_clear_nc


def G_and_s(k, N, K, pi_b, V1, Lbad, xi, kappa, Lp, T, rho, p_opp):
    free_ride = xi * V1 > (V1 - kappa)
    # GOOD coalition: true clearing payoff = +V1
    pcg, pcg_nc = type_perceived(k, pi_b, V1 - kappa, xi * V1, V1, Lbad, xi, kappa)
    solG = solve_unified(N, K, pcg, pcg_nc, kappa, Lp, T, rho, p_opp, k, free_ride)
    G = solG["pi_clear"]
    # BAD coalition: true clearing payoff = -Lbad
    pcb, pcb_nc = type_perceived(
        k, pi_b, -Lbad - kappa, xi * (-Lbad), V1, Lbad, xi, kappa
    )
    solB = solve_unified(N, K, pcb, pcb_nc, kappa, Lp, T, rho, p_opp, k, free_ride)
    s = 1 - solB["pi_clear"]
    return G, s


def W_of_k(k, N, K, pi_b, V1, Lbad, xi, kappa, Lp, T, rho, p_opp):
    G, s = G_and_s(k, N, K, pi_b, V1, Lbad, xi, kappa, Lp, T, rho, p_opp)
    return (1 - pi_b) * G * V1 - pi_b * (1 - s) * Lbad, G, s


def G_and_s_decoupled(
    k_type, k_piv, N, K, pi_b, V1, Lbad, xi, kappa, Lp, T, rho, p_opp
):
    """DECOUPLED two dials: k_type sets the type-optimism of the perceived payoff; k_piv sets the
    pivotality free-ride fraction. Tests whether the interior is a channel-BALANCE object (interior
    on BOTH dials) or a one-dial-coupling artifact (corner (k_type=1,k_piv=0))."""
    free_ride = xi * V1 > (V1 - kappa)
    pcg, pcg_nc = type_perceived(k_type, pi_b, V1 - kappa, xi * V1, V1, Lbad, xi, kappa)
    G = solve_unified(N, K, pcg, pcg_nc, kappa, Lp, T, rho, p_opp, k_piv, free_ride)[
        "pi_clear"
    ]
    pcb, pcb_nc = type_perceived(
        k_type, pi_b, -Lbad - kappa, xi * (-Lbad), V1, Lbad, xi, kappa
    )
    s = (
        1
        - solve_unified(N, K, pcb, pcb_nc, kappa, Lp, T, rho, p_opp, k_piv, free_ride)[
            "pi_clear"
        ]
    )
    return (1 - pi_b) * G * V1 - pi_b * (1 - s) * Lbad


def best_2d(N, K, pi_b, V1, Lbad, xi, kappa, Lp, T, rho, p_opp, grid=11):
    g = np.linspace(0, 1, grid)
    best, arg = -1e18, (0.0, 0.0)
    for kt in g:
        for kp in g:
            w = G_and_s_decoupled(
                float(kt), float(kp), N, K, pi_b, V1, Lbad, xi, kappa, Lp, T, rho, p_opp
            )
            if w > best:
                best, arg = w, (float(kt), float(kp))
    return arg, best


def kstar(N, K, pi_b, V1, Lbad, xi, kappa, Lp, T, rho, p_opp, grid=51):
    ks = np.linspace(0, 1, grid)
    ws = np.array(
        [
            W_of_k(float(k), N, K, pi_b, V1, Lbad, xi, kappa, Lp, T, rho, p_opp)[0]
            for k in ks
        ]
    )
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
    return kopt, kind, ks, ws, wopt


def main():
    print("=" * 96)
    print(
        "v9 S11 — DYNAMIC two-channel synthesis: interior k* WITH momentum? (closes v8 §7 #3)"
    )
    print(f"seed={SEED}")
    print("=" * 96)

    N, kappa, Lp, rho, p_opp, V1, Lbad, T = 10, 1.0, 3.0, 0.05, 0.6, 3.0, 3.0, 10
    theta = 0.30  # K=7: the contested band where the inverted-U interior in G(k) is decisive
    K = max(1, ceil(round((1 - theta) * N, 6)))
    print(
        f"\ncalib (v6/v7): N={N} theta={theta} K={K} V1={V1} Lbad={Lbad} kappa={kappa} Lp={Lp} "
        f"rho={rho} p_opp={p_opp} T={T}.  free-ride threshold xi*=1-kappa/V1={1 - kappa / V1:.3f}"
    )
    print(
        "  NOTE (v9.S11.2 correction): the v9.S11.1 'bang-bang / no interior' headline was FALSE -- it"
    )
    print(
        "  was calibration-cherry-picked at theta=0.40 (first value past a cliff) with a coarse xi grid,"
    )
    print(
        "  and the 'W~0.13 near-dead' justification matched NO computed number. A finer grid + theta=0.30"
    )
    print(
        "  show a DECISIVE LIVE interior. This script is the corrected version (adversarial-pass fix)."
    )

    # --- 1. the inverted-U in G(k): a DECISIVE live interior (theta=0.30, xi=0.667) ---
    print(
        "\n--- 1. DYNAMIC G(k), s(k), W(k) at xi=0.667 (pi_b=0.2): the inverted-U interior ---"
    )
    print(
        "   k     G(k)     s(k)     W(k)    (G=good clears; both endpoints near 0 -> interior wins)"
    )
    pi_b, xi = 0.2, 0.667
    for k in np.linspace(0, 1, 11):
        W, G, s = W_of_k(float(k), N, K, pi_b, V1, Lbad, xi, kappa, Lp, T, rho, p_opp)
        print(f"  {k:.2f}   {G:.4f}   {s:.4f}   {W:+.4f}")
    print(
        "  >> G(k) is an INVERTED-U with G(0)~0 (mask cannot ignite momentum) and G(1)~0 (free-ride"
    )
    print(
        "     fully unravels at full disclosure): the interior is the ONLY clearing point. The interior"
    )
    print(
        "     W beats BOTH bang-bang endpoints by the whole welfare range -- a genuine, LIVE interior k*."
    )

    # --- 2. k*(xi) with a FINE xi grid (the v9.S11.1 grid skipped the 0.72-0.78 band) ---
    print(
        "\n--- 2. k*(xi), FINE xi grid (pi_b=0.2): genuine interior PEAKs (W(k*) magnitude shown) ---"
    )
    print("   xi      k*      W(k*)    regime")
    peakcount = 0
    for xi in [
        0.0,
        0.3,
        0.5,
        0.6,
        0.65,
        0.667,
        0.70,
        0.72,
        0.75,
        0.78,
        0.80,
        0.85,
        0.90,
    ]:
        kopt, kind, _, _, wopt = kstar(
            N, K, 0.2, V1, Lbad, xi, kappa, Lp, T, rho, p_opp
        )
        if kind == "PEAK":
            peakcount += 1
        print(f"  {xi:.3f}   {kopt:.3f}   {wopt:+.4f}  {kind}")
    print(
        f"  >> genuine interior-PEAK cells (DYNAMIC, fine grid): {peakcount}/13 -- the interior is REAL."
    )

    # --- 3. theta-CLIFF: where (in theta) does the live interior exist? ---
    print(
        "\n--- 3. theta-CLIFF (pi_b=0.2, xi=0.667): live interior exists in a contested theta band ---"
    )
    print("  theta   K    k*      W(k*)    regime")
    for th in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        Kt = max(1, ceil(round((1 - th) * N, 6)))
        kopt, kind, _, _, wopt = kstar(
            N, Kt, 0.2, V1, Lbad, 0.667, kappa, Lp, T, rho, p_opp
        )
        print(f"  {th:.2f}   {Kt:2d}   {kopt:.3f}   {wopt:+.4f}  {kind}")
    print(
        "  >> the live interior lives in a CONTESTED theta band (roughly theta<=0.40); it gives way to"
    )
    print(
        "     DISCLOSE/MASK only at easier theta. v9.S11.1 picked theta=0.40 (just past the band) and"
    )
    print(
        "     mislabeled the harder-theta interiors as dead -- the error the adversarial pass caught."
    )

    # --- 4. DECOUPLED 2-DIAL test: is the interior a channel-BALANCE object? (the surviving claim) ---
    print(
        "\n--- 4. DECOUPLE the dials (k_type, k_piv): is the interior a channel-BALANCE object? ---"
    )
    print("  xi      argmax (k_type, k_piv)     W*       interpretation")
    for xi in [0.667, 0.70, 0.75]:
        (kt, kp), wstar = best_2d(N, K, 0.2, V1, Lbad, xi, kappa, Lp, T, rho, p_opp)
        interior_both = (0.05 < kt < 0.95) and (0.05 < kp < 0.95)
        tag = (
            "INTERIOR-on-both (balance)" if interior_both else "CORNER (not a balance)"
        )
        print(f"  {xi:.3f}   ({kt:.2f}, {kp:.2f})            {wstar:+.4f}  {tag}")
    print(
        "  >> the 2-D optimum is the CORNER (k_type=1, k_piv=0): you want FULL type-disclosure with"
    )
    print(
        "     ZERO free-ride leakage. So the single-dial interior is NOT a channel-BALANCE object --"
    )
    print(
        "     it exists ONLY because one physical dial YOKES type-disclosure to free-ride-leakage."
    )

    print("\n" + "=" * 96)
    print("SUMMARY (v9 S11) — CORRECTED after the adversarial pass")
    print(
        "  (1) The DYNAMIC two-channel model DOES host a genuine LIVE interior k* (e.g. theta=0.30,"
    )
    print(
        "      xi=0.667: G(k) inverted-U, both endpoints ~0, interior W beats both by the full range)."
    )
    print(
        "      v9.S11.1's 'bang-bang / no interior' was FALSE -- calibration-cherry-picked at theta=0.40"
    )
    print(
        "      with a coarse xi grid, and its 'W~0.13 near-dead' figure matched NO computed number."
    )
    print(
        "  (2) So adding dynamic momentum DOES create the interior k* that S9's STATIC model lacked --"
    )
    print(
        "      'type-optimism builds momentum, then free-ride unravels it' operates decisively."
    )
    print(
        "  (3) What SURVIVES (the deeper, defensible claim): the DECOUPLED 2-dial optimum is the CORNER"
    )
    print(
        "      (k_type=1, k_piv=0), so the interior is NOT a type-vs-pivotality channel-BALANCE object --"
    )
    print(
        "      it exists only because ONE physical dial yokes type-disclosure to free-ride-leakage."
    )
    print(
        "  (4) So S5's partial-masking has TWO distinct sources: the c-INTEGRITY DISTRIBUTION (v3/S9)"
    )
    print(
        "      AND the one-dial coupling (S11). NEITHER is a channel-balance object. HONEST: reduced"
    )
    print(
        "      model (one-dial coupling; reduced-form type-optimism + free-ride fraction); sign is the"
    )
    print("      deliverable, magnitudes calibration-specific.")
    print("=" * 96)


if __name__ == "__main__":
    main()


# ================================================================================================
# DEVELOPMENT LOG (numerics-first)
# ================================================================================================
# v9.S11.0: unified dynamic solver on the v7 solve_bad skeleton; one dial k drives type-optimism
#   (perceived payoff (1-k)*pooled+k*true) AND pivotality free-ride fraction (k*(1-piv_w) of movers
#   decline when inframarginal). G(k)=good clears, s(k)=bad screened, W(k) traded.
# v9.S11.1 (WRONG -- retracted): claimed "0/28 LIVE interior peaks / bang-bang / no interior even with
#   momentum" at theta=0.40, and dismissed harder-theta interiors as "NEAR-DEAD markets (W~0.13)".
# v9.S11.2 (CORRECTION, forced by the adversarial pass -- three independent reviewers + the audit agreed):
#   the v9.S11.1 headline was FALSE *on this model's own output*. (i) The "W~0.13 near-dead" figure
#   appears in NO computed output -- it was templated prose copied from a different cell; the theta=0.30
#   interiors are the run's LARGEST-W cells (W=+2.39, with W(0)=W(1)=0). (ii) theta=0.40 was calibration-
#   cherry-picked: a theta-cliff scan gives live interior PEAKs across theta~0.20-0.40 and they vanish
#   only at theta>=0.45-0.50; theta=0.40 is the first value past the cliff. (iii) Even at theta=0.40 a
#   FINER xi grid (the 0.72-0.78 band v9.S11.1 skipped by jumping 0.70->0.80) finds a live interior.
#   So the DYNAMIC two-channel model DOES host a genuine live interior k* (G(k) inverted-U, both
#   endpoints ~0). WHAT SURVIVES: the DECOUPLED 2-dial optimum is the CORNER (k_type=1, k_piv=0) -- so
#   the interior is NOT a channel-BALANCE object; it exists only because ONE physical dial yokes type-
#   disclosure to free-ride-leakage. S5's partial masking thus has TWO sources (c-distribution v3/S9 AND
#   the one-dial coupling here); neither is a channel-balance object. This script is the corrected version.
# ================================================================================================
