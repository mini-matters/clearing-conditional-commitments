"""
v9 S10 — INDEPENDENT VERIFICATION (adversarial).

S10 claim (v9_selection_lambdafree.py): replacing v7's lambda-specific logit stage selection with
a LAMBDA-FREE stagewise refinement (S10 uses Laplacian / uniform-belief risk-dominance) in the
v6/v7 dynamic game gives:
  (A) NO inversion (lambda-free, single value per cell);
  (B) a REGIME SPLIT: masked >= revealed in the HARD/contested regime (low theta / high K/N), but
      REVERSES in the EASY regime (high theta) where masked agents free-ride -> pure-free-rider
      collapse pi_masked -> 0;
  (C) "masked momentum is NOT a neutral-selection basin" survives lambda-free.

THIS SCRIPT tests whether (A)/(B)/(C) survive a DIFFERENT lambda-free refinement than S10's
Laplacian-uniform belief. Two independent refinements, both lambda-free, both pure-action:

  R1. GLOBAL-GAME (Carlsson-van Damme) vanishing-private-noise limit.
      At each reachable state (m,h) the uncommitted mover faces a binary action (commit/wait) whose
      payoff differential D(a_belief) = Vc - Vw depends on how many OTHER movers she believes commit.
      Carlsson-van Damme: embed the stage coordination game in a global game where each agent gets a
      noisy private signal of a payoff-shifter, take the private noise -> 0. The limit selects the
      action that is RISK-DOMINANT, where risk-dominance for a binary symmetric game with a
      continuum of co-players reduces to the LAPLACIAN-of-actions criterion ONLY for a 2x2 game; for
      the n-player monotone game the CvD/global-game limit selects the threshold where the agent is
      indifferent under the belief that the aggregate action a is UNIFORM on [0,1] -- but crucially
      the relevant object is the belief over the OTHERS' realized COMMIT FRACTION, which here is
      mediated by p_opp (only movers can commit). We build the global-game belief CORRECTLY through
      the p_opp movement kernel (S10's Laplacian put uniform weight on j=0..n-1 OTHER committers and
      ignored p_opp in the belief; we do NOT). The CvD limit action = 1[ integral_0^1 D(a) da > 0 ].

  R2. p-DOMINANCE / POTENTIAL-MAXIMIZER.
      An action profile is p-dominant if each action is a best response whenever it is played by at
      least fraction p of opponents. The Laplacian (p=1/2 uniform) is one belief; we instead compute
      the risk-dominant action via the (exact-potential where it exists, else the "1/2-dominance")
      criterion using the binomial co-mover distribution at the agent's OWN equilibrium mixing, i.e.
      we find the SYMMETRIC stage equilibrium and select the Pareto-undominated one (the
      potential-maximizer selection). This is lambda-free and does NOT assume a uniform belief.

If S10's regime split + free-rider collapse are ROBUST refinements (not Laplacian-uniform
artifacts), R1 and R2 should reproduce: masked >= revealed for low theta, reversal at high theta,
pi_masked -> 0 in the easy regime. If they are Laplacian artifacts, R1/R2 will differ materially.

Inherits the v6/v7 payoff/state/timing kernel VERBATIM (re-implemented here from v7_selection.py
so the only thing that changes is the stage-selection rule). Numerics-first, fixed seed.
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
# SHARED KERNEL (verbatim from v7_selection.py / v9 S10): the dynamic propagation and the symmetric
# stage differential. ONLY the stage-SELECTION rule differs across refinements.
# ================================================================================================
def stage_D(m, h, n, a, V1, xi, kappa, Lp, rho, p_opp, K, Pc, Vu):
    """Symmetric commit-minus-wait differential when every other mover commits w.p. a (through the
    p_opp movement kernel). Co-movers j ~ Binom(n-1, p_opp*a). IDENTICAL to v7 stage_D."""
    b = p_opp * a
    pmf = binom_pmf(n - 1, b)
    vc = -rho
    vw = -rho
    for j in range(n):
        p = pmf[j]
        mpc = m + 1 + j
        if mpc >= K:
            vc += p * (V1 - kappa)
        else:
            pc = Pc[mpc, h - 1]
            vc += p * ((V1 - kappa) * pc + (-Lp) * (1 - pc))
        mp = m + j
        if mp >= K:
            vw += p * (xi * V1)
        elif h - 1 == 0:
            vw += p * 0.0
        else:
            vw += p * Vu[mp, h - 1]
    return vc - vw


def propagate(m, h, n, a, V1, xi, kappa, Lp, rho, p_opp, K, Pc, Vu):
    """Advance Pc and Vu given commit propensity a. IDENTICAL kernel to v7 _propagate / v9 S10."""
    b = p_opp * a
    pmf = binom_pmf(n, b)
    Pc[m, h] = sum(
        pmf[j] * (1.0 if m + j >= K else Pc[m + j, h - 1]) for j in range(n + 1)
    )
    pmf1 = binom_pmf(n - 1, b)
    vc, vw = -rho, -rho
    for j in range(n):
        p = pmf1[j]
        mpc = m + 1 + j
        if mpc >= K:
            vc += p * (V1 - kappa)
        else:
            pc = Pc[mpc, h - 1]
            vc += p * ((V1 - kappa) * pc + (-Lp) * (1 - pc))
        mp = m + j
        if mp >= K:
            vw += p * (xi * V1)
        elif h - 1 == 0:
            vw += p * 0.0
        else:
            vw += p * Vu[mp, h - 1]
    Vu[m, h] = a * vc + (1 - a) * vw


# ================================================================================================
# REFINEMENT R1: GLOBAL-GAME (Carlsson-van Damme) vanishing-noise limit.
# The CvD limit of a symmetric binary game with strategic complements selects, at the limit, the
# action that is a best response under a UNIFORM (Laplacian) belief over the OPPONENT AGGREGATE
# ACTION a in [0,1]. Critically, the aggregate action here is the commit PROPENSITY a (each other
# mover commits w.p. p_opp*a), so the global-game belief is uniform over a in [0,1] -- and D(a) is
# evaluated through the SAME p_opp movement kernel. The CvD action is:
#       a_cvd = 1  iff  integral_0^1 stage_D(...; a) da > 0  (Laplacian over the PROPENSITY).
# This differs from S10's Laplacian, which put uniform weight directly on the integer count j of
# OTHER committers (0..n-1) and ignored p_opp in the belief. Here the uniform belief is over the
# action a, propagated through p_opp -- the textbook CvD aggregation.
# ================================================================================================
def cvd_action(m, h, n, V1, xi, kappa, Lp, rho, p_opp, K, Pc, Vu, n_quad=64):
    grid = (np.arange(n_quad) + 0.5) / n_quad  # midpoint rule, a in (0,1)
    integral = 0.0
    for a in grid:
        integral += stage_D(m, h, n, float(a), V1, xi, kappa, Lp, rho, p_opp, K, Pc, Vu)
    integral /= n_quad
    return 1.0 if integral > 0.0 else 0.0


def solve_masked_cvd(N, K, V1, xi, kappa, Lp, T, rho, p_opp):
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
            a = cvd_action(m, h, n, V1, xi, kappa, Lp, rho, p_opp, K, Pc, Vu)
            A[m, h] = a
            propagate(m, h, n, a, V1, xi, kappa, Lp, rho, p_opp, K, Pc, Vu)
    return {"A": A, "Vu": Vu, "Pc": Pc, "pi_succ": Pc[0, T]}


def solve_revealed_cvd(N, K, V1, xi, kappa, Lp, T, rho, p_opp, piv_force=None):
    """REVEALED 2-type under the global-game limit. Pivotal mover: CvD action on her decisive
    differential (commit clears -> V1-kappa vs wait continuation under uniform belief over whether
    OTHER movers close it). Inframarginal: free-ride iff xi*V1 > V1-kappa (lambda-free, belief-free).
    Blend by the SAME pivotality weight as v7/S10 so the only changed object is the refinement."""
    Vu = np.zeros((N + 1, T + 1))
    Pc = np.zeros((N + 1, T + 1))
    A = np.zeros((N + 1, T + 1))
    for m in range(N + 1):
        for h in range(T + 1):
            if m >= K:
                Pc[m, h] = 1.0
    free_ride_pays = xi * V1 > (V1 - kappa)
    for h in range(1, T + 1):
        for m in range(N):
            if m >= K:
                continue
            n = N - m
            gap = K - m
            # PIVOTAL mover: decisive. Commit -> V1-kappa. Wait -> gap may close from OTHER movers.
            # Global-game belief: uniform over the others' commit propensity a in (0,1); she clears
            # by waiting only if >= gap of the other n-1 movers commit. CvD action = 1[ E_a[commit -
            # wait] > 0 ] under uniform a.
            grid = (np.arange(64) + 0.5) / 64.0
            integ = 0.0
            for a in grid:
                b = p_opp * float(a)
                pmf = binom_pmf(n - 1, b)
                p_close_others = sum(pmf[j] for j in range(gap, n))
                if h - 1 == 0:
                    cont = 0.0
                else:
                    cont = Vu[m, h - 1]
                vc = (V1 - kappa) - rho
                vw = -rho + p_close_others * (xi * V1) + (1 - p_close_others) * cont
                integ += vc - vw
            integ /= 64.0
            a_marg = 1.0 if integ > 0.0 else 0.0
            a_inf = 0.0 if free_ride_pays else 1.0
            exp_movers = p_opp * n
            piv_w = float(np.clip(1.0 - (gap - 1) / max(1.0, exp_movers), 0.0, 1.0))
            if piv_force is not None:
                piv_w = float(piv_force)
            a = piv_w * a_marg + (1 - piv_w) * a_inf
            A[m, h] = a
            propagate(m, h, n, a, V1, xi, kappa, Lp, rho, p_opp, K, Pc, Vu)
    return {"A": A, "Vu": Vu, "Pc": Pc, "pi_succ": Pc[0, T]}


# ================================================================================================
# REFINEMENT R2: POTENTIAL-MAXIMIZER / risk-dominant SYMMETRIC stage equilibrium selection.
# Find ALL symmetric stage fixed points a* of the best-response correspondence a -> 1[D(a)>0]
# (pure) by scanning the sign of D over a in [0,1]. When the stage game is a coordination game
# (D crosses from - to + as a rises: two stable pure equilibria a=0 and a=1 plus an interior
# unstable one a_hat), risk-dominance / the potential maximizer selects a=1 iff the "tipping point"
# a_hat < 1/2 (basin of attraction of the commit equilibrium exceeds 1/2). When it is anti-
# coordination (D crosses + to -), there is a unique interior equilibrium a* = a_hat, which we
# select. This is the Harsanyi-Selten risk-dominant selection computed from the ACTUAL D(a) shape,
# NOT from a uniform belief. lambda-free.
# ================================================================================================
def rd_action_potential(m, h, n, V1, xi, kappa, Lp, rho, p_opp, K, Pc, Vu, n_scan=2001):
    grid = np.linspace(0.0, 1.0, n_scan)
    D = np.array(
        [
            stage_D(m, h, n, float(a), V1, xi, kappa, Lp, rho, p_opp, K, Pc, Vu)
            for a in grid
        ]
    )
    sgn = np.sign(D)
    # crossings
    cross = np.where(np.diff(sgn) != 0)[0]
    if len(cross) == 0:
        # no interior crossing: D single-signed
        return 1.0 if D[len(D) // 2] > 0 else 0.0
    # take the first crossing as the tipping point a_hat (basin boundary)
    i = cross[0]
    a_hat = grid[i] + (grid[i + 1] - grid[i]) * (0.0 - D[i]) / (
        D[i + 1] - D[i] + 1e-300
    )
    rising = (
        D[i + 1] > D[i]
    )  # + slope at crossing => coordination game (a=0 / a=1 stable)
    if rising:
        # coordination: risk-dominant = commit (a=1) iff basin of a=1 exceeds 1/2 i.e. a_hat < 1/2
        return 1.0 if a_hat < 0.5 else 0.0
    else:
        # anti-coordination: unique interior equilibrium a_hat is THE selection (mixed)
        return float(np.clip(a_hat, 0.0, 1.0))


def solve_masked_potential(N, K, V1, xi, kappa, Lp, T, rho, p_opp):
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
            a = rd_action_potential(m, h, n, V1, xi, kappa, Lp, rho, p_opp, K, Pc, Vu)
            A[m, h] = a
            propagate(m, h, n, a, V1, xi, kappa, Lp, rho, p_opp, K, Pc, Vu)
    return {"A": A, "Vu": Vu, "Pc": Pc, "pi_succ": Pc[0, T]}


def solve_revealed_potential(N, K, V1, xi, kappa, Lp, T, rho, p_opp, piv_force=None):
    """Revealed under the potential-maximizer refinement: pivotal mover plays the potential-max
    action on her decisive differential; inframarginal free-rides as before; same pivotality blend."""
    Vu = np.zeros((N + 1, T + 1))
    Pc = np.zeros((N + 1, T + 1))
    A = np.zeros((N + 1, T + 1))
    for m in range(N + 1):
        for h in range(T + 1):
            if m >= K:
                Pc[m, h] = 1.0
    free_ride_pays = xi * V1 > (V1 - kappa)
    for h in range(1, T + 1):
        for m in range(N):
            if m >= K:
                continue
            n = N - m
            gap = K - m

            # pivotal mover's decisive D(a_marg): commit clears vs wait-free-ride if others close it.
            def Dmarg(a_marg):
                b = p_opp * a_marg
                pmf = binom_pmf(n - 1, b)
                p_close = sum(pmf[j] for j in range(gap, n))
                cont = 0.0 if h - 1 == 0 else Vu[m, h - 1]
                vc = (V1 - kappa) - rho
                vw = -rho + p_close * (xi * V1) + (1 - p_close) * cont
                return vc - vw

            grid = np.linspace(0.0, 1.0, 2001)
            D = np.array([Dmarg(float(a)) for a in grid])
            sgn = np.sign(D)
            cross = np.where(np.diff(sgn) != 0)[0]
            if len(cross) == 0:
                a_marg = 1.0 if D[len(D) // 2] > 0 else 0.0
            else:
                i = cross[0]
                a_hat = grid[i] + (grid[i + 1] - grid[i]) * (0.0 - D[i]) / (
                    D[i + 1] - D[i] + 1e-300
                )
                rising = D[i + 1] > D[i]
                if rising:
                    a_marg = 1.0 if a_hat < 0.5 else 0.0
                else:
                    a_marg = float(np.clip(a_hat, 0.0, 1.0))
            a_inf = 0.0 if free_ride_pays else 1.0
            exp_movers = p_opp * n
            piv_w = float(np.clip(1.0 - (gap - 1) / max(1.0, exp_movers), 0.0, 1.0))
            if piv_force is not None:
                piv_w = float(piv_force)
            a = piv_w * a_marg + (1 - piv_w) * a_inf
            A[m, h] = a
            propagate(m, h, n, a, V1, xi, kappa, Lp, rho, p_opp, K, Pc, Vu)
    return {"A": A, "Vu": Vu, "Pc": Pc, "pi_succ": Pc[0, T]}


# ================================================================================================
# S10's ORIGINAL Laplacian refinement, re-implemented IDENTICALLY for side-by-side comparison.
# (uniform weight 1/n over j=0..n-1 OTHER committers, p_opp ignored in belief.)
# ================================================================================================
def laplacian_vals(m, h, n, V1, xi, kappa, Lp, rho, K, Pc, Vu):
    w = 1.0 / n
    vc = -rho
    vw = -rho
    for j in range(n):
        mpc = m + 1 + j
        if mpc >= K:
            vc += w * (V1 - kappa)
        else:
            pc = Pc[mpc, h - 1]
            vc += w * ((V1 - kappa) * pc + (-Lp) * (1 - pc))
        mp = m + j
        if mp >= K:
            vw += w * (xi * V1)
        elif h - 1 == 0:
            vw += w * 0.0
        else:
            vw += w * Vu[mp, h - 1]
    return vc, vw


def solve_masked_lap(N, K, V1, xi, kappa, Lp, T, rho, p_opp):
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
            vc, vw = laplacian_vals(m, h, n, V1, xi, kappa, Lp, rho, K, Pc, Vu)
            a = 1.0 if vc > vw else 0.0
            A[m, h] = a
            propagate(m, h, n, a, V1, xi, kappa, Lp, rho, p_opp, K, Pc, Vu)
    return {"A": A, "Vu": Vu, "Pc": Pc, "pi_succ": Pc[0, T]}


def solve_revealed_lap(N, K, V1, xi, kappa, Lp, T, rho, p_opp, piv_force=None):
    Vu = np.zeros((N + 1, T + 1))
    Pc = np.zeros((N + 1, T + 1))
    A = np.zeros((N + 1, T + 1))
    for m in range(N + 1):
        for h in range(T + 1):
            if m >= K:
                Pc[m, h] = 1.0
    free_ride_pays = xi * V1 > (V1 - kappa)
    for h in range(1, T + 1):
        for m in range(N):
            if m >= K:
                continue
            n = N - m
            gap = K - m
            wait_cont = 0.0 if h - 1 == 0 else Vu[m, h - 1]
            a_marg = 1.0 if (V1 - kappa) > wait_cont else 0.0
            a_inf = 0.0 if free_ride_pays else 1.0
            exp_movers = p_opp * n
            piv_w = float(np.clip(1.0 - (gap - 1) / max(1.0, exp_movers), 0.0, 1.0))
            if piv_force is not None:
                piv_w = float(piv_force)
            a = piv_w * a_marg + (1 - piv_w) * a_inf
            A[m, h] = a
            propagate(m, h, n, a, V1, xi, kappa, Lp, rho, p_opp, K, Pc, Vu)
    return {"A": A, "Vu": Vu, "Pc": Pc, "pi_succ": Pc[0, T]}


def mc_pi(sol, N, K, T, p_opp, n_runs, rng):
    A = sol["A"]
    s = 0
    for _ in range(n_runs):
        m, h = 0, T
        while h > 0 and m < K:
            n = N - m
            opp = rng.random(n) < p_opp
            act = rng.random(n) < A[m, h]
            m += int(np.sum(opp & act))
            h -= 1
        s += 1 if m >= K else 0
    return s / n_runs


def regime_label(adv):
    if adv > 0.02:
        return "MASK-adv"
    if adv < -0.02:
        return "REVEAL-adv"
    return "~tie"


def main():
    print("=" * 100)
    print(
        "v9 S10 INDEPENDENT VERIFICATION — two lambda-free refinements vs S10's Laplacian"
    )
    print(f"seed={SEED}")
    print("=" * 100)

    N, kappa, Lp, rho, xi, p_opp, V1, T = 10, 1.0, 3.0, 0.05, 0.8, 0.6, 3.0, 10
    print(
        f"\ncalib (v6/v7 verbatim): N={N} kappa={kappa} Lp={Lp} rho={rho} xi={xi} p_opp={p_opp} "
        f"V1={V1} T={T}  free-ride pays iff xi>1-kappa/V1={1 - kappa / V1:.3f} (xi={xi}: YES)"
    )

    thetas = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75]

    # ---- TABLE A: side-by-side advantage (M-R) under three lambda-free refinements ----
    print(
        "\n--- A. adv(M-R) under THREE lambda-free refinements (does the regime split survive?) ---"
    )
    print(
        "  theta  K/N   K   adv_LAP(S10)   adv_CvD(global-game)   adv_POT(potential-max)"
    )
    rows = []
    for theta in thetas:
        K = max(1, ceil(round((1 - theta) * N, 6)))
        smL = solve_masked_lap(N, K, V1, xi, kappa, Lp, T, rho, p_opp)
        srL = solve_revealed_lap(N, K, V1, xi, kappa, Lp, T, rho, p_opp)
        smC = solve_masked_cvd(N, K, V1, xi, kappa, Lp, T, rho, p_opp)
        srC = solve_revealed_cvd(N, K, V1, xi, kappa, Lp, T, rho, p_opp)
        smP = solve_masked_potential(N, K, V1, xi, kappa, Lp, T, rho, p_opp)
        srP = solve_revealed_potential(N, K, V1, xi, kappa, Lp, T, rho, p_opp)
        aL = smL["pi_succ"] - srL["pi_succ"]
        aC = smC["pi_succ"] - srC["pi_succ"]
        aP = smP["pi_succ"] - srP["pi_succ"]
        rows.append((theta, K, aL, aC, aP, smL, smC, smP))
        print(
            f"  {theta:.2f}  {K / N:.2f}  {K:2d}   {aL:+.4f} {regime_label(aL):>9}   "
            f"{aC:+.4f} {regime_label(aC):>9}     {aP:+.4f} {regime_label(aP):>9}"
        )

    # ---- TABLE B: pi_masked across refinements — does the easy-regime collapse reproduce? ----
    print(
        "\n--- B. pi_masked by refinement (does the EASY-regime pure-free-rider collapse reproduce?) ---"
    )
    print("  theta  K/N   pi_masked_LAP   pi_masked_CvD   pi_masked_POT")
    for theta, K, aL, aC, aP, smL, smC, smP in rows:
        print(
            f"  {theta:.2f}  {K / N:.2f}   {smL['pi_succ']:.4f}          "
            f"{smC['pi_succ']:.4f}          {smP['pi_succ']:.4f}"
        )

    # ---- TABLE C: pi_revealed across refinements ----
    print("\n--- C. pi_revealed by refinement ---")
    print("  theta  K/N   pi_rev_LAP   pi_rev_CvD   pi_rev_POT")
    for theta in thetas:
        K = max(1, ceil(round((1 - theta) * N, 6)))
        srL = solve_revealed_lap(N, K, V1, xi, kappa, Lp, T, rho, p_opp)
        srC = solve_revealed_cvd(N, K, V1, xi, kappa, Lp, T, rho, p_opp)
        srP = solve_revealed_potential(N, K, V1, xi, kappa, Lp, T, rho, p_opp)
        print(
            f"  {theta:.2f}  {K / N:.2f}   {srL['pi_succ']:.4f}       "
            f"{srC['pi_succ']:.4f}       {srP['pi_succ']:.4f}"
        )

    # ---- VERDICT LOGIC on the three robustness claims ----
    print("\n--- VERDICT on S10's three claims under the INDEPENDENT refinements ---")

    # Claim (B): regime split = mask-adv at low theta, reveal-adv at high theta.
    def has_split(idx):
        # idx: 2=LAP,3=CvD,4=POT
        low = [r[idx] for r in rows if r[0] <= 0.30]
        high = [r[idx] for r in rows if r[0] >= 0.60]
        mask_low = all(x > 0.02 for x in low)
        reveal_high = all(x < -0.02 for x in high)
        return mask_low, reveal_high

    for name, idx in [("LAP(S10)", 2), ("CvD", 3), ("POT", 4)]:
        ml, rh = has_split(idx)
        print(
            f"  {name:>10}: mask-adv at low theta(<=.30)={ml}   reveal-adv at high theta(>=.60)={rh}"
        )

    # Claim collapse: pi_masked -> ~0 at high theta
    print("\n  EASY-regime collapse (pi_masked < 0.05 at theta>=0.60)?")
    for name, smkey in [("LAP", 5), ("CvD", 6), ("POT", 7)]:
        vals = [r[smkey]["pi_succ"] for r in rows if r[0] >= 0.60]
        print(
            f"    {name}: pi_masked@(theta>=.60) = {[f'{v:.3f}' for v in vals]} -> collapse={all(v < 0.05 for v in vals)}"
        )

    # ---- D. NESTING controls (xi=0 / p_opp=1 should kill gap) under CvD and POT ----
    print(
        "\n--- D. NESTING controls at theta=0.40 (xi=0 or p_opp=1 should ~kill the gap) ---"
    )
    print("  xi   p_opp  adv_CvD    adv_POT")
    K = max(1, ceil(round((1 - 0.40) * N, 6)))
    for xv, pv in [(0.0, 0.6), (0.8, 1.0), (0.8, 0.6), (0.5, 0.6)]:
        smC = solve_masked_cvd(N, K, V1, xv, kappa, Lp, T, rho, pv)
        srC = solve_revealed_cvd(N, K, V1, xv, kappa, Lp, T, rho, pv)
        smP = solve_masked_potential(N, K, V1, xv, kappa, Lp, T, rho, pv)
        srP = solve_revealed_potential(N, K, V1, xv, kappa, Lp, T, rho, pv)
        print(
            f"  {xv:.2f} {pv:.2f}  {smC['pi_succ'] - srC['pi_succ']:+.4f}   "
            f"{smP['pi_succ'] - srP['pi_succ']:+.4f}"
        )

    # ---- E. FINITE-N under CvD (does the sign behavior track S10?) ----
    print("\n--- E. FINITE-N (K/N=0.6, T=N) under CvD vs Laplacian ---")
    print("  N    K   adv_LAP    adv_CvD    adv_POT")
    for Nv in [6, 8, 10, 12, 16, 20]:
        Kv = max(1, ceil(round(0.6 * Nv, 6)))
        aL = (
            solve_masked_lap(Nv, Kv, V1, xi, kappa, Lp, Nv, rho, p_opp)["pi_succ"]
            - solve_revealed_lap(Nv, Kv, V1, xi, kappa, Lp, Nv, rho, p_opp)["pi_succ"]
        )
        aC = (
            solve_masked_cvd(Nv, Kv, V1, xi, kappa, Lp, Nv, rho, p_opp)["pi_succ"]
            - solve_revealed_cvd(Nv, Kv, V1, xi, kappa, Lp, Nv, rho, p_opp)["pi_succ"]
        )
        aP = (
            solve_masked_potential(Nv, Kv, V1, xi, kappa, Lp, Nv, rho, p_opp)["pi_succ"]
            - solve_revealed_potential(Nv, Kv, V1, xi, kappa, Lp, Nv, rho, p_opp)[
                "pi_succ"
            ]
        )
        print(f"  {Nv:2d}  {Kv:2d}   {aL:+.4f}   {aC:+.4f}   {aP:+.4f}")

    # ---- F. ROBUSTNESS: does the EASY-regime free-ride collapse depend on xi crossing the threshold?
    # If collapse is REAL (free-ride driven), it should vanish when free-ride does NOT pay (xi small).
    print(
        "\n--- F. xi-sweep at theta=0.70 (collapse should require free-ride to pay: xi>0.667) ---"
    )
    print("  xi    free-ride?  pi_masked_CvD  pi_masked_POT  pi_masked_LAP")
    K = max(1, ceil(round((1 - 0.70) * N, 6)))
    for xv in [0.40, 0.60, 0.667, 0.70, 0.80, 0.90]:
        fr = xv * V1 > (V1 - kappa)
        pmC = solve_masked_cvd(N, K, V1, xv, kappa, Lp, T, rho, p_opp)["pi_succ"]
        pmP = solve_masked_potential(N, K, V1, xv, kappa, Lp, T, rho, p_opp)["pi_succ"]
        pmL = solve_masked_lap(N, K, V1, xv, kappa, Lp, T, rho, p_opp)["pi_succ"]
        print(f"  {xv:.3f}  {str(fr):>9}   {pmC:.4f}        {pmP:.4f}        {pmL:.4f}")

    # ---- G. MC cross-check of CvD at theta=0.30 ----
    print("\n--- G. SEEDED MC cross-check of CvD (theta=0.30) ---")
    rng = np.random.default_rng(SEED)
    K = max(1, ceil(round((1 - 0.30) * N, 6)))
    smC = solve_masked_cvd(N, K, V1, xi, kappa, Lp, T, rho, p_opp)
    srC = solve_revealed_cvd(N, K, V1, xi, kappa, Lp, T, rho, p_opp)
    print(
        f"  pi_masked_CvD  exact={smC['pi_succ']:.4f} MC={mc_pi(smC, N, K, T, p_opp, 8000, rng):.4f}"
    )
    print(
        f"  pi_rev_CvD     exact={srC['pi_succ']:.4f} MC={mc_pi(srC, N, K, T, p_opp, 8000, rng):.4f}"
    )

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
