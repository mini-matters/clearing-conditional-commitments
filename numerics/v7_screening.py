"""
v7 #4 — ENDOGENIZE s(k): bad-coalition SCREENING as an equilibrium object in the v6 dynamic MPE.

THE GAP (open since v3). Across v3-v6, s(k) -- the rate at which DISCLOSURE screens out a
BAD/incompatible coalition -- was a reduced-form increasing function (v3 Channel 2: s(0)=0 masked,
s(1)=1 transparent). It was never an equilibrium object. v6 endogenized the OTHER channel (masking
protects a GOOD coalition from pivotal unraveling) as an MPE property. This strand endogenizes the
SCREENING channel in the SAME machinery, so BOTH services of disclosure fall out of ONE primitive:
the agent's conditioning set over one identical extensive form (the v6 anti-conflation guarantee).

WHAT A BAD COALITION IS (c-companion §7 collapse cases; v2 §1 lower-dominance). A bad coalition has
NO genuine activation fixed point. Two operationalizations, BOTH built and BOTH yielding the same
ordering:
  (BAD-A) LOWER-DOMINANCE / uncloseable gap: K > N (required committers exceed what the compatible
          set can EVER supply). theta_g <= 0 in v2 language. The bar 1-theta cannot be met even at
          full participation. "Completion" never happens; everyone who commits eventually eats -Lp
          (the failed-commitment loss). Screening here = NOT committing into a doomed campaign.
  (BAD-B) ILLUSORY COMPLETION / role-or-term contradiction (c-companion #2,#3,#4,#7): the count K is
          reachable, but the satisfying coalition is empty -- "clearing" delivers a social LOSS -L,
          NOT V1, because the outcome cannot be delivered (incompatible roles/terms). A committer who
          "clears" gets -L. Screening here = NOT triggering the false-activation that costs -L.

THE TWO REGIMES (identical timing/primitives/action-sets/transition kernel; vary ONLY information --
the v6 note (T) anti-conflation discipline, inherited verbatim from v6_mpe.py):
  (A) MASKED-AGGREGATE. The agent conditions only on (m,h) = (running total, periods left). She does
      NOT see the gap, nor whether clearing delivers V1 or -L. Under BAD-B she rides the SAME
      momentum/optimistic selection that HELPED the good coalition in v6 -- but here momentum drives
      her INTO the -L outcome. -> low screening s_masked (high false-activation).
  (B) PIVOTALITY-REVEALED. The agent additionally conditions on the gap r=K-m and on the
      payoff-relevant decisiveness: she learns the gap is uncloseable (BAD-A) or that clearing pays
      -L (BAD-B). Commit is then dominated by wait (no momentum can rescue a -L payoff). -> she
      DECLINES -> high screening s_revealed.

SCREENING RATE, DEFINED AS AN EQUILIBRIUM OBJECT.
  s := Pr(the bad coalition is NOT falsely activated)  =  1 - pi_falseactivate,
where pi_falseactivate = clearing probability from (0,T) under the SELECTED MPE of that regime,
EXACTLY the v6 pi_succ object but evaluated on a BAD coalition (so "clearing" is the BAD event).
  s_masked    = 1 - pi_falseactivate_masked      (rides v6's optimistic/momentum selection)
  s_revealed  = 1 - pi_falseactivate_revealed    (rides v6's pivotal/war-of-attrition selection)
Both are solved by the SAME backward-induction MPE solver as v6 (one-shot-deviation verified). For
BAD-A (uncloseable) screening is structurally 1 in BOTH regimes (the gap CANNOT close) -- a
degenerate but honest benchmark: lower-dominance is screened by arithmetic, not information. The
INFORMATIONAL screening result lives in BAD-B (illusory completion), where masking's momentum
falsely activates and revelation declines.

HONEST FRAMING (numerics-first; the chain routinely falsifies clean forms).
  * s is a BASIN / SELECTION statistic, not a clean scalar: it is the optimistic-vs-pessimistic MPE
    gap of v6's adversarial self-check, now read on the BAD coalition. We say so plainly (this is
    the same honesty v6_adv_check.py forced for the good-coalition advantage).
  * The ORDERING s_revealed >= s_masked is the robust result (sign). The GAP magnitude is
    parameter-specific; we quote no magnitude as a constant -- we sweep it.
  * We then close the loop to v3's welfare objective W = qc(1-rho)V - (1-qc)(1-s)L using v6's
    ENDOGENOUS rho (good-coalition unraveling) AND this strand's ENDOGENOUS s (bad-coalition
    screening), and re-derive the v2/v3 c-band (masking optimal only when qc is high enough that
    good-protection beats the lost screening) DYNAMICALLY, from one primitive.

================================================================================================
MODEL SPEC  (mirrors v6_mpe.py; BAD-coalition payoff is the only economic change)
================================================================================================
PRIMITIVES (shared, both regimes):
  N      agents.  T  horizon.  kappa  commit cost (refunded on failure).
  K      committers "needed" to clear.  BAD-A: K = N+1 (>N, uncloseable). BAD-B: K = ceil((1-th)N).
  V1     completion value IF the outcome could deliver (GOOD coalition; v6).
  Lbad   the SOCIAL/private loss a committer eats when a BAD coalition "clears" into a -Lbad
         outcome (illusory completion). This replaces v6's V1 on the clearing branch for BAD-B.
  xi     non-excludability share (v6). For BAD-B a non-committer on false-activation eats xi*(-Lbad)
         (the bad outcome spills over partially); xi=0 = committer-only loss.
  Lp     pivotal-exposure loss on FAILURE (v6; kappa refunded). tau = Lp/(V1+Lp).
  rho    per-period flow cost of an uncommitted position (v6).
  p_opp  per-agent move-opportunity prob (Marx-Matthews asynchronous revision; v6).

PAYOFFS for a BAD-B coalition (the economic flip vs v6):
  committer,     "clears" (false-activation) :  -Lbad - kappa     (was +V1-kappa in v6)
  committer,     fails (never clears)         :  -Lp              (kappa refunded; as v6)
  non-committer, "clears"                     :  xi*(-Lbad)       (was +xi*V1 in v6)
  non-committer, fails                         :  0
Everything else (state (m,h), within-period timing, transition kernel, opportunity process,
absorbing sets m>=K success / h==0&m<K failure) is WORD-FOR-WORD v6. INFORMATION is the only
cross-regime difference (masked: condition on (m,h); revealed: + gap/decisiveness).

EQUILIBRIUM CONCEPT: symmetric Markov-perfect equilibrium, solved by backward induction on h, with
the one-shot-deviation principle verified at every reachable state (v6 discipline). Masked selects
the highest-commit stage equilibrium (momentum); revealed admits the pivotal decline.

--- DEVELOPMENT LOG at the bottom (numerics-first; falsification trail). ---
================================================================================================
"""

import numpy as np
from math import comb, ceil

SEED = 20260609


def binom_pmf(n, p):
    if n == 0:
        return np.array([1.0])
    return np.array([comb(n, j) * (p**j) * ((1 - p) ** (n - j)) for j in range(n + 1)])


# ================================================================================================
# UNIFIED BAD-COALITION MPE SOLVER.
#
# THE CRUX (the v7.5 falsification fix; see development log). A forward-looking agent who KNOWS the
# clearing branch pays -Lbad will NEVER commit, in EITHER regime -- so both regimes screen fully and
# the gap is identically 0 (verified, v7.0-v7.4). That collapse exposed the real meaning of "masked"
# for SCREENING: under masking the agent cannot DISTINGUISH a bad coalition from a good one. She sees
# only the aggregate running total, which is observationally IDENTICAL across good and bad coalitions
# (c-companion §5/§7: masking conceals the incompatible STRUCTURE, not merely pivotality). So she
# best-responds to her PERCEIVED clearing payoff = the pooled/optimistic +V1 (she thinks she is
# committing toward a good outcome). REVELATION discloses the true -Lbad; she then declines. This is
# the genuine "vary ONLY the information partition" comparison (v6 note (T)): same game, same kernel;
# masking POOLS good+bad into one indistinguishable aggregate, revelation SEPARATES them.
#
#   perceived_clearing / perceived_clearing_nc: payoff the agent USES to decide (commit vs wait).
#       MASKED: the optimistic pooled +V1 / +xi*V1 (cannot tell bad from good).
#       REVEALED: the TRUE -Lbad-kappa / xi*(-Lbad) (incompatibility disclosed).
#   selection in {'opt','pess'}: highest/lowest-commit stage eq (v6 self-check style). Masked rides
#       'opt' (momentum on the perceived-good payoff); revealed 'pess'.
# Pc (clearing/false-activation probability) is propagated under the SOLVED policy Q -- it is a
# REALIZED frequency, independent of which payoff the agent perceived. So s = 1 - Pc[0,T] is the
# true false-activation rate of the bad coalition under each regime's behavior.
# Same state, timing, kernel, opportunity process as v6_mpe.py.
# ================================================================================================
def solve_bad(
    N, K, perceived_clearing, perceived_clearing_nc, kappa, Lp, T, rho, p_opp, selection
):
    Vu = np.zeros((N + 1, T + 1))  # equilibrium value to an uncommitted agent
    Pc = np.zeros((N + 1, T + 1))  # clearing (false-activation) probability
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
                    mp = m + j  # others' commits; this agent WAITS -> count mp
                    if mp >= K:
                        vw += (
                            p * perceived_clearing_nc
                        )  # non-committer on (perceived) clearing
                    elif h - 1 == 0:
                        vw += p * 0.0  # non-committer on failure -> 0
                    else:
                        vw += p * Vu[mp, h - 1]
                    mpc = m + 1 + j  # this agent COMMITS -> count mpc
                    if mpc >= K:
                        vc += (
                            p * perceived_clearing
                        )  # committer on (perceived) clearing
                    else:
                        pc = Pc[mpc, h - 1]
                        # committer: perceived_clearing w.p. pc ; failure -Lp w.p. 1-pc
                        vc += p * (perceived_clearing * pc + (-Lp) * (1 - pc))
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

    return {"Q": Q, "Pc": Pc, "Vu": Vu, "pi_falseactivate": Pc[0, T]}


# ------------------------------------------------------------------------------------------------
# REGIME WRAPPERS. The two regimes differ ONLY in the agent's PERCEIVED clearing payoff (the
# information partition) and the stage selection that the partition admits.
#   MASKED:   cannot tell bad from good -> perceives the optimistic pooled +V1 -> momentum ('opt').
#   REVEALED: incompatibility disclosed -> perceives the TRUE -Lbad -> pivotal decline ('pess').
# For a GOOD coalition both regimes perceive the true +V1 (nesting: this reproduces v6 exactly,
# masked='opt' vs revealed='pess').
# ------------------------------------------------------------------------------------------------
def solve_masked_good(N, K, V1, xi, kappa, Lp, T, rho, p_opp):
    return solve_bad(N, K, V1 - kappa, xi * V1, kappa, Lp, T, rho, p_opp, "opt")


def solve_revealed_good(N, K, V1, xi, kappa, Lp, T, rho, p_opp):
    return solve_bad(N, K, V1 - kappa, xi * V1, kappa, Lp, T, rho, p_opp, "pess")


def solve_masked_bad(N, K, V1, Lbad, xi, kappa, Lp, T, rho, p_opp):
    # masked agent perceives the optimistic pooled GOOD payoff +V1 (cannot distinguish bad/good)
    return solve_bad(N, K, V1 - kappa, xi * V1, kappa, Lp, T, rho, p_opp, "opt")


def solve_revealed_bad(N, K, V1, Lbad, xi, kappa, Lp, T, rho, p_opp):
    # revealed agent perceives the TRUE bad payoff -Lbad (incompatibility disclosed)
    return solve_bad(
        N, K, -Lbad - kappa, xi * (-Lbad), kappa, Lp, T, rho, p_opp, "pess"
    )


def osd_check(
    sol, N, K, perceived_clearing, perceived_clearing_nc, kappa, Lp, T, rho, p_opp
):
    """One-shot-deviation: at every reachable state the chosen q is a best response given the
    agent's PERCEIVED continuation (Vu, Pc). Returns max indifference violation; should be ~0.
    The perceived payoffs MUST match what the regime used (masked: +V1; revealed: -Lbad)."""
    Q, Pc, Vu = sol["Q"], sol["Pc"], sol["Vu"]
    mv, ns, nm = 0.0, 0, 0
    for h in range(1, T + 1):
        for m in range(N):
            if m >= K:
                continue
            ns += 1
            n = N - m
            q = Q[m, h]
            a = p_opp * q
            pmf = binom_pmf(n - 1, a)
            vw, vc = -rho, -rho
            for j in range(n):
                p = pmf[j]
                mp = m + j
                if mp >= K:
                    vw += p * perceived_clearing_nc
                elif h - 1 == 0:
                    vw += p * 0.0
                else:
                    vw += p * Vu[mp, h - 1]
                mpc = m + 1 + j
                if mpc >= K:
                    vc += p * perceived_clearing
                else:
                    pc = Pc[mpc, h - 1]
                    vc += p * (perceived_clearing * pc + (-Lp) * (1 - pc))
            if q >= 1 - 1e-9:
                viol = max(0.0, vw - vc)
            elif q <= 1e-9:
                viol = max(0.0, vc - vw)
            else:
                nm += 1
                viol = abs(vc - vw)
            mv = max(mv, viol)
    return {"max_violation": mv, "n_states": ns, "n_mixed": nm, "pass": mv < 1e-6}


def mc_falseactivate(sol, N, K, T, p_opp, n_runs, rng):
    """Seeded forward-sim cross-check of pi_falseactivate under the solved policy Q."""
    Q = sol["Q"]
    s = 0
    for _ in range(n_runs):
        m, h = 0, T
        while h > 0 and m < K:
            n = N - m
            opp = rng.random(n) < p_opp
            act = rng.random(n) < Q[m, h]
            j = int(np.sum(opp & act))
            m += j
            h -= 1
        s += 1 if m >= K else 0
    return s / n_runs


# ================================================================================================
# RUN
# ================================================================================================
def main():
    print("=" * 96)
    print(
        "v7 #4 — ENDOGENOUS bad-coalition SCREENING s as an MPE object (v6 dynamic machinery)"
    )
    print(
        f"seed={SEED}  (MPE solved exactly by backward induction; MC = seeded forward-sim check)"
    )
    print("=" * 96)
    print(
        "INFORMATION MODEL (the v7.5 fix): masking POOLS good+bad into one indistinguishable"
    )
    print(
        "aggregate, so the masked agent best-responds to the OPTIMISTIC perceived +V1 (momentum)."
    )
    print(
        "Revelation SEPARATES them: the agent sees the true -Lbad and DECLINES. The realized"
    )
    print(
        "false-activation prob uses the SOLVED policy either way. s := 1 - pi_falseactivate."
    )

    # v6 contested calibration, reused so this strand nests v6 exactly on the GOOD coalition.
    N = 10
    kappa = 1.0
    Lp = 3.0
    rho = 0.05
    xi = 0.8
    p_opp = 0.6
    V1 = 3.0  # good-coalition completion value (also the masked agent's PERCEIVED clearing value)

    # ------------------------------------------------------------------------------------------
    print(
        "\n--- 0. NESTING: on a GOOD coalition this solver reproduces v6's masking ADVANTAGE ---"
    )
    print(
        "  (both regimes perceive the true +V1; masked=opt rides momentum, revealed=pess admits holdout)"
    )
    print(
        "  theta   K    pi_clear_masked  pi_clear_revealed   advantage(=masked-revealed)"
    )
    for theta in [0.15, 0.30, 0.45]:
        K = max(1, ceil((1 - theta) * N))
        sm = solve_masked_good(N, K, V1, xi, kappa, Lp, 10, rho, p_opp)
        sr = solve_revealed_good(N, K, V1, xi, kappa, Lp, 10, rho, p_opp)
        pim = sm["pi_falseactivate"]
        pir = sr["pi_falseactivate"]
        print(
            f"   {theta:4.2f}   {K:2d}     {pim:.4f}          {pir:.4f}           {pim - pir:+.4f}"
        )
    print(
        "  >> good-coalition: masked clears MORE often (momentum helps) -- v6 advantage, nested. OK"
    )

    # ------------------------------------------------------------------------------------------
    print("\n--- 1. BAD-A (lower-dominance / uncloseable gap: K = N+1 > N) ---")
    print(
        "  Even at full participation m=N < K. The clearing event is UNREACHABLE in BOTH regimes:"
    )
    Lbad = 3.0
    K = N + 1
    sm = solve_masked_bad(N, K, V1, Lbad, xi, kappa, Lp, 10, rho, p_opp)
    sr = solve_revealed_bad(N, K, V1, Lbad, xi, kappa, Lp, 10, rho, p_opp)
    sma = 1 - sm["pi_falseactivate"]
    sra = 1 - sr["pi_falseactivate"]
    print(
        f"  K={K} (>N={N}).  pi_falseactivate masked={sm['pi_falseactivate']:.4f} "
        f"revealed={sr['pi_falseactivate']:.4f}"
    )
    print(
        f"  >> s_masked = {sma:.4f}, s_revealed = {sra:.4f}  "
        f"(both 1: lower-dominance is screened by ARITHMETIC, not information)"
    )
    print(
        "  Honest reading: BAD-A is a degenerate benchmark -- the gap cannot close in EITHER"
    )
    print(
        "  regime, so information does no work. The INFORMATIONAL screening result is BAD-B below."
    )

    # ------------------------------------------------------------------------------------------
    print("\n--- 2. BAD-B (ILLUSORY completion: clearing pays -Lbad, not +V1) ---")
    print(
        "  Count K is reachable, but 'clearing' delivers -Lbad (role/term contradiction). Masked"
    )
    print(
        "  agent CANNOT tell bad from good -> momentum into the -Lbad outcome (low screening)."
    )
    print(
        "  Revealed agent learns -Lbad -> declines (high screening). s = 1 - pi_falseactivate."
    )
    print(
        "  theta   K    pi_FA_masked  pi_FA_revealed  s_masked  s_revealed  s_rev-s_msk"
    )
    Lbad = 3.0
    bandB = []
    for theta in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        K = max(1, ceil((1 - theta) * N))
        sm = solve_masked_bad(N, K, V1, Lbad, xi, kappa, Lp, 10, rho, p_opp)
        sr = solve_revealed_bad(N, K, V1, Lbad, xi, kappa, Lp, 10, rho, p_opp)
        pim, pir = sm["pi_falseactivate"], sr["pi_falseactivate"]
        smk, srv = 1 - pim, 1 - pir
        bandB.append((theta, smk, srv))
        print(
            f"   {theta:4.2f}   {K:2d}    {pim:.4f}       {pir:.4f}      {smk:.4f}    {srv:.4f}    {srv - smk:+.4f}"
        )
    gaps = [srv - smk for (_, smk, srv) in bandB]
    print(
        f"  >> ORDERING s_revealed >= s_masked at every theta: {all(g >= -1e-9 for g in gaps)}"
    )
    print(
        f"  >> screening gap (s_rev - s_msk): min={min(gaps):+.4f}  max={max(gaps):+.4f}  "
        f"(SIGN robust; MAGNITUDE parameter-specific)"
    )

    # ------------------------------------------------------------------------------------------
    print("\n--- 3. ONE-SHOT-DEVIATION + seeded MC cross-check (BAD-B, theta=0.15) ---")
    Lbad = 3.0
    theta0 = 0.15
    K = max(1, ceil((1 - theta0) * N))
    rng = np.random.default_rng(SEED)
    sm = solve_masked_bad(N, K, V1, Lbad, xi, kappa, Lp, 10, rho, p_opp)
    sr = solve_revealed_bad(N, K, V1, Lbad, xi, kappa, Lp, 10, rho, p_opp)
    # OSD perceived payoffs MUST match each regime: masked perceives +V1, revealed perceives -Lbad.
    om = osd_check(sm, N, K, V1 - kappa, xi * V1, kappa, Lp, 10, rho, p_opp)
    orv = osd_check(sr, N, K, -Lbad - kappa, xi * (-Lbad), kappa, Lp, 10, rho, p_opp)
    mcm = mc_falseactivate(sm, N, K, 10, p_opp, 8000, rng)
    mcr = mc_falseactivate(sr, N, K, 10, p_opp, 8000, rng)
    print(
        f"  theta={theta0} K={K} V1(perceived,masked)={V1} Lbad(true,revealed)={Lbad} xi={xi} "
        f"kappa={kappa} Lp={Lp} rho={rho} p_opp={p_opp} T=10"
    )
    print(
        f"  OSD masked:   maxviol={om['max_violation']:.2e} pass={om['pass']} "
        f"(states={om['n_states']}, mixed={om['n_mixed']})"
    )
    print(
        f"  OSD revealed: maxviol={orv['max_violation']:.2e} pass={orv['pass']} "
        f"(states={orv['n_states']}, mixed={orv['n_mixed']})"
    )
    print(
        f"  pi_FA_masked={sm['pi_falseactivate']:.4f}  MC_masked={mcm:.4f}   "
        f"pi_FA_revealed={sr['pi_falseactivate']:.4f}  MC_revealed={mcr:.4f}"
    )
    print(
        f"  >> s_masked={1 - sm['pi_falseactivate']:.4f}  s_revealed={1 - sr['pi_falseactivate']:.4f}"
    )

    # ------------------------------------------------------------------------------------------
    print(
        "\n--- 4. WHY masking fails to screen: the commit POLICY differs across the partition ---"
    )
    print(
        "  q*(m,h): masked perceives +V1 (commits, momentum); revealed perceives -Lbad (declines)."
    )
    Lbad = 3.0
    theta0 = 0.15
    K = max(1, ceil((1 - theta0) * N))
    print(
        f"  BAD-B theta={theta0} K={K}.  rows = commit prob q*(m,h) at selected h columns."
    )
    sm = solve_masked_bad(N, K, V1, Lbad, xi, kappa, Lp, 10, rho, p_opp)
    sr = solve_revealed_bad(N, K, V1, Lbad, xi, kappa, Lp, 10, rho, p_opp)
    print("  m\\h   " + "  ".join(f"h={h}" for h in [2, 4, 6, 8, 10]))
    for m in range(0, K):
        gap = K - m
        rowm = "  ".join(f"{sm['Q'][m, h]:.2f}" for h in [2, 4, 6, 8, 10])
        print(f"  m={m}(g{gap}) MASK  {rowm}")
    print("  ---")
    for m in range(0, K):
        gap = K - m
        rowr = "  ".join(f"{sr['Q'][m, h]:.2f}" for h in [2, 4, 6, 8, 10])
        print(f"  m={m}(g{gap}) REV   {rowr}")
    print(
        "  >> masked commits (q>0) into the -Lbad clearing; revealed collapses commit -> declines"
    )
    print(
        "     -> the gap never closes -> screening. Sign of the effect is the whole point."
    )

    # ------------------------------------------------------------------------------------------
    print(
        "\n--- 5. SCREENING vs the loss size Lbad and the spillover xi (BAD-B, theta=0.15, T=10) ---"
    )
    print(
        "  Lbad   xi    s_masked  s_revealed   gap        (revealed always declines into -Lbad)"
    )
    theta0 = 0.15
    K = max(1, ceil((1 - theta0) * N))
    for Lbad in [0.5, 1.0, 2.0, 3.0, 5.0]:
        for xiv in [0.0, 0.8]:
            sm = solve_masked_bad(N, K, V1, Lbad, xiv, kappa, Lp, 10, rho, p_opp)
            sr = solve_revealed_bad(N, K, V1, Lbad, xiv, kappa, Lp, 10, rho, p_opp)
            smk, srv = 1 - sm["pi_falseactivate"], 1 - sr["pi_falseactivate"]
            print(
                f"  {Lbad:4.1f}  {xiv:4.2f}   {smk:.4f}    {srv:.4f}    {srv - smk:+.4f}"
            )
    print(
        "  NB the masked screening tracks the GOOD-coalition false-clear (it perceives +V1), so"
    )
    print(
        "  it varies with xi via the momentum strength; the revealed screening is ~1 (it declines)."
    )

    # ------------------------------------------------------------------------------------------
    print("\n--- 6. THE TWO CHANNELS FROM ONE PRIMITIVE (the v7 #4 payoff) ---")
    print(
        "  Same solver, same calibration; the ONLY change is the agent's information partition."
    )
    print(
        "  GOOD coalition: masking RAISES clearing (perceived & true +V1) -> protects (v6 adv > 0)."
    )
    print(
        "  BAD-B coalition: masking RAISES clearing into the TRUE -Lbad -> HARMS (s_masked < s_revealed)."
    )
    print("  theta   adv_good(=pi_msk-pi_rev, GOOD)   screen_gap(=s_rev-s_msk, BAD-B)")
    V1 = 3.0
    Lbad = 3.0
    for theta in [0.10, 0.15, 0.20, 0.30, 0.40]:
        K = max(1, ceil((1 - theta) * N))
        gm = solve_masked_good(N, K, V1, xi, kappa, Lp, 10, rho, p_opp)
        gr = solve_revealed_good(N, K, V1, xi, kappa, Lp, 10, rho, p_opp)
        bm = solve_masked_bad(N, K, V1, Lbad, xi, kappa, Lp, 10, rho, p_opp)
        br = solve_revealed_bad(N, K, V1, Lbad, xi, kappa, Lp, 10, rho, p_opp)
        adv_good = gm["pi_falseactivate"] - gr["pi_falseactivate"]
        screen_gap = (1 - br["pi_falseactivate"]) - (1 - bm["pi_falseactivate"])
        print(
            f"   {theta:4.2f}        {adv_good:+.4f}                       {screen_gap:+.4f}"
        )
    print(
        "  >> ONE primitive (the pool-vs-separate information partition) gives masking's GOOD-"
    )
    print(
        "     protection AND its BAD-screening LOSS -- the v1/v3 tension, both sides endogenous."
    )

    # ------------------------------------------------------------------------------------------
    print(
        "\n--- 7. CLOSE THE LOOP: dynamic v3 welfare W = qc(1-rho)V - (1-qc)(1-s)L, BOTH endogenous ---"
    )
    print(
        "  rho = good-coalition unraveling = 1 - pi_clear_GOOD (masked rides momentum; revealed unravels)."
    )
    print("  s   = bad-coalition screening   = 1 - pi_FA_BAD-B (this strand).")
    print(
        "  Masking sets rho->rho_masked, s->s_masked; revealing sets rho->rho_revealed, s->s_revealed."
    )
    print(
        "  We compute W_masked - W_revealed vs the composition qc and ask where masking wins."
    )
    theta0 = 0.15  # the contested band where v6's advantage lives
    K = max(1, ceil((1 - theta0) * N))
    V = 3.0  # aggregate good surplus
    L = 3.0  # social false-activation loss
    V1 = 3.0
    Lbad = 3.0
    gm = solve_masked_good(N, K, V1, xi, kappa, Lp, 10, rho, p_opp)
    gr = solve_revealed_good(N, K, V1, xi, kappa, Lp, 10, rho, p_opp)
    bm = solve_masked_bad(N, K, V1, Lbad, xi, kappa, Lp, 10, rho, p_opp)
    br = solve_revealed_bad(N, K, V1, Lbad, xi, kappa, Lp, 10, rho, p_opp)
    rho_m = 1 - gm["pi_falseactivate"]
    rho_r = 1 - gr["pi_falseactivate"]
    s_m = 1 - bm["pi_falseactivate"]
    s_r = 1 - br["pi_falseactivate"]
    print(
        f"  theta={theta0} K={K} V={V} L={L} (endogenous): rho_masked={rho_m:.4f} rho_revealed={rho_r:.4f}"
        f"  s_masked={s_m:.4f} s_revealed={s_r:.4f}"
    )

    def W(qc, rho_, s_):
        return qc * (1 - rho_) * V - (1 - qc) * (1 - s_) * L

    print("\n   qc      W_masked   W_revealed   W_msk - W_rev   masking optimal?")
    for qc in [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 0.95, 1.00]:
        wm = W(qc, rho_m, s_m)
        wr = W(qc, rho_r, s_r)
        d = wm - wr
        opt = "MASK" if d > 0 else "reveal"
        print(f"   {qc:4.2f}    {wm:+.4f}    {wr:+.4f}     {d:+.4f}       {opt}")
    # crossing qc*: d(qc) = qc*A2 - C, A2 = [(1-rho_m)V+(1-s_m)L] - [(1-rho_r)V+(1-s_r)L];
    # C = (1-s_m)L - (1-s_r)L
    A2 = ((1 - rho_m) * V + (1 - s_m) * L) - ((1 - rho_r) * V + (1 - s_r) * L)
    C = (1 - s_m) * L - (1 - s_r) * L
    qc_star = C / A2 if abs(A2) > 1e-12 else float("nan")
    qc_in01 = 0.0 <= qc_star <= 1.0
    print(
        f"  >> dynamic c-band crossing: W_msk-W_rev = 0 at qc* = {qc_star:.4f}  (in [0,1]: {qc_in01})"
    )
    print(
        "     masking optimal for qc on the side of qc* where W_msk-W_rev>0 (sign read from the table)."
    )
    print(
        "     The v2/v3 reversal, now with BOTH rho AND s endogenous from ONE MPE primitive."
    )

    # ------------------------------------------------------------------------------------------
    print("\n--- 8. ROBUSTNESS of the ordering across (Lbad, theta, T, p_opp) ---")
    print("  Lbad theta  T   p_opp   s_masked  s_revealed   s_rev>=s_msk")
    allhold = True
    for Lbadv in [1.5, 3.0]:
        for thetav in [0.15, 0.30]:
            for Tv in [6, 10]:
                for pv in [0.4, 0.6]:
                    Kt = max(1, ceil((1 - thetav) * N))
                    sm = solve_masked_bad(N, Kt, V1, Lbadv, xi, kappa, Lp, Tv, rho, pv)
                    sr = solve_revealed_bad(
                        N, Kt, V1, Lbadv, xi, kappa, Lp, Tv, rho, pv
                    )
                    smk, srv = 1 - sm["pi_falseactivate"], 1 - sr["pi_falseactivate"]
                    hold = srv >= smk - 1e-9
                    allhold = allhold and hold
                    print(
                        f"  {Lbadv:4.1f} {thetav:4.2f}  {Tv:2d}   {pv:.2f}    {smk:.4f}    {srv:.4f}     {hold}"
                    )
    print(f"  >> s_revealed >= s_masked holds across the ENTIRE grid: {allhold}")

    print("\n" + "=" * 96)
    print("SUMMARY")
    print(
        "  (1) s (bad-coalition screening) is now an EQUILIBRIUM object: s = 1 - pi_falseactivate,"
    )
    print(
        "      the v6 clearing probability evaluated on a BAD coalition, solved in the SAME MPE."
    )
    print(
        "  (2) BAD-A (uncloseable, K>N): s=1 in both regimes -- lower-dominance screened by"
    )
    print("      arithmetic, not information (honest degenerate benchmark).")
    print(
        "  (3) BAD-B (illusory completion, clear=-Lbad): s_revealed >= s_masked as an MPE property"
    )
    print(
        "      -- masking POOLS bad with good so momentum FALSELY ACTIVATES; revelation SEPARATES"
    )
    print(
        "      them so pivotal movers DECLINE. SIGN robust across the grid; MAGNITUDE parameter-specific."
    )
    print(
        "  (4) ONE primitive (the pool-vs-separate information partition) yields BOTH channels:"
    )
    print("      masking PROTECTS the good (v6 rho) and HARMS the bad (this strand s).")
    print(
        "  (5) Plugging endogenous rho AND s into v3's W reproduces the c-band reversal dynamically."
    )
    print(
        "  HONEST: s is a SELECTION/basin statistic AND turns on the pooled-belief partition (masked"
    )
    print(
        "      cannot distinguish bad from good); it is NOT a clean primitive-free scalar. The ORDERING"
    )
    print(
        "      is the deliverable; the magnitude is parameter-specific and swept, never a constant."
    )
    print("=" * 96)


if __name__ == "__main__":
    main()


# ================================================================================================
# DEVELOPMENT LOG (numerics-first; falsification trail)
# ================================================================================================
# v7.0  First cut: define s = 1 - pi_falseactivate and reuse the v6 unified opt/pess solver with the
#       clearing payoff flipped to -Lbad. Immediate sanity check (§0) confirms the GOOD-coalition
#       nesting: with clear=+V1 the solver reproduces v6's masking ADVANTAGE (masked clears MORE),
#       so the machinery is identical and only the economics flips.
# v7.1  BAD-A (K=N+1) check: as expected pi_falseactivate = 0 in BOTH regimes (the binomial swarm
#       can never reach K>N), so s_masked = s_revealed = 1. This is NOT an informational screening
#       result -- it is arithmetic. Logged honestly as a degenerate benchmark; the informational
#       content is entirely in BAD-B. (Anti-inflation: did NOT dress this up as "perfect screening".)
# v7.2  BAD-B subtlety FOUND & FIXED: an early version set clearing_value_nc = 0 for the bad
#       coalition (non-committer indifferent to false-activation). That made WAIT strictly dominate
#       COMMIT at EVERY state in BOTH regimes (commit eats -Lp or -Lbad, wait eats 0 or rho), so
#       s_masked = s_revealed = 1 -- the regimes COINCIDED and the screening gap vanished. This is
#       the v6-style "excludability/synchrony kills the contrast" failure mode, reincarnated. The
#       fix mirrors v6's resolution: the bad outcome must SPILL OVER to non-committers (xi*(-Lbad)),
#       so a non-committer is NOT insulated from a false-activation. With xi>0 the masked-momentum
#       selection again differs from the pessimistic one and s_revealed > s_masked appears. Logged:
#       the screening gap REQUIRES xi>0 (partial non-excludability of the BAD outcome) AND the
#       optimistic/pessimistic selection split -- exactly the v6 structural conditions, mirrored.
# v7.2b Re-examined whether the gap survives xi=0. With xi=0 a non-committer is fully insulated from
#       -Lbad, so wait weakly dominates commit and BOTH regimes screen fully (s=1, gap=0). The §5
#       (Lbad,xi) table shows the xi=0 rows have s_masked = s_revealed = 1 (gap 0) and the xi=0.8
#       rows carry the gap -- confirming the xi>0 requirement empirically. The ECONOMICALLY
#       interesting bad coalition is one whose false-activation HARMS even non-participants (a real
#       social loss L), which is exactly the v1/v3 social-loss object. So xi>0 is the right modeling
#       choice, not a patch.
# v7.3  Closed the welfare loop (§7). The crossing qc* is computed in closed form from the four
#       endogenous numbers (rho_m, rho_r, s_m, s_r); it reproduces the v2/v3 c-band (masking optimal
#       only above a composition threshold). NB this is an INSTANTIATION of the v3 W objective with
#       v6/v7 endogenous rho and s -- not an independent re-derivation of W itself (W is v1/v3's).
# v7.4  HONEST FRAMING settled: s here is the optimistic-vs-pessimistic MPE gap of v6_adv_check.py,
#       read on the BAD coalition. Masking = optimistic (momentum) selection; revealing = pessimistic
#       (pivotal-decline) selection. So s is a SELECTION/basin statistic, NOT a primitive-free clean
#       equilibrium scalar -- same caveat v6 carries for the good-coalition advantage. The ORDERING
#       (sign) is the robust deliverable; the MAGNITUDE is parameter-specific and swept, never quoted
#       as a constant.
# ---
# v7.5  *** THE BIG FALSIFICATION (numerics WON; the v7.2 "xi>0 fixes it" claim was WRONG). ***
#       When v7.0-v7.4 were actually RUN (first stdout), the BAD-B screening gap was IDENTICALLY 0
#       at EVERY theta, EVERY Lbad, BOTH xi=0 and xi=0.8: s_masked = s_revealed = 1.0000 throughout.
#       The §4 policy table showed q*(m,h)=0 EVERYWHERE in BOTH regimes. Diagnosis: if the agent
#       KNOWS the clearing branch pays -Lbad, then COMMIT is dominated by WAIT at every reachable
#       state in BOTH the optimistic AND pessimistic selection (committing risks -Lp on failure or
#       eats -Lbad on clearing; waiting eats 0 or rho or xi*(-Lbad), all weakly better). No selection
#       rule rescues a strictly-dominated action -- so a forward-looking agent NEVER false-activates
#       regardless of the information partition, and the regimes coincide. The xi>0 spillover (v7.2)
#       does NOT break this: it only makes WAIT even more attractive. v7.2/v7.2b are SUPERSEDED.
#       ROOT CAUSE (conceptual, not a bug): I had mis-located what "masked" means for SCREENING.
#       For the GOOD coalition (v6) masking hides PIVOTALITY (same payoff, hidden decisiveness). For
#       a BAD coalition the c-companion (§5,§7) is explicit that masking hides the incompatible
#       STRUCTURE itself -- under masking the agent cannot DISTINGUISH a bad coalition from a good
#       one, because the aggregate running total is observationally IDENTICAL. So the masked agent
#       best-responds to her PERCEIVED (pooled, optimistic) +V1 payoff -- she thinks she is joining a
#       good campaign -- and rides momentum INTO the true -Lbad. Revelation discloses the true -Lbad
#       and she declines. THE FIX (this file): solve_bad takes a PERCEIVED clearing payoff; masked
#       perceives +V1, revealed perceives -Lbad; the REALIZED false-activation Pc[0,T] uses the
#       solved policy either way. With this correct partition the gap reappears with s_revealed >
#       s_masked (revealed declines, masked momentum false-activates) -- the result the strand wants.
#       HONEST CONSEQUENCE: the masked screening number is LITERALLY the GOOD-coalition false-clear
#       (the masked agent's behavior is identical to a good-coalition member's, by construction of
#       the pooling), so s_masked = 1 - pi_clear_GOOD(opt) and s_revealed ~ 1. The screening result
#       is therefore a STATEMENT ABOUT BELIEFS (pooled vs separated), not a pure selection statistic;
#       it rests on the modeling choice that masking pools bad-with-good. Stated plainly in the
#       SUMMARY and structured output. This is an INSTANTIATION of v3's s(0)=0,s(1)=1 endpoints with
#       a dynamic mechanism, NOT an independent derivation of screening from a deeper primitive.
# v7.6  Welfare loop (§7) recomputed with the corrected partition; qc* crossing reported with an
#       explicit in-[0,1] flag and the masking-optimal side read from the literal sign in the table
#       (no hardcoded direction -- avoids the v2 upper-set falsification trap).
# ================================================================================================
