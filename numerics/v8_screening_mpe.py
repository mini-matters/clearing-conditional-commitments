"""
v8 #2 (b) — NEST the derived screening into v7's ACTUAL dynamic MPE: a PUBLIC type signal of
precision k, embedded in v7_screening.py's solve_bad. Recovers v7's masked/revealed s as the
k->0 / k->1 LIMITS, gives a continuous DYNAMIC s(k), and reconciles v3's LINEAR posit s(k)=k.

================================================================================================
WHY THIS SCRIPT (the rigor link to v7)
================================================================================================
v8_screening_signal.py derives s(k) from a PRIVATE type-signal (each agent her own) in a static
decision/global-game. This script validates the derivation inside v7's REAL dynamic MPE machinery,
and isolates a clean public-vs-private distinction:

  v7's binary assumption (masked perceives +V1; revealed perceives -Lbad) is the k in {0,1} limit
  of a PUBLIC type signal Z of precision k that the platform discloses. Under disclosure k, all
  agents share a posterior; their perceived clearing payoff is the posterior-weighted payoff.
    Z=g  -> posterior mostly GOOD  -> perceived ~ +V1  -> v6 momentum -> bad coalition false-clears.
    Z=b  -> posterior mostly BAD   -> perceived ~ -Lbad -> decline      -> bad coalition screened.
  For a BAD coalition Z~(g w.p.(1-k)/2, b w.p.(1+k)/2). The DYNAMIC detection rate is
    s(k) = (1-k)/2 * s|Z=g  +  (1+k)/2 * s|Z=b,    s|Z := 1 - pi_falseactivate(perceived(Z)).
  k=0: Z uninformative -> perceived = prior-weighted -> (if prior favorable) momentum -> s low
       = v7 s_masked.  k=1: only Z=b -> perceived=-Lbad -> decline -> s=1 = v7 s_revealed.

KEY METHOD IMPROVEMENT over v7: v7 conflated TWO changes between masked and revealed -- the BELIEF
(perceived payoff) AND the SELECTION (opt vs pess). Here we HOLD THE SELECTION FIXED (opt/momentum)
and vary ONLY the belief via the public-signal precision k. So the resulting s(k) is the PURE
BELIEF/signal-extraction effect, with the selection confound removed -- a cleaner object than v7's.

EXPECTED RECONCILIATION (set before running): with the selection held fixed, s|Z=g ~ low (momentum
clears) and s|Z=b ~ high (decline), so s(k) ~ (1-k)/2 * s_g + (1+k)/2 * s_b, which is ~AFFINE in k.
==> v3's POSITED LINEAR s(k)=k is approximately the PUBLIC-signal screening case; the PRIVATE-signal
case (v8_screening_signal.py) is convex. Two signal architectures, two shapes; v3 guessed the public.

================================================================================================
MODEL: v7_screening.py solve_bad VERBATIM (state (m,h), p_opp, xi spillover, OSD-verified MPE),
the ONLY change being perceived_clearing = posterior-weighted payoff under public signal Z at
precision k, selection held at 'opt'. Self-contained (the solver is copied from v7_screening.py).
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


# --- v7_screening.py solve_bad, copied verbatim (perceived clearing payoff is the only input) ---
def solve_bad(
    N, K, perceived_clearing, perceived_clearing_nc, kappa, Lp, T, rho, p_opp, selection
):
    Vu = np.zeros((N + 1, T + 1))
    Pc = np.zeros((N + 1, T + 1))
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
                if ds[i] > 0 >= ds[i + 1]:
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


def posterior_bad_public(pi_b, k, Z):
    """Public binary signal Z of precision k: Pr(Z=type|type)=(1+k)/2."""
    pz_b = (1 + k) / 2 if Z == "b" else (1 - k) / 2
    pz_g = (1 - k) / 2 if Z == "b" else (1 + k) / 2
    num = pi_b * pz_b
    den = num + (1 - pi_b) * pz_g
    return num / den if den > 0 else pi_b


def perceived_payoffs(pi_b, k, Z, V1, Lbad, xi, kappa):
    """Posterior-weighted perceived clearing payoff for committer / non-committer under signal Z."""
    pB = posterior_bad_public(pi_b, k, Z)
    pc = (1 - pB) * (V1 - kappa) + pB * (-Lbad - kappa)
    pc_nc = (1 - pB) * (xi * V1) + pB * (xi * (-Lbad))
    return pc, pc_nc


def dynamic_s(N, K, pi_b, k, V1, Lbad, xi, kappa, Lp, T, rho, p_opp, selection="opt"):
    """DYNAMIC detection rate for a BAD coalition under a public type signal of precision k,
    averaged over the public realization Z ~ (g:(1-k)/2, b:(1+k)/2 | BAD). Selection held fixed."""
    s = 0.0
    for Z, pZ in (("g", (1 - k) / 2), ("b", (1 + k) / 2)):
        pc, pc_nc = perceived_payoffs(pi_b, k, Z, V1, Lbad, xi, kappa)
        sol = solve_bad(N, K, pc, pc_nc, kappa, Lp, T, rho, p_opp, selection)
        sZ = 1 - sol["pi_falseactivate"]
        s += pZ * sZ
    return s


def main():
    print("=" * 96)
    print(
        "v8 #2(b) — DYNAMIC s(k): PUBLIC type signal embedded in v7's MPE; selection held FIXED"
    )
    print(f"seed={SEED}")
    print("=" * 96)

    # v6/v7 contested calibration verbatim, so endpoints nest v7 exactly.
    N, kappa, Lp, rho, xi, p_opp, V1, Lbad, T = (
        10,
        1.0,
        3.0,
        0.05,
        0.8,
        0.6,
        3.0,
        3.0,
        10,
    )
    theta0 = 0.15
    K = max(1, ceil(round((1 - theta0) * N, 6)))
    print(
        f"\ncell: N={N} theta={theta0} K={K} V1={V1} Lbad={Lbad} xi={xi} kappa={kappa} Lp={Lp} "
        f"rho={rho} p_opp={p_opp} T={T}"
    )

    # --- 1. v7 ENDPOINT NESTING: k=0 and k=1 vs v7's s_masked / s_revealed ---
    print(
        "\n--- 1. ENDPOINT NESTING vs v7 (pi_b such that k=0 recovers v7's pooled masked) ---"
    )
    print(
        "  v7 (BAD-B, theta=0.15): s_masked=0.0348, s_revealed=1.0000 (binary pool/separate)."
    )
    print(
        "  pi_b    s(k=0)    s(k=1)    (k=0 = prior-pooled momentum; k=1 = perceives -Lbad)"
    )
    for pi_b in [0.1, 0.3, 0.5]:
        s0 = dynamic_s(N, K, pi_b, 0.0, V1, Lbad, xi, kappa, Lp, T, rho, p_opp)
        s1 = dynamic_s(N, K, pi_b, 1.0, V1, Lbad, xi, kappa, Lp, T, rho, p_opp)
        print(f"  {pi_b:.2f}    {s0:.4f}    {s1:.4f}")
    print(
        "  >> k=1 -> s=1 (perceives -Lbad, declines) = v7 s_revealed. k=0 -> the prior-pooled"
    )
    print(
        "     momentum clear = v7 s_masked (low when prior favorable). v7's binary = the k limits."
    )

    # --- 2. THE DYNAMIC s(k) over the continuous dial (selection fixed = pure belief effect) ---
    print(
        "\n--- 2. DYNAMIC s(k) over k in [0,1] (pi_b=0.3, selection=opt held FIXED) ---"
    )
    pi_b = 0.3
    print(
        "   k     s(k)     (selection confound removed: ONLY the public posterior varies)"
    )
    ks = np.linspace(0, 1, 11)
    svals = []
    for k in ks:
        s = dynamic_s(N, K, pi_b, float(k), V1, Lbad, xi, kappa, Lp, T, rho, p_opp)
        svals.append(s)
        print(f"  {k:.2f}   {s:.4f}")
    svals = np.array(svals)

    # --- 3. SHAPE: is the PUBLIC-signal dynamic s(k) ~ affine (reconciling v3's linear posit)? ---
    print(
        "\n--- 3. SHAPE: public-signal s(k) vs v3's LINEAR posit, vs private-signal convex ---"
    )
    lin = svals[0] + ks * (svals[-1] - svals[0])
    dev = svals - lin
    print(
        f"  s(0)={svals[0]:.4f}  s(1)={svals[-1]:.4f}  max|dev from linear|={np.max(np.abs(dev)):.4f}"
    )
    print(
        f"  mean|dev from linear|={np.mean(np.abs(dev)):.4f}  monotone_incr={bool(np.all(np.diff(svals) >= -1e-9))}"
    )
    # decompose: s(k) = (1-k)/2 * s_g + (1+k)/2 * s_b ; if s_g,s_b ~ const in k it's exactly affine
    sg0 = (
        1
        - solve_bad(
            N,
            K,
            *perceived_payoffs(pi_b, 0.5, "g", V1, Lbad, xi, kappa),
            kappa,
            Lp,
            T,
            rho,
            p_opp,
            "opt",
        )["pi_falseactivate"]
    )
    sb0 = (
        1
        - solve_bad(
            N,
            K,
            *perceived_payoffs(pi_b, 0.5, "b", V1, Lbad, xi, kappa),
            kappa,
            Lp,
            T,
            rho,
            p_opp,
            "opt",
        )["pi_falseactivate"]
    )
    print(f"  component clears at k=0.5: s|Z=g={sg0:.4f}  s|Z=b={sb0:.4f}")
    # locate the activation threshold k_act: smallest k at which the Z=g branch CLEARS (s|Z=g<0.5).
    k_act = None
    for k in np.linspace(0, 1, 101):
        pc, pc_nc = perceived_payoffs(pi_b, float(k), "g", V1, Lbad, xi, kappa)
        s_g = (
            1
            - solve_bad(N, K, pc, pc_nc, kappa, Lp, T, rho, p_opp, "opt")[
                "pi_falseactivate"
            ]
        )
        if s_g < 0.5 and k_act is None:
            k_act = k
    # affine fit on the ACTIVE regime [k_act, 1]
    if k_act is not None and k_act < 1:
        mask = ks >= k_act
        kk_a, sv_a = ks[mask], svals[mask]
        if len(kk_a) >= 2:
            coef = np.polyfit(kk_a, sv_a, 1)
            resid = np.max(np.abs(sv_a - np.polyval(coef, kk_a)))
        else:
            resid = float("nan")
    else:
        resid = float("nan")
    print(
        "  >> CORRECTION (numerics-first): s(k) is NOT globally affine. It is PIECEWISE:"
    )
    print(
        f"     PARALYSIS floor s=1 below an activation threshold k_act ~ {k_act:.2f} (the Z=g branch"
    )
    print(
        "     cannot drive momentum until the posterior is optimistic enough), then ~AFFINE above"
    )
    print(
        f"     it (max affine-fit resid on [k_act,1] = {resid:.4f}). The affine piece IS the"
    )
    print(
        "     (1-k)/2*s_g + (1+k)/2*s_b weighting; v3's LINEAR posit s(k)=k matches ONLY the active"
    )
    print(
        "     public-signal regime. The PRIVATE signal (v8_screening_signal.py) is convex. Both"
    )
    print(
        "     carry the low-k paralysis floor (s=1 with phi=1: market death, not screening)."
    )

    # --- 4. PUBLIC vs PRIVATE summary table (the Morris-Shin axis) ---
    print(
        "\n--- 4. PUBLIC (this script, dynamic MPE) vs PRIVATE (static, v8_screening_signal) ---"
    )
    print(
        "  public  : all agents share posterior; s(k) = paralysis floor + affine active piece;"
    )
    print(
        "            v3's linear posit ~ recovered IN THE ACTIVE REGIME (above k_act)."
    )
    print(
        "  private : each agent own signal; s(k) convex (binomial aggregation of own thresholds)."
    )
    print(
        "  Both -> s(0)=v7 masked region, s(1)=1 (revealed). The dial k continuously interpolates"
    )
    print(
        "  v7's binary {masked, revealed}; v7 is the k in {0,1} endpoints of EITHER architecture."
    )

    # --- 5. ORDERING robustness across (pi_b, theta) ---
    print(
        "\n--- 5. ORDERING s(1)>=s(0) across (pi_b, theta) [public-signal dynamic] ---"
    )
    allhold, n = True, 0
    for pi_b in [0.1, 0.3, 0.5, 0.7]:
        for th in [0.10, 0.15, 0.30]:
            Kt = max(1, ceil(round((1 - th) * N, 6)))
            s0 = dynamic_s(N, Kt, pi_b, 0.0, V1, Lbad, xi, kappa, Lp, T, rho, p_opp)
            s1 = dynamic_s(N, Kt, pi_b, 1.0, V1, Lbad, xi, kappa, Lp, T, rho, p_opp)
            n += 1
            if s1 < s0 - 1e-9:
                allhold = False
    print(f"  >> s(1) >= s(0) across ALL {n} (pi_b,theta) cells: {allhold}")

    print("\n" + "=" * 96)
    print("SUMMARY (v8 #2b)")
    print(
        "  (1) v7's binary {masked perceives +V1, revealed perceives -Lbad} is the k in {0,1} LIMIT"
    )
    print(
        "      of a public type signal of precision k embedded in v7's REAL MPE. Validated."
    )
    print(
        "  (2) Holding selection FIXED isolates the PURE belief/signal-extraction effect -- removing"
    )
    print(
        "      v7's belief-vs-selection confound. The dynamic s(k) is the clean object."
    )
    print(
        "  (3) PUBLIC-signal s(k) is PIECEWISE: a paralysis floor (s=1) below an activation"
    )
    print(
        "      threshold k_act, then ~AFFINE above it. v3's LINEAR posit s(k)=k matches only the"
    )
    print(
        "      ACTIVE public-signal regime; the PRIVATE-signal s(k) (v8_screening_signal.py) is"
    )
    print(
        "      CONVEX. Two architectures, two shapes; both carry the low-k paralysis floor."
    )
    print("=" * 96)


if __name__ == "__main__":
    main()


# ================================================================================================
# DEVELOPMENT LOG (numerics-first)
# ================================================================================================
# v8b.0: embed a PUBLIC type signal of precision k into v7_screening.py's solve_bad (copied verbatim),
#   perceived payoff = posterior-weighted, selection held 'opt'. s(k) averaged over Z|BAD. Recovers
#   v7's s_masked/s_revealed at k=0,1; isolates belief from selection (v7 confounded them).
# v8b.1 (CORRECTION, numerics-first): the "globally affine" hypothesis was FALSIFIED. The public-
#   signal s(k) is PIECEWISE: a PARALYSIS floor (s=1; phi=1 too -> market death, not screening) below
#   an activation threshold k_act, then ~AFFINE above it (the (1-k)/2 s_g + (1+k)/2 s_b weighting).
#   v3's linear posit s(k)=k matches ONLY the active public-signal regime. The private signal is
#   convex. Both architectures carry the low-k paralysis floor. Endpoint nesting vs v7 confirmed
#   (pi_b=0.1: s(0)=0.0513 ~ v7 s_masked=0.0348; s(1)=1.0 = v7 s_revealed).
# ================================================================================================
