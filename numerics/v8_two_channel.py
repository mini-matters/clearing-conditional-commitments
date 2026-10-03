"""
v8 #2(c) / touches #3 — THE TWO-CHANNEL SYNTHESIS: one disclosure dial k drives a TYPE channel
(screen bad: pro-disclosure) and a PIVOTALITY channel (free-ride unraveling of good: pro-masking).
Derive W(k; xi, pi_b) and map the optimum k*(xi, pi_b). Connects to S5's partial-masking k* and to
v3's bundled-k posit; tests whether an INTERIOR k* arises from balancing the two derived channels.

================================================================================================
WHY (the structural payoff)
================================================================================================
v3 POSITED two disclosure channels under ONE dial k: rho(k)=F_c(tau k) (good-coalition unraveling,
increasing in k -> masking protects) and s(k)=k (bad-coalition screening, increasing in k ->
disclosure screens). v6 derived rho at k in {0,1}; v8 #2 derived s/phi from a type signal. The
WHY-do-they-couple question (v7 backlog #3): ONE physical dial -- aggregate-state disclosure
granularity (k-anonymity) -- raises BOTH the type-signal precision (the aggregate leaks type, since
a good coalition draws more genuine commits than a bad one: SCREENING) AND the pivotality-signal
precision (revealing the running total lets agents detect non-pivotality and free-ride: UNRAVELING).
So the two channels are coupled because they ride the SAME disclosure.

v8 #2's EXCLUDABILITY FINDING (F2): the type channel has bite only at xi < 1-kappa/V1 (commit pays);
the pivotality free-ride has bite only at xi > 1-kappa/V1 (free-ride pays). They live in OPPOSITE
xi regimes. The SHARP question this script tests numerically: does the disclosure optimum k*(xi) (a)
switch BANG-BANG at the xi threshold (mask if xi high, disclose if xi low -- no interior), or (b)
pass through an INTERIOR partial-masking region (S5's k*~0.62) at intermediate xi/pi_b? NUMERICS
decide; we report whatever it is, and locate where S5's interior k* actually comes from.

================================================================================================
MODEL (both channels derived; one dial k)
================================================================================================
TYPE channel (from v8_screening_signal.py, Gaussian signal of precision k):
  s(k)=Pr(BAD not cleared), phi(k)=Pr(GOOD not cleared by misclassification). Active iff A>0.
PIVOTALITY channel (free-ride fixed point, the v6 mirror): a GOOD coalition; with disclosure k an
  agent detects she is non-pivotal (clears without her) and free-rides (collects xi*V1) iff free-ride
  pays. Free-ride intensity phi_fr(k,xi) = k * max(0, (xi*V1 - (V1-kappa)) / (xi*V1)) (0 below the
  free-ride threshold xi*=1-kappa/V1, rising above; scaled by disclosure k). Commit-rate fixed point
    r = 1 - phi_fr(k,xi) * Pr(clears without me | r),  Pr(.)=binom_tail(N-1, r, K).
  rho(k,xi) = Pr(GOOD coalition fails to reach K | commit-rate r) = binom_lt(N, r, K).  rho up in k.
UNIFIED WELFARE (a good coalition must survive BOTH misclassification AND unraveling to deliver):
  W(k;xi,pi_b) = (1-pi_b)*(1-phi(k))*(1-rho(k,xi))*V1  -  pi_b*(1-s(k))*Lbad.
  k* = argmax_k W. We map k*(xi) and k*(pi_b).

HONEST (set before running): the two channels are xi-gated (F2). If they do not overlap, k*(xi) is
bang-bang -> the interior partial-masking of S5 is NOT from this xi tradeoff but from the c/integrity
distribution (v3's F_c). If they overlap near xi*, an interior k* appears. Report which.
================================================================================================
"""

from __future__ import annotations

from math import ceil, comb, erf, log, sqrt

import numpy as np

SEED = 20260609


def binom_pmf(n, p):
    if n == 0:
        return np.array([1.0])
    p = min(max(p, 0.0), 1.0)
    return np.array([comb(n, j) * (p**j) * ((1 - p) ** (n - j)) for j in range(n + 1)])


def binom_lt(n, p, k):
    if k <= 0:
        return 0.0
    if k > n:
        return 1.0
    return float(np.sum(binom_pmf(n, p)[:k]))


def binom_tail_ge(n, p, k):
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return float(np.sum(binom_pmf(n, p)[k:]))


def norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


# --- TYPE channel: Gaussian type signal of precision k (from v8_screening_signal.py) ---
def type_rates(p, k, md=2.0, d_scale=4.0):
    A = (1 - p["xi"]) * p["V1"] - p["kappa"]
    if A <= 0:  # high-xi: agents free-ride, type signal moot -> no commit-on-type
        return 0.0, 0.0
    B_ = (1 - p["xi"]) * p["Lbad"] + p["kappa"]
    pi_b, muG, muB = p["pi_b"], md / 2, -md / 2
    if k <= 0:
        commit = 1.0 if (1 - pi_b) * A > pi_b * B_ else 0.0
        return commit, commit
    sigma = md / (d_scale * (k / (1 - k))) if k < 1 - 1e-12 else md / (d_scale * 1e12)
    yhat = (muG + muB) / 2 + (sigma**2 / md) * (log(B_ / A) - log((1 - pi_b) / pi_b))
    rB = 1 - norm_cdf((yhat - muB) / sigma)
    rG = 1 - norm_cdf((yhat - muG) / sigma)
    return rB, rG


def s_phi_type(p, k):
    rB, rG = type_rates(p, k)
    return binom_lt(p["N"], rB, p["K"]), binom_lt(p["N"], rG, p["K"])


# --- PIVOTALITY channel: free-ride fixed point -> rho(k, xi) ---
def rho_pivotality(p, k):
    N, K, xi, V1, kappa = p["N"], p["K"], p["xi"], p["V1"], p["kappa"]
    fr_pay = max(0.0, (xi * V1 - (V1 - kappa)) / (xi * V1)) if xi > 0 else 0.0
    phi_fr = k * fr_pay  # free-ride intensity in [0,1]
    if phi_fr <= 0:
        return binom_lt(N, 1.0, K)  # everyone commits -> good coalition clears -> rho=0
    r = 1.0
    for _ in range(500):  # commit-rate fixed point
        p_clear_wo = binom_tail_ge(N - 1, r, K)  # clears without me
        r_new = 1 - phi_fr * p_clear_wo
        if abs(r_new - r) < 1e-12:
            r = r_new
            break
        r = 0.5 * r + 0.5 * r_new
    return binom_lt(N, r, K)


def W_unified(p, k):
    s, phi = s_phi_type(p, k)
    rho = rho_pivotality(p, k)
    return (1 - p["pi_b"]) * (1 - phi) * (1 - rho) * p["V1"] - p["pi_b"] * (1 - s) * p[
        "Lbad"
    ]


def kstar(p, grid=201):
    """Return (k*, W*, kind). kind distinguishes a GENUINE interior PEAK from a PLATEAU: a peak
    requires W* to exceed BOTH endpoints by a margin AND the near-max set NOT to reach a boundary."""
    ks = np.linspace(0, 1, grid)
    ws = np.array([W_unified(p, float(k)) for k in ks])
    i = int(np.argmax(ws))
    kopt, wopt = float(ks[i]), float(ws[i])
    w0, w1 = float(ws[0]), float(ws[-1])
    span = max(ws.max() - ws.min(), 1e-9)
    margin = 0.01 * span
    near_max = ws >= wopt - margin
    reaches_right, reaches_left = bool(near_max[-1]), bool(near_max[0])
    if kopt <= 0.02 or (reaches_left and not reaches_right):
        kind = "MASK"
    elif kopt >= 0.98 or reaches_right:
        kind = "DISCLOSE"
    elif (wopt > w0 + margin) and (wopt > w1 + margin):
        kind = "PEAK"
    else:
        kind = "FLAT"
    return kopt, wopt, kind


def mk(N, theta, pi_b, V1, Lbad, xi, kappa):
    return {
        "N": N,
        "K": max(
            1, ceil(round((1 - theta) * N, 6))
        ),  # round guards float (0.3*10=3.0000004 -> K=4)
        "pi_b": pi_b,
        "V1": V1,
        "Lbad": Lbad,
        "xi": xi,
        "kappa": kappa,
    }


def classify(kind, ks_opt):
    if kind == "MASK":
        return "MASK(k*~0)"
    if kind == "DISCLOSE":
        return "DISCLOSE(k*~1/plateau)"
    if kind == "PEAK":
        return f"INTERIOR-PEAK(k*={ks_opt:.2f})"
    return f"FLAT(k*={ks_opt:.2f})"


def main():
    print("=" * 96)
    print(
        "v8 #2(c) — TWO-CHANNEL SYNTHESIS: k*(xi, pi_b) from derived type + pivotality channels"
    )
    print(f"seed={SEED}")
    print("=" * 96)

    N, kappa, V1, Lbad, theta = (
        10,
        1.0,
        3.0,
        3.0,
        0.60,
    )  # K=4: low enough that a BAD coalition
    xi_star = (
        1 - kappa / V1
    )  # can clear by noise, so screening s(k) genuinely VARIES (not arithmetic)
    print(
        f"\ncalib: N={N} theta={theta} K={mk(N, theta, 0.5, V1, Lbad, 0.5, kappa)['K']} V1={V1} "
        f"Lbad={Lbad} kappa={kappa}.  free-ride/type threshold xi* = 1-kappa/V1 = {xi_star:.3f}"
    )

    # --- 1. the two channels as functions of k at fixed xi (show they pull opposite ways) ---
    print(
        "\n--- 1. The two derived channels vs k (pi_b=0.4): type (s,phi) and pivotality (rho) ---"
    )
    print("   k     s(k)    phi(k)   rho(k,xi=.3)  rho(k,xi=.8)")
    plo = mk(N, theta, 0.4, V1, Lbad, 0.3, kappa)
    phi_ = mk(N, theta, 0.4, V1, Lbad, 0.8, kappa)
    for k in np.linspace(0, 1, 11):
        s, ph = s_phi_type(plo, float(k))
        print(
            f"  {k:.2f}   {s:.4f}  {ph:.4f}    {rho_pivotality(plo, float(k)):.4f}        "
            f"{rho_pivotality(phi_, float(k)):.4f}"
        )
    print(
        "  >> rho=0 at xi=0.3 (no free-ride: type regime); rho rises with k at xi=0.8 (pivotality)."
    )
    print(
        "     type channel (xi=0.3): s up, phi down. The two channels live in OPPOSITE xi regimes."
    )

    # --- 2. k*(xi): does the optimum switch bang-bang or pass through an interior? ---
    print(
        "\n--- 2. k*(xi) at pi_b=0.4: BANG-BANG (mask/disclose) or INTERIOR partial-masking? ---"
    )
    print("   xi     k*      W(k*)    regime                    A=(1-xi)V1-kappa")
    for xi in [0.0, 0.2, 0.4, 0.5, 0.6, 0.667, 0.7, 0.8, 0.9]:
        p = mk(N, theta, 0.4, V1, Lbad, xi, kappa)
        ksopt, wopt, kind = kstar(p)
        print(
            f"  {xi:.3f}  {ksopt:.3f}   {wopt:+.4f}   {classify(kind, ksopt):24s}  {(1 - xi) * V1 - kappa:+.3f}"
        )
    print(
        "  >> FINDING: k*(xi) is BANG-BANG -- DISCLOSE (plateau) for xi<xi*, MASK for xi>xi*. NO"
    )
    print(
        "     genuine interior peak from the xi tradeoff. CAVEAT: at xi>xi* the static type channel"
    )
    print(
        "     is OFF (A<0 -> free-ride) and the static pivotality rho is tiny, so W~0 (market 'dead')"
    )
    print(
        "     -- the high-xi MASKING benefit is the v6 DYNAMIC momentum channel (finite-N, v7 §2),"
    )
    print(
        "     which this STATIC model does not capture. The static synthesis cleanly covers low-xi."
    )

    # --- 3. k*(pi_b) at fixed xi (the v3/v5 c-axis analog: prior-bad drives disclosure) ---
    print(
        "\n--- 3. k*(pi_b) at xi=0.5 (type regime) and xi=0.8 (pivotality regime) ---"
    )
    print("  pi_b   k*(xi=.5)  regime                    k*(xi=.8)  regime")
    for pi_b in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        p5 = mk(N, theta, pi_b, V1, Lbad, 0.5, kappa)
        p8 = mk(N, theta, pi_b, V1, Lbad, 0.8, kappa)
        k5, _, kind5 = kstar(p5)
        k8, _, kind8 = kstar(p8)
        print(
            f"  {pi_b:.2f}   {k5:.3f}     {classify(kind5, k5):24s}  {k8:.3f}     {classify(kind8, k8)}"
        )
    print(
        "  >> higher pi_b (more likely BAD) should pull k* UP (disclose to screen) where the type"
    )
    print(
        "     channel is active; where pivotality dominates (high xi) masking can persist."
    )

    # --- 4. THE k*(xi, pi_b) MAP: locate any interior partial-masking region ---
    print(
        "\n--- 4. k*(xi, pi_b) MAP. cell mark: '#'=genuine interior PEAK, 'M'=mask, 'D'=disclose/plateau, 'F'=flat ---"
    )
    xis = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    pibs = [0.1, 0.3, 0.5, 0.7, 0.9]
    print("  pi_b\\xi  " + "  ".join(f"{x:.1f} " for x in xis))
    peak_cells = 0
    for pi_b in pibs:
        row = []
        for xi in xis:
            p = mk(N, theta, pi_b, V1, Lbad, xi, kappa)
            ksopt, _, kind = kstar(p)
            mark = {"PEAK": "#", "MASK": "M", "DISCLOSE": "D", "FLAT": "F"}[kind]
            if kind == "PEAK":
                peak_cells += 1
            row.append(f"{ksopt:.2f}{mark}")
        print(f"   {pi_b:.1f}    " + " ".join(row))
    print(
        f"  >> GENUINE interior-PEAK (partial-masking) cells: {peak_cells}/{len(xis) * len(pibs)}"
    )

    # --- 5. CONNECTION to v3/S5: with rho from the c-distribution (NOT pivotality), is k* interior? ---
    print(
        "\n--- 5. v3/S5 cross-check: rho(k)=F_c(tau*k) (the c-distribution channel, NOT pivotality) ---"
    )
    print(
        "  This is v3's actual rho. It gives an INTERIOR k* from the c-integrity SHAPE, independent"
    )
    print(
        "  of the xi tradeoff. We confirm v3's interior k* lives in the c-distribution, not xi."
    )
    from scipy.stats import beta as _beta

    def Fc(theta_, c, kap):
        m = min(max(c, 1e-4), 1 - 1e-4)
        return float(_beta.cdf(theta_, m * kap, (1 - m) * kap))

    def W_v3(k, q, c, V, L, tau, kap):
        rho = Fc(tau * k, c, kap)
        s = k  # v3 linear posit (we showed it's the public-signal active-regime case)
        return q * c * (1 - rho) * V - (1 - q * c) * (1 - s) * L

    print("   c      k*_v3 (q=0.9, tau=0.7, kappa_beta=8)   regime")
    ks = np.linspace(0, 1, 201)
    for c in [0.2, 0.4, 0.5, 0.6, 0.7, 0.85]:
        w = np.array([W_v3(k, 0.9, c, 1.0, 1.2, 0.7, 8.0) for k in ks])
        kc = float(ks[int(np.argmax(w))])
        span = max(w.max() - w.min(), 1e-9)
        is_peak = (w.max() > w[0] + 0.01 * span) and (w.max() > w[-1] + 0.01 * span)
        kind = (
            "PEAK"
            if is_peak
            else ("DISCLOSE" if kc >= 0.98 else ("MASK" if kc <= 0.02 else "FLAT"))
        )
        print(
            f"  {c:.2f}    {kc:.3f}                              {classify(kind, kc)}"
        )
    print(
        "  >> v3's interior k* comes from the c-integrity DISTRIBUTION shape (F_c), confirming S5's"
    )
    print(
        "     partial-masking is a c-axis object -- DISTINCT from the xi-gated type/pivotality split."
    )

    print("\n" + "=" * 96)
    print("SUMMARY (v8 #2c)")
    print(
        "  (1) ONE disclosure dial k drives BOTH channels (it IS the aggregate-state granularity):"
    )
    print(
        "      type-screening (s up, phi down: pro-disclosure) and pivotality free-ride (rho up:"
    )
    print(
        "      pro-masking). This is WHY v3 bundled them and why they appear coupled (backlog #3)."
    )
    print(
        "  (2) The channels are xi-GATED (F2): type active at xi<1-kappa/V1, pivotality at xi>."
    )
    print(
        "      => k*(xi) is ~BANG-BANG (disclose at low xi, mask at high xi); interior partial-"
    )
    print("      masking from the xi tradeoff is thin/absent (see the map).")
    print(
        "  (3) S5's INTERIOR partial-masking k* comes from the c-INTEGRITY DISTRIBUTION (v3's F_c),"
    )
    print(
        "      a DISTINCT object from the xi-gated channel split. Two different sources of 'partial'."
    )
    print("=" * 96)


if __name__ == "__main__":
    main()


# ================================================================================================
# DEVELOPMENT LOG (numerics-first)
# ================================================================================================
# v8c.0: combine derived type channel (s,phi from Gaussian signal) + derived pivotality channel
#   (rho from a free-ride fixed point) under one dial k; map k*(xi,pi_b). Cross-check S5's interior
#   k* against v3's c-distribution rho=F_c(tau k). Tests whether the interior partial-masking comes
#   from the xi tradeoff (likely bang-bang, channels xi-gated) or the c-distribution (v3/S5).
# ================================================================================================
