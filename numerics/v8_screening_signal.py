"""
v8 #2 — DERIVE the bad-coalition screening rate s(k) from an INFORMATION-THEORETIC PRIMITIVE
(signal-extraction on the coalition TYPE), replacing v7's operationalized binary pool-vs-separate.

================================================================================================
THE GAP (v7 sec 9 item 2)
================================================================================================
v7 #4 made s an MPE object but priced it as "INSTANTIATES, does not derive": the masked number is
LITERALLY the good-coalition false-clear (masking ASSUMED to pool good+bad so the agent perceives
+V1) and s_revealed ~ 1 because revelation is ASSUMED perfect separation. Both are degenerate
extremes of a BINARY belief assumption k in {0 pool, 1 separate}; the gap rests on a chosen payoff.

v8: disclosure granularity is a CONTINUOUS dial k in [0,1] = PRECISION of a private signal each
agent gets about the coalition TYPE (good=deliverable / bad=illusory). DERIVE the detection rate
s(k) as a Bayesian signal-extraction object: v7's extremes become LIMITS; the interior, the SHAPE,
and the GAP are derived from (signal precision k, prior pi_b, payoff asymmetry, closeness K/N).

================================================================================================
THREE FALSIFICATIONS THAT SHAPED v8 (numerics-first; full log at bottom)
================================================================================================
F1 (one-shot coordination game, v6 payoffs, xi=0.8, K/N=0.9): ALL cells -> all-decline (s=phi=1),
   nothing activates. The all-decline collapse is a PIVOTALITY/coordination artifact, NOT detection;
   it must be separated from the signal-extraction question.
F2 (xi=0.8): A:=(1-xi)V1-kappa = -0.4 < 0 -> free-ride dominates committing even to a GOOD clear,
   so the type SIGNAL is irrelevant -- pure volunteer's dilemma (the v6 PIVOTALITY channel). The
   TYPE/screening channel has bite ONLY at A>0  <=>  xi < 1-kappa/V1.  ==> the screening (type)
   and protection (pivotality) channels live in OPPOSITE EXCLUDABILITY REGIMES. (This is WHY v7 had
   to assume pooling: at xi=0.8 the type channel cannot produce screening on its own.)
F3 (binary 0/1 Bayes rule): s(k) is KNIFE-EDGE (jumps between commit-neither / commit-g / commit-
   both regimes). At low k "s=1" is PARALYSIS (commit-neither: phi=1 too -- good coalitions die),
   NOT screening. The smooth, honest derivation needs a CONTINUOUS (Gaussian) signal -- the
   canonical Morris-Shin / Carlsson-van Damme global-game structure, lambda-FREE.

================================================================================================
MODEL
================================================================================================
Binary type psi in {G,B}, prior pi_b=Pr(B). N agents, K=ceil((1-theta)N) commits to clear.
PAYOFFS (clearing branch = v6/v7; failure = 0 NO-CUSTODY refund, NOT -Lp; see F1):
  committer clears G: V1-kappa   committer clears B: -Lbad-kappa
  waiter    clears G: xi*V1      waiter    clears B: xi*(-Lbad)     failure: 0 both.
Single-agent Bayes-optimal commit rule GIVEN clearing:
  commit iff Pr(G|y)*A > Pr(B|y)*B_,  A=(1-xi)V1-kappa,  B_=(1-xi)Lbad+kappa   (needs A>0).
SIGNAL (two structures):
  BINARY channel quality k: Pr(z=psi|psi)=(1+k)/2.  (regime-structure view; knife-edge)
  GAUSSIAN: y=mu_psi+eps, eps~N(0,sigma(k)^2), discriminability d'(k)=d_scale*k/(1-k) so
            k->0 no info, k->1 perfect. Bayes cutoff y_hat from the break-even odds. (smooth view)
DETECTION (derived):  r_psi = Pr(commit | psi);  s(k)=Pr(Binom(N,r_B)<K) [screen bad],
  phi(k)=Pr(Binom(N,r_G)<K) [false-reject good = the disclosure COST].
WELFARE of the type channel:  W_type(k) = (1-pi_b)(1-phi(k))*V1 - pi_b*(1-s(k))*Lbad.

DERIVES: s(k), phi(k) as signal-extraction objects; limits, shape, gap from primitives; the
  excludability-regime split; that the type channel is a one-way ratchet toward disclosure.
INSTANTIATES: the signal family (binary AND Gaussian both built -> robustness); selection in the
  strategic layer (opt/pess/RD). No unique closed form claimed.
================================================================================================
"""

from __future__ import annotations

from math import comb, ceil, erf, log, sqrt

import numpy as np

SEED = 20260609


def binom_pmf(n: int, p: float) -> np.ndarray:
    if n == 0:
        return np.array([1.0])
    p = min(max(p, 0.0), 1.0)
    return np.array([comb(n, j) * (p**j) * ((1 - p) ** (n - j)) for j in range(n + 1)])


def binom_lt(n: int, p: float, k: int) -> float:
    if k <= 0:
        return 0.0
    if k > n:
        return 1.0
    return float(np.sum(binom_pmf(n, p)[:k]))


def binom_ge(n: int, p: float, k: int) -> float:
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return float(np.sum(binom_pmf(n, p)[k:]))


def norm_cdf(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


# ================================================================================================
# SIGNAL PRIMITIVES
# ================================================================================================
def posterior_bad_binary(pi_b, k, z):
    pz_b = (1 + k) / 2 if z == "b" else (1 - k) / 2
    pz_g = (1 - k) / 2 if z == "b" else (1 + k) / 2
    num = pi_b * pz_b
    den = num + (1 - pi_b) * pz_g
    return num / den if den > 0 else pi_b


def sig_prob_binary(k, z, type_):
    correct, wrong = (1 + k) / 2, (1 - k) / 2
    if type_ == "G":
        return correct if z == "g" else wrong
    return correct if z == "b" else wrong


def bayes_commit_binary(pi_b, k, z, V1, Lbad, xi, kappa):
    A = (1 - xi) * V1 - kappa
    if A <= 0:
        return 0
    B_ = (1 - xi) * Lbad + kappa
    pB = posterior_bad_binary(pi_b, k, z)
    return 1 if (1 - pB) * A > pB * B_ else 0


def commit_rates_binary(p, k):
    cg = bayes_commit_binary(p["pi_b"], k, "g", p["V1"], p["Lbad"], p["xi"], p["kappa"])
    cb = bayes_commit_binary(p["pi_b"], k, "b", p["V1"], p["Lbad"], p["xi"], p["kappa"])
    rB = sig_prob_binary(k, "g", "B") * cg + sig_prob_binary(k, "b", "B") * cb
    rG = sig_prob_binary(k, "g", "G") * cg + sig_prob_binary(k, "b", "G") * cb
    return rB, rG, cg, cb


# Gaussian: type means symmetric mu_G=+md/2, mu_B=-md/2 (md = mean separation). Signal noise
# sigma(k) set so discriminability d'(k)=md/sigma(k)=d_scale*k/(1-k): k->0 d'=0, k->1 d'->inf.
def commit_rates_gauss(p, k, md=2.0, d_scale=4.0):
    A = (1 - p["xi"]) * p["V1"] - p["kappa"]
    if A <= 0:
        return 0.0, 0.0
    B_ = (1 - p["xi"]) * p["Lbad"] + p["kappa"]
    pi_b = p["pi_b"]
    muG, muB = md / 2, -md / 2
    if k <= 0:  # no info: posterior=prior; commit iff prior clears the break-even
        commit = 1.0 if (1 - pi_b) * A > pi_b * B_ else 0.0
        return commit, commit
    if k >= 1 - 1e-12:
        sigma = md / (d_scale * 1e12)
    else:
        sigma = md / (d_scale * k / (1 - k))
    # Bayes cutoff y_hat: posterior-odds-good(y) = B_/A.  log-odds linear in y.
    # ln[(1-pi_b)/pi_b] + (md/sigma^2)*(y - (muG+muB)/2) = ln(B_/A)
    yhat = (muG + muB) / 2 + (sigma**2 / md) * (log(B_ / A) - log((1 - pi_b) / pi_b))
    rB = 1 - norm_cdf((yhat - muB) / sigma)  # Pr(y>yhat | B)
    rG = 1 - norm_cdf((yhat - muG) / sigma)
    return rB, rG


def s_phi(p, k, mode="binary", **kw):
    if mode == "binary":
        rB, rG, _, _ = commit_rates_binary(p, k)
    else:
        rB, rG = commit_rates_gauss(p, k, **kw)
    return binom_lt(p["N"], rB, p["K"]), binom_lt(p["N"], rG, p["K"])


def W_type(p, k, mode="gauss", **kw):
    s, phi = s_phi(p, k, mode, **kw)
    return (1 - p["pi_b"]) * (1 - phi) * p["V1"] - p["pi_b"] * (1 - s) * p["Lbad"]


# ================================================================================================
# STRATEGIC layer (one-shot global game on the type; failure=0 refund). opt/pess/RD selection.
# ================================================================================================
def cmw(z, qg, qb, p, k):
    N, K, pi_b = p["N"], p["K"], p["pi_b"]
    V1, Lbad, xi, kappa = p["V1"], p["Lbad"], p["xi"], p["kappa"]
    pB = posterior_bad_binary(pi_b, k, z)
    d = 0.0
    for type_, pt, cc, cw in (
        ("G", 1 - pB, V1 - kappa, xi * V1),
        ("B", pB, -Lbad - kappa, xi * (-Lbad)),
    ):
        if pt <= 0:
            continue
        r = sig_prob_binary(k, "g", type_) * qg + sig_prob_binary(k, "b", type_) * qb
        d += pt * (binom_ge(N - 1, r, K - 1) * cc - binom_ge(N - 1, r, K) * cw)
    return d


def eqs_strategic(p, k):
    out = []
    for s0g in (0.0, 1.0):
        for s0b in (0.0, 1.0):
            qg, qb = s0g, s0b
            for _ in range(200):
                ng = 1.0 if cmw("g", qg, qb, p, k) > 0 else 0.0
                nb = 1.0 if cmw("b", qg, qb, p, k) > 0 else 0.0
                if (ng, nb) == (qg, qb):
                    break
                qg, qb = ng, nb
            if (qg, qb) not in out:
                out.append((qg, qb))
    return out


def rd_eq(p, k):
    N, K, pi_b = p["N"], p["K"], p["pi_b"]
    V1, Lbad, xi, kappa = p["V1"], p["Lbad"], p["xi"], p["kappa"]
    pcc = sum(1 for j in range(N) if j >= K - 1) / N
    pcw = sum(1 for j in range(N) if j >= K) / N

    def c(z):
        pB = posterior_bad_binary(pi_b, k, z)
        vc = (1 - pB) * pcc * (V1 - kappa) + pB * pcc * (-Lbad - kappa)
        vw = (1 - pB) * pcw * (xi * V1) + pB * pcw * (xi * (-Lbad))
        return 1.0 if vc > vw else 0.0

    return c("g"), c("b")


def s_strategic(p, k, rule):
    if rule == "rd":
        qg, qb = rd_eq(p, k)
        neq = 1
    else:
        e = eqs_strategic(p, k)
        neq = len(e)
        qg, qb = (
            (max if rule == "opt" else min)(e, key=lambda t: t[0] + t[1])
            if e
            else (0.0, 0.0)
        )
    rB = sig_prob_binary(k, "g", "B") * qg + sig_prob_binary(k, "b", "B") * qb
    rG = sig_prob_binary(k, "g", "G") * qg + sig_prob_binary(k, "b", "G") * qb
    return binom_lt(p["N"], rB, p["K"]), binom_lt(p["N"], rG, p["K"]), neq


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


# ================================================================================================
# RUN
# ================================================================================================
def main():
    print("=" * 96)
    print(
        "v8 #2 — DERIVING s(k): bad-coalition screening from a private TYPE-SIGNAL of precision k"
    )
    print(f"seed={SEED}")
    print("=" * 96)

    N, kappa, V1, Lbad = 10, 1.0, 3.0, 3.0
    xi = 0.3  # type-channel regime: A=(1-xi)V1-kappa>0
    theta0, pi_b0 = 0.50, 0.35
    base = mk(N, theta0, pi_b0, V1, Lbad, xi, kappa)
    A = (1 - xi) * V1 - kappa
    print(
        f"\nTYPE-channel cell: N={N} theta={theta0} K={base['K']}(K/N={base['K'] / N:.2f}) V1={V1} "
        f"Lbad={Lbad} kappa={kappa} xi={xi} pi_b={pi_b0}  A={A:+.2f}(>0) 1-kappa/V1={1 - kappa / V1:.3f}"
    )

    # --- 0. EXCLUDABILITY REGIME (F2) ---
    print(
        "\n--- 0. EXCLUDABILITY REGIME (F2): the type channel has bite only at xi < 1-kappa/V1 ---"
    )
    print("  xi      A=(1-xi)V1-kappa   type-channel-active")
    for xv in [0.0, 0.3, 0.5, 0.667, 0.8]:
        print(
            f"  {xv:.3f}   {(1 - xv) * V1 - kappa:+8.3f}        {(1 - xv) * V1 - kappa > 1e-9}"
        )
    print(
        "  >> high xi (non-excludable) -> free-ride -> PIVOTALITY channel (v6, masking-favoring)."
    )
    print(
        "  >> low  xi (excludable)     -> commit pays -> TYPE channel (v8, disclosure-favoring)."
    )

    # --- 1. SMOOTH DERIVED s(k), phi(k): GAUSSIAN signal (the headline shape) ---
    print("\n--- 1. SMOOTH s(k), phi(k): GAUSSIAN type signal (md=2, d_scale=4) ---")
    print(
        "  s(k)=Pr(BAD not cleared)=SCREEN;  phi(k)=Pr(GOOD not cleared)=FALSE-REJECT (cost);"
    )
    print("  W_type(k)=(1-pi_b)(1-phi)V1 - pi_b(1-s)Lbad.")
    print("   k     rB       rG       s(k)     phi(k)    W_type")
    kk = np.linspace(0, 1, 11)
    for k in kk:
        rB, rG = commit_rates_gauss(base, float(k))
        s, phi = binom_lt(N, rB, base["K"]), binom_lt(N, rG, base["K"])
        print(
            f"  {k:.2f}   {rB:.4f}   {rG:.4f}   {s:.4f}   {phi:.4f}   {W_type(base, float(k)):+.4f}"
        )

    # --- 2. SHAPE + monotonicity (Gaussian) ---
    print(
        "\n--- 2. SHAPE of s(k), phi(k), W_type(k) (Gaussian, fine grid) vs v3's s(k)=k posit ---"
    )
    kf = np.linspace(0.001, 0.999, 200)
    sv = np.array(
        [binom_lt(N, commit_rates_gauss(base, float(k))[0], base["K"]) for k in kf]
    )
    pv = np.array(
        [binom_lt(N, commit_rates_gauss(base, float(k))[1], base["K"]) for k in kf]
    )
    wv = np.array([W_type(base, float(k)) for k in kf])
    lin = sv[0] + (kf - kf[0]) / (kf[-1] - kf[0]) * (sv[-1] - sv[0])
    print(
        f"  s(k):    monotone_incr={bool(np.all(np.diff(sv) >= -1e-9))}  "
        f"max|dev-from-linear|={np.max(np.abs(sv - lin)):.4f}  s(0+)={sv[0]:.3f} s(1-)={sv[-1]:.3f}"
    )
    print(
        f"  phi(k):  monotone_decr={bool(np.all(np.diff(pv) <= 1e-9))}  phi(0+)={pv[0]:.3f} phi(1-)={pv[-1]:.3f}"
    )
    print(
        f"  W_type:  monotone_incr={bool(np.all(np.diff(wv) >= -1e-9))}  argmax k={kf[np.argmax(wv)]:.3f} "
        f"(=1.0 => type channel wants FULL disclosure)"
    )
    print(
        "  >> v3 POSITED s(k)=k (linear); the derived shape is an S/convex curve, not linear."
    )

    # --- 3. BINARY-channel regime view (F3): paralysis vs screening ---
    print(
        "\n--- 3. BINARY channel (regime structure; F3): low-k 's=1' is PARALYSIS not screening ---"
    )
    print("   k     s(k)    phi(k)   commit(g,b)   regime")
    for k in np.linspace(0, 1, 11):
        rB, rG, cg, cb = commit_rates_binary(base, float(k))
        s, phi = binom_lt(N, rB, base["K"]), binom_lt(N, rG, base["K"])
        reg = (
            "commit-both"
            if (cg and cb)
            else ("SCREEN(g-only)" if cg else "PARALYSIS(neither)")
        )
        print(f"  {k:.2f}   {s:.4f}  {phi:.4f}    ({cg},{cb})       {reg}")
    print(
        "  >> in PARALYSIS s=1 AND phi=1 (good coalitions die too): NOT screening. Genuine"
    )
    print(
        "     screening (s high WHILE phi low) only appears once k passes the commit-on-good onset."
    )

    # --- 4. TWO-CHANNEL: type channel is one-way pro-disclosure ---
    print("\n--- 4. TWO-CHANNEL FINDING: type channel never favors masking ---")
    print(
        f"  Gaussian: s increasing={bool(np.all(np.diff(sv) >= -1e-9))}, "
        f"phi decreasing={bool(np.all(np.diff(pv) <= 1e-9))}, W_type max at k={kf[np.argmax(wv)]:.2f}"
    )
    print(
        "  >> MORE type-disclosure rejects more BAD and fewer GOOD AND raises welfare: a one-way"
    )
    print(
        "     ratchet to k=1. Masking-is-optimal lives in the PIVOTALITY channel (v6), not here."
    )

    # --- 5. GAP s(1)-s(0) and W_type(1)-W_type(0) from PRIMITIVES (Gaussian) ---
    print(
        "\n--- 5. screening GAP from primitives (Gaussian; s,W evaluated at k=0.02 and k=0.98) ---"
    )
    print(
        "  pi_b  Lbad/V1  theta(K/N)   s(.02)   s(.98)   gap_s    Wtype(.02)  Wtype(.98)"
    )
    for pi_b in [0.2, 0.5, 0.8]:
        for Lr in [1.0, 3.0]:
            for th in [0.30, 0.50]:
                p = mk(N, th, pi_b, V1, Lr * V1, xi, kappa)
                s0, _ = s_phi(p, 0.02, "gauss")
                s1, _ = s_phi(p, 0.98, "gauss")
                w0, w1 = W_type(p, 0.02), W_type(p, 0.98)
                print(
                    f"  {pi_b:.2f}  {Lr:5.1f}   {th:.2f}({p['K'] / N:.2f})   {s0:.4f}   {s1:.4f}   "
                    f"{s1 - s0:+.4f}   {w0:+.4f}    {w1:+.4f}"
                )

    # --- 6. STRATEGIC robustness ---
    print(
        "\n--- 6. STRATEGIC layer (one-shot global game, no -Lp): s(k) across selections ---"
    )
    print("   k     s_gauss  s_OPT   s_PESS  s_RD    n_eq")
    for k in [0.0, 0.25, 0.5, 0.75, 1.0]:
        sg, _ = s_phi(base, k, "gauss")
        so, _, neo = s_strategic(base, k, "opt")
        sp, _, _ = s_strategic(base, k, "pess")
        sr, _, _ = s_strategic(base, k, "rd")
        print(f"  {k:.2f}   {sg:.4f}  {so:.4f}  {sp:.4f}  {sr:.4f}   {neo}")

    # --- 7. ORDERING robustness (Gaussian) + MC ---
    print(
        "\n--- 7. ORDERING s(.98)>=s(.02) AND phi monotone-decr across grid (Gaussian) ---"
    )
    allhold, mono_phi, n = True, True, 0
    for pi_b in [0.2, 0.35, 0.5, 0.65, 0.8]:
        for Lr in [0.5, 1.0, 2.0, 3.0]:
            for th in [0.30, 0.50, 0.70]:
                for xv in [0.0, 0.3, 0.5]:
                    p = mk(N, th, pi_b, V1, Lr * V1, xv, kappa)
                    s0, _ = s_phi(p, 0.02, "gauss")
                    s1, _ = s_phi(p, 0.98, "gauss")
                    ph = np.array(
                        [
                            s_phi(p, float(k), "gauss")[1]
                            for k in np.linspace(0.02, 0.98, 25)
                        ]
                    )
                    n += 1
                    if s1 < s0 - 1e-9:
                        allhold = False
                    if np.any(np.diff(ph) > 1e-9):
                        mono_phi = False
    print(
        f"  >> s(.98)>=s(.02) across ALL {n} cells: {allhold};  phi(k) monotone-decreasing: {mono_phi}"
    )

    # --- 7b. C2 STRESS TEST (the adversarial-verifier refutation, reproduced here) ---
    # The independent global-game re-impl (v8_verify_reimpl.py) REFUTED the UNIVERSAL one-way-ratchet
    # form of C2: under a PESSIMISTIC prior (high pi_b) with a LOW threshold (high theta -> small K)
    # and a SMALL loss (low Lbad/V1), s(k) is U-SHAPED -- raising disclosure lets noisy-"good" signals
    # commit to a BAD coalition that then clears at low K, so screening DIPS before recovering. We
    # reproduce that here by sweeping the regime the section-7 grid missed, and report any net s(1)<s(0)
    # OR interior dip below both endpoints. This makes C2 CONDITIONAL (pro-disclosure only in the
    # optimistic-prior regime), as it must be after the refutation.
    print(
        "\n--- 7b. C2 STRESS TEST: is the type channel a UNIVERSAL one-way ratchet? (it is NOT) ---"
    )
    kk2 = np.linspace(0.02, 0.98, 25)
    viol_net, viol_dip, total, worst_net, worst_cell = 0, 0, 0, 0.0, None
    for pi_b in [0.5, 0.65, 0.8, 0.9]:
        for Lr in [0.3, 0.5, 1.0]:
            for th in [0.70, 0.80, 0.90]:
                for xv in [0.0, 0.2, 0.3]:
                    p = mk(N, th, pi_b, V1, Lr * V1, xv, kappa)
                    sv2 = np.array([s_phi(p, float(k), "gauss")[0] for k in kk2])
                    total += 1
                    net = sv2[-1] - sv2[0]
                    interior_dip = sv2.min() < min(sv2[0], sv2[-1]) - 1e-6
                    if net < -1e-6:
                        viol_net += 1
                        if net < worst_net:
                            worst_net, worst_cell = net, (pi_b, Lr, th, xv)
                    if interior_dip:
                        viol_dip += 1
    print(f"  swept {total} pessimistic/low-K/small-loss cells:")
    print(
        f"  >> net s(1)<s(0) (disclosure SCREENS WORSE than masking): {viol_net}/{total} cells"
    )
    print(
        f"  >> interior DIP (U-shaped s(k), screening non-monotone): {viol_dip}/{total} cells"
    )
    print(
        f"  >> worst net s(1)-s(0) = {worst_net:+.4f} at (pi_b,Lbad/V1,theta,xi)={worst_cell}"
    )
    print(
        "  >> CONCLUSION: s(k) is NON-MONOTONE (interior dip in the vast majority of these cells):"
    )
    print(
        "     from masking toward INTERMEDIATE k, screening gets WORSE (noisy-'good' signals commit to"
    )
    print(
        "     a bad coalition that clears at low K), then recovers by k=1. In THIS decision-theoretic"
    )
    print(
        "     model the dip recovers (net s(1)>=s(0)); the INDEPENDENT global-game re-impl finds NET"
    )
    print(
        "     s(1)<s(0) in ~59-62/240 cells (small, MC-confirmed). EITHER WAY C2's 'one-way ratchet'"
    )
    print("     is REFUTED as universal -> CONDITIONAL on the prior.")

    rng = np.random.default_rng(SEED)
    print("\n--- 8. SEEDED MC cross-check (Gaussian, base cell) ---")
    print("   k     s_exact  s_MC     |diff|")
    for k in [0.2, 0.5, 0.8]:
        rB, _ = commit_rates_gauss(base, k)
        s_exact = binom_lt(N, rB, base["K"])
        runs, fails = 20000, 0
        for _ in range(runs):
            commits = int(np.sum(rng.random(N) < rB))
            if commits < base["K"]:
                fails += 1
        print(
            f"  {k:.2f}   {s_exact:.4f}  {fails / runs:.4f}   {abs(s_exact - fails / runs):.4f}"
        )

    print("\n" + "=" * 96)
    print("SUMMARY (v8 #2)")
    print(
        "  (1) s(k) DERIVED as a Bayesian signal-extraction object (Pr a BAD coalition fails to"
    )
    print(
        "      reach K commits under a type signal of precision k). Smooth via the Gaussian signal;"
    )
    print(
        "      s(0),s(1) are LIMITS not posits. v3's linear s(k)=k is FALSE -- shape is non-linear"
    )
    print(
        "      (S/convex), BUT the precise curvature is inherited from the posited precision map"
    )
    print(
        "      d'(k)=d_scale*k/(1-k); 'derived' RELOCATES the assumption, it does not eliminate it."
    )
    print(
        "  (2) EXCLUDABILITY REGIME (robust, confirmed by independent re-impl): type channel has bite"
    )
    print(
        "      only at xi<1-kappa/V1; else free-ride (pivotality channel, v6). OPPOSITE xi regimes."
    )
    print(
        "  (3) TWO CHANNELS pull opposite ways: type favors DISCLOSURE (phi down, W up in the"
    )
    print(
        "      OPTIMISTIC-prior regime), pivotality favors MASKING. CONDITIONAL, NOT universal:"
    )
    print(
        "      s(k) is non-monotone and can be U-SHAPED under pessimistic priors + low K (sec 7b) --"
    )
    print(
        "      so the screening channel does NOT always want full disclosure. v3 bundled both in one k."
    )
    print(
        "  (4) HONEST: low-k 's=1' is PARALYSIS (phi=1 too), not screening. Genuine screening"
    )
    print(
        "      (s high WHILE phi low) requires k past the commit-on-good onset. s(1)>=s(0) holds on"
    )
    print(
        "      the sec-7 grid but FAILS in the sec-7b pessimistic regime (C2 is conditional)."
    )
    print("=" * 96)


if __name__ == "__main__":
    main()


# ================================================================================================
# DEVELOPMENT LOG (numerics-first; falsification trail)
# ================================================================================================
# v8.0 (DRAFT 1, FALSIFIED, F1): one-shot coordination game, v6 payoffs (xi=0.8, -Lp, K/N=0.9) ->
#   all-decline everywhere (s=phi=1). The collapse is a pivotality artifact, not detection.
# v8.1 (F2): xi=0.8 => A=(1-xi)V1-kappa<0 => free-ride dominates => type signal irrelevant. The
#   type channel needs xi<1-kappa/V1. The two disclosure channels live in opposite xi regimes.
# v8.2 (F3): a binary 0/1 Bayes rule makes s(k) knife-edge (commit-neither/g-only/both). Low-k
#   "s=1" is PARALYSIS (phi=1 too), not screening. Added the GAUSSIAN continuous-signal derivation
#   for a smooth, honest s(k): d'(k)=d_scale*k/(1-k), Bayes cutoff y_hat from break-even odds.
#   RESULTS (literal stdout, seed 20260609): s(k) non-linear (NOT v3's linear); phi(k) mostly
#   monotone-decreasing; W_type(k) monotone-increasing in the optimistic-prior regime.
#   Ordering s(.98)>=s(.02) holds on the sec-7 grid; MC matches to ~1e-3.
# v8.3 (C2 REFUTED-AS-UNIVERSAL by the adversarial verifier; folded in, sec 7b): an INDEPENDENT
#   continuous-fundamental global-game re-impl (v8_verify_reimpl.py: x~N(m0,1/tau0), bad iff x<0,
#   private Gaussian signals, threshold eq, FINITE quadratic precision ramp) confirmed C1/C3/C4 but
#   REFUTED the UNIVERSAL one-way-ratchet form of C2: in 62/240 excludable cells net s(1)<s(0)
#   (worst -0.082, MC-confirmed), s U-shaped under a PESSIMISTIC prior. Mechanism: pessimistic prior
#   => low-k paralysis screens bad; raising k lets noisy-"good" signals commit to a BAD coalition
#   that CLEARS at low K => s DIPS before recovering. sec 7b reproduces this in THIS framework. So C2
#   is CONDITIONAL (pro-disclosure only in the optimistic-prior regime), NOT universal. ALSO honest:
#   the convex shape (C4) is inherited from the POSITED precision map d'(k)=d_scale*k/(1-k) -- no
#   actual information-theoretic primitive (entropy/channel-capacity/k-anon leakage) is instantiated;
#   "derived" RELOCATES the assumption (v7's binary payoff -> v8's precision ramp), not eliminates it.
#   Float guard: K=ceil(round((1-theta)*N,6)) (theta=0.7 float gave K=4 not 3 in two robustness loops).
# ================================================================================================
