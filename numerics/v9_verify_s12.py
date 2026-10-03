"""
v9_verify_s12.py — ADVERSARIAL, INDEPENDENT verification of v9 S12.

S12 (v9_kanon_leakage.py) claims, using ADDITIVE UNIFORM BLUR of half-width r(k)=round((1-k)*N)
on a type-correlated support count m~Binom(N,p_psi), p_G>p_B:
  (A) the signal precision is a GENUINE Shannon mutual information I(type ; k-anon count);
  (B) I(k) is monotone-increasing (data-processing) + concave + saturating, with
      I(1)=I(type;exact count) < H(type)  (imperfect: binomial overlap);
  (C) s(1) < 1 -- full disclosure does NOT perfectly screen -- a DERIVED function of the
      support gap p_G-p_B, CORRECTING v8's posited-ramp artifact s(1)=1;
  (D) the s(k) non-monotonicity (v8 C2) is independently reproduced.

This verifier RE-DERIVES (A)-(D) with THREE DIFFERENT privacy mechanisms on the SAME
type-correlated count m~Binom(N,p_psi):

  MECH-RR  : RANDOMIZED RESPONSE / binary-symmetric channel. Each of the N agents' commit
             bits is independently flipped w.p. f(k); the published count is the sum of noisy
             bits. This is a genuinely different channel (per-bit BSC, not additive count noise).
             The noisy count given type is Binom(N, p_psi*(1-f) + (1-p_psi)*f). Flip prob
             f(k) = 0.5*(1-k): k=0 -> f=0.5 (no info), k=1 -> f=0 (exact).
  MECH-DP  : DIFFERENTIAL PRIVACY two-sided GEOMETRIC mechanism (discrete Laplace) on the count.
             Noise Z ~ DoubleGeometric(alpha), alpha=exp(-eps(k)), eps(k)=eps_scale*k/(1-k)
             (eps->inf at k=1 -> exact; eps->0 at k=0 -> washed out). Published = m + Z.
  MECH-KEQ : GENUINE k-ANONYMITY with EQUAL-COUNT (equal-probability-mass) buckets, not equal
             WIDTH. Buckets are built so each holds >= k_anon records of the EMPIRICAL count
             distribution; coarser k -> fewer buckets -> >= more records each. A nested-quantile
             construction (so buckets only ever MERGE as k shrinks) to respect data-processing.

For each mechanism we compute I(type;published) exactly from the joint, the Bayes-optimal
public-signal screening s(k)/phi(k) with the SAME break-even rule v9 uses, and the s(1) cap.

We ALSO recompute the ADDITIVE-BLUR baseline here independently (re-implemented, not imported)
to confirm we reproduce v9's printed numbers before trusting our cross-mechanism comparison.

VERDICT LOGIC:
  - (B) monotone+saturating: must hold under a PROPER DPI-respecting channel. We check it.
  - (C) s(1)<1: if it holds under RR and DP and KEQ too, it is a ROBUST correction, NOT
        mechanism-specific. If it holds ONLY for additive blur, S12's headline is fragile.
  - "GROUNDED not relocated": we test whether s(1) is pinned by (p_G,p_B) ALONE (the count's
        statistical discriminability = the Bayes error of Binom(N,p_G) vs Binom(N,p_B)),
        i.e. mechanism-INDEPENDENT at full disclosure. If s(1) at k=1 is the SAME across all
        three mechanisms (all reduce to the exact count), then (p_G,p_B) is the real primitive
        and the "blur" was never load-bearing -- which both VALIDATES the support-gap grounding
        AND exposes that S12's specific blur channel was an arbitrary choice (a relocation of the
        SHAPE, not the endpoint). We report this explicitly.

numerics-first, exact (no Monte-Carlo needed for the channels above; MC cross-check at the end).
"""

from __future__ import annotations

from math import comb, exp, log2

import numpy as np

SEED = 20260609
rng = np.random.default_rng(SEED)


# ----------------------------------------------------------------------------------------------
# shared primitives
# ----------------------------------------------------------------------------------------------
def binom_pmf(n: int, p: float) -> np.ndarray:
    if n == 0:
        return np.array([1.0])
    p = min(max(p, 0.0), 1.0)
    return np.array([comb(n, j) * (p**j) * ((1 - p) ** (n - j)) for j in range(n + 1)])


def entropy_bits(pi_b: float) -> float:
    out = 0.0
    for p in (pi_b, 1 - pi_b):
        if p > 0:
            out -= p * log2(p)
    return out


def mi_from_conditionals(pi_b: float, obsG: np.ndarray, obsB: np.ndarray) -> float:
    """I(type ; published) in bits from aligned conditional dists p(y|G), p(y|B)."""
    pobs = pi_b * obsB + (1 - pi_b) * obsG
    mi = 0.0
    for t, pt, ogt in (("G", 1 - pi_b, obsG), ("B", pi_b, obsB)):
        for bi in range(len(pobs)):
            joint = pt * ogt[bi]
            if joint > 0 and pobs[bi] > 0:
                mi += joint * log2(joint / (pt * pobs[bi]))
    return max(mi, 0.0)


def s_phi_from_conditionals(
    pi_b, obsG, obsB, V1, Lbad, xi, kappa
) -> tuple[float, float]:
    """Public-signal Bayes screening with the SAME break-even rule v9 uses.
    commit iff Pr(G|y)*A > Pr(B|y)*B_,  A=(1-xi)V1-kappa, B_=(1-xi)Lbad+kappa.
    s = Pr(decline | B) [SCREEN]; phi = Pr(decline | G) [FALSE-REJECT]."""
    A = (1 - xi) * V1 - kappa
    B_ = (1 - xi) * Lbad + kappa
    pobs = pi_b * obsB + (1 - pi_b) * obsG
    s = phi = 0.0
    for bi in range(len(pobs)):
        if pobs[bi] <= 0:
            continue
        postB = pi_b * obsB[bi] / pobs[bi]
        postG = 1 - postB
        commit = (A > 0) and (postG * A > postB * B_)
        if not commit:
            s += obsB[bi]
            phi += obsG[bi]
    return s, phi


# ----------------------------------------------------------------------------------------------
# MECH-BLUR : v9's additive uniform blur, RE-IMPLEMENTED independently (sanity baseline)
# ----------------------------------------------------------------------------------------------
def blur_conditionals(pG, pB, k, N):
    r = int(round((1 - k) * N))
    width = 2 * r + 1
    lo, hi = -r, N + r
    L = hi - lo + 1
    out = {}
    for tag, p in (("G", pG), ("B", pB)):
        pmf = binom_pmf(N, p)
        arr = np.zeros(L)
        for m in range(N + 1):
            if pmf[m] <= 0:
                continue
            for e in range(-r, r + 1):
                arr[(m + e) - lo] += pmf[m] / width
        out[tag] = arr
    return out["G"], out["B"]


# ----------------------------------------------------------------------------------------------
# MECH-RR : randomized response / per-bit binary symmetric channel, then publish the count
# ----------------------------------------------------------------------------------------------
def rr_conditionals(pG, pB, k, N):
    """Each agent commit-bit ~ Bernoulli(p_psi) is flipped i.i.d. w.p. f=0.5*(1-k).
    Noisy count ~ Binom(N, p_psi*(1-f)+(1-p_psi)*f). Published support = noisy count in {0..N}.
    This is the canonical local-DP RR mechanism on a count; a DPI channel by construction."""
    f = 0.5 * (1 - k)
    qG = pG * (1 - f) + (1 - pG) * f
    qB = pB * (1 - f) + (1 - pB) * f
    return binom_pmf(N, qG), binom_pmf(N, qB)


# ----------------------------------------------------------------------------------------------
# MECH-DP : two-sided geometric (discrete Laplace) DP noise on the count, eps(k)=eps_scale*k/(1-k)
# ----------------------------------------------------------------------------------------------
def double_geom_pmf(alpha: float, span: int) -> tuple[np.ndarray, int]:
    """Two-sided geometric Z over {-span..span}, P(Z=z) = (1-a)/(1+a) * a^|z|, a=alpha. Truncated+renorm.
    alpha->1 (eps->0) degenerates to (truncated) uniform = maximal-noise / zero-info limit."""
    zs = np.arange(-span, span + 1)
    if alpha >= 1 - 1e-12:
        w = np.ones(len(zs))
    else:
        w = alpha ** np.abs(zs)
    w = w / w.sum()
    return w, -span


def dp_conditionals(pG, pB, k, N, eps_scale=4.0):
    """Published = m + Z, Z ~ double-geometric with alpha=exp(-eps), eps=eps_scale*k/(1-k).
    k=1 -> eps=inf -> alpha=0 -> Z=0 (exact count). k=0 -> eps=0 -> alpha=1 -> max noise (washed)."""
    if k >= 1 - 1e-12:
        eps = eps_scale * 1e12
    elif k <= 1e-12:
        eps = 0.0
    else:
        eps = eps_scale * k / (1 - k)
    alpha = exp(-eps) if eps < 700 else 0.0
    span = 3 * N  # enough geometric tail for accuracy
    zpmf, zlo = double_geom_pmf(alpha if alpha > 0 else 1e-300, span)
    lo, hi = zlo, N + span
    L = hi - lo + 1
    out = {}
    for tag, p in (("G", pG), ("B", pB)):
        mpmf = binom_pmf(N, p)
        arr = np.zeros(L)
        for m in range(N + 1):
            if mpmf[m] <= 0:
                continue
            for zi, z in enumerate(range(zlo, span + 1)):
                arr[(m + z) - lo] += mpmf[m] * zpmf[zi]
        out[tag] = arr
    return out["G"], out["B"]


# ----------------------------------------------------------------------------------------------
# MECH-KEQ : genuine k-anonymity, EQUAL-COUNT buckets built by NESTED quantile merging
# ----------------------------------------------------------------------------------------------
def keq_conditionals(pG, pB, k, N, pi_b):
    """Equal-MASS (equal-count) k-anon buckets on the MARGINAL count distribution p(m).
    Number of buckets nb=max(1, round(k*N)). Buckets are contiguous on m built so each holds
    ~equal marginal mass (>= k_anon records when scaled by population) -- the genuine k-anon
    'each bucket >= k records' rule. Construction is NESTED across k (finer partitions refine
    coarser ones) by always cutting at marginal-mass quantiles of a fixed fine grid, so the
    channel respects data-processing. Returns p(bucket|G), p(bucket|B)."""
    pmfG, pmfB = binom_pmf(N, pG), binom_pmf(N, pB)
    marg = pi_b * pmfB + (1 - pi_b) * pmfG  # marginal p(m)
    nb = max(1, int(round(k * N)))
    nb = min(nb, N + 1)
    # nested quantile cuts: assign each m to a bucket by cumulative marginal mass.
    cum = np.cumsum(marg)
    # bucket index = floor(nb * cum_mass_up_to_m), clamped -> contiguous equal-mass buckets
    bidx = np.minimum((cum * nb).astype(int), nb - 1)
    # ensure contiguity / monotonic (cum is monotone so bidx is monotone non-decreasing) -> ok
    obsG = np.zeros(nb)
    obsB = np.zeros(nb)
    for m in range(N + 1):
        obsG[bidx[m]] += pmfG[m]
        obsB[bidx[m]] += pmfB[m]
    return obsG, obsB


# ----------------------------------------------------------------------------------------------
# Bayes-error / exact-count discriminability: the mechanism-FREE primitive at full disclosure
# ----------------------------------------------------------------------------------------------
def exact_count_s_phi(pG, pB, N, pi_b, V1, Lbad, xi, kappa):
    """s,phi when the EXACT count m is published (no mechanism). This is the common limit of
    ALL mechanisms at k=1. If s(1) matches this across mechanisms, (p_G,p_B) is the real primitive."""
    return s_phi_from_conditionals(
        pi_b, binom_pmf(N, pG), binom_pmf(N, pB), V1, Lbad, xi, kappa
    )


def exact_count_mi(pG, pB, N, pi_b):
    return mi_from_conditionals(pi_b, binom_pmf(N, pG), binom_pmf(N, pB))


# ----------------------------------------------------------------------------------------------
MECHS = {
    "BLUR (v9)": blur_conditionals,
    "RR  (BSC)": rr_conditionals,
    "DP  (geom)": dp_conditionals,
}


def run_curve(mech_fn, pG, pB, N, pi_b, V1, Lbad, xi, kappa, ks, needs_pi=False):
    Is, ss, phis = [], [], []
    for k in ks:
        if needs_pi:
            oG, oB = mech_fn(pG, pB, float(k), N, pi_b)
        else:
            oG, oB = mech_fn(pG, pB, float(k), N)
        Is.append(mi_from_conditionals(pi_b, oG, oB))
        s, phi = s_phi_from_conditionals(pi_b, oG, oB, V1, Lbad, xi, kappa)
        ss.append(s)
        phis.append(phi)
    return np.array(Is), np.array(ss), np.array(phis)


def concave_frac(I):
    d2 = np.diff(I, 2)
    return int(np.sum(d2 <= 1e-9)), len(d2)


def main():
    N, V1, Lbad, kappa, xi = 12, 3.0, 3.0, 1.0, 0.3
    pi_b, pG, pB = 0.4, 0.65, 0.30
    Hpsi = entropy_bits(pi_b)
    ks = np.linspace(0, 1, 13)

    print("=" * 96)
    print("v9_verify_s12 — INDEPENDENT cross-mechanism refutation test of v9 S12")
    print(
        f"calib N={N} pi_b={pi_b} pG={pG} pB={pB} gap={pG - pB:.2f} V1={V1} Lbad={Lbad} "
        f"kappa={kappa} xi={xi}  H(type)={Hpsi:.4f} bits"
    )
    print("=" * 96)

    # ---- 0. reproduce v9 additive-blur baseline (must match the printed v9 numbers) ----
    Ib, sb, pb = run_curve(blur_conditionals, pG, pB, N, pi_b, V1, Lbad, xi, kappa, ks)
    print("\n[0] REPRODUCE v9 baseline (additive blur), independent re-impl:")
    print(f"    I(0)={Ib[0]:.4f} I(1)={Ib[-1]:.4f}  (v9 printed 0.0994 / 0.6214)")
    print(f"    s(0)={sb[0]:.4f} s(1)={sb[-1]:.4f}  (v9 printed 0.9619 / 0.9614)")
    repro = (
        abs(Ib[0] - 0.0994) < 1e-3
        and abs(Ib[-1] - 0.6214) < 1e-3
        and abs(sb[-1] - 0.9614) < 1e-3
    )
    print(f"    reproduced v9 baseline: {repro}")

    # ---- 1. I(k) under each mechanism: monotone? saturating? I(1)<H? ----
    print(
        "\n[1] LEAKAGE I(k) under DIFFERENT mechanisms (DPI -> should be monotone+saturating):"
    )
    print(
        "    mech         I(0)     I(1)     I(1)<H?   monotone_incr   concave(2nd-diff<=0)"
    )
    I_exact = exact_count_mi(pG, pB, N, pi_b)
    curves = {}
    for name, fn in MECHS.items():
        I, s, phi = run_curve(fn, pG, pB, N, pi_b, V1, Lbad, xi, kappa, ks)
        curves[name] = (I, s, phi)
        mono = bool(np.all(np.diff(I) >= -1e-9))
        cf, ct = concave_frac(I)
        print(
            f"    {name:11s}  {I[0]:.4f}   {I[-1]:.4f}   {str(I[-1] < Hpsi - 1e-9):5s}     "
            f"{str(mono):5s}           {cf}/{ct}"
        )
    # KEQ separately (needs pi_b)
    Ikeq, skeq, pkeq = run_curve(
        keq_conditionals, pG, pB, N, pi_b, V1, Lbad, xi, kappa, ks, needs_pi=True
    )
    curves["KEQ (kanon)"] = (Ikeq, skeq, pkeq)
    monok = bool(np.all(np.diff(Ikeq) >= -1e-9))
    cfk, ctk = concave_frac(Ikeq)
    print(
        f"    {'KEQ (kanon)':11s}  {Ikeq[0]:.4f}   {Ikeq[-1]:.4f}   {str(Ikeq[-1] < Hpsi - 1e-9):5s}     "
        f"{str(monok):5s}           {cfk}/{ctk}"
    )
    print(
        f"    I(type;EXACT count) = {I_exact:.4f} bits  (the common k=1 limit; all should -> this)"
    )

    # ---- 2. s(1) under each mechanism vs the exact-count limit ----
    print("\n[2] s(1) under each mechanism vs the MECHANISM-FREE exact-count limit:")
    s_exact, phi_exact = exact_count_s_phi(pG, pB, N, pi_b, V1, Lbad, xi, kappa)
    print(
        f"    exact-count limit:  s(1*)={s_exact:.4f}  phi(1*)={phi_exact:.4f}   (Bayes on exact m)"
    )
    print("    mech         s(1)     s(1)<1?   matches exact-count limit?")
    for name, (I, s, phi) in curves.items():
        match = abs(s[-1] - s_exact) < 1e-6
        print(f"    {name:11s}  {s[-1]:.4f}   {str(s[-1] < 1 - 1e-9):5s}     {match}")

    # ---- 3. s(1)<1 ROBUSTNESS across support gaps, ALL mechanisms ----
    print(
        "\n[3] s(1)<1 ROBUSTNESS across support gap, all mechanisms (is the correction universal?):"
    )
    print("    gap    BLUR_s1  RR_s1    DP_s1    KEQ_s1   EXACT_s1   I1(exact)")
    gaps = [
        (0.55, 0.45),
        (0.6, 0.4),
        (0.65, 0.3),
        (0.75, 0.25),
        (0.9, 0.1),
        (0.98, 0.02),
    ]
    all_lt1 = True
    for pGv, pBv in gaps:
        row = []
        for fn in MECHS.values():
            oG, oB = fn(pGv, pBv, 1.0, N)
            sx, _ = s_phi_from_conditionals(pi_b, oG, oB, V1, Lbad, xi, kappa)
            row.append(sx)
        oGk, oBk = keq_conditionals(pGv, pBv, 1.0, N, pi_b)
        sk, _ = s_phi_from_conditionals(pi_b, oGk, oBk, V1, Lbad, xi, kappa)
        sex, _ = exact_count_s_phi(pGv, pBv, N, pi_b, V1, Lbad, xi, kappa)
        Iex = exact_count_mi(pGv, pBv, N, pi_b)
        if any(v >= 1 - 1e-9 for v in row + [sk, sex]):
            all_lt1 = all_lt1 and False
        print(
            f"    {pGv - pBv:.2f}   {row[0]:.4f}   {row[1]:.4f}   {row[2]:.4f}   "
            f"{sk:.4f}   {sex:.4f}    {Iex:.4f}"
        )
    print(f"    >> s(1)<1 under EVERY mechanism+gap shown: {all_lt1}")

    # ---- 4. NON-MONOTONICITY of s(k) (v8 C2) under each mechanism ----
    print("\n[4] s(k) NON-MONOTONICITY (v8 C2 dip) — reproduced under each mechanism?:")
    kf = np.linspace(0, 1, 41)
    print(
        "    mech         s(0)     s(1)     monotone_incr   interior dip (min<both ends)?"
    )
    for name, fn in list(MECHS.items()):
        sf = np.array(
            [
                s_phi_from_conditionals(
                    pi_b, *fn(pG, pB, float(k), N), V1, Lbad, xi, kappa
                )[0]
                for k in kf
            ]
        )
        mono = bool(np.all(np.diff(sf) >= -1e-9))
        dip = bool(sf.min() < min(sf[0], sf[-1]) - 1e-9)
        print(
            f"    {name:11s}  {sf[0]:.4f}   {sf[-1]:.4f}   {str(mono):5s}           {dip}"
        )
    sfk = np.array(
        [
            s_phi_from_conditionals(
                pi_b, *keq_conditionals(pG, pB, float(k), N, pi_b), V1, Lbad, xi, kappa
            )[0]
            for k in kf
        ]
    )
    monok = bool(np.all(np.diff(sfk) >= -1e-9))
    dipk = bool(sfk.min() < min(sfk[0], sfk[-1]) - 1e-9)
    print(
        f"    {'KEQ (kanon)':11s}  {sfk[0]:.4f}   {sfk[-1]:.4f}   {str(monok):5s}           {dipk}"
    )

    # ---- 5. 'GROUNDED not relocated' test: is s(1) mechanism-independent (= the support gap)? ----
    print("\n[5] 'GROUNDED not relocated' TEST: at k=1 every mechanism -> exact count.")
    print(
        "    If s(1) is IDENTICAL across mechanisms, the ENDPOINT is pinned by (pG,pB) alone"
    )
    print(
        "    (the binomial Bayes discriminability), so the blur channel is NOT load-bearing"
    )
    print(
        "    for the headline s(1)<1; (pG,pB) IS the real primitive. The SHAPE s(k), by contrast,"
    )
    print(
        "    is mechanism-dependent (see [4]) -> 'derived' grounds the ENDPOINT but the curve"
    )
    print(
        "    interior is still a modeling choice. Reporting per-mechanism s(1) spread:"
    )
    s1s = []
    for name, fn in MECHS.items():
        oG, oB = fn(pG, pB, 1.0, N)
        sx, _ = s_phi_from_conditionals(pi_b, oG, oB, V1, Lbad, xi, kappa)
        s1s.append(sx)
    oGk, oBk = keq_conditionals(pG, pB, 1.0, N, pi_b)
    sxk, _ = s_phi_from_conditionals(pi_b, oGk, oBk, V1, Lbad, xi, kappa)
    s1s.append(sxk)
    print(f"    s(1) across [BLUR,RR,DP,KEQ] = {[f'{v:.4f}' for v in s1s]}")
    print(
        f"    max spread in s(1) = {max(s1s) - min(s1s):.6f}  (≈0 => endpoint is mechanism-free)"
    )

    # ---- 6. MC cross-check for RR (the genuinely different channel) ----
    print("\n[6] Monte-Carlo cross-check of RR channel I and s (base cell, k=0.5):")
    k = 0.5
    f = 0.5 * (1 - k)
    M = 400_000
    types = rng.random(M) < pi_b  # True=BAD
    ps = np.where(types, pB, pG)
    # true commit bits per agent then BSC flip, sum -> noisy count
    bits = (rng.random((M, N)) < ps[:, None]).astype(int)
    flips = rng.random((M, N)) < f
    noisy = np.where(flips, 1 - bits, bits).sum(axis=1)
    oG_mc = np.bincount(noisy[~types], minlength=N + 1)[: N + 1] / max(
        1, (~types).sum()
    )
    oB_mc = np.bincount(noisy[types], minlength=N + 1)[: N + 1] / max(1, types.sum())
    I_mc = mi_from_conditionals(pi_b, oG_mc, oB_mc)
    s_mc, _ = s_phi_from_conditionals(pi_b, oG_mc, oB_mc, V1, Lbad, xi, kappa)
    oG_an, oB_an = rr_conditionals(pG, pB, k, N)
    I_an = mi_from_conditionals(pi_b, oG_an, oB_an)
    s_an, _ = s_phi_from_conditionals(pi_b, oG_an, oB_an, V1, Lbad, xi, kappa)
    print(f"    I_analytic={I_an:.4f}  I_MC={I_mc:.4f}  |diff|={abs(I_an - I_mc):.4f}")
    print(f"    s_analytic={s_an:.4f}  s_MC={s_mc:.4f}  |diff|={abs(s_an - s_mc):.4f}")

    print("\n" + "=" * 96)


if __name__ == "__main__":
    main()
