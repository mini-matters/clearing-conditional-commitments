"""
INDEPENDENT ADVERSARIAL VERIFICATION of v5 Route C (routeC-dynamics).

The develop agent claims: in a dynamic masked-running-total contribution game, the
completion basin (= realized A8 success-selection prob) SHRINKS when granular pivotality
is REVEALED, and the shrinkage is caused SPECIFICALLY by a war-of-attrition (w_wait), with
zero shrinkage at w_wait=0.

I re-derive with a DIFFERENT parameterization and a DIFFERENT/cleaner attrition model, and
I run several adversarial probes the develop script did NOT run:

  PROBE 1 (different params): N=80, T=10, p_opp=0.18, kappa=1, Lp=0.5, different seed.
           Does basin still rise in V1/kappa (folk threshold) and SHRINK under revelation?

  PROBE 2 (CODE-PATH SYMMETRY at w_wait=0): the develop script's falsification guard asserts
           revealed-basin == masked-baseline at w_wait=0. But the two functions are SEPARATE
           code paths. I run a STRICT identity test: with w_wait=0, are masked and revealed
           literally the SAME trajectory on the SAME seed? If not exactly equal, the "no other
           asymmetry" claim is WEAKER than advertised (residual path divergence).

  PROBE 3 (mechanism honesty): the belief _belief() has a hardcoded 0.30/0.5 projection and a
           SHARP=1.2 logistic. Is the result an ARTIFACT of those magic numbers? Sweep SHARP and
           the projection constant; check the sign of the masking advantage is robust.

  PROBE 4 (boundary stress): tau near 0 (V1 huge) and tau near 1 (V1 -> 0, Lp dominates);
           theta near 0 (G ~ N, nearly impossible) and theta near 1 (G ~ 0, trivial). Does the
           basin degenerate gracefully or pathologically?

  PROBE 5 (is the holdout the SOLE cause? an alternative attrition model): instead of the
           develop's "pivotal if gap <= 2*len(movers)", I use a cleaner decisiveness test
           (a mover is pivotal iff her single commitment crosses G, i.e. gap==1, OR gap<=movers)
           and deferral = skip this period. Confirm sign of shrinkage is model-robust, not an
           artifact of the specific pivotality predicate.

NUMERICS-FIRST. Fixed seed. Literal stdout reported.
"""

import numpy as np

SEED = 424242  # DIFFERENT seed from develop's 20260608


# ----------------------------------------------------------------------------------
# Independent re-implementation. I keep the SAME economic primitive (EV-commit gate,
# provision point G, conditional refund) but rewrite the belief and attrition cleanly.
# ----------------------------------------------------------------------------------
def belief(gap, movers_n, n_remaining, h, sharp, proj):
    """Logistic completion belief in slack = reachable - gap. proj = future-activity weight."""
    future = proj * n_remaining * max(0, h - 1)
    reachable = movers_n + 0.5 * future
    return 1.0 / (1.0 + np.exp(-sharp * (reachable - gap)))


def run_episode(
    N,
    G,
    V1,
    kappa,
    Lp,
    T,
    p_opp,
    rng,
    near_frac,
    reveal=False,
    w_wait=0.0,
    sharp=1.2,
    proj=0.30,
    pivot_rule="gap2",
):
    committed = np.zeros(N, dtype=bool)
    n0 = int(round(near_frac * N))
    if n0 > 0:
        committed[rng.choice(N, size=min(n0, N), replace=False)] = True
    g = int(committed.sum())
    for t in range(T):
        if g >= G:
            return 1
        h = T - t
        gap = G - g
        movers = np.where((~committed) & (rng.random(N) < p_opp))[0]
        if len(movers) == 0:
            continue
        n_rem = int((~committed).sum())
        p = belief(gap, len(movers), n_rem, h, sharp, proj)
        ev = V1 * p - kappa - Lp * (1 - p)
        if ev >= 0:
            if reveal:
                if pivot_rule == "gap2":
                    pivotal = gap <= 2 * len(movers)
                else:  # "decisive": a mover is pivotal iff movers can just barely close gap
                    pivotal = gap <= len(movers)
                if pivotal and len(movers) > 1 and w_wait > 0:
                    act = movers[rng.random(len(movers)) >= w_wait]
                    committed[act] = True
                else:
                    committed[movers] = True
            else:
                committed[movers] = True
            g = int(committed.sum())
    return 1 if g >= G else 0


def basin(
    N,
    V1,
    kappa,
    Lp,
    T,
    p_opp,
    thetas,
    nears,
    n_runs,
    seed,
    reveal=False,
    w_wait=0.0,
    sharp=1.2,
    proj=0.30,
    pivot_rule="gap2",
):
    rng = np.random.default_rng(seed)
    succ = tot = 0
    for th in thetas:
        G = max(1, int(np.ceil((1 - th) * N)))
        for nf in nears:
            for _ in range(n_runs):
                succ += run_episode(
                    N,
                    G,
                    V1,
                    kappa,
                    Lp,
                    T,
                    p_opp,
                    rng,
                    nf,
                    reveal=reveal,
                    w_wait=w_wait,
                    sharp=sharp,
                    proj=proj,
                    pivot_rule=pivot_rule,
                )
                tot += 1
    return succ / tot


def main():
    print("=" * 78)
    print("INDEPENDENT VERIFY of v5 Route C  (seed=%d, DIFFERENT params/method)" % SEED)
    print("=" * 78)

    # DIFFERENT parameterization from develop (N=50,T=12,p_opp=0.12)
    N, kappa, Lp, T, p_opp = 80, 1.0, 0.5, 10, 0.18
    thetas = np.linspace(0.05, 0.55, 11)
    nears = np.array([0.0, 0.1, 0.2, 0.3])
    n_runs = 50
    w_default = 0.7

    # ---- PROBE 1: folk threshold + basin shrinkage under revelation ----
    print(
        "\n[PROBE 1] folk threshold + masked-vs-revealed (N=80,T=10,p_opp=0.18,w=0.7)"
    )
    print("  V1/kappa   basin_masked   basin_revealed   shrink(M-R)")
    shrinks = []
    for r in [1.0, 1.5, 2.0, 4.0, 8.0]:
        V1 = r * kappa
        bm = basin(
            N,
            V1,
            kappa,
            Lp,
            T,
            p_opp,
            thetas,
            nears,
            n_runs,
            SEED + int(r * 100),
            reveal=False,
        )
        br = basin(
            N,
            V1,
            kappa,
            Lp,
            T,
            p_opp,
            thetas,
            nears,
            n_runs,
            SEED + int(r * 100),
            reveal=True,
            w_wait=w_default,
        )
        shrinks.append(bm - br)
        print(f"   {r:5.1f}      {bm:7.3f}        {br:7.3f}        {bm - br:+7.3f}")
    print(f"  >> mean shrink = {np.mean(shrinks):+.3f}  (claim: > 0)")
    print(f"  >> all shrinks positive? {all(s > 0 for s in shrinks)}")

    # ---- PROBE 2: STRICT code-path symmetry at w_wait=0 ----
    print("\n[PROBE 2] STRICT identity test masked vs revealed(w_wait=0) on SAME seed")
    V1 = 4.0
    eq = True
    diffs = 0
    for th in thetas:
        G = max(1, int(np.ceil((1 - th) * N)))
        for nf in nears:
            for k in range(80):
                s = SEED + 9000 + int(th * 1000) + int(nf * 10) + k
                rm = np.random.default_rng(s)
                rr = np.random.default_rng(s)
                a = run_episode(N, G, V1, kappa, Lp, T, p_opp, rm, nf, reveal=False)
                b = run_episode(
                    N, G, V1, kappa, Lp, T, p_opp, rr, nf, reveal=True, w_wait=0.0
                )
                if a != b:
                    eq = False
                    diffs += 1
    print(
        f"  masked == revealed(w=0) for ALL episodes on paired seed? {eq}  (mismatches={diffs})"
    )
    print(
        "  >> if True: w=0 collapses revealed onto masked EXACTLY -> attrition is the SOLE diff."
    )

    # ---- PROBE 3: robustness of masking advantage to belief magic numbers ----
    print("\n[PROBE 3] masking advantage vs belief steepness SHARP and projection proj")
    print("  SHARP   proj    basin_masked   basin_revealed   advantage")
    for sharp in [0.6, 1.2, 2.4]:
        for proj in [0.15, 0.30, 0.60]:
            bm = basin(
                N,
                4.0,
                kappa,
                Lp,
                T,
                p_opp,
                thetas,
                nears,
                n_runs,
                SEED + 111,
                reveal=False,
                sharp=sharp,
                proj=proj,
            )
            br = basin(
                N,
                4.0,
                kappa,
                Lp,
                T,
                p_opp,
                thetas,
                nears,
                n_runs,
                SEED + 111,
                reveal=True,
                w_wait=w_default,
                sharp=sharp,
                proj=proj,
            )
            print(
                f"  {sharp:4.1f}   {proj:4.2f}    {bm:7.3f}        {br:7.3f}        {bm - br:+7.3f}"
            )

    # ---- PROBE 4: boundary stress (tau->0, tau->1, theta->0, theta->1) ----
    print("\n[PROBE 4] boundary stress")
    # tau near 0 (V1 huge) vs tau near 1 (V1 tiny relative to Lp)
    for V1 in [0.05, 0.5, 100.0]:
        tau = Lp / (V1 + Lp)
        bm = basin(
            N, V1, kappa, Lp, T, p_opp, thetas, nears, n_runs, SEED + 222, reveal=False
        )
        print(f"  V1={V1:7.2f} tau={tau:.4f}: basin_masked={bm:.3f}")
    # theta near 0 (G~N, near-impossible) and near 1 (G~0, trivial), masked
    print("  theta extremes (masked, V1/kappa=4):")
    for th in [0.001, 0.02, 0.5, 0.98, 0.999]:
        G = max(1, int(np.ceil((1 - th) * N)))
        rng = np.random.default_rng(SEED + 333 + int(th * 1000))
        b = np.mean(
            [run_episode(N, G, 4.0, kappa, Lp, T, p_opp, rng, 0.2) for _ in range(300)]
        )
        print(f"    theta={th:.3f} G={G:3d}: basin={b:.3f}")

    # ---- PROBE 5: alternative pivotality predicate (model-robustness of the holdout) ----
    print(
        "\n[PROBE 5] alternative pivotality rule 'decisive' (gap<=movers, not gap<=2*movers)"
    )
    print("  V1/kappa   basin_masked   basin_revealed(decisive)   shrink")
    for r in [2.0, 4.0, 8.0]:
        V1 = r * kappa
        bm = basin(
            N,
            V1,
            kappa,
            Lp,
            T,
            p_opp,
            thetas,
            nears,
            n_runs,
            SEED + 444 + int(r * 100),
            reveal=False,
        )
        br = basin(
            N,
            V1,
            kappa,
            Lp,
            T,
            p_opp,
            thetas,
            nears,
            n_runs,
            SEED + 444 + int(r * 100),
            reveal=True,
            w_wait=w_default,
            pivot_rule="decisive",
        )
        print(
            f"   {r:5.1f}      {bm:7.3f}        {br:7.3f}                 {bm - br:+7.3f}"
        )

    # ---- PROBE 6: holdout-intensity monotonicity (independent params) ----
    print("\n[PROBE 6] revealed basin vs w_wait (monotone decreasing?), V1/kappa=4")
    print("  w_wait   basin_revealed")
    prev = None
    mono = True
    base_m = basin(
        N, 4.0, kappa, Lp, T, p_opp, thetas, nears, n_runs, SEED + 555, reveal=False
    )
    print(f"  (masked baseline = {base_m:.3f})")
    for w in [0.0, 0.25, 0.5, 0.75, 0.95]:
        br = basin(
            N,
            4.0,
            kappa,
            Lp,
            T,
            p_opp,
            thetas,
            nears,
            n_runs,
            SEED + 555 + int(w * 100),
            reveal=True,
            w_wait=w,
        )
        if prev is not None and br > prev + 1e-9:
            mono = False
        prev = br
        print(f"   {w:4.2f}    {br:7.3f}")
    print(f"  >> basin monotone non-increasing in w_wait? {mono}")

    print("\n" + "=" * 78)
    print(
        "VERDICT INPUTS: see PROBE 1 (shrink sign), PROBE 2 (sole-cause), PROBE 3 (artifact),"
    )
    print(
        "                PROBE 4 (boundary), PROBE 5 (model-robust), PROBE 6 (monotone)."
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
