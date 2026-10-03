# Numerical checks — Toy Model v2

Verification scripts for `docs/2026-06-08-ifwishlist-toy-model-v2.md`. Run:

```bash
uv run --with numpy --with scipy python check.py    # Lemma 1 + Prop 3 + Thm 2(a) single-crossing + upper-set test
uv run --with numpy --with scipy python shape.py    # masking-region shape at fixed q (interior band)
uv run --with numpy --with scipy python search.py   # 40k-draw structural scan: is the masking band nonempty?
```

## Results of record (2026-06-08)

- `check.py`: `θ̂ = τ` to 2.4e-15 (Lemma 1, σ-invariant); `ρ(c)=F_c(τ)` strictly decreasing
  (Prop 3); single-crossing in μ holds (Thm 2a); **upper-set-in-c FALSIFIED** (Thm 2 first
  draft) — counterexamples found.
- `shape.py`: at fixed q, masking region is an interior interval in c (hump-shaped policy).
- `search.py`: masking band nonempty in 24,711/40,000 (~62%) structural draws; concentrated at
  intermediate integrity (median winning c≈0.67, ρ≈0.84) and high pivotal exposure (median
  τ≈0.76).

`F_c` = Beta(mean=c, concentration=k). The falsification in `check.py` is what reshaped
Theorem 2 from "masking above a q·c threshold" to "masking in an interior band of c."

## Toy Model v3 (continuous disclosure k) — added 2026-06-08

```bash
uv run --with numpy --with scipy python v3.py        # Thm 3 concavity; triangular closed-form test (constraint + inversion)
uv run --with numpy --with scipy python v3_shape.py  # k*(c) shape on the Beta family (the real U-shape)
```

- `v3.py`: `W''<0` confirmed; triangular closed form matches argmax ONLY where `τk*≤θ_c`
  (115/217 draws), and within that family `k*(c)` is an **inverted-U** — so the closed form was
  demoted to a cautionary example.
- `v3_shape.py`: on v2's Beta family (fixed support, FOSD in c), `k*(c)` is robustly **U-shaped**,
  interior minimum at `c≈0.45–0.70` — masking (low k) in the middle integrity band, consistent
  with v2's hump. This is the result of record for Prop 4.

Lesson repeated from v2: the clean closed form was pathological; numerics on the well-behaved
family carry the comparative static.

## Toy Model v4 (two-signal Morris–Shin microfoundation) — added 2026-06-08

```bash
uv run --with numpy --with scipy python v4.py   # beta->0 nesting; uniqueness boundary; rho vs beta
```

Numerics-FIRST this time (model solved before claims written). Findings of record:

- `β→0` limit gives `θ*=τ` (recovers v2 Lemma 1) — solver correct.
- **Uniqueness boundary `β ≤ √(2πα)`** (Hellwig/Morris–Shin), reproduced: max-β-unique
  `{5.2,7.7,11.2,15.8}` vs `√(2πα) {5.6,7.9,11.2,15.9}` for `α{5,10,20,40}`.
- In the unique region, good-coalition failure `ρ` is INCREASING in public precision `β`
  (`0.09→0.28→0.35→0.40`) — public info destabilizes. So masking-as-strong-public-signal gives
  no protection there; v2's `ρ_M→0` lives only in the multiplicity region `β>√(2πα)` as
  equilibrium SELECTION (v4 Theorem 4, A8).

## Toy Model v5 (selection theory — A8 resolved) — added 2026-06-08

Three independent routes that DERIVE v4's A8 (success-selection) instead of assuming it. Each was
developed numerics-first and re-checked by an adversarial verifier with an independent
re-implementation (different seed/params/method). All three verdicts: `holds_with_correction`.

```bash
uv run --with numpy --with scipy python v5_routeA.py   # Lemma 3 (necessity): theta_RD=tau, no device => no benefit below tau
uv run --with numpy --with scipy python v5_routeB.py   # Theorem 5 (main): masked pass/fail frontier theta_p, 0<theta_p<tau
uv run --with numpy --with scipy python v5_routeC.py   # Theorem 6 (dynamic): masked running total sustains completion; reveal pivotality => holdout
# independent adversarial re-checks (different seed/params/method):
uv run --with numpy --with scipy python v5_A_verify.py
uv run --with numpy --with scipy python v5_B_verify.py
uv run --with numpy --with scipy python v5_C_verify.py
```

- `v5_routeA.py` (**Route A / Lemma 3 — necessity**): Laplacian/risk-dominant cutoff `θ_RD = τ`
  exactly, independent of `β` (MC to ~3e-4). Global-game vanishing-noise limit (`α→∞` at fixed `β`)
  selects `θ*(z=τ) → τ` for `β∈{3,8,20}` (`|θ*−τ|` ~`8e-6` at `α≈1.3e5`). Device-free failure is the
  step at `τ`. CONCLUSION: a bare global game manufactures no masking benefit below `τ`; a designed
  device is necessary. CAVEAT (verifier): this is a LIMIT result that exits multiplicity — at finite
  `β>√(2πα)` the two stable equilibria sit at `θ*≈{0,1}`, neither near `τ`; Route A does not select
  within multiplicity.
- `v5_routeB.py` (**Route B / Theorem 5 — the frontier, main result**): masked pass/fail
  `go=1{θ≥t}`; robust-selection frontier `θ_p≈0.264 < τ=0.40` (v4 calib `α=20`). A8 = `θ_g≥θ_p`;
  panic iff `θ_g<θ_p` (committing mass crosses `1−θ` tangentially at `θ_p`). `θ_p>0` (no first-best);
  rises toward `τ` as `α→∞` (`0.166→0.335`, `α=5→100`). VERIFIER CORRECTIONS folded into the doc: the
  same-machinery finite-`α` bare frontier is `0.434`, so the honest gap is `+0.169` (LARGER than the
  asserted `+0.136`); construction is a clip-Laplacian/binary hybrid so exact `0.264` is
  criterion-specific; `θ_p→0` as `τ→0` so claim holds for `τ≳0.1`.
- `v5_routeC.py` (**Route C / Theorem 6 — dynamic robustness arm**): masked running total → folk
  threshold in `V₁/κ` (basin `0.158→0.679` at `V₁/κ=1.0→1.5`) and horizon (`T` crosses basin `0.5`
  between `T=6,8`). Revealing pivotality SHRINKS the basin via war-of-attrition; isolated to holdout
  intensity (`w_wait=0` ⇒ no shrinkage; `0.716→0.122` as `w_wait:0→0.9`). VERIFIER: only the SIGN of
  the masking advantage is robust (magnitude predicate-specific `+0.07…+0.29`); "instantiates," not
  "derives" — a behavioral best-response sim, not a solved MPE (v6).

Doc of record: `docs/2026-06-08-ifwishlist-toy-model-v5.md`. Citation gaps to close before print
(provisional stubs added): `notes/frankelmorrispauzner2003equilibrium.md`,
`notes/marxmatthews2000dynamic.md`.

## Toy Model v6 (Markov-perfect dynamic selection — A8 endogenized) — added 2026-06-09

Upgrades v5 Route C (a behavioral sim with an exogenous holdout dial) to a genuine finite-horizon MPE
solved by backward induction, with the war-of-attrition waiting rate `w(m,h)` an ENDOGENOUS
best-response fixed point. Doc of record: `docs/2026-06-09-ifwishlist-toy-model-v6.md`.

```bash
uv run --with numpy --with scipy python v6_mpe.py        # MPE engine, both regimes, OSD + folk + comparative statics (~7s)
uv run --with numpy --with scipy python v6_mpe_explore.py # contested-regime parameter search, 2000 cells (~100s)
uv run --with numpy --with scipy python v6_adv_check.py   # fairness check: is the advantage a selection artifact? (~10s)
```

- `v6_mpe.py`: genuine MPE both regimes; one-shot-deviation principle holds to ~`4.4e-16` at every
  reachable state (29 mixed-masked / 21 interior-WoA states at the representative cell). Seeded MC
  cross-check. Folk thresholds of masked & revealed nearly COINCIDE (~`V1/k=4` at `theta=0.15`).
- `v6_adv_check.py`: the masking advantage IS the optimistic(momentum)−pessimistic(holdout) MPE
  selection gap (`0.9346−0.6756=+0.2590` at the representative cell `theta=0.40, V1/k=3, T=10`);
  `36/60` states have momentum as a genuine alternative equilibrium → info-driven SELECTION, not forced.
- `v6_mpe_explore.py`: large-advantage cells are KNIFE-EDGES at the completion-basin boundary; the
  robust effect is a MODEST band at intermediate `K/N` (`theta∈[0.30,0.40]`, advantage `+0.18→+0.26`).
- DIAGNOSTIC: full excludability (`xi=0`) OR synchronous moves (`p_opp=1`) → advantage exactly `0`
  (regimes coincide). The dynamic masking story requires a partial public good AND asynchronous moves.

TWO NUMERICS-FIRST CORRECTIONS (same pattern as v2–v5): (i) a continuation-value bug in the first
`solve_masked` draft (propagating the wait-action value `Vw` instead of the equilibrium continuation
`max(Vw,Vc)`) INFLATED the advantage to `+0.97`; fixing it deflated to the modest/conditional `+0.26`
band. (ii) the clean folk form `V1/k>=1/(1-xi)` does NOT match the aggregate threshold — demoted to a
cautionary heuristic. NET: the dynamic masking advantage is real, genuinely MPE, with endogenous
attrition — but MODEST and CONDITIONAL, and it inherits (does not eliminate) the A8 selection
dependence. FMP 2003 promoted to verified=yes; Marx–Matthews 2000 theorem body still unread.

## Toy Model v7 (cleared the v6 backlog: 5 strands) — added 2026-06-09

```bash
uv run --with numpy --with scipy python v7_selection.py              # S1 #1: micro-found the selection via a logit/QRE dynamic
uv run --with numpy --with scipy python v7_scale.py                  # S3 #2: scale the MPE (finite-N ladder + mean-field continuum) (~100s)
uv run --with numpy --with scipy python v7_screening.py              # S4 #3: endogenize bad-coalition screening s as an MPE object
uv run --with numpy --with scipy --with matplotlib python v7_axes.py # S6 #6: reconcile the four selection boundaries (emits v7_axes_phase.png)
uv run --with numpy --with scipy python v7_unify.py                  # S5 #5: the Block-4 single theorem — custody+spec+disclosure, one objective
```

- `v7_scale.py` (THE HEADLINE DEFLATION): holding `K/N`,`T/N` fixed, the v6 masking advantage decays
  `+0.259(N=10) → +0.007(N=16) → 0(N≥20)`; the mean-field continuum advantage is EXACTLY 0; the
  war-of-attrition holdout `w(0,T)` self-averages away as `N` grows. v5's STATIC gap `τ−θ_p=+0.136`
  SURVIVES the continuum; v6's DYNAMIC gap COLLAPSES — they are NOT the same object. The dynamic
  masking advantage is a finite-`N` pivotality effect (real for small campaigns, not a large-`N` theorem).
- `v7_selection.py`: a logit best-response dynamic from a neutral start DERIVES `π_masked ≥ π_revealed`
  (frontier `λ≈3` at contested `θ`), upgrading v6's selection-by-rule. CORRECTION (the script's own
  narration mislabels this): the `π_revealed` collapse is INFRAMARGINAL free-riding, not a marginal
  war-of-attrition holdout — forcing only the marginal action to withhold gives `π≈0.9919`, no collapse.
  Non-monotone in `λ`: adv `+0.0151`(λ20) → `−0.0097`(λ80). Sign robust over the band; magnitude `λ`-specific.
- `v7_screening.py`: bad-coalition screening `s := 1−Pr(false-activate)` is now an MPE-solved object;
  `s_revealed ≥ s_masked` across the robustness sweep (`s_masked=0.035, s_revealed=1.0` at `θ=0.15`,
  BAD-B illusory-completion). INSTANTIATES, does not derive: BAD-A (lower-dominance) is screened by
  arithmetic in both regimes; the gap lives entirely in the BAD-B operationalization.
- `v7_axes.py`: total unification of the four boundaries is FALSIFIED — they rhyme in location but
  diverge in regime parameter. `θ` (v5) and `K/N` (v6) are the SAME variable (`K/N=1−θ`, tautological);
  the only genuinely verified cross-axis object is the ANALYTIC gate `β/√α=√(2π)=2.5066` (the measured
  value is grid-fragile, `2.33–2.56`); the v6 dynamic band is EMPTY at the shared `τ=0.40` (onset `τ≈0.45`).
- `v7_unify.py` (THE BLOCK-4 SINGLE THEOREM): one objective `W(γ,σ,k;q,c)` makes all three minimalisms
  (no-custody, fuzzy-early, partial-masking) co-occur as minimal in a thin high-`q`/mid-`c` region
  (~3/114 cells) with reversal off it. Disclosure facet reproduces v3 (`max|Δk*|=0.020`). Separability is
  WEAK: argmax-separable 221/600, welfare gap median 0.43% but p95 92% (quasi-separable in threshold
  structure, coupled in welfare magnitude); leading interaction `σ×k=0.094`. Custody-minimalism is the
  BOUNDED Diamond-1984 claim (`γ*=0` only for small commitment-device benefit; `β_γ≥0.25 → custody wins`).

NET (v7): a second controlled deflation that SHARPENS the chain. The durable disclosure spine is STATIC
(v2 `c`-axis reversal + v5 Route B continuum `θ_p`); the dynamic MPE arm is finite-`N`. Selection is now
micro-founded (logit). Marx–Matthews 2000 CLOSED (`verified=yes`, Props 2/3/4 read verbatim — Prop 4 =
MPE completion via observable running total, folk `δ>γ`); FMP 2003 + Marx–Matthews 2000 rows added to the merged corpus (38 sources).

## Toy Model v8 (cleared v7 backlog #2: derive `s` from signal-extraction) — added 2026-06-09

```bash
uv run --with numpy --with scipy python v8_screening_signal.py  # S7: type signal -> derived s(k), phi(k), W_type; excludability regime
uv run --with numpy --with scipy python v8_screening_mpe.py     # S8: public signal in v7's MPE; nests v7; public(affine)/private(convex)
uv run --with numpy --with scipy python v8_two_channel.py       # S9: one dial -> two channels; k*(xi) bang-bang; S5 interior is a c-object
```

- `v8_screening_signal.py` (THE DERIVATION): disclosure `k`=precision of a private TYPE signal
  (good=deliverable/bad=illusory). `s(k)=Pr(BAD fails to reach K commits)` is a Bayesian
  signal-extraction object; v7's `s_masked/s_revealed` are the `k→0/1` LIMITS. Gaussian signal ⇒
  `s(k)` NON-LINEAR (NOT v3's posited linear `s(k)=k`); `φ(k)`=false-reject-of-good FALLS; `W_type(k)`
  monotone↑ in the OPTIMISTIC-prior regime. THREE numerics-first falsifications: F1 (one-shot
  coordination w/ v6 payoffs collapses to all-decline — pivotality artifact, not detection); F2 (the
  EXCLUDABILITY REGIME: type channel has bite only at `ξ<1−κ/V₁=0.667`; high `ξ`⇒free-ride⇒pivotality
  channel; the two disclosure channels live in OPPOSITE `ξ` regimes — WHY v7 had to assume pooling);
  F3 (binary 0/1 rule is knife-edge — low-`k` `s=1` is PARALYSIS, `φ=1` too, not screening; needs the
  continuous Gaussian signal, `λ`-free). **C2 REFUTED-AS-UNIVERSAL by the adversarial pass (sec 7b):**
  the "one-way ratchet" is CONDITIONAL — `s(k)` is NON-MONOTONE (interior dip in 105/108 stressed
  cells; the independent global-game re-impl finds net `s(1)<s(0)` in ~59–62/240 cells, MC-confirmed),
  because under a PESSIMISTIC prior more disclosure lets noisy-"good" signals commit to a bad coalition
  that clears at low K. ALSO HONEST: the curve shape is inherited from the POSITED precision map
  `d'(k)=d_scale·k/(1−k)` — no information-theoretic primitive (entropy/capacity/k-anon leakage) is
  instantiated, so "derived" RELOCATES the assumption (v7 §9 #2 only PARTIALLY closed).
- `v8_screening_mpe.py`: embeds a PUBLIC type signal of precision `k` in `v7_screening.py`'s `solve_bad`
  VERBATIM, SELECTION HELD FIXED (isolates belief from v7's belief-vs-selection confound). Endpoints
  NEST v7 (`π_b=0.1`: `s(0)=0.0513≈s_masked=0.0348`; `s(1)=1.0=s_revealed`). Public `s(k)` is PIECEWISE:
  a PARALYSIS floor `s=1` below `k_act≈0.36`, then ~AFFINE above (affine-fit resid 0.006). ⇒ v3's LINEAR
  `s(k)=k` is the PUBLIC active regime; the PRIVATE signal is CONVEX. The Morris–Shin public/private axis.
- `v8_two_channel.py` (SYNTHESIS, touches #3): ONE dial drives a TYPE channel (`s,φ`; disclosure-favoring)
  AND a PIVOTALITY free-ride channel (`ρ`; masking-favoring) — coupled because one disclosure leaks both
  type and decisiveness (the #3 "why they couple" answer). `k*(ξ)` is BANG-BANG (DISCLOSE plateau at
  `ξ<ξ*`, MASK at `ξ>ξ*`): 0/40 grid cells show a GENUINE interior peak. By contrast v3's `ρ(k)=F_c(τk)`
  (the `c`-integrity-distribution channel) DOES give interior peaks (`c=0.6→0.64, 0.7→0.71, 0.85→0.88`)
  ⇒ S5's interior partial-masking `k*` is a `c`-DISTRIBUTION object, NOT the `ξ` tradeoff. CAVEAT: the
  static synthesis under-represents the high-`ξ` regime (v6's dynamic momentum, finite-`N` per v7 §2).
- `v8_verify_reimpl.py`: the adversarial verifier's INDEPENDENT continuous-fundamental global-game
  re-implementation (`x~N(m0,1/tau0)`, bad iff `x<0`, threshold eq, finite quadratic precision ramp —
  different on every axis from v8). Confirms C1 (limits), C3 (excludability `ξ*=0.667` exactly), C4
  (non-linear); REFUTES C2-universal. Run: `uv run --with numpy --with scipy python v8_verify_reimpl.py`.

NET (v8): PARTIALLY closes v7 backlog #2 with an HONEST "relocation" (the precision→`k` map is a
standard Morris–Shin/Blackwell modeling choice, not an instantiated information measure — entropy/
capacity/k-anon leakage). The deepest content is a reorganization: the disclosure dial controls TWO
channels in OPPOSITE excludability regimes — a TYPE/screening channel (disclosure-favoring at `ξ<1−κ/V₁`,
but CONDITIONAL — `s(k)` non-monotone, can dip under pessimistic priors; adversarially refuted as a
universal one-way ratchet) and a PIVOTALITY channel (v6, finite-`N`, masking-favoring at high `ξ`).
Masking-is-optimal is thus narrower than v7 said (needs high `ξ` AND small `N` AND the dynamic
mechanism); S5's interior partial-masking is confirmed a `c`-axis (`F_c`) object, not the `ξ` tradeoff.
All three v8 scripts adversarially re-checked (independent re-impl + auditor + lit + critic): verdicts
`holds_with_correction`. Doc of record: `docs/2026-06-09-ifwishlist-toy-model-v8.md`. Merged corpus
unchanged (38 sources; no new sources added).

## Toy Model v9 (the 3 load-bearing backlog items; mostly-convergence) — added 2026-06-09

> NB: the adversarial pass FORCED A CORRECTION ON ALL THREE STRANDS — the first drafts of S10 and S11
> were WRONG (caught by independent re-implementations). The bullets below are the CORRECTED versions.

```bash
uv run --with numpy --with scipy python v9_kanon_leakage.py        # S12: Shannon mutual-info leakage primitive (v8 §7 #1)
uv run --with numpy --with scipy python v9_selection_lambdafree.py # S10: lambda-free risk-dominant selection (v7 §9 #1)
uv run --with numpy --with scipy python v9_dynamic_two_channel.py  # S11: dynamic two-channel synthesis (v8 §7 #3)
```

- `v9_kanon_leakage.py` (S12 — closes v8's MAIN deflation): the type-signal precision is now an ACTUAL
  information measure — the Shannon mutual information `I(type; k-anonymized support count)` — replacing
  v8's posited ramp `d'(k)=d_scale·k/(1−k)`. Mechanism: publish a type-correlated support count
  `m~Binom(N,p_psi)` (`p_G>p_B`) under k-anonymity = additive blur of half-width `r(k)=round((1−k)N)`.
  `I(k)` MONOTONE (data-processing; the first-draft equal-width re-bucketing was non-nested -> spurious
  non-monotone `I`, FALSIFIED and fixed via additive blur) + SATURATING (`I(1)=0.62 bits < H=0.97`; the
  exact count does NOT perfectly reveal type — binomial overlap). CORRECTION (adversarial): `I(k)` is
  CONVEX for additive blur (1/11 concave), NOT concave — the interior shape is mechanism-dependent (DP
  geometric is concave); saturation is the robust feature. KEY CORRECTION of v8: `s(1)<1` (`0.89` at
  support-gap 0.1) vs v8's ramp artifact `s(1)=1` — BUT regime-dependent: `s(1)->1` at large gap (>=0.80)
  = v8's perfect-separation LIMIT (v8 was a limiting case, not wrong). The s(k) non-monotonicity (v8 C2)
  is INDEPENDENTLY reproduced (and cross-mechanism: RR/DP/equal-count-k-anon all give monotone `I`,
  `s(1)<1`, the C2 dip). PARTIAL CLOSE: at k=1 every mechanism collapses to the exact count, so the
  endpoint is pinned by `(p_G,p_B)` ALONE; the close is for the ENDPOINT, not the curve. 'Derived' is
  re-grounded ONE LEVEL DEEPER (economic primitive + real mechanism), not eliminated; regress not escaped.
- `v9_selection_lambdafree.py` (S10 — closes v7 §9 #1 POSITIVELY): Carlsson-van Damme / RISK-DOMINANCE
  (a p_opp-CONSISTENT Laplacian belief, uniform over the commit PROPENSITY) replaces v7 S1's lambda-
  specific logit. NO lambda -> NO inversion; and it SUSTAINS masked momentum (`pi_masked~0.99` at every
  theta, no collapse). Masking advantage robustly POSITIVE in the CONTESTED regime (`theta<=0.30`: +0.99,
  the revealed inframarginal free-ride collapses revealed clearing there) and a TIE elsewhere — NO easy-
  regime reversal, NO masked collapse. CORRECTION (adversarial): the first-draft "pure-free-rider collapse
  pi_masked->0 / revelation-wins-easy regime split" was a BELIEF ARTIFACT (a Laplacian that ignored
  p_opp, asymmetric with the Binom(n,p_opp\*a) kernel); two independent lambda-free refinements (CvD +
  potential-maximizer) confirm it vanishes. Mechanism = inframarginal free-ride (piv_force). Finite-`N`
  (v7 §2) carries. License: cite FMP Sec 6.4 (LP-maximizer conditions), not Thm 1 alone.
- `v9_dynamic_two_channel.py` (S11 — closes v8 §7 #3; the headline FLIPPED after the adversarial pass):
  both channels inside the v6/v7 DYNAMIC (momentum) game on ONE dial k. CORRECTION (the first-draft
  "0/28 bang-bang / no interior" was FALSE ON ITS OWN OUTPUT): it was calibration-cherry-picked at
  theta=0.40 (first value past a theta-cliff) with a coarse xi grid skipping the 0.72-0.78 band, and its
  "W~0.13 near-dead" justification matched NO computed number. CORRECTED: the dynamic model DOES host a
  genuine LIVE interior k\* (theta=0.30, xi=0.667: `G(k)` inverted-U, both endpoints ~0, interior W=+2.39
  beats both; 6/13 fine-grid PEAK cells; live across theta~0.15-0.40). WHAT SURVIVES: the DECOUPLED two-
  dial optimum is the CORNER (k_type=1, k_piv=0) -> the interior is NOT a channel-BALANCE object; it is
  the CONSTRAINED optimum of the BUNDLED k-anon dial (you can't disclose type without leaking pivotality).
  So partial masking has TWO sources: the c-distribution (v3/S9) AND the one-dial coupling (S11), neither
  a balance.
- `v9_verify_s10.py` / `v9_verify_s11.py` / `v9_verify_s12.py`: the adversarial verifiers' INDEPENDENT
  re-implementations (CvD+potential-maximizer / decoupled two-dials + finer grid / RR+DP+equal-count
  k-anon) that FORCED the corrections above. See the v9 doc §5.

NET (v9): mostly-convergence, honestly. S10 and S12 RE-PRICE/CORRECT prior results (S10 confirms v7 S1's
contested-regime advantage lambda-free, positively; S12 corrects v8's s(1)=1 artifact + grounds the
precision one layer deeper) -- the diminishing-returns signature. BUT S11 produced a genuine NEW result
(a live dynamic interior k\*, constrained-optimal under the bundled dial). AND the adversarial pass had to
correct the first draft of EVERY strand -- itself a signal that single-pass results are now unreliable
and the marginal honest result costs a full adversarial cycle. The hard core (v2 `c`-axis reversal + v5
Route B continuum `θ_p`) survives untouched. vN+1 should be DATA-triggered (measure `p_G−p_B`, `ξ` per
campaign type, the campaign-size distribution, `π_b`/`c` — ifwishlist live 2026-06-08). Doc of record:
`docs/2026-06-09-ifwishlist-toy-model-v9.md`. Merged corpus unchanged (38 sources).
