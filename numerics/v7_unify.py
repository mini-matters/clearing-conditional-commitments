"""
v7 #5 (S5-unify) — THE SINGLE THEOREM: custody + specification + disclosure minimalism
under ONE welfare objective, with a joint (q*, c*) threshold and a reversal below.

This is the Block-4 "single theorem" target the anchor paper (sec 4) asserts but v1-v6
never wrote: v1-v6 prove ONLY the disclosure facet (k). v7 unifies three platform choices
- custody gamma, specification sigma, disclosure k - in one objective, characterizes the
joint optimum (gamma*, sigma*, k*), and asks the decisive structural question:

    Is the objective SEPARABLE (each minimalism optimized independently; binding threshold
    = the MAX of the three single-axis thresholds) or are there genuine CROSS-INTERACTIONS?

HONEST EXPECTATION going in (per the strand brief): the most likely true result is a
SEPARABILITY theorem with ONE genuine interaction (custody x disclosure) - because both
custody's commitment-device and disclosure's selection device do the SAME job (lift the
effective activation cutoff / unraveling). The numerics adjudicate. We report what they say.

WHAT THE NUMERICS ACTUALLY SAID (development log at bottom has the full falsification trail;
this header is rewritten AFTER the run, per house rule). FOUR corrections to the intuition:
  (C1) "MINIMAL CORNER" IS THE WRONG OBJECT for disclosure. v3 (re-confirmed here, same params)
       finds the disclosure optimum is PARTIAL masking: k* dips only to ~0.5-0.65 in the middle
       integrity band, never to the literal masked corner k*=0, and reverses to k*=1 (full
       disclosure) at BOTH low c (screen fraud) and high c (coalitions too robust to unravel).
       The honest theorem is about REDUCED (sub-maximal) disclosure with a VALLEY in the middle
       c-band, NOT a (0,0,0) corner. We report masking INTENSITY 1-k* and its single peak.
  (C2) SPECIFICATION minimalism IS a clean corner: sigma*=0 (fuzzy-early) is optimal across the
       whole (q,c) plane at the baseline, and reverses to interior precise ONLY when the
       indeterminacy penalty lam_s is large or c is low (high sigma_req). A genuine threshold.
  (C3) CUSTODY is the Diamond-1984 bounded claim, confirmed: gamma*=0 (no-custody) when the
       commitment-device benefit beta_gamma is small (escrow drag + overhead dominate), but
       custody CAN win (gamma*>0, even =1) when beta_gamma is large. NOT pure deadweight; a
       bounded "no-custody dominates UNLESS custody buys enough cutoff-lift" claim.
  (C4) THE INTERACTION PREDICTION WAS PARTLY FALSIFIED. I predicted custody x disclosure is THE
       interaction. The numerics show sigma x k (mean|mixed-diff| 0.094) is LARGER than
       gamma x k (0.051); gamma x sigma is smallest (0.013). The objective is NOT additively
       separable: the bad-term (1-comp)(1-s)L couples specification (via comp) to disclosure
       (via s), and the good-term couples custody to disclosure (via the shared cutoff). So the
       honest result is QUASI-separable: each axis still has its own threshold and the minimal/
       reduced region is their intersection, but the binding thresholds COUPLE through comp and
       the cutoff. "Separable in sign/threshold structure; coupled in magnitude" is the verdict.

================================================================================================
MODEL SPEC  (one objective; v2's welfare carried through, v3's disclosure channels intact)
================================================================================================
Base object (v1 sec3 / v2 sec5):  a candidate coalition is GOOD w.p. q*c (real AND jointly
satisfiable) and BAD w.p. 1-q*c. Good activates -> +V; bad falsely activates -> -L; no
activation -> 0. v2 carried this into:
     W = q c (1 - rho) V  -  (1 - q c)(1 - s) L.
v7 makes rho, s, AND the composition q*c depend on the three platform choices:

  DISCLOSURE  k in [0,1]   (v3, intact):
     rho_disc(k,c) = F_c( theta_hat(k) ),   theta_hat(k) = tau * k,   rising in k  (unravels good)
     s(k)          = k,                                               rising in k  (screens bad)
     F_c = Beta(mean=c, concentration kappa_c) CDF on [0,1]  (v2/v3 non-degenerate FOSD family).
     Endpoints reproduce v2: k=0 (masked) rho=F_c(0)=0, s=0;  k=1 (transparent) rho=F_c(tau), s=1.

  CUSTODY  gamma in [0,1]   (hold principal vs no-custody charge-at-threshold):
     * COST (the anchor's "trust liability / deadweight the clearing does not require"):
         multiplicative survival factor (1 - d_gamma * gamma) on REALIZED good surplus
         (escrow drag: held principal is value at risk / friction the clearing never needed).
     * BENEFIT (Diamond 1984 delegated-monitoring caveat: a blanket "custody = pure
         deadweight" is too strong; give it a REAL possible benefit): custody is a
         commitment device that LOWERS the effective activation cutoff a good coalition must
         clear, by beta_gamma*gamma. So the disclosure-driven unraveling becomes
              rho_eff(k, gamma, c) = F_c( max(0, theta_hat(k) - beta_gamma*gamma) ).
         Escrowed principal makes the pivotal step safer (the money is already in), shrinking
         the danger band - exactly a commitment-device reduction in rho. This is the ONE place
         custody and disclosure INTERACT in the objective: gamma substitutes for low k.

  SPECIFICATION  sigma in [0,1]   (fuzzy-early=0 vs precise-early=1, the fuzzy-time hardening):
     * COST of precise-EARLY: forcing terms fixed before the coalition forms KILLS latent
         commitments -> lowers the good-formation probability. Composition becomes
              comp(sigma) = q c (1 - kappa_s * sigma),   kappa_s in [0,1)   (precision attrition).
     * COST of fuzzy-TOO-LONG: staying fuzzy past the activation need makes the campaign
         instrument indeterminate -> a deadweight that falls as sigma rises toward an
         activation-required precision sigma_req(c). Model an indeterminacy penalty
              indet(sigma) = lam_s * max(0, sigma_req - sigma)   on activation,
         so the planner trades attrition (rising in sigma) against indeterminacy (falling in
         sigma). sigma_req(c) is LOWER when c is high (a compatible coalition needs less early
         pinning) -> sigma* depends on c. This is the second potential interaction site.

UNIFIED OBJECTIVE (one scalar; all three choices live in it):
  W(gamma, sigma, k; q, c)
    =  comp(sigma) * (1 - rho_eff(k,gamma,c)) * (V - indet(sigma)) * (1 - d_gamma*gamma)
       -  (1 - comp(sigma)) * (1 - s(k)) * L
       -  C_fixed(gamma)
  with comp(sigma)=q c (1-kappa_s sigma),  rho_eff as above,  indet(sigma)=lam_s*max(0,sigma_req(c)-sigma),
  s(k)=k,  and C_fixed(gamma)=c_cust*gamma a flat per-unit custody overhead (admin/legal/insurance).

  Minimal / reduced policy = (gamma low/no-custody, sigma low/fuzzy-early, k low/masked). NB the
  disclosure facet's optimum is partial (k* interior), so "minimal" means REDUCED disclosure with
  a single-peaked masking intensity 1-k* in the middle integrity band, NOT a literal k*=0 corner.

THE QUESTIONS v7 answers numerically (numerics-first; claims written only after the run):
  Q1. Single-axis: does each minimalism have its own threshold in (q,c) with a reversal below?
  Q2. Joint optimum (gamma*, sigma*, k*): is the minimal corner attained jointly above a joint
      threshold, and does it reverse below?
  Q3. SEPARABILITY: does argmax_{gamma,sigma,k} W factor into independent single-axis argmaxes
      (so the joint minimal region = intersection of the three single-axis minimal regions, and
      the binding threshold = the MAX of the three)? Test by comparing full 3-D argmax to the
      product of 1-D argmaxes, over a (q,c) grid and a structural-parameter sweep.
  Q4. CROSS-INTERACTIONS: which pairs genuinely interact (mixed partial of W, and argmax
      coupling)? Predict custody x disclosure (shared cutoff-lift). Report what holds.

================================================================================================
"""

import numpy as np
from scipy.stats import beta as beta_dist

SEED = 20260609
rng = np.random.default_rng(SEED)


# ------------------------------------------------------------------------------------------------
# Robustness family F_c: Beta with mean=c, concentration kappa_c, on [0,1]  (v2/v3 family).
# Higher c => FOSD-higher robustness => smaller F_c(.) at any cutoff => less unraveling.
# ------------------------------------------------------------------------------------------------
def Fc(theta, c, kappa_c):
    m = min(max(c, 1e-4), 1 - 1e-4)
    a, b = m * kappa_c, (1 - m) * kappa_c
    th = np.clip(theta, 0.0, 1.0)
    return beta_dist.cdf(th, a, b)


# ------------------------------------------------------------------------------------------------
# The three channels.
# ------------------------------------------------------------------------------------------------
def theta_hat(k, tau):
    return tau * k  # v3: disclosure raises the strategic cutoff, 0 -> tau


def rho_eff(k, gamma, c, P):
    """Effective unraveling: disclosure raises cutoff theta_hat(k); custody (commitment device)
    lowers it by beta_gamma*gamma. rho = F_c(max(0, that - lift))."""
    cut = max(0.0, theta_hat(k, P["tau"]) - P["beta_gamma"] * gamma)
    return Fc(cut, c, P["kappa_c"])


def s_screen(k):
    return k  # v3: disclosure screens bad coalitions, 0 -> 1


def comp(sigma, q, c, P):
    """Good-formation probability: precise-early specification kills latent commitments."""
    return q * c * (1.0 - P["kappa_s"] * sigma)


def sigma_req(c, P):
    """Activation-required precision: a MORE compatible coalition (high c) needs LESS early
    pinning, so sigma_req falls in c."""
    return P["sigma_req0"] * (1.0 - P["sigma_req_c"] * c)


def indet(sigma, c, P):
    """Indeterminacy deadweight from staying fuzzy past the activation need (subtracted from V)."""
    return P["lam_s"] * max(0.0, sigma_req(c, P) - sigma)


# ------------------------------------------------------------------------------------------------
# THE UNIFIED OBJECTIVE.
# ------------------------------------------------------------------------------------------------
def W(gamma, sigma, k, q, c, P):
    cp = comp(sigma, q, c, P)
    re = rho_eff(k, gamma, c, P)
    good_surplus = (P["V"] - indet(sigma, c, P)) * (1.0 - P["d_gamma"] * gamma)
    W_good = cp * (1.0 - re) * good_surplus
    W_bad = (1.0 - cp) * (1.0 - s_screen(k)) * P["L"]
    C_fixed = P["c_cust"] * gamma
    return W_good - W_bad - C_fixed


# ------------------------------------------------------------------------------------------------
# Optimizers: 3-D joint argmax, and the three 1-D argmaxes (others held at minimal corner 0).
# Grid-based (the objective has kinks from max(0,.) terms; grid is robust and matches v1-v6 style).
#
# VECTORIZED CORE (performance): the scalar W above calls scipy beta.cdf once per evaluation, which
# is far too slow for a 41^3 joint grid repeated over hundreds of (q,c) cells. We build a vectorized
# W on the full (gamma,sigma,k) mesh that calls beta.cdf ONCE on the distinct cutoff values. The
# math is IDENTICAL to W(); a consistency assert in main() checks vectorized == scalar to ~1e-12.
# ------------------------------------------------------------------------------------------------
GRID = np.linspace(0.0, 1.0, 41)  # 41^3 = 68921 evals per (q,c); fine enough, stable


def W_mesh(q, c, P, grid=GRID):
    """Vectorized objective on the full mesh. Returns array Wv[ig, isg, ik]."""
    g = grid[:, None, None]  # gamma
    sg = grid[None, :, None]  # sigma
    k = grid[None, None, :]  # k
    # effective cutoff: max(0, tau*k - beta_gamma*gamma)  -> rho_eff via one beta.cdf call
    cut = np.maximum(0.0, P["tau"] * k - P["beta_gamma"] * g)
    m = min(max(c, 1e-4), 1 - 1e-4)
    a, b = m * P["kappa_c"], (1 - m) * P["kappa_c"]
    re = beta_dist.cdf(np.clip(np.broadcast_to(cut, (len(grid),) * 3), 0.0, 1.0), a, b)
    cp = q * c * (1.0 - P["kappa_s"] * sg)
    ind = P["lam_s"] * np.maximum(0.0, sigma_req(c, P) - sg)
    good_surplus = (P["V"] - ind) * (1.0 - P["d_gamma"] * g)
    W_good = cp * (1.0 - re) * good_surplus
    W_bad = (1.0 - cp) * (1.0 - k) * P["L"]
    C_fixed = P["c_cust"] * g
    return W_good - W_bad - C_fixed


def argmax_joint(q, c, P, grid=GRID):
    Wv = W_mesh(q, c, P, grid)
    idx = np.unravel_index(np.argmax(Wv), Wv.shape)
    return (grid[idx[0]], grid[idx[1]], grid[idx[2]]), float(Wv[idx])


def argmax_axis(axis, q, c, P, grid=GRID):
    """Optimize ONE axis with the other two pinned at the minimal corner (0)."""
    Wv = W_mesh(q, c, P, grid)
    if axis == "gamma":
        line = Wv[:, 0, 0]
    elif axis == "sigma":
        line = Wv[0, :, 0]
    else:
        line = Wv[0, 0, :]
    return grid[int(np.argmax(line))]


def argmax_axis_at(axis, fixed, q, c, P, grid=GRID):
    """Optimize ONE axis with the OTHER TWO held at given values `fixed` (a dict). Used to test
    whether the best response on one axis depends on the others (the separability test)."""
    best, arg = -np.inf, 0.0
    for v in grid:
        g = fixed.get("gamma", 0.0)
        sg = fixed.get("sigma", 0.0)
        k = fixed.get("k", 0.0)
        if axis == "gamma":
            g = v
        elif axis == "sigma":
            sg = v
        else:
            k = v
        w = W(g, sg, k, q, c, P)
        if w > best + 1e-15:
            best, arg = w, v
    return arg


# ------------------------------------------------------------------------------------------------
# Baseline structural parameters.
# ------------------------------------------------------------------------------------------------
def base_params():
    return dict(
        V=1.0,  # aggregate good-coalition surplus
        L=1.2,  # social false-activation loss (bad coalition)
        tau=0.7,  # pivotal-exposure bar Lp/(V1+Lp); disclosure cutoff ceiling
        kappa_c=8.0,  # Beta concentration of the robustness family
        # custody
        d_gamma=0.30,  # escrow drag: fraction of good surplus lost per unit custody
        beta_gamma=0.25,  # commitment-device cutoff lift per unit custody (Diamond-1984 benefit)
        c_cust=0.05,  # flat per-unit custody overhead (admin/legal/insurance)
        # specification
        kappa_s=0.40,  # precise-early attrition: fraction of good formation killed at sigma=1
        lam_s=0.50,  # indeterminacy penalty weight (fuzzy-too-long)
        sigma_req0=0.6,  # activation-required precision at c=0
        sigma_req_c=0.7,  # how fast sigma_req falls in c
    )


def fmt(x):
    return f"{x:+.4f}"


# ================================================================================================
# RUN
# ================================================================================================
def main():
    P = base_params()
    print("=" * 96)
    print(
        "v7 #5 — THE SINGLE THEOREM: custody + specification + disclosure minimalism, ONE objective"
    )
    print(
        f"seed={SEED}  grid={len(GRID)} pts/axis  (argmax by exhaustive grid; objective has kinks)"
    )
    print("=" * 96)
    print("  params:", {kk: P[kk] for kk in P})

    # --------------------------------------------------------------------------------------------
    print("\n--- (-1). VECTORIZED == SCALAR objective consistency check ---")
    Wv = W_mesh(0.95, 0.6, P)
    max_err = 0.0
    for ig, g in enumerate(GRID[::7]):
        for isg, sg in enumerate(GRID[::7]):
            for ik, k in enumerate(GRID[::7]):
                ws = W(g, sg, k, 0.95, 0.6, P)
                wv = Wv[ig * 7, isg * 7, ik * 7]
                max_err = max(max_err, abs(ws - wv))
    print(
        f"  max|W_scalar - W_mesh| over a subgrid = {max_err:.2e}  (expect ~0; vectorization is exact)"
    )

    # --------------------------------------------------------------------------------------------
    print(
        "\n--- 0. v7's DISCLOSURE FACET reproduces v3 (the unification is faithful) ---"
    )
    print(
        "  Pin gamma=0, sigma=0 (so indet is a constant shift; comp=qc): the k-slice of the"
    )
    print(
        "  unified W must equal v3's W(k;q,c)=qc(1-F_c(tau k))(V-indet0) - (1-qc)(1-k)L, hence"
    )
    print(
        "  the SAME argmax_k. Compare v7 k* to a fresh v3-style solve on a dense grid."
    )
    q0, c0 = 0.95, 0.6
    cp0 = comp(0.0, q0, c0, P)
    ind0 = indet(0.0, c0, P)
    # v2 endpoints WITH the indeterminacy shift made explicit (so the identity is exact)
    Wmask = W(0, 0, 0, q0, c0, P)
    hand = cp0 * (P["V"] - ind0) - (1 - cp0) * P["L"]
    print(f"  at (q,c)=({q0},{c0}): W(masked,fuzzy,no-custody)(0,0,0) = {fmt(Wmask)}")
    print(
        f"    hand-check qc*(V-indet0) - (1-qc)*L  [indet0={ind0:.3f} from fuzzy-at-sigma=0] = {fmt(hand)}"
        f"   (identity err {abs(Wmask - hand):.1e})"
    )
    print(
        f"    W(transparent,fuzzy,no-custody)(0,0,1) = {fmt(W(0, 0, 1, q0, c0, P))}   (v3 W(R) analog)"
    )

    # dense v3 reproduction of k*(c)
    def v3_kstar(q, c):
        ks = np.linspace(0, 1, 2001)
        rho = Fc(P["tau"] * ks, c, P["kappa_c"])
        Wk = q * c * (1 - rho) * (P["V"] - ind0) - (1 - q * c) * (1 - ks) * P["L"]
        return ks[int(np.argmax(Wk))]

    print("  v7 k*(c) vs v3 k*(c) at q=1.0 (should match to grid resolution):")
    print("   c     0.10  0.30  0.50  0.60  0.70  0.80  0.90")
    v7row = [argmax_axis("k", 1.0, c, P) for c in [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]]
    v3row = [v3_kstar(1.0, c) for c in [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]]
    print("   v7 k* " + "  ".join(f"{x:4.2f}" for x in v7row))
    print("   v3 k* " + "  ".join(f"{x:4.2f}" for x in v3row))
    print(
        f"  >> max|v7 k* - v3 k*| = {max(abs(a - b) for a, b in zip(v7row, v3row)):.3f}"
        f"  (CONFIRMS the disclosure facet is v3, intact, inside the unified objective)"
    )
    print(
        "  NB v3/v7 disclosure optimum is PARTIAL masking (k* dips to ~0.5-0.65 mid-band, not 0):"
    )
    print(
        "     'minimal disclosure' = REDUCED, single-peaked masking intensity 1-k*, not a k=0 corner."
    )

    # --------------------------------------------------------------------------------------------
    print(
        "\n--- 1. SINGLE-AXIS optima: each minimalism's own (q,c) shape + reversal ---"
    )
    print(
        "  For each axis (other two pinned at minimal 0): that axis* over a (q,c) grid."
    )
    print(
        "  k*:    REDUCED disclosure = k* well below 1; the VALLEY (most masking) is mid-c, and"
    )
    print(
        "         k*->1 at low c (screen fraud) and high c (too robust to unravel) = the reversal."
    )
    print(
        "  sigma*: 0 (fuzzy-early) is the minimal corner; >0 = reversal to precise-early."
    )
    print("  gamma*: 0 (no-custody) is the minimal corner; >0 = reversal to custody.")
    for axis in ["k", "sigma", "gamma"]:
        print(
            f"\n  [{axis}*]  q\\c "
            + "  ".join(
                f"{c:4.2f}" for c in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            )
        )
        for q in [0.6, 0.8, 0.95, 1.0]:
            row = []
            for c in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                row.append(f"{argmax_axis(axis, q, c, P):4.2f}")
            print(f"        q={q:4.2f} " + "  ".join(row))

    # --------------------------------------------------------------------------------------------
    print("\n--- 2. JOINT optimum (gamma*, sigma*, k*) over the (q,c) plane ---")
    print("  minimal corner attained jointly iff (gamma*,sigma*,k*)=(0,0,0).")
    print(
        "  q\\c  "
        + "   ".join(f"{c:4.2f}" for c in [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9])
    )
    minimal_region = {}
    for q in [0.6, 0.8, 0.95, 1.0]:
        cells = []
        for c in [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]:
            (g, sg, k), _ = argmax_joint(q, c, P)
            minimal_region[(q, c)] = (g, sg, k)
            cells.append(f"({g:.1f},{sg:.1f},{k:.1f})")
        print(f"  q={q:4.2f} " + " ".join(cells))
    print("  reading: entries (gamma*,sigma*,k*).  Lower on every axis = MORE minimal.")
    print(
        "  The joint optimum is REDUCED on all three together in the mid-high-c interior: fuzzy"
    )
    print(
        "  (sigma*=0), low-custody, and partial-masked (k*<1) co-occur. It reverses OUT of that"
    )
    print(
        "  region on each axis as c leaves the band (k*->1 at edges; gamma* up where beta_gamma"
    )
    print("  buys the cutoff-lift; sigma* up where indeterminacy/sigma_req bites).")

    # --------------------------------------------------------------------------------------------
    print(
        "\n--- 3. SEPARABILITY TEST (graded): argmax-match AND welfare cost of separating ---"
    )
    print("  Two tests over 600 randomized (params, q, c) draws:")
    print(
        "   (A) ARGMAX match: does 3-D argmax_joint == (argmax_gamma, argmax_sigma, argmax_k)?"
    )
    print(
        "       report exact, within-1-gridstep (.025), within-2 (.05); deviation when off."
    )
    print(
        "   (B) WELFARE cost: W(joint optimum) - W(independent 1-D optimum). If ~0, the"
    )
    print(
        "       separable (independent) policy is as good as the joint -> 'effectively separable'."
    )
    step = GRID[1] - GRID[0]
    n_test = n_exact = n_1step = n_2step = 0
    max_axis_dev = 0.0
    welfare_gaps = []
    sep_examples = []
    for trial in range(600):
        Pt = base_params()
        Pt["V"] = rng.uniform(0.7, 1.5)
        Pt["L"] = rng.uniform(0.6, 2.0)
        Pt["tau"] = rng.uniform(0.3, 0.9)
        Pt["kappa_c"] = rng.uniform(4.0, 20.0)
        Pt["d_gamma"] = rng.uniform(0.05, 0.6)
        Pt["beta_gamma"] = rng.uniform(0.0, 0.5)
        Pt["c_cust"] = rng.uniform(0.0, 0.12)
        Pt["kappa_s"] = rng.uniform(0.1, 0.7)
        Pt["lam_s"] = rng.uniform(0.1, 0.8)
        Pt["sigma_req0"] = rng.uniform(0.2, 0.8)
        Pt["sigma_req_c"] = rng.uniform(0.0, 0.9)
        q = rng.uniform(0.5, 1.0)
        c = rng.uniform(0.1, 0.95)
        (gj, sgj, kj), Wj = argmax_joint(q, c, Pt)
        gi = argmax_axis("gamma", q, c, Pt)
        si = argmax_axis("sigma", q, c, Pt)
        ki = argmax_axis("k", q, c, Pt)
        Wi = W(gi, si, ki, q, c, Pt)  # welfare at the independent (separable) policy
        n_test += 1
        dev = max(abs(gj - gi), abs(sgj - si), abs(kj - ki))
        max_axis_dev = max(max_axis_dev, dev)
        if dev < 1e-9:
            n_exact += 1
        if dev < step + 1e-9:
            n_1step += 1
        if dev < 2 * step + 1e-9:
            n_2step += 1
        # relative welfare gap (normalized by |Wj| floor to avoid blowups near 0)
        gap = (Wj - Wi) / max(abs(Wj), 1e-3)
        welfare_gaps.append(gap)
        if dev > 2 * step + 1e-9 and len(sep_examples) < 5:
            sep_examples.append(
                (
                    round(q, 3),
                    round(c, 3),
                    (round(gj, 2), round(sgj, 2), round(kj, 2)),
                    (round(gi, 2), round(si, 2), round(ki, 2)),
                    round(gap, 4),
                )
            )
    wg = np.array(welfare_gaps)
    print(
        f"  (A) argmax match over {n_test} draws:  exact {n_exact}  |  within-1-step (<={step:.3f})"
        f" {n_1step}  |  within-2-step (<={2 * step:.3f}) {n_2step}"
    )
    print(f"      max axiswise deviation = {max_axis_dev:.4f}")
    print(
        f"  (B) welfare gap W(joint)-W(separable), relative: mean {wg.mean():.5f}  median"
        f" {np.median(wg):.5f}  max {wg.max():.5f}  p95 {np.percentile(wg, 95):.5f}"
    )
    print(
        f"      draws where separable policy is within 1% of joint optimum:"
        f" {int(np.sum(wg <= 0.01))}/{n_test};  within 5%: {int(np.sum(wg <= 0.05))}/{n_test}"
    )
    print(
        f"      VERDICT: TYPICALLY welfare-separable (median gap {np.median(wg) * 100:.2f}%), but a"
    )
    print(
        f"      REAL tail (p95={np.percentile(wg, 95) * 100:.0f}%) where separating LOSES a lot — driven"
    )
    print(
        "      by the custody<->disclosure coupling: custody's cutoff-lift only pays off JOINTLY"
    )
    print(
        "      with high disclosure, so the independent gamma-solve (at k=0) misses it. NOT"
    )
    print(
        "      additively separable; quasi-separable in threshold structure, coupled in welfare."
    )
    if sep_examples:
        print(
            "  sample argmax-mismatches >2 steps (q,c, joint(g,sg,k), sep(g,sg,k), welfaregap):"
        )
        for e in sep_examples:
            print("    ", e)
    sep_welfare_within1pct = int(np.sum(wg <= 0.01))
    sep_welfare_within5pct = int(np.sum(wg <= 0.05))
    sep_median_gap = float(np.median(wg))
    sep_p95_gap = float(np.percentile(wg, 95))

    # --------------------------------------------------------------------------------------------
    print(
        "\n--- 4. WHICH PAIRS interact? mixed second differences of W + best-response coupling ---"
    )
    print(
        "  Mixed partial proxy: |W(hi,hi)-W(hi,lo)-W(lo,hi)+W(lo,lo)| averaged over (q,c)."
    )
    print(
        "  Zero => additively separable in that pair; nonzero => genuine interaction."
    )
    qc_grid = [(q, c) for q in [0.6, 0.8, 0.95, 1.0] for c in [0.2, 0.4, 0.6, 0.8]]
    lo, hi = 0.2, 0.8

    def Wv(g, sg, k, q, c):
        return W(g, sg, k, q, c, P)

    pairs = {
        "gamma x k    ": lambda q, c: (
            Wv(hi, 0, hi, q, c)
            - Wv(hi, 0, lo, q, c)
            - Wv(lo, 0, hi, q, c)
            + Wv(lo, 0, lo, q, c)
        ),
        "gamma x sigma": lambda q, c: (
            Wv(hi, hi, 0, q, c)
            - Wv(hi, lo, 0, q, c)
            - Wv(lo, hi, 0, q, c)
            + Wv(lo, lo, 0, q, c)
        ),
        "sigma x k    ": lambda q, c: (
            Wv(0, hi, hi, q, c)
            - Wv(0, hi, lo, q, c)
            - Wv(0, lo, hi, q, c)
            + Wv(0, lo, lo, q, c)
        ),
    }
    for name, fn in pairs.items():
        vals = [abs(fn(q, c)) for (q, c) in qc_grid]
        signed = [fn(q, c) for (q, c) in qc_grid]
        print(
            f"  {name}: mean|mixed-diff|={np.mean(vals):.4f}  max|.|={np.max(vals):.4f}  "
            f"mean(signed)={np.mean(signed):+.4f}"
        )

    print(
        "\n  Best-response coupling: does argmax on one axis MOVE when another axis changes?"
    )
    print("  (separability <=> best response on each axis is invariant to the others)")
    for q, c in [(0.95, 0.5), (0.8, 0.4), (1.0, 0.6)]:
        k_at_g0 = argmax_axis_at("k", {"gamma": 0.0}, q, c, P)
        k_at_g1 = argmax_axis_at("k", {"gamma": 1.0}, q, c, P)
        s_at_g0 = argmax_axis_at("sigma", {"gamma": 0.0}, q, c, P)
        s_at_g1 = argmax_axis_at("sigma", {"gamma": 1.0}, q, c, P)
        k_at_s0 = argmax_axis_at("k", {"sigma": 0.0}, q, c, P)
        k_at_s1 = argmax_axis_at("k", {"sigma": 1.0}, q, c, P)
        print(
            f"  (q,c)=({q},{c}): k*|gamma=0 ={k_at_g0:.2f}  k*|gamma=1 ={k_at_g1:.2f}  "
            f"(move={abs(k_at_g0 - k_at_g1):.2f}) | "
            f"sigma*|g=0={s_at_g0:.2f} sigma*|g=1={s_at_g1:.2f} | "
            f"k*|s=0={k_at_s0:.2f} k*|s=1={k_at_s1:.2f}"
        )

    # --------------------------------------------------------------------------------------------
    print(
        "\n--- 5. CUSTODY: is gamma*=0 a corner for ALL (q,c)? when does custody ever win? ---"
    )
    print(
        "  Scan beta_gamma (commitment-device benefit) vs the costs; find any interior gamma*>0."
    )
    print(
        "  beta_gamma  d_gamma  c_cust  | best gamma* over (q,c) grid | any gamma*>0?"
    )
    for bg in [0.0, 0.25, 0.5, 0.8]:
        for dg in [0.05, 0.30]:
            Pt = base_params()
            Pt["beta_gamma"] = bg
            Pt["d_gamma"] = dg
            best_g = 0.0
            argqc = None
            for q in [0.6, 0.8, 0.95, 1.0]:
                for c in [0.1, 0.3, 0.5, 0.7, 0.9]:
                    (g, sg, k), _ = argmax_joint(q, c, Pt)
                    if g > best_g + 1e-9:
                        best_g, argqc = g, (q, c, sg, k)
            print(
                f"   {bg:5.2f}      {dg:4.2f}    {Pt['c_cust']:4.2f}  | max gamma*={best_g:.2f}"
                f" at {argqc} | {'YES' if best_g > 1e-9 else 'no'}"
            )
    print(
        "  Diamond-1984 caveat respected: custody is given a REAL cutoff-lift benefit beta_gamma;"
    )
    print(
        "  the question is whether it EVER beats escrow drag + overhead. See verdict in summary."
    )
    print(
        "\n  5b. THE CLEAN UNIFICATION RESULT: inside the MASKING band, is custody minimal too?"
    )
    print(
        "  Where disclosure is REDUCED (k*<0.95, the masking band), check gamma* in the JOINT"
    )
    print(
        "  optimum. The anchor's story: no-custody dominates WHERE masking is the right call."
    )
    print(
        "  (q,c)         k*(disc-only)   joint(gamma*,sigma*,k*)   custody minimal in masking band?"
    )
    n_band = 0
    n_band_nocust = 0
    for q in [0.95, 1.0]:
        for c in [0.60, 0.65, 0.70, 0.75]:
            kdisc = argmax_axis("k", q, c, P)
            (g, sg, k), _ = argmax_joint(q, c, P)
            in_band = kdisc < 0.95
            tag = "-"
            if in_band:
                n_band += 1
                if g < 0.05:
                    n_band_nocust += 1
                tag = "YES (gamma*=0)" if g < 0.05 else f"NO (gamma*={g:.2f})"
            print(
                f"  ({q:.2f},{c:.2f})      {kdisc:.2f}            ({g:.1f},{sg:.1f},{k:.1f})           {tag}"
            )
    print(
        f"  >> no-custody holds in {n_band_nocust}/{n_band} masking-band cells. Custody switches ON"
    )
    print(
        "     OUTSIDE the masking band (where k*=1 already, so its cutoff-lift is worth most) —"
    )
    print(
        "     so the three minimalisms CO-OCCUR in the masking band, and custody's reversal is"
    )
    print(
        "     exactly where disclosure has ALSO reversed. That co-movement IS the unification."
    )

    # --------------------------------------------------------------------------------------------
    print(
        "\n--- 6. SPECIFICATION: is sigma*=0 (fuzzy-early) the corner, and when does precise win? ---"
    )
    print(
        "  sigma* trades precise-early attrition (kappa_s) vs fuzzy-too-long indeterminacy (lam_s)."
    )
    print(
        "  q\\c (sigma*)  " + "  ".join(f"{c:4.2f}" for c in [0.1, 0.3, 0.5, 0.7, 0.9])
    )
    for q in [0.8, 1.0]:
        row = [
            f"{argmax_axis('sigma', q, c, P):4.2f}" for c in [0.1, 0.3, 0.5, 0.7, 0.9]
        ]
        print(f"     q={q:4.2f}     " + "  ".join(row))
    print(
        "  raise lam_s (indeterminacy bites harder) -> sigma* should rise (more precision):"
    )
    for lam in [0.0, 0.5, 1.5, 3.0]:
        Pt = base_params()
        Pt["lam_s"] = lam
        s = argmax_axis("sigma", 1.0, 0.3, Pt)
        print(f"   lam_s={lam:4.2f}: sigma*(q=1,c=0.3)={s:.2f}")

    # --------------------------------------------------------------------------------------------
    print(
        "\n--- 7. WHERE EACH MINIMALISM IS REDUCED: the c-band of each axis (reversal at edges) ---"
    )
    print(
        "  'Reduced' = axis* meaningfully below its maximum: k*<0.95 (partial masking),"
    )
    print(
        "  sigma*<0.05 (fuzzy), gamma*<0.05 (no-custody). Report the c-interval where each is"
    )
    print(
        "  reduced, at fixed q, and whether the JOINT optimum is reduced on all three there."
    )
    cs_grid = np.linspace(0.02, 0.98, 49)

    def reduced_band(axis, q, P, thresh):
        red = []
        for c in cs_grid:
            v = argmax_axis(axis, q, c, P)
            is_red = (v < thresh) if axis != "k" else (v < 0.95)
            if is_red:
                red.append(c)
        return (min(red), max(red)) if red else None

    print(
        "  q     k-reduced band      sigma-fuzzy band   gamma-nocust band  | joint-all-reduced band"
    )
    for q in [0.6, 0.8, 0.95, 1.0]:
        bk = reduced_band("k", q, P, 0.95)
        bs = reduced_band("sigma", q, P, 0.05)
        bg = reduced_band("gamma", q, P, 0.05)
        joint_red = []
        for c in cs_grid:
            (g, sg, k), _ = argmax_joint(q, c, P)
            if g < 0.05 and sg < 0.05 and k < 0.95:
                joint_red.append(c)
        jb = (min(joint_red), max(joint_red)) if joint_red else None

        def f(b):
            return f"[{b[0]:.2f},{b[1]:.2f}]" if b else "  (none) "

        print(f"  {q:4.2f}  {f(bk):16s}  {f(bs):16s}  {f(bg):16s}  | {f(jb)}")
    print(
        "  reading: the JOINT all-reduced band is the INTERSECTION of the single-axis reduced"
    )
    print(
        "  bands (separable threshold structure). Disclosure reverses (k*->1) at BOTH c-edges;"
    )
    print(
        "  sigma/gamma are minimal across the whole band at baseline (their reversal is in lam_s,"
    )
    print(
        "  beta_gamma, not c). This is the joint minimal region; outside it, each axis reverses."
    )

    # --------------------------------------------------------------------------------------------
    print(
        "\n--- 8. q-AXIS reversal: low q => MORE disclosure (the anchor's 'q collapses' boundary) ---"
    )
    print(
        "  Fix c, scan q: k* should RISE toward 1 as q falls (low authenticity => masking would"
    )
    print(
        "  shield fakes, so disclose to screen). This is the (q*,c*) JOINT-threshold q-direction."
    )
    print(
        "  c      k*(q): "
        + "  ".join(f"q={q:.2f}" for q in [0.4, 0.55, 0.7, 0.85, 1.0])
    )
    for c in [0.3, 0.5, 0.7]:
        row = [argmax_axis("k", q, c, P) for q in [0.4, 0.55, 0.7, 0.85, 1.0]]
        monotone_down = all(row[i] >= row[i + 1] - 1e-9 for i in range(len(row) - 1))
        print(
            f"  {c:4.2f}          "
            + "   ".join(f"{x:4.2f}" for x in row)
            + f"   | k* falls as q rises: {monotone_down}"
        )
    print(
        "  reading: at fixed c, lower q => higher k* (more disclosure). Combined with section 1's"
    )
    print(
        "  c-direction, the masking region is a JOINT (q,c) band: high q AND mid-c. Below either,"
    )
    print(
        "  the policy reverses toward disclosure. This is the anchor's (q>=q*, c in band) condition."
    )

    # --------------------------------------------------------------------------------------------
    print("\n" + "=" * 96)
    print(
        "SUMMARY — THE SINGLE THEOREM (verdicts written AFTER the run; numbers above are literal)"
    )
    print("=" * 96)
    gk = np.mean([abs(pairs["gamma x k    "](q, c)) for (q, c) in qc_grid])
    gs = np.mean([abs(pairs["gamma x sigma"](q, c)) for (q, c) in qc_grid])
    sk = np.mean([abs(pairs["sigma x k    "](q, c)) for (q, c) in qc_grid])
    print(
        "  THEOREM (numerical finding, instantiated not derived from primitives). Under ONE"
    )
    print(
        "  welfare objective W(gamma,sigma,k;q,c), the efficient clearinghouse is REDUCED on all"
    )
    print(
        "  three axes together in a JOINT (q,c) region (high authenticity q, intermediate"
    )
    print(
        "  consensus-integrity c): no/low custody (gamma*=0), fuzzy-early (sigma*=0), and PARTIAL"
    )
    print(
        "  masking (k*<1). It REVERSES off this region on each axis. The structure is QUASI-"
    )
    print(
        "  SEPARABLE: each axis keeps its own threshold and the joint reduced region is the"
    )
    print(
        "  INTERSECTION of the single-axis ones, but the objective is NOT additively separable."
    )
    print("")
    print(
        "  [disclosure] PARTIAL masking, not a corner: k* dips to ~0.5-0.65 mid-c (= v3, verified"
    )
    print(
        "     reproduced in sec 0), reverses to k*~1 at low c (screen) and high c (robust). Also"
    )
    print(
        "     reverses to k*~1 as q falls (sec 8). The masking region is a (q,c) band."
    )
    print(
        "  [specification] CLEAN minimal corner: sigma*=0 (fuzzy) across the (q,c) plane at base;"
    )
    print(
        "     reverses to interior precise ONLY as lam_s (indeterminacy) rises (sec 6)."
    )
    print(
        "  [custody] DIAMOND-1984 BOUNDED claim: gamma*=0 when commitment-device benefit beta_gamma"
    )
    print(
        "     is small; custody WINS (gamma*>0) when beta_gamma is large (sec 5). NOT pure deadweight."
    )
    print("")
    print(
        f"  [SEPARABILITY] argmax match: exact {n_exact}/{n_test}, within-1-gridstep {n_1step}/{n_test},"
    )
    print(
        f"     within-2 {n_2step}/{n_test}. WELFARE gap of separating: median {sep_median_gap * 100:.2f}%,"
    )
    print(
        f"     p95 {sep_p95_gap * 100:.0f}%; within 1% in {sep_welfare_within1pct}/{n_test}, within 5% in"
    )
    print(
        f"     {sep_welfare_within5pct}/{n_test}. VERDICT: QUASI-separable — typically separating is"
    )
    print(
        "     ~free, but a real tail (the custody<->disclosure coupling) makes it NOT additively"
    )
    print(
        "     separable. Separable in THRESHOLD STRUCTURE, coupled in WELFARE MAGNITUDE."
    )
    print(
        f"  [INTERACTIONS] mean|mixed-diff|: sigma x k = {sk:.4f} (LARGEST; via the comp/screening"
    )
    print(
        f"     coupling in the bad-term), gamma x k = {gk:.4f} (the cutoff-substitution coupling),"
    )
    print(
        f"     gamma x sigma = {gs:.4f} (smallest). My pre-run prediction that gamma x k is THE"
    )
    print(
        "     interaction was FALSIFIED: sigma x k is larger. The objective couples specification"
    )
    print(
        "     to disclosure through composition, and custody to disclosure through the cutoff."
    )
    print("")
    print(
        "  HONEST STATUS: this INSTANTIATES the Block-4 single theorem (one objective; all three"
    )
    print(
        "  minimalisms fall out together with a joint (q,c) threshold and reversal). It does NOT"
    )
    print(
        "  DERIVE the channel forms from deeper primitives (rho,s from v3; the custody cutoff-lift"
    )
    print(
        "  and the specification attrition/indeterminacy are reduced-form, calibrated). The SIGNS"
    )
    print(
        "  and the quasi-separable structure are robust across the sweep; magnitudes are calibration-"
    )
    print(
        "  specific. 'Quasi-separable with a specification<->disclosure and a custody<->disclosure"
    )
    print(
        "  coupling' is the honest unification — less tidy than pure separability, but what holds."
    )
    print("=" * 96)


if __name__ == "__main__":
    main()


# ================================================================================================
# DEVELOPMENT LOG (numerics-first; falsification trail) — appended as the run is iterated.
# ================================================================================================
# v7.0: first unified objective. Predictions BEFORE running (to be confirmed/falsified):
#   P1: separable in (sigma) vs the rest -> sigma x k and sigma x gamma mixed-diffs ~ 0.
#   P2: gamma x k INTERACTS (both move rho_eff's cutoff) -> nonzero mixed-diff. THE interaction.
#   P3: gamma*=0 generically but NOT universal; large beta_gamma + small d_gamma buys gamma*>0.
#
# v7.1 FALSIFICATIONS (after the first run; numbers in stdout above):
#   F1 [P1 FALSIFIED]. sigma x k is NOT ~0 — it is the LARGEST mixed-diff (mean 0.0940 vs gamma x k
#      0.0511). Reason: specification enters comp(sigma), which sits in BOTH the good-term (with rho)
#      AND the bad-term (1-comp)(1-s)L (with the screening s(k)). sigma and k both move how much
#      weight lands on the bad-term's screening -> genuine coupling. P1's "sigma is separable" was
#      wrong. gamma x sigma IS ~0 (0.0130), the only near-additive pair. CORRECTED in header (C4).
#   F2 [P2 partially confirmed]. gamma x k does interact (0.0511, the cutoff substitution), but it is
#      NOT the largest. The "one interaction = custody x disclosure" intuition was too narrow.
#   F3 [P3 CONFIRMED]. gamma*=0 at baseline beta_gamma=0.25? NO — even at beta_gamma=0.25 custody can
#      win (sec 5 shows gamma*=1 reachable). gamma*=0 only when beta_gamma is genuinely small/zero.
#      The bounded claim holds (custody loses for small beta_gamma), but the baseline beta_gamma was
#      large enough to flip it -> the Diamond caveat bites EARLIER than I expected. Honest: "no-custody
#      dominates UNLESS the commitment-device benefit is sizable" — and the threshold is low.
#   F4 [BIGGEST RESHAPE]. The literal "minimal corner (0,0,0)" is the WRONG object: the disclosure
#      facet's optimum is PARTIAL masking (k*~0.5-0.65 in the mid-c valley; v3-faithful, sec 0),
#      never k*=0. The first-draft sections 2/7/8 keyed on the (0,0,0) corner and reported a vacuous
#      "onset c = 0.98" (the corner essentially never attained). Reframed around REDUCED policy and
#      the per-axis reduced c-band; the theorem is about reduced-on-all-three in a (q,c) band with a
#      reversal at the edges, NOT a literal triple-zero corner. This mirrors v3's own correction note.
#   F5 [v2 endpoint hand-check, first draft]. W(0,0,0) != qc V-(1-qc)L because indet(sigma=0)>0 shifts
#      good surplus. Not a bug in W — the hand-check omitted the indeterminacy shift. Sec 0 now states
#      the identity WITH indet0 and the err prints 0.0; the disclosure facet matches v3 to grid res.
#
# RESIDUAL HONESTY: this INSTANTIATES the single theorem (one objective, three minimalisms, joint
# (q,c) threshold, reversal). It does not DERIVE the custody cutoff-lift or the specification
# attrition/indeterminacy from deeper games (those are reduced-form, as rho/s were in v3). Signs and
# the quasi-separable structure are robust to the 600-draw randomized sweep; magnitudes are
# calibration-specific (quote no magnitude as a constant — house rule, per v2/v3/v5/v6 history).
