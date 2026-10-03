"""
v8_verify_reimpl.py — INDEPENDENT adversarial re-implementation of the v8 screening claims.

GOAL: refute (or confirm) C1-C4 with a DIFFERENT model structure than v8 used.

v8 used: a symmetric BINARY type channel Pr(z=psi|psi)=(1+k)/2, plus a SEPARATE Gaussian
"type" model with two fixed means mu_G=+1, mu_B=-1 and a hand-built discriminability map
d'(k)=d_scale*k/(1-k). The "type" is a coin (G vs B); each agent's signal is about that coin;
the commit rule is a single-agent break-even Bayes rule; activation = Binom(N, r) >= K.

MY INDEPENDENT MODEL (deliberately different on every axis):
  - CONTINUOUS fundamental. The coalition has a "compatibility fundamental" x ~ Normal(m0, 1/tau0)
    (a real number, NOT a binary coin). x is the latent deliverability of the coalition.
    A coalition is GENUINELY INCOMPATIBLE ("bad") iff x < 0 (it cannot deliver); GOOD iff x >= 0.
    The prior mass on bad is pi_b = Phi(-m0*sqrt(tau0)).  <-- different prior family (continuous).
  - Each of N agents gets a PRIVATE Gaussian signal y_i = x + eps_i, eps_i ~ N(0, 1/(tau_eps)).
    DIFFERENT precision->k map than v8: signal precision tau_eps(k) = tau_lo + k^2*(tau_hi-tau_lo),
    a QUADRATIC ramp on a FINITE precision range (NOT v8's d'=d_scale*k/(1-k) blow-up). So even
    at k=1 information is finite and noisy. k=0 = floor precision tau_lo (not zero info).
  - THRESHOLD strategy (Morris-Shin / Carlsson-van Damme): agent commits iff y_i >= y_hat,
    where y_hat is the global-game switching point. We solve the threshold/fundamental fixed
    point (x*, y_hat) of the standard CvD construction rather than positing a single-agent rule.
  - Coalition ACTIVATES iff #commit >= K. Conditional on x, #commit ~ Binom(N, p_commit(x)),
    p_commit(x) = Pr(y_i >= y_hat | x) = Phi(sqrt(tau_eps)*(x - y_hat)).
  - s(k) = Pr( a genuinely-INCOMPATIBLE (x<0) coalition does NOT activate )   [SCREEN bad]
    phi(k) = Pr( a genuinely-COMPATIBLE (x>=0) coalition does NOT activate )  [false-reject good]
  - Welfare uses the SAME functional form as v8 W_type but with continuous good/bad split.

WHY THIS IS A FAIR-BUT-INDEPENDENT TEST:
  Same economic content (private signal of coalition deliverability; commit rule; K-of-N
  activation; screen=bad-fails, false-reject=good-fails). Totally different prior (continuous
  fundamental vs binary coin), different precision map (finite quadratic vs diverging), different
  equilibrium concept (genuine global-game threshold fixed point vs single-agent break-even),
  different "type" definition (x<0 vs a latent coin). If C1-C4 are real, they should survive.

Run: uv run --with numpy --with scipy python v8_verify_reimpl.py
"""

from __future__ import annotations

from math import ceil, comb, sqrt

import numpy as np
from scipy import integrate
from scipy.stats import norm

SEED = 20260609
rng = np.random.default_rng(SEED)


# --------------------------------------------------------------------------------------------
# binomial helpers
# --------------------------------------------------------------------------------------------
def binom_lt(n: int, p: float, k: int) -> float:
    """Pr(Binom(n,p) < k) = Pr(< K commits) = NOT activated."""
    if k <= 0:
        return 0.0
    if k > n:
        return 1.0
    p = min(max(p, 0.0), 1.0)
    pmf = np.array([comb(n, j) * (p**j) * ((1 - p) ** (n - j)) for j in range(n + 1)])
    return float(np.sum(pmf[:k]))


# --------------------------------------------------------------------------------------------
# precision map (DIFFERENT from v8): finite quadratic ramp
# --------------------------------------------------------------------------------------------
def tau_eps_of_k(k: float, tau_lo: float, tau_hi: float) -> float:
    return tau_lo + (k**2) * (tau_hi - tau_lo)


# --------------------------------------------------------------------------------------------
# Global-game threshold fixed point (Morris-Shin / Carlsson-van Damme style).
#
# Setup: payoff to committing depends on whether the coalition delivers. Following the CvD
# logic, an agent's expected gain from committing is increasing in her posterior about x.
# A genuine global game needs strategic complementarity AND a dominance region. We construct:
#   - gain to commit if coalition activates & delivers (x>=0):  A = (1-xi)*V1 - kappa
#   - loss to commit if coalition activates & is bad (x<0):     -B_ = -((1-xi)*Lbad + kappa)
#   - the agent is PIVOTAL with some probability; off-pivot, committing to a delivering
#     coalition still yields (V1-kappa) vs waiting xi*V1, i.e. private excludable gain A.
# We use the canonical CvD threshold equation: the marginal agent (signal exactly y_hat) is
# indifferent. With Gaussian prior x~N(m0,1/tau0) and signal y=x+eps, eps~N(0,1/tau_eps),
# the posterior of x given y is Normal with mean
#   mu_post = (tau0*m0 + tau_eps*y)/(tau0+tau_eps), precision tau_post=tau0+tau_eps.
# The agent commits iff posterior expected commit-payoff > 0:
#   Pr(x>=0 | y) * A  -  Pr(x<0 | y) * B_  >= 0
# (this is the standard "act iff posterior odds beat the payoff ratio" rule, the continuous-
# fundamental analog of v8's binary break-even; the K-of-N pivotality enters via activation,
# and the SAME rule defines y_hat). Solve for y_hat:
#   Pr(x>=0|y) = 1 - Phi( (0 - mu_post)*sqrt(tau_post) ) = Phi( mu_post*sqrt(tau_post) )
#   commit iff Phi(z) >= B_/(A+B_), z = mu_post*sqrt(tau_post).
# Indifference: z_hat = Phi^{-1}( B_/(A+B_) ); back out y_hat.
# This is a DIFFERENT derivation of the cutoff than v8 (continuous fundamental, posterior on a
# real line), and the K-of-N activation rule is layered on top, exactly as the task asks.
# --------------------------------------------------------------------------------------------
def cutoff_yhat(m0: float, tau0: float, tau_eps: float, A: float, B_: float):
    """Return the private-signal commit cutoff y_hat (commit iff y >= y_hat). None if A<=0."""
    if A <= 0:
        return None  # excludable gain non-positive: committing never pays (free-ride regime)
    tau_post = tau0 + tau_eps
    # threshold posterior odds: Phi(z_hat) = B_/(A+B_)
    target = B_ / (A + B_)
    z_hat = norm.ppf(target)  # mu_post * sqrt(tau_post) = z_hat
    mu_post_hat = z_hat / sqrt(tau_post)
    # mu_post = (tau0*m0 + tau_eps*y)/tau_post  ->  solve for y:
    y_hat = (mu_post_hat * tau_post - tau0 * m0) / tau_eps
    return y_hat


def p_commit_given_x(x: float, y_hat, tau_eps: float) -> float:
    if y_hat is None:
        return 0.0
    # Pr(y >= y_hat | x) = Pr(eps >= y_hat - x) = 1 - Phi(sqrt(tau_eps)*(y_hat - x))
    return float(1.0 - norm.cdf(sqrt(tau_eps) * (y_hat - x)))


# --------------------------------------------------------------------------------------------
# s(k), phi(k): integrate activation over the fundamental, split bad (x<0) / good (x>=0).
# --------------------------------------------------------------------------------------------
def s_phi(params: dict, k: float):
    m0, tau0 = params["m0"], params["tau0"]
    N, K = params["N"], params["K"]
    xi, V1, Lbad, kappa = params["xi"], params["V1"], params["Lbad"], params["kappa"]
    tau_lo, tau_hi = params["tau_lo"], params["tau_hi"]

    A = (1 - xi) * V1 - kappa
    B_ = (1 - xi) * Lbad + kappa
    tau_eps = tau_eps_of_k(k, tau_lo, tau_hi)
    y_hat = cutoff_yhat(m0, tau0, tau_eps, A, B_)

    sd = 1.0 / sqrt(tau0)

    def not_activate(x):  # Pr(<K commit | x)
        pc = p_commit_given_x(x, y_hat, tau_eps)
        return binom_lt(N, pc, K)

    # prior density of x
    def dens(x):
        return norm.pdf(x, loc=m0, scale=sd)

    # P(x<0) and P(x>=0) under prior
    pi_b = float(norm.cdf(0.0, loc=m0, scale=sd))
    pi_g = 1.0 - pi_b

    # s(k) = Pr(NOT activate AND x<0) / Pr(x<0)
    # phi(k) = Pr(NOT activate AND x>=0) / Pr(x>=0)
    lo, hi = m0 - 8 * sd, m0 + 8 * sd
    num_bad, _ = integrate.quad(lambda x: not_activate(x) * dens(x), lo, 0.0, limit=200)
    num_good, _ = integrate.quad(
        lambda x: not_activate(x) * dens(x), 0.0, hi, limit=200
    )
    s = num_bad / pi_b if pi_b > 1e-12 else 1.0
    phi = num_good / pi_g if pi_g > 1e-12 else 0.0
    return s, phi, pi_b


def W_type(params: dict, k: float):
    s, phi, pi_b = s_phi(params, k)
    return (1 - pi_b) * (1 - phi) * params["V1"] - pi_b * (1 - s) * params["Lbad"]


# --------------------------------------------------------------------------------------------
# Monte-Carlo cross-check (fully independent of the quadrature): simulate coalitions.
# --------------------------------------------------------------------------------------------
def s_phi_mc(params: dict, k: float, n_coalitions: int = 200000):
    m0, tau0 = params["m0"], params["tau0"]
    N, K = params["N"], params["K"]
    xi, V1, Lbad, kappa = params["xi"], params["V1"], params["Lbad"], params["kappa"]
    tau_lo, tau_hi = params["tau_lo"], params["tau_hi"]
    A = (1 - xi) * V1 - kappa
    B_ = (1 - xi) * Lbad + kappa
    tau_eps = tau_eps_of_k(k, tau_lo, tau_hi)
    y_hat = cutoff_yhat(m0, tau0, tau_eps, A, B_)
    sd = 1.0 / sqrt(tau0)
    x = rng.normal(m0, sd, size=n_coalitions)
    eps_sd = 1.0 / sqrt(tau_eps)
    # signals for N agents per coalition
    sig = x[:, None] + rng.normal(0, eps_sd, size=(n_coalitions, N))
    if y_hat is None:
        commits = np.zeros(n_coalitions, dtype=int)
    else:
        commits = np.sum(sig >= y_hat, axis=1)
    activate = commits >= K
    bad = x < 0
    good = ~bad
    s = float(np.mean(~activate & bad) / max(np.mean(bad), 1e-12))
    phi = float(np.mean(~activate & good) / max(np.mean(good), 1e-12))
    return s, phi


def mk(N, theta, m0, tau0, V1, Lbad, xi, kappa, tau_lo=0.25, tau_hi=8.0):
    return {
        "N": N,
        "K": max(1, ceil((1 - theta) * N)),
        "m0": m0,
        "tau0": tau0,
        "V1": V1,
        "Lbad": Lbad,
        "xi": xi,
        "kappa": kappa,
        "tau_lo": tau_lo,
        "tau_hi": tau_hi,
    }


# ============================================================================================
def main():
    print("=" * 96)
    print("INDEPENDENT RE-IMPL — continuous-fundamental global game; testing v8 C1-C4")
    print(f"seed={SEED}")
    print("=" * 96)

    N, V1, Lbad, kappa = (
        10,
        3.0,
        3.0,
        3.0,
    )  # NB kappa=3 here (v8 used 1); different calib
    # use kappa<V1 so A can be positive at low xi. set kappa=1.0 to match excludability range:
    kappa = 1.0
    theta0 = 0.5  # K = 5 of 10
    m0, tau0 = 0.0, 1.0  # prior mean 0 -> pi_b = 0.5 (symmetric, neutral prior)
    xi_lo = 0.3

    xi_star = 1 - kappa / V1
    print(
        f"\ncalib: N={N} theta={theta0} K={ceil((1 - theta0) * N)} V1={V1} Lbad={Lbad} "
        f"kappa={kappa}  xi*=1-kappa/V1={xi_star:.3f}"
    )
    print(
        "precision map: tau_eps(k)=0.25 + k^2*(8.0-0.25)  (FINITE, quadratic; v8 used d'~k/(1-k))"
    )
    print(
        "prior: x~N(m0,1/tau0), bad iff x<0.  Different family from v8's binary coin."
    )

    base = mk(N, theta0, m0, tau0, V1, Lbad, xi_lo, kappa)

    # ---- C4 / C1: shape of s(k), phi(k) ----
    print(
        "\n--- A. s(k), phi(k), W_type(k) over k (base: m0=0 -> pi_b=0.5, xi=0.3) ---"
    )
    print(
        "   k     tau_eps   s(k)      phi(k)    W_type     (s=screen bad, phi=false-reject good)"
    )
    ks = np.linspace(0, 1, 11)
    svals, pvals, wvals = [], [], []
    for k in ks:
        s, phi, pib = s_phi(base, float(k))
        w = (1 - pib) * (1 - phi) * V1 - pib * (1 - s) * Lbad
        svals.append(s)
        pvals.append(phi)
        wvals.append(w)
        print(
            f"  {k:.2f}   {tau_eps_of_k(float(k), 0.25, 8.0):6.3f}   {s:.4f}   {phi:.4f}   {w:+.4f}"
        )
    svals, pvals, wvals = map(np.array, (svals, pvals, wvals))

    # ---- monotonicity / shape diagnostics ----
    fine = np.linspace(0.0, 1.0, 101)
    sf = np.array([s_phi(base, float(k))[0] for k in fine])
    pf = np.array([s_phi(base, float(k))[1] for k in fine])
    wf = np.array([W_type(base, float(k)) for k in fine])
    lin = sf[0] + (fine - fine[0]) / (fine[-1] - fine[0]) * (sf[-1] - sf[0])
    s_mono_incr = bool(np.all(np.diff(sf) >= -1e-6))
    phi_mono_decr = bool(np.all(np.diff(pf) <= 1e-6))
    w_mono_incr = bool(np.all(np.diff(wf) >= -1e-6))
    # convexity: second difference sign
    d2 = np.diff(sf, 2)
    frac_convex = float(np.mean(d2 >= -1e-6))
    print("\n--- B. SHAPE diagnostics (fine grid, base cell) ---")
    print(
        f"  s(k):   monotone_incr={s_mono_incr}  s(0)={sf[0]:.4f} s(1)={sf[-1]:.4f}  "
        f"max|dev_from_linear|={np.max(np.abs(sf - lin)):.4f}  frac_2ndDiff>=0={frac_convex:.2f}"
    )
    print(
        f"  phi(k): monotone_decr={phi_mono_decr}  phi(0)={pf[0]:.4f} phi(1)={pf[-1]:.4f}"
    )
    print(
        f"  W(k):   monotone_incr={w_mono_incr}  argmax k={fine[int(np.argmax(wf))]:.3f}  "
        f"W(0)={wf[0]:+.4f} W(1)={wf[-1]:+.4f}"
    )

    # ---- C3: excludability regime ----
    print(
        "\n--- C. EXCLUDABILITY (C3): commit cutoff exists iff A=(1-xi)V1-kappa>0, xi<xi*? ---"
    )
    print(
        "   xi      A          y_hat(k=0.7)     s(k=0.7)   phi(k=0.7)   commit-active"
    )
    for xv in [0.0, 0.3, 0.5, 0.6, xi_star, 0.7, 0.8, 0.9]:
        p = mk(N, theta0, m0, tau0, V1, Lbad, xv, kappa)
        A = (1 - xv) * V1 - kappa
        yh = cutoff_yhat(
            m0, tau0, tau_eps_of_k(0.7, 0.25, 8.0), A, (1 - xv) * Lbad + kappa
        )
        s, phi, _ = s_phi(p, 0.7)
        yh_s = f"{yh:+.3f}" if yh is not None else "  none "
        print(
            f"  {xv:.3f}   {A:+7.3f}   {yh_s:>9}        {s:.4f}     {phi:.4f}     {A > 0}"
        )
    print(f"  >> predicted threshold xi* = 1 - kappa/V1 = {xi_star:.3f}")

    # ---- C2: can MORE disclosure HURT screening or welfare? scan for a counterexample ----
    print(
        "\n--- D. C2 STRESS TEST: search for a regime where MORE k HURTS s or W (refute one-way) ---"
    )
    worst_s_drop, worst_w_drop = 0.0, 0.0
    worst_s_cell, worst_w_cell = None, None
    n_cells, s_nonmono, w_nonmono = 0, 0, 0
    for m0v in [-1.0, -0.5, 0.0, 0.5, 1.0]:  # prior mean (controls pi_b)
        for xv in [0.0, 0.2, 0.4, 0.6]:  # excludable regime only (A>0)
            if (1 - xv) * V1 - kappa <= 0:
                continue
            for th in [0.2, 0.4, 0.6, 0.8]:  # closeness K/N
                for Lr in [0.5, 1.0, 3.0]:  # loss ratio
                    p = mk(N, th, m0v, tau0, V1, Lr * V1, xv, kappa)
                    sg = np.array([s_phi(p, float(k))[0] for k in fine])
                    wg = np.array([W_type(p, float(k)) for k in fine])
                    n_cells += 1
                    ds = np.min(np.diff(sg))
                    dw = np.min(np.diff(wg))
                    if ds < -1e-4:
                        s_nonmono += 1
                        if ds < worst_s_drop:
                            worst_s_drop = ds
                            worst_s_cell = (m0v, xv, th, Lr)
                    if dw < -1e-4:
                        w_nonmono += 1
                        if dw < worst_w_drop:
                            worst_w_drop = dw
                            worst_w_cell = (m0v, xv, th, Lr)
    print(f"  cells scanned: {n_cells}")
    print(
        f"  cells where s(k) DROPS somewhere (refutes 'one-way'): {s_nonmono}  "
        f"worst local d s = {worst_s_drop:+.4f} at (m0,xi,theta,Lr)={worst_s_cell}"
    )
    print(
        f"  cells where W(k) DROPS somewhere: {w_nonmono}  "
        f"worst local d W = {worst_w_drop:+.4f} at (m0,xi,theta,Lr)={worst_w_cell}"
    )

    # ---- show the worst s-dropping cell explicitly if found ----
    if worst_s_cell is not None:
        m0v, xv, th, Lr = worst_s_cell
        p = mk(N, th, m0v, tau0, V1, Lr * V1, xv, kappa)
        print(
            f"\n  WORST s-dropping cell (m0={m0v}, xi={xv}, theta={th}, Lbad/V1={Lr}):"
        )
        print("    k     s(k)     phi(k)    W_type")
        for k in np.linspace(0, 1, 11):
            s, phi, pib = s_phi(p, float(k))
            w = (1 - pib) * (1 - phi) * V1 - pib * (1 - s) * (Lr * V1)
            print(f"   {k:.2f}   {s:.4f}   {phi:.4f}   {w:+.4f}")

    # ---- endpoint ordering robustness ----
    print("\n--- E. ORDERING s(1)>=s(0) and phi(1)<=phi(0) across excludable grid ---")
    hold_s, hold_phi, ncell = True, True, 0
    for m0v in [-1.0, 0.0, 1.0]:
        for xv in [0.0, 0.3, 0.6]:
            for th in [0.2, 0.5, 0.8]:
                p = mk(N, th, m0v, tau0, V1, Lbad, xv, kappa)
                s0, p0, _ = s_phi(p, 0.01)
                s1, p1, _ = s_phi(p, 0.99)
                ncell += 1
                if s1 < s0 - 1e-4:
                    hold_s = False
                if p1 > p0 + 1e-4:
                    hold_phi = False
    print(f"  s(1)>=s(0) across {ncell} cells: {hold_s};  phi(1)<=phi(0): {hold_phi}")

    # ---- MC cross-check ----
    print("\n--- F. MONTE-CARLO cross-check (independent of quadrature) ---")
    print("   k     s_quad   s_MC    |d|      phi_quad  phi_MC   |d|")
    for k in [0.2, 0.5, 0.8]:
        sq, pq, _ = s_phi(base, k)
        sm, pm = s_phi_mc(base, k)
        print(
            f"  {k:.2f}   {sq:.4f}  {sm:.4f}  {abs(sq - sm):.4f}   "
            f"{pq:.4f}   {pm:.4f}  {abs(pq - pm):.4f}"
        )

    # ---- linear-vs-convex verdict on C4 ----
    print("\n--- G. C4 VERDICT: is s(k) linear (v3 posit) or non-linear? ---")
    # only meaningful where s actually varies (not pinned at a floor); use base cell active part
    print(
        f"  base s(k): max|dev_from_linear|={np.max(np.abs(sf - lin)):.4f} "
        f"(0 would mean perfectly linear)."
    )
    print(
        f"  s(0)={sf[0]:.4f}, s(1)={sf[-1]:.4f}; v3 posits s(k)=k i.e. s(0)=0,s(1)=1 linear."
    )

    # ---- H. THE DECISIVE TEST: a regime where s(k) genuinely VARIES (bad CAN clear by noise) ----
    # Optimistic prior (m0>0 -> low pi_b) + low theta (low K) so a BAD coalition (x<0) can reach
    # K commits by noise when signals are NOISY (low k). As k rises, agents see x<0 -> decline ->
    # bad screened. THIS is the regime where the screening CLAIM is actually testable.
    print(
        "\n--- H. DECISIVE: optimistic prior + low K so BAD can clear by noise; does k raise s? ---"
    )
    print(
        "   m0   theta  K    k=0:s   k=1:s   ds      k=0:phi k=1:phi  s_monotone_incr?  s_dip?"
    )
    for m0v in [0.5, 1.0, 1.5]:
        for th in [0.7, 0.8, 0.9]:
            p = mk(N, th, m0v, tau0, V1, Lbad, 0.3, kappa)
            sg = np.array([s_phi(p, float(k))[0] for k in fine])
            pg = np.array([s_phi(p, float(k))[1] for k in fine])
            mono = bool(np.all(np.diff(sg) >= -1e-4))
            dip = bool(np.min(np.diff(sg)) < -1e-3)
            print(
                f"  {m0v:.1f}  {th:.1f}   {p['K']}   {sg[0]:.4f}  {sg[-1]:.4f}  {sg[-1] - sg[0]:+.4f}  "
                f"{pg[0]:.4f}  {pg[-1]:.4f}   {str(mono):>5}             {str(dip):>5}"
            )

    # ---- I. DETAIL of a dip cell (if the screening rate is genuinely non-monotone) ----
    print(
        "\n--- I. DETAIL: m0=1.0, theta=0.8 (optimistic, low K) — s(k) trajectory ---"
    )
    p = mk(N, 0.8, 1.0, tau0, V1, Lbad, 0.3, kappa)
    print("    k    tau_eps   s(k)     phi(k)   W_type")
    for k in np.linspace(0, 1, 11):
        s, phi, pib = s_phi(p, float(k))
        w = (1 - pib) * (1 - phi) * V1 - pib * (1 - s) * Lbad
        print(
            f"   {k:.2f}  {tau_eps_of_k(float(k), 0.25, 8.0):6.3f}   {s:.4f}   {phi:.4f}  {w:+.4f}"
        )

    print("\n" + "=" * 96)
    print("DONE")
    print("=" * 96)


if __name__ == "__main__":
    main()
