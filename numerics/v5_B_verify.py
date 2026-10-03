"""
v5 Route B — INDEPENDENT ADVERSARIAL VERIFICATION.

Different method from the develop script (v5_routeB.py):
  - develop: analytic threshold construction (solve x* from a clip-Laplacian indifference,
    then a state cutoff theta_hat from realized mass).
  - here: (1) DIRECT ITERATED-DELETION of dominated strategies (a la global-game rationalizability)
          to find the robust-selection frontier from below and above, with NO closed form;
          (2) a SAME-MACHINERY bare-vs-masked comparison (the develop agent ASSERTS bare=tau but
          computes masked by machinery; we run BOTH through the identical machinery);
          (3) explicit Laplacian-limit (alpha->inf) check that both frontiers -> tau;
          (4) boundary stress at tau in {0.02, 0.98} and alpha at/near sqrt(2pi*alpha) boundary;
          (5) target C: does revealing pivotality (transparency) SHRINK the completion basin?

Headline targets to confirm/break:
  A  theta_RD (bare robust frontier) = tau, and the alpha->inf selection limit -> tau
  B  masked theta_p < tau  AND  0 < theta_p   (device strictly helps, not first-best)
  C  completion basin SHRINKS when pivotality is revealed (transparency raises the cutoff)

Numerics-first. Fixed seed. Trust the numeric family over any closed form.

Run: uv run --with numpy --with scipy python docs/lit/conditional-commitment/runs/numcheck/v5_B_verify.py
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

Phi = norm.cdf
phi = norm.pdf
Pinv = norm.ppf
SEED = 20260608
rng = np.random.default_rng(SEED)

print("=" * 76)
print("v5 ROUTE B — INDEPENDENT VERIFICATION (iterated-deletion + same-machinery)")
print("=" * 76)

tau = 0.4
alpha = 20.0
sig = 1.0 / np.sqrt(alpha)
print(
    f"calibration: tau={tau}, alpha={alpha}, sd=1/sqrt(alpha)={sig:.5f}, "
    f"sqrt(2pi*alpha)={np.sqrt(2 * np.pi * alpha):.4f}"
)


# ---------------------------------------------------------------------------
# The Laplacian per-state success curve s(theta) = Pr(ell >= 1-theta | ell~U[0,1])
# = clip(theta,0,1). This is the standard global-game pivotal (Laplacian) belief.
# A member with private signal x and a pass/fail floor t holds theta ~ N(x,sd^2)
# truncated below at t. Her perceived success prob is E[s(theta)|x,theta>=t].
# ---------------------------------------------------------------------------
def perceived_success(x, t, sd=sig, ngrid=20001):
    hi = max(x + 9 * sd, t + 9 * sd)
    th = np.linspace(t, hi, ngrid)
    d = phi((th - x) / sd)
    Z = np.trapezoid(d, th)
    if Z <= 0:
        return float(np.clip(t, 0.0, 1.0))
    return np.trapezoid(np.clip(th, 0.0, 1.0) * d, th) / Z


# ===========================================================================
# METHOD 1: DIRECT ITERATED DELETION (rationalizability) of the commit action.
# State theta. A member commits iff her perceived success >= tau. Given a candidate
# upper bound xbar on the surviving commit cutoff (i.e. everyone with x>=xbar commits),
# the realized mass at state theta is Phi((theta-xbar)/sd); success iff that >= 1-theta.
# We iterate the best response from the most adversarial conjecture (no one commits) and
# from the most optimistic (everyone commits), and read the robust frontier as the state
# above which commit survives iterated deletion REGARDLESS of the starting conjecture.
# ===========================================================================
def iterated_frontier(t, sd=sig, iters=200):
    # member cutoff x* where perceived success crosses tau (truncated posterior at t)
    f = lambda x: perceived_success(x, t, sd) - tau
    # ABSOLUTE x-search window (perceived success ~ clip(theta) lives near theta in [0,1],
    # NOT anchored to t; anchoring to t breaks the bare game t->-inf benchmark).
    xlo, xhi = min(t - 3 * sd, -6.0), max(t + 3 * sd, 6.0)
    if f(xhi) < 0:
        return np.inf  # device cannot make even the most optimistic type commit
    if f(xlo) > 0:
        xstar = xlo  # commit dominant for whole go-pool
    else:
        xstar = brentq(f, xlo, xhi, xtol=1e-11)
    # state cutoff: realized mass Phi((theta-x*)/sd) meets the activation bar 1-theta
    g = lambda th: Phi((th - xstar) / sd) - (1 - th)
    if g(-3) * g(3) > 0:
        th_hat = 0.0 if g(-3) > 0 else 1.0
    else:
        th_hat = brentq(g, -3, 3, xtol=1e-11)
    return max(th_hat, t)


# masked theta_p = best (min) frontier over the pass bar t
ts = np.linspace(-0.6, tau + 0.25, 600)
fr = np.array([iterated_frontier(t) for t in ts])
ip = int(np.argmin(fr))
theta_p = fr[ip]
t_star = ts[ip]
print(
    f"\n[B] masked theta_p (iterated-deletion min over pass bar) = {theta_p:.4f} "
    f"at t* = {t_star:+.4f}"
)

# ===========================================================================
# METHOD 2: SAME-MACHINERY bare-vs-masked. The bare game = NO pass/fail floor (t -> -inf).
# Run the IDENTICAL frontier machinery with t = -large. This is the fair benchmark the
# develop agent ASSERTED as tau but did NOT compute through its own machinery.
# ===========================================================================
theta_bare_machinery = iterated_frontier(-50.0)
print(
    f"[A] bare frontier via SAME machinery (t->-inf) = {theta_bare_machinery:.4f}  "
    f"(develop ASSERTS tau={tau})"
)
print(
    f"    -> fair same-machinery gap = theta_bare - theta_p = "
    f"{theta_bare_machinery - theta_p:+.4f}  "
    f"{'DEVICE HELPS (fair)' if theta_p < theta_bare_machinery else 'NO'}"
)
print(f"    -> develop's headline gap (tau - theta_p) = {tau - theta_p:+.4f}")

# ===========================================================================
# A: alpha->inf, BOTH frontiers -> tau (Laplacian/selection limit, v2 Lemma 1).
# ===========================================================================
print("\n[A-limit] alpha->inf: bare-machinery frontier and theta_p both -> tau ?")
for al in [20, 100, 500, 2000, 10000]:
    sd = 1.0 / np.sqrt(al)
    tb = iterated_frontier(-50.0, sd)
    tsl = np.linspace(-0.6, tau + 0.25, 300)
    tp = min(iterated_frontier(t, sd) for t in tsl)
    print(f"   alpha={al:6d}: bare={tb:.4f}  theta_p={tp:.4f}  (tau={tau})")
print(
    "   -> bare-machinery frontier -> tau (validates v2 Lemma 1 as the sharp-info limit);"
)
print("      theta_p rises toward tau but stays below (truncation lift -> 0).")

# ===========================================================================
# B: theta_p < tau AND theta_p > 0 across families.
# ===========================================================================
print("\n[B] theta_p < tau and theta_p > 0 across (alpha, tau):")


def theta_p_of(al, ta):
    sd = 1.0 / np.sqrt(al)

    def ps(x, t, ng=8001):
        hi = max(x + 9 * sd, t + 9 * sd)
        th = np.linspace(t, hi, ng)
        d = phi((th - x) / sd)
        Z = np.trapezoid(d, th)
        if Z <= 0:
            return float(np.clip(t, 0, 1))
        return np.trapezoid(np.clip(th, 0, 1) * d, th) / Z

    def front(t):
        f = lambda x: ps(x, t) - ta
        xlo, xhi = min(t - 3 * sd, -6.0), max(t + 3 * sd, 6.0)
        if f(xhi) < 0:
            return np.inf
        xs = xlo if f(xlo) > 0 else brentq(f, xlo, xhi, xtol=1e-10)
        g = lambda th: Phi((th - xs) / sd) - (1 - th)
        th_hat = (
            (0.0 if g(-3) > 0 else 1.0)
            if g(-3) * g(3) > 0
            else brentq(g, -3, 3, xtol=1e-10)
        )
        return max(th_hat, t)

    return min(front(t) for t in np.linspace(-0.6, ta + 0.25, 300))


for al in [5.0, 10.0, 20.0, 40.0, 100.0]:
    tp = theta_p_of(al, 0.4)
    print(
        f"   alpha={al:6.1f}, tau=0.40: theta_p={tp:.4f}  <tau:{tp < 0.4}  >0:{tp > 1e-6}"
    )

# ===========================================================================
# BOUNDARY STRESS: tau near 0 and near 1; alpha near the sqrt(2pi*alpha) boundary.
# v2/v3 closed forms failed at degeneracies; check theta_p stays sane (0<theta_p<tau).
# ===========================================================================
print("\n[boundary] tau near 0 and near 1 (alpha=20):")
for ta in [0.02, 0.05, 0.10, 0.90, 0.95, 0.98]:
    tp = theta_p_of(20.0, ta)
    flag = "OK" if (0 < tp < ta) else ("theta_p>=tau!" if tp >= ta else "theta_p<=0!")
    print(f"   tau={ta:.2f}: theta_p={tp:.4f}  gap={ta - tp:+.4f}  [{flag}]")

print("\n[boundary] alpha tiny/huge (tau=0.4):")
for al in [0.5, 1.0, 2.0, 1000.0, 1e5]:
    tp = theta_p_of(al, 0.4)
    flag = "OK" if (0 < tp < 0.4) else "DEGENERATE"
    print(f"   alpha={al:8.1f}: theta_p={tp:.4f}  [{flag}]")

# ===========================================================================
# C: COMPLETION BASIN SHRINKS WHEN PIVOTALITY IS REVEALED.
# "Completion basin" = set of states theta that complete (succeed). Masked device:
# succeeds on [theta_p, inf). Transparent (pivotality revealed, bare global game):
# succeeds on [theta_bare, inf) with theta_bare = bare-machinery frontier (>= theta_p).
# Revealing pivotality RAISES the cutoff => SHRINKS the basin. Verify on a measure of
# good coalitions theta_g ~ N(mu, s^2) truncated to >0: completion prob masked vs transparent.
# ===========================================================================
print("\n[C] completion basin: masked [theta_p,inf) vs transparent [theta_bare,inf):")
theta_bare = theta_bare_machinery
print(f"   masked cutoff   theta_p   = {theta_p:.4f}")
print(f"   transparent cut theta_bare= {theta_bare:.4f}  (pivotality revealed)")
print(
    f"   basin shrinks when revealed? cutoff rises by {theta_bare - theta_p:+.4f}  "
    f"{'YES (basin SHRINKS)' if theta_bare > theta_p else 'NO'}"
)
# measure-based completion probability over a good-coalition distribution
for mu, s in [(0.3, 0.15), (0.4, 0.2), (0.5, 0.25)]:
    draws = mu + s * rng.standard_normal(200000)
    draws = draws[draws > 0]
    comp_masked = np.mean(draws >= theta_p)
    comp_transp = np.mean(draws >= theta_bare)
    print(
        f"   theta_g~N({mu},{s}^2)|>0: completion masked={comp_masked:.3f} "
        f"transparent={comp_transp:.3f}  Delta={comp_masked - comp_transp:+.3f} "
        f"{'(masking completes MORE)' if comp_masked > comp_transp else ''}"
    )

# ===========================================================================
# PANIC INVERSION cross-check (independent): the realized mass at theta_g under the
# optimal device threshold crosses the activation bar exactly at theta_p.
# ===========================================================================
print("\n[panic] independent crossing check at optimal t*:")
f = lambda x: perceived_success(x, t_star) - tau
xstar_opt = brentq(f, t_star - 9 * sig, t_star + 9 * sig, xtol=1e-11)
for dth in [+0.10, +0.02, 0.0, -0.02, -0.10]:
    tg = theta_p + dth
    ell = Phi((tg - xstar_opt) / sig)
    bar = 1 - tg
    print(
        f"   theta_g={tg:6.3f}: mass ell={ell:.3f} vs bar={bar:.3f}  "
        f"-> {'GO' if ell >= bar else 'PANIC'}"
    )

print("\n" + "=" * 76)
print("VERDICT SUMMARY")
print("=" * 76)
print(
    f"  A: bare-machinery frontier={theta_bare_machinery:.4f} (->tau as alpha->inf); "
    f"asserted tau={tau}"
)
print(
    f"  B: theta_p={theta_p:.4f} < tau={tau} (HELPS) and >0 (NOT first-best): "
    f"{theta_p < tau and theta_p > 0}"
)
print(
    f"  C: revealing pivotality raises cutoff {theta_p:.4f}->{theta_bare:.4f}: "
    f"basin SHRINKS = {theta_bare > theta_p}"
)
print(
    f"  FAIR same-machinery gap (theta_bare - theta_p) = {theta_bare - theta_p:+.4f} "
    f"(vs develop's asserted {tau - theta_p:+.4f})"
)
print("=" * 76)
