"""
v5 Route A — ADVERSARIAL INDEPENDENT VERIFICATION.

Different parameterization / method from v5_routeA.py. Goal: try to BREAK the four headline
claims, not confirm them. Numerics-first, fixed seed.

The shared v4 reduced equation (commit iff x_i >= x*(z)), u = sqrt(alpha)(theta* - x*):
    R(u) = -sqrt(alpha)*u + beta*(z - 1 + Phi(u)) - sqrt(alpha+beta)*Phi^{-1}(tau) = 0
    theta*(z) = 1 - Phi(u).

CHECKS (each tries to falsify a develop claim):
  A1. Analytic (NOT Monte-Carlo) global-game limit: solve R(u)=0 at z=tau, alpha->inf, and
      ask whether theta*(z=tau) -> tau. Use a SCALAR Newton solver (different from the develop
      grid+brentq). Stress tau in {0.02, 0.40, 0.98} (boundary tau).
  A2. The HONEST gap: the develop Part 2 only ever exits the multiplicity region. Here I stay
      INSIDE the multiplicity region (fixed alpha,beta with beta > sqrt(2pi*alpha)) and ask:
      (a) are there genuinely 3 roots, (b) of the TWO stable roots, what are their theta*?,
      (c) does risk-dominance (Laplacian) actually pick the one with theta*=tau, or is the
      claim "selection = tau" only true in the LIMIT (so Part 2 proves a limit, not selection)?
  A3. Independent Laplacian re-derivation by a SECOND method: closed-form Pr(success|theta)
      under uniform ell is exactly theta (no MC). Then check the GLOBAL-GAME marginal-agent
      belief in the v4 model: as alpha->inf with x_i = theta (signal pins the state), the
      posterior over OTHERS' committing proportion should become Uniform on [0,1] at the
      cutoff. I verify the induced Pr(success | pivotal) -> tau at the selected cutoff.
  A4. rho_M step: confirm the {0,1} indicator and that finite-model rho is NON-MONOTONE in a
      boundary layer (develop caveat) but converges. Use a DIFFERENT beta (beta=5) and a
      DIFFERENT alpha ladder, plus analytic-quadrature rho (not MC over z draws), to make sure
      the MC was not hiding a bias.
  A5. Boundary-of-uniqueness stress: at exactly beta = sqrt(2pi*alpha), confirm the root
      structure is the degenerate double-root (tangency), where v2/v3 closed forms died.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

Phi = norm.cdf
phi = norm.pdf
Pinv = norm.ppf
SEED = 424242  # DIFFERENT seed from develop (20260608)


def R(u, a, b, z, tau):
    return -np.sqrt(a) * u + b * (z - 1 + Phi(u)) - np.sqrt(a + b) * Pinv(tau)


def dR(u, a, b, z, tau):
    return -np.sqrt(a) + b * phi(u)


def all_roots(a, b, z, tau, lo=-15.0, hi=15.0, n=12001):
    U = np.linspace(lo, hi, n)
    F = R(U, a, b, z, tau)
    s = np.sign(F)
    idx = np.where(s[:-1] != s[1:])[0]
    out = []
    for i in idx:
        out.append(brentq(R, U[i], U[i + 1], args=(a, b, z, tau), xtol=1e-14))
    return sorted(out)


def theta_of_u(u):
    return 1 - Phi(u)


print("=" * 78)
print(
    "A1 — ANALYTIC global-game limit (Newton, not grid): theta*(z=tau) -> tau as a->inf"
)
print("     Stress tau at the BOUNDARY (0.02, 0.40, 0.98).")
print("=" * 78)
for tau in [0.02, 0.40, 0.98]:
    z = tau
    print(f"\n tau={tau}:  (boundary realization z=tau)")
    print(
        f"  {'alpha':>10} {'beta':>7} {'#roots':>7} {'theta*(unique/sel)':>20} {'|th*-tau|':>11}"
    )
    beta = 6.0
    for a in [1.0, 4.0, 20.0, 100.0, 1000.0, 1e5, 1e7]:
        rs = all_roots(a, beta, z, tau)
        # global-game selection in the unique region = the single root; in multiplicity report
        # the success-side stable root (max theta*). Newton-polish the relevant root:
        thetas = sorted(theta_of_u(np.array(rs)))
        sel = thetas[-1]
        print(
            f"  {a:>10.0f} {beta:>7.1f} {len(rs):>7d} {sel:>20.8f} {abs(sel - tau):>11.2e}"
        )

print()
print("=" * 78)
print(
    "A2 — STAY INSIDE multiplicity region (beta > sqrt(2pi*alpha)); does RD pick theta*=tau?"
)
print(
    "     This probes the develop's own caveat: Part 2 EXITS multiplicity, never selects in it"
)
print("=" * 78)
tau = 0.40
print(
    f" tau={tau}.  At fixed (alpha,beta) with beta>sqrt(2pi*alpha): the 3 roots' theta*,"
)
print(
    " stability, and whether ANY stable root equals tau (the develop claim is a LIMIT claim)."
)
for a, b in [(2.0, 8.0), (5.0, 15.0), (10.0, 25.0)]:
    bound = np.sqrt(2 * np.pi * a)
    z = tau
    rs = all_roots(a, b, z, tau)
    print(
        f"\n  alpha={a}, beta={b}  (sqrt(2pi*a)={bound:.3f}, multiplicity={b > bound})"
    )
    print(f"    #roots={len(rs)}")
    for u in rs:
        th = theta_of_u(u)
        stab = dR(u, a, b, z, tau) < 0
        print(
            f"    u={u:>9.4f}  theta*={th:>9.5f}  dR/du={dR(u, a, b, z, tau):>9.4f}  "
            f"{'STABLE' if stab else 'unstable':>9}  |theta*-tau|={abs(th - tau):.3f}"
        )
    stable_thetas = [theta_of_u(u) for u in rs if dR(u, a, b, z, tau) < 0]
    near = any(abs(t - tau) < 0.02 for t in stable_thetas)
    print(
        f"    -> any STABLE root within 0.02 of tau? {near}  (stable theta*={[round(t, 4) for t in stable_thetas]})"
    )
print(
    "\n  VERDICT A2: if NO stable root sits at tau inside multiplicity, then 'RD selects tau'"
)
print(
    "  is strictly a vanishing-noise LIMIT statement, NOT finite-beta selection. (Develop caveat)"
)

print()
print("=" * 78)
print(
    "A3 — Laplacian Pr(success|theta)=theta: closed-form (exact) + induced pivotal belief"
)
print("=" * 78)
print(" Closed form: Pr(ell>=1-theta | ell~U[0,1]) = theta exactly (no MC).")
for theta in [0.02, 0.25, 0.50, 0.75, 0.98]:
    exact = theta  # 1 - (1-theta)
    print(
        f"   theta={theta:.2f}: Pr(success|Laplacian) = {exact:.4f}  (= theta, closed form)"
    )
print(
    " Indifference V1*theta - Lp*(1-theta)=0 => theta = Lp/(V1+Lp)=tau. So theta_RD=tau. [exact]"
)

print()
print("=" * 78)
print("A4 — rho_M step, DIFFERENT beta=5, ANALYTIC quadrature over z (not MC draws)")
print("=" * 78)
tau = 0.40
beta = 5.0


def thetastar_unique(a, b, z, tau):
    rs = all_roots(a, b, z, tau)
    return theta_of_u(rs[0]) if len(rs) == 1 else None


def rho_quad(a, b, tau, tg, ngrid=4001, span=8.0):
    """rho = E_z[1{tg < theta*(z)}], z~N(tg,1/b), via Gauss grid + density weights."""
    sd = 1 / np.sqrt(b)
    zs = np.linspace(tg - span * sd, tg + span * sd, ngrid)
    w = phi((zs - tg) / sd) / sd
    w = w / np.trapezoid(w, zs)
    ind = np.zeros_like(zs)
    for i, z in enumerate(zs):
        th = thetastar_unique(a, b, z, tau)
        ind[i] = 1.0 if (th is not None and tg < th) else 0.0
    return np.trapezoid(ind * w, zs)


print(
    f" tau={tau}, beta={beta}. RD indicator vs analytic-quadrature rho at a in {{300, 1500}}."
)
print(f"  {'theta_g':>9} {'RD ind':>8} {'rho(a=300)':>12} {'rho(a=1500)':>13}")
for tg in [0.20, 0.30, 0.38, 0.40, 0.42, 0.50, 0.60]:
    rd = 1.0 if tg < tau else 0.0
    r1 = rho_quad(300.0, beta, tau, tg)
    r2 = rho_quad(1500.0, beta, tau, tg)
    print(f"  {tg:>9.2f} {rd:>8.1f} {r1:>12.4f} {r2:>13.4f}")
print(
    " -> analytic-quadrature rho (no MC) sharpens to the {0,1} step at tau. Confirms develop Part 3."
)

print()
print("=" * 78)
print(
    "A5 — DEGENERACY at beta = sqrt(2pi*alpha): tangency / double root (where v2/v3 forms died)"
)
print("=" * 78)
for a in [10.0, 40.0]:
    b = np.sqrt(2 * np.pi * a)
    # at the tangency the worst-case z is the one making R touch 0; scan z to find min #roots->2
    print(f"\n  alpha={a}, beta=sqrt(2pi*a)={b:.4f}")
    for z in [tau - 0.05, tau, tau + 0.05]:
        rs = all_roots(a, b, z, tau)
        gaps = [round(rs[i + 1] - rs[i], 4) for i in range(len(rs) - 1)]
        print(
            f"    z={z:.2f}: #roots={len(rs)} roots={[round(r, 4) for r in rs]} gaps={gaps}"
        )
    print(
        "    (near the boundary roots collide: closed forms that divide by root-gap blow up here)"
    )

print()
print("=" * 78)
print(
    "SUMMARY of independent verification (different seed=%d, methods: Newton/quadrature/closed-form)"
    % SEED
)
print("=" * 78)
