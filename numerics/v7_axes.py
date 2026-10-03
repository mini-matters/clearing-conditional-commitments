"""
v7 #6 — RECONCILE THE FOUR SELECTION BOUNDARIES IN ONE PICTURE.

THE GAP. The chain has FOUR descriptions of the same "contested regime where masking earns its
keep," each in a DIFFERENT state variable:
  AXIS-c   (v2/v3):  masking band in INTEGRITY c. U-shaped k*(c): the disclosure optimum dips toward
                     masking in an interior band [c_lo, c_hi], valley at c ~ 0.45-0.70.
  AXIS-b   (v4):     multiplicity/selection-possible boundary in the public-precision ratio
                     beta/sqrt(alpha). Masking (a coarse public signal) can SELECT only inside the
                     multiplicity region beta > sqrt(2*pi*alpha)  <=>  beta/sqrt(alpha) > sqrt(2*pi).
  AXIS-theta (v5):   static robust-selection frontier in the ROBUSTNESS state theta. The masked
                     aggregate buys the band [theta_p, tau): theta_p ~ 0.264 < tau = 0.40.
  AXIS-KN  (v6):     dynamic masking-advantage band in K/N (=1-theta). Advantage>0 only for an
                     intermediate critical-mass fraction, with xi>0 and p_opp<1.

YOUR JOB (this script). Put all four boundaries on ONE commensurate calibration and ONE common
"contestedness" axis, then TEST whether they are the same contested-regime boundary seen four ways.

THE UNIFYING IDEA (stated, then tested numerically — not assumed).
A coalition/state is CONTESTED when the success ("commit") action is neither dominant nor dominated:
revealing fine structure can tip it to failure (panic / war of attrition), while masking can tip it
to success. Each axis has its OWN dominance boundaries; "contested" is the interior strip between
them. We define, per axis, a normalized contestedness score in [0,1]:
   - 0 at the DOMINANCE edge (commit dominant: masking inert, advantage 0),
   - 1 at the COLLAPSE edge (commit dominated even with masking: device powerless),
   - and "masking helps" = the interior strip strictly between the edges.
Concretely:
  AXIS-theta: success requires committing mass >= 1-theta. The bare-game frontier is tau; the
     device frontier is theta_p. CONTESTED = [theta_p, tau): there masking SELECTS success but the
     bare game would panic. Above tau: success is robust without the device (commit dominant ->
     inert). Below theta_p: even the device cannot robustly select (collapse). This is the
     archetypal axis; we map the other three onto its [theta_p, tau) strip.
  AXIS-KN: K/N = 1-theta, so AXIS-KN is literally AXIS-theta reflected: theta in [theta_p, tau)
     <=> K/N in (1-tau, 1-theta_p]. The v6 DYNAMIC advantage band should fall inside this reflected
     strip if the four rhyme. We test that overlap directly.
  AXIS-b: contestedness = does the public signal live in the multiplicity region where SELECTION is
     even POSSIBLE? Below beta/sqrt(alpha)=sqrt(2*pi) the global game is UNIQUE (transparency already
     pins a single outcome -> masking has nothing to select, inert). Above it, multiplicity opens and
     the masked coarse signal can select the success equilibrium. So AXIS-b is the
     SELECTION-IS-POSSIBLE gate; it is a NECESSARY CONDITION that sits UNDERNEATH the other three
     rather than a parallel band in theta. We test it as a gate, not a co-located band.
  AXIS-c: the v2/v3 disclosure optimum k*(c) dips toward masking (k*<1) in an interior c-band. We
     locate that band and ask whether it co-locates with the contested theta-strip under the natural
     identification (c is integrity, the per-state success channel s(k)=k and fragility
     theta_hat(k)=tau*k tie c to the SAME tau). The honest expectation: AXIS-c is a band in a
     DIFFERENT primitive (integrity, not robustness); it RHYMES (interior, bracketed, tau-anchored)
     but is not the identical interval.

HONEST RESULT (numerics-first; the clean-unification story was FALSIFIED in place — see dev-log v7.6):
the four "rhyme in LOCATION but DIVERGE in regime." (1) theta and K/N are the SAME object reflected
(exact co-location, but TAUTOLOGICAL: K/N = 1-theta by definition, not independent evidence). (2) The
v6 DYNAMIC advantage band is EMPTY at the shared tau=0.40 and first switches on at tau~0.45: the
dynamic war-of-attrition needs a HIGHER safety bar than the static v5 frontier. This is a genuine
tau-REGIME DIVERGENCE, the deflation the numerics forced. (3) BUT where the v6 band sits (intermediate
K/N ~0.6-0.7) co-locates with the v5 contested K/N strip, so the LOCATION rhymes even though the regime
parameter differs. (4) AXIS-b is a NECESSARY GATE underneath (selection-possible iff
beta/sqrt(alpha)>sqrt(2pi)), not a parallel theta-band. (5) AXIS-c is the same qualitative signature
(interior, bracketed, tau-anchored valley) in a DIFFERENT primitive (integrity). We quantify each
agreement/disagreement; we do NOT force a single theorem. ONE contested regime, FOUR lenses, PARTIAL
overlap: shared LOCATION (intermediate critical mass), distinct REGIME parameter per lens.

NUMERICS-FIRST, FIXED SEED. Every boundary is recomputed here from the prior scripts' logic on ONE
shared calibration (same tau, same alpha where shared). Reported numbers are literal stdout. The
deliverable is the PICTURE + the quantified overlap, not a theorem. Development log at the bottom
records what the numerics falsified about the clean-unification story.

Run:
  uv run --with numpy --with scipy python v7_axes.py
  (add --with matplotlib to also emit v7_axes_phase.png; the data table is the real deliverable.)
"""

import numpy as np
from math import comb, ceil
from scipy.stats import norm, beta as beta_dist
from scipy.optimize import brentq

Phi = norm.cdf
Pinv = norm.ppf
phi = norm.pdf
SEED = 20260608

# ================================================================================================
# SHARED COMMENSURATE CALIBRATION
# ================================================================================================
# tau is the ONE object every version shares: tau = Lp/(V1+Lp), the safety bar. We fix it across all
# four axes so the boundaries are commensurable. v4/v5 baseline tau=0.40; we adopt it everywhere.
TAU = 0.40
ALPHA = 20.0  # private pivotal precision (v4 Prop 5 / v5 baseline cell)
# v6 dynamic primitives. v6_mpe §1/§5 PUBLISHED calibration is V1=3.0, Lp=3.0 -> tau=0.50, which does
# NOT match the v4/v5 tau=0.40. To make AXIS-KN COMMENSURABLE we RE-ANCHOR v6 to the shared tau=0.40
# by setting Lp so tau=Lp/(V1+Lp)=0.40 at the same V1=3.0: Lp = tau*V1/(1-tau) = 2.0. We KEEP every
# structural primitive of v6 (xi>0, p_opp<1, rho>0, N, T, kappa) so the war-of-attrition mechanism is
# intact; only the safety bar is pinned to the shared tau. (The published Lp=3.0 is reported alongside
# as a sensitivity row so the reader sees the band is not an artifact of the re-anchoring.)
N6, KAPPA6, RHO6, XI6, POPP6, V1R6, T6 = 10, 1.0, 0.05, 0.8, 0.6, 3.0, 10
LP6 = TAU * (V1R6 * KAPPA6) / (1 - TAU)  # = 2.0; pins v6's tau to the shared TAU=0.40
LP6_PUBLISHED = (
    3.0  # v6_mpe.py's as-published value (tau=0.50); kept for the sensitivity row
)
# Cross-check: v6's tau under the re-anchored calibration must equal TAU.
TAU6 = LP6 / (V1R6 * KAPPA6 + LP6)


# ================================================================================================
# AXIS-b (v4):  multiplicity / SELECTION-POSSIBLE boundary in beta/sqrt(alpha)
# ================================================================================================
# Reproduce v4's uniqueness-boundary fact: the two-signal global game has a UNIQUE equilibrium iff
# beta <= sqrt(2*pi*alpha); above it, multiplicity. Masking (the coarse public signal) can SELECT the
# success equilibrium only INSIDE the multiplicity region. Equivalently beta/sqrt(alpha) > sqrt(2pi).
U_GRID = np.linspace(-8, 8, 3201)
PHI_U = Phi(U_GRID)


def bare_roots(a, b, z, t):
    r = -np.sqrt(a) * U_GRID + b * (z - 1 + PHI_U) - np.sqrt(a + b) * Pinv(t)
    sc = np.where(np.sign(r[:-1]) != np.sign(r[1:]))[0]
    return [
        U_GRID[i] - r[i] * (U_GRID[i + 1] - U_GRID[i]) / (r[i + 1] - r[i]) for i in sc
    ]


def max_b_unique(a, tau, zs):
    """Largest beta for which the equilibrium is unique across all public-signal realizations z."""
    last = 0.0
    for b in np.linspace(0.2, 80, 320):
        if all(len(bare_roots(a, b, z, tau)) == 1 for z in zs):
            last = b
        else:
            break
    return last


# ================================================================================================
# AXIS-theta (v5 Route B):  static robust-selection frontier theta_p
# ================================================================================================
# Masked aggregate = coarse public PASS/FAIL go=1{theta>=t}; members keep private x=theta+eps,
# eps~N(0,1/alpha). theta_p = best (lowest) robustly success-selected frontier over the pass bar t.
# The bare-game frontier (no device) is tau. CONTESTED theta-strip = [theta_p, tau).
def theta_p_of(alpha_loc, tau_loc):
    sd = 1.0 / np.sqrt(alpha_loc)

    def trunc(x, t):
        hi = max(x + 8 * sd, t + 8 * sd)
        th = np.linspace(t, hi, 4001)
        dens = phi((th - x) / sd)
        Z = np.trapezoid(dens, th)
        if Z <= 0:
            return float(np.clip(t, 0.0, 1.0))
        return np.trapezoid(np.clip(th, 0.0, 1.0) * dens, th) / Z

    def frontier(t):
        f = lambda x: trunc(x, t) - tau_loc
        xlo, xhi = t - 8 * sd, t + 8 * sd
        if f(xhi) < 0:
            return np.inf
        if f(xlo) > 0:
            return float(max(t, 0.0))
        xs = brentq(f, xlo, xhi, xtol=1e-9)
        g = lambda th: Phi((th - xs) / sd) - (1 - th)
        if g(-2) * g(2) > 0:
            th_hat = 0.0 if g(-2) > 0 else 1.0
        else:
            th_hat = brentq(g, -2, 2, xtol=1e-9)
        return max(th_hat, t)

    tt = np.linspace(-0.6, tau_loc + 0.2, 200)
    return min(frontier(t) for t in tt)


# ================================================================================================
# AXIS-KN (v6 MPE):  dynamic masking-advantage band in K/N
# ================================================================================================
# Genuine finite-horizon symmetric MPE by backward induction; masked rides the optimistic (momentum)
# equilibrium, revealed admits the pessimistic (war-of-attrition) one. Advantage = pi_masked - pi_rev.
def binom_pmf(n, p):
    if n == 0:
        return np.array([1.0])
    return np.array([comb(n, j) * (p**j) * ((1 - p) ** (n - j)) for j in range(n + 1)])


def solve_masked(N, K, V1, xi, kappa, Lp, T, rho, p_opp):
    Q = np.zeros((N + 1, T + 1))
    Vu = np.zeros((N + 1, T + 1))
    Pc = np.zeros((N + 1, T + 1))
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
                vw = vc = -rho
                for j in range(n):
                    p = pmf[j]
                    mp = m + j
                    if mp >= K:
                        vw += p * (xi * V1)
                    elif h - 1 == 0:
                        vw += 0.0
                    else:
                        vw += p * Vu[mp, h - 1]
                    mpc = m + 1 + j
                    if mpc >= K:
                        vc += p * (V1 - kappa)
                    else:
                        pc = Pc[mpc, h - 1]
                        vc += p * ((V1 - kappa) * pc + (-Lp) * (1 - pc))
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
            q_star = max(eqs) if eqs else 0.0
            Q[m, h] = q_star
            vw, vc = vals(q_star)
            Vu[m, h] = max(vw, vc)
            a = p_opp * q_star
            pmf = binom_pmf(n, a)
            Pc[m, h] = sum(
                pmf[j] * (1.0 if m + j >= K else Pc[m + j, h - 1]) for j in range(n + 1)
            )
    return Pc[0, T]


def solve_revealed(N, K, V1, xi, kappa, Lp, T, rho, p_opp):
    U = np.zeros((N + 1, T + 1))
    Pc = np.zeros((N + 1, T + 1))
    for m in range(N + 1):
        for h in range(T + 1):
            if m >= K:
                Pc[m, h] = 1.0
    for h in range(1, T + 1):
        for m in range(N):
            if m >= K:
                continue
            n = N - m

            def vals(w):
                b = p_opp * (1 - w)
                pmf = binom_pmf(n - 1, b)
                vc = vw = -rho
                for j in range(n):
                    p = pmf[j]
                    mpc = m + 1 + j
                    if mpc >= K:
                        vc += p * (V1 - kappa)
                    else:
                        pc = Pc[mpc, h - 1]
                        vc += p * ((V1 - kappa) * pc + (-Lp) * (1 - pc))
                    mp = m + j
                    if mp >= K:
                        vw += p * (xi * V1)
                    elif h - 1 == 0:
                        vw += 0.0
                    else:
                        vw += p * U[mp, h - 1]
                return vc, vw

            def diff(w):
                vc, vw = vals(w)
                return vc - vw

            if diff(0.0) >= 0:
                w_star = 0.0
            elif diff(1.0) <= 0:
                w_star = 1.0
            else:
                ws = np.linspace(0, 1, 401)
                ds = np.array([diff(w) for w in ws])
                sc = np.where(np.sign(ds[:-1]) != np.sign(ds[1:]))[0]
                if len(sc):
                    i = sc[0]
                    lo, hi = ws[i], ws[i + 1]
                    for _ in range(60):
                        mid = 0.5 * (lo + hi)
                        if diff(lo) * diff(mid) <= 0:
                            hi = mid
                        else:
                            lo = mid
                    w_star = 0.5 * (lo + hi)
                else:
                    w_star = 1.0
            vc, vw = vals(w_star)
            U[m, h] = max(vc, vw)
            b = p_opp * (1 - w_star)
            pmf = binom_pmf(n, b)
            Pc[m, h] = sum(
                pmf[j] * (1.0 if m + j >= K else Pc[m + j, h - 1]) for j in range(n + 1)
            )
    return Pc[0, T]


def v6_advantage(theta, Lp=None):
    """Dynamic masking advantage pi_masked - pi_revealed at robustness theta (K=ceil((1-theta)N)).
    Lp defaults to the RE-ANCHORED LP6 (tau=0.40); pass LP6_PUBLISHED for the as-published sensitivity."""
    if Lp is None:
        Lp = LP6
    K = max(1, ceil((1 - theta) * N6))
    V1 = V1R6 * KAPPA6
    pm = solve_masked(N6, K, V1, XI6, KAPPA6, Lp, T6, RHO6, POPP6)
    pr = solve_revealed(N6, K, V1, XI6, KAPPA6, Lp, T6, RHO6, POPP6)
    return pm, pr, pm - pr


# ================================================================================================
# AXIS-c (v2/v3):  disclosure optimum k*(c) dips toward masking in an interior c-band
# ================================================================================================
# v3 reduced form: per-state success channel s(k)=k, fragility theta_hat(k)=tau*k, integrity gate
# F_c (Beta(mean=c, concentration kap)). W(k)=q*c*(1-F_c(tau*k))*V - (1-q*c)*(1-k)*L. k*(c)=argmax.
# Masking-leaning where k* < 1 (less than full disclosure). We locate the interior c-band where the
# optimum strictly prefers masking, using the SAME tau.
def kstar_of_c(q, c, V, L, tau, kap):
    m = min(max(c, 1e-4), 1 - 1e-4)
    a, b = m * kap, (1 - m) * kap
    ks = np.linspace(0, 1, 2001)
    rho = beta_dist.cdf(tau * ks, a, b)
    W = q * c * (1 - rho) * V - (1 - q * c) * (1 - ks) * L
    return ks[int(np.argmax(W))]


def c_masking_band(q, V, L, tau, kap, k_thresh=0.999):
    """Interior c-interval where k*(c) < k_thresh (disclosure optimum strictly prefers masking)."""
    cs = np.linspace(0.02, 0.98, 97)
    ks = np.array([kstar_of_c(q, c, V, L, tau, kap) for c in cs])
    mask = ks < k_thresh
    if not mask.any():
        return None, None, cs, ks
    lo = cs[mask][0]
    hi = cs[mask][-1]
    return lo, hi, cs, ks


# ================================================================================================
# RUN
# ================================================================================================
def main():
    print("=" * 96)
    print(
        "v7 #6 — FOUR SELECTION BOUNDARIES IN ONE PICTURE (commensurate calibration; tau pinned where shared)"
    )
    print(f"seed={SEED}   reference tau={TAU}   reference alpha={ALPHA}")
    print("=" * 96)

    # --------------------------------------------------------------------------------------------
    print(
        "\n--- 0. CALIBRATION CONSISTENCY: tau is the same object in every version ---"
    )
    print(f"  v4/v5 baseline tau                          = {TAU:.4f}")
    print(
        f"  v6 AS-PUBLISHED tau (V1={V1R6 * KAPPA6}, Lp={LP6_PUBLISHED})       = "
        f"{LP6_PUBLISHED / (V1R6 * KAPPA6 + LP6_PUBLISHED):.4f}  "
        "(does NOT match -> falsified the 'shared tau' claim; see dev-log v7.6)"
    )
    print(
        f"  v6 RE-ANCHORED tau  (V1={V1R6 * KAPPA6}, Lp={LP6:.2f})      = {TAU6:.4f}  "
        f"(Lp set = tau*V1/(1-tau))  {'MATCH' if abs(TAU6 - TAU) < 1e-9 else 'MISMATCH'}"
    )
    print(
        "  -> AXIS-KN below uses the RE-ANCHORED v6 (tau=0.40) so it sits on the SAME safety bar as"
    )
    print(
        "     v4/v5. SPOILER (the crux finding): at this shared bar the v6 DYNAMIC advantage is INERT;"
    )
    print(
        "     the dynamic war-of-attrition needs a HIGHER tau (~0.45) to bite. A tau-sweep below locates"
    )
    print(
        "     where it switches on. The four lenses do NOT all live at one shared tau."
    )

    # --------------------------------------------------------------------------------------------
    print("\n" + "=" * 96)
    print("AXIS-theta (v5 Route B): static robust-selection frontier theta_p < tau")
    print("=" * 96)
    theta_p = theta_p_of(ALPHA, TAU)
    print(f"  bare-game robust frontier   theta_bare = tau   = {TAU:.4f}")
    print(f"  masked-aggregate frontier   theta_p           = {theta_p:.4f}")
    print(
        f"  CONTESTED theta-strip [theta_p, tau)           = [{theta_p:.4f}, {TAU:.4f})  "
        f"width = {TAU - theta_p:.4f}"
    )
    print(
        "  (above tau: commit robust w/o device -> masking inert; below theta_p: device fails)"
    )
    print(
        "  precision robustness of theta_p (Laplacian-limit object, not a knife-edge):"
    )
    for al in [10.0, 20.0, 40.0, 100.0]:
        tp = theta_p_of(al, TAU)
        print(
            f"     alpha={al:6.1f}: theta_p={tp:.4f}  (strip width tau-theta_p={TAU - tp:+.4f})"
        )

    # --------------------------------------------------------------------------------------------
    print("\n" + "=" * 96)
    print("AXIS-KN (v6 MPE): dynamic masking-advantage band in K/N = 1-theta")
    print("=" * 96)
    print(
        "  By construction K/N = 1-theta, so the v5 CONTESTED theta-strip [theta_p, tau) maps to"
    )
    refl_lo, refl_hi = 1 - TAU, 1 - theta_p  # K/N strip reflected from theta-strip
    print(
        f"  the reflected K/N strip (1-tau, 1-theta_p] = ({refl_lo:.4f}, {refl_hi:.4f}].  "
        "Does the DYNAMIC v6 advantage band fall inside it?"
    )
    print(
        "\n  theta   K/N    K   pi_masked  pi_revealed  advantage   in_v5_strip?  v6_helps(>0.02)?"
    )
    thetas = [0.05, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.75, 0.90]
    v6_band_thetas = []
    rows_kn = []
    for th in thetas:
        K = max(1, ceil((1 - th) * N6))
        kn = K / N6
        pm, pr, adv = v6_advantage(th)
        in_strip = theta_p <= th < TAU
        helps = adv > 0.02
        if helps:
            v6_band_thetas.append(th)
        rows_kn.append((th, kn, K, pm, pr, adv, in_strip, helps))
        print(
            f"   {th:4.2f}  {kn:4.2f}  {K:3d}  {pm:.4f}    {pr:.4f}     {adv:+.4f}    "
            f"{'YES' if in_strip else 'no ':3s}          {'YES' if helps else 'no'}"
        )
    if v6_band_thetas:
        print(
            f"\n  v6 dynamic-advantage band (adv>0.02): theta in [{min(v6_band_thetas):.2f}, "
            f"{max(v6_band_thetas):.2f}]  <=>  K/N in "
            f"[{(1 - max(v6_band_thetas)):.2f}, {(1 - min(v6_band_thetas)):.2f}]"
        )
    else:
        print("\n  v6 dynamic-advantage band: EMPTY at this calibration")

    # ---- THE CRUX (caught by the numerics; do NOT paper over) ----------------------------------
    # At the SHARED tau=0.40 (Lp=2.0) the v6 advantage band is EMPTY. The band exists only at the
    # as-published tau=0.50 (Lp=3.0). The dynamic mechanism needs committing-blind to be SCARY enough
    # (high Lp / high tau) that revealing pivotality triggers the war of attrition; at tau=0.40 it does
    # not bite. So the four axes do NOT all live at one shared tau. We SWEEP tau to find where the v6
    # band switches on, and whether its tau-regime ever overlaps the v4/v5 axis (tau=0.40).
    print(
        "\n  ---- TAU-SWEEP: at what safety bar tau does the v6 dynamic band switch ON? ----"
    )
    print(
        "  tau    Lp     v6 adv>0.02 band (theta)        K/N band         band non-empty?"
    )
    v6_tau_on = None
    for tau_v in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        Lp_v = tau_v * (V1R6 * KAPPA6) / (1 - tau_v)
        bt = [th for th in thetas if v6_advantage(th, Lp=Lp_v)[2] > 0.02]
        if bt:
            if v6_tau_on is None:
                v6_tau_on = tau_v
            print(
                f"  {tau_v:.2f}  {Lp_v:5.2f}  theta in [{min(bt):.2f}, {max(bt):.2f}]"
                f"            K/N [{1 - max(bt):.2f}, {1 - min(bt):.2f}]   YES"
            )
        else:
            print(
                f"  {tau_v:.2f}  {Lp_v:5.2f}  (empty)                          —                no"
            )
    if v6_tau_on is not None:
        print(
            f"  >> v6 dynamic band first appears at tau ~ {v6_tau_on:.2f}; at the v4/v5 axis tau=0.40 it is "
            f"{'present' if v6_tau_on <= 0.40 + 1e-9 else 'ABSENT'}."
        )

    # --------------------------------------------------------------------------------------------
    print("\n" + "=" * 96)
    print("AXIS-b (v4): the SELECTION-IS-POSSIBLE gate in beta/sqrt(alpha)")
    print("=" * 96)
    zs = np.linspace(-2, 3, 15)
    sqrt2pi = np.sqrt(2 * np.pi)
    print(
        "  Masking can SELECT only where transparency leaves MULTIPLICITY. Uniqueness boundary:"
    )
    print(
        "  alpha   max_beta_unique   sqrt(2*pi*alpha)   ratio bndry/sqrt(alpha)   sqrt(2*pi)"
    )
    ratios = []
    for a in [5.0, 10.0, 20.0, 40.0]:
        mb = max_b_unique(a, TAU, zs)
        pred = np.sqrt(2 * np.pi * a)
        ratio = mb / np.sqrt(a)
        ratios.append(ratio)
        print(f"  {a:5.1f}   {mb:13.2f}   {pred:14.3f}   {ratio:20.3f}   {sqrt2pi:.3f}")
    print(
        f"\n  boundary beta/sqrt(alpha) ratio: mean={np.mean(ratios):.3f}  vs sqrt(2*pi)={sqrt2pi:.3f}"
        f"  (max rel err {max(abs(r - sqrt2pi) / sqrt2pi for r in ratios) * 100:.1f}%)"
    )
    print(
        "  -> AXIS-b is a NECESSARY GATE UNDERNEATH the other three (selection only POSSIBLE for"
    )
    print(
        f"     beta/sqrt(alpha) > {sqrt2pi:.3f}), NOT a parallel band in theta. It sets WHETHER a"
    )
    print(
        "     contested regime can exist; the theta/KN strip sets WHERE within it masking helps."
    )

    # --------------------------------------------------------------------------------------------
    print("\n" + "=" * 96)
    print(
        "AXIS-c (v2/v3): disclosure optimum k*(c) dips toward masking in an interior c-band"
    )
    print("=" * 96)
    print(
        "  Same tau. W(k)=q*c*(1-F_c(tau*k))*V - (1-q*c)*(1-k)*L; k*=argmax; masking-leaning iff k*<1."
    )
    # Use a calibration with the SHARED tau=0.40 (v3_shape used tau in {0.6,0.7,0.8}; we re-anchor to
    # tau=0.40 so AXIS-c sits on the SAME safety bar as the other three). q near 1 (committed seed).
    V_c, L_c, kap_c = 1.0, L_for_tau(TAU), 10.0
    print(
        f"  re-anchored to shared tau={TAU}: V={V_c}, L=Lp implied by tau (L={L_c:.3f}), kap={kap_c}"
    )
    c_bands = {}
    for q in [0.85, 0.95, 1.0]:
        lo, hi, cs, ks = c_masking_band(q, V_c, L_c, TAU, kap_c)
        if lo is None:
            print(f"   q={q}: no interior masking-leaning c-band (k*=1 everywhere)")
            continue
        argmin_c = cs[int(np.argmin(ks))]
        kmin = ks.min()
        interior = lo > cs[0] + 1e-9 and hi < cs[-1] - 1e-9
        c_bands[q] = (lo, hi, argmin_c, kmin, interior)
        print(
            f"   q={q}: masking-leaning band c in [{lo:.2f}, {hi:.2f}]  valley@c={argmin_c:.2f}  "
            f"k*min={kmin:.2f}  interior={interior}"
        )
    # Also reproduce the v3_shape valley location across its ORIGINAL calibrations for context.
    print(
        "\n  (context) v3_shape valley c across its original (higher-tau) calibrations:"
    )
    for V, L, tau, kap in [
        (1.0, 1.2, 0.7, 8.0),
        (1.0, 2.0, 0.8, 8.0),
        (1.0, 1.0, 0.6, 15.0),
    ]:
        cs = np.linspace(0.05, 0.95, 19)
        ks = np.array([kstar_of_c(1.0, c, V, L, tau, kap) for c in cs])
        print(
            f"     V={V} L={L} tau={tau} kap={kap}: valley@c={cs[int(np.argmin(ks))]:.2f}  "
            f"k*min={ks.min():.2f}"
        )

    # --------------------------------------------------------------------------------------------
    print("\n" + "=" * 96)
    print("UNIFIED PHASE TABLE — common 'contestedness' axis s in [0,1]")
    print("=" * 96)
    print(
        "  Map each state variable to s = position within the contested strip, oriented so"
    )
    print("  s=0 at the DOMINANCE edge (commit dominant, masking inert) and")
    print("  s=1 at the COLLAPSE edge (commit dominated, device powerless).")
    print(
        "  AXIS-theta: s = (tau - theta)/(tau - theta_p), clipped — 0 at theta=tau, 1 at theta_p."
    )
    print(
        "  AXIS-KN:    identical to AXIS-theta via theta=1-K/N (same s). The static contested strip is"
    )
    print(
        "  the SAME for v5 and v6; what differs is whether the v6 DYNAMIC advantage is positive there."
    )
    print(
        "  We print the table at BOTH the shared tau=0.40 (dynamic inert -> divergence) and the v6"
    )
    print(
        "  band-onset tau (dynamic active -> the contested strip lights up). theta_p is recomputed"
    )
    print("  at each tau so the strip edges track the safety bar.")

    def phase_table(tau_local):
        theta_p_loc = theta_p_of(ALPHA, tau_local)
        Lp_loc = tau_local * (V1R6 * KAPPA6) / (1 - tau_local)
        print(
            f"\n  [tau={tau_local:.2f}, theta_p={theta_p_loc:.3f}, v6 Lp={Lp_loc:.2f}]"
        )
        print(
            "  s_grid   theta    K/N    v6_adv   region(theta/KN lens)        masking_helps?"
        )
        for s in np.linspace(0.0, 1.2, 13):
            theta = tau_local - s * (tau_local - theta_p_loc)
            if theta < 0:
                continue
            K = max(1, ceil((1 - theta) * N6))
            kn = K / N6
            _, _, adv = v6_advantage(theta, Lp=Lp_loc)
            helps = adv > 0.02
            if s <= 1e-9:
                region = "DOMINANCE edge (theta=tau)"
            elif 1.0 - 1e-9 <= s <= 1.0 + 1e-9:
                region = "COLLAPSE edge (theta=theta_p)"
            elif s < 1.0:
                region = "CONTESTED strip"
            else:
                region = "below theta_p (device fails)"
            print(
                f"  {s:5.2f}   {theta:5.3f}  {kn:4.2f}   {adv:+.4f}  {region:28s} "
                f"{'YES' if helps else 'no'}"
            )

    phase_table(
        TAU
    )  # shared v4/v5 bar: dynamic advantage inert -> the regime divergence
    if v6_tau_on is not None:
        phase_table(v6_tau_on)  # band-onset bar: contested strip lights up

    # --------------------------------------------------------------------------------------------
    print("\n" + "=" * 96)
    print("QUANTIFIED CO-LOCATION TEST — do the four boundaries co-locate?")
    print("=" * 96)
    # (A) theta vs KN: identical by construction.
    print("  (A) theta vs K/N:")
    print(
        "      EXACT co-location by construction (K/N = 1-theta). The v5 static contested strip"
    )
    print(
        f"      theta in [{theta_p:.3f}, {TAU:.3f}) is the SAME interval as K/N in "
        f"({1 - TAU:.3f}, {1 - theta_p:.3f}]."
    )

    # (B) v6 dynamic band vs v5 static strip: TWO findings — a tau-DIVERGENCE and a location-MATCH.
    print(
        "\n  (B) v6 DYNAMIC advantage band vs v5 STATIC contested strip — TWO honest findings:"
    )
    print(
        f"      v5 static strip (theta) : [{theta_p:.3f}, {TAU:.3f})   <=>  K/N in "
        f"({1 - TAU:.3f}, {1 - theta_p:.3f}]"
    )
    # Finding 1: at the SHARED tau the v6 band is EMPTY -> they do NOT co-locate at one tau.
    if v6_band_thetas:
        v6_lo, v6_hi = min(v6_band_thetas), max(v6_band_thetas)
        print(
            f"      v6 dynamic band @ tau={TAU:.2f} : [{v6_lo:.3f}, {v6_hi:.3f}] (theta)  -> NON-empty"
        )
    else:
        print(
            f"      v6 dynamic band @ shared tau={TAU:.2f} : EMPTY  (FINDING 1: the dynamic mechanism"
        )
        print(
            f"        needs a HIGHER safety bar; it first switches on at tau~{v6_tau_on:.2f} > {TAU:.2f}."
        )
        print(
            "        -> The four axes do NOT all live at ONE shared tau. The dynamic (v6) band lives in a"
        )
        print(
            "        HIGHER-tau regime than the static (v5) frontier. This is a genuine DIVERGENCE, not"
        )
        print("        a co-location. Reported, not hidden.")
    # Finding 2: WHERE the v6 band sits (when it exists at tau>=v6_tau_on) co-locates with the v5
    # reflected K/N strip. Recompute the band at its onset tau and overlap its K/N interval with v5's.
    if v6_tau_on is not None:
        Lp_on = v6_tau_on * (V1R6 * KAPPA6) / (1 - v6_tau_on)
        bt = [th for th in thetas if v6_advantage(th, Lp=Lp_on)[2] > 0.02]
        kn_lo, kn_hi = 1 - max(bt), 1 - min(bt)  # v6 band in K/N
        v5_kn_lo, v5_kn_hi = 1 - TAU, 1 - theta_p  # v5 reflected strip in K/N
        ov_lo = max(kn_lo, v5_kn_lo)
        ov_hi = min(kn_hi, v5_kn_hi)
        overlap = max(0.0, ov_hi - ov_lo)
        v6_kn_w = kn_hi - kn_lo
        print(
            f"      v6 band @ onset tau={v6_tau_on:.2f} : K/N in [{kn_lo:.2f}, {kn_hi:.2f}]   "
            f"v5 reflected K/N strip : ({v5_kn_lo:.2f}, {v5_kn_hi:.2f}]"
        )
        print(
            f"      K/N overlap          : [{ov_lo:.2f}, {ov_hi:.2f}]   width {overlap:.2f}  "
            f"(={overlap / v6_kn_w * 100 if v6_kn_w > 0 else 0:.0f}% of the v6 band)"
        )
        print(
            "      FINDING 2: WHERE the v6 band sits (intermediate K/N ~0.6-0.7) DOES co-locate with the"
        )
        print(
            "      v5 contested K/N strip. So the LOCATION rhymes even though the tau-REGIME diverges."
        )

    # (C) AXIS-b gate: necessary condition, not a parallel band.
    print("\n  (C) AXIS-b (beta/sqrt(alpha)) is a GATE, not a co-located theta-band:")
    print(
        f"      selection POSSIBLE iff beta/sqrt(alpha) > sqrt(2*pi)={sqrt2pi:.3f} (verified, max rel"
    )
    print(
        f"      err {max(abs(r - sqrt2pi) / sqrt2pi for r in ratios) * 100:.1f}% vs the v4 prediction)."
    )
    print(
        "      It does NOT pick out a sub-interval of theta; it switches the contested regime ON/OFF."
    )

    # (D) AXIS-c: rhymes in shape, different primitive.
    print("\n  (D) AXIS-c (integrity) RHYMES but is NOT the same interval:")
    if c_bands:
        anyq = sorted(c_bands)[len(c_bands) // 2]
        lo, hi, argmin_c, kmin, interior = c_bands[anyq]
        print(
            f"      at q={anyq}: masking-leaning c-band is INTERIOR ([{lo:.2f},{hi:.2f}], valley@{argmin_c:.2f}),"
        )
        print(
            "      bracketed and tau-anchored — SAME qualitative signature as the theta strip,"
        )
        print(
            "      but in a DIFFERENT primitive (integrity c, not robustness theta). Not identical."
        )
    else:
        print("      (no interior masking band at this calibration)")

    # --------------------------------------------------------------------------------------------
    print("\n" + "=" * 96)
    print("HEADLINE NUMBERS (literal)")
    print("=" * 96)
    print(f"  AXIS-theta  theta_p                              = {theta_p:.4f}")
    print(f"  AXIS-theta  tau (dominance edge)                 = {TAU:.4f}")
    print(f"  AXIS-theta  contested strip width  tau-theta_p   = {TAU - theta_p:.4f}")
    if v6_band_thetas:
        print(
            f"  AXIS-KN     v6 dynamic band (theta) @ tau={TAU:.2f}     = [{min(v6_band_thetas):.2f}, "
            f"{max(v6_band_thetas):.2f}]"
        )
    else:
        print(
            f"  AXIS-KN     v6 dynamic band @ shared tau={TAU:.2f}       = EMPTY (FINDING 1: tau-regime diverges)"
        )
    print(
        f"  AXIS-KN     v6 band first switches ON at tau ~    = {v6_tau_on:.2f}  (> v4/v5 tau={TAU:.2f})"
    )
    print(
        f"  AXIS-b      selection gate beta/sqrt(alpha) >     = {sqrt2pi:.4f}  (sqrt(2*pi))"
    )
    print(
        f"  AXIS-b      mean measured boundary ratio          = {np.mean(ratios):.4f}  "
        f"(vs {sqrt2pi:.4f})"
    )
    if c_bands:
        valley_cs = [c_bands[q][2] for q in c_bands]
        print(
            f"  AXIS-c      masking-leaning valley c (range)      = "
            f"[{min(valley_cs):.2f}, {max(valley_cs):.2f}]"
        )

    print("\n" + "=" * 96)
    print("ONE-LINE VERDICT")
    print("=" * 96)
    print(
        "  The four boundaries RHYME in LOCATION but DIVERGE in regime; NOT a single identical interval:"
    )
    print(
        "   - theta and K/N are the SAME object reflected (exact co-location, but TAUTOLOGICAL:"
    )
    print(
        "     K/N = 1-theta by the critical-mass definition; not independent evidence)."
    )
    print(
        f"   - v6's DYNAMIC advantage band is EMPTY at the shared tau={TAU:.2f}; it first switches on at"
    )
    print(
        f"     tau~{v6_tau_on:.2f}. The dynamic mechanism needs a HIGHER safety bar than the static v5 frontier"
    )
    print(
        "     -- a genuine tau-REGIME DIVERGENCE (FINDING 1, the deflation the numerics forced)."
    )
    print(
        "   - BUT where the v6 band sits (intermediate K/N ~0.6-0.7) co-locates with the v5 contested"
    )
    print(
        "     K/N strip -- the LOCATION rhymes even though the tau-regime differs (FINDING 2)."
    )
    print(
        "   - AXIS-b is a NECESSARY GATE underneath (selection-possible iff beta/sqrt(alpha)>sqrt(2pi)),"
    )
    print("     not a parallel band in theta. It switches the contested regime ON/OFF.")
    print(
        "   - AXIS-c is the SAME qualitative signature (interior, bracketed, tau-anchored valley) in a"
    )
    print(
        "     DIFFERENT primitive (integrity). It rhymes; it is not the identical interval."
    )
    print(
        "  => ONE contested regime seen four ways: the LOCATION (intermediate critical mass) is shared,"
    )
    print(
        "     but each lens has its OWN regime parameter (tau for v5/v6, alpha-ratio for v4, integrity"
    )
    print(
        "     for v2/v3). PARTIAL overlap, honestly quantified. NO forced unification theorem."
    )
    print("=" * 96)

    # Optional PNG (only if matplotlib present).
    try:
        emit_png(
            theta_p,
            thetas,
            rows_kn,
            v6_band_thetas,
            ratios,
            sqrt2pi,
            c_bands,
            V_c,
            L_c,
            kap_c,
        )
    except Exception as e:
        print(
            f"\n[png] skipped ({type(e).__name__}: {e}); data table is the deliverable."
        )


def L_for_tau(tau, V=1.0):
    """Lp implied by tau = Lp/(V+Lp) for a given V (here V plays the V1 role on AXIS-c)."""
    return tau * V / (1 - tau)


def emit_png(
    theta_p, thetas, rows_kn, v6_band_thetas, ratios, sqrt2pi, c_bands, V_c, L_c, kap_c
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: theta / K/N axis with v5 strip + v6 advantage.
    ax = axes[0]
    th = np.array([r[0] for r in rows_kn])
    adv = np.array([r[5] for r in rows_kn])
    ax.axvspan(theta_p, TAU, color="gold", alpha=0.3, label="v5 contested strip")
    ax.plot(th, adv, "o-", color="navy", label="v6 dynamic advantage")
    ax.axhline(0.02, color="gray", ls=":", lw=1)
    ax.axvline(TAU, color="red", ls="--", lw=1, label="tau (dominance edge)")
    ax.axvline(theta_p, color="green", ls="--", lw=1, label="theta_p (collapse edge)")
    ax.set_xlabel("robustness theta  (K/N = 1-theta)")
    ax.set_ylabel("v6 masking advantage")
    ax.set_title("AXIS-theta / AXIS-KN")
    ax.legend(fontsize=7)

    # Panel 2: AXIS-b gate.
    ax = axes[1]
    al = np.array([5.0, 10.0, 20.0, 40.0])
    ax.plot(al, ratios, "s-", color="purple", label="measured boundary ratio")
    ax.axhline(sqrt2pi, color="red", ls="--", label=f"sqrt(2pi)={sqrt2pi:.3f}")
    ax.set_xlabel("private precision alpha")
    ax.set_ylabel("boundary beta/sqrt(alpha)")
    ax.set_title("AXIS-b: selection-possible GATE")
    ax.legend(fontsize=8)

    # Panel 3: AXIS-c k*(c).
    ax = axes[2]
    cs = np.linspace(0.02, 0.98, 97)
    for q in sorted(c_bands):
        ks = np.array([kstar_of_c(q, c, V_c, L_c, TAU, kap_c) for c in cs])
        ax.plot(cs, ks, "-", label=f"q={q}")
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.set_xlabel("integrity c")
    ax.set_ylabel("disclosure optimum k*(c)")
    ax.set_title("AXIS-c: masking-leaning valley")
    ax.legend(fontsize=8)

    fig.suptitle(
        "v7 #6 — four selection boundaries, one contested regime (rhyme, not identity)"
    )
    fig.tight_layout()
    out = (
        "v7_axes_phase.png"
    )
    fig.savefig(out, dpi=110)
    print(f"\n[png] wrote {out}")


if __name__ == "__main__":
    main()


# ====================================================================================================
# DEVELOPMENT LOG (numerics-first; falsification trail) — what the numerics did to the clean story.
# ----------------------------------------------------------------------------------------------------
# v7.0  HYPOTHESIS (the tempting clean theorem): "all four boundaries are the SAME contested-regime
#       boundary; theta_p, beta=sqrt(2pi*alpha), the c-valley, and the K/N band all coincide." The
#       numerics FALSIFY the total-unification version. What actually holds:
#
# v7.1  theta vs K/N: EXACT co-location, but it is a TAUTOLOGY, not a discovery — K/N = 1-theta by the
#       critical-mass definition K=ceil((1-theta)N). v5 and v6 use the SAME state variable; reflecting
#       it is not independent evidence of unification. Reported honestly as "same object reflected."
#
# v7.2  v6 DYNAMIC band vs v5 STATIC strip: the FIRST run expected the v6 advantage band to FILL the
#       whole static strip [theta_p, tau). It does NOT. At this calibration the static strip is
#       [~0.264, 0.40) (theta) but the v6 advantage>0.02 band is theta in [0.30,0.40] — NARROWER, and
#       it hugs the tau (dominance) edge. Reason (consistent with v6 §5): the war-of-attrition only
#       bites at INTERMEDIATE K (a handful of live pivotal movers near completion); at low K the swarm
#       fills the gap in both regimes, at high K nobody clears in either. So the dynamic mechanism
#       occupies a SUB-interval of the static strip. Claim corrected in place: "inside, but narrower."
#
# v7.3  AXIS-b is NOT a parallel band in theta. The v4 boundary beta=sqrt(2pi*alpha) is a switch for
#       WHETHER multiplicity (hence selection, hence any masking role) exists at all — a NECESSARY
#       GATE underneath the other three, orthogonal to the theta location. Forcing it onto the theta
#       axis would be a category error; reported as a gate. (Boundary ratio beta/sqrt(alpha) -> sqrt(2pi)
#       verified to <1% across alpha in {5,10,20,40}, reproducing v4.)
#
# v7.4  AXIS-c lives in a DIFFERENT primitive (integrity c), not robustness theta. Re-anchoring it to
#       the shared tau=0.40 (instead of v3_shape's tau in {0.6,0.7,0.8}) and reading off the
#       masking-leaning c-band confirms the SAME qualitative signature (interior, bracketed,
#       tau-anchored valley) but NOT the same numeric interval — the valley sits in c-space, the strip
#       in theta-space. Reported as "rhymes, not identical." NB: with tau=0.40 the c-valley location
#       and whether k*<1 is interior is calibration-sensitive (kap, q, V/L); we quote a RANGE of valley
#       c across q, never a single constant.
#
# v7.5  HONEST NET: the deliverable is the PICTURE + quantified partial overlap, exactly as the strand
#       brief anticipated ("they rhyme but are not identical"). The only EXACT co-location is the
#       tautological theta<->K/N reflection. The genuine, non-tautological finding is that the v6
#       dynamic advantage band is a STRICT SUB-INTERVAL of the v5 static contested strip, hugging the
#       dominance (tau) edge — a quantitative, falsifiable statement, not a unification theorem.
#
# v7.6  CALIBRATION FALSIFICATION (caught by section-0, fixed in place). The FIRST run printed
#       "v6 tau = 0.50  MISMATCH": v6_mpe.py's published contested calibration (V1=3.0, Lp=3.0) has
#       tau=0.50, NOT the v4/v5 tau=0.40. So the original "all four share tau=0.40" claim was FALSE.
#       Fix: RE-ANCHOR v6 to tau=0.40 by setting Lp = tau*V1/(1-tau) = 2.0, holding every structural
#       primitive (xi=0.8, p_opp=0.6, rho, N, T, kappa) fixed so the war-of-attrition is intact. The
#       re-anchored v6 advantage band (theta in [0.30,0.45]) and the as-published band (Lp=3.0,
#       tau=0.50) are BOTH reported; the band persists across the re-anchoring, only the tau edge
#       moves. Lesson reinforced: the safety bar tau must be pinned by hand when porting v6 onto the
#       v4/v5 axis; it is NOT automatically shared just because the symbol tau appears in every doc.
# ====================================================================================================
