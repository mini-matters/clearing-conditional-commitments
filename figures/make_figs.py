"""Generate the three working-paper figures from the verified numcheck model scripts.

Run from the repository root:
  uv run --with numpy --with scipy --with matplotlib python figures/make_figs.py
"""

import sys
from math import ceil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

import os

HERE = os.path.dirname(os.path.abspath(__file__))
NUMCHECK = os.path.join(HERE, "..", "numerics")
OUT = os.environ.get("FIG_OUT", HERE)
sys.path.insert(0, NUMCHECK)

import v9_dynamic_two_channel as s11  # noqa: E402
import v9_selection_lambdafree as s10  # noqa: E402

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"

DBLUE = "#00308F"
DRED = "#8B0000"

# ================================================================================
# FIGURE 1 — fig-band.pdf : k*(c) interior masking band (v3_shape.py model, verbatim)
# ================================================================================
print("[fig 1] computing k*(c) on the v3_shape model ...")

V, L, TAU, KAP, Q = 1.0, 1.2, 0.7, 8.0, 0.95


def Fc(theta, c, kap):
    # verbatim from v3_shape.py (vectorized over theta)
    m = min(max(c, 1e-4), 1 - 1e-4)
    a, b = m * kap, (1 - m) * kap
    return beta.cdf(theta, a, b)


def W_p1(k, q, c):
    rho = Fc(TAU * k, c, KAP)
    return q * c * (1 - rho) * V - (1 - q * c) * (1 - k) * L


ks = np.arange(0.0, 1.0 + 1e-12, 0.002)
cs = np.arange(0.05, 0.98 + 1e-12, 0.001)
kstar = np.empty_like(cs)
for i, c in enumerate(cs):
    kstar[i] = ks[int(np.argmax(W_p1(ks, Q, c)))]

interior = kstar < 1.0 - 1e-9
idx = np.where(interior)[0]
c_lo, c_hi = float(cs[idx[0]]), float(cs[idx[-1]])
k_min = float(kstar[idx].min())
c_at_kmin = float(cs[idx[np.argmin(kstar[idx])]])
print(
    f"[fig 1] band edges: c in ({c_lo:.4f}, {c_hi:.4f}); "
    f"valley min k*={k_min:.3f} at c={c_at_kmin:.3f}"
)

fig, ax = plt.subplots(figsize=(4.2, 3.0))
ax.plot(cs, kstar, color=DBLUE, lw=1.5)
ax.axvline(c_lo, color="0.6", ls="--", lw=0.8)
ax.axvline(c_hi, color="0.6", ls="--", lw=0.8)
ax.set_xlabel(r"$c$")
ax.set_ylabel(r"$k^*(c)$")
ax.set_ylim(-0.03, 1.05)
ax.set_xlim(0.05, 0.98)
fig.tight_layout()
fig.savefig(f"{OUT}/fig-band.pdf", bbox_inches="tight")
plt.close(fig)
print("[fig 1] saved fig-band.pdf")

# ================================================================================
# FIGURE 2 — fig-Gk.pdf : G(k) inverted-U at the v9 S11 anchor (script functions reused)
# ================================================================================
print("[fig 2] computing G(k) at the v9 S11 anchor ...")

# anchor parameter block verbatim from v9_dynamic_two_channel.main()
N, kappa, Lp, rho, p_opp, V1, Lbad, T = 10, 1.0, 3.0, 0.05, 0.6, 3.0, 3.0, 10
theta = 0.30
K = max(1, ceil(round((1 - theta) * N, 6)))
pi_b, xi = 0.2, 0.667

kgrid = np.arange(0.0, 1.0 + 1e-12, 0.05)
Gs = np.empty_like(kgrid)
for i, k in enumerate(kgrid):
    G, s = s11.G_and_s(float(k), N, K, pi_b, V1, Lbad, xi, kappa, Lp, T, rho, p_opp)
    Gs[i] = G
    print(f"[fig 2]   k={k:.2f}  G={G:.4f}")

ipk = int(np.argmax(Gs))
print(
    f"[fig 2] peak: G={Gs[ipk]:.4f} at k={kgrid[ipk]:.2f}; "
    f"G(0)={Gs[0]:.4f}, G(1)={Gs[-1]:.4f}"
)

fig, ax = plt.subplots(figsize=(4.2, 3.0))
ax.plot(kgrid, Gs, color=DBLUE, lw=1.5)
ax.set_xlabel(r"$k$")
ax.set_ylabel(r"$G(k)$")
ax.set_xlim(0, 1)
fig.tight_layout()
fig.savefig(f"{OUT}/fig-Gk.pdf", bbox_inches="tight")
plt.close(fig)
print("[fig 2] saved fig-Gk.pdf")

# ================================================================================
# FIGURE 3 — fig-adv.pdf : pi_masked vs pi_revealed over theta (v9 S10 solvers reused)
# ================================================================================
print("[fig 3] computing masked/revealed clearing over theta ...")

# calibration block verbatim from v9_selection_lambdafree.main()
N3, kappa3, Lp3, rho3, xi3, p_opp3, V13, T3 = 10, 1.0, 3.0, 0.05, 0.8, 0.6, 3.0, 10
thetas = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75]
pm, pr = [], []
for th in thetas:
    Kt = max(1, ceil(round((1 - th) * N3, 6)))
    sm = s10.solve_masked_rd(N3, Kt, V13, xi3, kappa3, Lp3, T3, rho3, p_opp3)
    sr = s10.solve_revealed_rd(N3, Kt, V13, xi3, kappa3, Lp3, T3, rho3, p_opp3)
    pm.append(sm["pi_succ"])
    pr.append(sr["pi_succ"])
    print(
        f"[fig 3]   theta={th:.2f}  pi_masked={pm[-1]:.4f}  "
        f"pi_revealed={pr[-1]:.4f}  adv={pm[-1] - pr[-1]:+.4f}"
    )

fig, ax = plt.subplots(figsize=(4.2, 3.0))
ax.plot(thetas, pm, color=DBLUE, lw=1.5, ls="-", label=r"$\pi_{\mathrm{masked}}$")
ax.plot(thetas, pr, color=DRED, lw=1.5, ls="--", label=r"$\pi_{\mathrm{revealed}}$")
ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"$\pi$")
ax.set_ylim(-0.05, 1.05)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(f"{OUT}/fig-adv.pdf", bbox_inches="tight")
plt.close(fig)
print("[fig 3] saved fig-adv.pdf")
print("done")
