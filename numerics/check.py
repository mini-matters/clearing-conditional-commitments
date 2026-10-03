import numpy as np
from scipy.stats import norm, beta
from scipy.optimize import brentq

# --- 1. Lemma 1: solve (M)-(I) numerically, confirm theta_hat = tau, sigma-invariant ---
def solve_thetahat(sigma, tau):
    # indifference: Phi((xhat-thetahat)/sigma)=tau -> xhat = thetahat + sigma*Phi^{-1}(tau)
    # critical mass: Phi((thetahat-xhat)/sigma)=1-thetahat
    def f(th):
        xhat = th + sigma*norm.ppf(tau)
        return norm.cdf((th-xhat)/sigma) - (1-th)
    return brentq(f, -5, 5)

max_err = 0.0
for sigma in [0.05,0.1,0.5,1.0,2.0]:
    for tau in [0.05,0.2,0.5,0.7,0.95]:
        th = solve_thetahat(sigma, tau)
        max_err = max(max_err, abs(th - tau))
print(f"[Lemma 1] max |theta_hat - tau| over grid = {max_err:.2e}  (expect ~0, sigma-invariant)")

# --- 2. Prop 3: rho(c)=F_c(tau) decreasing in c; signs of d/dLp, d/dV1 ---
# F_c = Beta(a,b) on (0,1) with mean increasing in c. Use mean = c, fix concentration k.
def Fc_tau(c, tau, kconc=8.0):
    # Beta with mean=c (clip), concentration kconc => a=mean*k, b=(1-mean)*k
    m = min(max(c,1e-3),1-1e-3)
    a, b = m*kconc, (1-m)*kconc
    return beta.cdf(tau, a, b)

V1, Lp = 1.0, 0.5
tau = Lp/(V1+Lp)
cs = np.linspace(0.1,0.9,9)
rhos = [Fc_tau(c, tau) for c in cs]
mono_dec = all(x> y-1e-12 for x,y in zip(rhos, rhos[1:]))
print(f"[Prop 3] rho(c) strictly decreasing in c: {mono_dec}")
print(f"         rho at c=0.1..0.9: {[round(r,3) for r in rhos]}")
# d/dLp>0, d/dV1<0 via tau
tau_hi_Lp = (Lp*1.1)/(V1+Lp*1.1)
tau_hi_V1 = Lp/(V1*1.1+Lp)
c0=0.5
print(f"[Prop 3] dRho/dLp>0: {Fc_tau(c0,tau_hi_Lp) > Fc_tau(c0,tau)}   dRho/dV1<0: {Fc_tau(c0,tau_hi_V1) < Fc_tau(c0,tau)}")

# --- 3. Theorem 2: single-crossing in mu, and masking set upper-closed in c ---
V, L = 1.0, 0.4
def Delta(mu, c):
    rho = Fc_tau(c, tau)
    return rho*mu*V - (1-mu)*L

# single crossing: exactly one sign change over mu grid, for each c
rng = np.random.default_rng(0)
sc_ok = True
upper_ok = True
for _ in range(10000):
    c = rng.uniform(0.05,0.95)
    mus = np.linspace(0,1,400)
    d = np.array([Delta(m,c) for m in mus])
    signchanges = np.sum(np.diff(np.sign(d[d!=0]))!=0)
    if signchanges != 1: sc_ok=False
# upper set: m*(c) and boundary monotonicity -> if mu>=m*(c) then mu>=m*(c') for c'>c at same mu? 
# check: masking region membership is monotone in c at fixed mu
for _ in range(10000):
    mu = rng.uniform(0,1)
    c = rng.uniform(0.05,0.9)
    cp = rng.uniform(c,0.95)
    def mask(mu,c):
        rho=Fc_tau(c,tau); mstar=L/(L+rho*V); return mu>=mstar
    if mask(mu,c) and not mask(mu,cp): upper_ok=False
print(f"[Thm 2] single-crossing in mu (10k draws): {sc_ok}")
print(f"[Thm 2] masking set upper-closed in c (10k draws): {upper_ok}")
