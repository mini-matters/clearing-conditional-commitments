import numpy as np
from scipy.stats import beta
# Use v2's family for consistency: F_c = Beta(mean=c, concentration kappa) on [0,1].
# Channels: theta_hat(k)=tau*k, s(k)=k.  Find k*(c) numerically (no closed form).
def Fc(theta,c,kap):
    m=min(max(c,1e-4),1-1e-4); a,b=m*kap,(1-m)*kap
    return beta.cdf(theta,a,b)
def W(k,q,c,V,L,tau,kap):
    rho=Fc(tau*k,c,kap); s=k
    return q*c*(1-rho)*V-(1-q*c)*(1-s)*L
def kstar(q,c,V,L,tau,kap):
    ks=np.linspace(0,1,2001)
    Wg=[W(k,q,c,V,L,tau,kap) for k in ks]
    return ks[int(np.argmax(Wg))]

for (V,L,tau,kap) in [(1.0,1.2,0.7,8.0),(1.0,2.0,0.8,8.0),(1.0,1.0,0.6,15.0)]:
    print(f"\n--- V={V} L={L} tau={tau} kappa={kap} ---")
    for q in [0.8,0.95,1.0]:
        cs=np.linspace(0.05,0.95,19)
        k=np.array([kstar(q,c,V,L,tau,kap) for c in cs])
        # describe shape
        d=np.diff(k); sgn=np.sign(np.round(d,3)); sgn=sgn[sgn!=0]
        changes=int(np.sum(np.diff(sgn)!=0)) if len(sgn)>1 else 0
        amin=cs[int(np.argmin(k))]; amax=cs[int(np.argmax(k))]
        shape = "U" if (changes==1 and 0.1<amin<0.9) else ("hump" if (changes==1 and 0.1<amax<0.9) else ("incr" if k[-1]>k[0] else "decr"))
        print(f" q={q}: shape={shape} chg={changes} argmin_c={amin:.2f} argmax_c={amax:.2f}  k*: "+" ".join(f"{x:.2f}" for x in k[::3]))
