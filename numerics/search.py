import numpy as np
from scipy.stats import beta
# Does a nonempty masking region exist? Scan structural params.
def Fc(c,tau,k):
    m=min(max(c,1e-3),1-1e-3); a,b=m*k,(1-m)*k
    return beta.cdf(tau,a,b)
rng=np.random.default_rng(1)
found=[]
for _ in range(40000):
    V1=1.0; Lp=rng.uniform(0.1,5); tau=Lp/(V1+Lp)
    V=rng.uniform(0.5,3); L=rng.uniform(0.05,3); k=rng.uniform(2,40)
    # scan q,c
    best=None
    for q in np.linspace(0.3,1,15):
        for c in np.linspace(0.05,0.99,30):
            rho=Fc(c,tau,k); mu=q*c
            D=rho*mu*V-(1-mu)*L
            if D>0:
                if best is None or D>best[0]: best=(D,q,c,rho,mu)
    if best: found.append((Lp,tau,V,L,k,*best))
print(f"masking-nonempty param draws: {len(found)}/40000")
if found:
    import statistics as st
    # where does masking win? look at c, mu, rho at the winning point
    cs=[f[7] for f in found]; mus=[f[9] for f in found]; rhos=[f[8] for f in found]; taus=[f[1] for f in found]; Ls=[f[3] for f in found]
    print(f"winning c:   min={min(cs):.2f} med={st.median(cs):.2f} max={max(cs):.2f}")
    print(f"winning mu:  min={min(mus):.2f} med={st.median(mus):.2f} max={max(mus):.2f}")
    print(f"winning rho: min={min(rhos):.2f} med={st.median(rhos):.2f} max={max(rhos):.2f}")
    print(f"tau:         min={min(taus):.2f} med={st.median(taus):.2f} max={max(taus):.2f}")
    print(f"L(social):   min={min(Ls):.2f} med={st.median(Ls):.2f} max={max(Ls):.2f}")
    # is masking region an interval in c for a representative winner? pick one
    Lp,tau,V,L,k=found[len(found)//2][:5]
    cs2=np.linspace(0.01,0.99,99)
    for q in [0.9,1.0]:
        mask=[(q*c>=L/(L+Fc(c,tau,k)*V)) for c in cs2]
        reg=[round(c,2) for c,m in zip(cs2,mask) if m]
        print(f"  rep(Lp={Lp:.2f},V={V:.2f},L={L:.2f},k={k:.1f}) q={q}: mask c-range {reg[:1]+reg[-1:] if reg else 'empty'} n={len(reg)}")
