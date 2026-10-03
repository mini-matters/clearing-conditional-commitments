import numpy as np
from scipy.stats import beta
V1,Lp=1.0,0.5; tau=Lp/(V1+Lp)
V,L=1.0,0.4
def Fc(c,tau,k=8.0):
    m=min(max(c,1e-3),1-1e-3); a,b=m*k,(1-m)*k
    return beta.cdf(tau,a,b)
def mstar(c):
    rho=Fc(c,tau); return L/(L+rho*V)
# For fixed q, is masking region {c: q*c >= mstar(c)} an interval / what shape?
for q in [0.6,0.8,0.9,1.0]:
    cs=np.linspace(0.01,0.99,99)
    mask=[(q*c>=mstar(c)) for c in cs]
    region=[round(c,2) for c,m in zip(cs,mask) if m]
    if region:
        print(f"q={q}: masking optimal for c in [{region[0]}, {region[-1]}], contiguous={region==[round(c,2) for c in cs if region[0]-1e-9<=c<=region[-1]+1e-9]}, count={len(region)}")
    else:
        print(f"q={q}: masking never optimal")
# show mstar(c) and q*c crossing for q=0.9
print("\n c    q*c(.9)  mstar(c)  mask?")
for c in [0.1,0.3,0.5,0.6,0.7,0.8,0.9,0.95]:
    print(f"{c:.2f}  {0.9*c:.3f}    {mstar(c):.3f}    {0.9*c>=mstar(c)}")
