import numpy as np
from scipy.stats import norm
Phi=norm.cdf; Pinv=norm.ppf
# resid(u) = -sqrt(a)u + b(z-1+Phi(u)) - sqrt(a+b)Pinv(tau); theta*=1-Phi(u)
U=np.linspace(-8,8,3201); PhiU=Phi(U)
def roots(a,b,z,tau):
    r=-np.sqrt(a)*U + b*(z-1+PhiU) - np.sqrt(a+b)*Pinv(tau)
    sc=np.where(np.sign(r[:-1])!=np.sign(r[1:]))[0]
    # linear-interp roots
    us=[U[i]-r[i]*(U[i+1]-U[i])/(r[i+1]-r[i]) for i in sc]
    return us
def thetastar(a,b,z,tau):
    us=roots(a,b,z,tau)
    return (1-Phi(us[0]) if len(us)==1 else None, len(us))

tau=0.4
# 0. beta->0 limit
for b in [1e-5,1e-2]:
    th,n=thetastar(20.0,b,0.5,tau); print(f"[limit] beta={b:.0e}: theta*={th:.4f} (expect {tau}), roots={n}")

# 1. uniqueness boundary across z
zs=np.linspace(-2,3,15)
for a in [5,10,20,40]:
    last=0
    for b in np.linspace(0.2,80,160):
        if all(thetastar(a,b,z,tau)[1]==1 for z in zs): last=b
        else: break
    print(f"[uniq] a={a}: max_b_unique~{last:.1f} | sqrt(2pi*a)={np.sqrt(2*np.pi*a):.1f} | 2pi*... a^2/(2pi)={a*a/(2*np.pi):.1f} | a*sqrt(2pi)... {a/np.sqrt(2*np.pi):.1f}")

# 2. cutoff & failure prob vs beta (unique region). masking=high beta, transparency=low beta.
def fail_prob(a,b,tau,tg,nz=1500):
    rng=np.random.default_rng(0); zs=tg+rng.normal(0,1/np.sqrt(b),nz)
    f=0
    for z in zs:
        th,n=thetastar(a,b,z,tau)
        if th is not None and tg<th: f+=1
    return f/nz
a=20.0
print(f"\n[cutoff vs beta] a={a}, tau={tau}, theta_g=0.45:")
for b in [0.2,1,3,8,20,40]:
    th0,n=thetastar(a,b,0.45,tau); fp=fail_prob(a,b,tau,0.45)
    print(f"  beta={b:5.1f} roots={n}: theta*(z=tg)={th0 if th0 is None else round(th0,3)}  P(fail|good)={fp:.3f}")
