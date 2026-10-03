import numpy as np

# Triangular F_c on [0,theta_c]: F=theta^2/theta_c^2, f=2theta/theta_c^2
# Channels: theta_hat(k)=tau*k, s(k)=k. theta_c(c)=theta_max*c (FOSD in c).
def W(k, q, c, V, L, tau, theta_max):
    tc = theta_max*c
    that = tau*k
    rho = (that**2)/(tc**2) if that<=tc else 1.0
    s = k
    return q*c*(1-rho)*V - (1-q*c)*(1-s)*L

def kstar_closed(q,c,V,L,tau,theta_max):
    tc=theta_max*c
    return (1-q*c)*L*tc**2/(2*q*c*V*tau**2)

# --- 1. Theorem 3: closed form == argmax, interior, concave, tau*k*<=theta_c ---
V,L,tau,theta_max = 1.0, 1.2, 0.7, 1.4
ks=np.linspace(0,1,20001)
maxerr=0; n_interior=0; n_constraint_ok=0; ntest=0
rng=np.random.default_rng(3)
for _ in range(400):
    q=rng.uniform(0.4,1.0); c=rng.uniform(0.1,0.95)
    kc=kstar_closed(q,c,V,L,tau,theta_max)
    if not (0<kc<1): continue
    ntest+=1
    Wg=np.array([W(k,q,c,V,L,tau,theta_max) for k in ks])
    kgrid=ks[np.argmax(Wg)]
    maxerr=max(maxerr, abs(kgrid-kc))
    if 0<kc<1: n_interior+=1
    if tau*kc <= theta_max*c: n_constraint_ok+=1
print(f"[Thm 3] tested {ntest} interior draws; max|k*_closed - argmax_grid| = {maxerr:.4f} (grid res 5e-5)")
print(f"[Thm 3] interior 0<k*<1: {n_interior}/{ntest};  tau*k*<=theta_c holds: {n_constraint_ok}/{ntest}")

# --- 2. Prop 4: k*(c) U-shaped at fixed q (single interior min); 1-k* single-peaked ---
def kstar_clipped(q,c):
    return min(max(kstar_closed(q,c,V,L,tau,theta_max),0.0),1.0)
def ushape_count(q):
    cs=np.linspace(0.05,0.98,400)
    k=np.array([kstar_clipped(q,c) for c in cs])
    d=np.diff(k)
    sign=np.sign(d[np.abs(d)>1e-12])
    changes=np.sum(np.diff(sign)!=0)
    return changes, cs[np.argmin(k)], k.min(), k[0], k[-1]
for q in [0.7,0.85,0.95,1.0]:
    ch,cmin,kmin,k0,k1=ushape_count(q)
    print(f"[Prop 4] q={q}: sign-changes in k*(c)={ch} (U=>1), argmin c={cmin:.2f}, k*min={kmin:.2f}, k*(.05)={k0:.2f}, k*(.98)={k1:.2f}")

# --- 3. v2 consistency: c-region where k* mask-leaning (<0.5) is an interior interval ---
q=0.9
cs=np.linspace(0.05,0.98,400)
maskish=[(kstar_clipped(q,c)<0.5) for c in cs]
reg=[round(c,2) for c,m in zip(cs,maskish) if m]
print(f"[v2 consistency] q={q}: k*<0.5 (mask-leaning) for c in [{reg[0] if reg else 'none'},{reg[-1] if reg else 'none'}], interior={bool(reg) and reg[0]>0.05 and reg[-1]<0.98}")
