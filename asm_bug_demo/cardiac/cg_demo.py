import numpy as np
np.set_printoptions(precision=4, suppress=True)
# CG teaching demo: per-mode error trace on diagonal systems (mode = coordinate axis).
def cg_trace(A, b, x0, tag, nmax):
    print(f"\n===== {tag} =====")
    x=x0.astype(float).copy(); r=b-A@x; p=r.copy(); rr=r@r
    xstar=np.linalg.solve(A,b); lam=np.sort(np.linalg.eigvalsh(A))
    print("eigenvalues (per-mode lambda):", lam, " x* =", xstar)
    print(f"{'k':>2} {'||r||':>10}   x                      per-mode |x_i - x*_i|")
    print(f"{0:>2} {np.sqrt(rr):>10.4f}   {x}   {np.abs(x-xstar)}")
    for k in range(1,nmax+1):
        Ap=A@p; a=rr/(p@Ap); x=x+a*p; r=r-a*Ap; rr2=r@r
        print(f"{k:>2} {np.sqrt(rr2):>10.4f}   {x}   {np.abs(x-xstar)}")
        if np.sqrt(rr2)<1e-9: break
        p=r+(rr2/rr)*p; rr=rr2
# ex1: A=diag(1,5,25); modes=axes; fast mode (lam=25) killed first, slow (lam=1) last; exact in 3 steps
cg_trace(np.diag([1.,5.,25.]), np.array([1.,5.,25.]), np.zeros(3),
         "ex1: A=diag(1,5,25) -- big lambda dies first, small last", 4)
# ex2: 2 small eigenvalues + 10 clustered -> fast head (clustered modes) + slow tail (2 small modes)
d=np.concatenate([[0.2,0.6], np.linspace(8,10,10)]); A2=np.diag(d); xs=np.ones(12); b2=A2@xs
print("\n===== ex2: 2 small (slow) + 10 clustered (fast) -- head+tail =====")
print("eigenvalues:", np.sort(d))
x=np.zeros(12); r=b2-A2@x; p=r.copy(); rr=r@r
print(f"{'k':>2} {'||r||':>10}  slow1    slow2    fast(avg)")
for k in range(0,13):
    if k>0:
        Ap=A2@p; a=rr/(p@Ap); x=x+a*p; r=r-a*Ap; rr2=r@r; p=r+(rr2/rr)*p; rr=rr2
    e=np.abs(x-xs); print(f"{k:>2} {np.sqrt(r@r):>10.4f}   {e[0]:7.4f}  {e[1]:7.4f}  {e[2:].mean():9.4f}")
    if np.sqrt(r@r)<1e-9: break
