"""Final figure: the MFEM/PETSc/MPI 3D verdict."""
import csv, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

F=[r for r in csv.DictReader(open("twolevel_fair_nx48_L0.csv"))]
T=[r for r in csv.DictReader(open("twolevel_time_nx48.csv"))]
C=[r for r in csv.DictReader(open("aniso_nx48_n4_L0.csv"))]
Ps=[2,4,8,16]
st=[("BASIC",'grey','x:'),("sASM",'tab:red','o-'),("harm",'tab:green','D-'),
    ("harm+coarse",'tab:blue','s-'),("mult+coarse",'tab:orange','^--')]
it=lambda m,P,r: next((int(y["iter"]) for y in F if y["method"]==m and int(y["ranks"])==P and float(y["ratio"])==r),None)

fig,ax=plt.subplots(1,4,figsize=(21,4.7))

# (a) the anomaly, BASIC, vs contrast
g=lambda m,r,O: next(y for y in C if y["method"]==m and float(y["ratio"])==r and float(y["O"])==O)
rr=sorted({float(x["ratio"]) for x in C})
ax[0].semilogx(rr,[float(g("BASIC",r,3)["iter"])/float(g("BASIC",r,0)["iter"]) for r in rr],
               'o-',color='tab:red',lw=2.4,ms=8)
ax[0].axhline(1,color='k',lw=1); ax[0].fill_between([1,200],1,1.75,color='crimson',alpha=.08)
ax[0].set_xlim(1,110); ax[0].set_ylim(0.9,1.8)
ax[0].set_xlabel("contrast $r=\\sigma_\\ell/\\sigma_t$"); ax[0].set_ylabel("iter($O$=3)/iter($O$=0)")
ax[0].set_title("(a) the anomaly never flips in 3D\nPC_ASM_BASIC, 4 ranks"); ax[0].grid(alpha=.3)

# (b,c) scalability at equal TRUE residual
for k,r in enumerate((1.0,100.0)):
    a=ax[1+k]
    for m,c,mk in st:
        v=[it(m,P,r) for P in Ps]
        if any(v): a.plot(Ps,v,mk,color=c,lw=2.2,ms=7,label=m)
    a.set_xscale('log',base=2); a.set_xticks(Ps); a.set_xticklabels(Ps)
    a.set_xlabel("MPI ranks = subdomains (problem size fixed)")
    a.set_ylabel("CG iterations, equal TRUE residual")
    a.set_title("(%s) $r=%g$: only the two-level\nmethods go DOWN with $P$"%("bc"[k],r))
    a.grid(alpha=.3); a.legend(fontsize=8)

# (d) iterations vs wall time
tt=lambda m,P,r,f: next((float(y[f]) for y in T if y["method"]==m and int(y["ranks"])==P and float(y["ratio"])==r),None)
for m,c,mk in st:
    xs=[tt(m,P,r,"solve_s") for r in (1.0,100.0) for P in (8,16)]
    ys=[tt(m,P,r,"iter")    for r in (1.0,100.0) for P in (8,16)]
    if all(x is not None for x in xs):
        ax[3].plot(xs,ys,mk[0]+' ',color=c,ms=11,label=m,linestyle='None')
ax[3].set_xlabel("solve-only wall time (s), best of 3")
ax[3].set_ylabel("CG iterations")
ax[3].set_title("(d) iterations are not time:\nthe coarse level costs ~2x per step")
ax[3].grid(alpha=.3); ax[3].legend(fontsize=8)
plt.tight_layout(); plt.savefig("aniso/figD_mfem_final.png",dpi=135)
print("wrote aniso/figD_mfem_final.png")
