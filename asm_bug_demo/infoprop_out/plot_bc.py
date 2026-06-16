#!/usr/bin/env python3
"""Task 1 comparison: all-Dirichlet vs all-Neumann vs mixed BC, item by item.
(a) over-count N_hat = lambda_max(BASIC,exact) is BC-INDEPENDENT (geometric).
(b,c) CG iterations: BC moves the low end (Neumann/mixed harder one-level), but the
two-level coarse space COLLAPSES the BC spread. Reads bc_suite.txt."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows=[l.split() for l in open("bc_suite.txt") if l.strip() and not l.startswith("#")]
D={}
for r in rows:
    dim=int(r[0]); bc=r[1]
    D[(dim,bc)]=dict(Nhat=int(r[2]),lmaxBE=float(r[3]),lmaxSI=float(r[9]),
                     itBI=int(r[11]),itSI=int(r[12]),it2=int(r[13]))
BCs=["Dirichlet","Neumann","Mixed1D5N"]; BClab=["all-Dirichlet","all-Neumann","mixed (1D+5N)"]
col={"Dirichlet":"#1F3A5F","Neumann":"#8C2D04","Mixed1D5N":"#117733"}

fig,ax=plt.subplots(1,3,figsize=(15.2,5.0))

# (a) lambda_max(BASIC,exact) = N_hat, across BC
a=ax[0]; dims=[1,2,3]; x=np.arange(3); w=0.26
for j,bc in enumerate(BCs):
    vals=[D[(d,bc)]["lmaxBE"] for d in dims]
    a.bar(x+(j-1)*w, vals, w, color=col[bc], label=BClab[j], edgecolor="k", lw=0.4)
for d in dims:
    a.text(d-1, D[(d,"Dirichlet")]["Nhat"]+0.15, f"$\\hat N$={D[(d,'Dirichlet')]['Nhat']}", ha="center", fontsize=10, fontweight="bold")
a.set_xticks(x); a.set_xticklabels(["1D","2D","3D"]); a.set_ylabel(r"$\lambda_{\max}$(BASIC, exact)")
a.set_title("(a) over-count $\\hat N$ is BC-INDEPENDENT\n$\\lambda_{\\max}$(BASIC,exact)$=\\hat N=$2/4/8 for every BC",fontsize=10.5)
a.legend(fontsize=8.6,loc="upper left"); a.grid(alpha=0.3,axis="y"); a.set_ylim(0,10)

# (b,c) iterations for 2D and 3D
for col_i,dim in enumerate([2,3]):
    b=ax[1+col_i]; methods=["itBI","itSI","it2"]; mlab=["BASIC\nICC","sASM\nICC","two-level\nsASM"]
    xm=np.arange(3); w=0.26
    for j,bc in enumerate(BCs):
        vals=[D[(dim,bc)][m] for m in methods]
        bars=b.bar(xm+(j-1)*w, vals, w, color=col[bc], label=BClab[j], edgecolor="k", lw=0.4)
        for xi,v in zip(xm+(j-1)*w,vals): b.text(xi,v+1.5,str(v),ha="center",fontsize=7.5,color=col[bc],fontweight="bold")
    b.set_xticks(xm); b.set_xticklabels(mlab,fontsize=9)
    b.set_ylabel("CG iterations to $10^{-8}$"); b.set_title(
        f"({'b' if dim==2 else 'c'}) {dim}D iterations: BC spread (one-level) $\\to$ collapses (two-level)",fontsize=10.5)
    if col_i==0: b.legend(fontsize=8.6,loc="upper right")
    b.grid(alpha=0.3,axis="y")
    b.annotate("two-level: BC\ndifferences gone",xy=(2,D[(dim,'Neumann')]['it2']),
               xytext=(1.4,D[(dim,'Neumann')]['itBI']*0.7),fontsize=8.4,color="#444",
               arrowprops=dict(arrowstyle="->",color="#444",lw=1))

fig.suptitle("Task 1 — interpretability vs boundary conditions (unit geometry): the over-count is geometric (BC-free); "
             "the BC perturbs $\\lambda_{\\min}$, which the coarse space neutralizes",
             fontsize=11.6,fontweight="bold",color="#1F3A5F")
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig("fig_bc_compare.png",dpi=130)
print("wrote fig_bc_compare.png")
print("2D iters one-level sASM:", [D[(2,bc)]['itSI'] for bc in BCs], "-> two-level:", [D[(2,bc)]['it2'] for bc in BCs])
