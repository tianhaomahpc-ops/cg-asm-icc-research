#!/usr/bin/env python3
"""Fig 12: (3) locate the mechanism in the spectrum + the A×B phase map.
(a) full spectra (CG Ritz values): over-count = high-end outliers (sASM removes them),
    global mode = low-end cluster (common). (b) over-count penalty BASIC/sASM kappa
    over (overlap × ICC level): grows with omega (low level). Reads phasespec_*.txt."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACC="#8C2D04"; STRUCT="#1F3A5F"

fig, ax = plt.subplots(1, 2, figsize=(14.4, 5.7))

# ---- (a) full spectra ----
a = ax[0]
combos = [("BASIC_exact","BASIC, exact",ACC,0.0),
          ("sASM_exact","sASM, exact",STRUCT,1.0),
          ("BASIC_icc","BASIC, ICC(0)",ACC,2.0),
          ("sASM_icc","sASM, ICC(0)",STRUCT,3.0)]
for tag,lab,col,y in combos:
    ev = np.loadtxt(f"phasespec_eigs_{tag}.txt")
    style = "o" if "exact" in tag else "D"
    a.scatter(ev, np.full_like(ev, y), s=18, color=col, marker=style,
              alpha=0.55, edgecolors="none")
    a.text(0.0042, y+0.16, lab, fontsize=9, color=col, fontweight="bold")
    a.annotate(f"$\\lambda_{{max}}$={ev.max():.2f}", (ev.max(), y), textcoords="offset points",
               xytext=(4,-12), fontsize=8.2, color=col)
a.set_xscale("log"); a.set_xlim(0.004, 7); a.set_ylim(-0.5, 3.7)
a.set_yticks([]); a.set_xlabel(r"eigenvalue of $M^{-1}A$ (CG Ritz values, log)")
a.axvspan(2.2, 7, color=ACC, alpha=0.05)
a.text(3.6, 3.45, "high end:\nover-count $\\hat N$", fontsize=8.4, color=ACC, ha="center")
a.axvspan(0.004, 0.02, color="#555", alpha=0.06)
a.text(0.0072, 3.45, "low end:\nglobal mode ($\\lambda_{min}$)", fontsize=8.0, color="#555", ha="center")
a.set_title("(a) where each factor lives in the spectrum (64², 4×4, O=2)\n"
            "sASM removes the high-end over-count; the low-end $\\lambda_{\\min}$ cluster is common",
            fontsize=10.0)

# ---- (b) phase map: over-count penalty BASIC/sASM kappa over (O × ICC level) ----
d = np.loadtxt("phasespec_phase.txt")  # useSASM level O kappa
Os = [1,2,4,6,8]; Ls = [0,1,2,-1]; Llab = ["ICC(0)","ICC(1)","ICC(2)","exact"]
ratio = np.zeros((len(Ls), len(Os)))
for li,L in enumerate(Ls):
    for oi,O in enumerate(Os):
        kb = d[(d[:,0]==0)&(d[:,1]==L)&(d[:,2]==O),3][0]
        ks = d[(d[:,0]==1)&(d[:,1]==L)&(d[:,2]==O),3][0]
        ratio[li,oi] = kb/ks
b = ax[1]
im = b.imshow(ratio, cmap="OrRd", aspect="auto", vmin=1.0, vmax=3.0)
b.set_xticks(range(len(Os))); b.set_xticklabels(Os)
b.set_yticks(range(len(Ls))); b.set_yticklabels(Llab)
b.set_xlabel("overlap $O$"); b.set_ylabel("inner accuracy (ICC fill level)")
for li in range(len(Ls)):
    for oi in range(len(Os)):
        b.text(oi, li, f"{ratio[li,oi]:.1f}", ha="center", va="center",
               fontsize=9, color="white" if ratio[li,oi]>2.0 else "#333", fontweight="bold")
cb = fig.colorbar(im, ax=b, fraction=0.046, pad=0.03); cb.set_label(r"$\kappa_{\rm BASIC}/\kappa_{\rm sASM}$  (over-count penalty)")
b.set_title("(b) the A×B map: the over-count penalty grows with $\\omega$ (lower fill)\n"
            "96², 6×6: ~2.9× at ICC(0); ~1 at exact -- sASM helps where the solve is inexact",
            fontsize=10.0)

fig.suptitle("Fig. 12  —  Locating the mechanism: (a) over-count sits at the high end of the spectrum (sASM removes it); "
             "(b) its penalty scales with inner inexactness $\\omega$",
             fontsize=11.4, fontweight="bold", color=STRUCT)
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig("fig12_phasespec.png", dpi=132)
print("wrote fig12_phasespec.png")
