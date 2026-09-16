import csv, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
L = lambda f: [{k: (float(v) if k not in ("cfg","fine","coarse") else v)
                for k, v in r.items()} for r in csv.DictReader(open(f))]

# ---------------- FIG I: what does the anomaly actually track? ---------------
R = L("fig1_contrast.csv"); g = lambda a,r,O: [x for x in R if x['angle']==a and x['ratio']==r and x['O']==O][0]
ratios = sorted({x['ratio'] for x in R})
fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6))
for a, mk in ((0.0,'o-'), (45.0,'s--')):
    amp = [g(a,r,3)['it_basic']/g(a,r,0)['it_basic'] for r in ratios]
    ax[0].semilogx(ratios, amp, mk, label="fiber %.0f$\\degree$"%a, lw=2, ms=7)
ax[0].axhline(1.0, color='k', lw=1)
ax[0].fill_between([1,200],1,1.6,color='crimson',alpha=.08)
ax[0].text(1.25,1.45,"ANOMALY\noverlap makes CG worse",color='crimson',fontsize=10,weight='bold')
ax[0].text(25,0.65,"classical\nSchwarz",color='green',fontsize=10,weight='bold')
ax[0].set_xlim(1,110); ax[0].set_xlabel("conductivity contrast  $r=\\sigma_\\ell/\\sigma_t$")
ax[0].set_ylabel("iter($O$=3) / iter($O$=0),  PC_ASM_BASIC")
ax[0].set_title("(a) the anomaly is a LOW-CONTRAST disease"); ax[0].legend(); ax[0].grid(alpha=.3)

for a, mk in ((0.0,'o-'), (45.0,'s--')):
    ax[1].semilogx(ratios,[g(a,r,3)['omega'] for r in ratios], mk, color='tab:red', lw=2, ms=7,
                   label="$\\omega$, fiber %.0f$\\degree$"%a)
ax[1].set_ylim(1.0,1.9); ax[1].set_xlabel("conductivity contrast $r$")
ax[1].set_ylabel("$\\omega=\\lambda_{\\max}(M_i^{-1}A_i)$  (IC(0) inexactness)")
ax[1].set_title("(b) FALSIFIED: $\\omega$ does not track it\n(45$\\degree$ flat; 0$\\degree$ ANTI-correlated)")
ax[1].legend(); ax[1].grid(alpha=.3)

for a, ls in ((0.0,'-'), (45.0,'--')):
    ax[2].semilogx(ratios,[g(a,r,3)['lmax_basic']/g(a,r,0)['lmax_basic'] for r in ratios],
                   ls, color='tab:red', marker='o', lw=2, label="$\\lambda_{max}$ penalty (over-count) %.0f$\\degree$"%a)
    ax[2].semilogx(ratios,[g(a,r,3)['lmin_basic']/g(a,r,0)['lmin_basic'] for r in ratios],
                   ls, color='tab:blue', marker='s', lw=2, label="$\\lambda_{min}$ payoff (overlap) %.0f$\\degree$"%a)
ax[2].set_xlabel("conductivity contrast $r$"); ax[2].set_ylabel("factor gained going $O$=0 $\\to$ $O$=3")
ax[2].set_title("(c) the real mechanism: a RACE\n$\\lambda_{max}$ penalty is $\\sigma$-blind, $\\lambda_{min}$ payoff is not")
ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
plt.tight_layout(); plt.savefig("figA_contrast_mechanism.png", dpi=135); plt.close()

# ---------------- FIG II: the five partition-of-unity families --------------
R2 = L("fig2_pou.csv")
fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6))
styles = [("basic","PC_ASM_BASIC",'grey','x:'),("flat","sASM $1/\\sqrt{m_k}$ (today)",'tab:red','o-'),
          ("graded","graded $q^{depth}$",'tab:orange','^-'),("ramp","NEW-A  $\\delta$-ramp",'tab:blue','s-'),
          ("harm","NEW-B  $\\sigma$-harmonic",'tab:green','D-')]
for k,(r,ang) in enumerate([(1,45.0),(100,45.0),(100,0.0)]):
    rows = sorted([x for x in R2 if x['ratio']==r and x['angle']==ang], key=lambda x:x['O'])
    Os = [x['O'] for x in rows]
    for key,lab,col,mk in styles:
        ax[k].plot(Os,[x['it_'+key] for x in rows],mk,color=col,label=lab,lw=2,ms=6)
    ax[k].plot(Os,[x['it_exact'] for x in rows],'--',color='k',lw=1.2,label='exact Cholesky (bound)')
    ax[k].set_xlabel("overlap $O$"); ax[k].set_ylabel("CG iterations")
    ax[k].set_title("$r=%d$, fiber $%.0f\\degree$"%(r,ang)); ax[k].grid(alpha=.3)
    if k==0: ax[k].legend(fontsize=8)
plt.suptitle("Fig B — five partition-of-unity families, IC(0) sub-solve, 36 subdomains", y=1.02)
plt.tight_layout(); plt.savefig("figB_pou_families.png", dpi=135, bbox_inches='tight'); plt.close()

# ---------------- FIG III: coarse space + scalability -----------------------
R4 = L("fig4_scale.csv"); R5 = L("fig5_enrich.csv")
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
for k,(r,ang) in enumerate([(1,45.0),(100,45.0)]):
    cfgs = sorted({x['cfg'] for x in R4 if x['ratio']==r and x['angle']==ang})
    for c in cfgs:
        rows = sorted([x for x in R4 if x['ratio']==r and x['angle']==ang and x['cfg']==c], key=lambda x:x['nsub'])
        ax[k].plot([x['nsub'] for x in rows],[x['it'] for x in rows],'o-',lw=2,ms=6,label=c.strip())
    rows = sorted([x for x in R5 if x['ratio']==r and x['angle']==ang and '3 vec' in x['cfg']], key=lambda x:x['nsub'])
    if rows:
        ax[k].plot([x['nsub'] for x in rows],[x['it'] for x in rows],'D-',color='purple',lw=2.5,ms=7,
                   label="enriched reuse (harm, 3 vec/sub)")
    ax[k].set_xlabel("number of subdomains (subdomain size fixed)"); ax[k].set_ylabel("CG iterations")
    ax[k].set_title("$r=%d$, fiber $%.0f\\degree$, $O$=2"%(r,ang)); ax[k].grid(alpha=.3); ax[k].legend(fontsize=8)
plt.suptitle("Fig C — weak scaling: the classical $1/m_k$ coarse basis collapses under anisotropy", y=1.02)
plt.tight_layout(); plt.savefig("figC_scaling.png", dpi=135, bbox_inches='tight'); plt.close()
print("wrote figA_contrast_mechanism.png  figB_pou_families.png  figC_scaling.png")
