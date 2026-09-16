import anisopu as ap, numpy as np, csv
rows=[]
for r, ang in [(1,45.0),(100,45.0),(100,0.0)]:
    print("\n### r=%d fiber=%.0fdeg  (O=2, subdomain size fixed 20x20)"%(r,ang))
    Ps=[2,3,4,5,6]; res={}
    for P in Ps:
        sig=ap.sigma_tensor(r,ang); A,b,coords,free,n1=ap.build_fem(20*P,sig)
        own=ap.box_partition(coords,P,P); sets=ap.overlap_sets(A,own,P*P,2)
        Alocs=[A[i][:,i].tocsr() for i,_ in sets]
        for name,fine,cb in [("two-level reuse   (harm, 1 vec/sub)","harm","plain"),
                             ("enriched reuse    (harm, 3 vec/sub)","harm","enrich")]:
            w=ap.build_weights(fine,sets,Alocs,A.shape[0],2)
            M=ap.ASM(A,sets,w)
            R0 = ap.coarse_basis("harm",sets,Alocs,A.shape[0],2) if cb=="plain" \
                 else ap.coarse_basis_enriched("harm",sets,Alocs,A.shape[0],2,coords,ang)
            M.set_coarse(A,R0)
            _,it,(lo,hi)=ap.pcg(A,b,M,rtol=1e-6,maxit=6000)
            res.setdefault(name,[]).append(it)
            rows.append(dict(ratio=r,angle=ang,P=P,nsub=P*P,cfg=name,ncoarse=R0.shape[0],it=it,lmin=lo,lmax=hi))
    print("  #subdomains:                        "+"".join("%8d"%(P*P) for P in Ps)+"     growth")
    for k,v in res.items():
        print("  %-36s"%k+"".join("%8d"%x for x in v)+"     %.2fx"%(v[-1]/v[0]))
with open("fig5_enrich.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("\nwrote fig5_enrich.csv")
