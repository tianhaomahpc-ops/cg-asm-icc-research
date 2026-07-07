import numpy as np
np.set_printoptions(precision=3, suppress=True)
# 更像真实 S2 的谱:5 个分散的小 λ(慢模) + 20 个紧密聚集在 ~1 的 λ(快模 bulk)
np.random.seed(0)
small=np.array([0.005,0.01,0.02,0.05,0.1])          # 5 个慢模
bulk =np.linspace(0.95,1.05,20)                      # 20 个紧密快模
d=np.concatenate([small,bulk]); n=len(d)
A=np.diag(d); xs=np.ones(n); b=A@xs
print(f"谱: 5个小λ={small}  + 20个聚集λ在[0.95,1.05]")
print(f"条件数 = {d.max()/d.min():.0f}\n")

def cg(A,b,x0,tol=1e-8,nmax=200):
    x=x0.astype(float).copy(); r=b-A@x; p=r.copy(); rr=r@r; h=[np.sqrt(rr)]
    for k in range(nmax):
        Ap=A@p; a=rr/(p@Ap); x=x+a*p; r=r-a*Ap; rr2=r@r; h.append(np.sqrt(rr2))
        if np.sqrt(rr2)<tol*h[0]: break
        p=r+(rr2/rr)*p; rr=rr2
    return x,h

_,hp=cg(A,b,np.zeros(n))
print(f"普通 CG:              {len(hp)-1} 步")

# deflation: W = 5 个最慢模
m=5; W=np.zeros((n,m))
for i in range(m): W[i,i]=1
E=W.T@A@W
y=W@np.linalg.solve(E, W.T@b)     # 粗解:5个慢模一次性解精确
_,hd=cg(A,b,y)
print(f"deflated CG(摘掉5个慢模): {len(hd)-1} 步")
print(f"\n→ {len(hp)-1} 步  减到  {len(hd)-1} 步   (约 {(len(hp)-1)/(len(hd)-1):.1f}×)")
# 存残差历史给画图
np.save('/tmp/hp.npy',np.array(hp)); np.save('/tmp/hd.npy',np.array(hd))
