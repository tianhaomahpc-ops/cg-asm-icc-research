import matplotlib, numpy as np
matplotlib.use("Agg")
from matplotlib import font_manager
try:
    font_manager.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
    matplotlib.rcParams["font.family"]="WenQuanYi Zen Hei"
except Exception: pass
matplotlib.rcParams["axes.unicode_minus"]=False
import matplotlib.pyplot as plt
hp=np.load('/tmp/hp.npy'); hd=np.load('/tmp/hd.npy')
small=np.array([0.005,0.01,0.02,0.05,0.1]); bulk=np.linspace(0.95,1.05,20)

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,5.2))
fig.suptitle("deflation / 粗空间:把几个慢模摘掉,CG 只剩快模 → 几步收敛", fontsize=14, weight="bold", y=0.99)

# spectrum
ax1.plot(small,np.zeros_like(small),"v",color="#c0392b",ms=13,label="5 个小λ = 慢模(要摘掉的)")
ax1.plot(bulk,np.zeros_like(bulk),"|",color="#2980b9",ms=16,label="20 个聚集λ = 快模 bulk")
ax1.axvspan(-0.02,0.13,color="#fadbd8",alpha=0.5)
for s in small: ax1.annotate("",xy=(s,0.15),xytext=(s,0.0),arrowprops=dict(arrowstyle="->",color="#c0392b",lw=1.3))
ax1.text(0.05,0.2,"deflation:\n把这5个直接\n解掉(摘走)",color="#c0392b",fontsize=10,weight="bold",ha="center")
ax1.text(1.0,0.05,"CG 只需处理这团",color="#2980b9",fontsize=10,ha="center")
ax1.set_title("① 谱:5 个分散小λ(慢) + 20 个聚集λ(快)",fontsize=11)
ax1.set_xlabel("特征值 λ"); ax1.set_yticks([]); ax1.set_xlim(-0.05,1.15); ax1.set_ylim(-0.1,0.35)
ax1.legend(fontsize=9,loc="upper right")

# residual curves
ax2.semilogy(range(len(hp)),hp/hp[0],"o-",color="#c0392b",lw=2,ms=5,label=f"普通 CG:{len(hp)-1} 步(有慢尾)")
ax2.semilogy(range(len(hd)),hd/hd[0],"s-",color="#16a085",lw=2,ms=6,label=f"deflated CG:{len(hd)-1} 步")
ax2.axhline(1e-8,color="0.6",ls=":",lw=1); ax2.text(1,1.4e-8,"收敛线",fontsize=8,color="0.5")
ax2.set_title("② 残差:摘掉5个慢模 → 16步变6步(≈2.7×)",fontsize=11)
ax2.set_xlabel("CG 迭代数"); ax2.set_ylabel("相对残差 (log)")
ax2.legend(fontsize=10,loc="upper right"); ax2.grid(alpha=0.3); ax2.set_ylim(1e-9,3)
fig.savefig("fig_deflation.png",dpi=140,bbox_inches="tight"); print("wrote fig_deflation.png")
