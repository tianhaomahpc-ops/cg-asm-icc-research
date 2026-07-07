import matplotlib, numpy as np
matplotlib.use("Agg")
from matplotlib import font_manager
try:
    font_manager.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
    matplotlib.rcParams["font.family"]="WenQuanYi Zen Hei"
except Exception: pass
matplotlib.rcParams["axes.unicode_minus"]=False
import matplotlib.pyplot as plt
np_=[2,4,8]
fine=[70,78,76]; nic=[67,68,59]; rich=[51,48,40]
fig,ax=plt.subplots(figsize=(8.5,5.6))
ax.plot(np_,fine,"o-",color="#c0392b",lw=2.2,ms=8,label="只有细层 (bjacobi+ICC) — 无粗空间")
ax.plot(np_,nic,"s-",color="#e67e22",lw=2.2,ms=8,label="+ Nicolaides 粗空间 (每子域1个常数, dim=np)")
ax.plot(np_,rich,"^-",color="#16a085",lw=2.2,ms=8,label="+ 丰富粗空间 {1,x,y,z}/子域 (dim=4np)")
for xs,ys,c in [(np_,fine,"#c0392b"),(np_,nic,"#e67e22"),(np_,rich,"#16a085")]:
    for a,bv in zip(xs,ys): ax.text(a,bv+1.5,str(bv),ha="center",color=c,fontsize=9,weight="bold")
ax.set_title("真实 S2:粗空间/deflation 塌陷 67 步地板;粗空间越接近真慢模,塌得越深\n"
             "而且两层迭代随子域数不涨反降 → 强扩展友好",fontsize=11.5,weight="bold")
ax.set_xlabel("子域数 np"); ax.set_ylabel("CG 迭代数 (到 rtol 1e-8)")
ax.set_xticks(np_); ax.set_ylim(0,85); ax.grid(alpha=0.3); ax.legend(fontsize=9.5,loc="lower left")
ax.annotate("回收(Fischer)学到真慢模 → ~15 (−77%)\n几何粗空间便宜但粗糙 → 只到 40",
            xy=(8,40),xytext=(4.3,15),fontsize=9,color="#16a085",
            arrowprops=dict(arrowstyle="->",color="#16a085"))
fig.savefig("fig_deflate_real.png",dpi=140,bbox_inches="tight"); print("wrote fig_deflate_real.png")
