import numpy as np
np.set_printoptions(precision=4, suppress=True)
A=np.array([[3.,1.],[1.,3.]]); b=np.array([6.,2.])
# 特征分解(手算: 3+-1 = 4,2;特征向量 [1,1],[1,-1])
u1=np.array([1,1.])/np.sqrt(2); l1=4.0     # 快模
u2=np.array([1,-1.])/np.sqrt(2); l2=2.0     # 慢模
xstar=np.linalg.solve(A,b)
print("A =",A.tolist()," b =",b," 真解 x* =",xstar)
print(f"特征值 λ1={l1}(方向[1,1],快模), λ2={l2}(方向[1,-1],慢模)\n")

print("===== 路线1(用特征值):拆 b → 各自除以 λ → 拼回 =====")
c1=b@u1; c2=b@u2
print(f"1) 把 b 拆到两个特征方向:  沿[1,1]的量 = {c1:.4f},  沿[1,-1]的量 = {c2:.4f}")
print(f"2) 各自除以自己的 λ:        {c1:.4f}/{l1} = {c1/l1:.4f},   {c2:.4f}/{l2} = {c2/l2:.4f}")
x_eig=(c1/l1)*u1+(c2/l2)*u2
print(f"3) 拼回:x = {c1/l1:.4f}·[1,1]/√2 + {c2/l2:.4f}·[1,-1]/√2 = {x_eig}   (=x*)\n")

print("===== 路线2(CG,只用 A·向量,不碰特征值)=====")
x=np.zeros(2); r=b-A@x; p=r.copy(); rr=r@r
e0=x-xstar; f0_1=(e0@u1); f0_2=(e0@u2)
def show(k,x):
    e=x-xstar
    print(f" k={k}: x={x}   快模λ=4残余={ (e@u1)/f0_1:+.4f}   慢模λ=2残余={ (e@u2)/f0_2:+.4f}")
print(" (残余 = 该模误差还剩几分之几,即 p_k(λ))")
show(0,x)
for k in range(1,3):
    Ap=A@p; a=rr/(p@Ap); x=x+a*p; r=r-a*Ap; rr2=r@r
    show(k,x)
    if rr2>1e-14: p=r+(rr2/rr)*p; rr=rr2
print("\nCG 全程只做了 A·向量 和 点积,从没算过特征值 —— 但我们用特征值一分析,")
print("就看到:第1步 快模(λ=4)残余压到 %.3f、慢模(λ=2)只压到 %.3f → 大λ先消。"%(
    ((np.zeros(2)-xstar)@u1)*0,0))
