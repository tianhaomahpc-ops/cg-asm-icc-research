#!/usr/bin/env python3
"""analyze_stride.py -- how often should the u_e / torso solves run?

Reads every stride / spaced-window run in this directory and answers two
questions that are easy to conflate:

  cost      iterations and wall time, per solve AND over the same physical time
  accuracy  what the electrode trace loses when the torso is only solved every
            `stride` steps (measured against the full-rate reference trace)

Usage: python3 analyze_stride.py [fig_stride.png]
"""
import sys, re, glob, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

hdr_re = re.compile(r"#\s*asm_stride_test\s+n=(\d+)\s+nd=(\d+)\s+steps=(\d+)\s+"
                    r"dt=(\S+) ms \(T=(\S+) ms\) stride=(\d+).*?win=(\d+)")
wst_re = re.compile(r"wstride=(\d+)")
sys_re = re.compile(r"#\s*---\s*Sys(\d)\s*:\s*(\d+) solves")
row_re = re.compile(r"#\s+(.+?)\s+(\d+) iters\s+([\d.]+) /solve\s+\(\s*[-+][\d.]+%\)"
                    r"\s+([\d.]+) s\s+\(\s*[-+][\d.]+%\)\s+0-iter (\d+)/(\d+)")

def load(path):
    d = {"file": os.path.basename(path), "sys": {}, "wstride": 1}
    cur = None
    for line in open(path):
        if not line.startswith("#"):
            continue
        m = hdr_re.match(line)
        if m:
            d.update(n=int(m.group(1)), nd=int(m.group(2)), steps=int(m.group(3)),
                     dt=float(m.group(4)), T=float(m.group(5)),
                     stride=int(m.group(6)), win=int(m.group(7)))
            w = wst_re.search(line)
            if w: d["wstride"] = int(w.group(1))
            continue
        m = sys_re.match(line)
        if m:
            cur = int(m.group(1))
            d["sys"].setdefault(cur, {"nsolve": int(m.group(2))})
            continue
        if line.startswith("# --- pipeline") or line.startswith("# --- Sys2+Sys3"):
            cur = None; continue
        if cur is None:
            continue
        m = row_re.match(line)
        if m:
            d["sys"][cur][m.group(1).strip()] = dict(
                iters=int(m.group(2)), per=float(m.group(3)),
                time=float(m.group(4)), zero=int(m.group(5)))
    return d if "stride" in d and d["sys"] else None

runs = [r for r in (load(f) for f in sorted(glob.glob(os.path.join(HERE, "*.csv"))))
        if r and 2 in r["sys"]]
# the same (stride, wstride) can appear twice -- a parallel probe run and the
# sequential one.  Keep the sequential file: its wall times are uncontended.
best = {}
for r in runs:
    k = (r["stride"], r["wstride"])
    if k not in best or (not best[k]["file"].startswith("stride_")
                         and r["file"].startswith("stride_")):
        best[k] = r
runs = list(best.values())
if not runs:
    sys.exit("no finished runs found")

COLD, WARM = "cold(不做回收)", "warm(上一次解)"
CONS  = [k for k in runs[0]["sys"][2] if k.startswith("滑窗 m")][0]
def get(r, sy, key):
    for k, v in r["sys"][sy].items():
        if k.startswith(key): return v
    return None

base = [r for r in runs if r["wstride"] == 1]
base.sort(key=lambda r: r["stride"])
spaced = sorted([r for r in runs if r["wstride"] > 1], key=lambda r: r["wstride"])

h = base[0] if base else runs[0]
print(f"\n网格 {h['n']}^3 = {h['nd']} dof   dt = {h['dt']} ms   "
      f"sASM+IC(0), 4 子域, overlap 1, tol = 1e-8·‖b‖   窗口 m = {h['win']}\n")

# ================================ 1. the four configurations, side by side
by = {r["stride"]: r for r in base}
CFG = [("1", "每 0.01 ms", 1,   COLD, "不回收"),
       ("2", "每 1 ms",    100, COLD, "不回收"),
       ("3", "每 1 ms",    100, CONS, "+Fischer 滑窗"),
       ("4", "每 0.01 ms", 1,   CONS, "+Fischer 滑窗")]

r1 = None
print("=== 四个配置并排 (sASM+IC(0)+CG, 同样的 %g ms 物理时长) ===\n" % h["T"])
print(f"{'':<4}{'解算间隔':<12}{'初值':<16}{'求解次数':>9}"
      f"{'Sys2 迭代/次':>13}{'Sys3 迭代/次':>13}{'总迭代':>10}{'总时间 s':>10}{'零迭代':>8}")
rows = []
for tag, iv, st, key, kn in CFG:
    r = by.get(st)
    if not r: continue
    v2, v3 = get(r, 2, key), get(r, 3, key)
    if not (v2 and v3): continue
    ns = r["sys"][2]["nsolve"] + r["sys"][3]["nsolve"]
    it, tm = v2["iters"]+v3["iters"], v2["time"]+v3["time"]
    z = v2["zero"]+v3["zero"]
    rows.append((tag, iv, kn, ns, it, tm, v2, v3))
    print(f"{tag:<4}{iv:<12}{kn:<16}{ns:>9}{v2['per']:>13.1f}{v3['per']:>13.1f}"
          f"{it:>10}{tm:>10.2f}{100*z/ns:>7.0f}%")

if rows:
    b = rows[0]                                  # config 1 = the reference
    print(f"\n{'':<4}{'相对配置 1 (每 0.01 ms 冷启动)':<28}{'迭代':>10}{'时间':>10}")
    for tag, iv, kn, ns, it, tm, _, _ in rows:
        print(f"{tag:<4}{iv+' '+kn:<28}{b[4]/it:>9.1f}x{b[5]/tm:>9.1f}x")
    print("\n两两之间:")
    d = {r[0]: r for r in rows}
    if "1" in d and "2" in d:
        print(f"  少解 100 倍 (1->2):        迭代 {d['1'][4]/d['2'][4]:5.1f}x   "
              f"时间 {d['1'][5]/d['2'][5]:5.1f}x   "
              f"—— 每次求解的迭代数没变 ({d['1'][6]['per']:.1f} -> {d['2'][6]['per']:.1f})")
    if "2" in d and "3" in d:
        print(f"  在 1 ms 上加回收 (2->3):   迭代 {d['2'][4]/d['3'][4]:5.1f}x   "
              f"时间 {d['2'][5]/d['3'][5]:5.1f}x")
    if "1" in d and "4" in d:
        print(f"  在 0.01 ms 上加回收 (1->4):迭代 {d['1'][4]/d['4'][4]:5.1f}x   "
              f"时间 {d['1'][5]/d['4'][5]:5.1f}x")
    if "2" in d and "4" in d:
        print(f"  回收 vs 降频 (4 vs 2):     配置 4 仍比配置 2 贵 "
              f"{d['4'][4]/d['2'][4]:.0f}x 迭代 / {d['4'][5]/d['2'][5]:.0f}x 时间"
              f"  —— 回收补不回频率")
    r1 = (d["1"][4], d["1"][5], d["1"][3]) if "1" in d else None

# ============================================ 2. the whole pipeline, Sys1 included
print("\n=== 全流水线 (Sys1 每步都解,不受 stride 影响) ===")
print(f"{'':<42}{'Sys1':>10}{'Sys2':>9}{'Sys3':>9}{'总迭代':>10}{'总时间 s':>10}")
for lab, st, key in (("1. 每 0.01 ms,不回收", 1, COLD),
                     ("2. 每 1 ms,  不回收", 100, COLD),
                     ("3. 每 1 ms,  +Fischer", 100, CONS),
                     ("4. 每 0.01 ms,+Fischer", 1, CONS)):
    r = by.get(st)
    if not r: continue
    v = [get(r, q, key) for q in (1, 2, 3)]
    if any(x is None for x in v): continue
    it = sum(x["iters"] for x in v); tm = sum(x["time"] for x in v)
    print(f"{lab:<42}{v[0]['iters']:>10}{v[1]['iters']:>9}{v[2]['iters']:>9}"
          f"{it:>10}{tm:>10.2f}")

# ======================================== 3. the window-span law (stride sweep)
print("\n=== 解算密度 vs 回收效果(整段 %g ms 平均)===" % h["T"])
print("解得越密,相邻两次解越相关,窗口越有效 —— 单调,没有甜点。")
print(f"{'解算间隔':>10}{'窗口跨度':>10}{'Sys2 冷':>9}{'Sys2 窗':>9}{'降幅':>8}"
      f"{'Sys3 冷':>9}{'Sys3 窗':>9}{'降幅':>8}{'0迭代':>8}")
for r in base:
    span = r["win"]*r["stride"]*r["dt"]
    c2, w2 = get(r, 2, COLD), get(r, 2, CONS)
    c3, w3 = get(r, 3, COLD), get(r, 3, CONS)
    if not all((c2, w2, c3, w3)): continue
    z = w2["zero"]+w3["zero"]; ns = r["sys"][2]["nsolve"]+r["sys"][3]["nsolve"]
    print(f"{r['stride']*r['dt']:>8g}ms{span:>8g}ms{c2['per']:>9.1f}{w2['per']:>9.1f}"
          f"{100*(w2['per']/c2['per']-1):>7.0f}%{c3['per']:>9.1f}{w3['per']:>9.1f}"
          f"{100*(w3['per']/c3['per']-1):>7.0f}%{100*z/ns:>7.0f}%")

# ============================== 4. spaced window: solve every step, space the window
if spaced:
    SP  = [k for k in spaced[0]["sys"][2] if k.startswith("稀疏窗 m")][0]
    SPL = [k for k in spaced[0]["sys"][2] if k.startswith("稀疏窗 +")][0]
    print("\n=== 稀疏窗口(假设:窗口该覆盖更长物理时间)—— 被证伪 ===")
    print(f"每步都解,但窗口只收每第 w 次的解,跨度 = m·w·dt。")
    print(f"'连续窗' 是同一次运行里的 m 个相邻解(跨度只有 {h['win']*h['dt']:g} ms),它最好;")
    print("拉开窗口单调变差,连把最新解加回来也补不平。起作用的是快照离目标多近,不是覆盖多久。")
    print(f"{'w':>5}{'跨度':>9}{'Sys2 冷':>9}{'连续窗':>8}{'稀疏窗':>8}{'+最新':>8}"
          f"{'  |':>3}{'Sys3 冷':>9}{'连续窗':>8}{'稀疏窗':>8}{'+最新':>8}{'0迭代':>8}")
    for r in spaced:
        span = r["win"]*r["wstride"]*r["dt"]
        row = f"{r['wstride']:>5}{span:>7g}ms"
        for sy in (2, 3):
            c, w, s, sl = (get(r, sy, COLD), get(r, sy, CONS),
                           get(r, sy, SP), get(r, sy, SPL))
            if not all((c, w, s, sl)): row = None; break
            row += f"{c['per']:>9.1f}{w['per']:>8.1f}{s['per']:>8.1f}{sl['per']:>8.1f}"
            if sy == 2: row += f"{'  |':>3}"
        if row:
            z = sum(get(r, q, SPL)["zero"] for q in (2, 3))
            ns = sum(r["sys"][q]["nsolve"] for q in (2, 3))
            print(row + f"{100*z/ns:>7.0f}%")

# ================================================================ 5. accuracy
probe = os.path.join(HERE, "stride_probe_ref.csv")
acc = None
if os.path.exists(probe) and os.path.getsize(probe) > 1000:
  try:                      # the file may be mid-write if a run is in flight
    D = np.genfromtxt(probe, delimiter=",", names=True, invalid_raise=False)
    t = D["t_ms"]; E = np.vstack([D[f"e{i}"] for i in range(8)]).T
    keep = np.isfinite(t) & np.all(np.isfinite(E), axis=1)
    t, E = t[keep], E[keep]
    span = E.max(axis=0) - E.min(axis=0); span[span == 0] = 1.0
    print("\n=== 少解的代价:电极道只在解算时刻有值,中间要补 ===")
    print("(参考 = 每 0.01 ms 全率解;误差按每道峰峰值归一,8 道取最差)")
    print(f"{'解算间隔':>10}{'零阶保持 RMS':>15}{'零阶保持 max':>14}"
          f"{'线性插值 RMS':>15}{'线性插值 max':>14}")
    acc = []
    for st in (10, 25, 50, 100, 200):
        idx = np.arange(0, len(t), st)
        zoh = E[idx][np.clip(np.searchsorted(idx, np.arange(len(t)), "right")-1, 0, None)]
        lin = np.column_stack([np.interp(t, t[idx], E[idx, c]) for c in range(E.shape[1])])
        f = lambda X: (np.max(np.sqrt(np.mean(((X-E)/span)**2, axis=0))),
                       np.max(np.abs((X-E)/span)))
        rz, mz = f(zoh); rl, ml = f(lin)
        acc.append((st*h["dt"], rz, mz, rl, ml))
        print(f"{st*h['dt']:>8g}ms{rz:>15.2e}{mz:>14.2e}{rl:>15.2e}{ml:>14.2e}")
  except Exception as e:
    print("probe trace unreadable (a run may still be writing it):", e)
    acc = None

# ============================================ 5b. phase split (QRS vs plateau)
print("\n=== 相位拆分:回收的收益全在平台期 ===")
print("(激动期 = 波前扫过网格的 t < 20 ms;平台期 = 之后)")
print(f"{'解算间隔':>9}{'相位':>12}{'次数':>7}{'Sys2 冷':>9}{'Sys2 窗':>9}{'降幅':>7}"
      f"{'Sys3 冷':>9}{'Sys3 窗':>9}{'降幅':>7}{'零迭代':>8}")
for r in base:
    path = os.path.join(HERE, r["file"])
    try:
        A = np.genfromtxt(path, delimiter=",", names=True, comments="#",
                          invalid_raise=False)
    except Exception:
        continue
    t = A["t_ms"]
    ok = ~np.isnan(A["s2_v0"])
    for lo, hi, lab in ((0, 20, "激动期 QRS"), (20, 1e9, "平台期")):
        m = ok & (t >= lo) & (t < hi)
        if m.sum() == 0: continue
        c2, w2 = A["s2_v0"][m].mean(), A["s2_v2"][m].mean()
        c3, w3 = A["s3_v0"][m].mean(), A["s3_v2"][m].mean()
        z = ((A["s2_v2"][m] == 0).sum() + (A["s3_v2"][m] == 0).sum())/(2*m.sum())
        print(f"{r['stride']*r['dt']:>7g}ms{lab:>12}{m.sum():>7}{c2:>9.1f}{w2:>9.1f}"
              f"{100*(w2/c2-1):>6.0f}%{c3:>9.1f}{w3:>9.1f}{100*(w3/c3-1):>6.0f}%{100*z:>7.0f}%")

# =================================================== 5c. cost vs accuracy, one table
if acc and r1:
    amap = {a[0]: a for a in acc}
    print("\n=== 成本 vs 精度:同样的 %g ms 物理时长 ===" % h["T"])
    print(f"{'方案':<34}{'求解次数':>9}{'总迭代':>10}{'相对全率':>10}"
          f"{'波形 max 误差':>14}")
    for r in base:
        iv = r["stride"]*r["dt"]
        for key, lab in ((COLD, "不回收"), (CONS, "+Fischer 滑窗")):
            v = [get(r, q, key) for q in (2, 3)]
            if any(x is None for x in v): continue
            it = sum(x["iters"] for x in v)
            ns = sum(r["sys"][q]["nsolve"] for q in (2, 3))
            e = amap.get(iv)
            es = f"{e[4]*100:>12.2f}%" if e else f"{'(参考)':>13}"
            print(f"{('每 %g ms ' % iv)+lab:<34}{ns:>9}{it:>10}"
                  f"{it/r1[0]:>9.3f}x{es}")
    print("(波形误差 = 8 个电极道、线性插值重建、相对各道峰峰值的最大偏差)")

# ================================================================== 6. figure
if len(sys.argv) > 1 and base:
    import matplotlib; matplotlib.use("Agg")
    from matplotlib import font_manager
    try:
        font_manager.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
        matplotlib.rcParams["font.family"] = "WenQuanYi Zen Hei"
    except Exception: pass
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    sci = FuncFormatter(lambda v, p: "0" if v <= 0 else f"1e{int(round(np.log10(v)))}")
    plain = FuncFormatter(lambda v, p: f"{v:g}")

    fig, ax = plt.subplots(1, 3, figsize=(16.4, 5.4))
    fig.suptitle("胞外恢复 / 躯干求解该多久做一次? —— sASM+IC(0)+CG, "
                 f"{h['n']}^3={h['nd']} dof, 单域 dt={h['dt']} ms, tol=1e-8‖b‖",
                 fontsize=13, weight="bold", y=1.04)

    ms = [r["stride"]*r["dt"] for r in base]
    for sy, mk, nm in ((2, "o", "Sys2 $u_e$"), (3, "s", "Sys3 $u_T$")):
        ax[0].plot(ms, [get(r, sy, COLD)["per"] for r in base], mk+"-",
                   color="0.45", ms=5, label=f"{nm} 不回收")
        ax[0].plot(ms, [get(r, sy, CONS)["per"] for r in base], mk+"-",
                   color="#16a085", ms=5, label=f"{nm} +滑窗 m={h['win']}")
    if spaced:
        SPL = [k for k in spaced[0]["sys"][2] if k.startswith("稀疏窗 +")][0]
        ax[0].plot([r["wstride"]*r["dt"] for r in spaced],
                   [get(r, 2, SPL)["per"] for r in spaced], "*--",
                   color="#c0392b", ms=11,
                   label="Sys2 每步都解 + 稀疏窗(横轴=快照间隔)")
    ax[0].set_xscale("log"); ax[0].set_xlabel("解算间隔 (ms)")
    ax[0].set_ylabel("每次求解的 CG 迭代数")
    ax[0].set_title("① 每次求解要多少迭代", fontsize=11, weight="bold")
    ax[0].xaxis.set_major_formatter(plain)
    ax[0].grid(alpha=.3, which="both"); ax[0].legend(fontsize=7.6)

    for key, col, lab in ((COLD, "0.45", "不回收"), (CONS, "#16a085", "+滑窗")):
        ax[1].plot(ms, [sum(get(r, s, key)["iters"] for s in (2, 3)) for r in base],
                   "o-", color=col, ms=5, label=f"Sys2+Sys3 {lab}")
        ax[1].plot(ms, [sum(get(r, s, key)["iters"] for s in (1, 2, 3)) for r in base],
                   "^--", color=col, ms=5, alpha=.6, label=f"含 Sys1 全流水线 {lab}")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("解算间隔 (ms)"); ax[1].set_ylabel(f"{h['T']:g} ms 内的总迭代数")
    ax[1].xaxis.set_major_formatter(plain); ax[1].yaxis.set_major_formatter(sci)
    ax[1].set_title("② 同样物理时长下的总成本", fontsize=11, weight="bold")
    ax[1].grid(alpha=.3, which="both"); ax[1].legend(fontsize=7.6)

    if acc:
        a = np.array(acc)
        ax[2].loglog(a[:, 0], a[:, 1], "o-",  color="#c0392b", label="零阶保持 RMS")
        ax[2].loglog(a[:, 0], a[:, 3], "s-",  color="#1f6f8b", label="线性插值 RMS")
        ax[2].loglog(a[:, 0], a[:, 4], "s--", color="#1f6f8b", alpha=.6, label="线性插值 max")
        ax[2].axhline(1e-2, color="0.4", lw=.9, ls=":")
        ax[2].text(a[0, 0], 1.2e-2, "1% 峰峰值", fontsize=8, color="0.35")
        ax[2].xaxis.set_major_formatter(plain); ax[2].yaxis.set_major_formatter(sci)
        ax[2].legend(fontsize=8)
    ax[2].set_xlabel("解算间隔 (ms)"); ax[2].set_ylabel("电极道误差 (归一化)")
    ax[2].set_title("③ 少解的代价:电极波形误差", fontsize=11, weight="bold")
    ax[2].grid(alpha=.3, which="both")
    fig.savefig(sys.argv[1], dpi=140, bbox_inches="tight")
    print("\nwrote", sys.argv[1])
