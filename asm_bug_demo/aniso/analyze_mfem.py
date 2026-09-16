"""Analyse the MFEM/PETSc/MPI sigma-tensor sweep (aniso_nx*_n*_L*.csv).

Separates the two competing explanations of the overlap anomaly:
  H1  the ICC sub-solve inexactness omega drives it
  H2  the race between the lambda_max over-count penalty and the lambda_min
      overlap payoff drives it
by correlating the anomaly amplitude against each candidate.
"""
import csv, sys, math

path = sys.argv[1] if len(sys.argv) > 1 else "aniso_nx48_n4_L0.csv"
R = []
for row in csv.DictReader(open(path)):
    d = {}
    for k, v in row.items():
        if k in ("fiber", "method"):
            d[k] = v
        else:
            try: d[k] = float(v)
            except Exception: d[k] = float("nan")
    R.append(d)

g = lambda m, r, O: next((x for x in R if x["method"] == m and x["ratio"] == r and x["O"] == O), None)
ratios = sorted({x["ratio"] for x in R})
Os = sorted({x["O"] for x in R})
methods = ["BASIC", "sASM", "ramp", "harm"]

print("=" * 96)
print("(1) ITERATIONS  (MFEM 3D P1 tet, METIS, 4 MPI ranks, ICC(0), fibre %s)" % R[0]["fiber"])
print("=" * 96)
for m in methods:
    print("\n  %-6s   " % m + "".join("  O=%-6d" % O for O in Os) + "   amp=it(O3)/it(O0)")
    for r in ratios:
        row = [g(m, r, O) for O in Os]
        its = [x["iter"] if x else float("nan") for x in row]
        amp = its[-1] / its[0] if its[0] == its[0] else float("nan")
        mark = "  <-- ANOMALY" if amp > 1.0 else ""
        print("  r=%-6.0f " % r + "".join("%9d" % i for i in its) + "        %5.2f%s" % (amp, mark))

print()
print("=" * 96)
print("(2) MECHANISM: what does the anomaly amplitude actually track?  (BASIC, O=0 -> O=3)")
print("=" * 96)
print("    r   | lmax penalty | lmin payoff |  kappa net  | iter ratio |  omega   | kappa_sub")
rows = []
for r in ratios:
    a0, a3 = g("BASIC", r, 0), g("BASIC", r, 3)
    fx = a3["lmax"] / a0["lmax"]; fn = a3["lmin"] / a0["lmin"]
    ir = a3["iter"] / a0["iter"]
    print("  %5.0f |    %5.2f x    |   %5.2f x   |   %5.2f x   |   %5.2f x  |  %6.4f  |  %8.1f"
          % (r, fx, fn, fx / fn, ir, a3["omega"], a3["kappa_sub"]))
    rows.append((ir, fx, fn, a3["omega"], a3["kappa_sub"]))

def corr(u, v):
    mu, mv = sum(u)/len(u), sum(v)/len(v)
    num = sum((a-mu)*(b-mv) for a, b in zip(u, v))
    den = math.sqrt(sum((a-mu)**2 for a in u) * sum((b-mv)**2 for b in v))
    return num/den if den else float("nan")

it = [x[0] for x in rows]
print("\n  Pearson correlation of the anomaly amplitude with each candidate driver:")
print("     lambda_max amplification (over-count) : %+.3f" % corr(it, [x[1] for x in rows]))
print("     lambda_min amplification (overlap)    : %+.3f" % corr(it, [x[2] for x in rows]))
print("     omega  = lmax(M_i^-1 A_i)             : %+.3f" % corr(it, [x[3] for x in rows]))
print("     kappa_sub = cond(M_i^-1 A_i)          : %+.3f" % corr(it, [x[4] for x in rows]))

print()
print("=" * 96)
print("(3) WEIGHT COMPARISON: does the sigma-aware PU beat multiplicity scaling?")
print("=" * 96)
print("    r   |  O  | BASIC | sASM  | ramp  | harm  | harm vs sASM | sASM vs BASIC")
for r in ratios:
    for O in Os:
        if O == 0: continue
        b, s, p, h = (g(m, r, O) for m in methods)
        print("  %5.0f |  %d  | %5d | %5d | %5d | %5d |   %+6.1f%%    |   %+6.1f%%%s"
              % (r, O, b["iter"], s["iter"], p["iter"], h["iter"],
                 100*(h["iter"]-s["iter"])/s["iter"],
                 100*(s["iter"]-b["iter"])/b["iter"],
                 "  <-- sASM LOSES to BASIC" if s["iter"] > b["iter"] else ""))
    print()

print("=" * 96)
print("(4) lambda_max CAP: sqrt-PU theory says lambda_max(weighted) <= omega.  Does it hold?")
print("=" * 96)
print("    r   |  O  | omega  | lmax BASIC | lmax sASM | lmax harm | sASM/omega | harm/omega")
for r in ratios:
    for O in Os:
        if O == 0: continue
        b, s, h = g("BASIC", r, O), g("sASM", r, O), g("harm", r, O)
        om = b["omega"]
        print("  %5.0f |  %d  | %6.3f |   %7.3f  |  %7.3f  |  %7.3f  |   %5.2f    |   %5.2f"
              % (r, O, om, b["lmax"], s["lmax"], h["lmax"], s["lmax"]/om, h["lmax"]/om))
    print()
