"""The sweep writes the fibre as `1,1,1`, which splits into three CSV fields.
Rewrite it as `1|1|1` so the file is a well-formed 13-column CSV."""
import sys
src = sys.argv[1]
hdr = None
out = []
for i, line in enumerate(open(src)):
    line = line.rstrip("\n")
    if i == 0:
        hdr = line; ncol = len(hdr.split(",")); out.append(line); continue
    f = line.split(",")
    extra = len(f) - ncol
    if extra > 0:                       # fibre swallowed `extra` commas
        fib = "|".join(f[1:2 + extra])
        f = [f[0], fib] + f[2 + extra:]
    out.append(",".join(f))
open(src, "w").write("\n".join(out) + "\n")
print("normalised %s -> %d columns, %d rows" % (src, ncol, len(out) - 1))
