"""Side-by-side weights of two weight files (normalised units), with the tuner's default-frozen
features marked, and the per-feature sd of the CEM distribution at the end of a tune (dist.csv).

    python bot/tools/weight_diff.py bot/weights/normal.txt bot/results/survive/survive.txt [--dist bot/results/survive/survive.txt.dist.csv]
"""
import argparse

FROZEN = {"max_height", "wounds", "ready_repeat", "sz_single", "sz_double", "sz_triple",
          "lj_single", "lj_double", "lj_triple", "i_single", "i_double", "i_triple"}


def load(p):
    d, order = {}, []
    for l in open(p):
        f = l.split("#")[0].split()
        if len(f) == 2:
            try:
                d[f[0]] = float(f[1]); order.append(f[0])
            except ValueError:
                pass
    return d, order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--dist", default="")
    args = ap.parse_args()
    a, _ = load(args.a)
    b, order = load(args.b)
    sd = {}
    if args.dist:
        # rows "gen,mean,v1..vn" and "gen,cov,c11..cnn" (row-major); the last generation's diagonal
        rows = [l.rstrip("\n").split(",") for l in open(args.dist)][1:]
        cov = [r for r in rows if r[1] == "cov"]
        if cov:
            c = [float(x) for x in cov[-1][2:]]
            n = int(round(len(c) ** 0.5))
            if n == len(order):
                sd = {k: c[i * n + i] ** 0.5 for i, k in enumerate(order)}
    print("%-18s %9s %9s %9s %8s" % ("feature", "A", "B", "B - A", "final sd"))
    for k in order:
        x, y = a.get(k, 0.0), b[k]
        print("%-18s %9.2f %9.2f %+9.2f %8s %s" % (k, x, y, y - x, ("%.2f" % sd[k]) if k in sd else "", "frozen at 0" if k in FROZEN else ""))


if __name__ == "__main__":
    main()
