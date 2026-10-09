"""Paired comparison of two players on the same seeds: per-game attack over the full piece
budget (a top-out counts as 0 for the pieces not played), mean difference with its
standard error, a paired t statistic, and games won / lost / tied.

    python attentris/paired.py A.games "label A substring" B.games "label B substring"
"""
import json
import math
import sys


def blocks(path):
    out, cur = {}, None
    for line in open(path):
        if line.startswith("AtTenTris"):
            cur = line.strip()
            out[cur] = {}
        elif line.startswith("{") and cur:
            r = json.loads(line)
            out[cur][r["seed"]] = r
    return out


def pick(path, *subs):
    for k, v in blocks(path).items():
        if all(s in k for s in subs):
            return k, v
    raise SystemExit("no block in %s matching %s" % (path, subs))


def main():
    pieces = 500
    for garbage in ("no garbage", "garbage 10,3-4"):
        ka, a = pick(sys.argv[1], sys.argv[2], garbage)
        kb, b = pick(sys.argv[3], sys.argv[4], garbage)
        seeds = sorted(set(a) & set(b))
        d = [(a[s]["attack"] - b[s]["attack"]) / pieces for s in seeds]
        n = len(d)
        m = sum(d) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
        se = sd / math.sqrt(n)
        won = sum(x > 0 for x in d)
        lost = sum(x < 0 for x in d)
        print("%s, %d paired games: A %.3f vs B %.3f APPfull | diff %+.3f +- %.3f (t = %.2f) | A better in %d, worse in %d, tied %d" % (
            garbage, n, sum(a[s]["attack"] for s in seeds) / (n * pieces), sum(b[s]["attack"] for s in seeds) / (n * pieces),
            m, se, m / se if se else float("nan"), won, lost, n - won - lost))
        print("   A = %s\n   B = %s" % (ka[:110], kb[:110]))


if __name__ == "__main__":
    main()
