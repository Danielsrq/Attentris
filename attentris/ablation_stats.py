"""Paired death statistics for the d128 ablations: every model plays the same seeds, so for each garbage
setting the games are paired by (seed set, seed) and compared with an exact McNemar test (two-sided binomial
on the games only one of the two models died in). Seed sets: 33001+ (`<model>_<setting>.games`) and 53001+
(`<model>_<setting>_s53.games`, ablation_more.sh); a pair counts only when both models have it.
This measures game-to-game noise only: each model is one training run.

    python attentris/ablation_stats.py BASELINE MODEL [MODEL ...]
"""
import json
import os
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
DE = os.path.join(HERE, "data", "dagger3_eval")
SETTINGS = ["normal_g20", "normal_g50", "danger_g20", "danger_g50", "normal_d2g20", "normal_d2g50", "danger_d2g20", "danger_d2g50"]


def deaths(model, setting):
    out = {}
    for tag, suffix in (("s33", ""), ("s53", "_s53")):
        p = os.path.join(DE, "%s_%s%s.games" % (model, setting, suffix))
        if not os.path.exists(p):
            continue
        for line in open(p):
            if line.startswith("{"):
                r = json.loads(line)
                if r.get("done"):
                    out[(tag, r["seed"])] = bool(r["died"])
    return out


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def main():
    base, models = sys.argv[1], sys.argv[2:]
    short = lambda m: m.replace("v51_d128_", "").replace("_dagger5b_ft", "") or m
    print("paired deaths vs %s (exact McNemar; b = only the ablation died, c = only the baseline died)" % base)
    for md in models:
        tb = tc = 0
        print(short(md))
        for x in SETTINGS:
            g0, g1 = deaths(base, x), deaths(md, x)
            keys = sorted(set(g0) & set(g1))
            if not keys:
                continue
            b = sum(g1[k] and not g0[k] for k in keys)
            c = sum(g0[k] and not g1[k] for k in keys)
            tb += b
            tc += c
            print("  %-13s n=%3d  deaths %3d -> %3d  b %2d c %2d  p=%.4f" % (
                x, len(keys), sum(g0[k] for k in keys), sum(g1[k] for k in keys), b, c, mcnemar(b, c)))
        print("  pooled        b %3d c %3d  p=%.2g" % (tb, tc, mcnemar(tb, tc)))


if __name__ == "__main__":
    main()
