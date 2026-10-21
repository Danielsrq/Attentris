"""Shapley values of token groups for the game outcome (VS score, deaths) of one trained model.

Players: token groups (GROUPS). A coalition S plays with the groups in S present and the rest blanked at
inference (the well group is not blanked: without it the rows a well covers are play rows, so only the label,
its extras and the grouping go) (play.py --drop-kinds; the model was trained with all of them, as token_shap.py). v(S) = the
mean per-game VSfull ((attack + garbage cleared) / 500 pieces) over the same paired games: depth 1,
normal / danger x 80/20, 50/50, seeds 33001+ and 53001+ (64 each). Exact Shapley over all 2^n
coalitions, the pairwise Shapley interaction index, and a paired bootstrap over the seeds (the same
seeds in every coalition) for 90% intervals. The all-present coalition is the baseline's own games
(dagger3_eval, play is deterministic per seed).

    # 6 groups: 63 coalitions x 4 settings x 128 games ~ 32k games, ~2 h CPU; resumable per game file
    $CPUPY attentris/group_shap.py > attentris/group_shap.out 2>&1
    $CPUPY attentris/group_shap.py --report          # only the tables, from the game files
"""
import argparse
import itertools
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GROUPS = {   # name -> token-kind prefixes
    "fitx": "fit_s,fit_z,fit_l,fit_j,fit_i",
    "fitt": "fit_t",
    "setup": "setup_",
    "tstruct": "imperial,stsd,cspin,fractal,tsd,vertical",
    "pitrect": "pit,rect12",
    "well": None,   # a base kind: absent = no well tokens, the clean zone's rows tokenized as play rows (play.py --no-wells)
}
SETTINGS = {   # tag -> (style, garbage)
    "normal_g20": ("normal", "10,3-4,cheese0.2,0.2"),
    "normal_g50": ("normal", "10,3-4,cheese0.5,0.2"),
    "danger_g20": ("danger", "10,3-4,cheese0.2,0.2"),
    "danger_g50": ("danger", "10,3-4,cheese0.5,0.2"),
}
SEEDS = {"s33": 33001, "s53": 53001}


def coalition_name(present):
    return "+".join(g for g in GROUPS if g in present) or "none"


def game_file(args, present, tag, sk):
    if len(present) == len(GROUPS):   # all present: the baseline's own games
        return os.path.join(HERE, "data", "dagger3_eval", "%s_%s%s.games" % (args.base, tag, "" if sk == "s33" else "_s53"))
    absent = [g for g in GROUPS if g not in present]
    return os.path.join(args.dir, "%s_no-%s_%s_%s.games" % (args.base, "-".join(absent), tag, sk))


def finished(path):
    return os.path.exists(path) and any(l.startswith("AtTenTris") for l in open(path))


def play(args, present, tag, sk):
    path = game_file(args, present, tag, sk)
    if finished(path):
        return
    style, garbage = SETTINGS[tag]
    drop = ",".join(GROUPS[g] for g in GROUPS if g not in present and GROUPS[g])
    cmd = [sys.executable, os.path.join(HERE, "play.py"), "--model", os.path.join(HERE, "data", "model_%s.pt" % args.base),
           "--style", style, "--games", str(args.games), "--depth", "1", "--seed", str(SEEDS[sk]), "--garbage", garbage,
           "--drop-kinds", drop, "--out", path] + ([] if "well" in present else ["--no-wells"])
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    line = [l for l in out.splitlines() if l.startswith("AtTenTris")]
    print("%-36s %s %s: %s" % (coalition_name(present), tag, sk, line[0].split(": ", 1)[1][:110] if line else "FAILED"), flush=True)


def load(args, present, tag):
    """per game (seed set, seed) -> (VSfull, died)"""
    out = {}
    for sk in SEEDS:
        for l in open(game_file(args, present, tag, sk)):
            if l.startswith("{"):
                r = json.loads(l)
                out[(sk, r["seed"])] = ((r["attack"] + r["garbage_cleared"]) / args.pieces, float(r["died"]))
    return out


def shapley(v, n):
    """v: dict frozenset(player index) -> value (array: one value per bootstrap draw) -> phi (n, ...), I (n, n, ...)"""
    phi = [0.0] * n
    inter = [[0.0] * n for _ in range(n)]
    for i in range(n):
        rest = [k for k in range(n) if k != i]
        for r in range(n):
            for S in itertools.combinations(rest, r):
                S = frozenset(S)
                phi[i] = phi[i] + math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n) * (v[S | {i}] - v[S])
        for j in range(i + 1, n):
            rest2 = [k for k in range(n) if k not in (i, j)]
            for r in range(n - 1):
                for S in itertools.combinations(rest2, r):
                    S = frozenset(S)
                    w = math.factorial(r) * math.factorial(n - r - 2) / math.factorial(n - 1)
                    inter[i][j] = inter[i][j] + w * (v[S | {i, j}] - v[S | {i}] - v[S | {j}] + v[S])
                    inter[j][i] = inter[i][j]
    return phi, inter


def report(args):
    names = list(GROUPS)
    n = len(names)
    coals = [frozenset(c) for r in range(n + 1) for c in itertools.combinations(range(n), r)]
    rng = np.random.default_rng(0)
    pooled = {"VSfull": {c: 0.0 for c in coals}, "deaths": {c: 0.0 for c in coals}}
    keys0 = None
    tables = []
    for tag in SETTINGS:
        data = {c: load(args, {names[k] for k in c}, tag) for c in coals}
        keys = sorted(set.intersection(*(set(d) for d in data.values())))
        if keys0 is None:
            keys0 = len(keys)
            draws = rng.integers(0, len(keys), (args.boot, len(keys)))
        for m, mi in (("VSfull", 0), ("deaths", 1)):
            # value per coalition: point estimate and bootstrap draws (resampling the paired seeds); deaths per 128 games
            scale = 1.0 if m == "VSfull" else len(keys)
            v = {}
            for c in coals:
                x = np.array([data[c][k][mi] for k in keys]) * scale
                v[c] = np.concatenate([[x.mean()], x[draws].mean(axis=1)])
                pooled[m][c] = pooled[m][c] + v[c] / len(SETTINGS)
            tables.append((tag, m, v, len(keys)))
    tables += [("pooled (mean of the 4)", m, pooled[m], keys0) for m in ("VSfull", "deaths")]
    for tag, m, v, nk in tables:
        phi, inter = shapley(v, n)
        full, empty = v[frozenset(range(n))][0], v[frozenset()][0]
        print("\n== %s, %s (%d paired games; all present %.3f, none %.3f, sum of phi %.3f)" % (tag, m, nk, full, empty, full - empty))
        print("  %-9s %8s %18s %10s" % ("group", "phi", "90% interval", "leave-out"))
        for i, g in enumerate(names):
            lo, hi = np.percentile(phi[i][1:], [5, 95])
            loo = full - v[frozenset(range(n)) - {i}][0]   # leave-one-out at inference: all present minus all but i
            print("  %-9s %+8.3f   [%+7.3f, %+7.3f] %+10.3f" % (g, phi[i][0], lo, hi, loo))
        if m == "VSfull" or tag.startswith("pooled"):
            pairs = sorted(((i, j) for i in range(n) for j in range(i + 1, n)), key=lambda p: -abs(inter[p[0]][p[1]][0]))
            print("  interactions (Shapley interaction index; < 0: substitutes, > 0: complements):")
            for i, j in pairs[:4]:
                lo, hi = np.percentile(inter[i][j][1:], [5, 95])
                print("    %-8s x %-8s %+7.3f [%+6.3f, %+6.3f]" % (names[i], names[j], inter[i][j][0], lo, hi))
    print("\nvalue of every coalition, pooled: VSfull / deaths (of the paired games per setting):")
    for c in coals:
        print("  %-36s %.3f %6.1f" % (coalition_name({names[k] for k in c}), pooled["VSfull"][c][0], pooled["deaths"][c][0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="v51_d128_dagger5b_ft")
    ap.add_argument("--dir", default=os.path.join(HERE, "data", "group_shap"))
    ap.add_argument("--games", type=int, default=64, help="per seed set (two sets: 128 paired games per setting)")
    ap.add_argument("--pieces", type=int, default=500)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--report", action="store_true", help="only the tables")
    args = ap.parse_args()
    os.makedirs(args.dir, exist_ok=True)
    if not args.report:
        names = list(GROUPS)
        todo = [(frozenset(c), tag, sk) for r in range(len(names)) for c in itertools.combinations(names, r)
                for tag in SETTINGS for sk in SEEDS]
        left = sum(not finished(game_file(args, c, t, s)) for c, t, s in todo)
        print("group SHAP of %s: %d groups, %d coalitions to play, %d runs of %d games left (%s)" % (
            args.base, len(names), 2 ** len(names) - 1, left, args.games, time.strftime("%H:%M")), flush=True)
        for i, (c, tag, sk) in enumerate(todo):
            play(args, c, tag, sk)
        print("played (%s)" % time.strftime("%H:%M"), flush=True)
    report(args)


if __name__ == "__main__":
    main()
