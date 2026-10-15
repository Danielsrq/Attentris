"""DAgger round 2, step 2: pick the informative states from the student games (play.py --dump, with
--out for which games died), instead of labelling every visited state.

Kept (each state tagged with every reason that applies):
  death     within --before-death moves of a death (the decay that killed the game)
  danger    stack >= --height rows or >= --holes holes (where recovery skill matters)
  holes_up  the board's holes rose by >= --holes-up over the last --window student moves (where it
            starts going wrong, well before any death; holes, not the teacher's 1-ply value: greedy V
            alone plays worst of all, the teacher's strength is its search)
  unsure    the student's move distribution has entropy >= --entropy (not learned well)
  random    --random of the remaining states (ordinary coverage)
Exact duplicates (same board, pieces, hold) are dropped. Writes the selected states in the format
`bot label` reads, plus a summary.

    python attentris/dagger_select.py --games data/dagger2/*.games --states data/dagger2/*.jsonl --out data/dagger2/selected.jsonl
"""
import argparse
import collections
import glob
import json
import os
import random
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pack as P
import tokenizer as T

HERE = os.path.dirname(os.path.abspath(__file__))
BOT = os.path.join(HERE, "..", "bot", "target", "release", "bot.exe" if os.name == "nt" else "bot")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+", required=True, help="play.py --out files (glob ok)")
    ap.add_argument("--states", nargs="+", required=True, help="play.py --dump files (glob ok), same runs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--before-death", type=int, default=60)
    ap.add_argument("--height", type=int, default=15)
    ap.add_argument("--holes", type=int, default=10)
    ap.add_argument("--holes-up", type=int, default=4, help="holes added over the window")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--entropy", type=float, default=3.0, help="~ the top 10%% of student move entropies (dagger2: p90 = 3.02)")
    ap.add_argument("--random", type=float, default=0.10)
    args = ap.parse_args()
    rng = random.Random(0)
    files = lambda pats: sorted(f for p in pats for f in glob.glob(p))

    # deaths: (run tag, seed) -> pieces at death
    deaths = {}
    for f in files(args.games):
        tag = os.path.basename(f).rsplit(".", 1)[0]
        for l in open(f):
            if l.startswith("{"):
                r = json.loads(l)
                if r.get("died"):
                    deaths[(tag, r["seed"])] = r["pieces"]
    games = collections.defaultdict(list)
    for f in files(args.states):
        tag = os.path.basename(f).rsplit(".", 1)[0]
        for l in open(f):
            s = json.loads(l)
            s["run"] = tag
            games[(tag, s["seed"])].append(s)
    allst = [s for g in games.values() for s in sorted(g, key=lambda s: s["n"])]
    print("%d games, %d states, %d deaths" % (len(games), len(allst), len(deaths)), flush=True)

    for s in allst:
        b = T.board_from_replay(P.rows_to_strings(s["rows"]))
        s["_h"], s["_holes"] = int(T.heights(b).max()), T.holes_of(b)

    reasons = collections.Counter()
    picked, seen = [], set()
    for key, g in games.items():
        g.sort(key=lambda s: s["n"])
        dn = deaths.get(key)
        for i, s in enumerate(g):
            h, holes = s["_h"], s["_holes"]
            why = []
            if dn is not None and s["n"] >= dn - args.before_death:
                why.append("death")
            if h >= args.height or holes >= args.holes:
                why.append("danger")
            j = i - args.window
            if j >= 0 and holes - g[j]["_holes"] >= args.holes_up:
                why.append("holes_up")
            if s.get("entropy", 0.0) >= args.entropy:
                why.append("unsure")
            if not why and rng.random() < args.random:
                why.append("random")
            if not why:
                continue
            sig = (tuple(s["rows"]), s["piece"], tuple(s["queue"]), s["hold"])
            if sig in seen:
                reasons["duplicate (dropped)"] += 1
                continue
            seen.add(sig)
            for w in why:
                reasons[w] += 1
            rec = {k: s[k] for k in ("seed", "n", "rows", "b2b", "piece", "queue", "hold", "style")}
            rec["why"] = why
            picked.append(rec)
    with open(args.out, "w") as f:
        for r in picked:
            f.write(json.dumps(r) + "\n")
    print("selected %d of %d states (%.1f%%) -> %s" % (len(picked), len(allst), 100 * len(picked) / max(1, len(allst)), args.out))
    print("reasons (a state can have several): " + ", ".join("%s %d" % kv for kv in reasons.most_common()))


if __name__ == "__main__":
    main()
