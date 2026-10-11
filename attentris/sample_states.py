"""Pick a few states from play.py games (--out + --dump) for the viewers: in every game that died,
the states --before moves before the death (e.g. 30 and 10), plus --random states from the rest.
Writes them in the format `bot label` reads, and a notes file ("<seed>_<n>" -> why) for the viewer.

    python attentris/sample_states.py --games d1.games --states d1_states.jsonl --out picked.jsonl --notes picked_notes.json
"""
import argparse
import json
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", required=True)
    ap.add_argument("--states", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--notes", required=True)
    ap.add_argument("--before", default="30,10", help="moves before each death")
    ap.add_argument("--random", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    deaths = {}
    for l in open(args.games):
        if l.startswith("{"):
            r = json.loads(l)
            if r.get("died"):
                deaths[r["seed"]] = r["pieces"]
    states = {(s["seed"], s["n"]): s for s in map(json.loads, open(args.states))}
    picked, notes = [], {}
    for seed, dn in sorted(deaths.items()):
        for b in (int(x) for x in args.before.split(",")):
            n = max(0, dn - b)
            # the dump's n counts placements; take the nearest state at or before n
            ns = [k[1] for k in states if k[0] == seed and k[1] <= n]
            if ns:
                k = (seed, max(ns))
                picked.append(states[k])
                notes["%d_%d" % k] = "%d moves before the death at %d" % (dn - k[1], dn)
    rest = [k for k in states if k[0] not in deaths]
    for k in random.Random(args.seed).sample(rest, min(args.random, len(rest))):
        picked.append(states[k])
        notes["%d_%d" % k] = "random state (game survived)"
    with open(args.out, "w") as f:
        for s in picked:
            f.write(json.dumps({k: s[k] for k in ("seed", "n", "rows", "b2b", "piece", "queue", "hold", "style")}) + "\n")
    json.dump(notes, open(args.notes, "w"), indent=1)
    print("%d states (%d deaths) -> %s" % (len(picked), len(deaths), args.out))


if __name__ == "__main__":
    main()
