"""Tokenize every board of some replays and report token counts, overlay kinds, and a few
example parses (the first board where each structure kind appears).

    python attentris/survey.py data/bot/gate4_garbage.jsonl --replay normalrules
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tokenizer as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("replays", nargs="+")
    ap.add_argument("--replay", default="", help="only replays whose name contains this")
    ap.add_argument("--examples", default="imperial,stsd,cspin,fractal,tsd,rect12",
                    help="structure kinds to show one parse of")
    args = ap.parse_args()

    recs = []
    for path in args.replays:
        for line in open(path):
            r = json.loads(line)
            if args.replay in r["replay"]:
                recs.append(r)
    structs = T.structures_for([r["board"] for r in recs])
    n_tok = collections.Counter()
    n_base = collections.Counter()
    n_over = collections.Counter()
    kinds = collections.Counter()
    kinds_before_cap = collections.Counter()
    shown = set()
    want = args.examples.split(",")
    examples = []
    for r, s in zip(recs, structs):
        b = T.board_from_replay(r["board"])
        toks = T.tokenize(b, r["piece"], r["queue"], r["hold"], s)
        assert T.check_partition(toks)
        base = sum(t.base for t in toks)
        over = [t for t in toks if t.kind in T.OVERLAY_KINDS]
        n_tok[len(toks)] += 1
        n_base[base] += 1
        n_over[len(over)] += 1
        for t in over:
            kinds[t.kind] += 1
        _, h = T.base_tokens(b)
        for t in T.structure_tokens(s) + T.fit_tokens(b, h) + T.pit_tokens(b, h):
            kinds_before_cap[t.kind] += 1
        for k in want:
            if k not in shown and any(t.kind == k for t in over) and (k != "tsd" or r["piece"] == "t"):
                shown.add(k)
                examples.append((k, r, toks, b))

    n = len(recs)
    print("%d boards" % n)
    for name, c in (("tokens per board", n_tok), ("base board tokens", n_base), ("overlay tokens", n_over)):
        vals = np.repeat(list(c.keys()), list(c.values()))
        print("%-18s mean %.1f  median %d  max %d  min %d" % (name, vals.mean(), np.median(vals), vals.max(), vals.min()))
    print("overlay kinds (boards' share with >= 1, kept after the P_MAX cap / found before it):")
    for k in T.OVERLAY_KINDS:
        if kinds_before_cap[k]:
            print("  %-9s kept %6d  found %6d  (%.1f per board found)" % (k, kinds[k], kinds_before_cap[k], kinds_before_cap[k] / n))
    for k, r, toks, b in examples:
        print("\n=== first %s: %s placement %d, piece %s, chose %s %s" % (
            k, r["replay"], r["n"], r["piece"].upper(), r["place"]["type"].upper(), r["place"].get("spin", "")))
        print(T.render(b, toks))


if __name__ == "__main__":
    main()
