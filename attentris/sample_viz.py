"""Collect interesting boards (bot replays + made-up edge cases), tokenize them, and write
the JSON the tokenizer viewer (viz.html) embeds.

    python attentris/sample_viz.py            # writes attentris/viz.html
"""
import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tokenizer as T

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "bot")

SOURCES = [  # (file, replay name filter, label)
    ("gate4_garbage.jsonl", "normalrules", "gate4, normal rules + garbage"),
    ("gate4_stress.jsonl", "stress", "gate4, stress garbage"),
    ("gate6_stress.jsonl", "stress", "gate6, stress garbage"),
    ("normal.jsonl", "", "normal bot"),
    ("allspin_slots.jsonl", "", "all-spin bot"),
    ("gate3_garbage.jsonl", "", "gate3 + cheese/clean garbage"),
]


def ascii_board(lines):
    return T.board_from_ascii(lines)


def stsd_board():
    """An STSD built straight from the template (stem column 3, nub right)."""
    b = np.ones((T.ROWS, T.COLS), bool)
    b[7:, :] = False
    b[0:2, 9] = False                      # two clean garbage rows, hole column 9
    b[2, 3] = b[2, 4] = False              # stem bottom + h (the STSD's second slot)
    b[3, 3] = b[3, 4] = False              # stem middle + nub
    b[4, 3] = False                        # stem top (roof at (4,4) stays filled)
    b[5, 3:6] = False                      # the lane
    b[6, :] = False
    b[6, 0:4] = True                       # cap over the stem
    b[6, 6] = True                         # A
    return b


MADE_UP = [
    ("empty board", lambda: np.zeros((T.ROWS, T.COLS), bool)),
    ("tetris-ready: 4 clean rows, open well in column 9", lambda: ascii_board(
        ["#########."] * 4)),
    ("cheese: 6 garbage rows, a different hole each", lambda: ascii_board(
        ["##.#######", "#######.##", "#.########", "#####.####", "########.#", ".#########"])),
    ("clean garbage + cheese mix under a messy top", lambda: ascii_board(
        ["..#....##.", ".####.####", "######.###", "#######.##", "##.#######", "####.#####",
         "####.#####", "####.#####", "####.#####"])),
    ("STSD made from the template", stsd_board),
    ("TSD slot on a flat stack", lambda: ascii_board(
        ["##...##..#", "###...####", "####.#####", "#########."])),
    ("almost a TSD: no roof corner, so not a T-spin slot", lambda: ascii_board(
        ["##.......#", "###...####", "####.#####", "#########."])),
    ("3x4 and 4x3 pits (rect12) side by side", lambda: ascii_board(
        ["...#....##", "...#....##", "...#....##", "...#######", "#########."])),
    ("deep 2-wide well + 1-wide well", lambda: ascii_board(
        ["##..####.#", "##..####.#", "##..####.#", "##..####.#", "##..####.#", "##########"[:9] + "."])),
    ("swiss cheese: scattered covered holes", lambda: ascii_board(
        [".#.#.#.#..", "#.#.#.#.##", "##.##.##.#", "#.##.##.##", "###.###.##", "#.#####.##"])),
    ("tall bumpy stack, 17 high", lambda: ascii_board(
        ["#.........", "##........", "##.#......", "###.#.....", "####.#....", "#####.#...",
         "######.#..", "#######.#.", "########.#", "#######.##", "######.###", "#####.####",
         "####.#####", "###.######", "##.#######", "#.########", ".#########"])),
    ("topped out: stack reaches row 20 (no sky)", lambda: ascii_board(
        ["#.........", "#........."] + ["#########."] * 18)),
    ("overhang lid over a pit (unreachable cavity)", lambda: ascii_board(
        ["...####...", "...#..#...", "...#..#...", "#########."])),
]


def tok_json(t):
    rs, cs = np.nonzero(t.mask)
    return {"kind": t.kind, "cells": [[int(r), int(c)] for r, c in zip(rs, cs)], "extras": t.extras,
            "box": t.box, "base": t.base}


def entry(label, why, b, piece=None, queue=(), hold=None, structs=(), chose=None):
    toks = T.tokenize(b, piece, queue, hold, structs)
    assert T.check_partition(toks), label
    _, h = T.base_tokens(b)
    found = T.structure_tokens(structs) + T.pit_tokens(b, h)
    return {"label": label, "why": why, "board": [[int(v) for v in row] for row in b],
            "piece": piece, "queue": list(queue), "hold": hold, "chose": chose,
            "tokens": [tok_json(t) for t in toks], "overlays_found": len(found)}


def to_replay(b):
    rows = ["." * 10] * 40
    for r in range(T.ROWS):
        rows[39 - r] = "".join("#" if b[r, c] else "." for c in range(10))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "viz.html"), help=".html (viewer) or .json (data only)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    picked = []
    seen_kinds = set()
    for fname, filt, label in SOURCES:
        path = os.path.join(DATA, fname)
        if not os.path.exists(path):
            continue
        recs = [r for r in map(json.loads, open(path)) if filt in r["replay"]]
        structs = T.structures_for([r["board"] for r in recs])
        scored = []
        for r, s in zip(recs, structs):
            b = T.board_from_replay(r["board"])
            toks = T.tokenize(b, r["piece"], r["queue"], r["hold"], s)
            kinds = {t.kind for t in toks}
            h = T.heights(b)
            scored.append((r, s, b, toks, kinds, int(h.max()), sum(t.base for t in toks),
                           sum(t.kind == "well" for t in toks)))
        chosen = []

        def take(pred, why, key=None, k=1):
            pool = [x for x in scored if pred(x) and x[0] not in [c[0] for c in chosen]]
            if not pool:
                return
            pool = sorted(pool, key=key) if key else rng.sample(pool, len(pool))
            for x in pool[:k]:
                chosen.append((x[0], x[1], x[2], why))

        for kind in ("imperial", "stsd", "cspin", "fractal", "vertical", "rect12"):
            if kind not in seen_kinds:
                before = len(chosen)
                # prefer the T piece in hand, the structure's natural moment
                take(lambda x, k=kind: k in x[4] and x[0]["piece"] == "t", "has a %s overlay (T in hand)" % kind)
                if len(chosen) == before:
                    take(lambda x, k=kind: k in x[4], "has a %s overlay" % kind)
                if len(chosen) > before:
                    seen_kinds.add(kind)
        if "stress" in fname:
            take(lambda x: True, "tallest stack in these games", key=lambda x: -x[5])
            take(lambda x: True, "most well tokens (garbage)", key=lambda x: -x[7])
        take(lambda x: True, "most base tokens (budget / bands)", key=lambda x: -x[6])
        take(lambda x: x[5] >= 3, "random mid-game position", k=1)
        for r, s, b, why in chosen:
            pl = r["place"]
            picked.append(entry("%s · %s placement %d" % (label, r["replay"], r["n"]), why, b, r["piece"].upper(),
                                [q.upper() for q in r["queue"]], r["hold"].upper() if r["hold"] else None, s,
                                {"type": pl["type"].upper(), "spin": pl.get("spin", "none"),
                                 "cells": [[39 - y, x] for x, y in pl["cells"]]}))

    made = []
    boards = [(name, fn()) for name, fn in MADE_UP]
    structs = T.structures_for([to_replay(b) for _, b in boards])
    for (name, b), s in zip(boards, structs):
        made.append(entry("made up · " + name, "edge case", b, "T", list("IOSZJ"), None, s))

    # interleave so the viewer alternates real and made-up boards
    out = []
    while picked or made:
        if picked:
            out.append(picked.pop(0))
        if picked:
            out.append(picked.pop(0))
        if made:
            out.append(made.pop(0))
    data = json.dumps({"kinds": {"base": list(T.BASE_KINDS), "overlay": list(T.OVERLAY_KINDS)},
                       "k_board": T.K_BOARD, "p_max": T.P_MAX, "positions": out}, separators=(",", ":"))
    if args.out.endswith(".html"):
        template = open(os.path.join(HERE, "viz_template.html"), encoding="utf-8").read()
        data = template.replace("__DATA__", data)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(data)
    print("%d positions -> %s" % (len(out), args.out))
    for e in out:
        print("  %-75s %-38s %d tokens" % (e["label"][:75], e["why"], len(e["tokens"])))


if __name__ == "__main__":
    main()
