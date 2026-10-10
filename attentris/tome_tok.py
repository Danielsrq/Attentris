"""ToMe v3 pre-tokenization (no learning): the sky as one token per column, solid filled blocks as
one token each, every other cell (below the sky) as a single-cell token for the model to merge.

Sky (enclosure-aware): in each column, the empty cells from the LOWER of its two neighbours' tops
upwards (a wall counts as full height) -- or from its own top, if that is higher. So the open air
above the stack is sky, but a well or a gap below both neighbours stays as cells (a plain
"above the column's own top" rule would swallow every open well into a sky column).

Blocks: per row, the horizontal runs of filled cells; identical runs in consecutive rows stack
into one rectangle (a unique decomposition, no tie-breaking). A rectangle becomes a block only if
it is at least 2 x 2; smaller ones stay as single cells.

    python attentris/tome_tok.py --src attentris/tome_viz.html --out attentris/tome_tok.html
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def sky_start(h):
    """Row from which each column is sky (see the module doc); h = column heights."""
    pad = np.concatenate([[20], h, [20]])
    return np.maximum(h, np.minimum(pad[:-2], pad[2:]))


def pretokenize(board, min_w=2, min_h=2):
    """board: (20, 10) bool, row 0 = bottom. Returns a list of (kind, cells) with cells as flat
    indices r * 10 + c; kinds 'sky' (one per column with sky), 'block', 'cell'. Every cell of the
    board is in exactly one token."""
    g = np.asarray(board, bool)
    h = np.where(g, np.arange(1, 21)[:, None], 0).max(0)
    s0 = sky_start(h)
    toks = []
    for c in range(10):
        if s0[c] < 20:
            toks.append(("sky", [r * 10 + c for r in range(s0[c], 20)]))
    rects, open_runs = [], {}
    for r in range(20):
        runs, c = [], 0
        while c < 10:
            if g[r, c]:
                c0 = c
                while c < 10 and g[r, c]:
                    c += 1
                runs.append((c0, c - 1))
            else:
                c += 1
        new = {run: open_runs.get(run, r) for run in runs}
        rects += [(r0, r - 1, run[0], run[1]) for run, r0 in open_runs.items() if run not in new]
        open_runs = new
    rects += [(r0, 19, run[0], run[1]) for run, r0 in open_runs.items()]
    used = np.zeros(200, bool)
    for r0, r1, c0, c1 in sorted(rects):
        if r1 - r0 + 1 >= min_h and c1 - c0 + 1 >= min_w:
            cells = [r * 10 + c for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
            toks.append(("block", cells))
            used[cells] = True
    for r in range(20):
        for c in range(10):
            if r < s0[c] and not used[r * 10 + c]:
                toks.append(("cell", [r * 10 + c]))
    return toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "tome_viz.html"), help="take the positions from this ToMe viewer page")
    ap.add_argument("--out", default=os.path.join(HERE, "tome_tok.html"))
    args = ap.parse_args()
    s = open(args.src, encoding="utf-8").read()
    a = s.index("const DATA = ") + len("const DATA = ")
    src = json.JSONDecoder().raw_decode(s[a:])[0]["positions"]
    pos = []
    for q in src:
        b = np.array(q["board"], bool).reshape(20, 10)
        toks = pretokenize(b)
        pos.append({"why": q["why"], "teacher": q["teacher"], "seed": q["seed"], "n": q["n"], "board": q["board"],
                    "chosen": q["chosen"]["cells"], "tokens": [{"kind": k, "cells": c} for k, c in toks]})
    n = np.array([len(p["tokens"]) for p in pos])
    data = json.dumps({"positions": pos, "summary": {"mean": float(n.mean()), "min": int(n.min()), "max": int(n.max())}}, separators=(",", ":"))
    tpl = open(os.path.join(HERE, "tome_tok_template.html"), encoding="utf-8").read()
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(tpl.replace("__DATA__", data))
    print("%d positions -> %s (tokens per position: mean %.1f, min %d, max %d)" % (len(pos), args.out, n.mean(), n.min(), n.max()))


if __name__ == "__main__":
    main()
