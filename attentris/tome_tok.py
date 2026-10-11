"""ToMe v3 pre-tokenization (no learning): one sky token, solid filled blocks as one token each,
every other cell (below the sky) as a single-cell token for the model to merge.

Sky: the full-width rectangle of rows above the tallest column (AtTenTris's sky). The air below
it, between the stack surface and that row, deliberately stays as single cells, so merging
decides how to group it (wells and gaps included).

Blocks: greedy largest solid rectangles. Repeatedly take the largest all-filled rectangle of at
least 2 x 2 among the filled cells not yet used (ties: lowest row, then leftmost column, then the
taller one); stop when none is left. Every other filled cell stays a single cell.

    python attentris/tome_tok.py --src attentris/tome_viz.html --out attentris/tome_tok.html
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# every rectangle (r0, r1, c0, c1) of at least 2 x 2 on the 20 x 10 board
_R0, _R1, _C0, _C1 = [np.array(v) for v in zip(*[(r0, r1, c0, c1) for r0 in range(20) for r1 in range(r0 + 1, 20)
                                                    for c0 in range(10) for c1 in range(c0 + 1, 10)])]
_AREA = (_R1 - _R0 + 1) * (_C1 - _C0 + 1)
_ORDER = np.lexsort((-(_R1 - _R0), _C0, _R0, -_AREA))      # largest first; then lowest, leftmost, taller


def blocks_of(filled):
    """Greedy largest solid rectangles (>= 2 x 2) of a (20, 10) bool array -> list of (r0, r1, c0, c1)."""
    avail = np.asarray(filled, bool).copy()
    out = []
    while True:
        P = np.zeros((21, 11), np.int32)
        P[1:, 1:] = avail.cumsum(0).cumsum(1)
        s = P[_R1 + 1, _C1 + 1] - P[_R0, _C1 + 1] - P[_R1 + 1, _C0] + P[_R0, _C0]
        full = (s == _AREA)[_ORDER]
        if not full.any():
            return out
        k = _ORDER[np.argmax(full)]
        r0, r1, c0, c1 = int(_R0[k]), int(_R1[k]), int(_C0[k]), int(_C1[k])
        out.append((r0, r1, c0, c1))
        avail[r0:r1 + 1, c0:c1 + 1] = False


def pretokenize(board):
    """board: (20, 10) bool, row 0 = bottom. Returns a list of (kind, cells) with cells as flat
    indices r * 10 + c; kinds 'sky' (at most one), 'block', 'cell'. Every cell of the board is in
    exactly one token."""
    g = np.asarray(board, bool)
    top = int(np.where(g.any(1))[0].max()) + 1 if g.any() else 0      # rows 0..top-1 hold the stack
    toks = []
    if top < 20:
        toks.append(("sky", list(range(top * 10, 200))))
    used = np.zeros(200, bool)
    for r0, r1, c0, c1 in blocks_of(g):
        cells = [r * 10 + c for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        toks.append(("block", cells))
        used[cells] = True
    toks += [("cell", [k]) for k in range(top * 10) if not used[k]]
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
