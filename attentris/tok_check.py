"""Tokens only (no model): re-tokenize the attention viewer's positions with the current
tokenizer and draw them in the tokenizer viewer, to check slots / hangs / setups by eye.

    python attentris/tok_check.py --src attentris/attn_v5.html --out attentris/tok_check.html
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tokenizer as T

HERE = os.path.dirname(os.path.abspath(__file__))


def tok_json(t):
    rs, cs = np.nonzero(t.mask)
    e = {k: (v if not isinstance(v, (np.integer, np.floating)) else v.item()) for k, v in t.extras.items()}
    return {"kind": t.kind, "cells": [[int(r), int(c)] for r, c in zip(rs, cs)], "extras": e, "box": t.box, "base": t.base}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "attn_v5.html"))
    ap.add_argument("--out", default=os.path.join(HERE, "tok_check.html"))
    args = ap.parse_args()
    s = open(args.src, encoding="utf-8").read()
    src = json.JSONDecoder().raw_decode(s[s.index('{"summary"'):])[0]["positions"]
    boards, pieces = [], []
    for q in src:
        b = np.array(q["board"], bool).reshape(T.ROWS, T.COLS)
        pk = {t["kind"]: t["piece"] for t in q["tokens"] if t["kind"] in T.PIECE_KINDS}
        queue = [pk["preview%d" % i] for i in range(1, 6) if pk.get("preview%d" % i)]
        boards.append(b)
        pieces.append((pk["current"], queue, pk.get("hold")))
    rows40 = []
    for b in boards:
        r = ["." * 10] * 40
        for y in range(T.ROWS):
            r[39 - y] = "".join("#" if b[y, c] else "." for c in range(10))
        rows40.append(r)
    structs = T.structures_for(rows40, pieces=pieces)
    out = []
    for i, (q, b, (pc, qu, hd), st) in enumerate(zip(src, boards, pieces, structs)):
        toks = T.tokenize(b, pc, qu, hd, st)
        _, h = T.base_tokens(b)
        found = T.structure_tokens(st) + T.fit_tokens(b, h) + T.pit_tokens(b, h)
        ch = q["chosen"]
        out.append({"label": "%d. %s teacher · seed %d, placement %d" % (i + 1, q["teacher"], q["seed"], q["n"]),
                    "why": q["why"], "board": [[int(v) for v in row] for row in b], "piece": pc, "queue": qu, "hold": hd,
                    "chose": {"cells": [[c // 10, c % 10] for c in ch["cells"]], "type": ch["piece"] + (" from hold" if ch["hold"] else ""),
                              "spin": ["none", "mini", "full"][ch["spin"]], "lines": ch["lines"]},
                    "tokens": [tok_json(t) for t in toks], "overlays_found": len(found)})
    data = json.dumps({"kinds": {"base": list(T.BASE_KINDS), "overlay": list(T.OVERLAY_KINDS)}, "k_board": T.K_BOARD,
                       "p_max": T.P_MAX, "version": T.TOK_VERSION, "positions": out}, separators=(",", ":"))
    tpl = open(os.path.join(HERE, "tok_check_template.html"), encoding="utf-8").read()
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(tpl.replace("__DATA__", data))
    print("%d positions -> %s" % (len(out), args.out))


if __name__ == "__main__":
    main()
