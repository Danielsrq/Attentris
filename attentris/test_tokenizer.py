"""python -m pytest attentris  (or: python attentris/test_tokenizer.py)"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tokenizer as T

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = os.path.join(HERE, "..", "bot", "target", "release", "bot.exe")
REPLAYS = os.path.join(HERE, "..", "data", "bot", "gate4_garbage.jsonl")

# DESIGN.md's worked example (rows 10..1 from the top). (9,5) is an overhang zero.
TSD = T.board_from_ascii([
    ".....####.",
    "###...####",
    "####.#####",
    "###.######",
    "###.######",
    "###.######",
    "###.######",
    "###.######",
    "###.######",
    "###.######",
])


def to_replay(b):
    rows = ["." * 10] * 40
    for r in range(T.ROWS):
        rows[39 - r] = "".join("#" if b[r, c] else "." for c in range(10))
    return rows


def test_worked_example_base():
    toks, h = T.base_tokens(TSD)
    assert h.tolist() == [9, 9, 9, 8, 7, 10, 10, 10, 10, 9]
    kinds = [t.kind for t in toks]
    assert kinds == ["sky", "well", "play", "play", "play"], kinds
    sky, well = toks[0], toks[1]
    assert sky.mask.sum() == 100 and sky.box == (10, 19, 0, 9)
    assert well.extras["length"] == 7 and well.extras["hole"] == 3 and well.extras["covered"]
    assert T.check_partition(toks)


def test_worked_example_overlays():
    toks, h = T.base_tokens(TSD)
    pits = T.pit_tokens(TSD, h)
    assert [p.kind for p in pits] == ["pit2"], [p.kind for p in pits]
    cells = sorted(zip(*np.nonzero(pits[0].mask)))
    assert cells == [(7, 4), (8, 3), (8, 4)], cells       # 0-indexed rows: (8,4), (9,3), (9,4)


def test_structures_from_rust():
    if not os.path.exists(EXE):
        return
    s = T.structures_for([to_replay(TSD)], EXE)[0]
    kinds = [x["kind"] for x in s]
    assert "tsd" in kinds, s
    tsd = next(x for x in s if x["kind"] == "tsd")
    assert tsd["lines"] == 2
    toks = T.tokenize(TSD, "T", "IOSZJ", None, s)
    over = [t for t in toks if t.kind == "tsd"][0]
    assert sorted(zip(*np.nonzero(over.mask))) == [(7, 4), (8, 3), (8, 4), (8, 5)]
    # also an S-shaped recess: an upright S on the 8 / 7 step of columns 3-4 (it would bury the slot)
    fit = [t for t in toks if t.kind == "fit_s"][0]
    assert sorted(zip(*np.nonzero(fit.mask))) == [(7, 4), (8, 3), (8, 4), (9, 3)]
    assert len(toks) == 5 + 3 + 7, [t.kind for t in toks]


def test_sz_recesses():
    # position 10 of the viewer: Z flat on heights 11,10,10 (cols 2-4); position 11: Z upright on 4,5
    h10 = np.array([14, 13, 11, 10, 10, 11, 8, 8, 10, 12])
    b = np.zeros((T.ROWS, T.COLS), bool)
    for c, hc in enumerate(h10):
        b[:hc, c] = True
    fits = T.fit_tokens(b, T.heights(b))
    z = [t for t in fits if t.kind == "fit_z" and t.extras["orientation"] == "flat"]
    assert any(sorted(zip(*np.nonzero(t.mask))) == [(10, 3), (10, 4), (11, 2), (11, 3)] for t in z), fits
    h11 = np.array([8, 8, 6, 5, 5, 5, 6, 4, 5, 6])
    b = np.zeros((T.ROWS, T.COLS), bool)
    for c, hc in enumerate(h11):
        b[:hc, c] = True
    fits = T.fit_tokens(b, T.heights(b))
    assert any(t.kind == "fit_z" and t.extras["orientation"] == "upright" and t.box[2:] == (7, 8) for t in fits), fits


def test_rect12_and_well():
    # 3-wide 4-deep flat pit at columns 0-2, and a 1-wide well at column 9
    b = T.board_from_ascii([
        "...######.",
        "...######.",
        "...######.",
        "...######.",
        "#########.",
    ])
    toks = T.tokenize(b)
    kinds = sorted(t.kind for t in toks if t.kind in T.OVERLAY_KINDS)
    assert "rect12" in kinds and "pit1" in kinds, kinds
    rect = next(t for t in toks if t.kind == "rect12")
    assert rect.box == (1, 4, 0, 2)
    # bottom row is a single-hole row with an open hole column: the play area, not a well
    assert [t.kind for t in toks if t.base] == ["sky"] + ["play"] * 5


def test_partition_and_budget_on_replays():
    if not os.path.exists(REPLAYS):
        return
    n = 0
    with open(REPLAYS) as f:
        for line in f:
            rec = json.loads(line)
            b = T.board_from_replay(rec["board"])
            toks = T.tokenize(b, rec["piece"], rec["queue"], rec["hold"])
            assert T.check_partition(toks)
            assert sum(t.base for t in toks) <= T.K_BOARD
            n += 1
            if n >= 3000:
                break
    assert n > 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
