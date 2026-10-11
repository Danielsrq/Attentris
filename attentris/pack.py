"""Tokenize the raw dataset (attentris/data/raw/*.jsonl, from `bot dataset`) into training
arrays: attentris/data/{train,val}.npz. Games are split by seed, never by position, so
validation positions come from games the model has not seen.

    python attentris/pack.py [--val-every 10]

Per position (N rows), board-side tokens padded to NT, candidates padded to NC:
  cells       (N, 200)        uint8   board, row-major from the bottom row
  tok_kind    (N, NT)         int8    index into tokenizer.KINDS, -1 = padding
  tok_mask    (N, NT, 25)     uint8   np.packbits of the 200-cell mask
  tok_box     (N, NT, 4)      int8    row lo, row hi, col lo, col hi (-1 for pieces)
  tok_extra   (N, NT, NE)     float16 kind-specific numbers (see extras())
  tok_piece   (N, NT)         int8    piece id for piece tokens (7 = none), else -1
  cand_cells  (N, NC, 4)      int16   cell index r*10+c; -2 = above the visible board; -1 pad
  cand_meta   (N, NC, 5)      int8    piece, hold, spin (0/1/2), lines, attack
  cand_score  (N, NC, 2)      float32 1-ply score, best surviving search score (nan = pruned)
  n_cand      (N,)            int16
  chosen      (N,)            int16   index of the bot's move
  teacher     (N,)            int8    0 = normal, 1 = gate4
  seed, n     (N,)            int32
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pcores
import tokenizer as T

HERE = os.path.dirname(os.path.abspath(__file__))
NT = T.K_BOARD + T.P_MAX + 7
NC = 96
NE = 12
PIECE_ID = {p: i for i, p in enumerate(T.PIECES)}
TEACHERS = ("normal", "gate4", "danger")   # danger: CEM-survival on the students' trouble positions (DAgger 3)
KIND_ID = {k: i for i, k in enumerate(T.KINDS)}
FIT_ORIENT = ("flat", "upright", "nub down", "three on top", "hook on top")


def extras(t):
    """Fixed-length numbers per token, scaled to about 0..1."""
    e = np.zeros(NE, np.float32)
    x = t.extras
    k = t.kind
    if k == "sky":
        e[:10] = np.array(x["h"]) / 20.0
    elif k == "well":
        e[0], e[1], e[2] = x["length"] / 20.0, x["hole"] / 9.0, float(x["covered"])
    elif k in ("play", "band"):
        rows = x["rows"]
        e[0], e[1], e[2] = rows / 20.0, x["filled"] / (10.0 * rows), x["under"] / (10.0 * rows)
        if k == "band":
            e[3] = len(x["merged"]) / 20.0
    elif k in T.STRUCTURE_KINDS:
        e[0], e[1] = x["lines"] / 3.0, x["fill"]
    elif k == "rect12":
        e[0] = float(x["orientation"] == "4x3")
    elif k in T.SLOT_KINDS:
        if x["orientation"] in FIT_ORIENT:
            e[FIT_ORIENT.index(x["orientation"])] = 1.0
        e[5], e[6] = x["contacts"] / 8.0, x["gain"] / 6.0
        e[7], e[8] = float(x.get("covered", 0)), x.get("lines", 0) / 3.0
    elif k in T.SETUP_KINDS:
        e[0] = x["lines"] / 3.0
        e[1], e[2] = x.get("slot_r", 0.0) / 20.0, x.get("slot_c", 0.0) / 10.0   # where the slot will be (v5.1)
    elif k == "hang":
        e[0], e[1] = x["lines"] / 3.0, x["slots"] / 4.0
    elif k.startswith("pit"):
        e[0], e[1], e[2], e[3] = x["width"] / 4.0, x["depth"] / 20.0, x["level"] / 20.0, x["area"] / 40.0
    return e


def rows_to_strings(rows):
    return ["".join("#" if (r >> c) & 1 else "." for c in range(10)) for r in rows]


DOOMED = -500.0   # every move's search score below this: a lost position (all lines top out)


def doomed(r):
    best = [c[7] for c in r["cands"] if c[7] is not None]
    return bool(best) and max(best) < DOOMED


def pack(recs):
    """Arrays for the records; lost positions (see DOOMED) are left out -- with every line
    topping out, the teacher's pick among them is arbitrary and teaches nothing."""
    kept = [r for r in recs if not doomed(r)]
    lost = len(recs) - len(kept)
    recs = kept
    n = len(recs)
    out = {
        "cells": np.zeros((n, 200), np.uint8),
        "tok_kind": np.full((n, NT), -1, np.int8),
        "tok_mask": np.zeros((n, NT, 25), np.uint8),
        "tok_box": np.full((n, NT, 4), -1, np.int8),
        "tok_extra": np.zeros((n, NT, NE), np.float16),
        "tok_piece": np.full((n, NT), -1, np.int8),
        "cand_cells": np.full((n, NC, 4), -1, np.int16),
        "cand_meta": np.zeros((n, NC, 5), np.int8),
        "cand_score": np.full((n, NC, 2), np.nan, np.float32),
        "n_cand": np.zeros(n, np.int16),
        "chosen": np.zeros(n, np.int16),
        "teacher": np.zeros(n, np.int8),
        "seed": np.zeros(n, np.int32),
        "n": np.zeros(n, np.int32),
    }
    boards40 = [rows_to_strings(r["rows"]) for r in recs]
    structs = T.structures_for(boards40, pieces=[(r["piece"], r["queue"], r["hold"]) for r in recs])
    dropped = 0
    for i, (r, b40, s) in enumerate(zip(recs, boards40, structs)):
        b = T.board_from_replay(b40)
        toks = T.tokenize(b, r["piece"], r["queue"], r["hold"], s)
        out["cells"][i] = b.reshape(-1)
        for j, t in enumerate(toks[:NT]):
            out["tok_kind"][i, j] = KIND_ID[t.kind]
            out["tok_mask"][i, j] = np.packbits(t.mask.reshape(-1))
            if t.box is not None:
                out["tok_box"][i, j] = t.box
            out["tok_extra"][i, j] = extras(t)
            if t.kind in T.PIECE_KINDS:
                p = t.extras["piece"]
                out["tok_piece"][i, j] = PIECE_ID[p] if p else 7
        cands = r["cands"]
        if len(cands) > NC:
            # keep the chosen move and the best 1-ply scores
            order = sorted(range(len(cands)), key=lambda k: (k != r["chosen"], -cands[k][6]))[:NC]
            dropped += len(cands) - NC
        else:
            order = list(range(len(cands)))
        for j, k in enumerate(order):
            hold, piece, rot, spin, lines, attack, ply1, best, cells = cands[k]
            idx = []
            for q in range(4):
                x, y = cells[2 * q], cells[2 * q + 1]
                rr = 39 - y
                idx.append(rr * 10 + x if 0 <= rr < T.ROWS else -2)
            out["cand_cells"][i, j] = idx
            out["cand_meta"][i, j] = (PIECE_ID[piece], hold, spin, lines, min(int(attack), 127))
            out["cand_score"][i, j] = (ply1, np.nan if best is None else best)
            if k == r["chosen"]:
                out["chosen"][i] = j
        out["n_cand"][i] = len(order)
        out["teacher"][i] = TEACHERS.index(r["teacher"])
        out["seed"][i], out["n"][i] = r["seed"], r["n"]
    out["_lost"] = np.array([lost])
    return out, dropped


def pack_file(path):
    return pack([json.loads(l) for l in open(path)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(HERE, "data", "raw"))
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    ap.add_argument("--val-every", type=int, default=10, help="every k-th game (by seed) goes to validation")
    ap.add_argument("--only", default="", help="pack just this raw directory into OUT/NAME.npz, all of it (e.g. DAgger labels)")
    ap.add_argument("--name", default="dagger")
    ap.add_argument("--jobs", type=int, default=1, help="with --only: pack the game files in this many processes")
    args = ap.parse_args()
    pcores.pin()
    if args.only:
        files = sorted(glob.glob(os.path.join(args.only, "*.jsonl")))
        if args.jobs > 1:
            # one file at a time per process: a million JSON records at once would not fit
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(args.jobs) as pool:
                parts = pool.map(pack_file, files, chunksize=4)
            arrs = {k: np.concatenate([p[0][k] for p in parts]) for k in parts[0][0]}
            dropped = sum(p[1] for p in parts)
        else:
            recs = [json.loads(l) for f in files for l in open(f)]
            arrs, dropped = pack(recs)
        lost = int(arrs.pop("_lost").sum())
        path = os.path.join(args.out, args.name + ".npz")
        np.savez_compressed(path, **arrs)
        teach = np.bincount(arrs["teacher"], minlength=2)
        print("%s: %d positions (normal %d, gate4 %d), %d lost positions left out, %d candidates over NC dropped -> %s" % (
            args.name, len(arrs["chosen"]), teach[0], teach[1], lost, dropped, path))
        return
    files = sorted(glob.glob(os.path.join(args.raw, "*.jsonl")))
    splits = {"train": [], "val": []}
    for f in files:
        seed = int(os.path.basename(f).rsplit("_s", 1)[1].split(".")[0])
        recs = [json.loads(l) for l in open(f)]
        splits["val" if seed % args.val_every == 0 else "train"].extend(recs)
    for name, recs in splits.items():
        t0 = time.time()
        arrs, dropped = pack(recs)
        arrs.pop("_lost")
        path = os.path.join(args.out, name + ".npz")
        np.savez_compressed(path, **arrs)
        teach = np.bincount(arrs["teacher"], minlength=2)
        print("%s: %d positions (normal %d, gate4 %d), %d games, %.0fs, %d candidates over NC=%d dropped -> %s (%.0f MB)" % (
            name, len(recs), teach[0], teach[1], len({(r["teacher"], r["seed"]) for r in recs}), time.time() - t0,
            dropped, NC, path, os.path.getsize(path) / 1e6))


if __name__ == "__main__":
    main()
