"""Precompute the ToMe v3 pre-tokens (tome_tok.pretokenize) for packed data: per position a
(200,) int8 group id per cell -- 0 = sky, 1..MAX_BLOCKS = solid blocks, -1 = a single cell (later
blocks beyond MAX_BLOCKS stay single cells). Written next to the data as <name>.pretok.npy.

    python attentris/pretok.py --data attentris/data/v51 --files train.npz,dagger1.npz,train1m.npz,val.npz
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pcores
import tome_tok as TT

MAX_BLOCKS = 24


def groups(board):
    g = np.full(200, -1, np.int8)
    nb = 0
    for kind, cells in TT.pretokenize(board.reshape(20, 10)):
        if kind == "sky":
            g[cells] = 0
        elif kind == "block" and nb < MAX_BLOCKS:
            nb += 1
            g[cells] = nb
    return g


def work(chunk):
    pcores.pin(verbose=False)
    return np.stack([groups(b.astype(bool)) for b in chunk])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "v51"))
    ap.add_argument("--files", default="train.npz,dagger1.npz,train1m.npz,val.npz")
    ap.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args()
    pcores.pin()
    for f in args.files.split(","):
        out = os.path.join(args.data, f.replace(".npz", ".pretok.npy"))
        if os.path.exists(out):
            print("%s exists, skipped" % out)
            continue
        t0 = time.time()
        cells = np.load(os.path.join(args.data, f))["cells"]
        chunks = np.array_split(cells, max(1, len(cells) // 2000))
        with mp.get_context("spawn").Pool(args.jobs) as pool:
            g = np.concatenate(pool.map(work, chunks))
        np.save(out + ".tmp.npy", g)
        os.replace(out + ".tmp.npy", out)
        nt = (g == -1).sum(1) + np.array([len(np.unique(r[r >= 0])) for r in g[:5000]]).mean()
        print("%s: %d positions, %.0fs; single cells per position %.1f (+ sky and blocks) -> %s" % (
            f, len(g), time.time() - t0, (g == -1).sum(1).mean(), out), flush=True)


if __name__ == "__main__":
    main()
