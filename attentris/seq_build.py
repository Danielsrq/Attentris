"""Sequence builder for the multi-move (plan) variants: links each teacher position to the
positions that follow it in the same game, so step k of a plan has a target (the move the
teacher played k-1 moves later) and, for attentris-autoreg, that position's legal moves.

A link i -> j needs: same game (teacher, seed), n_j = n_i + 1, and the board of j equal to
the board of i after the chosen move (place its 4 cells, clear full rows). Garbage arriving
between the two moves, a dropped (doomed) position or a piece above the visible 20 rows
breaks the link, so a plan never runs across them.

Writes a sidecar next to each packed set, data/v51/<name>.seq.npz:
    nxt    (N, K) int32  index of the position k+1 moves later (k = 0..K-1), -1 if the chain is cut
    run    (N,)   int16  how many further positions follow without a cut (capped at K)
    played (N,)   int16  candidate index of the move actually played (it leads to the next
                         position's board); -1 when not linked. Differs from `chosen` where
                         the set was relabelled after the game (train / val come from
                         raw_v3, whose `chosen` is the relabelling search's pick, ~4% different)
Plan step 1 is the position itself; step k+1 is nxt[:, k-1]. Only teacher-game sets make
sense here (DAgger positions' next moves are the student's own).

    python attentris/seq_build.py                        # train, train1m, val (K = 5: plans up to N = 6)
    python attentris/seq_build.py --names val --k 5
"""
import argparse
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS, COLS = 20, 10


def after_move(cells, cand_cells, chosen):
    """Boards (M, 200) after each position's chosen move: its cells placed, full rows cleared
    (rows above fall down, empty rows enter at the top). ok = False when a cell lies above the
    visible board, where the next board cannot be predicted from the visible rows alone."""
    m = len(cells)
    idx = cand_cells[np.arange(m), chosen].astype(np.int64)        # (M, 4)
    return place(cells, idx)


def place(cells, idx):
    m = len(cells)
    ok = (idx >= 0).all(1)
    b = cells.astype(bool).copy()
    rows = np.repeat(np.arange(m), 4)
    flat = idx.reshape(-1)
    keep = flat >= 0
    b[rows[keep], flat[keep]] = True
    b = b.reshape(m, ROWS, COLS)
    full = b.all(2)                                                  # (M, 20), row 0 at the bottom
    order = np.argsort(full, axis=1, kind="stable")                  # kept rows first, bottom to top
    b = np.take_along_axis(b, order[:, :, None], axis=1)
    b[np.take_along_axis(full, order, axis=1)] = False               # cleared rows -> empty rows on top
    return b.reshape(m, ROWS * COLS), ok


def build(path, k, chunk=200_000):
    # an NpzFile re-reads an array on every z[key], so load the ones used once (~1 GB for train1m)
    with np.load(path) as f:
        z = {key: f[key] for key in ("teacher", "seed", "n", "cells", "cand_cells", "chosen", "n_cand")}
    teacher, seed, n = z["teacher"][:], z["seed"][:], z["n"][:]
    m = len(n)
    order = np.lexsort((n, seed, teacher))
    a, b = order[:-1], order[1:]
    same = (teacher[a] == teacher[b]) & (seed[a] == seed[b])
    step = same & (n[b] == n[a] + 1)
    link = np.full(m, -1, np.int64)
    played = np.full(m, -1, np.int16)
    why = {"game end": int((~same).sum()) + 1, "n gap (dropped position)": int((same & ~step).sum()),
           "above the board": 0, "board changed (garbage)": 0, "played != chosen": 0}
    cut_n = []
    for s in range(0, len(a), chunk):
        aa, bb, st = a[s:s + chunk], b[s:s + chunk], step[s:s + chunk]
        aa, bb = aa[st], bb[st]
        if len(aa) == 0:
            continue
        srt = np.argsort(aa)                    # mmap reads in file order
        aa, bb = aa[srt], bb[srt]
        pred, ok = after_move(z["cells"][aa], z["cand_cells"][aa], z["chosen"][aa])
        nb = np.argsort(bb)
        nxt_cells = np.empty((len(bb), ROWS * COLS), np.uint8)
        nxt_cells[nb] = z["cells"][bb[nb]]
        same_board = (pred == nxt_cells.astype(bool)).all(1)
        pl = np.where(ok & same_board, z["chosen"][aa], -1)
        # chosen does not lead to the next board: look for the candidate that does
        for i in np.where(~(ok & same_board))[0]:
            nc = int(z["n_cand"][aa[i]])
            cc = z["cand_cells"][aa[i], :nc].astype(np.int64)
            pb, pok = place(np.repeat(z["cells"][aa[i]][None], nc, 0), cc)
            hit = np.where(pok & (pb == nxt_cells[i].astype(bool)).all(1))[0]
            if len(hit):
                pl[i] = hit[0]
                why["played != chosen"] += 1
            elif not ok[i]:
                why["above the board"] += 1
            else:
                why["board changed (garbage)"] += 1
                cut_n.append(n[bb[i:i + 1]])
        good = pl >= 0
        link[aa[good]] = bb[good]
        played[aa[good]] = pl[good]
    nxt = np.full((m, k), -1, np.int64)
    cur = link
    for j in range(k):
        nxt[:, j] = cur
        cur = np.where(cur >= 0, link[np.maximum(cur, 0)], -1)
    run = (nxt >= 0).sum(1)       # chains are prefix-closed: a cut stays cut
    cut_n = np.concatenate(cut_n) if cut_n else np.zeros(0, np.int64)
    return nxt.astype(np.int32), run.astype(np.int16), played, why, cut_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(HERE, "data", "v51"))
    ap.add_argument("--names", default="train,train1m,val")
    ap.add_argument("--k", type=int, default=5, help="positions linked after each one (plan length N up to k + 1)")
    args = ap.parse_args()
    for name in args.names.split(","):
        t0 = time.time()
        path = os.path.join(args.dir, name + ".npz")
        nxt, run, played, why, cut_n = build(path, args.k)
        out = os.path.join(args.dir, name + ".seq.npz")
        tmp = out + ".tmp.npz"
        np.savez(tmp, nxt=nxt, run=run, played=played)
        os.replace(tmp, out)
        m = len(run)
        print("%s: %d positions, %.1fs -> %s" % (name, m, time.time() - t0, out))
        print("  links: " + ", ".join("%s %d" % kv for kv in why.items()))
        if len(cut_n):
            for e in (10, 15):
                print("  board-changed cuts landing on n %% %d == 0: %.1f%%" % (e, 100 * np.mean(cut_n % e == 0)))
        print("  full plan available from a position: " +
              ", ".join("N=%d %.1f%%" % (j + 1, 100 * np.mean(run >= j)) for j in range(1, args.k + 1)))


if __name__ == "__main__":
    main()
