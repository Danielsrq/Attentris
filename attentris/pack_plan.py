"""Plan targets for the multi-move variants, aligned row by row with the packed v5.1 sets.

From the plan labels (plan_label.sh: the teacher's planned line, depth 6, for every position)
this writes, per packed set NAME (data/v51/NAME.npz):

  data/v51/NAME.plan.npz      small, shipped with the data
      plan_len  (M,)    int8   plan steps usable from this position (1..6): step 1 is the
                               labelled move; steps 2.. only when the plan's move 1 IS that
                               label (else the plan continues another move: plan_len 1) and
                               while the step was replayed (bot plan-cands found it)
      coord     (M, 6)  int16  attentris-plan's target per step: hold*880 + rot*220 + x*22 + y
                               (1760 classes), on that step's own board; rot = the shape class
                               (O: 0, I / S / Z: 0-1, T / L / J: 0-3), x = leftmost column,
                               y = lowest row (0 = bottom); -1 when no step or y > 21
      piece_step  (M, 7) int8   per known piece (current, hold, preview 1..5): the plan step that places it
                                (1..6), 0 = not placed in a full 6-step plan (it stays in hold, or the
                                plan does not reach it), -1 = unknown (a shorter plan) or no piece (hold
                                empty); replayed from the plan's hold choices over the queue
      piece_place (M, 7) int16  its placement on its step's own board: rot*220 + x*22 + y (880 classes,
                                coord without the hold bit), -1 when not placed / unknown
  data/v51plan/NAME.pc.npy    large (~6 kB a position), rebuilt here, NOT shipped (data_bundle skips it)
      (M, 5) records of bot plan-cands for plan steps 2..6: legal moves in the root's frame,
      their meta, the planned move's index (attentris-autoreg's queries and target)

Rows are matched by (teacher, seed, n, visible board), so the packing order of each set does not
matter, and every row of every set must be found. Checks printed per set: rows matched, plan
move 1 = the label, the plan lengths, and that every planned cell (root frame) is empty on the
root board.

    python attentris/pack_plan.py [--names train,val,...] [--jobs 64]
    python attentris/pack_plan.py --pieces-only     # only (re)compute piece_step / piece_place, keep the rest
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
V51 = os.path.join(DATA, "v51")
OUT_BIG = os.path.join(DATA, "v51plan")
BOT = os.path.join(HERE, "..", "bot", "target", "release", "bot" + (".exe" if os.name == "nt" else ""))
sys.path.insert(0, HERE)
from pack import TEACHERS  # noqa: E402

PIECES = "IOTSZJL"
K = 5            # plan steps after the first
NC = 96
PC = np.dtype([("n", "<i2"), ("target", "<i2"), ("rot", "<i2"), ("own", "<i2", 8),
               ("cells", "<i2", (NC, 4)), ("meta", "i1", (NC, 5))])
# packed set -> the raw label directory (relative to data/) its records came from
SOURCES = {
    "train": "raw", "val": "raw", "train1m": "raw_1m", "dagger1": "raw_dagger1",
    "dagger2": "raw_dagger2_tr", "val_dagger2": "raw_dagger2_val",
    "danger": "dagger3/raw_tr", "val_danger": "dagger3/raw_val",
    "dagger4": "dagger4/raw_tr", "val_dagger4": "dagger4/raw_val",
    "dagger5all": "dagger5all/raw_tr", "val_dagger5all": "dagger5all/raw_val",
}


def rot_class(piece, rot):
    return 0 if piece == "O" else rot % 2 if piece in "ISZ" else rot


def shape(cells_xy):
    """Normalised shape of 4 cells (x, row from the bottom)."""
    x0, y0 = min(c[0] for c in cells_xy), min(c[1] for c in cells_xy)
    return tuple(sorted((x - x0, y - y0) for x, y in cells_xy))


def coord(hold, rclass, cells_xy):
    x, y = min(c[0] for c in cells_xy), min(c[1] for c in cells_xy)
    if not (0 <= x < 10 and 0 <= y < 22):
        return -1
    return hold * 880 + rclass * 220 + x * 22 + y


def visible_board(rows40):
    """(200,) uint8 like pack.py: row 0 = bottom, from the bot's 40 rows (top to bottom, bit x = column x)."""
    b = np.zeros(200, np.uint8)
    for r in range(20):
        v = rows40[39 - r]
        for c in range(10):
            if v >> c & 1:
                b[r * 10 + c] = 1
    return b


def plan_cands(job):
    """bot plan-cands for one raw file (skipped when its .bin exists)."""
    src, plan, out = job
    if os.path.exists(out):
        return
    tmp = out + ".tmp"
    subprocess.run([BOT, "plan-cands", "--mode", "normal", "--any-cores", "--states", src, "--plans", plan,
                    "--out", tmp, "--k", str(K), "--nc", str(NC)], check=True, stderr=subprocess.DEVNULL)
    os.replace(tmp, out)


def piece_targets(hold_empty, plan_len, holds, coords):
    """Which known piece each plan step places: slots 0 = current, 1 = hold, 2..6 = previews 1..5.
    holds[k] = step k+1's hold flag. Returns (step per slot (7,), placement per slot (7,))."""
    step = np.full(7, -1, np.int8)
    place = np.full(7, -1, np.int16)
    cur, held, nxt = 0, (None if hold_empty else 1), 2
    done = 0
    for k in range(plan_len):
        if not holds[k]:
            slot, cur, nxt = cur, nxt, nxt + 1
        elif held is None:                          # hold empty: the next piece is played, current held
            slot, held, cur, nxt = nxt, cur, nxt + 1, nxt + 2
        else:
            slot, held, cur, nxt = held, cur, nxt, nxt + 1
        if slot > 6:                                # a piece not yet visible: stop here
            break
        step[slot] = k + 1
        place[slot] = coords[k] % 880 if coords[k] >= 0 else -1
        done = k + 1
    if done == K + 1:                               # a full plan: every other known piece is not placed
        step[(step < 0)] = 0
        if hold_empty:
            step[1] = -1
    return step, place


def read_file(job):
    """Per record of one raw file: its key, and the plan's steps (hold, piece, rot, cells)."""
    src, plan = job
    keys, plans = [], []
    with open(src) as fs, open(plan) as fp:
        for i, (ls, lp) in enumerate(zip(fs, fp)):
            r, p = json.loads(ls), json.loads(lp)
            assert p["i"] == i
            keys.append((TEACHERS.index(r["teacher"]), r["seed"], r["n"], visible_board(r["rows"]).tobytes(), r["hold"] is None))
            plans.append([(s[0], s[1], s[2], [(s[3][2 * q], 39 - s[3][2 * q + 1]) for q in range(4)]) for s in p["plan"]])
    return keys, plans


def write_pieces(name, z, rows, plans, hold_empty, plan_len, coords):
    """NAME.plan.npz: plan_len, coord, and the piece targets (piece_targets) of every row."""
    m = len(rows)
    ps, pp = np.full((m, 7), -1, np.int8), np.full((m, 7), -1, np.int16)
    for i in range(m):
        j = rows[i]
        holds = [int(z["cand_meta"][i, z["chosen"][i], 1])] + [s[0] for s in plans[j][1:]]
        ps[i], pp[i] = piece_targets(bool(hold_empty[j]), int(plan_len[i]), holds, coords[i])
    # check: the piece token each step was assigned is the piece the plan places there
    kinds = np.arange(20, 27)                 # current, hold, preview1..5 (pack.KIND_ID)
    slot_piece = np.stack([z["tok_piece"][np.arange(m), np.argmax(z["tok_kind"] == k, 1)] for k in kinds], 1)
    bad = tot = 0
    for i in range(m):
        pl = plans[rows[i]]
        for slot in range(7):
            k = ps[i, slot]
            if k >= 2:                       # steps 2.. come from the plan (step 1 = the label)
                tot += 1
                bad += PIECES.index(pl[k - 1][1]) != slot_piece[i, slot]
    print("    %s: %d plan steps assigned to piece tokens, %d of a different piece type" % (name, tot, bad), flush=True)
    tmp = os.path.join(V51, name + ".plan.tmp.npz")
    np.savez(tmp, plan_len=plan_len, coord=coords, piece_step=ps, piece_place=pp)
    os.replace(tmp, os.path.join(V51, name + ".plan.npz"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default=",".join(SOURCES))
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--pieces-only", action="store_true", help="only add piece_step / piece_place to the existing NAME.plan.npz")
    args = ap.parse_args()
    names = args.names.split(",")
    os.makedirs(OUT_BIG, exist_ok=True)
    t0 = time.time()
    by_src = {}
    for name in names:
        by_src.setdefault(SOURCES[name], []).append(name)
    pool = Pool(args.jobs)
    for src_set, set_names in by_src.items():
        files = sorted(glob.glob(os.path.join(DATA, src_set, "*.jsonl")))
        files = [f for f in files if os.path.getsize(f) > 0]
        pairs = [(f, os.path.join(DATA, "plan", src_set, os.path.basename(f))) for f in files]
        bindir = os.path.join(OUT_BIG, "bin", src_set)
        os.makedirs(bindir, exist_ok=True)
        bins = [os.path.join(bindir, os.path.basename(f)[:-6] + ".bin") for f in files]
        if not args.pieces_only:
            pool.map(plan_cands, [(s, p, b) for (s, p), b in zip(pairs, bins)], chunksize=1)
        parts = pool.map(read_file, pairs, chunksize=4)
        keys = [k for ks, _ in parts for k in ks]
        plans = [p for _, ps in parts for p in ps]
        if not args.pieces_only:
            pc = np.concatenate([np.fromfile(b, PC).reshape(-1, K) for b in bins])
            assert len(pc) == len(keys), (src_set, len(pc), len(keys))
        hold_empty = np.array([k[4] for k in keys])
        index = {k[:4]: i for i, k in enumerate(keys)}
        # the rotation class of each (piece, shape), from the plans; must be one class per shape
        rclass = {}
        for pl in plans:
            for hold, piece, rot, cells in pl:
                key, c = (piece, shape(cells)), rot_class(piece, rot)
                assert rclass.setdefault(key, c) == c, ("two rotation classes for one shape", key)
        print("%s: %d records, %d files (%.0fs)" % (src_set, len(keys), len(files), time.time() - t0), flush=True)

        for name in set_names:
            with np.load(os.path.join(V51, name + ".npz")) as f:
                z = {k: f[k] for k in ("teacher", "seed", "n", "cells", "cand_cells", "cand_meta", "chosen", "tok_kind", "tok_piece")}
            m = len(z["n"])
            rows = np.array([index.get((int(t), int(s), int(n), c.tobytes()), -1)
                             for t, s, n, c in zip(z["teacher"], z["seed"], z["n"], z["cells"])])
            missing = int((rows < 0).sum())
            if missing:
                raise SystemExit("%s: %d of %d rows have no raw record" % (name, missing, m))
            if args.pieces_only:
                with np.load(os.path.join(V51, name + ".plan.npz")) as f:
                    plan_len, coords = f["plan_len"], f["coord"]
                write_pieces(name, z, rows, plans, hold_empty, plan_len, coords)
                print("  %-15s %8d rows: piece targets (%.0fs)" % (name, m, time.time() - t0), flush=True)
                continue
            plan_len = np.ones(m, np.int8)
            coords = np.full((m, K + 1), -1, np.int16)
            same1 = 0
            occupied = 0
            for i in range(m):
                j = rows[i]
                pl = plans[j]
                ch = z["chosen"][i]
                cc = z["cand_cells"][i, ch]
                hold, piece = int(z["cand_meta"][i, ch, 1]), PIECES[z["cand_meta"][i, ch, 0]]
                if (cc >= 0).all():
                    xy = [(int(c) % 10, int(c) // 10) for c in cc]
                    coords[i, 0] = coord(hold, rclass[(piece, shape(xy))], xy)
                if not pl:
                    continue
                h0, p0, _, c0 = pl[0]
                vis0 = sorted(y * 10 + x if 0 <= y < 20 else -2 for x, y in c0)
                if not (h0 == hold and vis0 == sorted(cc.tolist())):
                    continue
                same1 += 1
                for k in range(1, min(len(pl), K + 1)):
                    rec = pc[j, k - 1]
                    if rec["n"] <= 0 or rec["target"] < 0:
                        break
                    h, p, rot, cells = pl[k]
                    coords[i, k] = coord(h, rclass[(p, shape(cells))], cells)
                    t = rec["cells"][rec["target"]]
                    occupied += int(z["cells"][i][t[t >= 0]].any())
                    plan_len[i] = k + 1
            write_pieces(name, z, rows, plans, hold_empty, plan_len, coords)
            big = np.lib.format.open_memmap(os.path.join(OUT_BIG, name + ".pc.tmp.npy"), mode="w+", dtype=PC, shape=(m, K))
            for s in range(0, m, 100_000):
                big[s:s + 100_000] = pc[rows[s:s + 100_000]]
            big.flush()
            del big
            os.replace(os.path.join(OUT_BIG, name + ".pc.tmp.npy"), os.path.join(OUT_BIG, name + ".pc.npy"))
            print("  %-15s %8d rows matched; plan move 1 = label %.2f%%; plan_len %s; coord -1 at a used step %d; "
                  "planned cells on an occupied root cell %d (%.0fs)" % (
                      name, m, 100 * same1 / m,
                      " ".join("%d:%.1f%%" % (L, 100 * np.mean(plan_len >= L)) for L in range(2, K + 2)),
                      int(((coords < 0) & (np.arange(K + 1)[None] < plan_len[:, None])).sum()), occupied,
                      time.time() - t0), flush=True)
    pool.close()


if __name__ == "__main__":
    main()
