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
  data/v51plan/NAME.pc.npy    large (~6 kB a position), rebuilt here, NOT shipped (data_bundle skips it)
      (M, 5) records of bot plan-cands for plan steps 2..6: legal moves in the root's frame,
      their meta, the planned move's index (attentris-autoreg's queries and target)

Rows are matched by (teacher, seed, n, visible board), so the packing order of each set does not
matter, and every row of every set must be found. Checks printed per set: rows matched, plan
move 1 = the label, the plan lengths, and that every planned cell (root frame) is empty on the
root board.

    python attentris/pack_plan.py [--names train,val,...] [--jobs 64]
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


def read_file(job):
    """Per record of one raw file: its key, and the plan's steps (hold, piece, rot, cells)."""
    src, plan = job
    keys, plans = [], []
    with open(src) as fs, open(plan) as fp:
        for i, (ls, lp) in enumerate(zip(fs, fp)):
            r, p = json.loads(ls), json.loads(lp)
            assert p["i"] == i
            keys.append((TEACHERS.index(r["teacher"]), r["seed"], r["n"], visible_board(r["rows"]).tobytes()))
            plans.append([(s[0], s[1], s[2], [(s[3][2 * q], 39 - s[3][2 * q + 1]) for q in range(4)]) for s in p["plan"]])
    return keys, plans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default=",".join(SOURCES))
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
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
        pool.map(plan_cands, [(s, p, b) for (s, p), b in zip(pairs, bins)], chunksize=1)
        parts = pool.map(read_file, pairs, chunksize=4)
        keys = [k for ks, _ in parts for k in ks]
        plans = [p for _, ps in parts for p in ps]
        pc = np.concatenate([np.fromfile(b, PC).reshape(-1, K) for b in bins])
        assert len(pc) == len(keys), (src_set, len(pc), len(keys))
        index = {k: i for i, k in enumerate(keys)}
        # the rotation class of each (piece, shape), from the plans; must be one class per shape
        rclass = {}
        for pl in plans:
            for hold, piece, rot, cells in pl:
                key, c = (piece, shape(cells)), rot_class(piece, rot)
                assert rclass.setdefault(key, c) == c, ("two rotation classes for one shape", key)
        print("%s: %d records, %d files (%.0fs)" % (src_set, len(keys), len(files), time.time() - t0), flush=True)

        for name in set_names:
            with np.load(os.path.join(V51, name + ".npz")) as f:
                z = {k: f[k] for k in ("teacher", "seed", "n", "cells", "cand_cells", "cand_meta", "chosen")}
            m = len(z["n"])
            rows = np.array([index.get((int(t), int(s), int(n), c.tobytes()), -1)
                             for t, s, n, c in zip(z["teacher"], z["seed"], z["n"], z["cells"])])
            missing = int((rows < 0).sum())
            if missing:
                raise SystemExit("%s: %d of %d rows have no raw record" % (name, missing, m))
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
            np.savez(os.path.join(V51, name + ".plan.tmp.npz"), plan_len=plan_len, coord=coords)
            os.replace(os.path.join(V51, name + ".plan.tmp.npz"), os.path.join(V51, name + ".plan.npz"))
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
