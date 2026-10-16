"""CEM teacher's plan consistency on the val teacher games: how often the teacher's planned move k
(its depth-6 line at X, plan_label.sh) is the move played k-1 moves later, per step alone and for
steps 1..k all; move-based and board-based (the board after k moves equal: order-free). Windows
where all 6 steps are on the board (no garbage, no game end inside) only, the same windows for
every k. Step 1 < 100%: the plan is searched offline without the bag order the game's search knew.

    python attentris/plan_consistency_cem.py [--set val]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from seq_build import place  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="val")
    args = ap.parse_args()
    with np.load(os.path.join(HERE, "data", "v51", args.set + ".npz")) as f:
        seed, n, cells, cc = f["seed"], f["n"], f["cells"], f["cand_cells"]
    q = np.load(os.path.join(HERE, "data", "v51", args.set + ".seq.npz"))
    nxt, played = q["nxt"], q["played"]
    pos = {(int(s), int(m)): i for i, (s, m) in enumerate(zip(seed, n))}
    vis = lambda c: [(39 - c[2 * j + 1]) * 10 + c[2 * j] if 0 <= 39 - c[2 * j + 1] < 20 else -2 for j in range(4)]
    K = 6
    alone, prefix, board, tot = np.zeros(K), np.zeros(K), np.zeros(K), 0
    games = set()
    for pf in sorted(glob.glob(os.path.join(HERE, "data", "plan", "raw", "*.jsonl"))):
        for line in open(pf):
            p = json.loads(line)
            i = pos.get((p["seed"], p["n"]))
            if i is None or len(p["plan"]) < K or (nxt[i, :K - 1] < 0).any():
                continue
            chain = [i] + [int(j) for j in nxt[i, :K - 1]]
            tot += 1
            games.add(p["seed"])
            ok, b = True, cells[i][None].copy()
            for k in range(K):
                j = chain[k]
                hit = sorted(vis(p["plan"][k][3])) == sorted(cc[j, played[j]].tolist())
                ok = ok and hit
                alone[k] += hit
                prefix[k] += ok
                if k < K - 1:                       # board after k+1 planned moves vs the real board then
                    b, good = place(b, np.array([vis(p["plan"][k][3])]))
                    b = b.astype(np.uint8)
                    board[k] += bool(good[0]) and (b[0] == cells[chain[k + 1]]).all()     # after k+1 moves
    print("CEM teacher plan consistency, %s: %d games, %d windows with all %d steps on the board" % (args.set, len(games), tot, K))
    print("step k                       " + " ".join("%6d" % (k + 1) for k in range(K)))
    print("step k = move played (alone) " + " ".join("%5.1f%%" % (100 * x / tot) for x in alone))
    print("steps 1..k all match         " + " ".join("%5.1f%%" % (100 * x / tot) for x in prefix))
    print("board after k moves equal    " + " ".join("%5.1f%%" % (100 * x / tot) for x in board[:K - 1]) + "      -")


if __name__ == "__main__":
    main()
