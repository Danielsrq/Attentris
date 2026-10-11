"""What led to each death in a play.py --dump run: per dying game, the last moves' stack height,
holes, well depth and pieces, and the final board (row 1 at the bottom).

    python attentris/deaths.py data/deaths_v51.txt data/deaths_v51.jsonl
"""
import collections
import json
import sys

import numpy as np

sys.path_insert = None
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pack as P
import tokenizer as T


def board_of(s):
    return T.board_from_replay(P.rows_to_strings(s["rows"]))


def main():
    games_file, states_file = sys.argv[1], sys.argv[2]
    last = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    dead = {r["seed"]: r for r in (json.loads(l) for l in open(games_file) if l.startswith("{")) if r["died"]}
    by = collections.defaultdict(list)
    for l in open(states_file):
        s = json.loads(l)
        if s["seed"] in dead:
            by[s["seed"]].append(s)
    for seed in sorted(dead):
        ss = sorted(by[seed], key=lambda s: s["n"])
        print("== seed %d: died after %d pieces" % (seed, dead[seed]["pieces"]))
        for s in ss[-last:]:
            b = board_of(s)
            h = T.heights(b)
            well = int(np.max(np.minimum(np.r_[20, h[:-1]], np.r_[h[1:], 20]) - h))
            print("   n %3d  max h %2d  holes %2d  deepest gap %2d  piece %s hold %s next %s" % (
                s["n"], h.max(), T.holes_of(b), well, s["piece"], s["hold"], "".join(s["queue"][:3])))
        b = board_of(ss[-1])
        top = min(19, max([r for r in range(20) if b[r].any()] + [0]))
        print("   last board:")
        for r in range(top, -1, -1):
            print("     %2d %s" % (r + 1, "".join("#" if b[r, c] else "." for c in range(10))))


if __name__ == "__main__":
    main()
