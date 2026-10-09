"""Dump TetrisBattle decisions for the Rust parity check (`bot battle-parity`).

His bot (best.pt) plays games in his headless simulator, with random incoming garbage
for varied boards. For every decision we record the board, the current piece's full
placement set from his own search() (cells + T-spin flag), and the chosen placement's
outcome (lines, attack, combo / B2B before and after, perfect clear).

    cd external/tetrisbattle
    set PYTHONPATH=. & .venv\\Scripts\\python ..\\..\\bot\\tools\\battle_dump.py --games 20 --pieces 300 --out dump.jsonl
"""
import argparse
import json
import os
import random

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np

from TetrisBattle.ai.sim import SimPlayer
from TetrisBattle.ai.search import search
from TetrisBattle.ai.model import load_net, option_values, option_rewards, GAMMA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="best.pt")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--pieces", type=int, default=300)
    ap.add_argument("--garbage", type=float, default=0.15, help="chance per piece of a 1-4 line attack")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="battle_dump.jsonl")
    a = ap.parse_args()

    net, reward_cfg = load_net(a.model)
    n = 0
    with open(a.out, "w") as out:
        for g in range(a.games):
            random.seed(a.seed + g)
            np.random.seed(a.seed + g)
            s = SimPlayer()
            for _ in range(a.pieces):
                t = s.tetris
                grid = t.grid
                depth = len(grid[0])
                rows = ["".join("X" if grid[x][y] > 0 else "." for x in range(10)) for y in range(depth)]
                cur = t.block.block_type()
                res = search(grid, cur, t.px, t.py, t.block.current_shape_id, t.tspin)
                placements = [{"cells": [list(c) for c in cells], "t": int(tt)} for (cells, tt) in res.keys()]

                opts, b, sc, infos, tslots, frames = s.options()
                q = option_rewards(infos, tslots, reward_cfg)[0] + GAMMA * option_values(net, b, sc) * (infos[:, 5] == 0)
                p = opts[int(np.argmax(q))]
                rec = {"game": g, "rows": rows, "excess": depth - 20, "piece": cur, "px": t.px, "py": t.py,
                       "rot": t.block.current_shape_id, "tspin": int(t.tspin), "placements": placements,
                       "chosen": {"piece": p.piece_type, "cells": [list(c) for c in p.cells], "t": int(p.tspin),
                                  "hold": bool(p.use_hold)},
                       "combo_before": int(t.combo), "b2b_before": int(t.b2b_count)}
                o = s.apply(p)
                t = s.tetris
                rec["outcome"] = {"cleared": int(o["cleared"]), "attack": int(o["scores"]), "ko": bool(o["ko"]),
                                  "combo_after": int(t.combo), "b2b_after": int(t.b2b_count),
                                  "perfect": bool(o["cleared"] and not any(v > 0 for col in t.grid for v in col))}
                out.write(json.dumps(rec) + "\n")
                n += 1
                if o["ko"]:
                    break
                if random.random() < a.garbage:
                    s.add_garbage(random.randint(1, 4))
    print("wrote", n, "decisions to", a.out)


if __name__ == "__main__":
    main()
