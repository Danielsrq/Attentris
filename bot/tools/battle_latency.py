"""How slow would his value net be inside our beam search?

1. Plays a few pieces with our bot (`bot serve-battle --mode battle`) and records how many nodes
   (afterstates evaluated) its full search uses per piece.
2. Times his ValueNet (best.pt) on batches of real afterstate features of various sizes,
   single-threaded and with all torch threads.

    cd external/tetrisbattle && PYTHONPATH=. .venv/Scripts/python ../../bot/tools/battle_latency.py --exe ../../bot/target/release/bot.exe --weights ../../bot/results/battle/battle.txt
"""
import argparse
import json
import os
import subprocess
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import torch

from TetrisBattle.ai.model import load_net, option_values
from TetrisBattle.ai.sim import SimPlayer
from TetrisBattle.ai.search import enumerate_placements


def our_nodes(exe, weights, pieces, extra):
    p = subprocess.Popen([exe, "serve-battle", "--mode", "battle", "--weights", weights, "--threads", "1"] + extra,
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    s = SimPlayer(0)
    nodes, secs = [], []
    for _ in range(pieces):
        t = s.tetris
        grid = t.grid
        depth = len(grid[0])
        req = {"rows": ["".join("X" if grid[x][y] > 0 else "." for x in range(10)) for y in range(depth)],
               "excess": depth - 20, "piece": t.block.block_type(),
               "queue": [q.block_type() for q in t.buffer.now_list[:5]],
               "hold": t.held.block_type() if t.held is not None else None,
               "combo": t.combo, "b2b": t.b2b_count}
        t0 = time.perf_counter()
        p.stdin.write(json.dumps(req) + "\n")
        p.stdin.flush()
        rep = json.loads(p.stdout.readline())
        secs.append(time.perf_counter() - t0)
        if rep.get("dead"):
            break
        nodes.append(rep["nodes"])
        want = (tuple(sorted(tuple(c) for c in rep["cells"])), bool(rep["hold"]))
        opts = enumerate_placements(t)
        pick = next(o for o in opts if (tuple(sorted(tuple(c) for c in o.cells)), bool(o.use_hold)) == want)
        if s.apply(pick)["ko"]:
            break
    p.stdin.close()
    p.wait()
    return np.array(nodes), np.array(secs), s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="best.pt")
    ap.add_argument("--exe", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--pieces", type=int, default=60)
    args = ap.parse_args()

    for label, extra in (("full search", []), ("depth 1", ["--depth", "1", "--no-chance"])):
        nodes, secs, _ = our_nodes(args.exe, args.weights, args.pieces, extra)
        print("ours, %-11s nodes/piece mean %7.0f  median %7.0f  max %7.0f   time/piece %.1f ms (1 thread)" % (
            label, nodes.mean(), np.median(nodes), nodes.max(), 1000 * secs.mean()))
        if label == "full search":
            full_nodes = nodes.mean()

    # real afterstate features: the options of a mid-game position
    _, _, s = our_nodes(args.exe, args.weights, 30, [])
    opts, b, sc, infos, tslots, frames = s.options()
    print("his options per piece here: %d" % len(opts))
    net, _ = load_net(args.model)

    def batch(n):
        idx = np.arange(n) % len(opts)
        return np.ascontiguousarray(b[idx]), np.ascontiguousarray(sc[idx])

    for threads in (1, torch.get_num_threads() if torch.get_num_threads() > 1 else 8):
        torch.set_num_threads(threads)
        print("-- torch threads %d" % threads)
        for n in (1, 40, 256, 1024, 4096, 16384):
            bb, ss = batch(n)
            option_values(net, bb, ss)  # warm-up
            reps = max(3, int(2000 / n))
            t0 = time.perf_counter()
            for _ in range(reps):
                option_values(net, bb, ss)
            dt = (time.perf_counter() - t0) / reps
            print("batch %6d: %8.2f ms  (%6.1f us per state)" % (n, 1000 * dt, 1e6 * dt / n))
        bb, ss = batch(4096)
        t0 = time.perf_counter()
        for _ in range(3):
            option_values(net, bb, ss)
        per = (time.perf_counter() - t0) / 3 / 4096
        print("=> our full search (%.0f nodes/piece) with his net as V: ~%.0f ms per piece (batched, %d threads)" % (
            full_nodes, 1000 * per * full_nodes, threads))


if __name__ == "__main__":
    main()
