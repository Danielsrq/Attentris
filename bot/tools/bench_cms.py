"""cms-RL-982k's decision time per move, single-threaded, no other load: his bot plays one
game alone in his simulator and every decide() call is timed (move enumeration + features +
the batched value-net pass over all options).

    cd external/tetrisbattle && set PYTHONPATH=. && .venv\\Scripts\\python ..\\..\\bot\\tools\\bench_cms.py --pieces 300
"""
import argparse
import os
import random
import statistics
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "attentris"))

import torch

import pcores
from battle_match import HisBot, new_player


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="best.pt")
    ap.add_argument("--pieces", type=int, default=300)
    args = ap.parse_args()
    pcores.pin()
    torch.set_num_threads(1)
    random.seed(1)
    bot = HisBot(args.model)
    s = new_player(0, 1)
    ts = []
    for _ in range(args.pieces):
        t0 = time.perf_counter()
        p = bot.decide(s)
        ts.append(time.perf_counter() - t0)
        if p is None or s.apply(p)["ko"]:
            break
    ts.sort()
    print("cms-RL-982k, 1 thread, %d moves: median %.1f ms, mean %.1f ms, p90 %.1f ms per move" % (
        len(ts), 1000 * statistics.median(ts), 1000 * statistics.mean(ts), 1000 * ts[int(0.9 * len(ts))]))


if __name__ == "__main__":
    main()
