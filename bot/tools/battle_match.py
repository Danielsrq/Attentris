"""Bot vs bot in TetrisBattle's headless simulator: his RL bot (best.pt) against ours
(`bot serve --mode battle`, whose move generator reproduces his placement set exactly).

Both players get the same piece sequence and the same garbage-hole sequence. Attacks
cancel incoming garbage first; what is left lands at the receiver's next lock, as in his
self-play. A game ends when someone tops out (the survivor wins) or both reach --pieces.

Time (--speed):
  turn        one piece each in turn: equal speed, thinking time ignored
  realtime    a piece takes max(thinking time, execution time) -- execution = his engine's
              estimate of the frames the key presses take (Placement.frames, 100 FPS)
  cap:X       as realtime, but never faster than X pieces per second (both players)

    cd external/tetrisbattle
    set PYTHONPATH=.
    .venv\\Scripts\\python ..\\..\\bot\\tools\\battle_match.py --games 20 --speed turn --weights ..\\..\\bot\\weights\\normal.txt

VS score (as I understand TETR.IO's): 100 x (attack sent + garbage lines cleared) / seconds.
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import subprocess
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np

from TetrisBattle.settings import FPS, POSSIBLE_KEYS, PIECES_DICT
from TetrisBattle.tetris import Buffer, Piece, garbage_holes
from TetrisBattle.ai.sim import SimPlayer
from TetrisBattle.ai.search import enumerate_placements
from TetrisBattle.ai.model import load_net, option_values, option_rewards, GAMMA


class SeededBuffer(Buffer):
    """A 7-bag drawn from its own RNG, so two players can share one sequence."""

    def __init__(self, rng):
        self.rng = rng
        super().__init__()

    def fill(self, _list):
        keys = list(POSSIBLE_KEYS)
        self.rng.shuffle(keys)
        for key in keys:
            _list.append(Piece(key, PIECES_DICT[key]))


def new_player(pid, seed):
    s = SimPlayer(pid)
    t = s.tetris
    t.buffer = SeededBuffer(random.Random(seed))
    t.held = None
    t.block = t.buffer.new_block()
    t.reset_pos()
    s.hole_rng = random.Random(seed * 7919 + 1)
    return s


def garbage_rows(grid):
    return sum(1 for y in range(len(grid[0])) if any(grid[x][y] == 8 for x in range(10)))


class HisBot:
    name = "his_rl"

    def __init__(self, model, budget_frames=0):
        self.net, self.cfg = load_net(model)
        self.budget = budget_frames

    def decide(self, s):
        opts, b, sc, infos, tslots, frames = s.options(budget_frames=self.budget)
        r, done = option_rewards(infos, tslots, self.cfg, frames, self.budget)
        q = r + GAMMA * option_values(self.net, b, sc) * (~done)
        return opts[int(np.argmax(q))]


class OurBot:
    def __init__(self, exe, weights, extra, name):
        self.name = name
        self.p = subprocess.Popen([exe, "serve", "--mode", "battle", "--weights", weights, "--any-cores", "--threads", "1"] + extra,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.mismatch = 0

    def decide(self, s):
        t = s.tetris
        grid = t.grid
        depth = len(grid[0])
        req = {"rows": ["".join("X" if grid[x][y] > 0 else "." for x in range(10)) for y in range(depth)],
               "excess": depth - 20, "piece": t.block.block_type(),
               "queue": [p.block_type() for p in t.buffer.now_list[:5]],
               "hold": t.held.block_type() if t.held is not None else None,
               "combo": t.combo, "b2b": t.b2b_count}
        self.p.stdin.write(json.dumps(req) + "\n")
        self.p.stdin.flush()
        rep = json.loads(self.p.stdout.readline())
        opts = enumerate_placements(t)
        if rep.get("dead"):
            return opts[0] if opts else None
        want = (tuple(sorted(tuple(c) for c in rep["cells"])), bool(rep["hold"]))
        for p in opts:
            if (tuple(sorted(tuple(c) for c in p.cells)), bool(p.use_hold)) == want:
                return p
        self.mismatch += 1
        return opts[0] if opts else None

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def play_game(args, g):
    seed = args.seed + g
    random.seed(seed)
    np.random.seed(seed)
    cap = float(args.speed.split(":")[1]) if args.speed.startswith("cap:") else 0.0
    budget = FPS / cap if cap else 0
    bots = [HisBot(args.model, budget), OurBot(args.exe, args.weights, args.bot_args.split(), args.name)]
    if g % 2 == 1:
        bots.reverse()  # alternate who acts first in turn mode
    sims = [new_player(i, seed) for i in range(2)]
    st = [{"bot": b.name, "pieces": 0, "attack": 0, "garbage_in": 0, "garbage_cleared": 0, "think": 0.0,
           "clock": 0.0, "tsd": 0, "tst": 0, "quads": 0, "ko": False} for b in bots]
    # (lock time, placement) of each player's piece in flight
    pending = [None, None]
    moves = []   # every lock in order, for bot/tools/battle_watch.py

    def decide(i):
        t0 = time.perf_counter()
        p = bots[i].decide(sims[i])
        think = time.perf_counter() - t0
        st[i]["think"] += think
        if p is None:
            return None
        if args.speed == "turn":
            dur = 1.0
        else:
            dur = max(think, p.frames / FPS, 1.0 / cap if cap else 0.0)
        return (st[i]["clock"] + dur, p)

    for i in range(2):
        pending[i] = decide(i)
    turn = 0
    while True:
        if pending[0] is None or pending[1] is None:
            for i in range(2):
                st[i]["ko"] = st[i]["ko"] or pending[i] is None
            break
        if args.speed == "turn":
            i = turn % 2
            turn += 1
        else:
            i = 0 if pending[0][0] <= pending[1][0] else 1
        when, p = pending[i]
        s = sims[i]
        t = s.tetris
        q_before = [list(x) for x in t._garbage_queue]
        g_before = garbage_rows(t.grid)
        moves.append({"p": i, "t": round(when, 4), "hold": bool(p.use_hold), "rot": p.rot, "x": p.px, "y": p.py,
                      "tspin": p.tspin, "cells": sorted(list(c) for c in p.cells)})
        out = s.apply(p)
        added = len(garbage_holes(q_before, out["scores"])[:20])
        st[i]["garbage_cleared"] += max(0, g_before + added - garbage_rows(t.grid))
        st[i]["pieces"] += 1
        st[i]["attack"] += out["scores"]
        st[i]["clock"] = when
        if out["cleared"] == 4:
            st[i]["quads"] += 1
        if out["tspin"] == 2 and out["cleared"] in (2, 3):
            st[i]["tsd" if out["cleared"] == 2 else "tst"] += 1
        if out["scores"] > 0:
            o = 1 - i
            sims[o].tetris.add_attacked(out["scores"], sims[o].hole_rng.randrange(10))
            st[o]["garbage_in"] += out["scores"]
        if out["ko"]:
            st[i]["ko"] = True
            break
        if st[0]["pieces"] >= args.pieces and st[1]["pieces"] >= args.pieces:
            break
        pending[i] = decide(i) if st[i]["pieces"] < args.pieces else (float("inf"), None)
        if pending[i][1] is None and st[i]["pieces"] < args.pieces:
            pending[i] = None
    for b in bots:
        if isinstance(b, OurBot):
            for s_ in st:
                if s_["bot"] == b.name:
                    s_["mismatch"] = b.mismatch
            b.close()
    if st[0]["ko"] != st[1]["ko"]:
        winner = st[1]["bot"] if st[0]["ko"] else st[0]["bot"]
    else:
        winner = max(st, key=lambda x: x["attack"])["bot"] if st[0]["attack"] != st[1]["attack"] else "draw"
    for s_ in st:
        secs = s_["clock"] if args.speed != "turn" else s_["pieces"]
        s_["app"] = s_["attack"] / max(1, s_["pieces"])
        s_["pps"] = s_["pieces"] / secs if args.speed != "turn" and secs > 0 else None
        s_["vs"] = 100.0 * (s_["attack"] + s_["garbage_cleared"]) / secs if args.speed != "turn" and secs > 0 else None
        s_["vs_per_piece"] = (s_["attack"] + s_["garbage_cleared"]) / max(1, s_["pieces"])
        s_["think_ms"] = 1000.0 * s_["think"] / max(1, s_["pieces"])
    return {"game": g, "seed": seed, "speed": args.speed, "winner": winner, "players": st, "moves": moves}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="best.pt")
    ap.add_argument("--exe", default=os.path.join("..", "..", "bot", "target", "release", "bot.exe"))
    ap.add_argument("--weights", required=True)
    ap.add_argument("--name", default="ours")
    ap.add_argument("--bot-args", default="", help="extra args for bot serve, e.g. '--depth 1'")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--pieces", type=int, default=500, help="per player")
    ap.add_argument("--speed", default="turn")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default="match.jsonl")
    args = ap.parse_args()

    with mp.Pool(args.jobs) as pool, open(args.out, "w") as f:
        results = []
        for r in pool.imap_unordered(_run, [(args, g) for g in range(args.games)]):
            f.write(json.dumps(r) + "\n")
            f.flush()
            results.append(r)
            print("game %2d: winner %-8s %s" % (r["game"], r["winner"], "  ".join(
                "%s app %.3f pcs %d%s" % (p["bot"], p["app"], p["pieces"], " KO" if p["ko"] else "") for p in r["players"])), flush=True)
    wins = {}
    for r in results:
        wins[r["winner"]] = wins.get(r["winner"], 0) + 1
    print("wins:", wins)
    for name in sorted({p["bot"] for r in results for p in r["players"]}):
        ps = [p for r in results for p in r["players"] if p["bot"] == name]
        agg = lambda k: np.mean([p[k] for p in ps if p[k] is not None]) if any(p[k] is not None for p in ps) else float("nan")
        print("%-10s APP %.3f  VS/piece %.3f  pieces %.0f  think %.1f ms  PPS %s  VS %s  KOs %d  TSD %d  TST %d  quads %d  mismatches %s" % (
            name, agg("app"), agg("vs_per_piece"), agg("pieces"), agg("think_ms"),
            "%.2f" % agg("pps") if args.speed != "turn" else "-", "%.1f" % agg("vs") if args.speed != "turn" else "-",
            sum(p["ko"] for p in ps), sum(p["tsd"] for p in ps), sum(p["tst"] for p in ps), sum(p["quads"] for p in ps),
            sum(p.get("mismatch", 0) for p in ps)))


def _run(a):
    return play_game(*a)


if __name__ == "__main__":
    main()
