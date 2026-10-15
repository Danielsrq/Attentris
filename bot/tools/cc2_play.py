"""Cold Clear 2 (MinusKelvin, unmodified: external/cold-clear-2, cargo build --release) plays our
game: the Rust game loop (`bot play-ext`: bag, garbage, scoring by our rules) asks Cold Clear 2
for each move over TBP (the Tetris Bot Protocol it speaks natively) and plays it.

Rules for a fair test: `--mode standard --no-180` (SRS, T-spins only, no 180 rotation: Cold
Clear 2's own rule set; attack is scored by our table, which has no combo attack). Cold Clear 2
thinks for --think-ms per move: the clock starts when it has the new piece and ends at `suggest`
(it also keeps thinking while our process works; that is in its favour). One worker thread (its
default), each game pinned to one physical P-core, --jobs games at a time.

TBP state: started once per game and then advanced with `play` + `new_piece`; restarted (`start`
with board, queue, hold, combo, back-to-back, 7-bag state) whenever the board differs from the one
its move should have produced (garbage arrived, or our game did not accept its move). A move our
game does not accept is counted as illegal (bot play-ext plays its own depth-1 move instead). When
Cold Clear 2 sees no legal move the game ends as a death (play-ext {"resign": true}).

    attentris/.venv/Scripts/python bot/tools/cc2_play.py --games 64 --think-ms 100 --out cc2_100.jsonl
    ... --garbage 10,3-4,cheese0.2,0.2 --seed 33001
"""
import argparse
import concurrent.futures as cf
import ctypes
import json
import math
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
BOT = os.path.join(ROOT, "bot", "target", "release", "bot.exe" if os.name == "nt" else "bot")
CC2 = os.path.join(ROOT, "external", "cold-clear-2", "target", "release", "cold-clear-2.exe" if os.name == "nt" else "cold-clear-2")
sys.path.insert(0, os.path.join(ROOT, "attentris"))
import pcores  # noqa: E402

H = 40
# Cold Clear 2's piece shapes (data.rs), North orientation, (x, y) with y up
SHAPES = {"I": [(-1, 0), (0, 0), (1, 0), (2, 0)], "O": [(0, 0), (1, 0), (0, 1), (1, 1)],
          "T": [(-1, 0), (0, 0), (1, 0), (0, 1)], "L": [(-1, 0), (0, 0), (1, 0), (1, 1)],
          "J": [(-1, 0), (0, 0), (1, 0), (-1, 1)], "S": [(-1, 0), (0, 0), (0, 1), (1, 1)],
          "Z": [(-1, 1), (0, 1), (0, 0), (1, 0)]}
ROT = {"north": lambda x, y: (x, y), "east": lambda x, y: (y, -x),
       "south": lambda x, y: (-x, -y), "west": lambda x, y: (-y, x)}


def cells_of(loc):
    """Cold Clear 2 location -> our cells [(x, y)] (our y: 0 = top row of 40)."""
    r = ROT[loc["orientation"]]
    out = []
    for dx, dy in SHAPES[loc["type"]]:
        rx, ry = r(dx, dy)
        out.append((loc["x"] + rx, H - 1 - (loc["y"] + ry)))
    return out


def place(rows, cells):
    """Our rows (bitmasks, top to bottom) after placing cells and clearing full rows; lines."""
    rows = list(rows)
    for x, y in cells:
        rows[y] |= 1 << x
    keep = [r for r in rows if r != (1 << 10) - 1]
    lines = H - len(keep)
    return [0] * lines + keep, lines


def tbp_board(rows):
    """TBP board: 40 rows bottom to top, 10 cells each (None or a letter)."""
    return [[("G" if rows[H - 1 - y] >> x & 1 else None) for x in range(10)] for y in range(H)]


def cc_view(st):
    """The state as Cold Clear 2 models it: (reserve, queue). Its reserve is the hold piece, or
    the current piece when the hold is empty (then the previews are its queue)."""
    q = [st["piece"]] + st["queue"]
    if st["hold"] is None:
        return q[0], q[1:]
    return st["hold"], q


def suggest(cc, patience_ms=200):
    """Cold Clear 2's suggestion. Right after a `start` with a short think time its tree may not
    be expanded yet and it suggests nothing; ask again every millisecond (the wait counts as its
    thinking time) and only report no moves (topped out) after `patience_ms`."""
    t0 = time.perf_counter()
    while True:
        cc.send({"type": "suggest"})
        sug = cc.recv()
        if sug.get("moves") or time.perf_counter() - t0 > patience_ms / 1000.0:
            return sug
        time.sleep(0.001)


def pin_to(proc, mask):
    pcores.set_mask(mask, proc)


class Cc2:
    def __init__(self, mask):
        self.p = subprocess.Popen([CC2], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        if mask:
            pin_to(self.p, mask)
        info = self.recv()
        assert info["type"] == "info", info
        self.send({"type": "rules"})
        assert self.recv()["type"] == "ready"

    def send(self, m):
        self.p.stdin.write(json.dumps(m) + "\n")
        self.p.stdin.flush()

    def recv(self):
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("cold-clear-2 exited")
        return json.loads(line)

    def close(self):
        try:
            self.send({"type": "quit"})
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def play_game(seed, args, mask):
    cmd = [BOT, "play-ext", "--mode", "standard", "--no-180", "--combo", args.combo, "--pieces", str(args.pieces), "--seed", str(seed)]
    if args.garbage:
        cmd += ["--garbage", args.garbage]
    bot = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    cc = Cc2(mask)
    expect = None          # (rows, reserve, queue) Cold Clear 2 holds after its last move
    combo = 0
    restarts = garbage_restarts = no_move = 0
    nodes = []
    think = []
    log = open(os.path.join(args.log, "s%d.jsonl" % seed), "w") if args.log else None
    try:
        while True:
            line = bot.stdout.readline()
            if not line:
                raise RuntimeError("bot play-ext exited")
            st = json.loads(line)
            if st.get("done"):
                done = st
                break
            reserve, queue = cc_view(st)
            fresh = expect is None
            if not fresh:
                er, eres, eq = expect
                if er != st["rows"] or eres != reserve or queue[:len(eq)] != eq:
                    fresh = True
                    restarts += 1
                    if args.every and st["n"] % args.every == 0:
                        garbage_restarts += 1
            if fresh:
                if "combo" in st and st["combo"] >= 0:
                    combo = st["combo"] + 1   # TBP: consecutive clears so far; ours: 0 after the first
                bag = st.get("bag") or list("IJLOSTZ")
                cc.send({"type": "start", "board": tbp_board(st["rows"]), "queue": [st["piece"]] + st["queue"],
                         "hold": st["hold"], "combo": combo, "back_to_back": st["b2b"] >= 0,
                         "randomizer": {"type": "seven_bag", "bag_state": bag}})
            else:
                for p in queue[len(expect[2]):]:
                    cc.send({"type": "new_piece", "piece": p})
            t0 = time.perf_counter()
            time.sleep(args.think_ms / 1000.0)
            sug = suggest(cc)
            think.append(time.perf_counter() - t0)
            nodes.append(sug.get("move_info", {}).get("nodes", 0))
            if not sug.get("moves"):
                # Cold Clear 2 sees no legal move: it has topped out; the game ends as a death
                no_move += 1
                bot.stdin.write('{"resign": true}\n')
                bot.stdin.flush()
                continue
            mv = sug["moves"][0]
            loc = mv["location"]
            cells = cells_of(loc)
            use_hold = int(loc["type"] != st["piece"])
            bot.stdin.write(json.dumps({"hold": use_hold, "cells": [c for xy in cells for c in xy]}) + "\n")
            bot.stdin.flush()
            if log:
                log.write(json.dumps({"state": st, "move": mv, "hold": use_hold, "cells": cells, "fresh": fresh}) + "\n")
            cc.send({"type": "play", "move": mv})
            rows, lines = place(st["rows"], cells)
            combo = combo + 1 if lines else 0
            # Cold Clear 2's queue after advance: front popped; a move of the other piece makes the
            # front the new reserve
            front = queue[0]
            expect = (rows, front if loc["type"] != front else reserve, queue[1:])
    finally:
        if log:
            log.close()
        cc.close()
        if bot.poll() is None:
            bot.kill()
    done.update({"think_ms": args.think_ms, "restarts": restarts, "garbage_restarts": garbage_restarts, "cc2_no_move": no_move,
                 "cc2_nodes_per_move": sum(nodes) / max(1, len(nodes)),
                 "cc2_think_ms": 1000 * sum(think) / max(1, len(think))})
    return done


def summary(rs, label):
    n = len(rs)
    app = [r.get("app", 0.0) for r in rs if "app" in r]
    m = sum(app) / max(1, len(app))
    sd = math.sqrt(sum((a - m) ** 2 for a in app) / max(1, len(app) - 1)) if len(app) > 1 else 0.0
    died = sum(1 for r in rs if r.get("died"))
    pcs = sum(r.get("pieces", 0) for r in rs)
    per100 = lambda k: 100.0 * sum(r.get(k, 0) for r in rs) / max(1, pcs)
    return ("%s: APP %.3f +- %.3f (95%% CI, %d games), died %d/%d, TSD %.2f TST %.2f quads %.2f per 100 pieces, "
            "illegal %d, restarts %.1f/game (garbage %.1f), CC2 %.0f nodes/move, think %.1f ms/move") % (
        label, m, 1.96 * sd / math.sqrt(max(1, len(app))), len(app), died, n, per100("tsd"), per100("tst"), per100("quads"),
        sum(r.get("illegal", 0) for r in rs), sum(r["restarts"] for r in rs) / n, sum(r["garbage_restarts"] for r in rs) / n,
        sum(r["cc2_nodes_per_move"] for r in rs) / n, sum(r["cc2_think_ms"] for r in rs) / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pieces", type=int, default=500)
    ap.add_argument("--garbage", default="", help="as bot play --garbage, e.g. 10,3-4,cheese0.2,0.2")
    ap.add_argument("--think-ms", type=float, default=100)
    ap.add_argument("--combo", default="none", help="combo attack: none | guideline | tetrio (bot --combo)")
    ap.add_argument("--jobs", type=int, default=8, help="games at a time, one physical P-core each")
    ap.add_argument("--out", required=True, help="JSON lines, one per game (resumable)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--log", default=None, help="DIR: every turn's state and Cold Clear 2's move, s<seed>.jsonl (illegal-move checks)")
    args = ap.parse_args()
    if args.log:
        os.makedirs(args.log, exist_ok=True)
    args.every = int(args.garbage.split(",")[0]) if args.garbage else 0
    pcores.pin(verbose=False)
    pmask = pcores.performance_mask()
    cores = [(pmask >> (2 * k) & 3) << (2 * k) for k in range(32) if pmask >> (2 * k) & 3 == 3]
    done = {}
    if os.path.exists(args.out):
        for l in open(args.out):
            r = json.loads(l)
            done[r["seed"]] = r
    todo = [args.seed + i for i in range(args.games) if args.seed + i not in done]
    lock = threading.Lock()
    free = list(cores[:args.jobs]) or [0]
    t0 = time.time()

    def job(seed):
        with lock:
            mask = free.pop()
        try:
            return play_game(seed, args, mask)
        finally:
            with lock:
                free.append(mask)

    with open(args.out, "a") as f, cf.ThreadPoolExecutor(min(args.jobs, len(free))) as ex:
        for r in ex.map(job, todo):
            done[r["seed"]] = r
            f.write(json.dumps(r) + "\n")
            f.flush()
            print("seed %d: %s pieces, APP %.3f%s, %d illegal, %d restarts  (%d/%d, %.0fs)" % (
                r["seed"], r.get("pieces", "?"), r.get("app", 0.0), ", died" if r.get("died") else "", r.get("illegal", 0),
                r["restarts"], len(done), args.games, time.time() - t0), flush=True)
    rs = [done[args.seed + i] for i in range(args.games)]
    print(summary(rs, args.label or "Cold Clear 2 %gms" % args.think_ms))


if __name__ == "__main__":
    main()
