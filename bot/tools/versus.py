"""Versus matches between bots in our game engine (`bot play-ext --versus`, one process per player):
both players get the same piece sequence and the same sequence of garbage hole columns, and place
one piece each per turn (fixed time per move: thinking time does not change the pace).

Garbage (TETR.IO-like, as bot/tools/battle_match.py): a placement's attack first cancels the
player's own pending garbage, then what is left of the two players' attacks in the same turn cancels
out, and the rest goes to the opponent as one chunk (one hole column). Pending garbage that was
already waiting before this turn lands after the receiver's next placement that clears nothing,
at most --cap rows per placement; garbage sent this turn can still be cancelled next turn.
A match ends when a player tops out (the other wins) or both have placed --pieces pieces (a draw).

--realtime (uncapped speed): no turns. Each player has its own clock, advanced by its measured think time
per piece (wall time of computing the move, at least --min-ms), so a faster player places more pieces in
the same game time. The next lock in game time is applied first:
- its attack cancels its own pending garbage, and the rest is queued for the opponent, arriving at that
  time plus --garbage-delay-ms;
- a placement that clears nothing lands the pending garbage that has arrived by then (at most --cap rows).
Moves are decided from the board at the start of the think, so garbage landing mid-think lands at the lock,
as in a real game. Only one player computes at a time, so measured times are not inflated. Cold Clear 2
gets its next position (play / new_piece) only when its own clock starts, so it cannot think for free while
the other player moves. The match ends at a top-out, or as a draw after --max-seconds of game time.

Players (--a / --b):
  cem:WEIGHTS:DEPTH:BEAM[:nochance][:commitK]  our CEM bot, its own search inside play-ext (--self); nochance = no
                               expectimax ply over the piece after the preview (--no-chance); commitK = play
                               the search plan's next K-1 moves without searching (--commit K; garbage landing
                               cancels the rest of the plan)
  cc2:THINK_MS                 Cold Clear 2 (external/cold-clear-2, unmodified, TBP), THINK_MS per move
  cc2:match                    Cold Clear 2 thinking as long as its opponent has taken per move so far (running mean)
  nn:MODEL:STYLE[:DEPTH:BEAM]  AtTenTris (STYLE normal | danger | auto), depth 1 by default, else its Qs beam search
Rules: --mode coldclear (Cold Clear 1's attack table + guideline combos) or tetrio (TETR.IO table and
combo multiplier); both SRS, T-spins only, with --no-180.

    external/tetrisbattle/.venv/Scripts/python bot/tools/versus.py --a cem:bot/weights/normal.txt:6:64 --b cc2:27 \
        --mode coldclear --games 64 --out vs.jsonl
"""
import argparse
import concurrent.futures as cf
import ctypes
import json
import multiprocessing as mp
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BOT = os.path.join(ROOT, "bot", "target", "release", "bot.exe" if os.name == "nt" else "bot")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "attentris"))
import cc2_play as C  # noqa: E402
import pcores  # noqa: E402


def readjson(p):
    line = p.stdout.readline()
    if not line:
        raise RuntimeError("bot play-ext exited")
    return json.loads(line)


class Side:
    """One player: a `bot play-ext --versus` game plus whatever picks its moves."""

    def __init__(self, spec, args, seed):
        self.spec = spec
        kind, *rest = spec.split(":")
        self.kind = kind
        cmd = [BOT, "play-ext", "--mode", args.mode, "--no-180", "--versus", "--pieces", str(args.pieces), "--seed", str(seed), "--any-cores"]
        self.think = []
        if kind == "cem":
            w, depth, beam = rest[0], rest[1], rest[2]
            cmd += ["--self", "--weights", os.path.join(ROOT, w) if not os.path.isabs(w) else w, "--depth", depth, "--beam", beam, "--threads", "1"]
            for opt in rest[3:]:
                if opt == "nochance":
                    cmd += ["--no-chance"]
                elif opt.startswith("commit"):
                    cmd += ["--commit", opt[len("commit"):]]
                else:
                    raise SystemExit("unknown cem option " + opt)
        elif kind == "cc2":
            self.think_ms = None if rest[0] == "match" else float(rest[0])
            self.cc = C.Cc2(0)
            self.expect, self.combo, self.resigned = None, 0, False
            self.illegal_restarts = 0
        elif kind == "nn":
            import play as PL
            model = rest[0] if os.path.isabs(rest[0]) else os.path.join(ROOT, rest[0])
            self.style = rest[1] if len(rest) > 1 else "normal"
            depth = int(rest[2]) if len(rest) > 2 else 1
            beam = int(rest[3]) if len(rest) > 3 else 8
            self.styles = PL.STYLES
            self.pl = PL.Player(model, depth, beam, PL.STYLES["normal" if self.style == "auto" else self.style], args.mode,
                                expand_args=("--no-180",))
            self.T, self.P = PL.T, PL.P
        else:
            raise SystemExit("unknown player " + spec)
        self.game = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.done = None
        self.opp = None        # the other Side (cc2:match reads its think times)
        self.t_sent = None     # cem: when the last garbage line went out (its search starts then)
        self.defer = args.realtime   # cc2: hand it the played move only when its next think starts
        self.pending_play = None

    def turn(self):
        """Place one piece: the lock line, or the game's done line (topped out)."""
        if self.kind == "cem":
            t0 = self.t_sent
            r = readjson(self.game)
            if t0 is not None and "lock" in r:
                self.think.append(time.perf_counter() - t0)
        else:
            st = readjson(self.game)
            if st.get("done"):
                r = st
            else:
                t0 = time.perf_counter()
                if self.pending_play is not None:
                    self.cc2_played(*self.pending_play)
                    self.pending_play = None
                mv = self.cc2_move(st) if self.kind == "cc2" else self.nn_move(st)
                self.think.append(time.perf_counter() - t0)
                self.send(mv)
                r = readjson(self.game)
                if self.kind == "cc2" and "lock" in r and "tbp" in mv:
                    if self.defer:
                        self.pending_play = (st, mv, r)
                    else:
                        self.cc2_played(st, mv, r)
        if r.get("done"):
            self.done = r
        return r

    # --- Cold Clear 2 (as cc2_play.play_game: restart on any surprise, else play + new_piece) ---
    def cc2_move(self, st):
        reserve, queue = C.cc_view(st)
        fresh = self.expect is None
        if not fresh:
            er, eres, eq = self.expect
            if er != st["rows"] or eres != reserve or queue[:len(eq)] != eq:
                fresh = True
        if fresh:
            combo = st["combo"] + 1 if st.get("combo", -1) >= 0 else 0
            self.cc.send({"type": "start", "board": C.tbp_board(st["rows"]), "queue": [st["piece"]] + st["queue"], "hold": st["hold"],
                          "combo": combo, "back_to_back": st["b2b"] >= 0, "randomizer": {"type": "seven_bag", "bag_state": st.get("bag") or list("IJLOSTZ")}})
        else:
            for p in queue[len(self.expect[2]):]:
                self.cc.send({"type": "new_piece", "piece": p})
        ms = self.think_ms
        if ms is None:
            ms = 1000 * sum(self.opp.think) / len(self.opp.think) if self.opp.think else 40.0
        time.sleep(ms / 1000.0)
        sug = C.suggest(self.cc)
        if not sug.get("moves"):
            self.expect = None
            self.resigned = True
            return {"resign": True}
        m = sug["moves"][0]
        loc = m["location"]
        cells = C.cells_of(loc)
        return {"hold": int(loc["type"] != st["piece"]), "cells": [c for xy in cells for c in xy], "tbp": m, "_cells": cells, "_q": queue, "_r": reserve}

    def cc2_played(self, st, mv, lock):
        m = mv["tbp"]
        self.cc.send({"type": "play", "move": m})
        rows, _ = C.place(st["rows"], mv["_cells"])
        queue, reserve = mv["_q"], mv["_r"]
        front = queue[0]
        self.expect = (rows, front if m["location"]["type"] != front else reserve, queue[1:])
        if sorted(map(tuple, mv["_cells"])) != sorted(zip(lock["cells"][0::2], lock["cells"][1::2])):
            self.illegal_restarts += 1      # our game played its own move: the next turn restarts Cold Clear 2

    def send(self, m):
        """To our game process (the driver's private keys stripped)."""
        m = {k: v for k, v in m.items() if k in ("hold", "cells", "resign", "garbage", "stop")}
        self.game.stdin.write(json.dumps(m) + "\n")
        self.game.stdin.flush()

    # --- AtTenTris ---
    def nn_move(self, st):
        if self.style == "auto":
            b = self.T.board_from_replay(self.P.rows_to_strings(st["rows"]))
            hot = int(self.T.heights(b).max()) >= 12 or self.T.holes_of(b) >= 6
            self.pl.teacher = self.styles["danger" if hot else "normal"]
        c = self.pl.decide(st)
        return {"hold": c["hold"], "cells": c["cells"]} if c else {"resign": True}

    def finish(self):
        """Stop a game that is still running (the opponent topped out) and collect its stats."""
        if self.done is None:
            self.send({"stop": True})
            while True:
                r = readjson(self.game)
                if r.get("done"):
                    self.done = r
                    break
        self.game.wait()
        if self.kind == "cc2":
            self.cc.close()
            self.done["resigned"] = self.resigned
        if self.kind == "nn":
            self.pl.close()
        d = dict(self.done)
        if self.think:     # measured by the referee (cem: garbage sent -> lock received)
            d["think_ms"] = 1000 * sum(self.think) / len(self.think)
        elif "search_ms" in d:
            d["think_ms"] = d["search_ms"]
        return d


def realtime_loop(sides, args, holes, sent, received):
    """--realtime: event-driven play on two clocks (ms of game time). Returns (events, end time)."""
    clk = [0.0, 0.0]
    pending = [[], []]          # [rows, hole, arrival ms]
    hole_i = [0, 0]
    events = []                 # --record: one entry per lock, of one player

    def compute(i):
        t0 = time.perf_counter()
        r = sides[i].turn()
        dt = max(1000 * (time.perf_counter() - t0), args.min_ms)
        return r, clk[i] + dt, dt

    nxt = [compute(0), compute(1)]
    end = 0.0
    while True:
        i = 0 if nxt[0][1] <= nxt[1][1] else 1
        o = 1 - i
        l, t, dt = nxt[i]
        if t > 1000 * args.max_seconds:
            end = 1000 * args.max_seconds
            break
        end = t
        if l.get("done"):
            if args.record:
                events.append({"side": i, "t": round(t, 2), "dead": True, "ms": round(dt, 2)})
            break
        a = l["attack"]
        while a > 0 and pending[i]:
            c = min(a, pending[i][0][0])
            pending[i][0][0] -= c
            a -= c
            if pending[i][0][0] == 0:
                pending[i].pop(0)
        if a > 0:
            pending[o].append([a, holes[hole_i[o] % len(holes)], t + args.garbage_delay_ms])
            hole_i[o] += 1
            sent[i] += a
        g = []
        if l["lines"] == 0:
            room = args.cap
            while room > 0 and pending[i] and pending[i][0][2] <= t:
                c = min(room, pending[i][0][0])
                g.append([c, pending[i][0][1]])
                pending[i][0][0] -= c
                room -= c
                received[i] += c
                if pending[i][0][0] == 0:
                    pending[i].pop(0)
        if args.record:
            ev = {k: l[k] for k in ("piece", "cells", "rows", "attack", "lines", "spin", "b2b", "combo", "hold", "queue") if k in l}
            ev.update({"side": i, "t": round(t, 2), "ms": round(dt, 2), "garbage": g, "pending": sum(x[0] for x in pending[i]),
                       "opp_pending": sum(x[0] for x in pending[o]), "sent": sent[i]})
            events.append(ev)
        sides[i].send({"garbage": g})
        sides[i].t_sent = time.perf_counter()
        clk[i] = t
        nxt[i] = compute(i)
    return events, end


def match(job):
    seed, args, mask = job
    if args.record and args.record_first and seed >= args.seed + args.record_first:
        args.record = ""          # this job's own copy of args (pickled per job)
    if mask:
        pcores.set_mask(mask)
    # 1 ms timer resolution, so Cold Clear 2's think time is what we ask for (Windows' default tick ~15.6 ms;
    # Linux timers are already fine-grained)
    if sys.platform == "win32":
        ctypes.WinDLL("winmm").timeBeginPeriod(1)
    rng = random.Random(seed * 7919 + 17)
    holes = [rng.randrange(10) for _ in range(4096)]
    sides = [Side(args.a, args, seed), Side(args.b, args, seed)]
    sides[0].opp, sides[1].opp = sides[1], sides[0]
    pending = [[], []]          # [rows, hole, turn it arrived]
    hole_i = [0, 0]
    sent = [0, 0]
    received = [0, 0]
    turn = 0
    frames = []                 # --record: one entry per turn, both players
    pool = cf.ThreadPoolExecutor(2)
    end_ms = None
    try:
        if args.realtime:
            frames, end_ms = realtime_loop(sides, args, holes, sent, received)
        while not args.realtime:
            nthink = [len(s.think) for s in sides]
            locks = list(pool.map(lambda s: s.turn(), sides))
            dead = [bool(l.get("done")) for l in locks]
            if args.record:
                fr = []
                for i in range(2):
                    l = locks[i]
                    ms = 1000 * sides[i].think[-1] if len(sides[i].think) > nthink[i] else 0.0
                    fr.append({"dead": True, "ms": round(ms, 1)} if dead[i] else
                              {k: l[k] for k in ("piece", "cells", "rows", "attack", "lines", "spin", "b2b", "combo", "hold", "queue") if k in l} | {"ms": round(ms, 1)})
                frames.append(fr)
            if any(dead):
                break
            # cancel own pending garbage first
            rem = []
            for i in range(2):
                a = locks[i]["attack"]
                while a > 0 and pending[i]:
                    c = min(a, pending[i][0][0])
                    pending[i][0][0] -= c
                    a -= c
                    if pending[i][0][0] == 0:
                        pending[i].pop(0)
                rem.append(a)
            m = min(rem)                 # simultaneous attacks cancel each other
            for i in range(2):
                r = rem[i] - m
                if r > 0:
                    o = 1 - i
                    pending[o].append([r, holes[hole_i[o] % len(holes)], turn])
                    hole_i[o] += 1
                    sent[i] += r
            # land garbage that was waiting before this turn, after a placement that cleared nothing
            for i in range(2):
                g = []
                if locks[i]["lines"] == 0:
                    room = args.cap
                    while room > 0 and pending[i] and pending[i][0][2] < turn:
                        c = min(room, pending[i][0][0])
                        g.append([c, pending[i][0][1]])
                        pending[i][0][0] -= c
                        room -= c
                        received[i] += c
                        if pending[i][0][0] == 0:
                            pending[i].pop(0)
                sides[i].send({"garbage": g})
                sides[i].t_sent = time.perf_counter()
                if args.record:
                    frames[-1][i]["garbage"] = g
                    frames[-1][i]["pending"] = sum(x[0] for x in pending[i])
                    frames[-1][i]["sent"] = sent[i]
            turn += 1
    finally:
        pool.shutdown()
    res = [s.finish() for s in sides]
    died = [bool(r.get("died")) for r in res]
    winner = None if died[0] == died[1] else (1 if died[0] else 0)
    out = {"seed": seed, "a_spec": args.a, "b_spec": args.b, "mode": args.mode, "turns": turn, "winner": ["a", "b"][winner] if winner is not None else "draw"}
    if args.realtime:
        out["realtime"] = True
        out["seconds"] = round(end_ms / 1000.0, 3)
    for i, k in enumerate("ab"):
        r = res[i]
        out[k] = {"died": died[i], "pieces": r.get("pieces"), "attack": r.get("attack"), "sent": sent[i], "received": received[i],
                  "garbage_cleared": r.get("garbage_cleared"), "tsd": r.get("tsd"), "quads": r.get("quads"),
                  "think_ms": round(r.get("think_ms", 0.0), 2), "illegal": r.get("illegal", 0), "resigned": r.get("resigned", False)}
        if args.realtime:
            out[k]["pps"] = round((r.get("pieces") or 0) / max(end_ms / 1000.0, 1e-9), 2)
    if args.record:
        # replay for bot/tools/versus_watch.py: both boards turn by turn, with the measured think times;
        # budget_ms = the pace a turn takes at speed x1 when the bots are faster (2 PPS: 500)
        os.makedirs(args.record, exist_ok=True)
        with open(os.path.join(args.record, "%s_s%d.json" % (args.tag or "match", seed)), "w") as f:
            json.dump({"seed": seed, "a": args.a, "b": args.b, "mode": args.mode, "budget_ms": args.budget_ms, "winner": out["winner"],
                       "turns": turn, "realtime": bool(args.realtime), "frames": frames}, f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--mode", default="coldclear", choices=("coldclear", "tetrio", "standard", "normal"))
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--seed", type=int, default=50001)
    ap.add_argument("--pieces", type=int, default=1000, help="per player; both reach it = a draw")
    ap.add_argument("--cap", type=int, default=8, help="garbage rows landing per placement at most")
    ap.add_argument("--jobs", type=int, default=4, help="matches at a time (two physical P-cores each)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--record", default="", help="DIR: also write each match as a replay (bot/tools/versus_watch.py)")
    ap.add_argument("--record-first", type=int, default=0, help="--record only the first N seeds (0 = all)")
    ap.add_argument("--tag", default="", help="replay file name prefix")
    ap.add_argument("--budget-ms", type=float, default=0.0, help="replay pace per turn at speed x1 when both bots are faster (2 PPS: 500)")
    ap.add_argument("--realtime", action="store_true", help="uncapped speed: each player on its own clock (see the docstring)")
    ap.add_argument("--max-seconds", type=float, default=120.0, help="--realtime: a draw after this much game time")
    ap.add_argument("--min-ms", type=float, default=0.0, help="--realtime: each piece takes at least this long (0 = uncapped)")
    ap.add_argument("--garbage-delay-ms", type=float, default=0.0, help="--realtime: garbage can land this long after it was sent")
    args = ap.parse_args()
    if args.realtime and args.pieces == 1000:
        args.pieces = 100000     # the game-time limit ends realtime matches, not a piece count
    pcores.pin(verbose=False)
    pm = pcores.performance_mask()
    masks = [(pm >> (4 * k) & 15) << (4 * k) for k in range(16) if pm >> (4 * k) & 15 == 15][:args.jobs] or [0]
    done = {}
    if os.path.exists(args.out):
        for l in open(args.out):
            r = json.loads(l)
            done[r["seed"]] = r
    todo = [args.seed + i for i in range(args.games) if args.seed + i not in done]
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with open(args.out, "a") as f, ctx.Pool(len(masks)) as pool:
        jobs = [(s, args, masks[i % len(masks)]) for i, s in enumerate(todo)]
        for r in pool.imap_unordered(match, jobs):
            done[r["seed"]] = r
            f.write(json.dumps(r) + "\n")
            f.flush()
            when = ("%.2f s game time (a %.1f PPS, b %.1f PPS)" % (r["seconds"], r["a"]["pps"], r["b"]["pps"]) if r.get("realtime")
                    else "%d turns" % r["turns"])
            print("seed %d: %s in %s; a sent %d recv %d (%.1f ms), b sent %d recv %d (%.1f ms)  (%d/%d, %.0fs)" % (
                r["seed"], r["winner"], when, r["a"]["sent"], r["a"]["received"], r["a"]["think_ms"],
                r["b"]["sent"], r["b"]["received"], r["b"]["think_ms"], len(done), args.games, time.time() - t0), flush=True)
    rs = [done[args.seed + i] for i in range(args.games) if args.seed + i in done]
    print(summary(rs))


def summary(rs):
    n = len(rs)
    wa = sum(r["winner"] == "a" for r in rs)
    wb = sum(r["winner"] == "b" for r in rs)
    dr = n - wa - wb
    line = []
    for k in "ab":
        pcs = sum(r[k]["pieces"] or 0 for r in rs)
        att = sum(r[k]["attack"] or 0 for r in rs)
        clr = sum(r[k]["garbage_cleared"] or 0 for r in rs)
        secs = sum(r.get("seconds", 0.0) for r in rs)
        line.append("APP %.3f, VS %.3f, sent %.1f/match, %.1f ms/move, illegal %d%s" % (
            att / max(1, pcs), (att + clr) / max(1, pcs), sum(r[k]["sent"] for r in rs) / n, sum(r[k]["think_ms"] for r in rs) / n,
            sum(r[k]["illegal"] or 0 for r in rs), ", %.1f PPS" % (pcs / secs) if secs else ""))
    rt = rs[0].get("realtime")
    return "%s vs %s (%s%s, %d matches): a wins %d, b wins %d, draws %d (a score %.3f)%s\n  a: %s\n  b: %s" % (
        rs[0]["a_spec"], rs[0]["b_spec"], rs[0]["mode"], ", realtime" if rt else "", n, wa, wb, dr, (wa + 0.5 * dr) / n,
        ", mean %.2f s game time" % (sum(r.get("seconds", 0.0) for r in rs) / n) if rt else "", line[0], line[1])


if __name__ == "__main__":
    main()
