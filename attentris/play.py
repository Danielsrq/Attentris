"""AtTenTris plays real games: the Rust game loop (`bot play-ext`: bag, garbage, scoring) asks
for every move; legal moves and their result boards come from `bot expand`.

  --depth 1   no search: the model's top move
  --depth k   policy beam: keep the --beam most likely lines (sum of log-probabilities,
              the model scoring each position's moves) for k pieces, play the first move of
              the best line. The model has no value head, so likelihood is the only score.

    <python with torch> attentris/play.py --games 16 --depth 1 [--garbage 10,3-4,cheese0.2,0.2] [--style normal]

Each game runs in its own process (one torch thread each), pinned to the P-cores.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pack as P
import pcores
import tokenizer as T

HERE = os.path.dirname(os.path.abspath(__file__))
BOT = os.path.join(HERE, "..", "bot", "target", "release", "bot.exe")


def readjson(proc):
    while True:
        line = proc.stdout.readline()
        if not line:
            raise EOFError("bot exited")
        if line.startswith("{"):
            return json.loads(line)


def arrays(states, teacher):
    """Model inputs for a list of (state, expand-result) pairs; returns the batch dict."""
    import torch
    n = len(states)
    NT, NC, NE = P.NT, P.NC, P.NE
    a = {"cells": np.zeros((n, 200), np.int64), "tok_kind": np.full((n, NT), -1, np.int64),
         "tok_mask": np.zeros((n, NT, 200), bool), "tok_box": np.full((n, NT, 4), -1, np.int64),
         "tok_extra": np.zeros((n, NT, NE), np.float32), "tok_piece": np.full((n, NT), -1, np.int64),
         "cand_cells": np.full((n, NC, 4), -1, np.int64), "cand_meta": np.zeros((n, NC, 5), np.int64),
         "teacher": np.full(n, teacher, np.int64)}
    for i, (st, ex) in enumerate(states):
        b = T.board_from_replay(P.rows_to_strings(st["rows"]))
        toks = T.tokenize(b, st["piece"], st["queue"], st["hold"], ex["structures"])
        a["cells"][i] = b.reshape(-1)
        for j, t in enumerate(toks[:NT]):
            a["tok_kind"][i, j] = P.KIND_ID[t.kind]
            a["tok_mask"][i, j] = t.mask.reshape(-1)
            if t.box is not None:
                a["tok_box"][i, j] = t.box
            a["tok_extra"][i, j] = P.extras(t)
            if t.kind in T.PIECE_KINDS:
                p = t.extras["piece"]
                a["tok_piece"][i, j] = P.PIECE_ID[p] if p else 7
        for j, c in enumerate(ex["cands"][:NC]):
            cells = c["cells"]
            a["cand_cells"][i, j] = [(39 - cells[2 * q + 1]) * 10 + cells[2 * q] if 0 <= 39 - cells[2 * q + 1] < 20 else -2
                                     for q in range(4)]
            a["cand_meta"][i, j] = (P.PIECE_ID[c["piece"]], c["hold"], c["spin"], c["lines"], max(0, min(c["attack"], 127)))
    return {k: torch.from_numpy(v) for k, v in a.items()}


class Player:
    def __init__(self, model_path, depth, beam, teacher, mode):
        import torch
        import model as M
        torch.set_num_threads(1)
        ck = torch.load(model_path, map_location="cpu", weights_only=False)
        self.net = M.AtTenTris(d=ck["args"]["d"], layers=ck["args"]["layers"])
        self.net.load_state_dict(ck["model"])
        self.net.eval()
        self.torch = torch
        self.depth, self.beam, self.teacher = depth, beam, teacher
        self.ex = subprocess.Popen([BOT, "expand", "--mode", mode], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def expand(self, st):
        self.ex.stdin.write(json.dumps(st) + "\n")
        self.ex.stdin.flush()
        return readjson(self.ex)

    def logprobs(self, pairs):
        with self.torch.inference_mode():
            logit = self.net(arrays(pairs, self.teacher))
        return logit.log_softmax(-1).numpy()

    def decide(self, st):
        root = self.expand(st)
        if not root["cands"]:
            return None
        if self.depth <= 1:
            lp = self.logprobs([(st, root)])[0]
            return root["cands"][int(np.argmax(lp[:len(root["cands"])]))]
        # policy beam: (log-prob of the line, first move, state, its expansion)
        frontier = [(0.0, None, st, root)]
        best_line = None
        for ply in range(self.depth):
            lps = self.logprobs([(s, e) for _, _, s, e in frontier])
            kids = []
            for (score, first, s, e), lp in zip(frontier, lps):
                for k, c in enumerate(e["cands"][:P.NC]):
                    kids.append((score + float(lp[k]), first if first is not None else c, s, c))
            kids.sort(key=lambda x: -x[0])
            kids = kids[:self.beam]
            best_line = kids[0]
            if ply == self.depth - 1:
                break
            frontier = []
            for score, first, s, c in kids:
                q = s["queue"]
                used = c["used"]
                if used > len(q):          # no known piece left for this line
                    continue
                child = {"rows": c["rows"], "piece": q[used - 1], "queue": q[used:], "hold": c["hold_after"], "b2b": c["b2b"]}
                ce = self.expand(child)
                if ce["cands"]:
                    frontier.append((score, first, child, ce))
            if not frontier:
                break
        return best_line[1]

    def close(self):
        self.ex.stdin.close()
        self.ex.wait()


def play_game(args, seed):
    pcores.pin(verbose=False)
    pl = Player(args.model, args.depth, args.beam, {"normal": 0, "gate4": 1}[args.style], args.mode)
    cmd = [BOT, "play-ext", "--mode", args.mode, "--pieces", str(args.pieces), "--seed", str(seed)]
    if args.garbage:
        cmd += ["--garbage", args.garbage]
    game = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    times = []
    states = []
    while True:
        st = readjson(game)
        if st.get("done"):
            break
        if args.dump:
            states.append({"seed": seed, "n": st["n"], "rows": st["rows"], "b2b": st["b2b"], "piece": st["piece"],
                           "queue": st["queue"], "hold": st["hold"], "style": args.style})
        t0 = time.perf_counter()
        c = pl.decide(st)
        times.append(time.perf_counter() - t0)
        move = {"hold": c["hold"], "cells": c["cells"]} if c else {"hold": 0, "cells": []}
        game.stdin.write(json.dumps(move) + "\n")
        game.stdin.flush()
    game.wait()
    pl.close()
    st["ms_per_move"] = 1000 * float(np.mean(times)) if times else 0.0
    st["ms_p90"] = 1000 * float(np.percentile(times, 90)) if times else 0.0
    st["states"] = states
    return st


def _run(a):
    return play_game(*a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "data", "model.pt"))
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--pieces", type=int, default=500)
    ap.add_argument("--seed", type=int, default=5001)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--style", default="normal", choices=("normal", "gate4"))
    ap.add_argument("--mode", default="normal")
    ap.add_argument("--garbage", default="")
    ap.add_argument("--jobs", type=int, default=0, help="parallel games (0 = P-core logical processors)")
    ap.add_argument("--out", default="")
    ap.add_argument("--dump", default="", help="write every state reached (seed, n, rows, b2b) to this JSONL")
    args = ap.parse_args()
    n_p = pcores.pin() or 8
    jobs = args.jobs or n_p
    t0 = time.time()
    with mp.get_context("spawn").Pool(min(jobs, args.games)) as pool:
        res = pool.map(_run, [(args, args.seed + g) for g in range(args.games)])
    pieces = sum(r["pieces"] for r in res)
    attack = sum(r["attack"] for r in res)
    full = attack / (args.games * args.pieces)
    line = ("AtTenTris depth %d%s style %s %s: APP %.3f (APPfull %.3f), died %d/%d, pieces %.0f avg, TSD %d, TST %d, quads %d, "
            "max height %d, illegal %d, %.1f ms/move (p90 %.1f), %.0fs") % (
        args.depth, "" if args.depth <= 1 else " beam %d" % args.beam, args.style, ("garbage " + args.garbage) if args.garbage else "no garbage",
        attack / max(1, pieces), full, sum(r["died"] for r in res), args.games, pieces / args.games,
        sum(r["tsd"] for r in res), sum(r["tst"] for r in res), sum(r["quads"] for r in res),
        max(r["max_height"] for r in res), sum(r["illegal"] for r in res),
        float(np.mean([r["ms_per_move"] for r in res])), float(np.mean([r["ms_p90"] for r in res])), time.time() - t0)
    print(line, flush=True)
    if args.dump:
        with open(args.dump, "w") as f:
            for r in res:
                for s_ in r["states"]:
                    f.write(json.dumps(s_) + "\n")
    for r in res:
        r.pop("states", None)
    if args.out:
        with open(args.out, "a") as f:
            f.write(line + "\n")
            for r in res:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
