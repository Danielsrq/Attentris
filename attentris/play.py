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
import model as M
import pack as P
import pcores
import tokenizer as T

HERE = os.path.dirname(os.path.abspath(__file__))
BOT = os.path.join(HERE, "..", "bot", "target", "release", "bot.exe")
STYLES = {t: i for i, t in enumerate(P.TEACHERS)}     # the style input: normal 0, gate4 1, danger 2


def readjson(proc):
    while True:
        line = proc.stdout.readline()
        if not line:
            raise EOFError("bot exited")
        if line.startswith("{"):
            return json.loads(line)


def arrays(states, teacher, version=T.TOK_VERSION, drop=()):
    """Model inputs for a list of (state, expand-result) pairs; returns the batch dict. drop: token-kind
    prefixes to blank after tokenizing (an ablation checkpoint's train.py --drop-kinds: same as training)."""
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
        toks = T.tokenize(b, st["piece"], st["queue"], st["hold"], ex["structures"], version=version)
        a["cells"][i] = b.reshape(-1)
        for j, t in enumerate(toks[:NT]):
            if drop and any(t.kind.startswith(p) for p in drop):
                continue                                         # blanked, as in training (the slot stays padding)
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
    def __init__(self, model_path, depth, beam, teacher, mode, policy="logit", propose=0, temp=0.0, seed=0, merge="", merge_override="",
                 expand_args=()):
        import torch
        import model as M
        torch.set_num_threads(1)
        ck = torch.load(model_path, map_location="cpu", weights_only=False)
        tome = ck["args"].get("arch") == "tome"
        if tome:      # AtTenTris-ToMe: reads only the cells, the piece tokens and the candidates
            import model_tome as MT
            self.net = MT.AtTenTrisToMe(d=ck["args"]["d"], layers=ck["args"]["layers"],
                                        merge=[float(f) for f in (merge or ck["args"]["merge"]).split(",")], sky=ck["args"].get("tome_sky", False),
                                        pretok=ck["args"].get("tome_pretok", False), flat_bias=ck["args"].get("tome_flat", False),
                                        streams=ck["args"].get("tome_streams", 1), n_teachers=ck["model"]["style.weight"].shape[0],
                                        budget=MT.budget_of(ck["args"].get("tome_budget", "")), split=ck["args"].get("tome_split", "token"), pure=ck["args"].get("tome_pure", False))
            self.net.frac_override = MT.override_of(merge_override)
            self.n_styles = ck["model"]["style.weight"].shape[0]
        else:
            self.net = M.AtTenTris(d=ck["args"]["d"], layers=ck["args"]["layers"], n_kinds=ck["model"]["kind.weight"].shape[0],
                          cross_layers=ck["args"].get("cross_layers", 1), cross_loops=ck["args"].get("cross_loops", 1),
                          n_teachers=ck["model"]["style.weight"].shape[0])
            self.n_styles = ck["model"]["style.weight"].shape[0]
        missing = self.net.load_state_dict(ck["model"], strict=False).missing_keys
        if policy != "logit" and any(k.startswith(policy + ".") for k in missing):
            raise SystemExit("%s has no %s head (a v1 checkpoint?)" % (model_path, policy))
        self.net.eval()
        self.tok_version = 4 if tome else ck.get("tok_version", 5 if ck["model"]["kind.weight"].shape[0] > 31 else 4)
        self.no_setups = "setup_" in ck["args"].get("drop_kinds", "")
        self.drop = tuple(p for p in ck["args"].get("drop_kinds", "").split(",") if p)
        self.policy = policy
        self.propose = propose
        self.temp, self.rng, self.last_entropy = temp, np.random.default_rng(seed), None
        self.qscale = M.Q_SCALE
        self.torch = torch
        self.depth, self.beam, self.teacher = depth, beam, teacher
        # expand_args: more rule flags for `bot expand` (bot/tools/versus.py: --no-180)
        self.ex = subprocess.Popen([BOT, "expand", "--mode", mode, *expand_args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def expand(self, st):
        q = dict(st, slots=True) if self.tok_version >= 5 else st
        if self.no_setups:       # a setup-ablation checkpoint: the bot skips the setup lookahead
            q = dict(q, setups=False)
        self.ex.stdin.write(json.dumps(q) + "\n")
        self.ex.stdin.flush()
        return readjson(self.ex)

    def scores(self, pairs):
        """Per position: the policy's log-probabilities, and Q in lines of attack from the
        chosen value head (None when moves are chosen by the policy)."""
        with self.torch.inference_mode():
            a = arrays(pairs, self.teacher, self.tok_version, self.drop)
            if getattr(self.net, "pretok", False):      # ToMe v3: the rule pre-tokens of each position
                import pretok as PT
                a["pretok"] = self.torch.from_numpy(np.stack([PT.groups(a["cells"][i].numpy().astype(bool)) for i in range(len(pairs))]).astype(np.int64))
            out = self.net(a, heads=True)
        logp = out["logit"].log_softmax(-1).numpy()
        q = None if self.policy == "logit" else (out[self.policy] * self.qscale).numpy()
        return logp, q

    def allowed(self, logp, n):
        """Candidate indices Q may choose from: all, or the policy's top `propose` (so Q is
        only asked about moves like the ones it was trained on)."""
        order = np.argsort(-logp[:n])
        return order[:self.propose] if self.propose else order

    def shortlist(self, st, k):
        """The policy's top-k first moves (intuition), for the Rust search to choose among."""
        root = self.expand(st)
        if not root["cands"]:
            return []
        logp, _ = self.scores([(st, root)])
        n = len(root["cands"])
        return [root["cands"][int(i)] for i in np.argsort(-logp[0][:n])[:k]]

    def decide(self, st):
        root = self.expand(st)
        if not root["cands"]:
            return None
        n = len(root["cands"])
        if self.depth <= 1:
            logp, q = self.scores([(st, root)])
            lp = logp[0][:n]
            pr = np.exp(lp - lp.max())
            pr /= pr.sum()
            self.last_entropy = float(-(pr * np.log(np.clip(pr, 1e-12, None))).sum())   # for DAgger selection
            if q is None:
                if self.temp > 0:                     # sample (DAgger collection: more varied student games)
                    t = np.exp((lp - lp.max()) / self.temp)
                    return root["cands"][int(self.rng.choice(len(t), p=t / t.sum()))]   # the scored moves (<= NC)
                return root["cands"][int(np.argmax(lp))]
            ids = self.allowed(logp[0], n)
            return root["cands"][int(ids[np.argmax(q[0][ids])])]
        # beam: (score so far, first move, state, its expansion). Policy: a line's score is
        # the sum of its log-probabilities. Q head: attack sent earlier in the line + Q of
        # the last move (attack + what follows), as the teacher's own beam scores lines.
        frontier = [(0.0, None, st, root)]
        best_line = None
        for ply in range(self.depth):
            logps, qs = self.scores([(s, e) for _, _, s, e in frontier])
            kids = []
            for i, (score, first, s, e) in enumerate(frontier):
                cands = e["cands"][:P.NC]
                for k in self.allowed(logps[i], len(cands)):
                    c = cands[k]
                    if qs is None:
                        line = carry = score + float(logps[i][k])
                    else:
                        line, carry = score + float(qs[i][k]), score + c["attack"]
                    kids.append((line, first if first is not None else c, s, c, carry))
            kids.sort(key=lambda x: -x[0])
            kids = kids[:self.beam]
            best_line = kids[0]
            if ply == self.depth - 1:
                break
            frontier = []
            for _, first, s, c, score in kids:
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
    auto = args.style == "auto"
    rule_args = ("--no-180",) if args.no_180 else ()
    pl = Player(args.model, args.depth, args.beam, STYLES["normal" if auto else args.style], args.mode, args.policy, args.propose,
                temp=args.temp, seed=seed, merge=args.merge, merge_override=args.merge_override, expand_args=rule_args)
    cmd = [BOT, "play-ext", "--mode", args.mode, *rule_args, "--pieces", str(args.pieces), "--seed", str(seed)]
    if auto and getattr(pl, "n_styles", 2) < 3:
        raise SystemExit("--style auto needs a model with a danger style (3 style rows)")
    if args.shortlist or args.takeover >= 0:
        w = args.teacher_weights or (os.path.join(HERE, "..", "bot", "weights", "normal.txt") if args.style != "gate4" else
                                     os.path.join(HERE, "..", "bot", "results", "gate4", "gate4.txt"))
        cmd += ["--shortlist", "--weights", w, "--threads", "1", "--any-cores"]
    if args.garbage:
        cmd += ["--garbage", args.garbage]
    if args.replay:   # Godot replay of the game (bot play-ext --replay)
        name = os.path.basename(args.model).replace("model_", "").replace(".pt", "") + "_d%d" % args.depth
        if args.merge:      # ToMe at another merge schedule than it was trained with: say so in the replay's name
            name += "_merge" + "-".join(f.split(".")[-1] if "." in f else f for f in args.merge.split(","))
        os.makedirs(args.replay, exist_ok=True)
        cmd += ["--replay", os.path.join(args.replay, "%s_s%d.jsonl" % (name, seed)), "--label", name]
    game = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(os.path.join(args.bot_log, "game_s%d.err" % seed), "w") if args.bot_log else subprocess.DEVNULL, text=True, bufsize=1)
    times = []
    states = []
    danger_moves = 0
    while True:
        st = readjson(game)
        if st.get("done"):
            break
        if auto:      # the danger style on trouble boards, normal otherwise (decided per move)
            b = T.board_from_replay(P.rows_to_strings(st["rows"]))
            mh, nh = int(T.heights(b).max()), T.holes_of(b)
            hot = mh >= args.auto_height or nh >= args.auto_holes
            if not hot and pl.teacher == STYLES["danger"] and args.auto_exit_height >= 0:
                # hysteresis: once in danger, stay until the board is this clean
                hot = mh >= args.auto_exit_height or nh > args.auto_exit_holes
            pl.teacher = STYLES["danger"] if hot else STYLES["normal"]
            danger_moves += hot
        if args.dump:
            states.append({"seed": seed, "n": st["n"], "rows": st["rows"], "b2b": st["b2b"], "piece": st["piece"],
                           "queue": st["queue"], "hold": st["hold"], "style": args.style})
            dumped = states[-1]
        t0 = time.perf_counter()
        if args.takeover >= 0:
            # the model plays until move `takeover`, then the teacher's own search (every legal move offered)
            if st["n"] < args.takeover:
                c = pl.decide(st)
                moves = [c] if c else []
            else:
                moves = pl.shortlist(st, 10000)
            move = {"moves": [{"hold": c["hold"], "cells": c["cells"]} for c in moves]}
        elif args.shortlist:
            moves = pl.shortlist(st, args.shortlist)
            move = {"moves": [{"hold": c["hold"], "cells": c["cells"]} for c in moves]}
        else:
            c = pl.decide(st)
            move = {"hold": c["hold"], "cells": c["cells"]} if c else {"hold": 0, "cells": []}
            if args.dump and pl.last_entropy is not None:
                dumped["entropy"] = round(pl.last_entropy, 4)
        times.append(time.perf_counter() - t0)
        game.stdin.write(json.dumps(move) + "\n")
        game.stdin.flush()
    game.wait()
    pl.close()
    st["danger_moves"] = danger_moves
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
    ap.add_argument("--style", default="normal", choices=("normal", "gate4", "danger", "auto"),
                    help="the style input; auto = danger on trouble boards (--auto-height / --auto-holes), normal otherwise")
    ap.add_argument("--policy", default="logit", choices=("logit", "q1", "qs"), help="score moves by the policy or a value head")
    ap.add_argument("--propose", type=int, default=0, help="with a value head: choose only among the policy's top-k moves (0 = all)")
    ap.add_argument("--replay", default="", help="directory: write each game as a Godot replay (JSONL, the viewer's format)")
    ap.add_argument("--temp", type=float, default=0.0, help="depth 1: sample moves from the policy at this temperature (0 = argmax)")
    ap.add_argument("--auto-height", type=int, default=12, help="--style auto: the danger style when the stack is at least this high")
    ap.add_argument("--auto-holes", type=int, default=6, help="--style auto: ... or has at least this many holes")
    ap.add_argument("--auto-exit-height", type=int, default=-1, help="--style auto, hysteresis: once in danger, back to normal only "
                    "when the stack is below this (and holes <= --auto-exit-holes); -1 = no hysteresis")
    ap.add_argument("--auto-exit-holes", type=int, default=0)
    ap.add_argument("--takeover", type=int, default=-1, help="the model plays until this move, then the teacher's full search takes over (0 = teacher alone)")
    ap.add_argument("--teacher-weights", default="", help="--shortlist / --takeover: the teacher's weights (default normal.txt, gate4.txt for --style gate4)")
    ap.add_argument("--shortlist", type=int, default=0, help="intuition + calculation: the policy's top-k moves, then the "
                    "teacher's full search (its handcrafted V) picks among them")
    ap.add_argument("--mode", default="normal")
    ap.add_argument("--no-180", action="store_true", help="no 180 rotation (Cold Clear's rules; bot --no-180)")
    ap.add_argument("--garbage", default="")
    ap.add_argument("--jobs", type=int, default=0, help="parallel games (0 = P-core logical processors)")
    ap.add_argument("--out", default="")
    ap.add_argument("--bot-log", default="", help="directory for each game's bot stderr (debugging)")
    ap.add_argument("--merge-override", default="", help="ToMe: at inference, LAYER:FRACTION[,...] -- that layer merges this share "
                    "of its tokens instead of its schedule / budget step (e.g. 6:0.30)")
    ap.add_argument("--merge", default="", help="ToMe: play with this merge schedule (per layer, e.g. 0,0.17,0.25,0.25) "
                                                "instead of the one it was trained with; no retraining")
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
    line = ("AtTenTris %s%s%s%s depth %d%s style %s %s: APP %.3f (APPfull %.3f), died %d/%d, pieces %.0f avg, TSD %d, TST %d, quads %d, "
            "max height %d, illegal %d, %.1f ms/move (p90 %.1f) + search %.1f ms, %.0fs") % (
        args.policy, (" merge " + args.merge) if args.merge else "", (" top%d" % args.propose) if args.propose else "", (" shortlist %d + V search" % args.shortlist) if args.shortlist else "", args.depth, "" if args.depth <= 1 else " beam %d" % args.beam, args.style, ("garbage " + args.garbage) if args.garbage else "no garbage",
        attack / max(1, pieces), full, sum(r["died"] for r in res), args.games, pieces / args.games,
        sum(r["tsd"] for r in res), sum(r["tst"] for r in res), sum(r["quads"] for r in res),
        max(r["max_height"] for r in res), sum(r["illegal"] for r in res),
        float(np.mean([r["ms_per_move"] for r in res])), float(np.mean([r["ms_p90"] for r in res])),
        float(np.mean([r.get("search_ms", 0.0) for r in res])), time.time() - t0)
    if args.style == "auto":
        line += ", danger style on %.1f%% of moves (height >= %d or holes >= %d)" % (
            100 * sum(r["danger_moves"] for r in res) / max(1, pieces), args.auto_height, args.auto_holes)
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
