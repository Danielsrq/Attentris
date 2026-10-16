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
BOT = os.path.join(HERE, "..", "bot", "target", "release", "bot.exe" if os.name == "nt" else "bot")
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
        elif ck["args"].get("plan_variant"):     # attentris-autoreg / -plan: move 1 includes the step-1 embedding
            import model_plan as MP
            self.net = MP.AtTenTrisPlan(d=ck["args"]["d"], layers=ck["args"]["layers"], n_kinds=ck["model"]["kind.weight"].shape[0],
                                        cross_layers=ck["args"].get("cross_layers", 1), cross_loops=ck["args"].get("cross_loops", 1),
                                        n_teachers=ck["model"]["style.weight"].shape[0], variant=ck["args"]["plan_variant"],
                                        plan_layers=ck["args"].get("plan_layers", 1))
            self.n_styles = ck["model"]["style.weight"].shape[0]
        else:
            self.net = M.AtTenTris(d=ck["args"]["d"], layers=ck["args"]["layers"], n_kinds=ck["model"]["kind.weight"].shape[0],
                          cross_layers=ck["args"].get("cross_layers", 1), cross_loops=ck["args"].get("cross_loops", 1),
                          n_teachers=ck["model"]["style.weight"].shape[0])
            self.n_styles = ck["model"]["style.weight"].shape[0]
        self.variant = ck["args"].get("plan_variant", "")
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
        self.n_committed = 0                  # --commit 2: moves played from the previous decision
        self.n_illegal = self.n_plan_off = 0  # --commit 2, pieces: step 2 not legal / step 1 not the move played
        self.committed = None
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
        self.last_root = root
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

    def plan_window(self, st, root, c, K):
        """attentris-autoreg, --plan-check K: the model's own plan from here, steps 2..K, each step
        chained on the previous predicted move (its legal moves from Rust, mapped into this board's
        frame through the earlier planned clears). Returns [(hold, sorted cells)] for steps 2..K
        (shorter when the chain cannot go on: the known queue runs out)."""
        import torch
        a = arrays([(st, root)], self.teacher, self.tok_version, self.drop)
        i = root["cands"].index(c)
        preds, pcs = [], []
        with torch.inference_mode():
            _, enc, _ = self.net.move1(a)
            prev = [(a["cand_cells"][:, i], a["cand_meta"][:, i].clamp(min=0))]
            state, move, origin = st, c, list(range(40))
            for k in range(2, K + 1):
                s2 = self.step2(state, move, origin)
                if s2 is None:
                    break
                nxt, cands2, pc, origin = s2
                pcs.append(pc)
                b = {key: torch.cat([q[key] for q in pcs], 1) for key in pc}
                logits, _ = self.net.autoreg_steps(b, enc, a["teacher"], k, prev)
                j = int(logits[0, -1, :len(cands2)].argmax())
                m = cands2[j]
                preds.append((m["hold"], sorted(zip(m["cells"][0::2], m["cells"][1::2]))))
                prev.append((pc["pc_cells"][:, 0, j], pc["pc_meta"][:, 0, j].clamp(min=0)))
                state, move = nxt, m
        return preds

    def step2(self, st, c, origin0=None):
        """After our move c at st: the next state, its legal moves (bot expand) and those moves as
        plan-step inputs in st's frame (pc_cells / pc_meta / pc_n / pc_target, as pack_plan.py).
        None when the next state cannot be predicted (queue too short, or a rule the replay below
        does not model)."""
        import torch
        used, queue = c["used"], st["queue"]
        if len(queue) < used:                  # the next piece is not known
            return None
        nxt = {"rows": c["rows"], "piece": queue[used - 1], "queue": queue[used:], "hold": c["hold_after"], "b2b": c["b2b"]}
        # this board's rows after our move: placed, full rows cleared; origin[y] = the row it came from
        rows = list(st["rows"])
        for q in range(4):
            rows[c["cells"][2 * q + 1]] |= 1 << c["cells"][2 * q]
        full = [y for y in range(40) if rows[y] == 1023]
        kept_rows = [y for y in range(40) if rows[y] != 1023]
        if [0] * len(full) + [rows[y] for y in kept_rows] != list(c["rows"]):
            return None
        origin = [-(len(full) - i) for i in range(len(full))] + kept_rows
        if origin0 is not None:     # st is itself a planned state: map on into the root's frame
            origin = [origin0[o] if o >= 0 else o - 1000 for o in origin]
        cands2 = self.expand(nxt)["cands"][:P.NC]
        if not cands2:
            return None
        cc = np.full((1, 1, P.NC, 4), -1, np.int64)
        cm = np.zeros((1, 1, P.NC, 5), np.int64)
        for j, m in enumerate(cands2):
            cl = m["cells"]
            cc[0, 0, j] = [(39 - origin[cl[2 * q + 1]]) * 10 + cl[2 * q] if 0 <= 39 - origin[cl[2 * q + 1]] < 20 else -2 for q in range(4)]
            cm[0, 0, j] = (P.PIECE_ID[m["piece"]], m["hold"], m["spin"], m["lines"], max(0, min(m["attack"], 127)))
        pc = {"pc_cells": torch.from_numpy(cc), "pc_meta": torch.from_numpy(cm),
              "pc_n": torch.tensor([[len(cands2)]]), "pc_target": torch.tensor([[-1]])}
        return nxt, cands2, pc, origin

    def rot_table(self):
        """(piece, normalised shape) -> rotation class, as pack_plan.py derives it (O 0, I / S / Z
        rot % 2, T / L / J rot), read from the plan labels until all 19 shapes are seen."""
        if getattr(self, "_rot", None) is None:
            import glob
            import pack_plan as PP
            self._rot = {}
            for f in sorted(glob.glob(os.path.join(HERE, "data", "plan", "raw", "*.jsonl"))):
                for line in open(f):
                    for h, pc, rot, cl in json.loads(line)["plan"]:
                        xy = [(cl[2 * q], 39 - cl[2 * q + 1]) for q in range(4)]
                        self._rot[(pc, PP.shape(xy))] = PP.rot_class(pc, rot)
                if len(self._rot) >= 19:
                    break
        return self._rot

    def place_of(self, m):
        """A legal move's 880-way placement (pack_plan.py coord without the hold bit), or -1."""
        import pack_plan as PP
        xy = [(m["cells"][2 * q], 39 - m["cells"][2 * q + 1]) for q in range(4)]
        rc = self.rot_table().get((m["piece"], PP.shape(xy)))
        c = -1 if rc is None else PP.coord(0, rc, xy)
        return c

    def decide_commit_pieces(self, st):
        """attentris-plan (pieces), --commit 2: move 1 from the policy; the pieces heads' step 2 (the
        piece most likely placed at step 2, at its most likely place) is played at X+1 without a new
        decision when (a) their step 1 is the move just played, (b) the board is the predicted one
        (no garbage), and (c) that placement is a legal move at X+1; else a normal decision there
        ((c) failing counts as illegal)."""
        import torch
        kept, self.committed = getattr(self, "committed", None), None
        if kept is not None and kept["n"] == st["n"] and kept["rows"] == st["rows"] and kept["hold"] == st["hold"] \
                and kept["piece"] == st["piece"]:
            root = self.expand(st)
            hit = [m for m in root["cands"] if m["piece"] == kept["piece_type"] and self.place_of(m) == kept["place"]]
            if hit:
                self.n_committed += 1
                return hit[0]
            self.n_illegal += 1
        root = self.expand(st)
        self.last_root = root
        if not root["cands"]:
            return None
        n = min(len(root["cands"]), P.NC)
        a = arrays([(st, root)], self.teacher, self.tok_version, self.drop)
        with torch.inference_mode():
            out = self.net(a, heads=True, n_steps=6)
        c = root["cands"][int(out["logit"][0, :n].argmax())]
        when = out["when_logits"][0].log_softmax(-1)        # (7 slots, 7 classes)
        where = out["where_logits"][0].argmax(-1)           # (7,)
        slot1, slot2 = int(when[:, 1].argmax()), int(when[:, 2].argmax())
        pieces = [st["piece"], st["hold"]] + list(st["queue"])
        if slot1 != slot2 and int(where[slot1]) == self.place_of(c) and pieces[slot1] == c["piece"] and slot2 < len(pieces) and pieces[slot2]:
            used, queue = c["used"], st["queue"]
            if len(queue) >= used:
                self.committed = {"n": st["n"] + 1, "rows": c["rows"], "hold": c["hold_after"], "piece": queue[used - 1],
                                  "piece_type": pieces[slot2], "place": int(where[slot2])}
        else:
            self.n_plan_off += 1
        return c

    def decide_autoreg_beam(self, st):
        """attentris-autoreg, --depth 2 --autoreg-beam: the depth-2 policy beam (decide()'s: the
        policy's top --beam first moves, line = log p(move 1) + log p(move 2), play the best line's
        first move), but move 2's probabilities come from the step-2 cross pass over the ROOT's
        encoding: the children are expanded (their legal moves) but never tokenized or encoded."""
        import torch
        root = self.expand(st)
        self.last_root = root
        if not root["cands"]:
            return None
        n = min(len(root["cands"]), P.NC)
        a = arrays([(st, root)], self.teacher, self.tok_version, self.drop)
        with torch.inference_mode():
            out, enc, _ = self.net.move1(a)
            lp1 = out["logit"][0, :n].log_softmax(-1)
            top = [int(i) for i in lp1.argsort(descending=True)[:self.beam]]
            kids = []
            for i in top:
                s2 = self.step2(st, root["cands"][i])
                if s2 is not None:
                    kids.append((i, s2))
            if not kids:
                return root["cands"][top[0]]
            B = len(kids)
            rep_enc = {k: (v.expand(B, *v.shape[1:]) if torch.is_tensor(v) else v) for k, v in enc.items()}
            b = {"pc_cells": torch.cat([s2[2]["pc_cells"] for _, s2 in kids]), "pc_meta": torch.cat([s2[2]["pc_meta"] for _, s2 in kids]),
                 "pc_n": torch.cat([s2[2]["pc_n"] for _, s2 in kids]), "pc_target": torch.cat([s2[2]["pc_target"] for _, s2 in kids])}
            ids = torch.tensor([i for i, _ in kids])
            prev = [(a["cand_cells"][0, ids], a["cand_meta"][0, ids].clamp(min=0))]
            logits, _ = self.net.autoreg_steps(b, rep_enc, a["teacher"].expand(B), 2, prev)
            best, best_i = -1e30, kids[0][0]
            for r, (i, (_, cands2, _)) in enumerate(kids):
                line = float(lp1[i]) + float(logits[r, 0, :len(cands2)].log_softmax(-1).max())
                if line > best:
                    best, best_i = line, i
        return root["cands"][best_i]

    def decide_commit(self, st):
        """attentris-autoreg, --commit 2: two moves per encoder pass. At X: move 1 by the policy, then
        move 2 from the same board encoding (one more cross pass over the next state's legal moves),
        kept for X+1. At X+1 the kept move is played without a new decision when the state is the one
        predicted (no garbage arrived; else a normal decision)."""
        import torch
        kept, self.committed = getattr(self, "committed", None), None
        if kept is not None and kept["n"] == st["n"] and kept["rows"] == st["rows"] and kept["hold"] == st["hold"] \
                and kept["piece"] == st["piece"] and kept["queue"] == st["queue"][:len(kept["queue"])]:
            self.n_committed += 1
            return kept["move"]
        root = self.expand(st)
        self.last_root = root
        if not root["cands"]:
            return None
        n = len(root["cands"])
        a = arrays([(st, root)], self.teacher, self.tok_version, self.drop)
        with torch.inference_mode():
            out, enc, _ = self.net.move1(a)
            i = int(out["logit"][0, :min(n, P.NC)].argmax())
            c = root["cands"][i]
            step2 = self.step2(st, c)
            if step2 is not None:
                nxt, cands2, pc = step2[:3]
                a.update(pc)
                prev = [(a["cand_cells"][:, i], a["cand_meta"][:, i].clamp(min=0))]
                logits, _ = self.net.autoreg_steps(a, enc, a["teacher"], 2, prev)
                m = cands2[int(logits[0, 0, :len(cands2)].argmax())]
                self.committed = {"n": st["n"] + 1, "rows": c["rows"], "hold": nxt["hold"], "piece": nxt["piece"],
                                  "queue": nxt["queue"], "move": m}
        return c

    def close(self):
        self.ex.stdin.close()
        self.ex.wait()


def play_game(args, seed):
    pcores.pin(verbose=False)
    auto = args.style == "auto"
    rule_args = ("--no-180",) if args.no_180 else ()
    if (args.plan_check or args.commit > 1) and (args.depth > 1 or args.style == "auto"):
        raise SystemExit("--plan-check / --commit: depth 1, one style")
    if args.autoreg_beam and (args.depth != 2 or args.style == "auto"):
        raise SystemExit("--autoreg-beam: --depth 2 (an N = 2 attentris-autoreg model), one style")
    if args.commit > 2:
        raise SystemExit("--commit: 2 for now (an N = 2 model)")
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
    windows, played = {}, {}      # --plan-check: the plan made at move n; the move played at n
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
        elif args.autoreg_beam:
            c = pl.decide_autoreg_beam(st)
            move = {"hold": c["hold"], "cells": c["cells"]} if c else {"hold": 0, "cells": []}
        elif args.commit > 1:
            c = pl.decide_commit_pieces(st) if pl.variant == "pieces" else pl.decide_commit(st)
            move = {"hold": c["hold"], "cells": c["cells"]} if c else {"hold": 0, "cells": []}
        else:
            c = pl.decide(st)
            move = {"hold": c["hold"], "cells": c["cells"]} if c else {"hold": 0, "cells": []}
            if args.plan_check and c:
                windows[st["n"]] = pl.plan_window(st, pl.last_root, c, args.plan_check)
            if c:
                played[st["n"]] = (c["hold"], sorted(zip(c["cells"][0::2], c["cells"][1::2])))
            if args.dump and pl.last_entropy is not None:
                dumped["entropy"] = round(pl.last_entropy, 4)
        times.append(time.perf_counter() - t0)
        game.stdin.write(json.dumps(move) + "\n")
        game.stdin.flush()
    game.wait()
    pl.close()
    st["danger_moves"] = danger_moves
    if args.plan_check:
        # windows X..X+K-1 with every move played and no garbage arriving inside (game.rs: a batch
        # lands before move n when n % every == 0), as plan_consistency_cem.py counts the teacher
        K = args.plan_check
        every = int(args.garbage.split(",")[0]) if args.garbage else 0
        alone, prefix, n_win, short = np.zeros(K + 1), np.zeros(K + 1), 0, 0
        for x, preds in windows.items():
            if any(x + j not in played for j in range(K)) or (every and any((x + j) % every == 0 for j in range(1, K))):
                continue
            if len(preds) < K - 1:
                short += 1
                continue
            n_win += 1
            ok = True
            for k in range(2, K + 1):
                hit = preds[k - 2] == played[x + k - 1]
                ok = ok and hit
                alone[k] += hit
                prefix[k] += ok
        st["plan_win"], st["plan_alone"], st["plan_prefix"], st["plan_short"] = n_win, alone.tolist(), prefix.tolist(), short
    st["committed"] = getattr(pl, "n_committed", 0)
    st["commit_illegal"], st["commit_plan_off"] = getattr(pl, "n_illegal", 0), getattr(pl, "n_plan_off", 0)
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
    ap.add_argument("--plan-check", type=int, default=0, help="attentris-autoreg, K >= 2: plan consistency, the model's "
                    "own plan of steps 2..K at every move X vs the moves it then plays at X+1..X+K-1 (step alone and all "
                    "steps so far; windows with garbage inside skipped), as plan_consistency_cem.py for the teacher")
    ap.add_argument("--autoreg-beam", action="store_true", help="attentris-autoreg, --depth 2: the policy beam with move 2 "
                    "from the step-2 pass over the root's encoding (children expanded, never re-encoded)")
    ap.add_argument("--commit", type=int, default=1, help="attentris-autoreg: moves played per decision (2: move 2 "
                    "comes from the same encoder pass as move 1, unless garbage changed the board)")
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
    if args.autoreg_beam:
        line += ", move 2 from the autoreg step-2 pass (children not encoded)"
    if args.commit > 1:
        line += ", %.1f%% of moves committed" % (100 * sum(r["committed"] for r in res) / max(1, pieces))
        if sum(r["commit_illegal"] + r["commit_plan_off"] for r in res):
            line += " (pieces: step 2 illegal %d, step 1 not the policy's move %d)" % (
                sum(r["commit_illegal"] for r in res), sum(r["commit_plan_off"] for r in res))
    if args.plan_check:
        nw = sum(r["plan_win"] for r in res)
        al, pr = np.sum([r["plan_alone"] for r in res], 0), np.sum([r["plan_prefix"] for r in res], 0)
        line += ", plan check over %d windows (%d plans cut short): step k alone %s; steps 2..k all %s" % (
            nw, sum(r["plan_short"] for r in res), " ".join("%d:%.1f%%" % (k, 100 * al[k] / max(1, nw)) for k in range(2, args.plan_check + 1)),
            " ".join("%d:%.1f%%" % (k, 100 * pr[k] / max(1, nw)) for k in range(2, args.plan_check + 1)))
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
