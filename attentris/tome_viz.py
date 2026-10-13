"""AtTenTris-ToMe blob viewer: for the attention viewer's positions, the blobs after each merging
layer and the model's move probabilities with each move's cross-attention over the final blobs.
CPU only (never shares the GPU with a training run).

    attentris/.venv/Scripts/python attentris/tome_viz.py --src attentris/attn_v51.html --out attentris/tome_viz.html

--merges "A;B" renders every position under each merge schedule (the model was trained with its own;
ToMe can merge more or less at inference without retraining), with a toggle in the page, and scores
the held-out positions under each. --games FILE.npz adds positions from games (packed, with the
teacher's labels), labelled "game at merge <--games-merge>".
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_tome as MT
import pcores
import tokenizer as T
import train as TR

HERE = os.path.dirname(os.path.abspath(__file__))
SLOTS = ("current", "hold", "preview1", "preview2", "preview3", "preview4", "preview5")


def piece_tokens(d, i):
    toks = []
    for j, kd in enumerate(d["tok_kind"][i]):
        if kd >= 0 and T.KINDS[kd] in SLOTS:
            pc = int(d["tok_piece"][i, j])
            toks.append({"kind": T.KINDS[kd], "piece": T.PIECES[pc] if 0 <= pc < 7 else None})
    return toks


def contact(a, b):
    """A pair of 4-adjacent cells, one in cell list a and one in b (for drawing a link between two blobs)."""
    sb = set(b)
    for c in a:
        r, q = divmod(c, 10)
        for nb, okn in ((c + 1, q < 9), (c - 1, q > 0), (c + 10, r < 19), (c - 10, r > 0)):
            if okn and nb in sb:
                return [int(c), int(nb)]
    return [int(a[0]), int(b[0])]


def merge_step(ev, k, layer, st, max_blocked=16):
    """One merge step of position k, stream st: the blobs going in (with their A / B side, if the split
    has sides), the merges made, and -- for A / B splits -- the blocked pairs: adjacent, same side, and
    more similar than the weakest merge this step made (so the merge would have been taken without the
    sides)."""
    cand, stream, mask = ev["cand"][k], ev["stream"][k], ev["mask"][k] > 0.5
    toks = [j for j in range(len(cand)) if bool(cand[j]) and int(stream[j]) == st and bool(mask[j].any())]
    at = {j: n for n, j in enumerate(toks)}
    blobs = [np.nonzero(mask[j].numpy())[0].tolist() for j in toks]
    S, adj = ev["S"][k], ev["adj"][k]
    sides = None
    if ev["A"] is not None:
        sides = ["A" if bool(ev["A"][k, j]) else "B" if bool(ev["Bs"][k, j]) else "" for j in toks]
    merges = []
    for j in toks:
        if bool(ev["sel"][k, j]):
            q = int(ev["dst"][k, j])
            merges.append([at[j], at[q]] + contact(blobs[at[j]], blobs[at[q]]) + [round(float(S[j, q]), 3)])
    weakest = min((m[4] for m in merges), default=float("inf"))
    blocked = []
    if sides is not None:
        for x in range(len(toks)):
            for y in range(x + 1, len(toks)):
                i, j = toks[x], toks[y]
                if sides[x] == sides[y] and bool(adj[i, j]) and float(S[i, j]) > weakest:
                    blocked.append([x, y] + contact(blobs[x], blobs[y]) + [round(float(S[i, j]), 3)])
        blocked.sort(key=lambda b: -b[4])
    return {"layer": layer, "stream": st, "blobs": blobs, "sides": sides, "merges": merges, "asked": int(ev["r"][k, st]),
            "weakest": None if weakest == float("inf") else weakest, "n_blocked": len(blocked), "blocked": blocked[:max_blocked]}


def render(net, d, idx, src):
    """Per position: blobs per merging layer (and stream), the final tokens, and the top moves."""
    events = []                                                 # every merge step's inputs and choices (model_tome.MERGE_HOOK)
    MT.MERGE_HOOK = lambda e: events.append({k: e[k] for k in ("A", "Bs", "S", "sel", "dst", "mask", "cand", "stream", "adj", "r")})
    try:
        with torch.no_grad():
            out, info = net(TR.batch_of(d, idx, "cpu"), return_attn=True)
    finally:
        MT.MERGE_HOOK = None
    prob = out.softmax(-1).numpy()
    cross = info["cross"].mean(1).numpy()                       # (P, NC, N_final), mean of heads
    merge_layers = [li for li, f in enumerate(net.merge) if f > 0]
    res = []
    for k, (q, i) in enumerate(zip(src, idx)):
        layers = []
        for li in merge_layers:
            t = info["merges"][li]
            mk, al = t["mask"][k].numpy() > 0.5, t["alive"][k].numpy()
            sm = t["stream"][k].numpy() if "stream" in t else np.zeros(len(al), int)
            for st in range(getattr(net, "streams", 1)):
                layers.append({"layer": li + 1, "stream": st,
                               "blobs": [np.nonzero(mk[j])[0].tolist() for j in np.nonzero(al)[0] if mk[j].any() and sm[j] == st]})
        last = info["merges"][-1]
        mk, al = last["mask"][k].numpy() > 0.5, last["alive"][k].numpy()
        sm = last["stream"][k].numpy() if "stream" in last else np.zeros(len(al), int)
        final = []                                              # the tokens the candidates attend to, in tensor order
        for j in range(len(al)):
            if not al[j]:
                final.append(None)
            elif mk[j].any():
                final.append({"cells": np.nonzero(mk[j])[0].tolist(), "stream": int(sm[j])})
            else:
                final.append({"piece": True})
        # name the piece tokens: the model keeps them in their packed order (current, hold, previews)
        pk = [t_ for t_ in q["tokens"] if t_["kind"] in SLOTS]
        pi = 0
        for f in final:
            if f is not None and f.get("piece"):
                f["kind"] = SLOTS[pi] if pi < len(SLOTS) else "piece"
                f["p"] = pk[pi]["piece"] if pi < len(pk) else None
                pi += 1
        nc = int(d["n_cand"][i])
        order = np.argsort(-prob[k][:nc])
        ch = int(d["chosen"][i])

        def cand(c):
            m = d["cand_meta"][i, c]
            return {"i": int(c), "cells": [int(x) for x in d["cand_cells"][i, c] if x >= 0], "piece": T.PIECES[m[0]],
                    "hold": int(m[1]), "spin": int(m[2]), "lines": int(m[3]), "attack": int(m[4]), "p": float(prob[k][c]),
                    "attn": [float(cross[k, c, j]) if final[j] is not None else 0.0 for j in range(len(final))]}
        steps = [merge_step(ev, k, li + 1, st) for ev, li in zip(events, merge_layers) for st in range(getattr(net, "streams", 1))]
        res.append({"layers": layers, "steps": steps, "final": final, "chosen": cand(ch), "top": [cand(int(c)) for c in order[:8]],
                    "rank": int(np.where(order == ch)[0][0]) + 1})
    return res


def val_top1(net, d, batch=256):
    """Top-1 agreement with the teacher on a packed set, per teacher."""
    hit, cnt = np.zeros(2), np.zeros(2)
    with torch.no_grad():
        for s in range(0, len(d["chosen"]), batch):
            idx = np.arange(s, min(len(d["chosen"]), s + batch))
            out = net(TR.batch_of(d, idx, "cpu"))
            top = out.argmax(-1).numpy() == d["chosen"][idx]
            for t in (0, 1):
                m = d["teacher"][idx] == t
                hit[t] += top[m].sum(); cnt[t] += m.sum()
    return {name: float(hit[t] / max(1, cnt[t])) for t, name in ((0, "normal"), (1, "gate4"))}


def load_packed(path, pretok):
    d = TR.load(path)
    if pretok:
        d["pretok"] = np.load(path.replace(".npz", ".pretok.npy"))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "data", "model_tome.pt"))
    ap.add_argument("--data", default=os.path.join(HERE, "data", "v51", "val.npz"))
    ap.add_argument("--src", default=os.path.join(HERE, "attn_v51.html"), help="take the positions (and their labels) from this viewer page")
    ap.add_argument("--out", default=os.path.join(HERE, "tome_viz.html"))
    ap.add_argument("--label", default="")
    ap.add_argument("--random", type=int, default=0, help="also show this many random validation positions")
    ap.add_argument("--merges", default="", help="merge schedules to render, ';'-separated (default: the trained one)")
    ap.add_argument("--games", default="", help="packed positions from games (pack.py + pretok.py) to add")
    ap.add_argument("--games-merge", default="", help="the merge schedule those games were played at (their label)")
    ap.add_argument("--games-notes", default="", help="JSON '<seed>_<n>' -> note for the game positions (sample_states.py)")
    ap.add_argument("--play", default="", help="JSON list of {value, label} play-test results for the page summary")
    ap.add_argument("--overrides", default="", help="inference-time merge overrides to render, ';'-separated LAYER:FRACTION "
                    "specs ('none' = as trained; 'split=hash' renders with the direction-blind A/B split), e.g. "
                    "'none;6:0.30;split=hash' (model_tome frac_override / split)")
    ap.add_argument("--title", default="", help="page title (default: the template's)")
    ap.add_argument("--val-eval", action="store_true", help="score the held-out set under each schedule (top-1)")
    args = ap.parse_args()
    pcores.pin()
    torch.set_num_threads(8)
    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    a = ck["args"]
    trained = a["merge"]
    scheds = args.merges.split(";") if args.merges else [trained]
    # variants: (merge schedule, override spec, name shown in the page)
    own_split = a.get("tome_split", "token")
    if args.overrides:
        variants = []
        for o in args.overrides.split(";"):
            items = [] if o == "none" else o.split(",")
            sp = [t_.split("=", 1)[1] for t_ in items if t_.startswith("split=")]
            fr = [t_ for t_ in items if not t_.startswith("split=")]
            name = [{"hash": "hash split at inference (bug-fixed; not retrained)", "token": "token split (as trained up to v4.1)",
                     "greedy": "greedy matching at inference (no A/B; not retrained)"}.get(sp[0], sp[0] + " split")] if sp else []
            name += ["layer %s merges %d%%" % (k, round(100 * float(v))) for k, v in (q.split(":") for q in fr)]
            variants.append((trained, ",".join(fr), "as trained" if not name else "+ " + ", ".join(name), sp[0] if sp else own_split))
    else:
        variants = [(sc, "", None, own_split) for sc in scheds]
    net = MT.AtTenTrisToMe(d=a["d"], layers=a["layers"], merge=[float(f) for f in trained.split(",")], sky=a.get("tome_sky", False),
                          pretok=a.get("tome_pretok", False), flat_bias=a.get("tome_flat", False),
                          streams=a.get("tome_streams", 1), budget=MT.budget_of(a.get("tome_budget", "")), split=a.get("tome_split", "token"))
    net.load_state_dict(ck["model"])
    net.eval()
    d = load_packed(args.data, a.get("tome_pretok"))
    s = open(args.src, encoding="utf-8").read()
    src = json.JSONDecoder().raw_decode(s[s.index('{"summary"'):])[0]["positions"]
    at = {(int(d["teacher"][i]), int(d["seed"][i]), int(d["n"][i])): i for i in range(len(d["chosen"]))}
    idx = [at[(["normal", "gate4"].index(q["teacher"]), q["seed"], q["n"])] for q in src]
    if args.random:
        rng = np.random.default_rng(7)
        extra = [int(i) for i in rng.permutation(len(d["chosen"])) if int(i) not in set(idx)][:args.random]
        for i in extra:
            src.append({"why": "random sample", "teacher": ["normal", "gate4"][d["teacher"][i]], "seed": int(d["seed"][i]),
                        "n": int(d["n"][i]), "tokens": piece_tokens(d, i)})
        idx += extra
    sets = [(d, np.array(idx), src, 0)]
    if args.games:
        g = load_packed(args.games, a.get("tome_pretok"))
        notes = json.load(open(args.games_notes)) if args.games_notes else {}
        gsrc = [{"why": "game at merge " + (args.games_merge or trained) + (
                     " · " + notes["%d_%d" % (g["seed"][i], g["n"][i])] if "%d_%d" % (g["seed"][i], g["n"][i]) in notes else ""), "teacher": ["normal", "gate4"][g["teacher"][i]],
                 "seed": int(g["seed"][i]), "n": int(g["n"][i]), "tokens": piece_tokens(g, i)} for i in range(len(g["chosen"]))]
        home = [v[0] for v in variants].index(args.games_merge) if args.games_merge in [v[0] for v in variants] else 0
        sets.append((g, np.arange(len(g["chosen"])), gsrc, home))

    pos, vals = [], []
    def use(v):
        net.merge = tuple(float(f) for f in v[0].split(","))
        net.frac_override = MT.override_of(v[1])
        net.split = v[3]
    for v in variants:
        use(v)
        if args.val_eval:
            vals.append(val_top1(net, d))
            print("%s: val top1 normal %.3f gate4 %.3f" % (v[2] or v[0], vals[-1]["normal"], vals[-1]["gate4"]), flush=True)
    for dd, ii, ss, home in sets:
        per = []
        for v in variants:
            use(v)
            per.append(render(net, dd, ii, ss))
        for k, (q, i) in enumerate(zip(ss, ii)):
            pk = [t_ for t_ in q["tokens"] if t_["kind"] in SLOTS]
            pos.append({"why": q["why"], "teacher": q["teacher"], "seed": q["seed"], "n": q["n"], "home": home,
                        "board": dd["cells"][i].tolist(), "pieces": {t_["kind"]: t_["piece"] for t_ in pk},
                        "n_cand": int(dd["n_cand"][i]), "sched": [per[si][k] for si in range(len(variants))]})
    log = ck["log"]
    label = args.label or "AtTenTris-ToMe · 200 cells + 7 pieces · adjacency-constrained merging · %d epochs" % len(log)
    last_val = log[-1]["val"] if log else {}
    data = json.dumps({"model": label, "streams": getattr(net, "streams", 1), "trained": trained, "schedules": [v[0] for v in variants],
                       "sched_names": [v[2] for v in variants] if args.overrides else None,
                       "val": {k: v["top1"] for k, v in last_val.items()}, "val_sched": vals,
                       "play": json.load(open(args.play)) if args.play else [],
                       "positions": pos}, separators=(",", ":"))
    tpl = open(os.path.join(HERE, "tome_viz_template.html"), encoding="utf-8").read()
    with open(args.out, "w", encoding="utf-8") as f:
        if args.title:
            tpl = tpl.replace("<title>AtTenTris-ToMe Blobs</title>", "<title>%s</title>" % args.title, 1)
        f.write(tpl.replace("__DATA__", data))
    print("%d positions x %d variants -> %s" % (len(pos), len(variants), args.out))


if __name__ == "__main__":
    main()
