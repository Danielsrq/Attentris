"""Token-level SHAP: which tokens drove the model's move?

For one position the players are its tokens (board-side structure tokens and the 7 piece tokens). A
coalition S keeps the tokens in S and blanks the rest as padding (as train.py --drop-kinds does); the board
cells and the candidates stay. The value is the log-probability of the move the model chooses with every
token present:
    v(S) = log softmax(logits(S))[a*],   a* = argmax logits(all tokens)
so a token gets credit for separating the chosen move from the alternatives, not for raising all logits.
Shapley values are estimated by permutation sampling in antithetic pairs (an order and its reverse), every
permutation telescoping exactly: per position sum(phi) = v(all) - v(none).

With --top N every one of the model's N best moves gets its own attribution (the same coalitions, so
no extra cost), like the per-move cross-attention of the attention viewer; --keep PAGE takes the
positions of an attention page (attn_viz.py output, from val.npz) instead of a random sample.

Positions: --per N from each --source file (validation sets in data/v51), split into calm / hot (stack
>= 12 or >= 6 holes, as play.py's auto style). Output: --out FILE.jsonl, one record per position (kinds,
boxes, phi, the move), and a summary by token kind: share of the total |phi|, mean phi, how often the
kind's tokens pushed toward the move, overall and per subset (calm / hot; the move: spin clear, plain
clear, quad, no clear).

    attentris/.venv/bin/python attentris/token_shap.py --model data/model_v51_d128_dagger5b_ft.pt \
        --source val.npz,val_danger.npz,val_dagger5all.npz --per 100 --perms 64 --out data/shap/dagger5b_ft.jsonl
    attentris/.venv/bin/python attentris/token_shap.py --summary data/shap/dagger5b_ft.jsonl
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M
import pcores
import tokenizer as T
import train as TR

HERE = os.path.dirname(os.path.abspath(__file__))


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    net = M.AtTenTris(d=a.get("d", 64), layers=a.get("layers", 4), n_kinds=ck["model"]["kind.weight"].shape[0],
                      cross_layers=a.get("cross_layers", 1), cross_loops=a.get("cross_loops", 1),
                      n_teachers=ck["model"]["style.weight"].shape[0], registers=a.get("registers", 0),
                      flat=not a.get("no_flat", False))
    net.load_state_dict(ck["model"])
    for mod in net.modules():                                     # coalitions without any token: attention outputs 0
        if isinstance(mod, M.Attention):
            mod.allow_empty = True
    drop = tuple(p for p in a.get("drop_kinds", "").split(",") if p)
    return net.eval(), drop


def board_stats(cells):
    b = cells.reshape(20, 10).astype(bool)                     # row 0 = bottom
    h = np.array([(np.nonzero(b[:, c])[0].max() + 1) if b[:, c].any() else 0 for c in range(10)])
    holes = sum(int((~b[:h[c], c]).sum()) for c in range(10))
    return int(h.max()), holes


def move_type(meta):
    piece, hold, spin, lines, attack = (int(x) for x in meta)
    if lines == 0:
        return "no clear"
    if spin > 0:
        return "spin clear"
    return "quad" if lines == 4 else "plain clear"


def blank(b, keep):
    """b: a 1-position batch; keep: (K, NT) bool over token slots -> a K-position batch."""
    K = keep.shape[0]
    out = {k: v.expand(K, *v.shape[1:]).clone() for k, v in b.items()}
    off = ~keep
    out["tok_kind"][off] = -1
    out["tok_mask"][off] = False
    out["tok_box"][off] = -1
    out["tok_extra"][off] = 0.0
    out["tok_piece"][off] = -1
    return out


@torch.no_grad()
def values(net, b, keep, cands, chunk):
    """log p of each move in cands, per coalition: (K, len(cands))."""
    vs = []
    for s in range(0, keep.shape[0], chunk):
        logit = net(blank(b, keep[s:s + chunk]))
        vs.append(logit.log_softmax(-1)[:, cands])
    return torch.cat(vs)


def shap_position(net, b, perms, rng, chunk, top=1, extra=None):
    """Shapley values of the tokens for log p of the model's top `top` moves (the same coalitions serve
    every move: Shapley values are linear in the value function). Returns the token slots, the moves
    (best first), phi (n tokens x moves), and v(all) / v(none) per move."""
    kind = b["tok_kind"][0]
    slots = torch.nonzero(kind >= 0).squeeze(-1)
    n, NT = len(slots), kind.shape[0]
    with torch.no_grad():
        full = net(b)[0]
    nc = int(torch.isfinite(full).sum())
    cands = torch.argsort(full, descending=True)[:min(top, nc)]
    if extra is not None and extra not in cands.tolist():          # e.g. the teacher's move
        cands = torch.cat([cands, torch.tensor([extra])])
    orders = []
    for _ in range(perms // 2):
        o = rng.permutation(n)
        orders += [o, o[::-1].copy()]
    keep = torch.zeros(len(orders), n + 1, NT, dtype=torch.bool)
    for i, o in enumerate(orders):
        for j in range(1, n + 1):
            keep[i, j, slots[o[:j]]] = True
    C = len(cands)
    v = values(net, b, keep.view(-1, NT), cands, chunk).view(len(orders), n + 1, C)
    phi = torch.zeros(n, C)
    for i, o in enumerate(orders):
        phi[torch.as_tensor(o.copy())] += (v[i, 1:] - v[i, :-1])
    phi /= len(orders)
    return slots.tolist(), cands.tolist(), phi, v[0, -1].tolist(), v[0, 0].tolist()


def run(args):
    pcores.pin()
    torch.set_num_threads(args.threads)
    net, drop = load_model(args.model)
    rng = np.random.default_rng(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    done = set()
    if os.path.exists(args.out):                                  # resumable
        for line in open(args.out):
            r = json.loads(line)
            done.add((r["source"], r["index"]))
    fh = open(args.out, "a")
    t0, count = time.time(), 0
    for src in args.source.split(","):
        d = TR.load(os.path.join(args.data, src))
        if drop:
            ids = [i for i, k in enumerate(T.KINDS) if any(k.startswith(p) for p in drop)]
            m = np.isin(d["tok_kind"], ids)
            for k, fill in (("tok_kind", -1), ("tok_mask", 0), ("tok_box", -1), ("tok_extra", 0), ("tok_piece", -1)):
                d[k] = d[k].copy()
                d[k][m] = fill
        if args.keep:                                             # the attention page's positions (attn_viz.py --keep)
            old = open(args.keep, encoding="utf-8").read()
            old = json.JSONDecoder().raw_decode(old[old.index('{"summary"'):])[0]["positions"]
            at = {(int(d["teacher"][i]), int(d["seed"][i]), int(d["n"][i])): i for i in range(len(d["chosen"]))}
            idx_all = [at[(["normal", "gate4"].index(q["teacher"]), q["seed"], q["n"])] for q in old]
        else:
            idx_all = np.random.default_rng(args.seed).choice(len(d["chosen"]), size=min(args.per, len(d["chosen"])), replace=False)
        for i in idx_all:
            i = int(i)
            if (src, i) in done:
                continue
            b = TR.batch_of(d, np.array([i]))
            slots, cands, phi, vfull, vnone = shap_position(net, b, args.perms, rng, args.chunk, args.top,
                                                            int(d["chosen"][i]) if args.top > 1 else None)
            target = cands[0]
            kinds = [T.KINDS[int(b["tok_kind"][0, s])] for s in slots]
            boxes = [b["tok_box"][0, s].tolist() for s in slots]
            height, holes = board_stats(d["cells"][i])
            meta = d["cand_meta"][i, target]
            rec = {"source": src, "index": i, "hot": bool(height >= 12 or holes >= 6), "height": height, "holes": holes,
                   "teacher_move": int(d["chosen"][i]), "move": target, "move_type": move_type(meta),
                   "move_meta": [int(x) for x in meta], "p_move": float(np.exp(vfull[0])), "v_all": vfull[0], "v_none": vnone[0],
                   "kinds": kinds, "boxes": boxes, "slots": slots, "phi": [round(float(x), 5) for x in phi[:, 0]]}
            if args.top > 1:                                      # every top move: its own attribution over the tokens
                rec["cands"] = [{"i": c, "p": float(np.exp(vfull[k])), "v_all": vfull[k], "v_none": vnone[k],
                                 "meta": [int(x) for x in d["cand_meta"][i, c]], "phi": [round(float(x), 5) for x in phi[:, k]]}
                                for k, c in enumerate(cands)]
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            count += 1
            if count % 20 == 0:
                print("%d positions, %.1f s each" % (count, (time.time() - t0) / count), flush=True)
    fh.close()
    summary(args.out)


def group(kind):
    """Token kinds grouped for the summary: piece slots by role, the rest by name."""
    if kind.startswith("preview"):
        return "preview"
    return kind


def summary(path):
    recs = [json.loads(l) for l in open(path)]
    subsets = [("all", lambda r: True), ("calm", lambda r: not r["hot"]), ("hot", lambda r: r["hot"])]
    for mt in ("spin clear", "plain clear", "quad", "no clear"):
        subsets.append((mt, lambda r, mt=mt: r["move_type"] == mt))
    print("token SHAP: %s, %d positions (hot %d); value = log p(chosen move); sum phi = v(all) - v(none), mean %.2f"
          % (path, len(recs), sum(r["hot"] for r in recs), np.mean([r["v_all"] - r["v_none"] for r in recs])))
    print("share = the kind's part of each position's total |phi|, averaged over positions (with the kind present);"
          " pres = share of positions that have the kind; mean phi; toward = share of its tokens with phi > 0")
    for name, f in subsets:
        rs = [r for r in recs if f(r)]
        if not rs:
            continue
        agg = {}
        for r in rs:
            tot = sum(abs(x) for x in r["phi"]) or 1.0
            per = {}
            for k, p in zip(r["kinds"], r["phi"]):
                g = group(k)
                e = per.setdefault(g, [0.0, 0.0, 0, 0])
                e[0] += abs(p) / tot
                e[1] += p
                e[2] += p > 0
                e[3] += 1
            for g, e in per.items():
                a = agg.setdefault(g, [0.0, 0.0, 0, 0, 0])
                a[0] += e[0]
                a[1] += e[1]
                a[2] += e[2]
                a[3] += e[3]
                a[4] += 1
        print("\n[%s] %d positions" % (name, len(rs)))
        print("  %-10s %6s %6s %8s %7s" % ("kind", "share", "pres", "mean phi", "toward"))
        for g, a in sorted(agg.items(), key=lambda kv: -kv[1][0] / len(rs)):
            print("  %-10s %6.3f %6.2f %8.3f %7.2f" % (g, a[0] / a[4], a[4] / len(rs), a[1] / a[3], a[2] / a[3]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--data", default=os.path.join(HERE, "data", "v51"))
    ap.add_argument("--source", default="val.npz,val_danger.npz,val_dagger5all.npz")
    ap.add_argument("--per", type=int, default=100, help="positions per source file")
    ap.add_argument("--perms", type=int, default=64, help="permutations per position (antithetic pairs)")
    ap.add_argument("--top", type=int, default=1, help="attribute the model's top N moves (1 = the chosen move only; "
                    "with N > 1 the teacher's move too)")
    ap.add_argument("--keep", default="", help="an attention page (attn_viz.py output): its positions, from --source (val.npz)")
    ap.add_argument("--chunk", type=int, default=512, help="coalitions per forward pass")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "shap", "shap.jsonl"))
    ap.add_argument("--summary", default="", help="only print the summary of this output file")
    args = ap.parse_args()
    if args.summary:
        summary(args.summary)
    else:
        run(args)


if __name__ == "__main__":
    main()
