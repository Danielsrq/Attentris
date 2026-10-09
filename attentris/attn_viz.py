"""Where does AtTenTris look? For validation positions, draw the cross-attention of a
candidate placement (the bot's move, or the model's own pick) over the board tokens,
projected onto the board through the token masks, and report the "pointing game":
for T-spin moves into a T structure, how often the most-attended token is that structure.

    <python with torch> attentris/attn_viz.py [--model attentris/data/model.pt] [--out attentris/attn.html]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M
import pcores
import tokenizer as T
import train as TR

HERE = os.path.dirname(os.path.abspath(__file__))
STRUCT_IDS = {T.KINDS.index(k) for k in T.STRUCTURE_KINDS}
PIECE_IDS = {T.KINDS.index(k) for k in T.PIECE_KINDS}
T_ID = T.PIECES.index("T")


def run(net, d, idx):
    with torch.no_grad():
        logit, att = net(TR.batch_of(d, idx), return_attn=True)
    return logit.softmax(-1).numpy(), att["cross"].mean(1).numpy()       # (B, NC), (B, NC, NT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "data", "model.pt"))
    ap.add_argument("--data", default=os.path.join(HERE, "data", "val.npz"))
    ap.add_argument("--out", default=os.path.join(HERE, "attn.html"))
    ap.add_argument("--positions", type=int, default=30)
    args = ap.parse_args()
    n_p = pcores.pin()
    torch.set_num_threads(int(os.environ.get("THREADS", n_p or 8)))
    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    net = M.AtTenTris(d=ck["args"]["d"], layers=ck["args"]["layers"])
    net.load_state_dict(ck["model"])
    net.eval()
    d = TR.load(args.data)
    n = len(d["chosen"])

    # --- whole validation set: agreement + pointing game ---
    top1, point_hits, point_n, point_base = 0, 0, 0, 0.0
    tspin_rows = []
    # epsilon-agreement: a pick counts if its search score is within EPS of the best one
    # (a pick the beam pruned is a miss); same for greedy V (argmax of the 1-ply score)
    EPS = (0.5, 1.0)
    agree = {who: {t: {e: 0 for e in EPS} for t in (0, 1)} for who in ("model", "greedy V")}
    hit1 = {who: {t: 0 for t in (0, 1)} for who in ("model", "greedy V")}
    per_t = {0: 0, 1: 0}
    for s in range(0, n, 512):
        idx = np.arange(s, min(n, s + 512))
        prob, cross = run(net, d, idx)
        ch = d["chosen"][idx]
        top1 += int((prob.argmax(-1) == ch).sum())
        nc = d["n_cand"][idx]
        ply1 = d["cand_score"][idx][..., 0].copy()
        for k in range(len(idx)):
            ply1[k, nc[k]:] = -np.inf
        picks = {"model": prob.argmax(-1), "greedy V": ply1.argmax(-1)}
        for k, i in enumerate(idx):
            t = int(d["teacher"][i])
            per_t[t] += 1
            bs = d["cand_score"][i, :, 1]
            bmax = np.nanmax(bs[:nc[k]]) if not np.all(np.isnan(bs[:nc[k]])) else np.nan
            for who, pk in picks.items():
                hit1[who][t] += int(pk[k] == ch[k])
                v = bs[pk[k]]
                for e in EPS:
                    agree[who][t][e] += int(not np.isnan(v) and bmax - v <= e)
        for k, i in enumerate(idx):
            meta = d["cand_meta"][i, ch[k]]
            kinds = d["tok_kind"][i]
            structs = [j for j, kd in enumerate(kinds) if kd in STRUCT_IDS]
            if meta[0] == T_ID and meta[2] == 2 and meta[3] >= 1 and structs:
                a = cross[k, ch[k]]
                board_toks = [j for j, kd in enumerate(kinds) if kd >= 0 and kd not in PIECE_IDS]
                best = max(board_toks, key=lambda j: a[j])
                # the structure the T actually went into: most overlap with its cells
                cells = [c for c in d["cand_cells"][i, ch[k]] if c >= 0]
                masks = M.unpack_masks(d["tok_mask"][i])
                into = max(structs, key=lambda j: masks[j][cells].sum())
                if masks[into][cells].sum() >= 3:
                    point_n += 1
                    point_hits += int(best == into)
                    point_base += 1.0 / len(board_toks)
                    tspin_rows.append(int(i))
    names = {0: "normal", 1: "gate4"}
    table = {who: {names[t]: {"top1": hit1[who][t] / max(1, per_t[t]),
                              **{"within %.1f" % e: agree[who][t][e] / max(1, per_t[t]) for e in EPS}} for t in (0, 1)}
             for who in agree}
    summary = {"positions": n, "top1": top1 / n, "agreement": table, "pointing_n": point_n,
               "pointing": point_hits / max(1, point_n), "pointing_random": point_base / max(1, point_n)}
    print(json.dumps(summary, indent=1))

    # --- positions to show: T-spins into structures, recess fills, random ---
    rng = np.random.default_rng(3)
    show = list(rng.choice(tspin_rows, size=min(10, len(tspin_rows)), replace=False)) if tspin_rows else []
    fit_ids = {T.KINDS.index(k) for k in T.FIT_KINDS}
    fits = []
    for i in rng.permutation(n)[:3000]:
        ch = d["chosen"][i]
        cells = sorted(c for c in d["cand_cells"][i, ch] if c >= 0)
        masks = M.unpack_masks(d["tok_mask"][i])
        for j, kd in enumerate(d["tok_kind"][i]):
            if kd in fit_ids and sorted(np.nonzero(masks[j])[0].tolist()) == cells:
                fits.append(int(i))
                break
        if len(fits) >= 10:
            break
    show += fits
    show += [int(i) for i in rng.choice(n, size=max(0, args.positions - len(show)), replace=False)]
    pos = []
    for i in show:
        prob, cross = run(net, d, np.array([i]))
        prob, cross = prob[0], cross[0]
        nc = int(d["n_cand"][i])
        ch = int(d["chosen"][i])
        order = np.argsort(-prob[:nc])
        masks = M.unpack_masks(d["tok_mask"][i])
        kinds = d["tok_kind"][i]
        toks = [{"kind": T.KINDS[kd], "cells": np.nonzero(masks[j])[0].tolist(),
                 "piece": T.PIECES[d["tok_piece"][i, j]] if d["tok_piece"][i, j] >= 0 and d["tok_piece"][i, j] < 7 else None}
                for j, kd in enumerate(kinds) if kd >= 0]

        def cand(c):
            m = d["cand_meta"][i, c]
            return {"i": int(c), "cells": [int(x) for x in d["cand_cells"][i, c] if x >= 0], "piece": T.PIECES[m[0]],
                    "hold": int(m[1]), "spin": int(m[2]), "lines": int(m[3]), "attack": int(m[4]),
                    "p": float(prob[c]), "attn": [float(cross[c, j]) for j, kd in enumerate(kinds) if kd >= 0]}
        why = "T-spin into a T structure" if i in tspin_rows else ("bot filled a piece-shaped recess" if i in fits else "random position")
        pos.append({"why": why, "teacher": ["normal", "gate4"][d["teacher"][i]], "seed": int(d["seed"][i]), "n": int(d["n"][i]),
                    "board": d["cells"][i].tolist(), "tokens": toks, "chosen": cand(ch),
                    "top": [cand(int(c)) for c in order[:3]], "rank": int(np.where(order == ch)[0][0]) + 1, "n_cand": nc})
    tpl = open(os.path.join(HERE, "attn_template.html"), encoding="utf-8").read()
    data = json.dumps({"summary": summary, "epochs": len(ck["log"]), "log": ck["log"], "positions": pos}, separators=(",", ":"))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(tpl.replace("__DATA__", data))
    print("%d positions -> %s" % (len(pos), args.out))


if __name__ == "__main__":
    main()
