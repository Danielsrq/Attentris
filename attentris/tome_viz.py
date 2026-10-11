"""AtTenTris-ToMe blob viewer: for the attention viewer's positions, the blobs after each merging
layer and the model's move probabilities with each move's cross-attention over the final blobs.
CPU only (never shares the GPU with a training run).

    attentris/.venv/Scripts/python attentris/tome_viz.py --src attentris/attn_v51.html --out attentris/tome_viz.html
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "data", "model_tome.pt"))
    ap.add_argument("--data", default=os.path.join(HERE, "data", "v51", "val.npz"))
    ap.add_argument("--src", default=os.path.join(HERE, "attn_v51.html"), help="take the positions (and their labels) from this viewer page")
    ap.add_argument("--out", default=os.path.join(HERE, "tome_viz.html"))
    ap.add_argument("--label", default="")
    ap.add_argument("--random", type=int, default=0, help="also show this many random validation positions")
    args = ap.parse_args()
    pcores.pin()
    torch.set_num_threads(8)
    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    a = ck["args"]
    net = MT.AtTenTrisToMe(d=a["d"], layers=a["layers"], merge=[float(f) for f in a["merge"].split(",")], sky=a.get("tome_sky", False),
                          pretok=a.get("tome_pretok", False), flat_bias=a.get("tome_flat", False),
                          streams=a.get("tome_streams", 1))
    net.load_state_dict(ck["model"])
    net.eval()
    d = TR.load(args.data)
    if a.get("tome_pretok"):
        d["pretok"] = np.load(args.data.replace(".npz", ".pretok.npy"))
    s = open(args.src, encoding="utf-8").read()
    src = json.JSONDecoder().raw_decode(s[s.index('{"summary"'):])[0]["positions"]
    at = {(int(d["teacher"][i]), int(d["seed"][i]), int(d["n"][i])): i for i in range(len(d["chosen"]))}
    idx = [at[(["normal", "gate4"].index(q["teacher"]), q["seed"], q["n"])] for q in src]
    if args.random:
        rng = np.random.default_rng(7)
        extra = [int(i) for i in rng.permutation(len(d["chosen"])) if int(i) not in set(idx)][:args.random]
        for i in extra:
            toks = []
            for j, kd in enumerate(d["tok_kind"][i]):
                if kd >= 0 and T.KINDS[kd] in SLOTS:
                    pc = int(d["tok_piece"][i, j])
                    toks.append({"kind": T.KINDS[kd], "piece": T.PIECES[pc] if 0 <= pc < 7 else None})
            src.append({"why": "random sample", "teacher": ["normal", "gate4"][d["teacher"][i]], "seed": int(d["seed"][i]),
                        "n": int(d["n"][i]), "tokens": toks})
        idx += extra
    idx = np.array(idx)
    with torch.no_grad():
        out, info = net(TR.batch_of(d, idx, "cpu"), return_attn=True)
    prob = out.softmax(-1).numpy()
    cross = info["cross"].mean(1).numpy()                       # (P, NC, N_final), mean of heads
    merge_layers = [li for li, f in enumerate(net.merge) if f > 0]
    pos = []
    for k, (q, i) in enumerate(zip(src, idx)):
        layers = []
        for li in merge_layers:
            t = info["merges"][li]
            mk, al = t["mask"][k].numpy() > 0.5, t["alive"][k].numpy()
            layers.append({"layer": li + 1, "blobs": [np.nonzero(mk[j])[0].tolist() for j in np.nonzero(al)[0] if mk[j].any()]})
        last = info["merges"][-1]
        mk, al = last["mask"][k].numpy() > 0.5, last["alive"][k].numpy()
        final = []                                              # the tokens the candidates attend to, in tensor order
        for j in range(len(al)):
            if not al[j]:
                final.append(None)
            elif mk[j].any():
                final.append({"cells": np.nonzero(mk[j])[0].tolist()})
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
        pos.append({"why": q["why"], "teacher": q["teacher"], "seed": q["seed"], "n": q["n"],
                    "board": d["cells"][i].tolist(), "layers": layers, "final": final,
                    "pieces": {t_["kind"]: t_["piece"] for t_ in pk},
                    "chosen": cand(ch), "top": [cand(int(c)) for c in order[:8]],
                    "rank": int(np.where(order == ch)[0][0]) + 1, "n_cand": nc})
    log = ck["log"]
    label = args.label or "AtTenTris-ToMe · 200 cells + 7 pieces · adjacency-constrained merging (layers 2–4) · %d epochs" % len(log)
    last_val = log[-1]["val"] if log else {}
    data = json.dumps({"model": label, "val": {k: v["top1"] for k, v in last_val.items()}, "positions": pos}, separators=(",", ":"))
    tpl = open(os.path.join(HERE, "tome_viz_template.html"), encoding="utf-8").read()
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(tpl.replace("__DATA__", data))
    print("%d positions -> %s" % (len(pos), args.out))


if __name__ == "__main__":
    main()
