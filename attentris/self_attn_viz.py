"""Self-attention viewer for AtTenTris (v5.x): pick any token (a piece token -- current, hold, a preview -- or a
board token) and see, per layer and head, where it looks among the tokens, drawn on the board through the token
masks; or the other way round, who looks at it. The positions (and their token lists) are an attention page's
(attn_viz.py), plus --random positions where a queued piece has a fit / setup token of its own piece.
The page also shows, over the whole validation set, the "own-piece lift": for each piece token, the attention per
fit_X token of its own piece / per fit_X token of another piece (1 = no preference). CPU only.

    $CPUPY attentris/self_attn_viz.py --model attentris/data/model_v51_d128_dagger5b_ft.pt \
        --src attentris/attn_v51_d128_dagger5b.html --out attentris/self_attn_v51_d128_dagger5b.html
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
QUERIES = ["current", "hold", "preview1", "preview2", "preview3", "preview4", "preview5"]


def load_net(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    net = M.AtTenTris(d=a["d"], layers=a["layers"], n_kinds=ck["model"]["kind.weight"].shape[0],
                      cross_layers=a.get("cross_layers", 1), cross_loops=a.get("cross_loops", 1),
                      n_teachers=ck["model"]["style.weight"].shape[0], registers=a.get("registers", 0),
                      flat=not a.get("no_flat", False))
    net.load_state_dict(ck["model"])
    net.eval()
    return net, ck


def self_attn(net, d, idx):
    """(B, L, H, NT, NT) self-attention over the packed token slots (registers, if any, dropped as keys)."""
    with torch.no_grad():
        _, info = net(TR.batch_of(d, np.asarray(idx), "cpu"), return_attn=True)
    nt = d["tok_kind"].shape[1]
    return torch.stack(info["self"], 1)[..., :nt, :nt].numpy()


def own_piece_lift(net, d, fam="fit"):
    """Whole set: per query slot, layer, head -- sum of the mean attention to own-piece fam_X tokens and to
    other-piece ones, over positions with both (lift = ratio of the sums)."""
    tk = {T.KINDS.index(fam + "_" + p.lower()): p for p in T.PIECES if fam + "_" + p.lower() in T.KINDS}
    qk = {T.KINDS.index(q): q for q in QUERIES}
    pid = {p: i for i, p in enumerate(T.PIECES)}
    same = other = None
    npos = np.zeros(len(QUERIES), int)
    for s in range(0, len(d["chosen"]), 500):
        idx = np.arange(s, min(len(d["chosen"]), s + 500))
        A = self_attn(net, d, idx)
        if same is None:
            same, other = np.zeros((len(QUERIES),) + A.shape[1:3]), np.zeros((len(QUERIES),) + A.shape[1:3])
        for b, i in enumerate(idx):
            kinds, pieces = d["tok_kind"][i], d["tok_piece"][i]
            tp = {j: pid[tk[int(k)]] for j, k in enumerate(kinds) if int(k) in tk}
            if not tp:
                continue
            for j, k in enumerate(kinds):
                if int(k) not in qk or pieces[j] < 0:
                    continue
                sj = [t for t, p in tp.items() if p == int(pieces[j])]
                oj = [t for t, p in tp.items() if p != int(pieces[j])]
                if sj and oj:
                    q = QUERIES.index(qk[int(k)])
                    same[q] += A[b, :, :, j][:, :, sj].mean(-1)
                    other[q] += A[b, :, :, j][:, :, oj].mean(-1)
                    npos[q] += 1
    return {"queries": QUERIES, "n": npos.tolist(),
            "head": (same / np.maximum(other, 1e-12)).round(2).tolist(),                       # (Q, L, H)
            "mean": (same.sum(-1) / np.maximum(other.sum(-1), 1e-12)).round(2).tolist()}       # (Q, L)


def note(d, i, j, kd):
    k, e = T.KINDS[kd], d["tok_extra"][i, j]
    if k in T.SLOT_KINDS and len(T.KINDS) > 27:
        lines = int(round(float(e[8]) * 3))
        return ("covered" if e[7] > 0.5 else "open") + (" · %d line%s" % (lines, "s" * (lines > 1)) if lines else "")
    if k in T.SETUP_KINDS:
        return "slot for %d lines" % int(round(float(e[0]) * 3))
    return None


def position(d, i, A, why):
    kinds = d["tok_kind"][i]
    real = [j for j, kd in enumerate(kinds) if kd >= 0]
    masks = M.unpack_masks(d["tok_mask"][i])
    toks = [{"kind": T.KINDS[kinds[j]], "note": note(d, i, j, kinds[j]), "cells": np.nonzero(masks[j])[0].tolist(),
             "piece": T.PIECES[d["tok_piece"][i, j]] if 0 <= d["tok_piece"][i, j] < 7 else None} for j in real]
    a = A[:, :, real][:, :, :, real]                                   # (L, H, n, n)
    ch = int(d["chosen"][i])
    m = d["cand_meta"][i, ch]
    return {"why": why, "teacher": ["normal", "gate4"][d["teacher"][i]], "seed": int(d["seed"][i]), "n": int(d["n"][i]),
            "board": d["cells"][i].tolist(), "tokens": toks,
            "move": {"cells": [int(x) for x in d["cand_cells"][i, ch] if x >= 0], "piece": T.PIECES[m[0]], "hold": int(m[1]),
                     "spin": int(m[2]), "lines": int(m[3]), "attack": int(m[4])},
            # attention in 1/10000 (rows: query, columns: key), per layer and head
            "A": np.rint(a * 10000).astype(int).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "data", "model_v51_d128_dagger5b_ft.pt"))
    ap.add_argument("--data", default=os.path.join(HERE, "data", "v51", "val.npz"))
    ap.add_argument("--src", default=os.path.join(HERE, "attn_v51_d128_dagger5b.html"), help="take the positions from this attention page")
    ap.add_argument("--random", type=int, default=30, help="add this many positions where a queued piece has a fit / setup token of its own piece")
    ap.add_argument("--out", default=os.path.join(HERE, "self_attn_v51_d128_dagger5b.html"))
    ap.add_argument("--label", default="v5.1 d128 dagger5b-ft")
    ap.add_argument("--title", default="AtTenTris Self-Attention")
    args = ap.parse_args()
    n_p = pcores.pin()
    torch.set_num_threads(int(os.environ.get("THREADS", n_p or 8)))
    net, ck = load_net(args.model)
    d = TR.load(args.data)
    lift = own_piece_lift(net, d)
    print("own-piece lift (fit_X), layer means:", {q: lift["mean"][k] for k, q in enumerate(QUERIES)}, flush=True)
    s = open(args.src, encoding="utf-8").read()
    src = json.JSONDecoder().raw_decode(s[s.index('{"summary"'):])[0]["positions"]
    at = {(int(d["teacher"][i]), int(d["seed"][i]), int(d["n"][i])): i for i in range(len(d["chosen"]))}
    show = [(at[(["normal", "gate4"].index(q["teacher"]), q["seed"], q["n"])], q["why"]) for q in src]
    for (i, _), q in zip(show, src):                                   # the page's tokens are this file's
        assert [t["kind"] for t in q["tokens"]] == [T.KINDS[k] for k in d["tok_kind"][i] if k >= 0], "token lists differ"
    taken = {i for i, _ in show}
    rng = np.random.default_rng(11)
    fam = {T.KINDS.index(f + "_" + p.lower()): p for f in ("fit", "setup") for p in T.PIECES if f + "_" + p.lower() in T.KINDS}
    qids = {T.KINDS.index(q) for q in QUERIES if q != "current"}
    extra = []
    for i in rng.permutation(len(d["chosen"])):
        if len(extra) >= args.random:
            break
        i = int(i)
        if i in taken:
            continue
        kinds = d["tok_kind"][i]
        queued = {T.PIECES[d["tok_piece"][i, j]] for j, k in enumerate(kinds) if int(k) in qids and 0 <= d["tok_piece"][i, j] < 7}
        mine = sorted({fam[int(k)] for k in kinds if int(k) in fam} & queued)
        if mine:
            extra.append((i, "a queued piece has its own slot (%s)" % ", ".join(mine)))
    show += extra
    pos = []
    for k in range(0, len(show), 64):
        part = show[k:k + 64]
        A = self_attn(net, d, [i for i, _ in part])
        pos += [position(d, i, A[b], why) for b, (i, why) in enumerate(part)]
    data = json.dumps({"model": args.label, "layers": int(ck["args"]["layers"]), "heads": len(pos[0]["A"][0]),
                       "registers": int(ck["args"].get("registers", 0)), "val_n": int(len(d["chosen"])), "lift": lift,
                       "positions": pos}, separators=(",", ":"))
    tpl = open(os.path.join(HERE, "self_attn_template.html"), encoding="utf-8").read()
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(tpl.replace("__TITLE__", args.title).replace("__DATA__", data))
    print("%d positions (%d from %s, %d added) -> %s" % (len(pos), len(src), os.path.basename(args.src), len(extra), args.out))


if __name__ == "__main__":
    main()
