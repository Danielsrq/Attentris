"""How specialised are AtTenTris's attention heads? Over validation positions, per head of the
cross-attention (moves -> tokens) and of each self-attention layer (token <-> token):

  entropy     the head's attention distribution, normalised by log(number of tokens)
              (0 = all on one token, 1 = uniform), averaged over queries
  confidence  the largest attention weight per query, averaged (Voita et al., 2019)
  importance  drop in top-1 agreement when the head's output is switched off (Michel et al., 2019)

and, for the cross-attention, each head's attention share by token kind and by where the token
sits relative to the move (overlapping, within 2 rows, farther). CPU only.

    python attentris/head_stats.py --model attentris/data/model_v51.pt --data attentris/data/v51/val.npz
"""
import argparse
import collections
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
GROUPS = [("sky", ("sky",)), ("well", ("well",)), ("play rows", ("play", "band")),
          ("slots fit_x", T.SLOT_KINDS), ("setups", T.SETUP_KINDS), ("T structures", T.STRUCTURE_KINDS),
          ("pits", ("pit1", "pit2", "pit3", "pit4", "rect12")), ("pieces", T.PIECE_KINDS)]


def kind_group(k):
    for name, ks in GROUPS:
        if k in ks:
            return name
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "data", "model_v51.pt"))
    ap.add_argument("--data", default=os.path.join(HERE, "data", "v51", "val.npz"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(HERE, "results_heads_v51.json"))
    args = ap.parse_args()
    pcores.pin()
    torch.set_num_threads(8)
    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    net = M.AtTenTris(d=ck["args"]["d"], layers=ck["args"]["layers"], n_kinds=ck["model"]["kind.weight"].shape[0],
                          cross_layers=ck["args"].get("cross_layers", 1), cross_loops=ck["args"].get("cross_loops", 1),
                          n_teachers=ck["model"]["style.weight"].shape[0], registers=ck["args"].get("registers", 0))
    net.load_state_dict(ck["model"])
    net.eval()
    d = TR.load(args.data)
    idx = np.sort(np.random.default_rng(0).choice(len(d["chosen"]), size=min(args.n, len(d["chosen"])), replace=False))
    H = net.cross.att.h
    L = len(net.blocks)

    def forward(part):
        b = TR.batch_of(d, part)
        with torch.no_grad():
            logit, att = net(b, return_attn=True)
        return b, logit, att

    def top1():
        hit = 0
        for s in range(0, len(idx), 500):
            part = idx[s:s + 500]
            _, logit, _ = forward(part)
            hit += int((logit.argmax(-1).numpy() == d["chosen"][part]).sum())
        return hit / len(idx)

    # --- entropy, confidence, kind / relation shares ---
    ent = collections.defaultdict(list)
    conf = collections.defaultdict(list)
    share_kind = np.zeros((H, len(GROUPS) + 1))
    share_rel = np.zeros((H, 3))
    nq = 0
    for s in range(0, len(idx), 500):
        part = idx[s:s + 500]
        b, logit, att = forward(part)
        tokpad = b["tok_kind"] < 0                                             # (B, NT)
        ntok = (~tokpad).sum(-1).float()                                       # (B,)
        candpad = b["cand_cells"][..., 0] == -1                                # (B, NC)
        lnorm = torch.log(ntok.clamp(min=2))
        # cross-attention (B, H, NC, NT)
        a = att["cross"].clamp(min=1e-12)
        e = -(a * a.log()).sum(-1) / lnorm[:, None, None]                      # (B, H, NC)
        real = (~candpad)[:, None, :].expand_as(e)
        for h in range(H):
            ent[("cross", h)].append(e[:, h][real[:, h]].numpy())
            conf[("cross", h)].append(att["cross"][:, h].amax(-1)[real[:, h]].numpy())
        # self-attention layers (B, H, NT, NT): queries = real tokens
        for li, sa in enumerate(att["self"]):
            a_ = sa.clamp(min=1e-12)
            e_ = -(a_ * a_.log()).sum(-1) / lnorm[:, None, None]
            rq = (~tokpad)[:, None, :].expand_as(e_)
            for h in range(H):
                ent[("self%d" % (li + 1), h)].append(e_[:, h][rq[:, h]].numpy())
                conf[("self%d" % (li + 1), h)].append(sa[:, h].amax(-1)[rq[:, h]].numpy())
        # cross-attention shares for the teacher's move, by kind and by relation to the move
        masks = M.unpack_masks(d["tok_mask"][part])                            # (B, NT, 200)
        ch = d["chosen"][part]
        for k in range(len(part)):
            i = part[k]
            mv = [c for c in d["cand_cells"][i, ch[k]] if c >= 0]
            mrow = min(c // 10 for c in mv) if mv else 0
            for j, kd in enumerate(d["tok_kind"][i]):
                if kd < 0:
                    continue
                g = kind_group(T.KINDS[kd])
                gi = [n for n, _ in GROUPS].index(g) if g != "other" else len(GROUPS)
                w = att["cross"][k, :, ch[k], j].numpy()                       # (H,)
                share_kind[:, gi] += w
                cells = np.nonzero(masks[k, j])[0]
                if len(cells):
                    if any(c in mv for c in cells):
                        share_rel[:, 0] += w
                    elif min(abs(c // 10 - mrow) for c in cells) <= 2:
                        share_rel[:, 1] += w
                    else:
                        share_rel[:, 2] += w
            nq += 1
    share_kind /= nq
    share_rel /= nq

    # --- importance: switch each head off ---
    base = top1()
    imp = {}
    for blk_name, att_mod in [("cross", net.cross.att)] + [("self%d" % (li + 1), blk.att) for li, blk in enumerate(net.blocks)]:
        for h in range(H):
            m = torch.ones(H)
            m[h] = 0
            att_mod.head_mask = m
            imp[(blk_name, h)] = base - top1()
            att_mod.head_mask = None
        print("  importance done:", blk_name, flush=True)

    rows = []
    for blk_name in ["cross"] + ["self%d" % (li + 1) for li in range(L)]:
        for h in range(H):
            rows.append({"block": blk_name, "head": h + 1, "entropy": float(np.concatenate(ent[(blk_name, h)]).mean()),
                         "confidence": float(np.concatenate(conf[(blk_name, h)]).mean()), "importance": float(imp[(blk_name, h)])})
    print("\nbase top-1 on %d validation positions: %.3f" % (len(idx), base))
    print("\n%-6s %4s  %7s  %10s  %10s" % ("block", "head", "entropy", "confidence", "importance"))
    for r in rows:
        print("%-6s %4d  %7.2f  %10.2f  %+9.1f pt" % (r["block"], r["head"], r["entropy"], r["confidence"], -100 * r["importance"]))
    names = [n for n, _ in GROUPS] + ["other"]
    print("\ncross-attention share of the teacher's move, by token kind (each head's row sums to 100%):")
    print("  head  " + "  ".join("%11s" % n for n in names))
    for h in range(H):
        print("  %4d  " % (h + 1) + "  ".join("%10.1f%%" % (100 * share_kind[h, g]) for g in range(len(names))))
    print("\ncross-attention share by where the token sits relative to the move:")
    print("  head    overlapping   within 2 rows   farther")
    for h in range(H):
        print("  %4d  %11.1f%%  %13.1f%%  %8.1f%%" % (h + 1, *(100 * share_rel[h])))
    with open(args.out, "w") as f:
        json.dump({"base_top1": base, "heads": rows, "kind_names": names, "share_kind": share_kind.tolist(),
                   "share_rel": share_rel.tolist()}, f, indent=1)
    print("\n-> %s" % args.out)


if __name__ == "__main__":
    main()
