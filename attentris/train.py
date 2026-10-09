"""Train AtTenTris on attentris/data/{train,val}.npz (pack.py). CPU only.

Target per position: half one-hot on the move the bot played, half a softmax over the
search scores of the moves that survived the beam (temperature --tau, in attack units),
so near-equal alternatives are not punished as hard as blunders. Teacher scores are
targets only, never inputs.

    <python with torch> attentris/train.py [--epochs 8 --batch 256 --threads 16]

Checkpoint after every epoch (attentris/data/model.pt, plus the log); --resume continues.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M
import tokenizer as T

HERE = os.path.dirname(os.path.abspath(__file__))
assert len(T.KINDS) <= M.N_KINDS


def load(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def batch_of(d, idx):
    b = {
        "cells": torch.from_numpy(d["cells"][idx].astype(np.int64)),
        "tok_kind": torch.from_numpy(d["tok_kind"][idx].astype(np.int64)),
        "tok_mask": torch.from_numpy(M.unpack_masks(d["tok_mask"][idx])),
        "tok_box": torch.from_numpy(d["tok_box"][idx].astype(np.int64)),
        "tok_extra": torch.from_numpy(d["tok_extra"][idx].astype(np.float32)),
        "tok_piece": torch.from_numpy(d["tok_piece"][idx].astype(np.int64)),
        "cand_cells": torch.from_numpy(d["cand_cells"][idx].astype(np.int64)),
        "cand_meta": torch.from_numpy(d["cand_meta"][idx].astype(np.int64)).clamp(min=0),
        "teacher": torch.from_numpy(d["teacher"][idx].astype(np.int64)),
    }
    return b


def targets(d, idx, tau):
    chosen = torch.from_numpy(d["chosen"][idx].astype(np.int64))
    best = torch.from_numpy(d["cand_score"][idx][..., 1])                     # (B, NC), nan = pruned
    soft = torch.where(torch.isnan(best), torch.full_like(best, float("-inf")), best / tau).softmax(-1)
    onehot = F.one_hot(chosen, best.shape[1]).float()
    return chosen, 0.5 * onehot + 0.5 * soft.nan_to_num(0.0)


def evaluate(net, d, batch, tau):
    net.eval()
    n = len(d["chosen"])
    stats = {t: [0, 0, 0, 0.0] for t in (0, 1)}        # positions, top1, top3, loss
    with torch.no_grad():
        for s in range(0, n, batch):
            idx = np.arange(s, min(n, s + batch))
            logit = net(batch_of(d, idx))
            chosen, tgt = targets(d, idx, tau)
            loss = -(tgt * logit.log_softmax(-1).nan_to_num(0.0, neginf=0.0)).sum(-1)
            top = logit.topk(3, -1).indices
            t1 = top[:, 0] == chosen
            t3 = (top == chosen.unsqueeze(1)).any(-1)
            teach = d["teacher"][idx]
            for t in (0, 1):
                m = torch.from_numpy(teach == t)
                stats[t][0] += int(m.sum())
                stats[t][1] += int(t1[m].sum())
                stats[t][2] += int(t3[m].sum())
                stats[t][3] += float(loss[m].sum())
    net.train()
    out = {}
    for t, name in ((0, "normal"), (1, "gate4")):
        p, a1, a3, l = stats[t]
        out[name] = {"top1": a1 / max(1, p), "top3": a3 / max(1, p), "loss": l / max(1, p), "n": p}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--out", default=None, help="checkpoint (default DATA/model.pt)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N training positions (smoke test)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    out = args.out or os.path.join(args.data, "model.pt")

    tr = load(os.path.join(args.data, "train.npz"))
    va = load(os.path.join(args.data, "val.npz"))
    if args.limit:
        tr = {k: v[:args.limit] for k, v in tr.items()}
    n = len(tr["chosen"])
    net = M.AtTenTris(d=args.d, layers=args.layers)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * math.ceil(n / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    start, log = 0, []
    if args.resume and os.path.exists(out):
        ck = torch.load(out, weights_only=False)
        net.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start, log = ck["epoch"] + 1, ck["log"]
    print("params %d, train %d positions, val %d, %d epochs x %d batches" % (
        sum(p.numel() for p in net.parameters()), n, len(va["chosen"]), args.epochs, math.ceil(n / args.batch)), flush=True)
    rng = np.random.default_rng(0)
    for ep in range(start, args.epochs):
        t0 = time.time()
        perm = rng.permutation(n)
        tot, cnt = 0.0, 0
        nb = math.ceil(n / args.batch)
        for bi in range(nb):
            idx = np.sort(perm[bi * args.batch:(bi + 1) * args.batch])
            logit = net(batch_of(tr, idx))
            _, tgt = targets(tr, idx, args.tau)
            loss = -(tgt * logit.log_softmax(-1).nan_to_num(0.0, neginf=0.0)).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(idx); cnt += len(idx)
            if bi % 50 == 0:
                el = time.time() - t0
                print("  epoch %d batch %d/%d loss %.3f  (%.0fs, epoch ETA %.0fs)" % (
                    ep + 1, bi, nb, tot / cnt, el, el / (bi + 1) * (nb - bi - 1)), flush=True)
        ev = evaluate(net, va, 512, args.tau)
        rec = {"epoch": ep + 1, "train_loss": tot / cnt, "val": ev, "secs": time.time() - t0}
        log.append(rec)
        print("epoch %d: train loss %.3f | val normal top1 %.3f top3 %.3f | val gate4 top1 %.3f top3 %.3f | %.0fs" % (
            ep + 1, tot / cnt, ev["normal"]["top1"], ev["normal"]["top3"], ev["gate4"]["top1"], ev["gate4"]["top3"],
            time.time() - t0), flush=True)
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep,
                    "log": log, "args": vars(args)}, out)
        with open(out + ".log.json", "w") as f:
            json.dump(log, f, indent=1)


if __name__ == "__main__":
    main()
