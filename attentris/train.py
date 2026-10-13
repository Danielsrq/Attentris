"""Train AtTenTris on attentris/data/{train,val}.npz (pack.py). CPU only.

Target per position: half one-hot on the move the bot played, half a softmax over the
search scores of the moves that survived the beam (temperature --tau, in attack units),
so near-equal alternatives are not punished as hard as blunders. Teacher scores are
targets only, never inputs.

    <python with torch> attentris/train.py [--epochs 8 --batch 256 --threads 16]

Checkpoint after every epoch (attentris/data/model.pt, plus the log) and every --save-every
batches within an epoch (the epoch's batch order is saved too); --resume continues from the last
one, mid-epoch included.
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
import model_tome as MT
import pcores
import tokenizer as T

HERE = os.path.dirname(os.path.abspath(__file__))
assert len(T.KINDS) <= M.N_KINDS
TEACH = ("normal", "gate4", "danger")      # pack.TEACHERS


def load(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def batch_of(d, idx, device="cpu"):
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
    if "pretok" in d:
        b["pretok"] = torch.from_numpy(d["pretok"][idx].astype(np.int64))
    if device != "cpu":
        b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
    return b


def targets(d, idx, tau, device="cpu"):
    chosen = torch.from_numpy(d["chosen"][idx].astype(np.int64))
    best = torch.from_numpy(d["cand_score"][idx][..., 1])                     # (B, NC), nan = pruned
    soft = torch.where(torch.isnan(best), torch.full_like(best, float("-inf")), best / tau).softmax(-1)
    onehot = F.one_hot(chosen, best.shape[1]).float()
    return chosen.to(device), (0.5 * onehot + 0.5 * soft.nan_to_num(0.0)).to(device)


def compute_loss(out, d, idx, args, device):
    """Per-position loss (B,), the chosen indices, and the score targets.

      policy  cross-entropy to 0.5 one-hot(bot's move) + 0.5 softmax(search score / tau)
      dense   cross-entropy to softmax(1-ply score / tau_dense) over EVERY legal move: the
              graded "how bad" of each move the teacher rejected (weight w_dense)
      q1, qs  Huber regression of the value heads to the teacher's 1-ply score (every move)
              and 6-ply search score (moves the beam kept), in units of M.Q_SCALE (weight w_q)
    """
    logit = out["logit"]
    chosen, tgt = targets(d, idx, args.tau, device)
    logp = logit.log_softmax(-1).nan_to_num(0.0, neginf=0.0)
    loss = -(tgt * logp).sum(-1)
    sc = torch.from_numpy(d["cand_score"][idx]).to(device)
    ply1, best = sc[..., 0], sc[..., 1]
    real = torch.from_numpy(d["cand_cells"][idx][..., 0] != -1).to(device)
    if args.w_dense:
        dense = torch.where(real, ply1.nan_to_num(0.0) / args.tau_dense, torch.full_like(ply1, float("-inf"))).softmax(-1)
        loss = loss + args.w_dense * -(dense * logp).sum(-1)
    w_q1 = args.w_q if args.w_q1 is None else args.w_q1
    w_qs = args.w_q if args.w_qs is None else args.w_qs
    if w_q1 or w_qs:
        lo, hi = -30.0, 10.0      # clamp: -1000 death lines would otherwise dominate the regression
        t1 = (ply1.nan_to_num(0.0) / M.Q_SCALE).clamp(lo, hi)
        m1 = real.float()
        e1 = (F.smooth_l1_loss(out["q1"], t1, reduction="none") * m1).sum(-1) / m1.sum(-1).clamp(min=1)
        ms = (real & ~torch.isnan(best)).float()
        ts = (best.nan_to_num(0.0) / M.Q_SCALE).clamp(lo, hi)
        es = (F.smooth_l1_loss(out["qs"], ts, reduction="none") * ms).sum(-1) / ms.sum(-1).clamp(min=1)
        loss = loss + w_q1 * e1 + w_qs * es
    return loss, chosen, ply1, best, real


def evaluate(net, d, batch, args, device="cpu"):
    net.eval()
    n = len(d["chosen"])
    keys = ("n", "top1", "top3", "loss", "q1_top1", "qs_top1", "q1_err", "q1_cnt", "qs_err", "qs_cnt")
    stats = {t: dict.fromkeys(keys, 0.0) for t in range(len(TEACH))}
    with torch.no_grad():
        for s in range(0, n, batch):
            idx = np.arange(s, min(n, s + batch))
            out = net(batch_of(d, idx, device), heads=True)
            loss, chosen, ply1, best, real = compute_loss(out, d, idx, args, device)
            logit = out["logit"]
            q1 = out["q1"].masked_fill(~real, float("-inf"))
            qs = out["qs"].masked_fill(~real, float("-inf"))
            top = logit.topk(3, -1).indices
            t1 = top[:, 0] == chosen
            t3 = (top == chosen.unsqueeze(1)).any(-1)
            q1h, qsh = q1.argmax(-1) == chosen, qs.argmax(-1) == chosen
            e1 = ((out["q1"] * M.Q_SCALE - ply1).abs() * real).nan_to_num(0.0)
            bm = real & ~torch.isnan(best)
            es = ((out["qs"] * M.Q_SCALE - best).abs() * bm).nan_to_num(0.0)
            teach = d["teacher"][idx]
            for t in range(len(TEACH)):
                m = torch.from_numpy(teach == t).to(device)
                st = stats[t]
                st["n"] += int(m.sum()); st["top1"] += int(t1[m].sum()); st["top3"] += int(t3[m].sum())
                st["loss"] += float(loss[m].sum()); st["q1_top1"] += int(q1h[m].sum()); st["qs_top1"] += int(qsh[m].sum())
                st["q1_err"] += float(e1[m].sum()); st["q1_cnt"] += int(real[m].sum())
                st["qs_err"] += float(es[m].sum()); st["qs_cnt"] += int(bm[m].sum())
    net.train()
    res = {}
    for t, name in enumerate(TEACH):
        st = stats[t]
        p = max(1, st["n"])
        res[name] = {"top1": st["top1"] / p, "top3": st["top3"] / p, "loss": st["loss"] / p, "n": st["n"],
                     "q1_top1": st["q1_top1"] / p, "qs_top1": st["qs_top1"] / p,
                     "q1_mae": st["q1_err"] / max(1, st["q1_cnt"]), "qs_mae": st["qs_err"] / max(1, st["qs_cnt"])}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--threads", type=int, default=0, help="0 = one per P-core logical processor")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--out", default=None, help="checkpoint (default DATA/model.pt; DATA/model_tome.pt for --arch tome)")
    ap.add_argument("--cross-layers", type=int, default=1, help="attentris: independent cross-attention blocks (v5.2-untied: 2)")
    ap.add_argument("--cross-loops", type=int, default=1, help="attentris: apply the cross block this many times, same weights (v5.2-loop: 2)")
    ap.add_argument("--arch", default="attentris", choices=("attentris", "tome"), help="tome = AtTenTris-ToMe (model_tome.py)")
    ap.add_argument("--merge", default="0,0.45,0.45,0.45", help="tome: fraction of cell tokens merged after attention, per layer")
    ap.add_argument("--tome-sky", action="store_true", help="tome: the sky starts as one token per column (model_tome sky=True)")
    ap.add_argument("--tome-pretok", action="store_true", help="tome v3: one sky + solid blocks + single cells (needs <file>.pretok.npy, pretok.py)")
    ap.add_argument("--tome-flat", action="store_true", help="tome: FLAT-2D pair-feature bias in the self-attention")
    ap.add_argument("--tome-budget", default="", help="tome: per-position merge plan BASE,AT,SLOPE,FIRST (e.g. 20,36,0.25,0.10): "
                    "budget BASE + (n0 - AT) * SLOPE blobs per stream, first step FIRST, then equal ratios; --merge > 0 marks the merging layers")
    ap.add_argument("--tome-split", default="token", choices=("token", "hash", "greedy"),
                    help="A/B split after the first merge: token = alternate in token order (row-major, biased sideways; up to v4.1), "
                         "hash = a hashed bit per (board, step, stream, first cell), direction-blind; "
                         "greedy = no A/B sets, a greedy matching over all adjacent pairs")
    ap.add_argument("--tome-streams", type=int, default=1, help="tome v4: merging streams (partitions), stream s merges by head s's keys")
    ap.add_argument("--micro", type=int, default=1, help="split each batch into this many micro-batches (same gradient, less GPU memory)")
    ap.add_argument("--bucket", type=int, default=0, help="group batches by token count within chunks of this many batches (0 = off)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--save-every", type=int, default=300, help="mid-epoch checkpoint every N batches (0 = epoch ends only)")
    ap.add_argument("--seed", type=int, default=0, help="weight init and batch order")
    ap.add_argument("--drop-kinds", default="", help="ablation: token kinds to blank, by prefix, comma-separated (e.g. setup_)")
    ap.add_argument("--n-teachers", type=int, default=2, help="style rows (3: + danger); a smaller --init checkpoint grows, new rows copied from normal")
    ap.add_argument("--init", default="", help="warm start: load these model weights, fresh optimizer and schedule")
    ap.add_argument("--limit", type=int, default=0, help="use a random N-position subset (smoke tests, learning curves)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gpu-mem-frac", type=float, default=0.9, help="cap on the GPU memory this run may use (0 = no cap)")
    ap.add_argument("--train", default="train.npz",
                    help="training file(s) in DATA, comma-separated; FILE@F uses F of that file per epoch (a fresh random "
                         "draw each epoch; F > 1 repeats it), e.g. a 70/30 fine-tuning mix")
    ap.add_argument("--val2", default="", help="a second validation file in DATA (e.g. the students' own positions)")
    ap.add_argument("--w-dense", type=float, default=0.0, help="weight of the dense 1-ply ranking term (v2: 0.5)")
    ap.add_argument("--tau-dense", type=float, default=2.0)
    ap.add_argument("--w-q", type=float, default=0.0, help="weight of both value-head regressions (v2: 1.0)")
    ap.add_argument("--w-q1", type=float, default=None, help="weight of the Q1 regression (default --w-q)")
    ap.add_argument("--w-qs", type=float, default=None, help="weight of the Qs regression (default --w-q)")
    args = ap.parse_args()
    n_p = pcores.pin()
    torch.set_num_threads(args.threads or n_p or 8)
    if args.device == "cuda" and args.gpu_mem_frac > 0:
        # cap the caching allocator: on Windows (WDDM) the driver otherwise lets allocations past the
        # card's memory spill silently into system RAM (a 6-layer ToMe grew to 17 GB of RAM); with a cap
        # PyTorch frees its cache and retries first, and a real shortage is a clear OOM error
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_frac)
    torch.manual_seed(args.seed)
    out = args.out or os.path.join(args.data, "model_tome.pt" if args.arch == "tome" else "model.pt")
    if os.path.exists(out) and not args.resume:
        prev = torch.load(out, map_location="cpu", weights_only=False).get("args", {}).get("arch", "attentris")
        if prev != args.arch:
            raise SystemExit("%s holds a %s checkpoint; refusing to overwrite it with %s (pick another --out)" % (out, prev, args.arch))

    specs = [(f.split("@")[0], float(f.split("@")[1]) if "@" in f else 1.0) for f in args.train.split(",")]
    parts = [load(os.path.join(args.data, f)) for f, _ in specs]
    if args.tome_pretok:
        for p, (f, _) in zip(parts, specs):
            p["pretok"] = np.load(os.path.join(args.data, f.replace(".npz", ".pretok.npy")))
    sizes = [len(p["chosen"]) for p in parts]
    tr = {}
    for k in list(parts[0]):          # one array at a time, each file's copy freed as it is merged (peak ~1x the data, not 2x)
        tr[k] = np.concatenate([p.pop(k) for p in parts])
    del parts
    va = load(os.path.join(args.data, "val.npz"))
    va2 = load(os.path.join(args.data, args.val2)) if args.val2 else None
    if va2 is not None and args.tome_pretok:
        va2["pretok"] = np.load(os.path.join(args.data, args.val2.replace(".npz", ".pretok.npy")))
    if args.tome_pretok:
        va["pretok"] = np.load(os.path.join(args.data, "val.pretok.npy"))
    if args.drop_kinds:      # ablation: blank these token kinds everywhere (training and validation)
        ids = [i for i, k in enumerate(T.KINDS) if any(k.startswith(p) for p in args.drop_kinds.split(","))]
        n_tr = int(np.isin(tr["tok_kind"], ids).sum())
        for dd in [tr, va] + ([va2] if va2 is not None else []):
            m = np.isin(dd["tok_kind"], ids)
            dd["tok_kind"][m] = -1
            dd["tok_mask"][m] = 0
            dd["tok_box"][m] = -1
            dd["tok_extra"][m] = 0
            dd["tok_piece"][m] = -1
        print("dropped token kinds %s: %d tokens in training (%.3f per position)" % (
            ",".join(T.KINDS[i] for i in ids), n_tr, n_tr / len(tr["chosen"])), flush=True)
    if args.limit:
        keep = np.sort(np.random.default_rng(123).choice(len(tr["chosen"]), size=min(args.limit, len(tr["chosen"])), replace=False))
        tr = {k: v[keep] for k, v in tr.items()}   # a random subset (files are in teacher order)
    n = len(tr["chosen"])
    mixed = any(fr != 1.0 for _, fr in specs)
    if mixed and args.limit:
        raise SystemExit("--limit and FILE@F do not mix")
    starts = np.cumsum([0] + sizes[:-1])
    takes = [int(round(fr * sz)) for (_, fr), sz in zip(specs, sizes)]
    m = sum(takes) if mixed else n                            # positions per epoch

    def draw(rng):
        """The epoch's positions in random order: every position once, or with FILE@F a fresh draw of
        F of each file (whole repeats, then a random part without replacement)."""
        if not mixed:
            return rng.permutation(n)
        idx = []
        for s0, sz, k in zip(starts, sizes, takes):
            idx += [s0 + np.arange(sz)] * (k // sz) + [s0 + rng.choice(sz, size=k % sz, replace=False)]
        return rng.permutation(np.concatenate(idx))
    if args.arch == "tome":
        net = MT.AtTenTrisToMe(d=args.d, layers=args.layers, merge=[float(f) for f in args.merge.split(",")],
                               sky=args.tome_sky, pretok=args.tome_pretok, flat_bias=args.tome_flat,
                               streams=args.tome_streams, n_teachers=args.n_teachers, budget=MT.budget_of(args.tome_budget), split=args.tome_split).to(args.device)
    else:
        net = M.AtTenTris(d=args.d, layers=args.layers, cross_layers=args.cross_layers, cross_loops=args.cross_loops,
                          n_teachers=args.n_teachers).to(args.device)
    print("device", args.device, torch.cuda.get_device_name(0) if args.device == "cuda" else "", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * math.ceil(m / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    start, log = 0, []
    start_batch, perm0, tot0, cnt0 = 0, None, 0.0, 0
    if args.init:
        ck0 = torch.load(args.init, map_location=args.device, weights_only=False)
        sw = ck0["model"].get("style.weight")
        if sw is not None and sw.shape[0] < args.n_teachers:   # a new style: starts as a copy of normal's
            ck0["model"]["style.weight"] = torch.cat([sw, sw[:1].repeat(args.n_teachers - sw.shape[0], 1)])
            print("style table grown %d -> %d rows (new rows = normal)" % (sw.shape[0], args.n_teachers), flush=True)
        net.load_state_dict(ck0["model"])
        print("warm start from %s (%d epochs)" % (args.init, len(ck0.get("log", []))), flush=True)
    if args.resume and os.path.exists(out):
        ck = torch.load(out, map_location=args.device, weights_only=False)
        net.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        net.to(args.device)
        start, log = ck["epoch"] + 1, ck["log"]
        if ck.get("batch") is not None:                       # a mid-epoch checkpoint
            start, start_batch, perm0, tot0, cnt0 = ck["epoch"], ck["batch"] + 1, ck["perm"], ck["tot"], ck["cnt"]
        print("resuming at epoch %d, batch %d" % (start + 1, start_batch), flush=True)
    print("params %d, train %d positions (%d per epoch: %s), val %d%s, %d epochs x %d batches" % (
        sum(p.numel() for p in net.parameters()), n, m, ", ".join("%s %d" % (f, k) for (f, _), k in zip(specs, takes)),
        len(va["chosen"]), " + val2 %d" % len(va2["chosen"]) if va2 is not None else "", args.epochs,
        math.ceil(m / args.batch)), flush=True)
    rng = np.random.default_rng(args.seed)
    for _ in range(start):
        draw(rng)                                             # the same batch order as an uninterrupted run
    length = None
    if args.bucket and "pretok" in tr:
        length = (tr["pretok"] == -1).sum(1) + tr["pretok"].max(1).astype(np.int64) + 1     # single cells + sky + blocks
    elif args.bucket:
        # tokens a position starts with (stack cells + sky columns): batches of similar length pad less
        g = tr["cells"].reshape(-1, 20, 10).astype(bool)
        h = np.where(g, np.arange(1, 21)[None, :, None], 0).max(1)
        length = (np.arange(20)[None, :, None] < h[:, None, :]).sum((1, 2)) + (h < 20).sum(1)

    def order_of(perm):
        """The epoch's position order: random, or (with --bucket) sorted by length inside chunks of
        `bucket` batches, the batches then shuffled, so every batch is a random draw of similar size."""
        if length is None:
            return perm
        span = args.bucket * args.batch
        out = []
        for s0 in range(0, len(perm), span):
            ch = perm[s0:s0 + span]
            out.append(ch[np.argsort(length[ch], kind="stable")])
        o = np.concatenate(out)
        nb_ = math.ceil(len(perm) / args.batch)
        bo = np.random.default_rng(int(perm[0])).permutation(nb_)
        return np.concatenate([o[b * args.batch:(b + 1) * args.batch] for b in bo])

    def save(path, ep, extra):
        torch.save(dict({"model": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep,
                         "log": log, "args": vars(args), "tok_version": T.TOK_VERSION}, **extra), path + ".tmp")
        os.replace(path + ".tmp", path)                       # never a half-written checkpoint

    for ep in range(start, args.epochs):
        t0 = time.time()
        perm = order_of(draw(rng))
        tot, cnt = 0.0, 0
        first = 0
        if perm0 is not None and ep == start:
            perm, first, tot, cnt = perm0, start_batch, tot0, cnt0
        nb = math.ceil(len(perm) / args.batch)
        for bi in range(first, nb):
            idx = np.sort(perm[bi * args.batch:(bi + 1) * args.batch])
            opt.zero_grad()
            loss = 0.0
            for part in np.array_split(idx, args.micro):        # same batch, less memory: gradients add up
                o = net(batch_of(tr, part, args.device), heads=True)
                lp = compute_loss(o, tr, part, args, args.device)[0].sum() / len(idx)
                lp.backward()
                loss = loss + lp.detach()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(idx); cnt += len(idx)
            if bi % 50 == 0:
                el = time.time() - t0
                print("  epoch %d batch %d/%d loss %.3f  (%.0fs, epoch ETA %.0fs)" % (
                    ep + 1, bi, nb, tot / cnt, el, el / (bi - first + 1) * (nb - bi - 1)), flush=True)
            if args.save_every and bi % args.save_every == args.save_every - 1 and bi < nb - 1:
                save(out, ep, {"batch": bi, "perm": perm, "tot": tot, "cnt": cnt})
        eb = max(64, 512 // args.micro)       # validation in micro-batches too (sums per position: same numbers)
        ev = evaluate(net, va, eb, args, args.device)
        rec = {"epoch": ep + 1, "train_loss": tot / cnt, "val": ev, "secs": time.time() - t0}
        if va2 is not None:
            rec["val2"] = evaluate(net, va2, eb, args, args.device)
        log.append(rec)
        print("epoch %d: train loss %.3f | val normal top1 %.3f top3 %.3f | val gate4 top1 %.3f top3 %.3f | "
              "argmax Qs top1 %.3f / %.3f, Q1 top1 %.3f / %.3f | MAE Q1 %.2f Qs %.2f | %.0fs" % (
            ep + 1, tot / cnt, ev["normal"]["top1"], ev["normal"]["top3"], ev["gate4"]["top1"], ev["gate4"]["top3"],
            ev["normal"]["qs_top1"], ev["gate4"]["qs_top1"], ev["normal"]["q1_top1"], ev["gate4"]["q1_top1"],
            (ev["normal"]["q1_mae"] + ev["gate4"]["q1_mae"]) / 2, (ev["normal"]["qs_mae"] + ev["gate4"]["qs_mae"]) / 2,
            time.time() - t0), flush=True)
        if va2 is not None:
            v2 = max(rec["val2"].values(), key=lambda v: v["n"])     # the teacher the val2 set is labelled by
            print("epoch %d: val2 (%s, %d) top1 %.3f top3 %.3f loss %.3f | argmax Qs top1 %.3f | MAE Qs %.2f" % (
                ep + 1, args.val2, v2["n"], v2["top1"], v2["top3"], v2["loss"], v2["qs_top1"], v2["qs_mae"]), flush=True)
        save(out, ep, {"batch": None})
        with open(out + ".log.json", "w") as f:
            json.dump(log, f, indent=1)


if __name__ == "__main__":
    main()
