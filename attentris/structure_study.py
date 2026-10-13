"""Does incidental structure recovery make a ToMe model choose better? (structure_study.sh)

Each ToMe model is scored on its OWN game positions, labelled by the teacher (CEM-depth6: every move's
depth-6 search score). Per rule token (tsd, fit_X open / covered, pit1-2, setup_X) of each position:
  coverage = the largest share of the token's cells inside one final blob (either stream), and
  leak = that blob's size / the token's size. Recovery levels (generous):
    clean       IoU >= 0.8
    contained   coverage >= 0.75 and leak <= 4 (a TSD running into its well / the air above counts; a
                sky-sized blob does not)
    partial     coverage 0.5 - 0.75 (or contained but leak > 4)
    fragmented  coverage < 0.5
Decisions, on the positions where the teacher's move uses the token (>= 3 of its cells inside it):
  the share where the model's move also uses it, the model's top-1 with the teacher, and its regret
  (the teacher's best score minus the score of the model's move; scored moves only). The control (a
  rule-token model, e.g. v5.1 dagger-ft) is scored on the same positions: a level that is merely easier
  shows up there too. Also the recovery rates per kind over all positions.

    attentris/.venv/Scripts/python attentris/structure_study.py --tome NAME=CKPT=PACKED.npz ... --control CKPT
"""
import argparse
import collections
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M
import model_tome as MT
import pcores
import tokenizer as T
import train as TR
from tome_recovery_play import load_control, load_tome

LEVELS = ("clean", "contained", "partial", "fragmented")


def level(cov, leak, iou):
    if iou >= 0.8:
        return "clean"
    if cov >= 0.75 and leak <= 4:
        return "contained"
    if cov >= 0.5:
        return "partial"
    return "fragmented"


def kind_of(d, i, j, kd):
    kind = T.KINDS[kd]
    if kind in T.SLOT_KINDS:
        kind = kind + (" covered" if d["tok_extra"][i, j, 7] > 0.5 else " open")
    return kind


def run(net, d, device, batch=128, tome=True):
    """picks (N,), and per position the final blobs (list of bool (200,) arrays) if tome."""
    picks, blobs = [], []
    with torch.no_grad():
        for s in range(0, len(d["chosen"]), batch):
            idx = np.arange(s, min(len(d["chosen"]), s + batch))
            b = TR.batch_of(d, idx, device)
            if tome:
                logit, info = net(b, return_attn=True)
                last = info["merges"][-1]
                mk = last["mask"].cpu().numpy() > 0.5
                al = last["alive"].cpu().numpy()
                for k in range(len(idx)):
                    blobs.append(mk[k][al[k] & mk[k].any(-1)])
            else:
                logit = net(b)
            picks.append(logit.argmax(-1).cpu().numpy())
    return np.concatenate(picks), blobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tome", action="append", required=True, help="name=checkpoint=packed.npz (its own positions; repeatable)")
    ap.add_argument("--control", required=True)
    ap.add_argument("--device", default="cpu", help="cpu by default: never share the GPU with a training run")
    ap.add_argument("--min-n", type=int, default=20)
    args = ap.parse_args()
    pcores.pin()
    kinds_show = ["tsd"] + ["fit_%s %s" % (p, c) for p in "szltji" for c in ("open", "covered")] + ["pit1", "pit2"] + \
                 ["setup_%s" % p for p in "tszlji"]
    for spec in args.tome:
        name, ck, data = spec.split("=", 2)
        d = TR.load(data)
        d["pretok"] = np.load(data.replace(".npz", ".pretok.npy"))
        net = load_tome(ck).to(args.device)
        pick, blobs = run(net, d, args.device)
        cnet = load_control(args.control).to(args.device)
        cpick, _ = run(cnet, d, args.device, tome=False)
        best = d["cand_score"][..., 1].astype(np.float64)
        top = np.nanmax(best, axis=1)
        N = len(pick)
        reg = top - best[np.arange(N), pick]
        creg = top - best[np.arange(N), cpick]
        hit, chit = pick == d["chosen"], cpick == d["chosen"]
        rate = collections.defaultdict(collections.Counter)                  # kind -> level counts (all tokens)
        dec = collections.defaultdict(lambda: collections.defaultdict(list))  # (kind, level) -> metric -> values
        for i in range(N):
            masks = M.unpack_masks(d["tok_mask"][i])
            cells = lambda c: set(int(x) for x in d["cand_cells"][i, c] if x >= 0)
            ch, mp, cp = cells(d["chosen"][i]), cells(pick[i]), cells(cpick[i])
            B = blobs[i].astype(np.int32)
            for j, kd in enumerate(d["tok_kind"][i]):
                if kd < 0 or not masks[j].any() or T.KINDS[kd] in T.PIECE_KINDS:
                    continue
                kind = kind_of(d, i, j, kd)
                if kind not in kinds_show:
                    continue
                mm = masks[j].reshape(200)
                tset = set(np.nonzero(mm)[0].tolist())
                inter = B @ mm.astype(np.int32)
                if len(inter) == 0:
                    continue
                b = int(inter.argmax())
                cov = inter[b] / len(tset)
                iou = (inter / (B.sum(1) + len(tset) - inter)).max()
                lv = level(cov, B[b].sum() / len(tset), iou)
                rate[kind][lv] += 1
                if len(ch & tset) >= 3:                                         # the teacher's move uses it
                    r = dec[(kind, lv)]
                    r["model uses"].append(len(mp & tset) >= 3); r["ctl uses"].append(len(cp & tset) >= 3)
                    r["model top1"].append(hit[i]); r["ctl top1"].append(chit[i])
                    r["model regret"].append(reg[i]); r["ctl regret"].append(creg[i])
        print("\n######## %s on its own %d positions (teacher CEM-depth6): top-1 %.3f, regret %.2f; control %s top-1 %.3f, regret %.2f" % (
            name, N, hit.mean(), np.nanmean(reg), os.path.basename(args.control), chit.mean(), np.nanmean(creg)))
        print("   recovery of each kind, all tokens:       " + "  ".join("%-10s" % l for l in LEVELS) + "   n")
        for kind in kinds_show:
            c = rate[kind]
            n = sum(c.values())
            if n:
                print("   %-22s                   " % kind + "  ".join("%5.1f%%    " % (100 * c[l] / n) for l in LEVELS) + " %6d" % n)
        print("   decisions where the teacher's move uses the structure:")
        print("   kind                   recovery        n   model uses  ctl uses   model top-1  ctl top-1   model regret  ctl regret")
        for kind in kinds_show:
            for lv in LEVELS:
                r = dec.get((kind, lv))
                if not r or len(r["model top1"]) < args.min_n:
                    continue
                m = {k: np.nanmean(np.array(v, dtype=float)) for k, v in r.items()}
                print("   %-22s %-11s %6d      %.3f      %.3f       %.3f      %.3f        %5.2f       %5.2f" % (
                    kind, lv, len(r["model top1"]), m["model uses"], m["ctl uses"], m["model top1"], m["ctl top1"],
                    m["model regret"], m["ctl regret"]))


if __name__ == "__main__":
    main()
