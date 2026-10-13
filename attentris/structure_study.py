"""Does structure recovery make a ToMe model choose better? (structure_study.sh)

Each ToMe model is scored on its OWN game positions, labelled by the teacher (CEM-depth6: every move's
depth-6 search score). Per rule token (tsd, fit_X open / covered, pit1-4, setup_X, T structures):
  coverage = the largest share of the token's cells inside one final blob (either stream), and
  leak = that blob's size / the token's size. Recovery levels (generous):
    clean       IoU >= 0.8
    contained   coverage >= 0.75 and leak <= 4 (a TSD running into its well / the air above counts; a
                sky-sized blob does not)
    partial     coverage 0.5 - 0.75 (or contained but leak > 4)
    fragmented  coverage < 0.5
"recovered" = clean or contained.

1. Correlational, per kind and level, split by whether the teacher's move uses the token (>= 3 of its
   cells inside it): the share where the model's move uses it, top-1 with the teacher, and regret (the
   teacher's best score minus the score of the model's move), with a rule-token control model (v5.1)
   on the same positions: "excess" = model regret - control regret, so a level that is merely easier
   cancels out.
2. Causal (--force KINDS, for the --force-models): the same positions decided again with that kind
   forced into blobs (model_tome.MERGE_GUIDE: blobs of one token get a similarity bonus, so they merge
   first; everything else is the model's own merging). Natural vs forced recovery, uses, top-1 and regret,
   overall and on the treated positions (not recovered naturally, recovered when forced).

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
KINDS = ["tsd", "stsd", "fractal", "imperial", "cspin"] + ["fit_%s %s" % (p, c) for p in "tszlji" for c in ("open", "covered")] + \
        ["pit1", "pit2", "pit3", "pit4"] + ["setup_%s" % p for p in "tszlji"]


def level(cov, leak, iou):
    if iou >= 0.8:
        return "clean"
    if cov >= 0.75 and leak <= 4:
        return "contained"
    if cov >= 0.5:
        return "partial"
    return "fragmented"


def tokens_of(d, i):
    """[(kind, mask (200,) bool)] of position i's rule tokens of the studied kinds."""
    masks = M.unpack_masks(d["tok_mask"][i])
    out = []
    for j, kd in enumerate(d["tok_kind"][i]):
        if kd < 0 or not masks[j].any():
            continue
        kind = T.KINDS[kd]
        if kind in T.SLOT_KINDS:
            kind = kind + (" covered" if d["tok_extra"][i, j, 7] > 0.5 else " open")
        if kind in KINDS:
            out.append((kind, masks[j].reshape(200)))
    return out


def run(net, d, device, toks=None, force=None, batch=128, tome=True):
    """picks (N,) and, for ToMe, the final blobs per position; force = a kind to force into blobs."""
    picks, blobs = [], []
    with torch.no_grad():
        for s in range(0, len(d["chosen"]), batch):
            idx = np.arange(s, min(len(d["chosen"]), s + batch))
            if force is not None:
                g = np.full((len(idx), 200), -1, np.int64)
                for k, i in enumerate(idx):
                    for t, (kind, mm) in enumerate(toks[i]):
                        if kind == force:
                            g[k][mm & (g[k] < 0)] = t
                MT.MERGE_GUIDE = torch.from_numpy(g)
            try:
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
            finally:
                MT.MERGE_GUIDE = None
            picks.append(logit.argmax(-1).cpu().numpy())
    return np.concatenate(picks), blobs


def recovery(B, mm):
    tset = mm.sum()
    inter = B.astype(np.int32) @ mm.astype(np.int32)
    if len(inter) == 0:
        return "fragmented"
    b = int(inter.argmax())
    iou = (inter / (B.sum(1) + tset - inter)).max()
    return level(inter[b] / tset, B[b].sum() / tset, iou)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tome", action="append", required=True, help="name=checkpoint=packed.npz (its own positions; repeatable)")
    ap.add_argument("--control", required=True)
    ap.add_argument("--force", default="tsd,fit_t covered,fit_t open,fit_s open,fit_z open,fit_l open,fit_j open,fit_i open,pit1,pit2,setup_t",
                    help="kinds for the forced-recovery test ('' = none)")
    ap.add_argument("--force-models", default="tome_v42_l4g_dagger_ft,tome_v42_l6g_dagger_ft")
    ap.add_argument("--device", default="cpu", help="cpu by default: never share the GPU with a training run")
    ap.add_argument("--min-n", type=int, default=20)
    args = ap.parse_args()
    pcores.pin()
    for spec in args.tome:
        name, ck, data = spec.split("=", 2)
        d = TR.load(data)
        d["pretok"] = np.load(data.replace(".npz", ".pretok.npy"))
        N = len(d["chosen"])
        toks = [tokens_of(d, i) for i in range(N)]
        net = load_tome(ck).to(args.device)
        pick, blobs = run(net, d, args.device)
        cpick, _ = run(load_control(args.control).to(args.device), d, args.device, tome=False)
        best = d["cand_score"][..., 1].astype(np.float64)
        top = np.nanmax(best, axis=1)
        regret = lambda p: top - best[np.arange(N), p]
        reg, creg = regret(pick), regret(cpick)
        cells = lambda i, c: set(int(x) for x in d["cand_cells"][i, c] if x >= 0)
        uses = lambda i, c, mm: len(cells(i, c) & set(np.nonzero(mm)[0].tolist())) >= 3
        print("\n######## %s on its own %d positions (teacher CEM-depth6): top-1 %.3f, regret %.2f; control %s: top-1 %.3f, regret %.2f" % (
            name, N, (pick == d["chosen"]).mean(), np.nanmean(reg), os.path.basename(args.control), (cpick == d["chosen"]).mean(), np.nanmean(creg)))
        # 1. correlational
        rate = collections.defaultdict(collections.Counter)
        dec = collections.defaultdict(lambda: collections.defaultdict(list))
        for i in range(N):
            for kind, mm in toks[i]:
                lv = recovery(blobs[i], mm)
                rate[kind][lv] += 1
                tu = "teacher uses" if uses(i, d["chosen"][i], mm) else "teacher doesn't"
                r = dec[(kind, tu, lv)]
                r["uses"].append(uses(i, pick[i], mm)); r["ctl uses"].append(uses(i, cpick[i], mm))
                r["top1"].append(pick[i] == d["chosen"][i]); r["ctl top1"].append(cpick[i] == d["chosen"][i])
                r["regret"].append(reg[i]); r["ctl regret"].append(creg[i])
        print("   recovery of each kind, all tokens:      " + "  ".join("%-10s" % l for l in LEVELS) + "      n")
        for kind in KINDS:
            c = rate[kind]
            n = sum(c.values())
            if n:
                print("   %-22s                  " % kind + "  ".join("%5.1f%%    " % (100 * c[l] / n) for l in LEVELS) + " %6d" % n)
        print("   decisions by recovery (excess = model regret - control regret: lower = better relative to the control):")
        print("   kind                   teacher          recovery        n   uses / ctl     top-1 / ctl     regret / ctl    excess")
        for kind in KINDS:
            for tu in ("teacher uses", "teacher doesn't"):
                for lv in LEVELS:
                    r = dec.get((kind, tu, lv))
                    if not r or len(r["top1"]) < args.min_n:
                        continue
                    m = {k: np.nanmean(np.array(v, dtype=float)) for k, v in r.items()}
                    print("   %-22s %-16s %-11s %6d   %.3f / %.3f  %.3f / %.3f  %5.2f / %5.2f   %+5.2f" % (
                        kind, tu, lv, len(r["top1"]), m["uses"], m["ctl uses"], m["top1"], m["ctl top1"], m["regret"], m["ctl regret"],
                        m["regret"] - m["ctl regret"]))
        # 2. causal: the same positions with one kind forced into blobs
        if name not in args.force_models.split(",") or not args.force:
            continue
        print("   forced recovery (same positions; natural = the model's own merging):")
        print("   kind                   teacher             n   recovered nat / forced   uses nat / forced   top-1 nat / forced   regret nat / forced"
              "   | treated: n   regret nat / forced   uses nat / forced")
        for kind in args.force.split(","):
            fpick, fblobs = run(net, d, args.device, toks=toks, force=kind)
            freg = regret(fpick)
            for tu in ("teacher uses", "teacher doesn't"):
                rows = []
                for i in range(N):
                    for k2, mm in toks[i]:
                        if k2 != kind:
                            continue
                        if (uses(i, d["chosen"][i], mm)) != (tu == "teacher uses"):
                            continue
                        nat, frc = recovery(blobs[i], mm), recovery(fblobs[i], mm)
                        rows.append((nat in ("clean", "contained"), frc in ("clean", "contained"), uses(i, pick[i], mm), uses(i, fpick[i], mm),
                                     pick[i] == d["chosen"][i], fpick[i] == d["chosen"][i], reg[i], freg[i]))
                if len(rows) < args.min_n:
                    continue
                a = np.array(rows, dtype=float)
                tr = a[(a[:, 0] == 0) & (a[:, 1] == 1)]
                tline = "| %5d     %5.2f / %5.2f        %.3f / %.3f" % (len(tr), np.nanmean(tr[:, 6]), np.nanmean(tr[:, 7]),
                                                                    tr[:, 2].mean(), tr[:, 3].mean()) if len(tr) >= args.min_n else "| %5d" % len(tr)
                print("   %-22s %-16s %6d       %.3f / %.3f        %.3f / %.3f      %.3f / %.3f       %5.2f / %5.2f        %s" % (
                    kind, tu, len(a), a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean(), a[:, 3].mean(), a[:, 4].mean(), a[:, 5].mean(),
                    np.nanmean(a[:, 6]), np.nanmean(a[:, 7]), tline), flush=True)


if __name__ == "__main__":
    main()
