"""Does a ToMe model choose better on the positions where its blobs recover the structure? Per validation
position and rule-token kind (tsd, covered fit_t, pit1, ...): the best IoU between each such token and
the model's final blobs (both streams), and the model's decision quality there: top-1 with the teacher,
and regret = the teacher's search score of the best move minus that of the model's pick (when the pick
was scored). Positions are bucketed by recovery; a rule-token model (the control, e.g. v5.1) is scored on
the same positions, so a bucket that is merely easier shows up in the control too. "uses it": the
teacher's move puts >= 3 cells into that token's mask (the structure is the one that matters).

    attentris/.venv/Scripts/python attentris/tome_recovery_play.py --tome l4g=data/model_tome_v42_l4g.pt --control data/model_v51.pt
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

HERE = os.path.dirname(os.path.abspath(__file__))
BINS = ((0.0, 0.5, "IoU < 0.5"), (0.5, 0.8, "0.5-0.8"), (0.8, 1.01, ">= 0.8"))


def load_tome(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    net = MT.AtTenTrisToMe(d=a["d"], layers=a["layers"], merge=[float(f) for f in a["merge"].split(",")], sky=a.get("tome_sky", False),
                          pretok=a.get("tome_pretok", False), flat_bias=a.get("tome_flat", False),
                          streams=a.get("tome_streams", 1), budget=MT.budget_of(a.get("tome_budget", "")), split=a.get("tome_split", "token"),
                          pure=a.get("tome_pure", False), n_teachers=ck["model"]["style.weight"].shape[0])
    net.load_state_dict(ck["model"])
    return net.eval()


def load_control(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    net = M.AtTenTris(d=a.get("d", 64), layers=a.get("layers", 4), cross_layers=a.get("cross_layers", 1),
                      cross_loops=a.get("cross_loops", 1), n_teachers=ck["model"]["style.weight"].shape[0])
    net.load_state_dict(ck["model"])
    return net.eval()


def decisions(logit, d, idx):
    """top-1 hit and regret per position (regret nan when the pick has no search score)."""
    pick = logit.argmax(-1).numpy()
    best = d["cand_score"][idx][..., 1].astype(np.float64)                     # search score, nan = pruned
    top = np.nanmax(best, axis=1)
    reg = top - best[np.arange(len(idx)), pick]
    return pick == d["chosen"][idx], reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tome", action="append", required=True, help="name=checkpoint (repeatable)")
    ap.add_argument("--control", default=os.path.join(HERE, "data", "model_v51.pt"))
    ap.add_argument("--data", default=os.path.join(HERE, "data", "v51", "val.npz"))
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--kinds", default="tsd,fit_t covered,pit1")
    args = ap.parse_args()
    pcores.pin()
    torch.set_num_threads(8)
    d = TR.load(args.data)
    d["pretok"] = np.load(args.data.replace(".npz", ".pretok.npy"))
    idx = np.arange(min(args.n, len(d["chosen"])))
    kinds = args.kinds.split(",")
    # rule tokens per position: (kind, mask, used by the teacher's move)
    toks = []
    for i in idx:
        masks = M.unpack_masks(d["tok_mask"][i])
        ch = set(int(c) for c in d["cand_cells"][i, d["chosen"][i]] if c >= 0)
        row = []
        for j, kd in enumerate(d["tok_kind"][i]):
            if kd < 0 or not masks[j].any():
                continue
            kind = T.KINDS[kd]
            if kind in T.SLOT_KINDS:
                kind = kind + (" covered" if d["tok_extra"][i, j, 7] > 0.5 else " open")
            if kind in kinds:
                mm = masks[j].reshape(200)
                row.append((kind, mm, len(ch & set(np.nonzero(mm)[0].tolist())) >= 3))
        toks.append(row)
    # the control's decisions (rule tokens: unaffected by any ToMe blob)
    ctl_hit, ctl_reg = [], []
    cnet = load_control(args.control)
    with torch.no_grad():
        for s in range(0, len(idx), 256):
            part = idx[s:s + 256]
            h, r = decisions(cnet(TR.batch_of(d, part)), d, part)
            ctl_hit.append(h); ctl_reg.append(r)
    ctl_hit, ctl_reg = np.concatenate(ctl_hit), np.concatenate(ctl_reg)
    print("control %s: top-1 %.3f on %d positions" % (os.path.basename(args.control), ctl_hit.mean(), len(idx)))
    for spec in args.tome:
        name, path = spec.split("=", 1)
        net = load_tome(path)
        hit, reg, rec = [], [], []
        with torch.no_grad():
            for s in range(0, len(idx), 256):
                part = idx[s:s + 256]
                logit, info = net(TR.batch_of(d, part), return_attn=True)
                h, r = decisions(logit, d, part)
                hit.append(h); reg.append(r)
                last = info["merges"][-1]
                mk = last["mask"].numpy() > 0.5
                al = last["alive"].numpy()
                for k, i in enumerate(part):
                    blobs = mk[k][al[k] & mk[k].any(-1)].astype(np.int32)        # (blobs, 200), both streams
                    out = []
                    for kind, mm, used in toks[i]:
                        inter = blobs @ mm.astype(np.int32)
                        iou = inter / (blobs.sum(1) + mm.sum() - inter)
                        out.append((kind, float(iou.max()), used))
                    rec.append(out)
        hit, reg = np.concatenate(hit), np.concatenate(reg)
        print("\n== %s: top-1 %.3f overall (control %.3f)" % (name, hit.mean(), ctl_hit.mean()))
        print("   kind            subset     bucket       n    ToMe top-1  control top-1  ToMe - control   ToMe regret  control regret")
        for kind in kinds:
            for subset in ("all", "uses it"):
                for lo, hi, lab in BINS:
                    sel = []
                    for p, out in enumerate(rec):
                        v = [iou for kd, iou, used in out if kd == kind and (subset == "all" or used)]
                        if v and lo <= max(v) < hi:
                            sel.append(p)
                    if len(sel) < 20:
                        continue
                    sel = np.array(sel)
                    th, ch = hit[sel].mean(), ctl_hit[sel].mean()
                    tr, cr = np.nanmean(reg[sel]), np.nanmean(ctl_reg[sel])
                    print("   %-15s %-9s  %-9s %6d       %.3f        %.3f         %+.3f         %5.2f        %5.2f" % (
                        kind, subset, lab, len(sel), th, ch, th - ch, tr, cr))


if __name__ == "__main__":
    main()
