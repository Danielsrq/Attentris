"""Does AtTenTris-ToMe rediscover the rule tokens? For validation positions, every blob alive after
each merging layer is a candidate region; for each rule token (sky, wells, play rows, T
structures, slots, setups, pits; tokenizer v5.1) take the best IoU with any blob. Compared with
the same model merging random adjacent pairs on the same schedule (connectivity alone makes
compact blobs, so some overlap is expected by chance).

    attentris/.venv/Scripts/python attentris/tome_iou.py --model attentris/data/model_tome.pt
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


def blobs(net, d, idx, device, random_merge):
    net.random_merge = random_merge
    torch.manual_seed(0)
    with torch.no_grad():
        _, info = net(TR.batch_of(d, idx, device), return_attn=True)
    out = []                                          # per position: list of (layer, cell set)
    for i in range(len(idx)):
        nodes = []
        for li, t in enumerate(info["merges"]):
            if net.merge[li] <= 0:
                continue
            mk, al = t["mask"][i].cpu().numpy() > 0.5, t["alive"][i].cpu().numpy()
            for j in np.nonzero(al)[0]:
                if mk[j].any():
                    nodes.append((li + 1, mk[j]))
        out.append(nodes)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "data", "model_tome.pt"))
    ap.add_argument("--data", default=os.path.join(HERE, "data", "v51", "val.npz"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--merge-override", default="", help="LAYER:FRACTION at inference (model_tome frac_override)")
    ap.add_argument("--device", default="cpu", help="cpu by default: never share the GPU with a training run")
    args = ap.parse_args()
    pcores.pin()
    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    a = ck["args"]
    net = MT.AtTenTrisToMe(d=a["d"], layers=a["layers"], merge=[float(f) for f in a["merge"].split(",")], sky=a.get("tome_sky", False),
                          pretok=a.get("tome_pretok", False), flat_bias=a.get("tome_flat", False),
                          streams=a.get("tome_streams", 1), budget=MT.budget_of(a.get("tome_budget", "")))
    net.load_state_dict(ck["model"])
    net.to(args.device).eval()
    net.frac_override = MT.override_of(args.merge_override)
    d = TR.load(args.data)
    if a.get("tome_pretok"):
        d["pretok"] = np.load(args.data.replace(".npz", ".pretok.npy"))
    idx = np.random.default_rng(0).choice(len(d["chosen"]), size=min(args.n, len(d["chosen"])), replace=False)
    idx.sort()
    res = {False: collections.defaultdict(list), True: collections.defaultdict(list)}
    layer_of_best = collections.defaultdict(collections.Counter)
    for s in range(0, len(idx), 256):
        part = idx[s:s + 256]
        got = {rm: blobs(net, d, part, args.device, rm) for rm in (False, True)}
        for k, i in enumerate(part):
            masks = M.unpack_masks(d["tok_mask"][i])
            labs, kinds = [], []
            for j, kd in enumerate(d["tok_kind"][i]):
                if kd < 0 or T.KINDS[kd] in T.PIECE_KINDS or not masks[j].any():
                    continue
                kind = T.KINDS[kd]
                if kind in T.SLOT_KINDS:
                    kind = kind + (" covered" if d["tok_extra"][i, j, 7] > 0.5 else " open")
                labs.append(masks[j])
                kinds.append(kind)
            L = np.array(labs, np.int32)                                   # (labels, 200)
            for rm in (False, True):
                lay = np.array([li for li, _ in got[rm][k]])
                Nd = np.array([nm for _, nm in got[rm][k]], np.int32)      # (blobs, 200)
                inter = L @ Nd.T
                iou = inter / (L.sum(1)[:, None] + Nd.sum(1)[None, :] - inter)
                best = iou.max(1)
                for kind, bv, arg in zip(kinds, best, iou.argmax(1)):
                    res[rm][kind].append(float(bv))
                    if not rm:
                        layer_of_best[kind][int(lay[arg])] += 1
        print("  %d / %d positions" % (min(s + 256, len(idx)), len(idx)), flush=True)
    print("\nrule token kind      count   ToMe: mean best IoU, share >= 0.5   random merges: mean, share >= 0.5   best blob's layer (ToMe)")
    for kind in sorted(res[False], key=lambda k: -len(res[False][k])):
        v, r = np.array(res[False][kind]), np.array(res[True][kind])
        lay = layer_of_best[kind]
        print("  %-18s %6d      %.2f   %3.0f%%                     %.2f   %3.0f%%            %s" % (
            kind, len(v), v.mean(), 100 * (v >= 0.5).mean(), r.mean(), 100 * (r >= 0.5).mean(),
            " ".join("L%d:%d%%" % (l, 100 * c / len(v)) for l, c in sorted(lay.items()))))


if __name__ == "__main__":
    main()
