"""How often does ToMe merge filled cells with empty ones? For validation positions and each split
(token = as trained up to v4.1, hash, greedy), every merge step's merges by the kind of the two blobs:
empty + empty, filled + filled, empty + filled (two pure blobs of different kinds), or involving an
already mixed blob; plus how many board blobs go into each step and how many merges the budget asks
for; and at the last layer the share of mixed blobs (and of the cells in them), per stream.

    attentris/.venv/Scripts/python attentris/tome_mix.py --model attentris/data/model_tome_v41.pt --variants "split=token;split=greedy;split=greedy,pure"
"""
import argparse
import collections
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_tome as MT
import pcores
import train as TR

HERE = os.path.dirname(os.path.abspath(__file__))
KINDS = ("empty+empty", "filled+filled", "empty+filled", "with a mixed blob")


def kind_of(mask, cells):
    """mask (B, N, 200), cells (B, 200) -> (B, N): 0 empty, 1 filled, 2 mixed (-1 no cells)."""
    nf = (mask * cells[:, None, :]).sum(-1)
    n = mask.sum(-1)
    k = torch.where(nf == 0, 0, torch.where(nf == n, 1, 2))
    return torch.where(n > 0, k, -1)


def run(net, d, idx, spec, device):
    name = MT.apply_variant(net, spec)
    per_step = collections.defaultdict(lambda: np.zeros(len(KINDS) + 2))     # step -> kinds..., blobs in, asked
    final = np.zeros((2, 4))                                                  # stream -> blobs, mixed blobs, cells, cells in mixed
    for s in range(0, len(idx), 128):
        part = idx[s:s + 128]
        b = TR.batch_of(d, part, device)
        cells = b["cells"].float().view(len(part), 200)
        step = [0]

        def hook(e):
            step[0] += 1
            k = kind_of(e["mask"], cells)
            sel, dst, cand = e["sel"], e["dst"], e["cand"]
            a = per_step[step[0]]
            bi, ni = torch.nonzero(sel, as_tuple=True)
            ka, kb = k[bi, ni], k[bi, dst[bi, ni]]
            a[0] += int(((ka == 0) & (kb == 0)).sum())
            a[1] += int(((ka == 1) & (kb == 1)).sum())
            a[2] += int((((ka == 0) & (kb == 1)) | ((ka == 1) & (kb == 0))).sum())
            a[3] += int(((ka == 2) | (kb == 2)).sum())
            a[4] += int((cand & (k >= 0)).sum())
            a[5] += int(e["r"].sum())
        MT.MERGE_HOOK = hook
        with torch.no_grad():
            _, info = net(b, return_attn=True)
        MT.MERGE_HOOK = None
        last = info["merges"][-1]
        k = kind_of(last["mask"], cells)
        size = last["mask"].sum(-1)
        for st in (0, 1):
            m = last["alive"] & (k >= 0) & (last["stream"] == st)
            final[st] += [int(m.sum()), int((m & (k == 2)).sum()), float(size[m].sum()), float(size[m & (k == 2)].sum())]
        print("  %s: %d / %d positions" % (split, min(s + 128, len(idx)), len(idx)), flush=True)
    MT.apply_variant(net, "none")
    return name, per_step, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "data", "model_tome_v41.pt"))
    ap.add_argument("--data", default=os.path.join(HERE, "data", "v51", "val.npz"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--variants", default="split=token;split=hash;split=greedy", help="';'-separated model_tome.apply_variant specs")
    ap.add_argument("--device", default="cpu", help="cpu by default: never share the GPU with a training run")
    args = ap.parse_args()
    pcores.pin()
    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    a = ck["args"]
    net = MT.AtTenTrisToMe(d=a["d"], layers=a["layers"], merge=[float(f) for f in a["merge"].split(",")], sky=a.get("tome_sky", False),
                          pretok=a.get("tome_pretok", False), flat_bias=a.get("tome_flat", False),
                          streams=a.get("tome_streams", 1), budget=MT.budget_of(a.get("tome_budget", "")), split=a.get("tome_split", "token"), pure=a.get("tome_pure", False))
    net.load_state_dict(ck["model"])
    net.to(args.device).eval()
    d = TR.load(args.data)
    if a.get("tome_pretok"):
        d["pretok"] = np.load(args.data.replace(".npz", ".pretok.npy"))
    idx = np.random.default_rng(0).choice(len(d["chosen"]), size=min(args.n, len(d["chosen"])), replace=False)
    idx.sort()
    layers = [li + 1 for li, f in enumerate(net.merge) if f > 0]
    for spec in args.variants.split(";"):
        name, per_step, final = run(net, d, idx, spec, args.device)
        print("\n== %s, %d positions (both streams together)" % (name, len(idx)))
        print("   layer   blobs in / merges made / asked per position+stream   " + "   ".join("%-15s" % k_ for k_ in KINDS))
        tot = np.zeros(len(KINDS))
        for st_, a_ in sorted(per_step.items()):
            n = a_[:4].sum()
            tot += a_[:4]
            print("   %d       %5.1f / %4.1f / %4.1f (%4.1f%% of the blobs)        " % (
                layers[st_ - 1], a_[4] / (2 * len(idx)), n / (2 * len(idx)), a_[5] / (2 * len(idx)), 100 * n / a_[4]) +
                "   ".join("%5.1f%%         " % (100 * a_[j] / n) for j in range(4)))
        print("   all merges                                            " + "   ".join("%5.1f%%         " % (100 * t / tot.sum()) for t in tot))
        for st in (0, 1):
            f = final[st]
            print("   after the last layer, stream %d: %.1f board blobs per position, %.1f%% mixed (holding %.1f%% of the cells)" % (
                st + 1, f[0] / len(idx), 100 * f[1] / f[0], 100 * f[3] / f[2]))


if __name__ == "__main__":
    main()
