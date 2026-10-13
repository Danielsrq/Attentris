"""How much do trained models rely on FLAT-2D's pairwise geometry? At inference, the bias of every
board-token pair is replaced by its per-head mean over the position's pairs (distance / overlap no
longer matter; the learned constant for piece-token pairs and the box embeddings stay), in the
self-attention, the candidates' cross-attention, or both. Top-1 with the teacher on the validation set
per variant. This measures reliance, not necessity: a model trained without the bias could compensate
(box embeddings carry absolute position).

    attentris/.venv/Scripts/python attentris/flat_ablate.py --model v51_d3=data/model_v51_dagger3_ft.pt --model l4g_d3=data/model_tome_v42_l4g_dagger3_ft.pt
"""
import argparse
import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M
import model_tome as MT
import pcores
import train as TR

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    nt = ck["model"]["style.weight"].shape[0]
    if a.get("arch") == "tome":
        net = MT.AtTenTrisToMe(d=a["d"], layers=a["layers"], merge=[float(f) for f in a["merge"].split(",")], sky=a.get("tome_sky", False),
                              pretok=a.get("tome_pretok", False), flat_bias=a.get("tome_flat", False),
                              streams=a.get("tome_streams", 1), budget=MT.budget_of(a.get("tome_budget", "")),
                              split=a.get("tome_split", "token"), pure=a.get("tome_pure", False), n_teachers=nt)
    else:
        net = M.AtTenTris(d=a.get("d", 64), layers=a.get("layers", 4), cross_layers=a.get("cross_layers", 1),
                          cross_loops=a.get("cross_loops", 1), n_teachers=nt)
    net.load_state_dict(ck["model"])
    return net.to(device).eval(), a.get("arch") == "tome"


def flatten(bias_mod):
    """Make a FLAT-2D Bias module position-blind: board pairs get the per-head mean of their bias."""
    def fwd(self, feats, both):
        b = self.mlp(feats)                                                       # (B, Nq, Nk, H)
        w = both.unsqueeze(-1).float()
        mean = (b * w).sum((1, 2), keepdim=True) / w.sum((1, 2), keepdim=True).clamp(min=1.0)
        b = torch.where(both.unsqueeze(-1), mean.expand_as(b), self.nobox)
        return b.permute(0, 3, 1, 2)
    bias_mod.forward = types.MethodType(fwd, bias_mod)


def top1(net, d, device, batch=256):
    hit = {0: [0, 0], 1: [0, 0]}
    with torch.no_grad():
        for s in range(0, len(d["chosen"]), batch):
            idx = np.arange(s, min(len(d["chosen"]), s + batch))
            pick = net(TR.batch_of(d, idx, device)).argmax(-1).cpu().numpy()
            ok = pick == d["chosen"][idx]
            for t in (0, 1):
                m = d["teacher"][idx] == t
                hit[t][0] += int(ok[m].sum()); hit[t][1] += int(m.sum())
    return [hit[t][0] / max(1, hit[t][1]) for t in (0, 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="name=checkpoint (repeatable)")
    ap.add_argument("--data", default=os.path.join(HERE, "data", "v51", "val.npz"))
    ap.add_argument("--device", default="cpu", help="cpu by default: never share the GPU with a training run")
    args = ap.parse_args()
    pcores.pin()
    d = TR.load(args.data)
    d["pretok"] = np.load(args.data.replace(".npz", ".pretok.npy"))
    print("%-14s %-22s  top-1 normal / gate4   (drop vs full)" % ("model", "variant"))
    for spec in args.model:
        name, path = spec.split("=", 1)
        full = None
        for variant in ("full", "self geometry off", "cross geometry off", "both off"):
            net, tome = load(path, args.device)
            if variant in ("self geometry off", "both off"):
                if isinstance(net.bias, M.Bias):
                    flatten(net.bias)
                else:
                    print("%-14s %-22s  (self bias is not FLAT-2D, skipped)" % (name, variant)); continue
            if variant in ("cross geometry off", "both off"):
                flatten(net.c_bias)
                for b in getattr(net, "c_bias_extra", []):
                    flatten(b)
            t = top1(net, d, args.device)
            full = full or t
            print("%-14s %-22s  %.3f / %.3f          (%+.3f / %+.3f)" % (name, variant, t[0], t[1], t[0] - full[0], t[1] - full[1]), flush=True)


if __name__ == "__main__":
    main()
