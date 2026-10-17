"""Does a model use its register tokens? (model.py --registers R)
1. Attention share: per self-attention layer, the mean attention weight that board-side tokens put on the
   registers (summed over the R registers, averaged over heads and real tokens); for the cross block, the share
   the candidates put on them. For reference, a register treated like an average token would get R / (NT + R).
2. Ablations, scored with train.py's evaluate on the validation files (top-1 against the teacher's move):
   - knock-out: no query can attend to the registers (they still exist, nobody reads them);
   - mean: after every layer the registers' states are replaced by their mean over the validation positions,
     so they carry no position-specific information.
   A drop in top-1 means the model relies on what the registers hold.
    <python with torch> attentris/register_usage.py --model data/model_v51_d128_reg4_ft.pt [--val val.npz,val_dagger5all.npz]
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model as M  # noqa: E402
import pcores  # noqa: E402
import train as TR  # noqa: E402


def load(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck.get("args", {})
    net = M.AtTenTris(d=a.get("d", 64), layers=a.get("layers", 4), cross_layers=a.get("cross_layers", 1),
                      cross_loops=a.get("cross_loops", 1), n_teachers=ck["model"]["style.weight"].shape[0],
                      n_kinds=ck["model"]["kind.weight"].shape[0], registers=a.get("registers", 0)).to(device)
    net.load_state_dict(ck["model"])
    return net.eval(), a


def attention_share(net, d, device, n=4096, batch=256):
    R = net.registers
    L = len(net.blocks)
    tot_self, tot_cross, cnt_self, cnt_cross = np.zeros(L), 0.0, 0.0, 0.0
    ref = []
    with torch.no_grad():
        for s in range(0, min(n, len(d["chosen"])), batch):
            idx = np.arange(s, min(n, len(d["chosen"]), s + batch))
            b = TR.batch_of(d, idx, device)
            _, at = net(b, return_attn=True)
            real = b["tok_kind"] >= 0                                   # (B, NT) real board tokens
            NT = real.shape[1]
            for li, a in enumerate(at["self"]):                         # (B, H, NT+R, NT+R)
                w = a[:, :, :NT, NT:].sum(-1).mean(1)                   # (B, NT)
                tot_self[li] += float((w * real).sum())
            cnt_self += float(real.sum())
            ref.append(float((R / (real.sum(1) + R)).mean()))
            ca = at["cross"]                                            # (B, H, NC, NT+R)
            creal = b["cand_cells"][..., 0] != -1
            w = ca[..., NT:].sum(-1).mean(1)
            tot_cross += float((w * creal).sum())
            cnt_cross += float(creal.sum())
    return tot_self / cnt_self, tot_cross / cnt_cross, float(np.mean(ref))


def register_means(net, d, device, n=4096, batch=256):
    net.reg_record = {}
    with torch.no_grad():
        for s in range(0, min(n, len(d["chosen"])), batch):
            idx = np.arange(s, min(n, len(d["chosen"]), s + batch))
            net(TR.batch_of(d, idx, device))
    means = [torch.cat(net.reg_record[li]).mean(0).to(device) for li in range(len(net.blocks))]
    net.reg_record = None
    return means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=os.path.join(HERE, "data", "v51"))
    ap.add_argument("--val", default="val.npz,val_dagger5all.npz")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    pcores.pin(verbose=False)
    net, a = load(args.model, args.device)
    if not net.registers:
        sys.exit("%s has no registers" % args.model)
    name = os.path.basename(args.model)
    print("%s: %d registers; reg_k %s  reg_c %s" % (name, net.registers, np.round(net.reg_k.detach().cpu().numpy(), 2),
                                                    np.round(net.reg_c.detach().cpu().numpy(), 2)))
    targs = argparse.Namespace(tau=a.get("tau", 2.0), w_dense=0.0, tau_dense=2.0, w_q=0.0, w_q1=0.0, w_qs=1.0)
    for vf in args.val.split(","):
        d = TR.load(os.path.join(args.data, vf))
        sh_self, sh_cross, ref = attention_share(net, d, args.device)
        print("%s: attention on the registers: self layers %s (an average token's share: %.3f), cross %.3f" % (
            vf, " ".join("%.3f" % x for x in sh_self), ref, sh_cross))
        means = register_means(net, d, args.device)
        rows = []
        for mode in ("normal", "knock-out", "mean"):
            net.reg_knockout = mode == "knock-out"
            net.reg_mean = means if mode == "mean" else None
            ev = TR.evaluate(net, d, 512, targs, args.device)
            net.eval()
            parts = ["%s %.4f (n %d)" % (t, v["top1"], v["n"]) for t, v in ev.items() if v["n"]]
            rows.append((mode, parts))
        net.reg_knockout, net.reg_mean = False, None
        for mode, parts in rows:
            print("  %-9s top1: %s" % (mode, ", ".join(parts)))


if __name__ == "__main__":
    main()
