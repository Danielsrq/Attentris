"""Peak GPU memory of one AtTenTris training step (forward + loss + backward + Adam step) at batch 512 on
real positions, per model width, before committing to a run (8 GB RTX 3060 Ti).

    attentris/.venv/Scripts/python attentris/vram_probe.py --d 64,96,128
"""
import argparse
import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M
import train as TR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", default="64,96,128")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "v51", "dagger2.npz"),
                    help="positions to probe with (DAgger 2 boards have the most tokens)")
    args = ap.parse_args()
    d = TR.load(args.data)
    la = types.SimpleNamespace(tau=2.0, tau_dense=2.0, w_dense=0.0, w_q=0.0, w_q1=0.0, w_qs=1.0)
    # the largest boards: sort by token count so the probe is the worst case
    ntok = (d["tok_kind"] >= 0).sum(1) + (d["chosen"] * 0)
    idx = np.argsort(-ntok)[:args.batch]
    dev = "cuda"
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    print("GPU %s, %.1f GiB; batch %d of the boards with the most rule tokens (%d-%d)" % (
        torch.cuda.get_device_name(0), total, args.batch, ntok[idx].min(), ntok[idx].max()))
    for width in [int(x) for x in args.d.split(",")]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        net = M.AtTenTris(d=width, layers=4, n_teachers=3).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
        nparam = sum(p.numel() for p in net.parameters())
        try:
            for _ in range(2):
                out = net(TR.batch_of(d, idx, dev), heads=True)
                loss = TR.compute_loss(out, d, idx, la, dev)[0].mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            torch.cuda.synchronize()
            print("d=%-4d params %8d  peak allocated %.2f GiB  reserved %.2f GiB" % (
                width, nparam, torch.cuda.max_memory_allocated() / 2 ** 30, torch.cuda.max_memory_reserved() / 2 ** 30), flush=True)
        except torch.cuda.OutOfMemoryError:
            print("d=%-4d params %8d  OUT OF MEMORY at batch %d" % (width, nparam, args.batch), flush=True)
        del net, opt
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
