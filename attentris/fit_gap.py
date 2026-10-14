"""Overfitting check: a checkpoint's top-1 / top-3 / loss / Qs error on a random sample of positions it was
trained on vs the held-out validation set (same teacher labels). A small gap = not overfitting; then more
data mainly helps if the model is data-limited, not capacity- or label-noise-limited.

    attentris/.venv/Scripts/python attentris/fit_gap.py --model data/model_v51.pt --train train1m.npz --n 20000
"""
import argparse
import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pcores
import train as TR
from tome_recovery_play import load_control


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "v51"))
    ap.add_argument("--train", default="train1m.npz", help="comma-separated training files to sample from")
    ap.add_argument("--val", default="val.npz")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    pcores.pin(verbose=False)
    net = load_control(args.model).to(args.device)
    la = types.SimpleNamespace(tau=2.0, tau_dense=2.0, w_dense=0.0, w_q=0.0, w_q1=0.0, w_qs=1.0)
    rng = np.random.default_rng(0)
    print("%-28s %7s %7s %7s %7s %8s" % ("set", "n", "top1", "top3", "loss", "Qs MAE"))
    for f in args.train.split(",") + [args.val]:
        d = TR.load(os.path.join(args.data, f))
        n = len(d["chosen"])
        if n > args.n:
            idx = np.sort(rng.choice(n, args.n, replace=False))
            d = {k: v[idx] for k, v in d.items()}
        r = TR.evaluate(net, d, 512, la, args.device)
        for name, s in r.items():
            if s["n"]:
                print("%-28s %7d %7.3f %7.3f %7.3f %8.2f" % ("%s [%s]" % (f, name), s["n"], s["top1"], s["top3"], s["loss"], s["qs_mae"]), flush=True)


if __name__ == "__main__":
    main()
