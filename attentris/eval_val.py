"""Score a checkpoint on a packed validation file (train.py's evaluate), e.g. a baseline on the
students' own positions:

    <python with torch> attentris/eval_val.py --model data/model_v51.pt --data data/v51 --val val_dagger2.npz
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M
import pcores
import train as TR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", default="val.npz")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--as-teacher", default="", help="score every position with this style input (normal / gate4 / danger)")
    args = ap.parse_args()
    pcores.pin()
    ck = torch.load(args.model, map_location=args.device, weights_only=False)
    a = ck.get("args", {})
    net = M.AtTenTris(d=a.get("d", 64), layers=a.get("layers", 4), cross_layers=a.get("cross_layers", 1),
                      cross_loops=a.get("cross_loops", 1), n_teachers=ck["model"]["style.weight"].shape[0]).to(args.device)
    net.load_state_dict(ck["model"])
    targs = argparse.Namespace(tau=a.get("tau", 2.0), w_dense=0.0, tau_dense=2.0, w_q=0.0, w_q1=0.0, w_qs=1.0)
    d = TR.load(os.path.join(args.data, args.val))
    if args.as_teacher:      # score with this style input (e.g. a model without the data's style)
        net_teacher = TR.TEACH.index(args.as_teacher)
        real = d["teacher"].copy()
        d["teacher"] = np.full_like(real, net_teacher)
        print("scoring %s as style %s" % (args.val, args.as_teacher))
    ev = TR.evaluate(net, d, 512, targs, args.device)
    for name, v in ev.items():
        if v["n"]:
            print("%s on %s (%s, %d): top1 %.3f top3 %.3f loss %.3f | argmax Qs top1 %.3f | MAE Qs %.2f" % (
                os.path.basename(args.model), args.val, name, v["n"], v["top1"], v["top3"], v["loss"], v["qs_top1"], v["qs_mae"]))


if __name__ == "__main__":
    main()
