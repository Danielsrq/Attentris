"""Where does each model's attention go when it scores the bot's T-spin into a T structure?
Mean share of the chosen candidate's cross-attention (mean of heads) by token group, over
every such position in the validation set, for several checkpoints.

    <python with torch> attentris/attn_shift.py model_a.pt model_b.pt ...
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn_viz as AV
import model as M
import pcores
import tokenizer as T
import train as TR

HERE = os.path.dirname(os.path.abspath(__file__))
GROUPS = {
    "T structure": set(T.STRUCTURE_KINDS),
    "piece queue (current, hold, previews)": set(T.PIECE_KINDS),
    "wells": {"well"},
    "play rows / bands": {"play", "band"},
    "sky": {"sky"},
    "recesses + pits + rect12": set(T.FIT_KINDS) | {"pit1", "pit2", "pit3", "pit4", "rect12"},
}


def main():
    n_p = pcores.pin(verbose=False)
    torch.set_num_threads(n_p or 8)
    d = TR.load(os.path.join(HERE, "data", "v3", "val.npz"))
    n = len(d["chosen"])
    struct_ids = {T.KINDS.index(k) for k in T.STRUCTURE_KINDS}
    rows = []
    for i in range(n):
        ch = d["chosen"][i]
        m = d["cand_meta"][i, ch]
        if m[0] == AV.T_ID and m[2] == 2 and m[3] >= 1 and any(k in struct_ids for k in d["tok_kind"][i]):
            rows.append(i)
    rows = np.array(rows)
    print("%d T-spin positions with a T structure on the board" % len(rows))
    gid = {T.KINDS.index(k): g for g, ks in GROUPS.items() for k in ks}
    for path in sys.argv[1:]:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        net = M.AtTenTris(d=ck["args"]["d"], layers=ck["args"]["layers"], n_kinds=ck["model"]["kind.weight"].shape[0])
        net.load_state_dict(ck["model"], strict=False)
        net.eval()
        tot = {g: 0.0 for g in GROUPS}
        for s in range(0, len(rows), 256):
            idx = rows[s:s + 256]
            _, cross = AV.run(net, d, idx)
            for k, i in enumerate(idx):
                a = cross[k, d["chosen"][i]]
                for j, kd in enumerate(d["tok_kind"][i]):
                    if kd >= 0:
                        tot[gid.get(int(kd), "sky")] += float(a[j])
        print("\n%s" % os.path.basename(path))
        for g in GROUPS:
            print("  %-40s %5.1f%%" % (g, 100 * tot[g] / len(rows)))


if __name__ == "__main__":
    main()
