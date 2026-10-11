"""Model checks from DESIGN.md step 4: shapes, and permutation equivariance -- shuffling the
board-side tokens (with their masks and metadata) must not change any candidate's logit,
and shuffling the candidates must permute the logits the same way.

    <python with torch> attentris/test_model.py [path/to/some.npz]
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M
import train as TR

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "data", "val.npz")
    d = TR.load(path)
    idx = np.arange(min(8, len(d["chosen"])))
    torch.manual_seed(0)
    net = M.AtTenTris().eval()
    b = TR.batch_of(d, idx)
    with torch.no_grad():
        logit, att = net(b, return_attn=True)
    B, NC = b["cand_cells"].shape[:2]
    NT = b["tok_kind"].shape[1]
    assert logit.shape == (B, NC)
    assert att["cross"].shape == (B, 4, NC, NT) and att["pool"].shape == (B, NT, 200)
    assert all(a.shape == (B, 4, NT, NT) for a in att["self"])
    ok = ~torch.isinf(logit)
    assert (ok == (b["cand_cells"][..., 0] != -1)).all(), "padding candidates must be -inf, real ones finite"

    # shuffle board-side tokens
    perm = torch.randperm(NT)
    b2 = dict(b)
    for k in ("tok_kind", "tok_mask", "tok_box", "tok_extra", "tok_piece"):
        b2[k] = b[k][:, perm]
    with torch.no_grad():
        l2 = net(b2)
    assert torch.allclose(logit[ok], l2[ok], atol=1e-4), (logit[ok] - l2[ok]).abs().max()

    # shuffle candidates
    cperm = torch.randperm(NC)
    b3 = dict(b)
    for k in ("cand_cells", "cand_meta"):
        b3[k] = b[k][:, cperm]
    with torch.no_grad():
        l3 = net(b3)
    assert torch.allclose(logit[:, cperm][ok[:, cperm]], l3[ok[:, cperm]], atol=1e-4)
    print("ok: shapes, token permutation invariance, candidate permutation equivariance (%d positions)" % B)


if __name__ == "__main__":
    main()
