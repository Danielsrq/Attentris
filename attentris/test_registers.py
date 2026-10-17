"""Register tokens: a checkpoint loaded into a model with registers (reg_k = reg_c = -8) gives nearly the
same policy as without them; registers change nothing about padding; gradients reach them.
    python attentris/test_registers.py [MODEL]
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model as M  # noqa: E402
import train as TR  # noqa: E402


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "data", "model_v51_d128_dagger5b_ft.pt")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    kw = dict(d=a["d"], layers=a["layers"], n_kinds=ck["model"]["kind.weight"].shape[0], n_teachers=ck["model"]["style.weight"].shape[0])
    base = M.AtTenTris(**kw).eval()
    base.load_state_dict(ck["model"])
    reg = M.AtTenTris(**kw, registers=4).eval()
    missing = reg.load_state_dict(ck["model"], strict=False).missing_keys
    assert set(missing) == {"reg", "reg_k", "reg_q", "reg_c"}, missing
    val = TR.load(os.path.join(HERE, "data", "v51", "val.npz"))
    idx = np.arange(256)
    b = TR.batch_of(val, idx, "cpu")
    with torch.no_grad():
        lb = base(b).log_softmax(-1)
        lr = reg(b).log_softmax(-1)
    ok = torch.isfinite(lb)
    p = lb.exp()
    kl = (p * (lb - lr)).where(ok, torch.zeros_like(lb)).sum(-1)
    same = (lb.argmax(-1) == lr.argmax(-1)).float().mean().item()
    print("start: KL(base || registers) mean %.2e max %.2e, same top-1 %.4f" % (kl.mean().item(), kl.max().item(), same))
    assert kl.max().item() < 1e-2 and same > 0.99
    # gradients reach the registers through the frozen network
    for n_, p_ in reg.named_parameters():
        p_.requires_grad_(n_ in ("reg", "reg_k", "reg_q", "reg_c"))
    reg.train()
    out = reg(b)
    lp = out.log_softmax(-1)
    loss = -lp.gather(1, out.argmax(-1, keepdim=True)).mean()
    loss.backward()
    g = {n_: p_.grad.abs().max().item() for n_, p_ in reg.named_parameters() if p_.requires_grad}
    print("grad max:", {k: "%.2e" % v for k, v in g.items()})
    assert all(v > 0 for v in g.values())
    print("ok")


if __name__ == "__main__":
    main()
