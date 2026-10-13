"""Checks for model_tome's A/B split (the sideways-merging bug, tome_orient.py):

1. regression: a split="token" model (every checkpoint up to v4.1) gives the same logits as the
   model_tome.py of commit bfa434c (before the split option existed);
2. symmetry: with split="hash", side-by-side (H) and stacked (V) adjacent pairs are eligible (one in
   A, one in B) equally often at every merge step after the first -- trained v4.1 weights and a
   random-init model; the token split is shown for contrast;
3. determinism / batch independence: a position's logits under split="hash" are the same alone and
   inside a batch, and on a second run;
4. the two streams get different splits (share of tokens on the same side in both ~ 1/2);
5. the hash split is the same on CUDA as on the CPU (training on the GPU, play on the CPU), if CUDA is free;
6. split="greedy" (no A/B sets): every merge step equals a sequential greedy matching on the same
   similarities; no blob used twice; only adjacent same-stream board blobs; deterministic, batch-independent
   (and CUDA = CPU with --cuda); the budget shortfall per step; ms per move (one position, CPU) vs hash;
7. pure merges (model_tome.apply_variant "pure"): no blob ever holds both filled and empty cells, with greedy
   and hash; "6:x2" doubles the merges layer 6 is asked for (as far as the pure pairs allow).

    attentris/.venv/Scripts/python attentris/test_tome_split.py [--cuda]
"""
import importlib.util
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_tome as MT
import tome_orient as TO
import train as TR

HERE = os.path.dirname(os.path.abspath(__file__))


def build(mod, a, split=None):
    kw = {} if split is None else {"split": split}
    return mod.AtTenTrisToMe(d=a["d"], layers=a["layers"], merge=[float(f) for f in a["merge"].split(",")], sky=a.get("tome_sky", False),
                             pretok=a.get("tome_pretok", False), flat_bias=a.get("tome_flat", False),
                             streams=a.get("tome_streams", 1), budget=mod.budget_of(a.get("tome_budget", "")), **kw).eval()


def eligibility(net, d, idx):
    st = TO.Stats()
    for s in range(0, len(idx), 128):
        part = idx[s:s + 128]
        b = TR.batch_of(d, part)
        st.batch0, st.layer = s, 0
        MT.MERGE_HOOK = st.hook(b["cells"].float().view(len(part), 200))
        with torch.no_grad():
            net(b)
        MT.MERGE_HOOK = None
    out = {}
    for (L, cls, sub), a in st.pair.items():
        if sub == "all" and cls in ("H", "V") and a[0]:
            out[(L, cls)] = a[1] / a[0]
    return out


def greedy_checks(a, ck, d, idx):
    import time
    ok = True
    net = build(MT, a, "greedy")
    net.load_state_dict(ck["model"])
    events = []
    MT.MERGE_HOOK = lambda e: events.append({k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in e.items()})
    with torch.no_grad():
        out = net(TR.batch_of(d, idx[:128]))
    MT.MERGE_HOOK = None
    bad_ref = bad_twice = bad_valid = 0
    short, want = np.zeros(len(events)), np.zeros(len(events))
    for L, e in enumerate(events):
        S, adj, cand, stream, sel, dst, mask, r = (e[k] for k in ("S", "adj", "cand", "stream", "sel", "dst", "mask", "r"))
        anchor = mask.argmax(-1)
        for b in range(S.shape[0]):
            got = {(int(i), int(dst[b, i])) for i in torch.nonzero(sel[b]).flatten()}
            used = [x for pr in got for x in pr]
            bad_twice += len(used) != len(set(used))
            for i, j in got:
                bad_valid += not (bool(cand[b, i]) and bool(cand[b, j]) and bool(adj[b, i, j]) and int(stream[b, i]) == int(stream[b, j]) and i != j)
            for st in range(r.shape[1]):
                toks = [i for i in range(S.shape[1]) if bool(cand[b, i]) and int(stream[b, i]) == st]
                edges = []
                for x_ in range(len(toks)):
                    for y_ in range(x_ + 1, len(toks)):
                        i, j = toks[x_], toks[y_]
                        if bool(adj[b, i, j]):
                            lo, hi = sorted((int(anchor[b, i]), int(anchor[b, j])))
                            edges.append((float(S[b, i, j].double()) - 1e-9 * (lo * 200 + hi) / 40000.0, i, j))
                edges.sort(reverse=True)
                taken, pick = set(), []
                for w, i, j in edges:                         # the slow sequential greedy reference
                    if i not in taken and j not in taken:
                        taken |= {i, j}; pick.append((j, i))  # later token merges into the earlier
                rr = int(r[b, st])
                ref = set(pick[:rr])
                mine = {pr for pr in got if int(stream[b, pr[0]]) == st}
                bad_ref += ref != mine
                want[L] += rr; short[L] += max(0, rr - len(pick))
    print("6. greedy: steps differing from the sequential reference %d, blobs used twice %d, invalid pairs %d" % (bad_ref, bad_twice, bad_valid))
    print("   budget shortfall per step (merges asked / missing): " + "  ".join("%d/%d" % (w, s_) for w, s_ in zip(want, short)))
    ok &= bad_ref == 0 and bad_twice == 0 and bad_valid == 0
    with torch.no_grad():
        again = net(TR.batch_of(d, idx[:128]))
        worst = 0.0
        for i in (0, 9, 77, 127):
            one = net(TR.batch_of(d, idx[i:i + 1]))[0]
            f = torch.isfinite(one)
            worst = max(worst, (one[f] - out[i][f]).abs().max().item())
    rep = (out[torch.isfinite(out)] - again[torch.isfinite(again)]).abs().max().item()
    print("   rerun max diff %.2e, alone vs in a batch of 128 max diff %.2e" % (rep, worst))
    ok &= rep == 0.0 and worst < 1e-4
    if "--cuda" in sys.argv and torch.cuda.is_available():
        net.cuda()
        with torch.no_grad():
            g = net(TR.batch_of(d, idx[:128], "cuda")).cpu()
        net.cpu()
        f = torch.isfinite(out)
        gd = (g[f] - out[f]).abs().max().item()
        print("   greedy CUDA vs CPU logits max diff %.2e" % gd)
        ok &= gd < 1e-3 and bool((torch.isfinite(g) == f).all())
    times = {}
    for split in ("hash", "greedy"):
        n2 = build(MT, a, split); n2.load_state_dict(ck["model"])
        bs = [TR.batch_of(d, idx[i:i + 1]) for i in range(64)]
        with torch.no_grad():
            for b_ in bs[:8]:
                n2(b_)
            t0 = time.perf_counter()
            for b_ in bs:
                n2(b_)
        times[split] = 1000 * (time.perf_counter() - t0) / len(bs)
    print("   ms per position (one at a time, CPU, 4 threads): hash %.1f, greedy %.1f" % (times["hash"], times["greedy"]))
    return ok


def pure_checks(a, ck, d, idx):
    ok = True
    b = TR.batch_of(d, idx[:128])
    cells = b["cells"].float().view(-1, 200)
    asked = {}
    for spec in ("split=greedy", "split=greedy,pure", "split=hash,pure", "split=greedy,pure,6:x2"):
        net = build(MT, a)
        net.load_state_dict(ck["model"])
        MT.apply_variant(net, spec)
        ev = []
        MT.MERGE_HOOK = lambda e: ev.append((e["mask"].clone(), e["sel"].clone(), e["r"].clone()))
        with torch.no_grad():
            _, info = net(b, return_attn=True)
        MT.MERGE_HOOK = None
        masks = [m for m, _, _ in ev] + [info["merges"][-1]["mask"]]
        mixed = 0
        for m in masks:
            nf = (m * cells[:, None, :]).sum(-1)
            n = m.sum(-1)
            mixed += int(((nf > 0.5) & (nf < n - 0.5)).sum())
        asked[spec] = [int(r.sum()) for _, _, r in ev]
        made = [int(sl.sum()) for _, sl, _ in ev]
        print("7. %-24s mixed blobs %5d   merges made per step %s, asked %s" % (spec, mixed, made, asked[spec]))
        if "pure" in spec:
            ok &= mixed == 0
    ok &= asked["split=greedy,pure,6:x2"][-1] == sum(2 * x for x in [asked["split=greedy,pure"][-1]]) or \
        abs(asked["split=greedy,pure,6:x2"][-1] - 2 * asked["split=greedy,pure"][-1]) <= 128   # floor per position
    ok &= asked["split=greedy,pure,6:x2"][:-1] == asked["split=greedy,pure"][:-1]
    return ok


def main():
    torch.set_num_threads(4)
    ck = torch.load(os.path.join(HERE, "data", "model_tome_v41.pt"), map_location="cpu", weights_only=False)
    a = ck["args"]
    d = TR.load(os.path.join(HERE, "data", "v51", "val.npz"))
    d["pretok"] = np.load(os.path.join(HERE, "data", "v51", "val.pretok.npy"))
    idx = np.arange(512)
    ok = True

    # 1. regression against the pre-change model_tome.py
    old_src = subprocess.run(["git", "show", "bfa434c:attentris/model_tome.py"], capture_output=True, text=True, cwd=HERE, check=True).stdout
    tmp = os.path.join(os.environ.get("CLAUDE_JOB_DIR", HERE), "tmp", "model_tome_bfa434c.py")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    open(tmp, "w", encoding="utf-8").write(old_src)
    spec = importlib.util.spec_from_file_location("model_tome_old", tmp)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    n_old, n_new = build(old, a), build(MT, a, a.get("tome_split", "token"))
    n_old.load_state_dict(ck["model"]); n_new.load_state_dict(ck["model"])
    b = TR.batch_of(d, idx[:128])
    with torch.no_grad():
        lo, ln = n_old(b), n_new(b)
    fin = torch.isfinite(lo)
    diff = (lo[fin] - ln[fin]).abs().max().item()
    same_mask = bool((fin == torch.isfinite(ln)).all())
    print("1. token split vs bfa434c: max |logit diff| %.2e, same padding %s" % (diff, same_mask))
    ok &= diff == 0.0 and same_mask

    # 2. symmetry of eligibility, trained and random-init weights
    print("2. eligible share of adjacent same-stream pairs, H vs V, merge steps 2-5")
    for name, sd in (("v4.1 weights", ck["model"]), ("random init", None)):
        for split in ("token", "hash"):
            torch.manual_seed(1)
            net = build(MT, a, split)
            if sd is not None:
                net.load_state_dict(sd)
            el = eligibility(net, d, idx)
            row = []
            for L in range(2, 6):
                h, v = el.get((L, "H"), float("nan")), el.get((L, "V"), float("nan"))
                row.append("step %d H %.1f%% V %.1f%%" % (L, 100 * h, 100 * v))
                if split == "hash":
                    ok &= abs(h - v) < 0.03 and 0.4 < h < 0.6
            print("   %-12s %-5s %s" % (name, split, "   ".join(row)))

    # 3. determinism and batch independence (hash)
    net = build(MT, a, "hash")
    net.load_state_dict(ck["model"])
    with torch.no_grad():
        big = net(TR.batch_of(d, idx[:64]))
        again = net(TR.batch_of(d, idx[:64]))
        worst = 0.0
        for i in (0, 7, 33, 63):
            one = net(TR.batch_of(d, idx[i:i + 1]))[0]
            f = torch.isfinite(one)
            worst = max(worst, (one[f] - big[i][f]).abs().max().item())
    rep = (big[torch.isfinite(big)] - again[torch.isfinite(again)]).abs().max().item()
    print("3. hash: rerun max diff %.2e, alone vs in a batch of 64 max diff %.2e" % (rep, worst))
    ok &= rep == 0.0 and worst < 1e-4

    # 4. the streams' splits differ
    cells = torch.from_numpy(d["cells"][:64].astype(np.int64))
    anchor = torch.arange(200).expand(64, 200)
    agree = np.mean([(MT.hash_side(cells, anchor, st_, 0) == MT.hash_side(cells, anchor, st_, 1)).float().mean().item() for st_ in range(1, 6)])
    print("4. hash: share of cells on the same side in stream 0 and stream 1: %.2f" % agree)
    ok &= 0.45 < agree < 0.55

    # 5. CUDA = CPU (only when asked: never share the GPU with a training run)
    if "--cuda" in sys.argv and torch.cuda.is_available():
        cells = torch.from_numpy(d["cells"][:256].astype(np.int64))
        anchor = torch.arange(200).expand(256, 200)
        same = all(bool((MT.hash_side(cells, anchor, st_, s_) == MT.hash_side(cells.cuda(), anchor.cuda(), st_, s_).cpu()).all())
                   for st_ in range(1, 6) for s_ in (0, 1))
        net.cuda()
        with torch.no_grad():
            g = net(TR.batch_of(d, idx[:64], "cuda")).cpu()
        net.cpu()
        f = torch.isfinite(big)
        gd = (g[f] - big[f]).abs().max().item()
        print("5. hash sides CUDA == CPU: %s; logits max diff %.2e" % (same, gd))
        ok &= same and gd < 1e-3 and bool((torch.isfinite(g) == f).all())

    ok &= greedy_checks(a, ck, d, idx)
    ok &= pure_checks(a, ck, d, idx)
    print("ALL OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
