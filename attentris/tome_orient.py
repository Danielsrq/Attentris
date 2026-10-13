"""Does ToMe merge horizontally more than vertically, and why? For validation positions:

1. every merge decision, by the direction of contact between the two blobs (H = side by side,
   V = one above the other, mixed): how many adjacent pairs are eligible under the A/B split (one in
   A, one in B), their key cosine, and how often they merge; and for A tokens with both an eligible H
   and an eligible V neighbour, how often they pick the V one (eligibility controlled);
2. a position code in the keys at the first merge (single cells): key cosine of two empty cells by
   their row / column offset, and the share of key variance explained by the row vs by the column;
3. cohesion at the last layer: for each internal empty-empty edge of a rule token (well: its hole column), how often its
   two cells end in the same blob (stream 0, stream 1, either), H and V edges separately; plus all
   empty-empty and filled-filled edges as a reference;
4. TSD slots: does the stem join the row of 3? split by whether the current / hold T also has a
   T-spin single ("|-") using the stem; best IoU of each stream's blobs with the T shape and with the
   |- shape.

--split own|token|hash|col|random re-runs with the A/B split after the first merge as the model was
trained, in token order (row-major, <= v4.1), by a fixed random bit per (step, stream, first cell), in
column-major order or random (model_tome.SPLIT_MODE), with policy top-1 on the same
positions so the cost of the off-distribution split is visible.

    attentris/.venv/Scripts/python attentris/tome_orient.py --model attentris/data/model_tome_v41.pt --split token,col,random
"""
import argparse
import collections
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M
import model_tome as MT
import pcores
import tokenizer as T
import train as TR

HERE = os.path.dirname(os.path.abspath(__file__))
T_ID = T.PIECES.index("T")
KINDS = ("well", "tsd", "pit1", "pit2", "fit_t covered", "fit_t open", "play")


def contacts(mask):
    """mask (B, N, 200) -> H, V (B, N, N): number of side-by-side / one-above-the-other cell contacts."""
    B, N, _ = mask.shape
    g = mask.view(B, N, 20, 10)
    h = torch.einsum("bnrc,bmrc->bnm", g[..., :, :-1], g[..., :, 1:])
    v = torch.einsum("bnrc,bmrc->bnm", g[..., :-1, :], g[..., 1:, :])
    return h + h.transpose(1, 2), v + v.transpose(1, 2)


class Stats:
    def __init__(self):
        self.pair = collections.defaultdict(lambda: np.zeros(5))     # (layer, class) -> n, eligible, cos sum, merged, merged&eligible
        self.choice = collections.defaultdict(lambda: np.zeros(2))   # layer -> A tokens with both options, picked V
        self.keys = []                                                # first merge: (stream, row, col, filled, key)
        self.layer = 0

    def hook(self, cells):
        def f(e):
            self.layer += 1
            L = self.layer
            mask, cand, stream, S = e["mask"], e["cand"], e["stream"], e["S"]
            A, Bs, sel, dst = e["A"], e["Bs"], e["sel"], e["dst"]
            if A is None:                              # greedy matching: no sides, every adjacent pair is eligible
                A = Bs = cand
            H, V = contacts(mask)
            same = (stream[:, :, None] == stream[:, None, :]) & cand[:, :, None] & cand[:, None, :]
            iu = torch.triu(torch.ones_like(same[0]), 1)[None]
            pairs = same & iu & ((H + V) > 0)
            elig = (A[:, :, None] & Bs[:, None, :]) | (Bs[:, :, None] & A[:, None, :])
            N = mask.shape[1]
            m = torch.zeros_like(same)
            bi, ni = torch.nonzero(sel, as_tuple=True)
            m[bi, ni, dst[bi, ni]] = True
            m = m | m.transpose(1, 2)
            single = (e["size"] == 1)
            filled = (mask * cells[:, None, :]).sum(-1) > 0.5
            for cls, cm in (("H", (H > 0) & (V == 0)), ("V", (V > 0) & (H == 0)), ("mixed", (H > 0) & (V > 0))):
                for sub, sm in (("all", None), ("empty-empty singles", single[:, :, None] & single[:, None, :] & ~filled[:, :, None] & ~filled[:, None, :])):
                    p = pairs & cm if sm is None else pairs & cm & sm
                    a = self.pair[(L, cls, sub)]
                    a += [p.sum().item(), (p & elig).sum().item(), S[p].sum().item(), (p & m).sum().item(), (p & m & elig).sum().item()]
            # eligibility-controlled choice: A tokens with an eligible H-only and an eligible V-only neighbour
            ok = same & (A[:, :, None] & Bs[:, None, :]) & ((H + V) > 0)
            hv = ok & (H > 0) & (V == 0)
            vv = ok & (V > 0) & (H == 0)
            both = hv.any(-1) & vv.any(-1) & sel
            if both.any():
                bi, ni = torch.nonzero(both, as_tuple=True)
                picked = dst[bi, ni]
                isv = vv[bi, ni, picked]
                self.choice[L] += [len(bi), isv.sum().item()]
            if e["first"]:
                k = e["keys"]
                Bn, N, d = k.shape
                heads = 4
                kh = torch.nn.functional.normalize(k.view(Bn, N, heads, d // heads), dim=-1)
                ks = kh.gather(2, (stream % heads)[:, :, None, None].expand(Bn, N, 1, d // heads)).squeeze(2)
                one = cand & single
                cell = mask.argmax(-1)
                bi, ni = torch.nonzero(one, as_tuple=True)
                c = cell[bi, ni]
                self.keys.append(np.concatenate([
                    bi.cpu().numpy()[:, None] + self.batch0, stream[bi, ni].cpu().numpy()[:, None], (c // 10).cpu().numpy()[:, None],
                    (c % 10).cpu().numpy()[:, None], filled[bi, ni].cpu().numpy()[:, None], ks[bi, ni].cpu().numpy()], 1))
        return f


def position_code(K):
    """K: rows (pos, stream, row, col, filled, key...). Cosine by (|dr|, |dc|) for empty-empty pairs in
    the same position and stream; eta^2 of row and of column over the key dimensions."""
    out = []
    for st in np.unique(K[:, 1]).astype(int):
        S = K[(K[:, 1] == st) & (K[:, 4] < 0.5)]
        cos = collections.defaultdict(list)
        for pos in np.unique(S[:, 0])[:600]:
            P = S[S[:, 0] == pos]
            if len(P) < 2:
                continue
            kk = P[:, 5:]
            C = kk @ kk.T
            dr = np.abs(P[:, 2][:, None] - P[:, 2][None])
            dc = np.abs(P[:, 3][:, None] - P[:, 3][None])
            for a, b in ((1, 0), (0, 1), (2, 0), (0, 2), (3, 0), (0, 3), (1, 1)):
                sel = (dr == a) & (dc == b)
                cos[(a, b)] += list(C[sel])
        kk = S[:, 5:]
        tot = ((kk - kk.mean(0)) ** 2).sum()
        def eta(col):
            v = 0.0
            for g in np.unique(S[:, col]):
                x = kk[S[:, col] == g]
                v += len(x) * ((x.mean(0) - kk.mean(0)) ** 2).sum()
            return v / tot
        out.append((st, len(S), {k: (np.mean(v), len(v)) for k, v in cos.items() if v}, eta(2), eta(3)))
    return out


def run(net, d, idx, device, split):
    MT.SPLIT_MODE = None if split == "own" else split
    st = Stats()
    coh = collections.defaultdict(lambda: np.zeros(4))     # (kind, H/V) -> edges, same s0, same s1, same either
    tsd = collections.defaultdict(lambda: np.zeros(6))     # alt? -> n, stem joined s0, s1, either, whole T one blob (either), n with T piece
    shape = collections.defaultdict(lambda: np.zeros(5))   # alt? -> n, IoU T s0, IoU T s1, IoU |- s0, IoU |- s1
    top1 = [0, 0]
    torch.manual_seed(0)
    for s in range(0, len(idx), 128):
        part = idx[s:s + 128]
        b = TR.batch_of(d, part, device)
        st.batch0 = s
        st.layer = 0
        MT.MERGE_HOOK = st.hook(b["cells"].float().view(len(part), 200))
        with torch.no_grad():
            logit, info = net(b, return_attn=True)
        MT.MERGE_HOOK = None
        st.layer = 0
        top1[0] += int((logit.argmax(-1).cpu().numpy() == d["chosen"][part]).sum())
        top1[1] += len(part)
        last = info["merges"][-1]
        mk = last["mask"].cpu().numpy() > 0.5
        al = last["alive"].cpu().numpy()
        sm = last["stream"].cpu().numpy()
        cm = b["cand_meta"].cpu().numpy()
        cc = b["cand_cells"].cpu().numpy()
        for k, i in enumerate(part):
            lab = -np.ones((2, 200), int)                               # blob id per cell per stream
            for j in np.nonzero(al[k])[0]:
                if mk[k, j].any():
                    lab[sm[k, j] % 2, mk[k, j]] = j
            filled = d["cells"][i].reshape(200) > 0
            def same(a, c):
                s0 = lab[0, a] >= 0 and lab[0, a] == lab[0, c]
                s1 = lab[1, a] >= 0 and lab[1, a] == lab[1, c]
                return s0, s1
            # reference: every empty-empty / filled-filled edge below row 19
            for a in range(200):
                r, c = divmod(a, 10)
                for o, nb in (("H", a + 1 if c < 9 else -1), ("V", a + 10 if r < 19 else -1)):
                    if nb < 0 or filled[a] != filled[nb]:
                        continue
                    if lab[0, a] < 0 and lab[1, a] < 0:
                        continue
                    s0, s1 = same(a, nb)
                    coh[("(all) " + ("filled" if filled[a] else "empty"), o)] += [1, s0, s1, s0 or s1]
            masks = M.unpack_masks(d["tok_mask"][i])
            Tcands = [(cm[k, j], cc[k, j]) for j in range(cm.shape[1]) if cm[k, j, 0] == T_ID and cm[k, j, 2] > 0 and cc[k, j, 0] >= 0]
            for j, kd in enumerate(d["tok_kind"][i]):
                if kd < 0 or not masks[j].any():
                    continue
                kind = T.KINDS[kd]
                if kind in T.SLOT_KINDS:
                    kind = kind + (" covered" if d["tok_extra"][i, j, 7] > 0.5 else " open")
                if kind not in KINDS:
                    continue
                mm = masks[j].reshape(200)
                cells_ = np.nonzero(mm)[0]
                for a in cells_:
                    r, c = divmod(a, 10)
                    for o, nb in (("H", a + 1 if c < 9 else -1), ("V", a + 10 if r < 19 else -1)):
                        if nb >= 0 and mm[nb] and not filled[a] and not filled[nb]:
                            s0, s1 = same(a, nb)
                            coh[(kind, o)] += [1, s0, s1, s0 or s1]
                if kind == "tsd" and len(cells_) == 4:
                    rows = cells_ // 10
                    stem = cells_[rows == rows.min()]
                    if len(stem) != 1:
                        continue
                    stem = stem[0]
                    centre = stem + 10
                    alts = [cl for mt, cl in Tcands if mt[3] == 1 and stem in cl]
                    hasT = any(cm[k, j2, 0] == T_ID for j2 in range(cm.shape[1]))
                    top = stem + 20
                    topk = "centre's top open" if top < 200 and not filled[top] else "centre's top covered"
                    key = "no T piece" if not hasT else ("|- TSS also available" if alts else "no |- TSS")
                    key = key + ", " + topk
                    s0, s1 = same(stem, centre)
                    whole = any(len(set(lab[q, cells_])) == 1 and lab[q, cells_[0]] >= 0 for q in (0, 1))
                    tsd[key] += [1, s0, s1, s0 or s1, whole, hasT]
                    if alts:
                        alt = np.zeros(200, bool)
                        alt[alts[0][alts[0] >= 0]] = True
                        def best(q, target):
                            v = 0.0
                            for bid in set(lab[q][lab[q] >= 0]):
                                bm = lab[q] == bid
                                v = max(v, (bm & target).sum() / (bm | target).sum())
                            return v
                        shape[key] += [1, best(0, mm), best(1, mm), best(0, alt), best(1, alt)]
        print("  %s: %d / %d positions" % (split, min(s + 128, len(idx)), len(idx)), flush=True)
    return st, coh, tsd, shape, top1


def report(split, st, coh, tsd, shape, top1, f):
    p = lambda *a: print(*a, file=f, flush=True) or print(*a, flush=True)
    p("\n================ split after the first merge: %s   (policy top-1 on these positions %.3f)" % (split, top1[0] / top1[1]))
    p("\n1. adjacent same-stream pairs at each merge (layer = merge step; step 1 = model layer 2, checkerboard split)")
    p("   step class  subset                   pairs   eligible   key cos   merged   merged|eligible")
    for (L, cls, sub), a in sorted(st.pair.items()):
        if a[0] == 0:
            continue
        p("   %d    %-5s  %-22s %8d    %5.1f%%     %5.3f    %5.1f%%     %5.1f%%" % (
            L, cls, sub, a[0], 100 * a[1] / a[0], a[2] / a[0], 100 * a[3] / a[0], 100 * a[4] / max(a[1], 1)))
    p("\n   A tokens that merged and had both an eligible H-only and an eligible V-only neighbour: share that picked V")
    for L, a in sorted(st.choice.items()):
        p("   step %d: %6d tokens, picked V %.1f%%" % (L, a[0], 100 * a[1] / max(a[0], 1)))
    if st.keys:
        K = np.concatenate(st.keys)
        p("\n2. position code in the keys at the first merge (empty single cells; each stream with its own head)")
        for s_, n, cos, er, ec in position_code(K):
            p("   stream %d (%d cells): share of key variance explained by row %.3f, by column %.3f" % (s_, n, er, ec))
            p("     mean key cosine by offset: " + "  ".join("dr%d dc%d %.3f" % (a, b, cos[(a, b)][0]) for a, b in sorted(cos)))
    p("\n3. cohesion at the last layer: share of a token's internal empty-empty edges whose two cells share a blob")
    p("   kind                 dir   edges   stream0  stream1   either")
    for (kind, o), a in sorted(coh.items()):
        p("   %-20s %-3s %8d    %5.1f%%   %5.1f%%   %5.1f%%" % (kind, o, a[0], 100 * a[1] / a[0], 100 * a[2] / a[0], 100 * a[3] / a[0]))
    p("\n4. TSD slots: stem joined to the centre of the row of 3 / whole T in one blob")
    p("   case                     n    stem s0   stem s1   stem either   whole T (either)")
    for key, a in sorted(tsd.items()):
        p("   %-44s %6d    %5.1f%%    %5.1f%%    %5.1f%%        %5.1f%%" % (key, a[0], 100 * a[1] / a[0], 100 * a[2] / a[0], 100 * a[3] / a[0], 100 * a[4] / a[0]))
    for key, a in sorted(shape.items()):
        p("   %s: best IoU with the T shape s0 %.2f s1 %.2f; with the |- shape s0 %.2f s1 %.2f   (n %d)" % (
            key, a[1] / a[0], a[2] / a[0], a[3] / a[0], a[4] / a[0], a[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "data", "model_tome_v41.pt"))
    ap.add_argument("--data", default=os.path.join(HERE, "data", "v51", "val.npz"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--variants", default="", help="';'-separated model_tome.apply_variant specs (e.g. 'none;split=greedy,pure,6:x2'); replaces --split")
    ap.add_argument("--split", default="own,col,random", help="own = the model's own split (token for <= v4.1); token / hash / col / random override it")
    ap.add_argument("--out", default=os.path.join(HERE, "results_tome_orient.txt"))
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
    with open(args.out, "a") as f:
        print("\n######## %s, %d positions of %s" % (os.path.basename(args.model), len(idx), os.path.basename(args.data)), file=f)
        for item in (args.variants.split(";") if args.variants else args.split.split(",")):
            if args.variants:
                MT.apply_variant(net, item)
            report(item, *run(net, d, idx, args.device, "own" if args.variants else item), f)
        MT.apply_variant(net, "none")


if __name__ == "__main__":
    main()
