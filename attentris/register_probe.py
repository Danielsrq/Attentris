"""Linear probes on register tokens (model.py --registers R): what do the registers hold?

collect: run the model on positions and save, per position, the state of every register after each self-attention
layer (R x 128 per layer), the mean of the real board tokens after each layer (128, for comparison), simple board
features, the game seed (for a split by game) and labels.
  --source val   the held-out validation files (val, val_dagger2, val_danger, val_dagger4, val_dagger5all)
  --source play  the model's own games (probe_games.sh: play.py --dump + --out), every --stride-th position
Labels (1 / 0; -1 = unknown, left out of that probe):
  board now   hot (height >= 12 or holes >= 6, the --style auto rule), height >= 12, holes >= 6,
              T-structure token present (tsd / stsd / cspin / imperial / fractal / vertical)
  own move    the model's own top move (not the teacher's): a T-spin, clears lines, uses hold, sends attack
  future      play only: the game ends within 10 / 30 moves (survived games: unknown in their last 30 moves)
probe: logistic regression (linear, no hidden layer, L2), fitted on 80% of the games and scored by AUC on the other
20% (split by game seed). Feature sets per layer: each register alone, the 4 registers together, the board-token mean,
and (baseline) 6 board features (max / mean height, holes, bumpiness, deepest well, filled cells). A random-label
control shows how far a probe gets on noise. Before layer 1 the registers are learned constants (no information).
    <python with torch> attentris/register_probe.py collect --source val
    <python with torch> attentris/register_probe.py collect --source play
    <python with torch> attentris/register_probe.py probe
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model as M  # noqa: E402
import pcores  # noqa: E402
import tokenizer as T  # noqa: E402

PROBE = os.path.join(HERE, "data", "probe")
MODEL = os.path.join(HERE, "data", "model_v51_d128_reg4_dagger5b_ft.pt")
VALS = ("val.npz", "val_dagger2.npz", "val_danger.npz", "val_dagger4.npz", "val_dagger5all.npz")
T_KINDS = [T.KINDS.index(k) for k in ("tsd", "stsd", "cspin", "imperial", "fractal", "vertical")]
T_PIECE = "IOTSZJL".index("T")
TARGETS = ("hot", "height12", "holes6", "tslot", "own_tspin", "own_clear", "own_hold", "own_attack", "die10", "die30")


def board_feats(b):
    """(20, 10) bool, row 0 at the bottom -> max height, mean height, holes, bumpiness, deepest well, filled cells."""
    h = T.heights(b).astype(np.float32)
    nb = np.concatenate([[20.0], h, [20.0]])
    well = np.maximum(0.0, np.minimum(nb[:-2], nb[2:]) - h).max()
    return np.array([h.max(), h.mean(), T.holes_of(b), np.abs(np.diff(h)).sum(), well, b.sum()], np.float32)


def load_net(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    net = M.AtTenTris(d=a["d"], layers=a["layers"], n_kinds=ck["model"]["kind.weight"].shape[0],
                      n_teachers=ck["model"]["style.weight"].shape[0], registers=a.get("registers", 0))
    net.load_state_dict(ck["model"])
    return net.eval()


def run_batch(net, b):
    """Register states (L, B, R, d), board means (L, B, d), own top move's meta (B, 5) for one batch."""
    net.reg_record = {}
    with torch.inference_mode():
        logit = net(b)
    L = len(net.blocks)
    regs = torch.stack([torch.cat(net.reg_record[li]) for li in range(L)]).numpy().astype(np.float16)
    brd = torch.stack([torch.cat(net.reg_record[("board", li)]) for li in range(L)]).numpy().astype(np.float16)
    net.reg_record = None
    top = logit.argmax(-1)
    meta = b["cand_meta"][torch.arange(len(top)), top].numpy()
    tslot = np.isin(b["tok_kind"].numpy(), T_KINDS).any(1)
    return regs, brd, meta, tslot


def labels_from(cells, meta, tslot):
    n = len(cells)
    feats = np.stack([board_feats(cells[i].reshape(20, 10).astype(bool)) for i in range(n)])
    lab = {"height12": feats[:, 0] >= 12, "holes6": feats[:, 2] >= 6, "tslot": tslot,
           "own_tspin": (meta[:, 0] == T_PIECE) & (meta[:, 2] > 0), "own_clear": meta[:, 3] > 0,
           "own_hold": meta[:, 1] > 0, "own_attack": meta[:, 4] > 0}
    lab["hot"] = lab["height12"] | lab["holes6"]
    return feats, {k: v.astype(np.int8) for k, v in lab.items()}


def collect_val(args):
    import train as TR
    net = load_net(args.model)
    torch.set_num_threads(pcores.pin(verbose=False) or 8)
    parts = []
    for vf in VALS:
        d = TR.load(os.path.join(HERE, "data", "v51", vf))
        n = len(d["chosen"])
        for s in range(0, n, 512):
            idx = np.arange(s, min(n, s + 512))
            b = TR.batch_of(d, idx, "cpu")
            regs, brd, meta, tslot = run_batch(net, b)
            feats, lab = labels_from(d["cells"][idx], meta, tslot)
            parts.append((regs, brd, feats, lab, d["seed"][idx].astype(np.int64), np.full(len(idx), VALS.index(vf), np.int8)))
        print("%s: %d positions" % (vf, n), flush=True)
    save(args, parts, "val")


def _play_worker(job):
    states, model = job
    pcores.pin(verbose=False)
    import play as PL
    torch.set_num_threads(1)
    pl = PL.Player(model, 1, 8, PL.STYLES["normal"], "normal")
    out = []
    for s0 in range(0, len(states), 64):
        chunk = states[s0:s0 + 64]
        pairs = []
        for st in chunk:
            root = pl.expand(st)
            if root["cands"]:
                pairs.append((st, root))
        if not pairs:
            continue
        a = PL.arrays(pairs, pl.teacher, pl.tok_version, pl.drop)
        regs, brd, meta, tslot = run_batch(pl.net, a)
        cells = a["cells"].numpy()
        feats, lab = labels_from(cells, meta, tslot)
        extra = np.array([[st["seed"], st["n"]] for st, _ in pairs], np.int64)
        out.append((regs, brd, feats, lab, extra))
    pl.close()
    return out


def collect_play(args):
    results = {}
    states = []
    for name in ("g20", "g50"):
        for line in open(os.path.join(PROBE, "play_%s.games" % name)):
            if line.startswith("{"):
                r = json.loads(line)
                results[r["seed"]] = (bool(r["died"]), int(r["pieces"]))
        for k, line in enumerate(open(os.path.join(PROBE, "play_%s.jsonl" % name))):
            st = json.loads(line)
            if st["n"] % args.stride == 0:
                states.append(st)
    jobs = pcores.pin(verbose=False) or 8
    chunks = [states[i::jobs] for i in range(jobs)]
    print("%d positions from %d games, %d workers" % (len(states), len(results), jobs), flush=True)
    with mp.get_context("spawn").Pool(jobs) as pool:
        outs = pool.map(_play_worker, [(c, args.model) for c in chunks])
    parts = []
    for out in outs:
        for regs, brd, feats, lab, extra in out:
            seed, n = extra[:, 0], extra[:, 1]
            for K in (10, 30):
                y = np.full(len(seed), -1, np.int8)
                for i in range(len(seed)):
                    died, pieces = results[int(seed[i])]
                    left = pieces - n[i]
                    if died:
                        y[i] = 1 if left <= K else 0
                    elif left > K:
                        y[i] = 0
                lab["die%d" % K] = y
            parts.append((regs, brd, feats, lab, seed, np.full(len(seed), 9, np.int8)))
    save(args, parts, "play")


def save(args, parts, name):
    lab = {k: np.concatenate([p[3].get(k, np.full(len(p[4]), -1, np.int8)) for p in parts]) for k in TARGETS}
    np.savez(os.path.join(PROBE, "feat_%s.npz" % name), regs=np.concatenate([p[0] for p in parts], 1),
             board=np.concatenate([p[1] for p in parts], 1), feats=np.concatenate([p[2] for p in parts]),
             seed=np.concatenate([p[4] for p in parts]), src=np.concatenate([p[5] for p in parts]), **{"y_" + k: v for k, v in lab.items()})
    print("saved %s: %d positions" % (name, len(lab["hot"])), flush=True)


def auc(score, y):
    order = np.argsort(score)
    r = np.empty(len(score))
    r[order] = np.arange(1, len(score) + 1)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0)


def fit_probe(Xtr, ytr, Xte, dev, l2=1e-3, steps=150):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = torch.from_numpy((Xtr - mu) / sd).float().to(dev)
    Xte = torch.from_numpy((Xte - mu) / sd).float().to(dev)
    y = torch.from_numpy(ytr).float().to(dev)
    w = torch.zeros(Xtr.shape[1], device=dev, requires_grad=True)
    b0 = torch.zeros(1, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([w, b0], lr=1.0, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xtr @ w + b0, y) + l2 * (w * w).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return (Xte @ w + b0).cpu().numpy()


def probe(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = open(os.path.join(HERE, "results_probes.txt"), "a")

    def say(s=""):
        print(s, flush=True)
        out.write(s + "\n")
    say("== register probes on %s (AUC on held-out games; 0.5 = chance)" % os.path.basename(args.model))
    for name in ("val", "play"):
        path = os.path.join(PROBE, "feat_%s.npz" % name)
        if not os.path.exists(path):
            continue
        z = np.load(path)
        regs, brd, feats, seed = z["regs"], z["board"], z["feats"], z["seed"]
        L, N, R, d = regs.shape
        test = (seed * 2654435761 % 1000) < 200
        say("-- %s positions: %d (%d test), layers %d, registers %d" % (name, N, test.sum(), L, R))
        say("%-12s %6s %5s  %s" % ("target", "pos%", "n", "layer: reg1  reg2  reg3  reg4 | all regs | board mean   (board features)"))
        for t in TARGETS:
            y = z["y_" + t]
            ok = y >= 0
            if ok.sum() < 200 or y[ok].min() == y[ok].max():
                continue
            tr, te = ok & ~test, ok & test
            if (y[te] == 1).sum() < 20 or (y[te] == 0).sum() < 20:
                continue
            base = auc(fit_probe(feats[tr], y[tr], feats[te], dev), y[te])
            rows = []
            for li in range(L):
                cols = [auc(fit_probe(regs[li, tr, r].astype(np.float32), y[tr], regs[li, te, r].astype(np.float32), dev), y[te])
                        for r in range(R)]
                allr = auc(fit_probe(regs[li, tr].reshape(tr.sum(), -1).astype(np.float32), y[tr],
                                     regs[li, te].reshape(te.sum(), -1).astype(np.float32), dev), y[te])
                bm = auc(fit_probe(brd[li, tr].astype(np.float32), y[tr], brd[li, te].astype(np.float32), dev), y[te])
                rows.append("L%d: %s | %.3f | %.3f" % (li + 1, " ".join("%.3f" % c for c in cols), allr, bm))
            say("%-12s %5.1f%% %5d  %s   (%.3f)" % (t, 100 * y[ok].mean(), ok.sum(), rows[0], base))
            for r_ in rows[1:]:
                say("%-12s %6s %5s  %s" % ("", "", "", r_))
        # control: shuffled labels of "hot" (what a probe reaches on noise), all registers at the last layer
        rng = np.random.default_rng(0)
        y = z["y_hot"].copy()
        rng.shuffle(y)
        tr, te = ~test, test
        c = auc(fit_probe(regs[-1, tr].reshape(tr.sum(), -1).astype(np.float32), y[tr],
                          regs[-1, te].reshape(te.sum(), -1).astype(np.float32), dev), y[te])
        say("control: shuffled 'hot' labels, all registers, last layer: AUC %.3f" % c)
    out.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("collect", "probe"))
    ap.add_argument("--source", choices=("val", "play"), default="val")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--stride", type=int, default=2, help="play: every n-th position of each game")
    args = ap.parse_args()
    os.makedirs(PROBE, exist_ok=True)
    if args.cmd == "collect":
        (collect_val if args.source == "val" else collect_play)(args)
    else:
        probe(args)


if __name__ == "__main__":
    main()
