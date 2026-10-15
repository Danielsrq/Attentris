"""How long does one AtTenTris decision take, with no search? Pinned to the P-cores.

  1. forward pass only (positions already tokenized), batch 1 = per-move latency,
     at 1 / 4 / 16 torch threads
  2. batched throughput (batch 256), per position
  3. the per-move pipeline in Python: T-structure scan (a persistent `bot structures`
     process), tokenizer, packing to tensors, forward. Move generation is not included:
     the candidates come from the dataset (in play they would come from the Rust movegen).

    <python with torch> attentris/bench_infer.py [--model attentris/data/model.pt]
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M
import pack as P
import pcores
import tokenizer as T
import train as TR

HERE = os.path.dirname(os.path.abspath(__file__))


def ms(xs):
    xs = sorted(xs)
    return "median %.2f ms, p90 %.2f ms" % (1000 * statistics.median(xs), 1000 * xs[int(0.9 * len(xs))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "data", "model.pt"))
    ap.add_argument("--data", default=os.path.join(HERE, "data", "val.npz"))
    ap.add_argument("--n", type=int, default=400)
    args = ap.parse_args()
    n_p = pcores.pin()
    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    net = M.AtTenTris(d=ck["args"]["d"], layers=ck["args"]["layers"], n_kinds=ck["model"]["kind.weight"].shape[0])
    net.load_state_dict(ck["model"])
    net.eval()
    d = TR.load(args.data)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(d["chosen"]), size=args.n, replace=False)
    print("model: %d params; %d positions, mean %.1f candidates each" % (
        sum(p.numel() for p in net.parameters()), args.n, d["n_cand"][idx].mean()))

    # 1. batch-1 forward latency
    batches = [TR.batch_of(d, np.array([i])) for i in idx]
    for th in (1, 4, n_p or 16):
        torch.set_num_threads(th)
        with torch.inference_mode():
            for b in batches[:20]:
                net(b)
            ts = []
            for b in batches:
                t0 = time.perf_counter()
                net(b)
                ts.append(time.perf_counter() - t0)
        print("forward, batch 1, %2d threads: %s" % (th, ms(ts)))

    # 2. batched throughput
    torch.set_num_threads(n_p or 16)
    big = TR.batch_of(d, np.sort(rng.choice(len(d["chosen"]), size=256, replace=False)))
    with torch.inference_mode():
        net(big)
        t0 = time.perf_counter()
        for _ in range(5):
            net(big)
        dt = (time.perf_counter() - t0) / 5
    print("forward, batch 256, %d threads: %.1f ms per batch = %.3f ms per position" % (n_p or 16, 1000 * dt, 1000 * dt / 256))

    # 3. per-move pipeline from a raw board
    torch.set_num_threads(1)
    proc = subprocess.Popen([os.path.join(HERE, "..", "bot", "target", "release", "bot.exe" if os.name == "nt" else "bot"), "structures"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    raw = [json.loads(l) for l in open(os.path.join(HERE, "data", "raw", "gate4_s2010.jsonl"))][:args.n]
    t_struct, t_tok, t_pack, t_fwd = [], [], [], []
    with torch.inference_mode():
        for r in raw:
            t0 = time.perf_counter()
            b40 = P.rows_to_strings(r["rows"])
            proc.stdin.write(json.dumps({"board": b40}) + "\n")
            proc.stdin.flush()
            s = json.loads(proc.stdout.readline())["structures"]
            t_struct.append(time.perf_counter() - t0)
        for r in raw:
            b40 = P.rows_to_strings(r["rows"])
            proc.stdin.write(json.dumps({"board": b40}) + "\n")
            proc.stdin.flush()
            s = json.loads(proc.stdout.readline())["structures"]
            t0 = time.perf_counter()
            b = T.board_from_replay(b40)
            toks = T.tokenize(b, r["piece"], r["queue"], r["hold"], s)
            t1 = time.perf_counter()
            one = single_position_arrays(r, b, toks)
            batch = TR.batch_of(one, np.array([0]))
            t2 = time.perf_counter()
            net(batch)
            t3 = time.perf_counter()
            t_tok.append(t1 - t0); t_pack.append(t2 - t1); t_fwd.append(t3 - t2)
    proc.stdin.close(); proc.wait()
    tot = [a + b + c + e for a, b, c, e in zip(t_struct, t_tok, t_pack, t_fwd)]
    print("per move from a raw board (1 thread): structures %s | tokenizer %s | packing %s | forward %s" % (
        ms(t_struct), ms(t_tok), ms(t_pack), ms(t_fwd)))
    print("per move total: %s" % ms(tot))


def single_position_arrays(r, b, toks):
    """The arrays pack() writes, for one position, without the structure scan."""
    NT, NC, NE = P.NT, P.NC, P.NE
    out = {
        "cells": b.reshape(1, -1).astype(np.uint8),
        "tok_kind": np.full((1, NT), -1, np.int8), "tok_mask": np.zeros((1, NT, 25), np.uint8),
        "tok_box": np.full((1, NT, 4), -1, np.int8), "tok_extra": np.zeros((1, NT, NE), np.float16),
        "tok_piece": np.full((1, NT), -1, np.int8), "cand_cells": np.full((1, NC, 4), -1, np.int16),
        "cand_meta": np.zeros((1, NC, 5), np.int8), "teacher": np.array([P.TEACHERS.index(r["teacher"])], np.int8),
    }
    for j, t in enumerate(toks[:NT]):
        out["tok_kind"][0, j] = P.KIND_ID[t.kind]
        out["tok_mask"][0, j] = np.packbits(t.mask.reshape(-1))
        if t.box is not None:
            out["tok_box"][0, j] = t.box
        out["tok_extra"][0, j] = P.extras(t)
        if t.kind in T.PIECE_KINDS:
            p = t.extras["piece"]
            out["tok_piece"][0, j] = P.PIECE_ID[p] if p else 7
    for j, c in enumerate(r["cands"][:NC]):
        hold, piece, rot, spin, lines, attack, ply1, best, cells = c
        idx = []
        for q in range(4):
            x, y = cells[2 * q], cells[2 * q + 1]
            rr = 39 - y
            idx.append(rr * 10 + x if 0 <= rr < T.ROWS else -2)
        out["cand_cells"][0, j] = idx
        out["cand_meta"][0, j] = (P.PIECE_ID[piece], hold, spin, lines, min(int(attack), 127))
    return out


if __name__ == "__main__":
    main()
