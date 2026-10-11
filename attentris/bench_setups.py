"""How much does the setup overhang filter (`bot expand` with "setup_filter": true) save, and what
does it miss? Expands the same states with and without it (single process, one at a time) and
compares the setup tokens.

    python attentris/bench_setups.py --states data/v5_d1_states.jsonl data/dagger3/v51_dagger_ft_g20b.jsonl --n 1000
"""
import argparse
import json
import os
import random
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BOT = os.path.join(HERE, "..", "bot", "target", "release", "bot.exe")


def run(states, filt, slots=True):
    p = subprocess.Popen([BOT, "expand", "--mode", "normal"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    out, t = [], []
    for s in states:
        q = {k: s[k] for k in ("rows", "piece", "queue", "hold", "b2b")}
        q["slots"] = slots
        if filt:
            q["setup_filter"] = True
        t0 = time.perf_counter()
        p.stdin.write(json.dumps(q) + "\n")
        p.stdin.flush()
        while True:
            line = p.stdout.readline()
            if line.startswith("{"):
                break
        t.append(time.perf_counter() - t0)
        r = json.loads(line)
        out.append(sorted(json.dumps(x, sort_keys=True) for x in r["structures"] if x.get("kind", "").startswith("setup_")))
    p.stdin.close()
    p.wait()
    return out, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=1000, help="states per file")
    args = ap.parse_args()
    rng = random.Random(0)
    for f in args.states:
        allst = [json.loads(l) for l in open(f)]
        st = rng.sample(allst, min(args.n, len(allst)))
        base, tb = run(st, False)
        filt, tf = run(st, True)
        _, tn = run(st, False, slots=False)
        print("%s: expand with no slot / setup tokens at all (v4-style) %.2f ms/state" % (os.path.basename(f), 1000 * sum(tn) / len(tn)))
        nb = sum(len(x) for x in base)
        nf = sum(len(x) for x in filt)
        missed = sum(len(set(b) - set(a)) for b, a in zip(base, filt))
        extra = sum(len(set(a) - set(b)) for b, a in zip(base, filt))
        lost_pos = sum(1 for b, a in zip(base, filt) if set(b) - set(a))
        print("%s (%d states): expand %.2f ms/state without the filter, %.2f ms with (%.1fx); setup tokens %d -> %d "
              "(missed %d, %.1f%%; new %d); positions with any setup missed %d (%.1f%%)" % (
                  os.path.basename(f), len(st), 1000 * sum(tb) / len(tb), 1000 * sum(tf) / len(tf), sum(tb) / max(1e-9, sum(tf)),
                  nb, nf, missed, 100 * missed / max(1, nb), extra, lost_pos, 100 * lost_pos / len(st)), flush=True)


if __name__ == "__main__":
    main()
