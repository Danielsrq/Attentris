"""Score positions with the teacher's V: where does AtTenTris go that the teachers never do?

    python attentris/v_audit.py LABEL=FILE_OR_DIR [LABEL=...] [--weights bot/weights/normal.txt]

Files are JSONL states with seed, n, rows (40 bitmask rows), b2b: play.py --dump output or
`bot dataset` game files (a directory of them). The first LABEL is the reference (the
teacher); for each other player it reports V and the board features by game phase, how V
looks just before a top-out, and when each game first leaves the reference's range.
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pcores

HERE = os.path.dirname(os.path.abspath(__file__))
BOT = os.path.join(HERE, "..", "bot", "target", "release", "bot.exe" if os.name == "nt" else "bot")
NAMES = ["holes", "hole_rows", "cover", "bump", "bump_sq", "row_trans", "col_trans", "max_height", "danger"]
PHASES = [(0, 25), (25, 50), (50, 100), (100, 200), (200, 500)]
# eval::features divides by eval::SCALE; multiply back for raw units (holes, rows, cells)
RAW = {"holes": 10.0, "hole_rows": 20.0, "max_height": 20.0, "bump": 20.0}


def load(path):
    files = sorted(glob.glob(os.path.join(path, "*.jsonl"))) if os.path.isdir(path) else [path]
    out = []
    for f in files:
        for line in open(f):
            r = json.loads(line)
            out.append({"seed": r["seed"], "n": r["n"], "rows": r["rows"], "b2b": r["b2b"]})
    return out


def score(states, weights):
    inp = "".join(json.dumps({"rows": s["rows"], "b2b": s["b2b"]}) + "\n" for s in states)
    res = subprocess.run([BOT, "value", "--mode", "normal", "--weights", weights], input=inp, capture_output=True, text=True, check=True)
    rows = [json.loads(l) for l in res.stdout.splitlines() if l.startswith("{")]
    assert len(rows) == len(states)
    return np.array([r["v"] for r in rows]), np.array([r["f"] for r in rows]), np.array([r["c"] for r in rows])


def main():
    args = [a for a in sys.argv[1:] if "=" in a and not a.startswith("--")]
    weights = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--weights=")), os.path.join(HERE, "..", "bot", "weights", "normal.txt"))
    pcores.pin(verbose=False)
    eval_names = None
    data = {}
    for a in args:
        label, path = a.split("=", 1)
        st = load(path)
        v, f, c = score(st, weights)
        for name, k in RAW.items():
            f[:, NAMES.index(name)] *= k
        data[label] = (st, v, f, c)
    # feature indices (the first 9 features of eval::NAMES)
    idx = {n: i for i, n in enumerate(NAMES)}
    ref_label = args[0].split("=", 1)[0]
    ref_v = data[ref_label][1]
    p01, p05 = np.percentile(ref_v, 1), np.percentile(ref_v, 5)
    print("V = the normal teacher's evaluation (%s); higher is better. Reference: %s, V 5th pct %.1f, 1st pct %.1f\n" % (
        os.path.basename(weights), ref_label, p05, p01))
    print("%-22s %-10s %7s %7s %7s %6s %6s %6s %6s" % ("player", "pieces", "n", "V mean", "V p10", "height", "holes", "hrows", "bump"))
    for label, (st, v, f, c) in data.items():
        n = np.array([s["n"] for s in st])
        for lo, hi in PHASES:
            m = (n >= lo) & (n < hi)
            if m.sum() == 0:
                continue
            print("%-22s %-10s %7d %7.1f %7.1f %6.1f %6.2f %6.2f %6.1f" % (
                label, "%d-%d" % (lo, hi), m.sum(), v[m].mean(), np.percentile(v[m], 10), f[m, idx["max_height"]].mean(),
                f[m, idx["holes"]].mean(), f[m, idx["hole_rows"]].mean(), f[m, idx["bump"]].mean()))
        print()
    rn_all = np.array([s["n"] for s in data[ref_label][0]])
    phase_p01 = {ph: np.percentile(ref_v[(rn_all >= ph[0]) & (rn_all < ph[1])], 1) for ph in PHASES}
    phase_of = lambda k: next(ph for ph in PHASES if ph[0] <= k < ph[1])
    print("Games that topped out: the last 10 positions; and the first piece where V fell below the reference's 1st percentile for that game phase")
    for label, (st, v, f, c) in data.items():
        seeds = sorted({s["seed"] for s in st})
        last, first_out, lengths = [], [], []
        for sd in seeds:
            ids = [i for i, s in enumerate(st) if s["seed"] == sd]
            ids.sort(key=lambda i: st[i]["n"])
            lengths.append(len(ids))
            below = [st[i]["n"] for i in ids if v[i] < phase_p01[phase_of(st[i]["n"])]]
            first_out.append(below[0] if below else None)
            if len(ids) < 500:
                last.extend(ids[-10:])
        died = sum(1 for L in lengths if L < 500)
        fo = [x for x in first_out if x is not None]
        msg = "%-22s died %2d/%d" % (label, died, len(seeds))
        if last:
            last = np.array(last)
            msg += " | last 10 before top-out: V %.1f, height %.1f, holes %.1f" % (v[last].mean(), f[last, idx["max_height"]].mean(), f[last, idx["holes"]].mean())
        msg += " | left the reference range in %d/%d games%s" % (len(fo), len(seeds), (", median at piece %d" % np.median(fo)) if fo else "")
        print(msg)
    # which features make the difference (mean weight x feature), player vs reference, first 50 pieces
    print("\nWhere V is lost, pieces 0-50 (mean contribution w*f, player minus reference; most negative first)")
    names34 = None
    out = subprocess.run([BOT, "weights", "--weights", weights], capture_output=True, text=True).stdout
    names34 = [l.split()[0] for l in out.splitlines() if l and not l.startswith("#")][:data[ref_label][3].shape[1]]
    rs, rv, rf, rc = data[ref_label]
    rn = np.array([s["n"] for s in rs])
    for label, (st, v, f, c) in data.items():
        if label == ref_label:
            continue
        n = np.array([s["n"] for s in st])
        d = c[n < 50].mean(0) - rc[rn < 50].mean(0)
        order = np.argsort(d)[:6]
        print("%-22s V gap %+.1f: %s" % (label, d.sum(), ", ".join("%s %+.1f" % (names34[i], d[i]) for i in order)))


if __name__ == "__main__":
    main()
