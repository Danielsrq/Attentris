"""Checks the plan labels (plan_label.sh) against the records they came from: one plan record per
input line, and move 1 of the plan = the record's labelled move (`chosen`). Move 1 should match
almost always when the plan was searched by the same teacher and beam as the label; a low rate
means the wrong weights or beam for that set.

    python attentris/plan_verify.py raw raw_1m dagger4/raw_tr ...     (sets relative to attentris/data)
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("ATT_DATA", os.path.join(HERE, "data"))     # another data dir (tests)


def cells_key(hold, cells):
    return int(hold), tuple(sorted(zip(cells[0::2], cells[1::2])))


def main():
    bad = False
    for s in sys.argv[1:]:
        src_files = sorted(glob.glob(os.path.join(DATA, s, "*.jsonl")))
        n = lines = match = empty = missing = 0
        lengths = {}
        for f in src_files:
            pf = os.path.join(DATA, "plan", s, os.path.basename(f))
            src = [l for l in open(f) if l.strip()]
            lines += len(src)
            if not os.path.exists(pf):
                missing += 1
                continue
            plans = [json.loads(l) for l in open(pf)]
            if [p["i"] for p in plans] != list(range(len(src))):
                print("  %s: plan records do not line up with the input lines" % pf)
                bad = True
                continue
            for line, p in zip(src, plans):
                r = json.loads(line)
                n += 1
                lengths[len(p["plan"])] = lengths.get(len(p["plan"]), 0) + 1
                if not p["plan"]:
                    empty += 1
                    continue
                c = r["cands"][r["chosen"]]
                match += cells_key(c[0], c[8]) == cells_key(p["plan"][0][0], p["plan"][0][3])
        rate = match / max(1, n - empty)
        bad |= missing > 0 or rate < 0.9
        print("%-20s %8d states, %d files missing, move 1 = label %.2f%%, no move %d, plan lengths %s" % (
            s, n, missing, 100 * rate, empty, dict(sorted(lengths.items()))))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
