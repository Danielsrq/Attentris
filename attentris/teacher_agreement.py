"""How consistent is the teacher with itself? Relabel the validation games with a stronger
search (--beam 128 by default) and count how often its chosen move differs from the
original beam-64 label. That rate bounds how much top-1 agreement means: a student cannot
meaningfully agree with the teacher more often than the teacher agrees with itself.

    python attentris/teacher_agreement.py [--beam 128] [--games 20]
"""
import argparse
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pcores

HERE = os.path.dirname(os.path.abspath(__file__))
BOT = os.path.join(HERE, "..", "bot", "target", "release", "bot.exe")
W = {"normal": os.path.join(HERE, "..", "bot", "weights", "normal.txt"),
     "gate4": os.path.join(HERE, "..", "bot", "results", "gate4", "gate4.txt")}


def key(c):
    return (c[0], tuple(sorted(zip(c[8][0::2], c[8][1::2]))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(HERE, "data", "raw_v3"))
    ap.add_argument("--beam", type=int, default=128)
    ap.add_argument("--val-every", type=int, default=10)
    args = ap.parse_args()
    pcores.pin(verbose=False)
    out = {}
    for f in sorted(glob.glob(os.path.join(args.raw, "*.jsonl"))):
        seed = int(os.path.basename(f).rsplit("_s", 1)[1].split(".")[0])
        if seed % args.val_every:
            continue
        teacher = os.path.basename(f).split("_")[0]
        recs = {r["n"]: r for r in map(json.loads, open(f))}
        tmp = os.path.join(HERE, "data", "agree_%s_s%d.jsonl" % (teacher, seed))
        with open(f) as src:
            subprocess.run([BOT, "label", "--mode", "normal", "--weights", W[teacher], "--label", teacher,
                            "--threads", "16", "--beam", str(args.beam), "--out", tmp], stdin=src, check=True,
                           stderr=subprocess.DEVNULL)
        same = tot = 0
        for r2 in map(json.loads, open(tmp)):
            r1 = recs.get(r2["n"])
            if r1 is None:
                continue
            tot += 1
            same += key(r1["cands"][r1["chosen"]]) == key(r2["cands"][r2["chosen"]])
        os.remove(tmp)
        a = out.setdefault(teacher, [0, 0])
        a[0] += same
        a[1] += tot
    for t, (s, n) in out.items():
        print("%s teacher: beam %d picks the same move as beam 64 in %.1f%% of %d validation positions" % (t, args.beam, 100 * s / max(1, n), n))


if __name__ == "__main__":
    main()
