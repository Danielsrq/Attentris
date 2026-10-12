"""Weights of several weight files side by side (normalised units), the tuner's default-frozen features
marked. Rows are features in the order of the first file; columns are NAME=FILE arguments.

    python bot/tools/weight_table.py normal=bot/weights/normal.txt survive=bot/results/survive/survive.txt ...
"""
import sys

FROZEN = {"max_height", "wounds", "ready_repeat", "sz_single", "sz_double", "sz_triple",
          "lj_single", "lj_double", "lj_triple", "i_single", "i_double", "i_triple"}
DEAD_IN_NORMAL = {"burst_armed", "combo"}      # always 0 under normal rules


def load(p):
    d, order = {}, []
    for l in open(p):
        f = l.split("#")[0].split()
        if len(f) == 2:
            try:
                d[f[0]] = float(f[1]); order.append(f[0])
            except ValueError:
                pass
    return d, order


def main():
    runs = [(a.split("=", 1)[0], load(a.split("=", 1)[1])) for a in sys.argv[1:]]
    order = []
    for _, (_, o) in runs:
        order += [k for k in o if k not in order]
    print("%-16s" % "feature" + "".join("%11s" % n for n, _ in runs) + "  note")
    for k in order:
        note = "frozen at 0 by the tuner" if k in FROZEN else ("no effect under normal rules" if k in DEAD_IN_NORMAL else "")
        print("%-16s" % k + "".join("%11.2f" % w.get(k, 0.0) for _, (w, _) in runs) + "  " + note)


if __name__ == "__main__":
    main()
