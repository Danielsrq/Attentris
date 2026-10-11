"""APP and VS score from `bot play --game-log` files. VS here is TETR.IO's versus score without
the PPS factor: (lines sent + garbage cleared) / pieces. Reported per piece played (as TETR.IO) and
over the full piece budget (games x --pieces: a game that dies early scores 0 for the rest, as the
tuner's APPfull fitness).

    python attentris/vs_score.py --pieces 500 NAME=FILE ...
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="NAME=FILE")
    ap.add_argument("--pieces", type=int, default=500)
    args = ap.parse_args()
    print("%-34s %6s %8s %7s %8s %7s %8s %9s" % ("run", "died", "pieces", "APP", "APPfull", "VS", "VSfull", "gb clr"))
    for s in args.runs:
        name, path = s.split("=", 1)
        g = [json.loads(l) for l in open(path) if l.startswith("{")]
        pcs = sum(r["pieces"] for r in g)
        att = sum(r["attack"] for r in g)
        clr = sum(r["garbage_cleared"] for r in g)
        gin = sum(r["garbage_in"] for r in g)
        full = len(g) * args.pieces
        print("%-34s %3d/%-2d %8.0f %7.3f %8.3f %7.3f %8.3f %9s" % (
            name, sum(r["died"] for r in g), len(g), pcs / len(g), att / pcs, att / full, (att + clr) / pcs, (att + clr) / full,
            "%d/%d" % (clr, gin) if gin else "-"))


if __name__ == "__main__":
    main()
