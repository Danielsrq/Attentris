"""DAgger round 5, routing ("danger mixed into normal"): every selected position already carries
CEM-survival's label as the danger style (bot label --label danger). Positions from the student's
NORMAL-style games (--normal-seeds) are also routed into the normal style by the auto rule
(play.py --style auto: hot = stack >= 12 or >= 6 holes):
    hot  -> a copy of CEM-survival's record tagged "normal" (the normal style learns to survive)
    calm -> the state goes to --calm-out, to be labelled by normal.txt as "normal" (bot label)
so the normal style plays attack on calm boards and survival on hot ones without a style switch.

    python attentris/dagger5_route.py --labelled data/dagger5/labelled_surv.jsonl --selected data/dagger5/selected.jsonl \
        --normal-seeds 41001-41100,42001-42100 --out data/dagger5/routed.jsonl --calm-out data/dagger5/calm_states.jsonl
"""
import argparse
import json

H, W = 40, 10


def hot(rows, height=12, holes=6):
    tops = [next((y for y in range(H) if rows[y] >> x & 1), H) for x in range(W)]
    hmax = H - min(tops)
    nh = sum(1 for x in range(W) for y in range(tops[x] + 1, H) if not rows[y] >> x & 1)
    return hmax >= height or nh >= holes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labelled", required=True, help="CEM-survival labels of the selected positions (teacher danger)")
    ap.add_argument("--selected", required=True, help="the selected states (dagger_select.py)")
    ap.add_argument("--normal-seeds", required=True, help="seed ranges of the normal-style games, e.g. 41001-41100,42001-42100")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calm-out", required=True)
    args = ap.parse_args()
    ranges = [tuple(int(v) for v in r.split("-")) for r in args.normal_seeds.split(",")]
    normal = lambda s: any(a <= s <= b for a, b in ranges)
    n = {"danger": 0, "normal_hot": 0, "calm": 0}
    with open(args.out, "w") as out:
        for line in open(args.labelled):
            r = json.loads(line)
            out.write(line if line.endswith("\n") else line + "\n")
            n["danger"] += 1
            if normal(r["seed"]) and hot(r["rows"]):
                r["teacher"] = "normal"
                out.write(json.dumps(r) + "\n")
                n["normal_hot"] += 1
    with open(args.calm_out, "w") as out:
        for line in open(args.selected):
            s = json.loads(line)
            if normal(s["seed"]) and not hot(s["rows"]):
                out.write(line if line.endswith("\n") else line + "\n")
                n["calm"] += 1
    print("routed: %d danger records, %d hot normal-style positions copied as normal, %d calm normal-style states for normal.txt" % (
        n["danger"], n["normal_hot"], n["calm"]))


if __name__ == "__main__":
    main()
