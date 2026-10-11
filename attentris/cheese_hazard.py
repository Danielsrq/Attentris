"""Do deaths follow cheese? For games with logged garbage batches ([piece, rows, cheese]; bot play
--game-log or play.py --out with the play-ext "garbage" field), compare the death rate per piece
alive in the W pieces after a cheese batch with the rate elsewhere (a hazard ratio), and how many
games the runs lose on the same seeds.

    python attentris/cheese_hazard.py --runs "teacher=data/cheese/teacher.jsonl" "ft=data/cheese/dagger_ft.games"
"""
import argparse
import json


def load(path):
    games = {}
    for l in open(path):
        if l.startswith("{"):
            r = json.loads(l)
            games[r["seed"]] = r
    return games


def hazard(games, w):
    """Deaths and pieces alive inside / outside the W pieces after any cheese batch."""
    d_in = d_out = e_in = e_out = 0
    for r in games.values():
        end = r["pieces"]
        cheese = [g[0] for g in r["garbage"] if g[2]]
        inside = [any(c <= n < c + w for c in cheese) for n in range(end)]
        e_in += sum(inside); e_out += end - sum(inside)
        if r["died"]:
            # the death happens on the last piece played (or as garbage arrives)
            last = end - 1
            if any(c <= last < c + w for c in cheese) or any(c == end for c in cheese):
                d_in += 1
            else:
                d_out += 1
    return d_in, e_in, d_out, e_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="NAME=FILE")
    ap.add_argument("--windows", default="10,20,30,50")
    args = ap.parse_args()
    runs = [(s.split("=", 1)[0], load(s.split("=", 1)[1])) for s in args.runs]
    seeds = sorted(set.intersection(*(set(g) for _, g in runs)))
    print("%d common seeds; garbage every 10 pieces, 3-4 rows, 20%% of batches cheese" % len(seeds))
    for name, g in runs:
        g = {s: g[s] for s in seeds}
        died = [s for s in seeds if g[s]["died"]]
        print("\n== %s: survived %d/%d, pieces %.0f avg, APP %.3f" % (
            name, len(seeds) - len(died), len(seeds), sum(r["pieces"] for r in g.values()) / len(seeds),
            sum(r["attack"] for r in g.values()) / max(1, sum(r["pieces"] for r in g.values()))))
        for w in (int(x) for x in args.windows.split(",")):
            d_in, e_in, d_out, e_out = hazard(g, w)
            r_in, r_out = d_in / max(1, e_in), d_out / max(1, e_out)
            print("  within %2d pieces after cheese: %3d deaths / %6d pieces alive (%.1f%% of the time) -> %.2f per 1000; "
                  "elsewhere %3d / %6d -> %.2f per 1000; hazard ratio %s" % (
                      w, d_in, e_in, 100 * e_in / max(1, e_in + e_out), 1000 * r_in, d_out, e_out, 1000 * r_out,
                      "%.1f" % (r_in / r_out) if r_out > 0 else ("inf" if r_in > 0 else "-")))
        if died:
            gaps = []
            for s in died:
                ch = [c[0] for c in g[s]["garbage"] if c[2] and c[0] <= g[s]["pieces"]]
                gaps.append(g[s]["pieces"] - ch[-1] if ch else None)
            known = sorted(x for x in gaps if x is not None)
            print("  pieces from the last cheese batch to the death: median %s, quartiles %s / %s; %d deaths with no cheese before" % (
                known[len(known) // 2] if known else "-", known[len(known) // 4] if known else "-",
                known[3 * len(known) // 4] if known else "-", sum(x is None for x in gaps)))
        # cheese load early in the game vs survival
        k = 150
        rows = [(sum(c[1] for c in g[s]["garbage"] if c[2] and c[0] < k), g[s]["died"]) for s in seeds]
        rows.sort()
        third = len(rows) // 3
        parts = [rows[:third], rows[third:2 * third], rows[2 * third:]]
        print("  cheese rows in the first %d pieces (terciles): " % k + "; ".join(
            "%d-%d rows: %d/%d died" % (p[0][0], p[-1][0], sum(x[1] for x in p), len(p)) for p in parts if p))
    if len(runs) == 2:
        (na, a), (nb, b) = runs
        da, db = {s for s in seeds if a[s]["died"]}, {s for s in seeds if b[s]["died"]}
        print("\nsame seeds: both died %d, only %s %d, only %s %d, neither %d" % (
            len(da & db), na, len(da - db), nb, len(db - da), len(seeds) - len(da | db)))
        both = sorted(da & db)
        if both:
            diff = [b[s]["pieces"] - a[s]["pieces"] for s in both]
            print("  where both died: %s lasted %+.0f pieces on average vs %s" % (nb, sum(diff) / len(diff), na))


if __name__ == "__main__":
    main()
