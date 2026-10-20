"""Spin, B2B and survival stats from `bot play` replays.

    python bot/tools/spin_stats.py NAME=REPLAYS[,GAMELOG] ...

REPLAYS is a `bot play --out` file (one record per locked piece; .jsonl or .jsonl.gz), GAMELOG the
matching `--game-log` file (died, garbage cleared). Per run:
  died        games that topped out
  APP         attack per piece played; APPfull over the full piece budget (a death scores 0 after)
  VSfull      (attack + garbage cleared) / full piece budget (TETR.IO versus score without PPS)
  clr/100     line clears per 100 pieces
  spin%       share of clears that are spins (any piece); nonT% share of those spins by S/Z/L/J/I/O
  B2B kept    of the clears made while a B2B chain was alive, the share that kept it (spin or quad)
  chain       mean / max B2B chain length in clears (every chain counted when it ends or the game does)
  alive%      share of pieces placed with a chain alive
  rep%        share of clears that repeated the previous clear type (all-spin: each inserts a wound)
  wnd/100     wound lines inserted per 100 pieces
  s1/s2/s3    spin singles / doubles / triples per 100 pieces; q quads per 100 pieces
"""
import gzip
import json
import sys


def opener(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def run_stats(replays, gamelog, budget):
    logs = {}
    if gamelog:
        for l in opener(gamelog):
            g = json.loads(l)
            logs[g["seed"]] = g
    games = {}
    for l in opener(replays):
        r = json.loads(l)
        games.setdefault(r["replay"], []).append(r)
    s = dict(games=0, died=0, pieces=0, attack=0, gbclr=0, clears=0, spins=0, nonT=0, quads=0,
             alive_clears=0, kept=0, alive_pieces=0, repeats=0, chains=[], s1=0, s2=0, s3=0)
    for name, recs in games.items():
        recs.sort(key=lambda r: r["n"])
        seed = int(name.rsplit("_s", 1)[1])
        g = logs.get(seed, {})
        s["games"] += 1
        s["died"] += bool(g.get("died", len(recs) < budget))
        s["gbclr"] += g.get("garbage_cleared", 0)
        prev = -1
        for r in recs:
            s["pieces"] += 1
            s["attack"] += r.get("attack", 0)
            if prev >= 0:
                s["alive_pieces"] += 1
            if r.get("cleared", 0) > 0:
                s["clears"] += 1
                spin = r.get("spin", "none") != "none"
                s["spins"] += spin
                s["nonT"] += spin and r["place"]["type"] != "t"
                s["quads"] += (not spin) and r["cleared"] == 4
                if spin:
                    s["s%d" % min(r["cleared"], 3)] += 1
                s["repeats"] += bool(r.get("repeat"))
                b = r.get("b2b", -1)
                if prev >= 0:
                    s["alive_clears"] += 1
                    if b > prev:
                        s["kept"] += 1
                    else:
                        s["chains"].append(prev + 1)
                prev = b
        if prev >= 0:
            s["chains"].append(prev + 1)
    return s


def main():
    budget = 500
    rows = []
    for a in sys.argv[1:]:
        if a.startswith("--pieces="):
            budget = int(a.split("=")[1])
            continue
        name, paths = a.split("=", 1)
        rep, _, log = paths.partition(",")
        rows.append((name, run_stats(rep, log or None, budget)))
    print("%-34s %6s %6s %7s %7s %7s %6s %6s %5s %8s %10s %6s %5s %7s" % (
        "run", "died", "pcs", "APP", "APPfull", "VSfull", "clr/100", "spin%", "nonT%", "B2B kept", "chain", "alive%", "rep%", "wnd/100")
          + " %5s %5s %5s %5s" % ("s1", "s2", "s3", "q"))
    for name, s in rows:
        n, p, c = s["games"], max(s["pieces"], 1), max(s["clears"], 1)
        full = n * budget
        ch = s["chains"] or [0]
        print("%-34s %6s %6.0f %7.3f %7.3f %7.3f %6.1f %6.1f %5.1f %8.3f %10s %6.1f %5.2f %7.2f" % (
            name, "%d/%d" % (s["died"], n), p / max(n, 1), s["attack"] / p, s["attack"] / full,
            (s["attack"] + s["gbclr"]) / full, 100 * s["clears"] / p, 100 * s["spins"] / c,
            100 * s["nonT"] / max(s["spins"], 1), s["kept"] / max(s["alive_clears"], 1),
            "%.1f / %d" % (sum(ch) / len(ch), max(ch)), 100 * s["alive_pieces"] / p,
            100 * s["repeats"] / c, 100 * s["repeats"] / p)
              + " %5.1f %5.1f %5.2f %5.2f" % tuple(100 * s[k] / p for k in ("s1", "s2", "s3", "quads")))


if __name__ == "__main__":
    main()
