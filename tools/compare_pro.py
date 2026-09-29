"""Per-game comparison of one pro's clean (garbage-free) play against the bot.

    python tools/compare_pro.py kezdabez data/bot/standard.jsonl

Pro clears are scored with this project's attack table (core/attack_table.gd) under
standard rules: non-T spins count as plain clears. Only the `strict` + `extended`
tiers are used -- the stretch of each Zenith run before unrecorded garbage landed.
Each bot game is cut to the same length as the pro game it is paired with, so
openings and mid-game are weighted the same on both sides.
"""
import collections
import json
import sys

LINES = {1: 0, 2: 1, 3: 2, 4: 4}
T_FULL = {1: 2, 2: 4, 3: 6}
T_MINI = {1: 1, 2: 2}


def score(records):
    s = collections.Counter()
    b2b = -1
    for r in records:
        lines = int(r.get("cleared", 0))
        piece = r["place"]["type"].lower()
        spin = r.get("spin", "none") if piece == "t" else "none"
        s["pieces"] += 1
        if lines == 0:
            continue
        difficult = lines == 4 or spin != "none"
        b2b = b2b + 1 if difficult else -1
        if spin == "full":
            atk, kind = T_FULL.get(lines, 0), ["", "TSS", "TSD", "TST"][min(lines, 3)]
        elif spin == "mini":
            atk, kind = T_MINI.get(lines, 0), "mini"
        else:
            atk, kind = LINES[min(lines, 4)], "quad" if lines == 4 else "plain"
        if difficult and b2b >= 1:
            atk += 1
        # Perfect clear: every cell on the board, plus the piece's four, was cleared.
        filled = sum(len(row) - row.count(".") for row in r["board"])
        if filled + 4 == 10 * lines:
            atk += 10
            s["PC"] += 1
        s["attack"] += atk
        s[kind] += 1
        s["b2b"] = max(s["b2b"], b2b)
    return s


def row(name, s):
    n = max(s["pieces"], 1)
    per = lambda k: 100.0 * s[k] / n
    return (f"{name:<24} {s['pieces']:>6} {s['attack'] / n:>6.3f} {per('quad'):>6.2f} {per('TSD'):>6.2f} "
            f"{per('TST'):>6.2f} {per('TSS'):>6.2f} {per('mini'):>6.2f} {per('plain'):>6.2f} {per('PC'):>6.2f} {s['b2b']:>4}")


def main():
    user = sys.argv[1] if len(sys.argv) > 1 else "kezdabez"
    bot_path = sys.argv[2] if len(sys.argv) > 2 else "data/bot/standard.jsonl"
    pro = collections.OrderedDict()
    for line in open("data/zenith_full.jsonl"):
        r = json.loads(line)
        if r["user"] == user and r["tier"] in ("strict", "extended"):
            pro.setdefault(r["replay"], []).append(r)
    games = sorted(pro.items(), key=lambda kv: -len(kv[1]))[:5]
    bot = collections.OrderedDict()
    for line in open(bot_path):
        r = json.loads(line)
        bot.setdefault(r["replay"], []).append(r)

    head = f"{'game':<24} {'pieces':>6} {'APP':>6} {'quad%':>6} {'TSD%':>6} {'TST%':>6} {'TSS%':>6} {'mini%':>6} {'1-3%':>6} {'PC%':>6} {'b2b':>4}"
    print(head)
    tot_p, tot_b = collections.Counter(), collections.Counter()
    for (name, recs), (bname, brecs) in zip(games, bot.items()):
        n = len(recs)
        sp, sb = score(recs), score(brecs[:n])
        print(row(f"{user} {name[-6:]}", sp))
        print(row(f"  bot {bname} [:{n}]", sb))
        for k, v in sp.items():
            tot_p[k] = max(tot_p[k], v) if k == "b2b" else tot_p[k] + v
        for k, v in sb.items():
            tot_b[k] = max(tot_b[k], v) if k == "b2b" else tot_b[k] + v
    print("-" * len(head))
    print(row(f"{user} (5 games)", tot_p))
    print(row("bot (same lengths)", tot_b))


if __name__ == "__main__":
    main()
