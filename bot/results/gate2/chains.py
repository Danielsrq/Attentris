"""Paid gate chains in a replay: game, placement numbers, clears, payout."""
import json
import sys
from collections import Counter, defaultdict

path = sys.argv[1]
games = defaultdict(list)
for line in open(path):
    r = json.loads(line)
    games[r["replay"]].append(r)


def value(r):
    spin, lines = r["spin"], r["cleared"]
    if spin == "full" and r["place"]["type"] == "t" and lines == 2:
        return 4
    if spin == "full" and r["place"]["type"] == "t" and lines == 3:
        return 10
    if lines == 4:
        return 1
    return 0


kinds = Counter()
lines_out = []
for name, steps in games.items():
    # Gate rules: a full TSD/TST opens; TSD/TST/quad join and restart the 2-placement
    # window; a chain pays once it holds two spin clears.
    left = 0
    chain = []
    def close():
        if sum(1 for c in chain if c[3]) >= 2:
            lines_out.append((name, list(chain)))
    for i, r in enumerate(steps):
        v = value(r)
        spin = r["spin"] == "full" and v > 1
        if left > 0:
            if v > 0:
                chain.append((i + 1, r["clear"], v, spin))
                left = 2
                continue
            left -= 1
            if left == 0:
                close()
                chain = []
        elif spin:
            chain = [(i + 1, r["clear"], v, True)]
            left = 2
    if left > 0:
        close()

for name, chain in lines_out:
    kinds[" -> ".join(c[1] for c in chain)] += 1
print(f"{len(lines_out)} paid chains in {len(games)} games")
for k, c in kinds.most_common():
    print(f"  {c:3d}  {k}")
print()
for name, chain in lines_out:
    print(f"  {name}  placements {', '.join(str(c[0]) for c in chain):<14}  {' -> '.join(c[1] for c in chain)}  = {sum(c[2] for c in chain)}")
