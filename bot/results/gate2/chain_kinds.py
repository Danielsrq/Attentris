"""Classify the paid gate chains by how each T went in: V = vertical (rot 1/3: STSD,
imperial cross), F = flat (rot 2). For flat -> flat, 'stacked' when the second T lands
in the rows directly under where the first T was (a fractal), else 'apart'."""
import json
import sys
from collections import Counter, defaultdict

games = defaultdict(list)
for line in open(sys.argv[1]):
    r = json.loads(line)
    games[r["replay"]].append(r)


def value(r):
    if r["spin"] == "full" and r["place"]["type"] == "t" and r["cleared"] in (2, 3):
        return 4 if r["cleared"] == 2 else 10
    return 1 if r["cleared"] == 4 else 0


kinds = Counter()
examples = defaultdict(list)
for name, steps in games.items():
    left, chain = 0, []

    def close():
        spins = [c for c in chain if c[1]]
        if len(spins) >= 2:
            a, b = steps[spins[0][0]], steps[spins[1][0]]
            ka = "V" if a["place"]["r"] in (1, 3) else "F"
            kb = "V" if b["place"]["r"] in (1, 3) else "F"
            k = f"{ka}->{kb}"
            if k == "F->F":
                # Rows (from the bottom) of each T's cells: stacked if the second T sits in
                # the rows the first T's clear exposed, i.e. same columns.
                ca = {x for x, _ in a["place"]["cells"]}
                cb = {x for x, _ in b["place"]["cells"]}
                k += " stacked" if ca == cb else " apart"
            kinds[k] += 1
            examples[k].append(f"{name} {spins[0][0] % 500 + 1}->{spins[1][0] % 500 + 1}")

    for i, r in enumerate(steps):
        v = value(r)
        spin = v > 1
        if left > 0:
            if v > 0:
                chain.append((i, spin))
                left = 2
                continue
            left -= 1
            if left == 0:
                close()
                chain = []
        elif spin:
            chain, left = [(i, True)], 2
    if left > 0:
        close()

for k, c in kinds.most_common():
    print(f"  {c:3d}  {k:<16} e.g. {', '.join(examples[k][:3])}")
