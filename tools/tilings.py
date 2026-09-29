#!/usr/bin/env python3
"""Enumerate every way whole tetrominoes tile a w x h rectangle.

Structures.scan() accepts any tiling rather than matching templates, so this is how you
find out what a new rectangle size would actually let players build -- and therefore
what it pays, since Structures.payout() pays one per piece into its group.

    python tools/tilings.py 4 3      # the size Game scans by default
    python tools/tilings.py 4 3 -v   # with an ASCII diagram per tiling

Output is grouped by composition, which is what matters for the economy: a composition
is a payout. Lower-case letters in the diagrams are a second piece of the same type,
matching the convention _build() uses in tests/run_tests.gd.
"""
import sys
from collections import Counter
from itertools import product

SPAWN = {
    "I": [(0, 0), (1, 0), (2, 0), (3, 0)],
    "O": [(0, 0), (1, 0), (0, 1), (1, 1)],
    "T": [(0, 0), (1, 0), (2, 0), (1, 1)],
    "S": [(1, 0), (2, 0), (0, 1), (1, 1)],
    "Z": [(0, 0), (1, 0), (1, 1), (2, 1)],
    "J": [(0, 0), (0, 1), (1, 1), (2, 1)],
    "L": [(2, 0), (0, 1), (1, 1), (2, 1)],
}
GROUP = {"S": "SZ", "Z": "SZ", "L": "LJ", "J": "LJ", "I": "I", "O": "O", "T": "T"}


def rotations(cells):
    out = []
    for _ in range(4):
        cells = [(-y, x) for x, y in cells]
        mx = min(x for x, y in cells)
        my = min(y for x, y in cells)
        norm = tuple(sorted((x - mx, y - my) for x, y in cells))
        if norm not in out:
            out.append(norm)
    return out


def placements(w, h):
    out = {}
    for name, spawn in SPAWN.items():
        seen = set()
        for r in rotations(spawn):
            pw = max(x for x, y in r) + 1
            ph = max(y for x, y in r) + 1
            for oy in range(h - ph + 1):
                for ox in range(w - pw + 1):
                    seen.add(frozenset((x + ox, y + oy) for x, y in r))
        out[name] = sorted(seen, key=sorted)
    return out


def tile(w, h):
    """Every tiling, found by always filling the top-left empty cell -- which visits
    each tiling exactly once instead of once per piece ordering."""
    place = placements(w, h)
    solutions = []

    def go(remaining, used):
        if not remaining:
            solutions.append(list(used))
            return
        cell = min(remaining, key=lambda c: (c[1], c[0]))
        for name, ps in place.items():
            for p in ps:
                if cell in p and p <= remaining:
                    go(remaining - p, used + [(name, p)])

    go(frozenset(product(range(w), range(h))), [])
    return solutions


def render(sol, w, h):
    grid = [["."] * w for _ in range(h)]
    count = Counter()
    for name, p in sol:
        ch = name if count[name] == 0 else name.lower()
        count[name] += 1
        for x, y in p:
            grid[y][x] = ch
    return ["".join(row) for row in grid]


def payout(names):
    pay = Counter(GROUP[n] for n in names)
    return " ".join("%s%d" % (g, n) for g, n in sorted(pay.items()))


def main(argv):
    verbose = "-v" in argv
    argv = [a for a in argv if a != "-v"]
    w = int(argv[1]) if len(argv) > 1 else 4
    h = int(argv[2]) if len(argv) > 2 else 3
    if (w * h) % 4:
        print("%dx%d is %d cells, not a multiple of 4 -- no tiling" % (w, h, w * h))
        return
    sols = tile(w, h)
    print("%dx%d: %d tilings, %d pieces each" % (w, h, len(sols), w * h // 4))
    by_comp = {}
    for s in sols:
        by_comp.setdefault(tuple(sorted(n for n, _ in s)), []).append(s)
    for comp, group in sorted(by_comp.items(), key=lambda kv: -len(kv[1])):
        print("  %-12s %-14s %d tiling(s)" % ("".join(comp), payout(comp), len(group)))
        if verbose:
            for s in group:
                print("      " + " / ".join(render(s, w, h)))


if __name__ == "__main__":
    main(sys.argv)
