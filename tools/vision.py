"""Reads a board back out of a rendered frame by per-cell colour matching.

The premise this tests: a Tetris board is a fixed grid of flat, known colours, so
recovering its state needs no convolution or correlation -- just the centre pixel of
each cell matched against a palette. If that holds, a replay can be reconstructed by
watching it rather than simulating it, which sidesteps every unmodelled rule (Zenith's
garbage, the allspin penalty) because the result is observed instead of derived.

Calibrated here against our own viewer, where the ground truth is known. The same code
re-pointed at TETR.IO frames would need new geometry and palette constants, nothing more.
"""
import sys, json
from PIL import Image

ORIGIN = (161, 60)      # top-left of the visible field, in pixels
CELL = 30               # cell size
COLS, ROWS = 10, 20

PALETTE = {
    "i": (0, 240, 240), "o": (240, 240, 0), "t": (160, 0, 240), "s": (0, 240, 0),
    "z": (240, 0, 0),   "j": (0, 0, 240),   "l": (240, 160, 0),
    "G": (107, 112, 122),          # garbage grey
    ".": (18, 18, 23),             # empty cell / field background
}

def nearest(px):
    best, bd = ".", 1 << 30
    for k, c in PALETTE.items():
        d = sum((a - b) ** 2 for a, b in zip(px[:3], c))
        if d < bd: best, bd = k, d
    return best, bd

def read_board(path):
    im = Image.open(path).convert("RGB")
    out, worst = [], 0
    for r in range(ROWS):
        row = ""
        for c in range(COLS):
            x = ORIGIN[0] + c * CELL + CELL // 2
            y = ORIGIN[1] + r * CELL + CELL // 2
            k, d = nearest(im.getpixel((x, y)))
            worst = max(worst, d)
            row += k
        out.append(row)
    return out, worst

if __name__ == "__main__":
    got, worst = read_board(sys.argv[1])
    truth = [l.rstrip("\n") for l in open(sys.argv[2], encoding="utf-8")]
    truth = [t for t in truth if not t.startswith("PIECE")][-ROWS:]   # visible rows only
    ok = sum(a == b for g, t in zip(got, truth) for a, b in zip(g, t))
    total = ROWS * COLS
    print("cells matched: %d / %d  (%.2f%%)" % (ok, total, 100 * ok / total))
    print("worst colour distance: %d (0 = exact)" % worst)
    if ok != total:
        print("\n  detected            truth")
        for g, t in zip(got, truth):
            print("  %s   %s   %s" % (g, t, "" if g == t else "  <-- differs"))
