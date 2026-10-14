"""Watch recorded versus matches (bot/tools/versus.py --record DIR) in a pygame window: both boards side
by side, turn by turn, at the pace the bots actually played (a turn takes the slower bot's measured think
time, at least the match's budget_ms: 2 PPS replays play at 500 ms per piece), with a speed multiplier.

    external/tetrisbattle/.venv/Scripts/python bot/tools/versus_watch.py bot/results/versus/replays [--speed 0.5]

Keys: space pause | left / right one turn back / forward (shift: 10) | up / down speed x2 / /2 |
1 / 2 / 3 / 4 speed x1 / x0.5 / x0.25 / x0.125 | n / p next / previous match | r restart | esc quit
"""
import argparse
import glob
import json
import os
import sys

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

H, W, VIS = 40, 10, 20          # board rows, columns, visible rows (rows 20..39)
SHOW = 22                       # rows drawn: 2 buffer rows above the visible 20
CELL = 24
COLORS = {"I": (0, 200, 220), "O": (230, 210, 0), "T": (160, 60, 200), "S": (60, 190, 60), "Z": (220, 50, 50),
          "J": (40, 80, 220), "L": (240, 140, 20), "G": (120, 120, 120), "?": (90, 90, 110)}
BG, GRID, TEXT, DIM, WARN = (18, 18, 26), (40, 40, 55), (230, 230, 235), (140, 140, 155), (230, 60, 60)
SPEEDS = {pygame.K_1: 1.0, pygame.K_2: 0.5, pygame.K_3: 0.25, pygame.K_4: 0.125}


def name_of(spec):
    k, *rest = spec.split(":")
    if k == "cem":
        w = rest[0].replace("\\", "/").split("/")[-1]
        base = "CEM-survival" if "survive" in w else ("CEM" if w == "normal.txt" else "CEM (%s)" % w)
        return "%s d%s b%s%s" % (base, rest[1], rest[2], " no-chance" if len(rest) > 3 else "")
    if k == "cc2":
        return "Cold Clear 2 (%s)" % ("time-matched" if rest[0] == "match" else rest[0] + " ms")
    if k == "nn":
        d = ("d%s b%s" % (rest[2], rest[3])) if len(rest) > 3 else "d1"
        return "AtTenTris %s %s" % (os.path.basename(rest[0]).replace("model_", "").replace(".pt", ""), d)
    return spec


def clear_name(piece, spin, lines, combo, pc):
    if lines == 0:
        return ""
    n = ["", "Single", "Double", "Triple", "Quad"][min(lines, 4)]
    s = ("%s-spin %s%s" % (piece, "Mini " if spin == 1 else "", n)) if spin else n
    if pc:
        s += "  PERFECT CLEAR"
    if combo and combo >= 1:
        s += "  %d combo" % combo
    return s


class Side:
    """One player's board rebuilt from the frames: colours follow the pieces (garbage grey)."""

    def __init__(self):
        self.grid = [["." for _ in range(W)] for _ in range(H)]
        self.attack = self.pieces = self.sent = self.received = self.pending = 0
        self.ms = 0.0
        self.last = ""
        self.b2b = -1
        self.combo = -1
        self.dead = False
        self.cells = []

    def apply(self, f):
        if f.get("dead"):
            self.dead = True
            return
        self.cells = [(f["cells"][i], f["cells"][i + 1]) for i in range(0, len(f["cells"]), 2)]
        for x, y in self.cells:
            if 0 <= y < H and 0 <= x < W:
                self.grid[y][x] = f["piece"]
        keep = [r for r in self.grid if not all(c != "." for c in r)]
        self.grid = [["."] * W for _ in range(H - len(keep))] + keep
        # trust the recorded board for what is filled (colours from our rebuild)
        rows = f.get("rows")
        if rows:
            for y in range(H):
                for x in range(W):
                    filled = rows[y] >> x & 1
                    if filled and self.grid[y][x] == ".":
                        self.grid[y][x] = "?"
                    elif not filled:
                        self.grid[y][x] = "."
        pc = rows is not None and not any(rows)
        self.attack += f["attack"]
        self.pieces += 1
        self.b2b, self.combo = f.get("b2b", -1), f.get("combo", -1)
        self.ms = f.get("ms", 0.0)
        if f["lines"]:
            self.last = clear_name(f["piece"], f.get("spin", 0), f["lines"], self.combo, pc) + ("  +%d" % f["attack"] if f["attack"] else "")
        for c, hole in f.get("garbage", []):
            self.grid = self.grid[c:] + [["G" if x != hole else "." for x in range(W)] for _ in range(c)]
            self.cells = [(x, y - c) for x, y in self.cells]
            self.received += c
        self.pending = f.get("pending", 0)
        self.sent = f.get("sent", self.sent)


class Playback:
    def __init__(self, path):
        self.path = path
        self.rec = json.load(open(path))
        self.frames = self.rec["frames"]
        self.names = [name_of(self.rec["a"]), name_of(self.rec["b"])]
        budget = self.rec.get("budget_ms", 0.0) or 0.0
        self.dt = [max(budget, f[0].get("ms", 0.0), f[1].get("ms", 0.0), 1.0) / 1000.0 for f in self.frames]
        self.seek(0)

    def seek(self, k):
        k = max(0, min(k, len(self.frames)))
        self.sides = [Side(), Side()]
        for f in self.frames[:k]:
            for i in range(2):
                self.sides[i].apply(f[i])
        self.k = k
        self.clock = sum(self.dt[:k])

    def step(self):
        if self.k < len(self.frames):
            for i in range(2):
                self.sides[i].apply(self.frames[self.k][i])
            self.clock += self.dt[self.k]
            self.k += 1

    def done(self):
        return self.k >= len(self.frames)


def draw_side(screen, font, small, s, name, x0, y0, winner):
    screen.blit(font.render(name, True, TEXT), (x0, y0 - 34))
    top = H - SHOW
    pygame.draw.rect(screen, GRID, (x0 - 2, y0 - 2, W * CELL + 4, SHOW * CELL + 4), 2)
    for r in range(SHOW):
        y = top + r
        for x in range(W):
            c = s.grid[y][x]
            rect = (x0 + x * CELL, y0 + r * CELL, CELL - 1, CELL - 1)
            if c == ".":
                pygame.draw.rect(screen, (28, 28, 38) if y >= H - VIS else (22, 22, 30), rect)
            else:
                col = COLORS.get(c, COLORS["?"])
                if (x, y) in s.cells:
                    col = tuple(min(255, v + 50) for v in col)
                pygame.draw.rect(screen, col, rect)
    # incoming garbage meter (red), up to 20 rows
    p = min(s.pending, VIS)
    mx = x0 + W * CELL + 8
    pygame.draw.rect(screen, GRID, (mx, y0 + (SHOW - VIS) * CELL, 8, VIS * CELL), 1)
    if p:
        pygame.draw.rect(screen, WARN, (mx + 1, y0 + SHOW * CELL - p * CELL, 6, p * CELL))
    lines = ["pieces %d   attack %d   APP %.3f" % (s.pieces, s.attack, s.attack / max(1, s.pieces)),
             "sent %d   received %d   incoming %d" % (s.sent, s.received, s.pending),
             "B2B %s   combo %s   %.1f ms/move" % (s.b2b if s.b2b >= 0 else "-", s.combo if s.combo >= 0 else "-", s.ms),
             s.last]
    for i, t in enumerate(lines):
        screen.blit(small.render(t, True, TEXT if i < 3 else (250, 220, 120)), (x0, y0 + SHOW * CELL + 10 + 20 * i))
    if s.dead:
        screen.blit(font.render("TOPPED OUT", True, WARN), (x0 + 50, y0 + 200))
    elif winner:
        screen.blit(font.render("WINS", True, (120, 230, 120)), (x0 + 85, y0 + 200))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("replays", nargs="+", help="replay .json files or directories of them (versus.py --record)")
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()
    files = []
    for p in args.replays:
        files += sorted(glob.glob(os.path.join(p, "*.json"))) if os.path.isdir(p) else [p]
    if not files:
        sys.exit("no replays found")
    pygame.init()
    screen = pygame.display.set_mode((900, 760))
    pygame.display.set_caption("versus replay")
    font, small = pygame.font.SysFont("consolas", 22), pygame.font.SysFont("consolas", 16)
    clock = pygame.time.Clock()
    gi, pb = 0, Playback(files[0])
    speed, paused, acc = args.speed, False, 0.0
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                return
            if e.type == pygame.KEYDOWN:
                big = 10 if e.mod & pygame.KMOD_SHIFT else 1
                if e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key == pygame.K_RIGHT:
                    pb.seek(pb.k + big)
                elif e.key == pygame.K_LEFT:
                    pb.seek(pb.k - big)
                elif e.key == pygame.K_UP:
                    speed *= 2
                elif e.key == pygame.K_DOWN:
                    speed /= 2
                elif e.key in SPEEDS:
                    speed = SPEEDS[e.key]
                elif e.key == pygame.K_r:
                    pb.seek(0)
                elif e.key in (pygame.K_n, pygame.K_p):
                    gi = (gi + (1 if e.key == pygame.K_n else -1)) % len(files)
                    pb = Playback(files[gi])
                acc = 0.0
        dt = clock.tick(120) / 1000.0
        if not paused and not pb.done():
            acc += dt * speed
            while not pb.done() and acc >= pb.dt[pb.k]:
                acc -= pb.dt[pb.k]
                pb.step()
        screen.fill(BG)
        rec = pb.rec
        win = rec["winner"] if pb.done() else None
        draw_side(screen, font, small, pb.sides[0], pb.names[0], 60, 70, win == "a")
        draw_side(screen, font, small, pb.sides[1], pb.names[1], 500, 70, win == "b")
        pps = pb.k / pb.clock if pb.clock > 0 else 0.0
        status = "%s  seed %d  match %d/%d  turn %d/%d  t=%.1fs (%.1f PPS each)  x%g%s" % (
            rec["mode"], rec["seed"], gi + 1, len(files), pb.k, len(pb.frames), pb.clock, pps, speed, "  PAUSED" if paused else "")
        screen.blit(small.render(status, True, DIM), (20, 12))
        screen.blit(small.render("space pause  <-/-> step (shift 10)  up/down x2 /2  1-4 = x1 x0.5 x0.25 x0.125  n/p match  r restart  esc",
                                 True, DIM), (20, 735))
        pygame.display.flip()


if __name__ == "__main__":
    main()
