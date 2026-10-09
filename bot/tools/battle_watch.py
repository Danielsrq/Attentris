"""Watch a recorded bot-vs-bot game (bot/tools/battle_match.py --out ...jsonl) in TetrisBattle's
own pygame UI. The game is rebuilt from its seed and the recorded placements, so it is
exactly what was played; realtime / cap games play back on their own clock.

    cd external/tetrisbattle
    set PYTHONPATH=.
    .venv\\Scripts\\python ..\\..\\bot\\tools\\battle_watch.py ..\\..\\bot\\results\\battle\\match_battle_realtime.jsonl --game 3

Keys: space pause, left / right arrow one piece back / forward (shift: 10), up / down speed x2 / /2,
n / p next / previous game, r restart, esc quit.
"""
import argparse
import json
import os
import sys

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pygame

from TetrisBattle.settings import SCREENWIDTH, SCREENHEIGHT, load_imgs
from TetrisBattle.renderer import Renderer
from TetrisBattle.ai.search import enumerate_placements
_driver = os.environ.get("SDL_VIDEODRIVER")
from battle_match import new_player  # noqa: E402
if _driver is None:
    os.environ.pop("SDL_VIDEODRIVER", None)  # battle_match defaults it to dummy; we want a window

# from game/tetris_game.py (left, right board)
POS = [
    {"drawscreen": (112, 138), "transparent": (110, 135), "you_win": (120, 230), "you_lose": (115, 230),
     "ko": (140, 233), "attack_clean": (298, 140, 3, 360), "attack_alarm": (298, 481, 3, 18), "label": (112, 112)},
    {"drawscreen": (495, 138), "transparent": (494, 135), "you_win": (520, 230), "you_lose": (515, 230),
     "ko": (527, 233), "attack_clean": (680, 140, 3, 360), "attack_alarm": (680, 481, 3, 18), "label": (495, 112)},
]


RENAME = {"his_rl": "cms-RL"}  # older match files used his_rl


class Playback:
    def __init__(self, rec, turn_pps):
        self.rec = rec
        self.turn_pps = turn_pps
        self.sims = [new_player(i, rec["seed"]) for i in range(2)]
        self.raw = [p["bot"] for p in rec["players"]]
        self.names = [RENAME.get(n, n) for n in self.raw]
        # the match alternates who moves first; keep his bot on the right whatever the order
        self.side = [1 if n in RENAME.values() else 0 for n in self.names]
        if self.side[0] == self.side[1]:
            self.side = [0, 1]
        self.moves = rec["moves"]
        if rec["speed"] == "turn":  # no clock: show both players at turn_pps each
            self.times = [(k + 1) / (2.0 * turn_pps) for k in range(len(self.moves))]
        else:
            self.times = [m["t"] for m in self.moves]
        self.k = 0
        self.sent = [0, 0]
        self.pieces = [0, 0]
        self.last = ["", ""]
        self.ko = [False, False]

    def seek(self, k):
        '''The playback after k moves: forward from here, backward by replaying from the start.'''
        k = max(0, min(k, len(self.moves)))
        pb = self if k >= self.k else Playback(self.rec, self.turn_pps)
        while pb.k < k:
            pb.step()
        return pb

    def done(self):
        return self.k >= len(self.moves)

    def step(self):
        m = self.moves[self.k]
        self.k += 1
        i = m["p"]
        s = self.sims[i]
        t = s.tetris
        want = (m["hold"], m["rot"], m["x"], m["y"], m["tspin"])
        opts = enumerate_placements(t)
        p = next((o for o in opts if (bool(o.use_hold), o.rot, o.px, o.py, o.tspin) == want), None)
        if p is None:  # fall back to the cells (older records)
            cells = sorted(tuple(c) for c in m["cells"])
            p = next(o for o in opts if bool(o.use_hold) == m["hold"] and sorted(tuple(c) for c in o.cells) == cells)
        out = s.apply(p)
        self.pieces[i] += 1
        self.sent[i] += out["scores"]
        kind = ""
        if out["cleared"]:
            kind = {0: "", 1: "T-spin mini ", 2: "T-spin "}[out["tspin"]] + \
                ["", "single", "double", "triple", "quad"][out["cleared"]]
            if out["scores"]:
                kind += "  +%d" % out["scores"]
        self.last[i] = kind
        if out["scores"] > 0:
            o = 1 - i
            self.sims[o].tetris.add_attacked(out["scores"], self.sims[o].hole_rng.randrange(10))
        if out["ko"]:
            self.ko[i] = True


def draw(screen, r, font, small, pb, clock_s, speed, paused):
    r.drawByName("gamescreen", 0, 0)
    for i in range(2):
        t = pb.sims[i].tetris
        pos = POS[pb.side[i]]
        r.drawScreen(t, *pos["drawscreen"])
        pygame.draw.rect(screen, (30, 30, 30), pos["attack_clean"])
        for j in range(min(t._attacked, 20)):
            a = list(pos["attack_alarm"])
            a[1] -= 18 * j
            pygame.draw.rect(screen, (255, 0, 0), a)
        app = pb.sent[i] / max(1, pb.pieces[i])
        # average over the elapsed game time (turn games have no clock of their own)
        pps = "-" if pb.rec["speed"] == "turn" or clock_s <= 0 else "%.2f" % (pb.pieces[i] / clock_s)
        lx, ly = pos["label"]
        pygame.draw.rect(screen, (15, 15, 20), (lx - 2, ly - 2, 184, 22))
        screen.blit(font.render(pb.names[i], True, (240, 240, 240)), (lx, ly))
        pygame.draw.rect(screen, (15, 15, 20), (lx - 2, 503, 290, 54))
        screen.blit(small.render("pcs %d  sent %d  APP %.2f  PPS %s" % (pb.pieces[i], pb.sent[i], app, pps),
                                 True, (220, 220, 220)), (lx, 505))
        b2b = "-" if t.b2b_count < 0 else "%d" % t.b2b_count   # -1 = no chain
        screen.blit(small.render("b2b %s  combo %d" % (b2b, max(0, t.combo)), True, (220, 220, 220)), (lx, 522))
        screen.blit(small.render(pb.last[i], True, (255, 220, 120)), (lx, 539))
        if pb.done():
            r.drawByName("transparent", *pos["transparent"])
            win = pb.rec["winner"] == pb.raw[i]
            r.drawByName("you_win" if win else "you_lose", *pos["you_win" if win else "you_lose"])
            if pb.ko[i]:
                r.drawByName("ko", *pos["ko"])
    status = "game %d  %s  t=%.1fs  piece %d/%d  x%g%s  space pause  <-/-> step (shift 10)  up/down speed  n/p game  r restart" % (
        pb.rec["game"], pb.rec["speed"], clock_s, pb.k, len(pb.moves), speed, "  PAUSED" if paused else "")
    pygame.draw.rect(screen, (15, 15, 20), (0, SCREENHEIGHT - 22, SCREENWIDTH, 22))
    screen.blit(small.render(status, True, (200, 200, 200)), (6, SCREENHEIGHT - 19))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("match", help="jsonl written by battle_match.py")
    ap.add_argument("--game", type=int, default=0)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    ap.add_argument("--turn-pps", type=float, default=3.0, help="per-player PPS when showing turn-mode games")
    args = ap.parse_args()

    recs = sorted((json.loads(l) for l in open(args.match)), key=lambda r: r["game"])
    if not recs or "moves" not in recs[0]:
        sys.exit("no moves in %s -- re-run battle_match.py (it records them now)" % args.match)
    gi = next((k for k, r in enumerate(recs) if r["game"] == args.game), 0)

    pygame.init()
    screen = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
    r = Renderer(screen, load_imgs())
    font = pygame.font.SysFont("consolas", 16, bold=True)
    small = pygame.font.SysFont("consolas", 13)
    clock = pygame.time.Clock()
    speed = args.speed
    paused = False
    pb = Playback(recs[gi], args.turn_pps)
    pygame.display.set_caption("%s vs %s" % tuple(n for _, n in sorted(zip(pb.side, pb.names))))
    t = 0.0
    while True:
        dt = clock.tick(60) / 1000.0
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return
            if e.type != pygame.KEYDOWN:
                continue
            if e.key == pygame.K_ESCAPE:
                return
            if e.key == pygame.K_SPACE:
                paused = not paused
            elif e.key in (pygame.K_RIGHT, pygame.K_LEFT):
                n = 10 if e.mod & pygame.KMOD_SHIFT else 1
                pb = pb.seek(pb.k + (n if e.key == pygame.K_RIGHT else -n))
                t = pb.times[pb.k - 1] if pb.k else 0.0
                paused = True
            elif e.key == pygame.K_UP:
                speed *= 2
            elif e.key == pygame.K_DOWN:
                speed /= 2
            elif e.key in (pygame.K_n, pygame.K_p, pygame.K_r):
                gi = (gi + {pygame.K_n: 1, pygame.K_p: -1, pygame.K_r: 0}[e.key]) % len(recs)
                pb = Playback(recs[gi], args.turn_pps)
                t = 0.0
        if not paused:
            t += dt * speed
            while not pb.done() and pb.times[pb.k] <= t:
                pb.step()
        draw(screen, r, font, small, pb, t, speed, paused)
        pygame.display.flip()


if __name__ == "__main__":
    main()
