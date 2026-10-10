"""AtTenTris tokenizer, version 1 (see DESIGN.md): a Tetris board -> a few structure tokens,
each with a kind, a box and a 200-cell mask. Rule-based and fixed before the network runs.

Coordinates: r = row from the bottom (0..19), c = column (0..9). Replay boards (40 rows,
top to bottom) map with r = 39 - y; the 20 rows above the visible board are ignored.

Base tokens partition the 200 cells exactly once:
  sky    every row above the highest filled cell (one token; empty if the stack is 20 high)
  well   a clean well: a run of rows with exactly one hole each, all in the same column
         (length 1 = a cheese row). Only in the clean zone at the bottom of the stack.
  play   a row of the top play area: between the clean zone and the sky. The zone's top
         run counts as play instead while it is a single row whose hole is not sealed by
         the cell directly above (a TSD's stem row, an imperial cross's foot).
  band   two or more adjacent stack tokens merged to keep the board within K tokens

Overlay tokens (may overlap base tokens), at most P_MAX, by priority:
  T structures (from `bot structures`): tsd, fractal, imperial, stsd, cspin, vertical
  rect12  an empty flat-floored pit 3 wide x 4 deep or 4 wide x 3 deep: exactly the region
          L+J+O, L+S+J or L+Z+J (also L+L+O, J+J+O) fill
  fit_s, fit_z, fit_t, fit_l, fit_j  a piece-shaped recess: a resting place where a piece with
          a shaped bottom (S / Z either way up, T nub down, L / J three on top or hook on
          top) touches the surface under every column -- no gap left -- and sits snugly:
          at least MIN_CONTACTS of its cells' sides touch a filled cell or the board edge
          (a piece balanced on a peak only touches the step it rests on), and the surface
          around it (its columns +-1) is no bumpier afterwards. Mask = its cells.
  pit1..pit4  negative space: a run of 1-4 columns below both its side walls (an edge counts
          as a wall), filled with "water" up to the lower wall. pit1 = a well.

Piece tokens (no footprint): current, hold, preview 1..5.
"""
from dataclasses import dataclass, field
import json
import os
import subprocess

import numpy as np

ROWS, COLS = 20, 10
K_BOARD = 12      # base board tokens at most (sky + stack tokens)
P_MAX = 12        # overlay tokens at most (v5: 12, was 8)
# Tokenizer versions: 4 = pre-v5 checkpoints; 5 = v5 (covered slot masks include their hang,
# setup masks include their future slot); 51 = v5.1 (slot masks are exactly the tetromino, the
# overhang is its own "hang" token, setup masks are the setup placement only). Checkpoints
# record the version they were trained with ("tok_version").
TOK_VERSION = 51
MIN_CONTACTS = 2  # side contacts for a piece-shaped recess
PIECES = "IOTSZJL"
STRUCTURE_KINDS = ("imperial", "stsd", "cspin", "fractal", "tsd", "vertical")
BASE_KINDS = ("sky", "well", "play", "band")
FIT_KINDS = ("fit_s", "fit_z", "fit_t", "fit_l", "fit_j")
OVERLAY_KINDS = STRUCTURE_KINDS + ("rect12",) + FIT_KINDS + ("pit1", "pit2", "pit3", "pit4")

# Orientations with a shaped bottom: cells (dx, dy) from the lowest cell's row. Flat-bottomed
# ones (I, O, flat T / L / J, the L / J foot-down verticals) fit any flat stretch and would
# only add noise.
FIT_SHAPES = {
    "fit_s": [("flat", [(0, 0), (1, 0), (1, 1), (2, 1)]), ("upright", [(1, 0), (1, 1), (0, 1), (0, 2)])],
    "fit_z": [("flat", [(1, 0), (2, 0), (0, 1), (1, 1)]), ("upright", [(0, 0), (0, 1), (1, 1), (1, 2)])],
    "fit_t": [("nub down", [(1, 0), (0, 1), (1, 1), (2, 1)])],
    "fit_l": [("three on top", [(0, 0), (0, 1), (1, 1), (2, 1)]), ("hook on top", [(1, 0), (1, 1), (1, 2), (0, 2)])],
    "fit_j": [("three on top", [(2, 0), (0, 1), (1, 1), (2, 1)]), ("hook on top", [(0, 0), (0, 1), (0, 2), (1, 2)])],
}
PIECE_KINDS = ("current", "hold") + tuple("preview%d" % i for i in range(1, 6))
# v5 additions, appended so the indices of every earlier kind stay the same (old packed data
# and checkpoints keep working): covered I slots, and setups (an overhang one piece away).
SETUP_KINDS = ("setup_s", "setup_z", "setup_t", "setup_l", "setup_j", "setup_i")
NEW_KINDS = ("fit_i",) + SETUP_KINDS + ("hang",)    # "hang": v5.1
SLOT_KINDS = FIT_KINDS + ("fit_i",)       # fit_X = a slot for piece X, open (recess) or covered (spin)
KINDS = BASE_KINDS + OVERLAY_KINDS + PIECE_KINDS + NEW_KINDS
OVERLAY_KINDS = OVERLAY_KINDS + NEW_KINDS


@dataclass
class Token:
    kind: str
    mask: np.ndarray                      # (20, 10) bool; all False for piece tokens
    extras: dict = field(default_factory=dict)

    @property
    def box(self):
        """(row lo, row hi, col lo, col hi), inclusive; None for piece tokens."""
        if not self.mask.any():
            return None
        rs, cs = np.nonzero(self.mask)
        return int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max())

    @property
    def base(self):
        return self.kind in BASE_KINDS


def board_from_replay(rows40):
    """Replay board (40 strings, top to bottom, '.' empty) -> (20, 10) bool, row 0 at the bottom."""
    b = np.zeros((ROWS, COLS), bool)
    for r in range(ROWS):
        line = rows40[39 - r]
        for c, ch in enumerate(line[:COLS]):
            b[r, c] = ch not in ". "
    return b


def board_from_ascii(lines):
    """Rows top to bottom, '#' filled, anything else empty; pads with empty rows on top."""
    lines = [l for l in lines if l.strip()]
    b = np.zeros((ROWS, COLS), bool)
    for i, line in enumerate(reversed(lines)):
        for c, ch in enumerate(line[:COLS]):
            b[i, c] = ch == "#"
    return b


def heights(b):
    h = np.zeros(COLS, int)
    for c in range(COLS):
        filled = np.nonzero(b[:, c])[0]
        h[c] = filled.max() + 1 if len(filled) else 0
    return h


def _rows_mask(r0, r1):
    m = np.zeros((ROWS, COLS), bool)
    m[r0:r1 + 1, :] = True
    return m


def base_tokens(b, k_board=K_BOARD):
    h = heights(b)
    top = int(h.max())
    toks = []
    if top < ROWS:
        toks.append(Token("sky", _rows_mask(top, ROWS - 1), {"h": h.tolist()}))

    # Clean zone: bottom-up runs of single-hole rows, grouped by hole column.
    runs = []           # (r0, r1, hole col)
    r = 0
    while r < top and (~b[r]).sum() == 1:
        hole = int(np.nonzero(~b[r])[0][0])
        if runs and runs[-1][2] == hole and runs[-1][1] == r - 1:
            runs[-1] = (runs[-1][0], r, hole)
        else:
            runs.append((r, r, hole))
        r += 1

    def covered(run):
        r0, r1, hole = run
        return bool(b[r1 + 1:, hole].any())

    def sealed(run):
        r0, r1, hole = run
        return r1 + 1 < ROWS and bool(b[r1 + 1, hole])

    # The zone's top run is the active surface, not a well, while it is one row whose hole
    # is not sealed directly above (a TSD's stem row, an imperial cross's foot, a
    # half-built row...). Cheese holes are sealed by the next garbage row.
    while runs and runs[-1][0] == runs[-1][1] and not sealed(runs[-1]):
        runs.pop()
    zone_top = runs[-1][1] + 1 if runs else 0

    stack = []          # bottom to top: (r0, r1, kind, extras)
    for run in runs:
        r0, r1, hole = run
        stack.append((r0, r1, "well", {"length": r1 - r0 + 1, "hole": hole, "covered": covered(run)}))
    for r in range(zone_top, top):
        stack.append((r, r, "play", {}))

    # Budget: merge the lowest adjacent pair until the board fits in k_board tokens.
    while len(stack) + len(toks) > k_board and len(stack) >= 2:
        (a0, _, ka, ea), (_, b1, kb, eb) = stack[0], stack[1]
        stack[0:2] = [(a0, b1, "band", {"merged": ea.get("merged", [ka]) + eb.get("merged", [kb])})]

    for r0, r1, kind, ex in stack:
        m = _rows_mask(r0, r1)
        ex = dict(ex)
        ex.update({"rows": r1 - r0 + 1, "filled": int(b[r0:r1 + 1].sum()),
                   "under": int((~b[r0:r1 + 1] & (np.arange(ROWS)[r0:r1 + 1, None] < h[None, :])).sum())})
        toks.append(Token(kind, m, ex))
    return toks, h


def pit_tokens(b, h):
    """Negative space above the surface: every column run of width 1-4 lying below both
    side walls (an edge counts as an infinitely high wall, but not on both sides), filled
    up to the lower wall. A flat-floored 3x4 / 4x3 one is a rect12 instead."""
    out = []
    inf = 10 ** 6
    for a in range(COLS):
        for z in range(a, min(a + 4, COLS)):
            wl = h[a - 1] if a > 0 else inf
            wr = h[z + 1] if z < COLS - 1 else inf
            level = min(wl, wr)
            if level >= inf or h[a:z + 1].max() >= level:
                continue
            w = z - a + 1
            m = np.zeros((ROWS, COLS), bool)
            for c in range(a, z + 1):
                m[h[c]:level, c] = True
            if m.sum() < 2:
                continue
            floor = h[a:z + 1]
            depth = int(level - floor.max())   # depth over the highest floor cell
            flat = floor.min() == floor.max()
            if flat and ((w == 3 and depth >= 4) or (w == 4 and depth >= 3)):
                rect = np.zeros((ROWS, COLS), bool)
                f = int(floor[0])
                rect[f:f + (4 if w == 3 else 3), a:z + 1] = True
                out.append(Token("rect12", rect, {"width": w, "orientation": "3x4" if w == 3 else "4x3"}))
            else:
                out.append(Token("pit%d" % w, m, {"width": w, "depth": int(level - floor.min()), "level": int(level),
                                                  "area": int(m.sum())}))
    return out


def bumpiness(h, lo, hi):
    return int(sum(abs(int(h[c + 1]) - int(h[c])) for c in range(max(lo, 0), min(hi, COLS - 1))))


def holes_of(b):
    """Empty cells below the top of their column."""
    filled = np.maximum.accumulate(b[::-1], axis=0)[::-1]
    return int((filled & ~b).sum())


def fit_tokens(b, h, post_clear=True):
    """Open piece slots ("recesses", see the module doc). v5: judged on the board after the
    piece's line clears, so a T dropping into a T-shaped gap over a well, which completes two
    rows and leaves the well open, counts (position 28 of the attention viewer)."""
    out = []
    holes0 = holes_of(b)
    for kind, shapes in FIT_SHAPES.items():
        for orient, cells in shapes:
            w = max(dx for dx, _ in cells) + 1
            low = [min(dy for dx, dy in cells if dx == k) for k in range(w)]
            top = [max(dy for dx, dy in cells if dx == k) for k in range(w)]
            for x in range(COLS - w + 1):
                a = max(h[x + k] - low[k] for k in range(w))      # where it comes to rest
                if a + max(top) >= ROWS:
                    continue
                m = np.zeros((ROWS, COLS), bool)
                for dx, dy in cells:
                    m[a + dy, x + dx] = True
                nb = b | m
                full = nb.all(axis=1)
                lines = int(full.sum())
                gap = any(h[x + k] - low[k] != a for k in range(w))
                if lines:
                    nb = np.concatenate([nb[~full], np.zeros((lines, COLS), bool)])
                if gap and (not post_clear or lines == 0 or holes_of(nb) > holes0):
                    continue                                       # it would bury a cell
                if not post_clear:
                    lines = 0                                      # v4: no line clears, no lines extra
                if lines:
                    after = heights(nb)
                else:
                    after = h.copy()
                    for k in range(w):
                        after[x + k] = a + top[k] + 1
                # snugness: piece-cell sides against a filled cell or the board edge
                contacts = 0
                for dx, dy in cells:
                    r, c = a + dy, x + dx
                    for cc in (c - 1, c + 1):
                        if cc < 0 or cc >= COLS or (not m[r, cc] and b[r, cc]):
                            contacts += 1
                if contacts < MIN_CONTACTS:
                    continue
                gain = bumpiness(h, x - 1, x + w) - bumpiness(after, x - 1, x + w)
                if gain < 0:
                    continue  # snug but leaves the area bumpier (a tower against a wall)
                out.append(Token(kind, m, {"orientation": orient, "contacts": contacts, "gain": gain, "covered": 0, "lines": lines}))
    return out


def structure_tokens(structures, version=TOK_VERSION):
    """`bot structures` output (replay coordinates) -> overlay tokens on the visible board.
    T structures keep their kind; covered piece slots ("slot_x", v5) become fit_x tokens with
    covered = 1, their mask exactly the piece's cells; the filled overhang above a covered
    slot becomes a "hang" token (one per distinct overhang, shared by the slots under it);
    setups keep their kind, mask = the setup placement only (the slot it creates does not exist
    yet), the slot's centre in the extras. version=5 reproduces the v5 masks."""
    out = []
    hangs = {}
    for s in structures:
        m = np.zeros((ROWS, COLS), bool)
        extra_cells = s.get("hang", []) + s.get("slot", []) if version < 51 else []
        for x, y in s["cells"] + extra_cells:
            r = 39 - y
            if 0 <= r < ROWS and 0 <= x < COLS:
                m[r, x] = True
        if not m.any():
            continue
        k = s["kind"]
        if k.startswith("slot_"):
            out.append(Token("fit_" + k[5:], m, {"orientation": "spin", "contacts": 0, "gain": 0, "covered": 1, "lines": s["lines"]}))
            hc = frozenset((39 - y, x) for x, y in s.get("hang", []) if 0 <= 39 - y < ROWS)
            if version >= 51 and hc:
                n, lines = hangs.get(hc, (0, 0))
                hangs[hc] = (n + 1, max(lines, s["lines"]))
        elif k.startswith("setup_"):
            sl = [(39 - y, x) for x, y in s.get("slot", [])]
            sr = float(np.mean([r for r, _ in sl])) if sl else 0.0
            sc = float(np.mean([c for _, c in sl])) if sl else 0.0
            out.append(Token(k, m, {"lines": s["lines"], "by": s.get("by"), "slot_r": sr, "slot_c": sc, "slot_cells": sl}))
        else:
            out.append(Token(k, m, {"lines": s["lines"], "fill": s["fill"]}))
    for hc, (n, lines) in hangs.items():
        m = np.zeros((ROWS, COLS), bool)
        for r, c in hc:
            m[r, c] = True
        out.append(Token("hang", m, {"slots": n, "lines": lines}))
    return out


def overlay_priority(t):
    if t.kind in STRUCTURE_KINDS:
        return (0, STRUCTURE_KINDS.index(t.kind), -t.extras["lines"], -t.extras["fill"])
    if t.kind in SLOT_KINDS and t.extras.get("covered"):
        return (1, -t.extras["lines"], SLOT_KINDS.index(t.kind), 0)
    if t.kind == "hang":
        return (1.5, -t.extras["lines"], -t.extras["slots"], 0)
    if t.kind in SETUP_KINDS:
        return (2, -t.extras["lines"], SETUP_KINDS.index(t.kind), 0)
    if t.kind == "rect12":
        return (3, 0, 0, 0)
    if t.kind in SLOT_KINDS:
        return (4, -t.extras.get("lines", 0), -t.extras["contacts"], -t.extras["gain"])
    return (5, -t.extras["area"], t.extras["width"], 0)


def tokenize(b, piece=None, queue=(), hold=None, structures=(), k_board=K_BOARD, p_max=P_MAX, version=TOK_VERSION):
    """Board (20, 10) bool + pieces -> list of Tokens: base, then overlays, then pieces.
    version=4 reproduces the tokens pre-v5 checkpoints were trained on (T structures only,
    open fits without line clears, at most 8 overlays)."""
    toks, h = base_tokens(b, k_board)
    if version < 5:
        structures = [s for s in structures if s["kind"] in STRUCTURE_KINDS]
        p_max = min(p_max, 8)
    overlays = structure_tokens(structures, version) + fit_tokens(b, h, post_clear=version >= 5) + pit_tokens(b, h)
    overlays.sort(key=overlay_priority)
    toks += overlays[:p_max]
    if piece is not None:
        none = np.zeros((ROWS, COLS), bool)
        toks.append(Token("current", none, {"piece": piece.upper()}))
        toks.append(Token("hold", none, {"piece": hold.upper() if hold else None}))
        for i, p in enumerate(list(queue)[:5]):
            toks.append(Token("preview%d" % (i + 1), none, {"piece": p.upper()}))
    return toks


def mask_matrix(toks):
    """M: n x 200 (row-major from the bottom row)."""
    return np.stack([t.mask.reshape(-1) for t in toks]).astype(np.float32)


def check_partition(toks):
    base = [t for t in toks if t.base]
    cover = np.sum([t.mask.astype(int) for t in base], axis=0)
    return bool((cover == 1).all())


def structures_for(boards40, exe=None, pieces=None):
    """Runs `bot structures` over replay boards (one subprocess for the whole list). With
    `pieces` (per board: (piece, queue, hold)) it also returns covered piece slots and setups."""
    exe = exe or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bot", "target", "release", "bot.exe")
    reqs = []
    for i, b in enumerate(boards40):
        r = {"board": b}
        if pieces is not None:
            pc, q, hd = pieces[i]
            r.update({"piece": pc.upper(), "queue": [x.upper() for x in q], "hold": hd.upper() if hd else None})
        reqs.append(r)
    inp = "".join(json.dumps(r) + "\n" for r in reqs)
    res = subprocess.run([exe, "structures"], input=inp, capture_output=True, text=True, check=True)
    return [json.loads(l)["structures"] for l in res.stdout.splitlines() if l.strip()]


SYMBOL = {"sky": ".", "well": "W", "band": "B"}


def render(b, toks):
    """Text picture: board, base partition, and each overlay's cells."""
    base = [t for t in toks if t.base]
    over = [t for t in toks if t.kind in OVERLAY_KINDS]
    label = np.full((ROWS, COLS), "?", dtype="<U1")
    names = []
    play_i = 0
    for t in base:
        if t.kind == "play":
            sym = "0123456789abcdefghij"[play_i]
            play_i += 1
        else:
            sym = SYMBOL[t.kind]
        label[t.mask] = sym
        names.append("%s=%s%s" % (sym, t.kind, "" if not t.extras or t.kind == "sky" else
                                  " " + ",".join("%s:%s" % kv for kv in t.extras.items() if kv[0] not in ("merged",))))
    top = max(int(heights(b).max()) + 2, 6)
    cols = [["board"], ["base"]] + [["%d" % (i + 1)] for i in range(len(over))]
    lines = []
    for r in range(min(top, ROWS) - 1, -1, -1):
        parts = ["".join("#" if b[r, c] else "." for c in range(COLS)),
                 "".join(label[r])]
        for t in over:
            parts.append("".join("o" if t.mask[r, c] else ("#" if b[r, c] else ".") for c in range(COLS)))
        lines.append("%2d  " % (r + 1) + "  ".join(parts))
    head = "    " + "  ".join(n[0].ljust(COLS) for n in cols)
    out = [head] + lines + ["base: " + "; ".join(names)]
    for i, t in enumerate(over):
        out.append("overlay %d: %s %s" % (i + 1, t.kind, t.extras))
    pieces = [t for t in toks if t.kind in PIECE_KINDS]
    if pieces:
        out.append("pieces: " + " ".join("%s=%s" % (t.kind, t.extras["piece"]) for t in pieces))
    out.append("tokens: %d board + %d overlay + %d piece = %d" % (len(base), len(over), len(pieces), len(toks)))
    return "\n".join(out)
