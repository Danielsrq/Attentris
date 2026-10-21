"""README figures as static HTML pages (inline SVG), rendered to PNG with headless Chrome.

    python attentris/readme_figs.py model --page attentris/attn_shap_v51_d128_dagger5b.html --pos 1 2 26 --out DIR
    python attentris/readme_figs.py results --out DIR

`model`: one position of an attention / SHAP page (attn_viz.py + shap_page.py) per figure: (a) the position and the
model's top moves, (b) the 4 cross-attention heads for the top move, (c) head 2 on the top-1 vs the top-2 move,
(d) the FLAT-2D geometry bias, (e) token SHAP toward the top move. `results`: the d128 ablations and the group SHAP
(numbers from NOTES.md, "Ablations at d = 128" and "Group SHAP at play time"). Each page is also saved as a PNG
when Chrome is found (--chrome PATH to override).
"""
import argparse
import json
import os
import shutil
import subprocess

HEAT, COOL = (214, 84, 20), (38, 104, 196)
EMPTY, FILLED, GRID, FG, MUTED, LINE, PICK, MOVE = "#ebeae4", "#3a3f47", "#ffffff", "#1d2127", "#5c626b", "#d8d5ca", "#2c55c9", "#111418"
HEAD_ROLE = ["generic", "placement cells", "piece queue", "surroundings"]
SPIN = ["", " mini", "-spin"]
PIECE_CELLS = {"I": [(0, 0), (1, 0), (2, 0), (3, 0)], "O": [(1, 0), (2, 0), (1, 1), (2, 1)], "T": [(0, 0), (1, 0), (2, 0), (1, 1)],
               "S": [(0, 0), (1, 0), (1, 1), (2, 1)], "Z": [(1, 0), (2, 0), (0, 1), (1, 1)], "J": [(0, 0), (1, 0), (2, 0), (0, 1)],
               "L": [(0, 0), (1, 0), (2, 0), (2, 1)]}
CSS = """
body { margin: 0; background: #fff; color: %s; font: 14px/1.4 "Segoe UI", system-ui, sans-serif; }
.fig { padding: 22px 26px; display: inline-block; }
h1 { font-size: 19px; margin: 0 0 2px; } .lede { color: %s; margin: 0 0 16px; font-size: 13.5px; }
.grid { display: grid; gap: 18px 16px; align-items: start; }
.panel { display: grid; gap: 4px; justify-items: center; }
.panel h3 { margin: 0; font-size: 13px; font-weight: 600; } .panel .s { color: %s; font-size: 12px; text-align: center; }
.tag { font-weight: 700; margin-right: 4px; }
.group { border-top: 1px solid %s; padding-top: 10px; }
.gh { font-size: 13px; font-weight: 600; margin: 0 0 8px; } .gh span { color: %s; font-weight: 400; }
.legend { display: flex; gap: 14px; flex-wrap: wrap; color: %s; font-size: 12px; margin-top: 12px; align-items: center; }
.sw { display: inline-block; width: 12px; height: 12px; border-radius: 3px; vertical-align: -2px; margin-right: 4px; }
""" % (FG, MUTED, MUTED, LINE, MUTED, MUTED)


def load_page(path):
    s = open(path, encoding="utf-8").read()
    a = s.index("const DATA = ") + len("const DATA = ")
    return json.JSONDecoder().raw_decode(s[a:])[0]


def rgba(c, a):
    return "rgba(%d,%d,%d,%.3f)" % (c + (a,))


def cand_label(c):
    s = c["piece"] + (" (hold)" if c["hold"] else "") + SPIN[c["spin"]]
    if c["spin"] == 2 and c["piece"] == "T":
        s = "T-spin " + ["", "single", "double", "triple"][c["lines"]]
    elif c["lines"]:
        s += ", clears %d" % c["lines"]
    if c["attack"]:
        s += " (sends %d)" % c["attack"]
    return s


def cols_of(c):
    xs = [i % 10 for i in c["cells"]]
    return (" · col %d" % xs[0]) if min(xs) == max(xs) else " · cols %d-%d" % (min(xs), max(xs))


def outline(cells, cs, inset, rows=20):
    """SVG path around a set of cells (index = row * 10 + col, row 0 = bottom)."""
    st, d = set(cells), ""
    for i in cells:
        r, c = divmod(i, 10)
        x0, x1 = c * cs + inset, (c + 1) * cs - inset
        y0, y1 = (rows - 1 - r) * cs + inset, (rows - r) * cs - inset
        if i + 10 not in st:
            d += "M%.1f %.1fH%.1f" % (x0, y0, x1)
        if i - 10 not in st:
            d += "M%.1f %.1fH%.1f" % (x0, y1, x1)
        if c == 0 or i - 1 not in st:
            d += "M%.1f %.1fV%.1f" % (x0, y0, y1)
        if c == 9 or i + 1 not in st:
            d += "M%.1f %.1fV%.1f" % (x1, y0, y1)
    return d


def board_svg(p, heat=None, signed=False, cs=13, show=None, dashed=None, rows=20):
    """The board with a per-cell heat map (unsigned: 0..max, signed: -max..max), the shown move solid, another dashed."""
    w, h = 10 * cs, rows * cs
    out = ['<svg width="%d" height="%d" viewBox="-1 -1 %d %d">' % (w + 2, h + 2, w + 2, h + 2)]
    mx = max((abs(v) for v in heat), default=0) or 1 if heat is not None else 1
    for i in range(rows * 10):
        r, c = divmod(i, 10)
        x, y = c * cs, (rows - 1 - r) * cs
        out.append('<rect x="%d" y="%d" width="%d" height="%d" fill="%s" stroke="%s" stroke-width="1"/>' % (x, y, cs, cs, EMPTY, GRID))
        if heat is not None and heat[i]:
            v = heat[i] / mx
            if signed:
                out.append('<rect x="%d" y="%d" width="%d" height="%d" fill="%s"/>' % (x, y, cs, cs, rgba(HEAT if v > 0 else COOL, 0.9 * abs(v))))
            elif v > 0:
                out.append('<rect x="%d" y="%d" width="%d" height="%d" fill="%s"/>' % (x, y, cs, cs, rgba(HEAT, 0.9 * v)))
        if p["board"][i]:
            out.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="1.5" fill="%s" opacity="0.82"/>' % (x + 2.5, y + 2.5, cs - 5, cs - 5, FILLED))
    if dashed:
        out.append('<path d="%s" stroke="%s" stroke-width="1.6" stroke-dasharray="3 2.5" fill="none"/>' % (outline(dashed["cells"], cs, 1.2, rows), MOVE))
    if show:
        out.append('<path d="%s" stroke="%s" stroke-width="2.4" fill="none"/>' % (outline(show["cells"], cs, 2.2, rows), PICK))
    out.append("</svg>")
    return "".join(out)


def mini_piece(piece, cs=7):
    if not piece:
        return '<svg width="%d" height="%d"></svg>' % (4 * cs, 2 * cs)
    return '<svg width="%d" height="%d">' % (4 * cs + 2, 2 * cs + 2) + "".join(
        '<rect x="%d" y="%d" width="%d" height="%d" rx="1" fill="%s"/>' % (1 + x * cs, 1 + (1 - y) * cs, cs - 1, cs - 1, FILLED)
        for x, y in PIECE_CELLS[piece]) + "</svg>"


def per_cell(p, vals, how):
    """Token values onto cells. dens: each token's value spread over its cells and summed where tokens overlap
    (attention, SHAP); mean: the mean over the tokens covering the cell (a bias, which does not add up)."""
    heat, cover = [0.0] * 200, [0] * 200
    for t, v in zip(p["tokens"], vals):
        if v is None:
            continue
        for i in t["cells"]:
            heat[i] += v / len(t["cells"]) if how == "dens" else v
            cover[i] += 1
    if how == "mean":
        heat = [h / n if n else 0.0 for h, n in zip(heat, cover)]
    return heat


def piece_share(p, vals):
    return sum(v for t, v in zip(p["tokens"], vals) if not t["cells"])


def head_vals(cand, key, h):
    return cand["H"][key][h] if h is not None else cand[key]


def geo_rel(p, cand, h=None):
    g = head_vals(cand, "geo", h)
    board = [j for j, t in enumerate(p["tokens"]) if t["cells"]]
    m = sum(g[j] for j in board) / max(1, len(board))
    return [g[j] - m if p["tokens"][j]["cells"] else None for j in range(len(g))]


def panel(tag, title, sub, svg):
    return '<div class="panel"><h3><span class="tag">%s</span>%s</h3>%s<div class="s">%s</div></div>' % (tag, title, svg, sub)


def fig_model(d, pos, h2=1):
    p = d["positions"][pos - 1]
    top = p["top"]
    t1, t2 = top[0], top[1]
    bot = p["chosen"]
    played = None        # the teacher's move is left out: the figure is about the model's own ranking
    BS = 11                                   # cell size of the small boards
    # rows drawn: the stack and the shown moves, plus 4 rows of air (at least 10)
    used = [i for i in range(200) if p["board"][i]] + [i for c in (t1, t2, bot) for i in c["cells"]]
    R = min(20, max(10, max(i // 10 for i in used) + 5))
    # (a) the position: queue and the model's top 3
    q = {t["kind"]: t["piece"] for t in p["tokens"] if not t["cells"]}
    slot = lambda k, lab: '<div style="display:grid;justify-items:center;gap:2px">%s<span>%s</span></div>' % (mini_piece(q.get(k)), lab)
    qhtml = ('<div style="display:flex;gap:14px;align-items:end;font-size:11px;color:%s">%s%s<div style="display:grid;justify-items:center;gap:2px">'
             '<div style="display:flex;gap:6px">%s</div><span>next</span></div></div>') % (
        MUTED, slot("current", "now"), slot("hold", "hold"), "".join(mini_piece(q.get("preview%d" % k)) for k in range(1, 6) if "preview%d" % k in q))
    probs = "".join('<div style="display:grid;grid-template-columns:14px 200px 60px 34px;gap:6px;align-items:center;font-size:12px">'
                    '<span style="color:%s">%d</span><span>%s</span><span style="height:8px;border-radius:2px;background:#e0763a;width:%.0f%%"></span>'
                    '<span style="text-align:right;font-variant-numeric:tabular-nums">%.0f%%</span></div>' % (MUTED, k + 1, cand_label(c) + cols_of(c), 100 * c["p"] / top[0]["p"], 100 * c["p"])
                    for k, c in enumerate(p["probs"][:3]))
    a = ('<div class="panel"><h3><span class="tag">a</span>Position</h3>%s%s<div style="display:grid;gap:3px;margin-top:8px">'
         '<div class="s" style="text-align:left">the model\'s top moves (of %d legal)</div>%s</div></div>') % (
        board_svg(p, show=t1, dashed=played, cs=16, rows=R), qhtml, p["n_cand"], probs)
    # (b) the 4 heads for the top move
    heads = "".join(panel("", "Head %d" % (h + 1), "usually: %s<br>on the pieces: %.0f%%" % (HEAD_ROLE[h], 100 * piece_share(p, head_vals(t1, "attn", h))),
                          board_svg(p, per_cell(p, head_vals(t1, "attn", h), "dens"), show=t1, cs=BS, rows=R)) for h in range(4))
    b = ('<div class="panel" style="align-content:start;justify-items:start"><h3><span class="tag">b</span>The 4 cross-attention heads, move #1</h3>'
         '<div class="grid" style="grid-template-columns:repeat(4,auto);gap:12px 22px">%s</div></div>') % heads
    # (c) head 2 on the top-1 vs the top-2 move
    c = ('<div class="panel" style="align-content:start"><h3><span class="tag">c</span>Head %d: move #1 vs move #2</h3>'
         '<div class="grid" style="grid-template-columns:repeat(2,auto)">%s%s</div></div>') % (
        h2 + 1,
        panel("", "", "#1 %s<br>p = %.0f%%" % (cand_label(t1), 100 * t1["p"]), board_svg(p, per_cell(p, head_vals(t1, "attn", h2), "dens"), show=t1, cs=BS, rows=R)),
        panel("", "", "#2 %s<br>p = %.0f%%" % (cand_label(t2), 100 * t2["p"]), board_svg(p, per_cell(p, head_vals(t2, "attn", h2), "dens"), show=t2, cs=BS, rows=R)))
    # (d) geometry bias, (e) SHAP, both for move #1
    dpanel = panel("", "", "<b>d</b> FLAT-2D bias alone:<br>where the move looks from geometry",
                   board_svg(p, per_cell(p, t1["gattn"], "dens"), show=t1, cs=BS, rows=R))
    sh = t1.get("shap")
    epanel = panel("", "", "<b>e</b> Token SHAP of p(move #1)<br>(queue %+.2f, total %+.2f)" % (
        piece_share(p, sh), t1["shap_all"] - t1["shap_none"]), board_svg(p, per_cell(p, sh, "dens"), signed=True, show=t1, cs=BS, rows=R)) if sh else ""
    de = ('<div class="panel" style="align-content:start"><h3>Geometry and attribution, move #1</h3>'
          '<div class="grid" style="grid-template-columns:repeat(2,auto)">%s%s</div></div>') % (dpanel, epanel)
    legend = ('<div class="legend"><span><svg width="22" height="12"><path d="M1 6H21" stroke="%s" stroke-width="2.4"/></svg>move shown</span>%s'
              '<span><span class="sw" style="background:%s"></span>attention (b, c, d) · pushes toward the move (e)</span>'
              '<span><span class="sw" style="background:%s"></span>pushes away (e)</span>'
              '<span>Heat per board cell: a token\'s value spread over its cells; each panel scaled to its own max.</span></div>') % (
        PICK, '<span><svg width="22" height="12"><path d="M1 6H21" stroke="%s" stroke-width="1.6" stroke-dasharray="3 2.5"/></svg>teacher\'s move</span>' % MOVE if played else "",
        rgba(HEAT, 0.8), rgba(COOL, 0.8))
    body = ('<div class="fig"><h1>What Attentris looks at</h1><p class="lede">%s. Model %s: each candidate move is a token that '
            'cross-attends over the board\'s structure tokens, then gets a probability in one forward pass.</p>'
            '<div class="grid" style="grid-template-columns:auto auto;gap:22px 36px"><div style="grid-row:span 2">%s</div>%s'
            '<div class="grid" style="grid-template-columns:auto auto;gap:18px 36px;justify-content:start">%s%s</div></div>%s</div>') % (
        p["why"][0].upper() + p["why"][1:], d.get("model", "").split(" (")[0], a, b, c, de, legend)
    return body


# d128 ablations through the full lineage (NOTES.md, "Ablations at d = 128"): pooled McNemar over 8 garbage settings,
# games lost only by the ablated model vs only by the baseline, 128 paired games per setting.
ABLATIONS = [("all overlay tokens", "nooverlay", 236, 94, "3e-15"), ("non-T slots (fit_s/z/l/j/i)", "nofitx", 201, 93, "3e-10"),
             ("FLAT-2D geometry bias", "noflat", 186, 120, "0.0002"), ("setup tokens", "nosetup", 175, 132, "0.016"),
             ("T slot (fit_t)", "nofitt", 170, 128, "0.017"), ("pits (pit1-4, rect12)", "nopit", 159, 148, "0.57")]
APP_NONE = [("baseline", 0.739), ("nooverlay", 0.729), ("nofitx", 0.715), ("noflat", 0.724), ("nosetup", 0.722), ("nofitt", 0.728), ("nopit", 0.754)]
# Group SHAP at play time (NOTES.md): phi deaths per 128 games [90%], and the retrained ablation's extra deaths
# (mean of the 4 depth-1 settings); None = no retrained model.
GROUPS = [("non-T slots", -51.5, -53.3, -49.6, 19.0), ("pits", -23.6, -24.7, -22.5, 2.0), ("well", -3.3, -4.2, -2.3, None),
          ("T slot", -1.7, -2.8, -0.7, 6.8), ("T structures", -1.1, -2.1, 0.0, None), ("setups", -0.1, -1.1, 0.9, 9.0)]


def fig_results():
    # (a) ablations: extra deaths per 1024 paired games (8 settings x 128)
    W, L, R, rowh = 620, 190, 210, 30
    mx = max(a - b for _, _, a, b, _ in ABLATIONS)
    s = ['<svg width="%d" height="%d">' % (W, 40 + rowh * len(ABLATIONS))]
    for k, (name, var, a, b, pv) in enumerate(ABLATIONS):
        y = 22 + k * rowh
        bw = (W - L - R) * (a - b) / mx
        sig = float(pv.replace("e", "E")) < 0.05
        s.append('<text x="%d" y="%d" text-anchor="end" font-size="12.5" fill="%s">%s</text>' % (L - 10, y + 14, FG, name))
        s.append('<rect x="%d" y="%d" width="%.1f" height="18" rx="3" fill="%s"/>' % (L, y, max(bw, 1), rgba(HEAT, 0.85 if sig else 0.3)))
        s.append('<text x="%.1f" y="%d" font-size="12" fill="%s">+%d <tspan fill="%s">(%d vs %d) &#160;p = %s</tspan></text>' % (
            L + bw + 6, y + 14, FG, a - b, MUTED, a, b, pv))
    s.append('<text x="%d" y="%d" font-size="11.5" fill="%s">net extra games lost, out of 1,024 paired games</text>' % (
        L, 32 + rowh * len(ABLATIONS), MUTED))
    s.append("</svg>")
    apps = " · ".join("%s %.3f" % (n, v) for n, v in APP_NONE)
    a = ('<div class="panel" style="justify-items:start"><h3><span class="tag">a</span>Ablations: retrained without one component</h3>'
         '<div class="s" style="text-align:left;max-width:600px">Each model is retrained without the component through the full training lineage, then it and the '
         'baseline play the same 1,024 games (8 garbage settings x 128 seeds). Bar: games only the ablated model lost minus games only the '
         'baseline lost (e.g. 236 vs 94 = +142). p: exact McNemar test, i.e. how likely so lopsided a split would be if both models were '
         'equally good. Faded: not significant (p &gt; 0.05).</div>%s<div class="s" style="text-align:left;max-width:600px">Clean-board APP barely moves (%s): '
         'every model stacks well enough on a clean board; the cost shows up <b>under garbage</b>, where the stack gets hard to keep in shape.</div></div>') % ("".join(s), apps)
    # (b) group SHAP: deaths prevented, with the retrained ablation's extra deaths as a marker
    W, L, R = 560, 110, 60
    mx = 55.0
    X = lambda v: L + (W - L - R) * v / mx
    s = ['<svg width="%d" height="%d">' % (W, 56 + rowh * len(GROUPS))]
    for v in (0, 10, 20, 30, 40, 50):
        s.append('<line x1="%.1f" x2="%.1f" y1="14" y2="%d" stroke="%s"/>' % (X(v), X(v), 18 + rowh * len(GROUPS), LINE))
        s.append('<text x="%.1f" y="%d" text-anchor="middle" font-size="11" fill="%s">%d</text>' % (X(v), 32 + rowh * len(GROUPS), MUTED, v))
    for k, (name, phi, lo, hi, ret) in enumerate(GROUPS):
        y = 20 + k * rowh
        s.append('<text x="%d" y="%d" text-anchor="end" font-size="12.5" fill="%s">%s</text>' % (L - 10, y + 14, FG, name))
        s.append('<rect x="%.1f" y="%d" width="%.1f" height="18" rx="3" fill="%s"/>' % (X(0), y, max(X(-phi) - X(0), 1), rgba(COOL, 0.8)))
        s.append('<line x1="%.1f" x2="%.1f" y1="%d" y2="%d" stroke="%s" stroke-width="1.5"/>' % (X(max(-hi, 0)), X(-lo), y + 9, y + 9, FG))
        if ret is not None:
            s.append('<path d="M%.1f %dl6 6l-6 6l-6 -6z" fill="%s" stroke="#fff"/>' % (X(ret), y + 3, rgba(HEAT, 1)))
    s.append('<text x="%d" y="%d" font-size="11.5" fill="%s">deaths per 128 games</text>' % (L, 50 + rowh * len(GROUPS), MUTED))
    s.append("</svg>")
    b = ('<div class="panel" style="justify-items:start"><h3><span class="tag">b</span>Group SHAP at play time vs retraining</h3>'
         '<div class="s" style="text-align:left;max-width:560px">one model trained with all tokens; token groups blanked in every combination '
         '(64 coalitions x 4 garbage settings x 128 paired games)</div>%s'
         '<div class="legend" style="margin-top:2px"><span><span class="sw" style="background:%s"></span>deaths the group prevents (Shapley value, 90%% CI)</span>'
         '<span><svg width="12" height="12"><path d="M6 0l6 6l-6 6l-6 -6z" fill="%s"/></svg> extra deaths when retrained without it</span></div>'
         '<div class="s" style="text-align:left;max-width:560px">The model leans on the non-T slots and the pits. Retrained without pits it reads '
         'the same from the cells (+2, n.s.); without the non-T slots it cannot (+19): <b>the slots carry what the cells give only at a cost.</b></div></div>') % (
        "".join(s), rgba(COOL, 0.8), rgba(HEAT, 1))
    return ('<div class="fig"><h1>What matters: survival comes from the slots</h1><p class="lede">Attentris v5.1 d=128 (1.06M params), '
            'imitating a CEM-tuned search bot; play tests at depth 1 and depth 2 under 80/20 and 50/50 garbage.</p>'
            '<div class="grid" style="grid-template-columns:auto auto;gap:18px 40px">%s%s</div></div>') % (a, b)


def write(path, title, body, chrome, width):
    html = '<!doctype html><html><head><meta charset="utf-8"><title>%s</title><style>%s</style></head><body>%s</body></html>' % (title, CSS, body)
    open(path, "w", encoding="utf-8").write(html)
    if chrome:
        png = path[:-5] + ".png"
        subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                        "--window-size=%d,%d" % width, "--screenshot=" + os.path.abspath(png), "file:///" + os.path.abspath(path).replace("\\", "/")],
                       check=True, capture_output=True)
        trim(png)
        print("wrote", png)
    else:
        print("wrote", path)


def trim(png, pad=24):
    """Crop the white margin (the window is sized generously); needs Pillow, else the PNG stays as rendered."""
    try:
        from PIL import Image, ImageChops
    except ImportError:
        return
    im = Image.open(png).convert("RGB")
    box = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))).getbbox()
    if box:
        im.crop((max(0, box[0] - pad), max(0, box[1] - pad), min(im.width, box[2] + pad), min(im.height, box[3] + pad))).save(png)


def find_chrome(arg):
    for c in [arg, shutil.which("chrome"), shutil.which("google-chrome"), shutil.which("chromium"),
              r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]:
        if c and os.path.exists(c):
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["model", "results"])
    ap.add_argument("--page", default=os.path.join(os.path.dirname(__file__), "attn_shap_v51_d128_dagger5b.html"))
    ap.add_argument("--pos", type=int, nargs="+", default=[1], help="1-based positions of the page")
    ap.add_argument("--head", type=int, default=2, help="head for panel (c), 1-based")
    ap.add_argument("--out", default="docs/img")
    ap.add_argument("--chrome")
    ap.add_argument("--size", default=None, help="window W,H in CSS px")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    chrome = find_chrome(args.chrome)
    if args.what == "model":
        d = load_page(args.page)
        size = tuple(map(int, args.size.split(","))) if args.size else (1000, 900)
        for pos in args.pos:
            write(os.path.join(args.out, "fig_model_p%d%s.html" % (pos, "" if args.head == 2 else "_h%d" % args.head)), "Attentris Attention", fig_model(d, pos, args.head - 1), chrome, size)
    else:
        size = tuple(map(int, args.size.split(","))) if args.size else (1260, 420)
        write(os.path.join(args.out, "fig_results.html"), "Attentris Results", fig_results(), chrome, size)


if __name__ == "__main__":
    main()
