"""Builds dashboard.html: the tuning runs and the weight landscape, data embedded.

    python build_dashboard.py

Reads (all in this folder unless noted):
  cem.txt.log / cma.txt.log                learning curves
  cem.txt.dist.csv / cma.txt.dist.csv      per-generation mean + covariance (normalised units)
  trajectory_cem.txt.json / ..._cma ...    samples and clouds projected on each run's plane
  slices.csv, plane.csv                    landscape views 1 and 3
  heldout_*.txt                            held-out benchmark rows (optional)
  ../cem_norm_vs_raw/normalised.txt.log    the earlier CEM run, for run-to-run spread
"""
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def path(*p):
    return os.path.join(HERE, *p)


def read_log(p):
    if not os.path.exists(p):
        return None
    rows = []
    with open(p) as f:
        head = f.readline().split()
        for line in f:
            v = line.split()
            if len(v) >= 4:
                rows.append({head[i]: float(v[i]) for i in range(min(len(head), len(v)))})
    return rows


def read_dist(p):
    if not os.path.exists(p):
        return None
    means, covs = {}, {}
    with open(p) as f:
        f.readline()
        for line in f:
            g, kind, vals = line.rstrip("\n").split(",", 2)
            arr = np.array([float(x) for x in vals.split(",")])
            (means if kind == "mean" else covs)[int(g)] = arr
    gens = sorted(means)
    n = len(means[gens[0]])
    out = {"gens": gens, "mean": [means[g].round(4).tolist() for g in gens]}
    # Per generation: per-weight spread (sqrt of the diagonal) and the spectrum of the
    # covariance -- the std along each principal axis, largest first.
    spread, spectrum, axes = [], [], []
    for g in gens:
        c = covs[g].reshape(n, n)
        c = 0.5 * (c + c.T)
        spread.append(np.sqrt(np.clip(np.diag(c), 0, None)).round(4).tolist())
        vals, vecs = np.linalg.eigh(c)
        order = np.argsort(vals)[::-1]
        spectrum.append(np.sqrt(np.clip(vals[order], 0, None)).round(5).tolist())
        axes.append(vecs[:, order])
    out["spread"] = spread
    out["spectrum"] = spectrum
    # Loadings of the final distribution's widest and narrowest principal axes.
    last = axes[-1]
    out["axis_widest"] = last[:, 0].round(4).tolist()
    out["axis_narrowest"] = last[:, -1].round(4).tolist()
    return out


def read_csv(p):
    if not os.path.exists(p):
        return None
    with open(p) as f:
        head = f.readline().strip().split(",")
        return [dict(zip(head, line.strip().split(","))) for line in f if line.strip()]


def read_heldout():
    rows = []
    for name in sorted(os.listdir(HERE)):
        m = re.match(r"heldout_(.+)\.txt$", name)
        if not m:
            continue
        with open(path(name)) as f:
            first = f.readline().split()
        # name APP died pcs quad TSD TST TSS mini spin 1-3 gbclr b2b rep wnd ms
        rows.append({"name": m.group(1), "app": float(first[1]), "died": first[2], "quad": float(first[4]),
                     "tsd": float(first[5]), "tss": float(first[7]), "plain": float(first[10]), "b2b": int(first[12])})
    return rows


BURST = os.path.join(HERE, "..", "burst")


def read_weights_file(p):
    """Name -> value from a normalised weights file (tune output or weights/*.txt)."""
    out = {}
    with open(p) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if line:
                k, v = line.split()[:2]
                out[k] = float(v)
    return out


def read_burst():
    """The burst-rules study in ../burst: held-out rows, per-piece attack sequences,
    Shapley estimates, tune logs and weights."""
    if not os.path.isdir(BURST):
        return None
    out = {"heldout": [], "seqs": {}, "shapley": {}, "log": {}, "weights": {}}
    for name in sorted(os.listdir(BURST)):
        m = re.match(r"heldout_(.+)_(burst|normal)\.txt$", name)
        if not m:
            continue
        with open(os.path.join(BURST, name)) as f:
            lines = f.read().splitlines()
        v = lines[0].split()
        row = {"name": m.group(1), "mode": m.group(2), "app": float(v[1]), "died": v[2], "quad": float(v[4]),
               "tsd": float(v[5]), "tst": float(v[6]), "tss": float(v[7]), "plain": float(v[10]), "b2b": int(v[12])}
        b = next((l for l in lines if l.startswith("burst")), "")
        mm = re.search(r"7-pc window APP best (\S+) p90 (\S+), 14-pc window APP best (\S+) p90 (\S+); pair rate (\S+)% of (\d+) qualifying clears; burst-pair attack share (\S+)%", b)
        if mm:
            g = [float(x) for x in mm.groups()]
            row.update(w7best=g[0], w7p90=g[1], w14best=g[2], w14p90=g[3], pair_rate=g[4], qual=int(g[5]), pair_share=g[6])
        out["heldout"].append(row)
        seq = os.path.join(BURST, name[:-4] + ".seq.csv")
        if os.path.exists(seq):
            games = {}
            with open(seq) as f:
                f.readline()
                for line in f:
                    gi, _n, atk, _q, _d = line.strip().split(",")
                    games.setdefault(int(gi), []).append(int(atk))
            out["seqs"][m.group(1) + "|" + m.group(2)] = [games[k] for k in sorted(games)]
    for key in ("burst", "normal"):
        p = os.path.join(BURST, f"shapley_{key}.csv")
        if not os.path.exists(p):
            continue
        rows, meta = [], {}
        with open(p) as f:
            f.readline()
            for line in f:
                parts = line.strip().split(",")
                if parts[0].startswith("#"):
                    meta[parts[0][1:]] = float(parts[1])
                else:
                    rows.append({"feature": parts[0], "phi": float(parts[1]), "se": float(parts[2]),
                                 "weight": float(parts[3]), "baseline": float(parts[4])})
        out["shapley"][key] = {"rows": rows, "v_empty": meta.get("v_empty", 0), "v_full": meta.get("v_full", 0),
                               "perms": int(meta.get("perms", 0))}
    for key in ("burst", "normal_frozen"):
        log = read_log(os.path.join(BURST, f"{key}.txt.log"))
        if log:
            out["log"][key] = log
    files = {"burst": os.path.join(BURST, "burst.txt"), "normal_frozen": os.path.join(BURST, "normal_frozen.txt"),
             "normal_txt": os.path.join(HERE, "..", "..", "weights", "normal.txt")}
    found = {k: read_weights_file(p) for k, p in files.items() if os.path.exists(p)}
    # Feature order from a tune output (all 30 names); normal.txt predates burst_armed.
    names = list(found.get("burst") or found.get("normal_frozen") or found.get("normal_txt") or [])
    out["weights"] = {"names": names, **{k: [w.get(n, 0.0) for n in names] for k, w in found.items()}}
    return out


DESCRIPTIONS = {
    "holes": "empty cells with something above them",
    "hole_rows": "rows that contain at least one hole",
    "cover": "filled cells stacked above each column's topmost hole",
    "bump": "height steps between neighbouring columns",
    "bump_sq": "the same height steps, squared (punishes big steps)",
    "row_trans": "filled/empty changes along each row (fragmentation)",
    "col_trans": "filled/empty changes down each column",
    "max_height": "height of the tallest column",
    "danger": "(stack height - 10) squared, zero below 10 rows",
    "well_rows": "rows full except the deepest column, up to 4: a quad in waiting",
    "extra_wells": "other 1-wide notches deeper than 2 (needing a second I)",
    "t_single": "a T-slot that would clear 1 row",
    "t_double": "a T-slot that would clear 2 rows (TSD)",
    "t_triple": "a T-slot that would clear 3 rows (TST)",
    "t_fill": "how full the best T-slot's rows are",
    "b2b": "a back-to-back chain is live",
    "b2b_level": "chain length, capped at 8",
    "b2b_next": "bonus the next difficult clear would earn",
    "wounds": "wound lines on the board (all-spin only; always 0 here)",
    "ready_repeat": "a ready clear would repeat the last type (all-spin only)",
    "sz_single": "an S/Z spin spot that would clear 1 row",
    "sz_double": "an S/Z spin spot that would clear 2 rows",
    "sz_triple": "an S/Z spin spot that would clear 3 rows",
    "lj_single": "an L/J spin spot that would clear 1 row",
    "lj_double": "an L/J spin spot that would clear 2 rows",
    "lj_triple": "an L/J spin spot that would clear 3 rows",
    "i_single": "an I spin spot that would clear 1 row",
    "i_double": "an I spin spot that would clear 2 rows",
    "i_triple": "an I spin spot that would clear 3 rows",
    "burst_armed": "the last placement was a qualifying clear (burst rules)",
}


def main():
    names = None
    data = {"runs": {}}
    for key, label in [("cem", "CEM"), ("cma", "CMA-ES")]:
        log = read_log(path(f"{key}.txt.log"))
        if log is None:
            continue
        run = {"label": label, "log": log, "dist": read_dist(path(f"{key}.txt.dist.csv"))}
        tp = path(f"trajectory_{key}.txt.json")
        if os.path.exists(tp):
            with open(tp) as f:
                run["trajectory"] = json.load(f)
                names = run["trajectory"]["names"]
        data["runs"][key] = run
    # The earlier CEM run (../cem_norm_vs_raw) is reproduced exactly by cem.txt -- tuning
    # is deterministic -- so it is not plotted as a separate series.

    slices = read_csv(path("slices.csv"))
    if slices:
        by = {}
        for r in slices:
            by.setdefault(r["feature"], []).append((int(r["k"]), float(r["weight"]), float(r["fitness"])))
        data["slices"] = [{"feature": k, "points": sorted(v)} for k, v in by.items()]
    plane = read_csv(path("plane.csv"))
    if plane:
        data["plane"] = [{"kind": r["kind"], "x": float(r["x"]), "y": float(r["y"]), "f": float(r["fitness"])} for r in plane]
    data["heldout"] = read_heldout()
    # Synthetic benchmark tables: function, noise, CEM, CMA-ES per budget.
    synth = []
    for name in sorted(os.listdir(HERE)):
        m = re.match(r"synth_(\d+)\.txt$", name)
        if not m:
            continue
        gens = int(m.group(1))
        with open(path(name)) as f:
            for line in f:
                mm = re.match(r"^(.+?)\s+(\d+)%\s+(\S+)\s+(\S+)\s+(CEM|CMA-ES)\s*$", line.rstrip())
                if mm:
                    synth.append({"gens": gens, "fn": mm.group(1).strip(), "noise": int(mm.group(2)),
                                  "cem": float(mm.group(3)), "cma": float(mm.group(4))})
    data["synth"] = synth
    if names is None:
        # Fall back to the slices' feature order.
        names = [s["feature"] for s in data.get("slices", [])]
    data["names"] = names
    data["descriptions"] = DESCRIPTIONS
    # Start and final weights (normalised units) from the distribution logs.
    weights = {}
    for key in ("cem", "cma"):
        run = data["runs"].get(key)
        if run and run.get("dist"):
            weights.setdefault("start", run["dist"]["mean"][0])
            weights[key] = run["dist"]["mean"][-1]
    data["weights"] = weights
    data["burst"] = read_burst()

    with open(path("dashboard.template.html"), encoding="utf-8") as f:
        html = f.read()
    blob = json.dumps(data, separators=(",", ":"))
    html = html.replace("/*__DATA__*/null", blob)
    with open(path("dashboard.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote dashboard.html ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
