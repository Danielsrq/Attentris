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
    earlier = read_log(path("..", "cem_norm_vs_raw", "normalised.txt.log"))
    if earlier:
        data["runs"]["cem0"] = {"label": "CEM (earlier run)", "log": earlier, "dist": None}

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

    with open(path("dashboard.template.html"), encoding="utf-8") as f:
        html = f.read()
    blob = json.dumps(data, separators=(",", ":"))
    html = html.replace("/*__DATA__*/null", blob)
    with open(path("dashboard.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote dashboard.html ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
