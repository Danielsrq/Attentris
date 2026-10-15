"""Can the teacher (CEM linear V, depth 6 + beam 64 search) save the games a model lost? For each
death in a play.py run, the model replays its game (deterministic) up to k moves before the death,
then the teacher's full search takes over on the same piece sequence and the same garbage (the
garbage schedule depends on the seed only); also the teacher alone from move 0 on that seed.
Reports survival (500 pieces) and the final stack.

    python attentris/takeover.py --deaths v51:data/deaths_v51.txt tome_v4:data/deaths_tome_v4.txt
    python attentris/takeover.py --deaths v51_dagger_ft:data/dagger3/v51_dagger_ft.games --garbage 10,3-4,cheese0.2,0.2 \
        --teacher-weights bot/results/survive/survive.txt --sample 12
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PYC = os.path.join(HERE, "..", "external", "tetrisbattle", ".venv", *(("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")))
if not os.path.exists(PYC):   # no TetrisBattle copy (Linux training machine): use this python
    PYC = sys.executable


def run(model, seed, takeover, garbage="", teacher_weights="", style="normal"):
    cmd = [PYC, os.path.join(HERE, "play.py"), "--model", model, "--games", "1", "--jobs", "1", "--seed", str(seed),
           "--depth", "1", "--takeover", str(takeover), "--out", os.devnull, "--style", style]
    if garbage:
        cmd += ["--garbage", garbage]
    if teacher_weights:
        cmd += ["--teacher-weights", teacher_weights]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    line = [l for l in p.stdout.splitlines() if l.startswith("AtTenTris")]
    if not line:
        print("no result for seed %d takeover %d: %s" % (seed, takeover, (p.stderr or p.stdout)[-600:]), file=sys.stderr, flush=True)
        return None
    died = "died 1/1" in line[0]
    pieces = int(line[0].split("pieces ")[1].split(" ")[0])
    app = float(line[0].split("APP ")[1].split(" ")[0])
    return died, pieces, app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deaths", nargs="+", required=True, help="name:games_file pairs (play.py --out of a depth-1 run)")
    ap.add_argument("--before", default="100,50,20,10")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--garbage", default="", help="the garbage the games were played with")
    ap.add_argument("--teacher-weights", default="", help="the teacher's weights (default normal.txt)")
    ap.add_argument("--sample", type=int, default=0, help="only this many random deaths per model (0 = all)")
    ap.add_argument("--no-alone", action="store_true", help="skip the teacher-alone-from-move-0 runs")
    ap.add_argument("--style", default="normal", help="the model's style input while it plays (normal / gate4 / danger / auto)")
    args = ap.parse_args()
    jobs = []
    rng = random.Random(0)
    for spec in args.deaths:
        name, f = spec.split(":", 1)                        # the path may be C:/...
        model = os.path.join(HERE, "data", "model_%s.pt" % name)
        path = f if os.path.isabs(f) else os.path.join(HERE, f)
        deaths = [r for r in (json.loads(l) for l in open(path) if l.startswith("{")) if r["died"]]
        if args.sample and len(deaths) > args.sample:
            deaths = rng.sample(deaths, args.sample)
        for r in deaths:
            if not args.no_alone:
                jobs.append((name, model, r["seed"], r["pieces"], 0))
            for k in (int(x) for x in args.before.split(",")):
                if r["pieces"] - k > 0:
                    jobs.append((name, model, r["seed"], r["pieces"], r["pieces"] - k))
    print("%-14s %5s %10s  %-28s  %s" % ("model", "seed", "died at", "teacher takes over at", "result"), flush=True)
    res = []
    with cf.ThreadPoolExecutor(args.jobs) as ex:
        futs = {ex.submit(run, j[1], j[2], j[4], args.garbage, args.teacher_weights, args.style): j for j in jobs}
        for f in cf.as_completed(futs):                     # each result as soon as it is known
            name, _, seed, dpieces, tk = futs[f]
            r = f.result()
            res.append((name, dpieces - tk if tk else None, r))
            when = "move 0 (teacher alone)" if tk == 0 else "move %d (%d before death)" % (tk, dpieces - tk)
            txt = "no result" if r is None else ("DIED at piece %d" % r[1] if r[0] else "survived to 500 (APP %.3f)" % r[2])
            print("%-14s %5d %10d  %-28s  %s" % (name, seed, dpieces, when, txt), flush=True)
    print("\nsummary: teacher survives / tries")
    for name in sorted({x[0] for x in res}):
        for k in sorted({x[1] for x in res if x[0] == name}, key=lambda v: -1 if v is None else v):
            rr = [x[2] for x in res if x[0] == name and x[1] == k and x[2] is not None]
            print("  %-14s %-26s %d/%d" % (name, "alone from move 0" if k is None else "takes over %d before death" % k,
                                           sum(not r[0] for r in rr), len(rr)))


if __name__ == "__main__":
    main()
