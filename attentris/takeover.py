"""Can the teacher (CEM linear V, depth 6 + beam 64 search) save the games a model lost? For each
death in a play.py run, the model replays its game (deterministic) up to k moves before the death,
then the teacher's full search takes over on the same piece sequence; also the teacher alone from
move 0 on that seed. Reports survival (500 pieces) and the final stack.

    python attentris/takeover.py --deaths v51:data/deaths_v51.txt tome_v4:data/deaths_tome_v4.txt
"""
import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PYC = os.path.join(HERE, "..", "external", "tetrisbattle", ".venv", "Scripts", "python.exe")


def run(model, seed, takeover):
    out = subprocess.run([PYC, os.path.join(HERE, "play.py"), "--model", model, "--games", "1", "--jobs", "1", "--seed", str(seed),
                          "--depth", "1", "--takeover", str(takeover), "--out", os.devnull],
                         capture_output=True, text=True, timeout=1800).stdout
    line = [l for l in out.splitlines() if l.startswith("AtTenTris")]
    if not line:
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
    args = ap.parse_args()
    jobs = []
    for spec in args.deaths:
        name, f = spec.split(":")
        model = os.path.join(HERE, "data", "model_%s.pt" % name)
        for r in (json.loads(l) for l in open(os.path.join(HERE, f)) if l.startswith("{")):
            if not r["died"]:
                continue
            jobs.append((name, model, r["seed"], r["pieces"], 0))
            for k in (int(x) for x in args.before.split(",")):
                if r["pieces"] - k > 0:
                    jobs.append((name, model, r["seed"], r["pieces"], r["pieces"] - k))
    print("%-8s %5s %10s  %-28s  %s" % ("model", "seed", "died at", "teacher takes over at", "result"), flush=True)
    with cf.ThreadPoolExecutor(args.jobs) as ex:
        futs = {ex.submit(run, j[1], j[2], j[4]): j for j in jobs}
        for f in cf.as_completed(futs):                     # each result as soon as it is known
            name, _, seed, dpieces, tk = futs[f]
            r = f.result()
            when = "move 0 (teacher alone)" if tk == 0 else "move %d (%d before death)" % (tk, dpieces - tk)
            txt = "no result" if r is None else ("DIED at piece %d" % r[1] if r[0] else "survived to 500 (APP %.3f)" % r[2])
            print("%-8s %5d %10d  %-28s  %s" % (name, seed, dpieces, when, txt), flush=True)


if __name__ == "__main__":
    main()
