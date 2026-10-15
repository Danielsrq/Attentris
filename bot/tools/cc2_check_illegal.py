"""Which Cold Clear 2 moves does our game reject? (cc2_play.py --log DIR) Every logged turn's state
goes through `bot expand --mode standard --no-180` (all our legal moves); a Cold Clear 2 move whose
(hold, cells) is not among them is printed with the board, the piece, Cold Clear 2's location / spin
and the stack height, to tell a spawn / kick / rule difference from a driver bug.

    attentris/.venv/Scripts/python bot/tools/cc2_check_illegal.py DIR
"""
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BOT = os.path.join(HERE, "..", "target", "release", "bot.exe" if os.name == "nt" else "bot")


def show(rows, cells):
    cs = set(map(tuple, cells))
    top = min([y for y in range(40) if rows[y]] + [y for _, y in cs] + [39])
    out = []
    for y in range(max(0, top - 1), 40):
        out.append("%2d " % y + "".join("@" if (x, y) in cs else ("#" if rows[y] >> x & 1 else ".") for x in range(10)))
    return "\n".join(out)


def main():
    files = sorted(glob.glob(os.path.join(sys.argv[1], "s*.jsonl")))
    p = subprocess.Popen([BOT, "expand", "--mode", "standard", "--no-180"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    turns = bad = 0
    for f in files:
        for line in open(f):
            t = json.loads(line)
            st = t["state"]
            p.stdin.write(json.dumps({"rows": st["rows"], "piece": st["piece"], "queue": st["queue"], "hold": st["hold"], "b2b": st["b2b"]}) + "\n")
            p.stdin.flush()
            r = json.loads(p.stdout.readline())
            want = sorted(map(tuple, t["cells"]))
            legal = {(c["hold"], tuple(sorted(zip(c["cells"][0::2], c["cells"][1::2])))) for c in r["cands"]}
            turns += 1
            if (t["hold"], tuple(want)) not in legal:
                bad += 1
                height = 40 - min([y for y in range(40) if st["rows"][y]] + [40])
                loc = t["move"]["location"]
                print("%s n=%d: %s %s x=%d y=%d spin=%s hold=%d, stack height %d, fresh start %s, %d legal moves" % (
                    os.path.basename(f), st["n"], loc["type"], loc["orientation"], loc["x"], loc["y"], t["move"].get("spin"),
                    t["hold"], height, t["fresh"], len(r["cands"])))
                print(show(st["rows"], want))
    p.stdin.close()
    print("%d of %d Cold Clear 2 moves not legal in our game" % (bad, turns))


if __name__ == "__main__":
    main()
