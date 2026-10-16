"""Filter for plan_label.sh on DAgger 5 records (stdin -> stdout, one line each, order kept): a
"normal"-tagged record on a hot board is a copy of CEM-survival's label (dagger5_route.py), so it
gets "w": "danger" and bot label --plan searches it with CEM-survival's weights."""
import json
import sys

from dagger5_route import hot

for line in sys.stdin:
    if line.strip():
        r = json.loads(line)
        if r.get("teacher") == "normal" and hot(r["rows"]):
            r["w"] = "danger"
            line = json.dumps(r, separators=(",", ":")) + "\n"
    sys.stdout.write(line)
