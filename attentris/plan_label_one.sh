#!/bin/sh
# One input file for plan_label.sh: plan_label_one.sh OUT_DIR BEAM ROUTE FILE.jsonl -> OUT_DIR/FILE.jsonl
# (written atomically; skipped when it exists). ROUTE = route: DAgger 5 records go through
# plan_route.py (hot "normal" records searched by CEM-survival). Needs BOT, PY, TW, DIR.
out="$1/$(basename "$4")"
[ -f "$out" ] && exit 0
if [ "$3" = route ]; then
    "$PY" "$DIR/plan_route.py" < "$4"
else
    cat "$4"
fi | "$BOT" label --plan --mode normal --teacher-weights "$TW" --beam "$2" --threads 1 --out "$out.tmp" 2>/dev/null &&
    mv "$out.tmp" "$out"
