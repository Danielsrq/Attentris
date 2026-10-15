#!/bin/sh
# ToMe v4 at a heavier merge schedule than it was trained with (0,0.17,0.25,0.25 instead of
# 0,0.17,0.17,0.17), inference only: 64 games at depth 1 (states dumped for the viewer) and at depth
# 2 beam 4 (no depth 3), same seeds as the trained-schedule tests (5001+).
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR"; . "$ATT/plat.sh"
PYC="$CPUPY"
M=0,0.17,0.25,0.25
OUT="$DIR/results_tome_v4_m25.txt"
D="$DIR/data/tome_m25"
mkdir -p "$D"
echo "== ToMe v4, merge $M (trained with 0,0.17,0.17,0.17), $(date)" > "$OUT"
"$PYC" "$DIR/play.py" --model "$DIR/data/model_tome_v4.pt" --games 64 --depth 1 --merge $M \
    --out "$(wpath "$D/d1.games")" --dump "$(wpath "$D/d1_states.jsonl")" | grep "^AtTenTris" | tee -a "$OUT"
"$PYC" "$DIR/play.py" --model "$DIR/data/model_tome_v4.pt" --games 64 --depth 2 --beam 4 --merge $M | grep "^AtTenTris" | tee -a "$OUT"
echo "tome m25 done" | tee -a "$OUT"
