#!/bin/sh
# Value heads restricted to the policy's proposals (the policy's top-k moves, then argmax Q):
# the standard guard against a Q function picking moves it was never trained on.
DIR=$(cd "$(dirname "$0")" && pwd)
PY="$DIR/../external/tetrisbattle/.venv/Scripts/python.exe"
OUT=${OUT:-$DIR/results_play_v2_propose.txt}
MODEL=${MODEL:-$DIR/data/model_v2.pt}
STRESS="10,3-4,cheese0.2,0.2"
: > "$OUT"
for g in "" "$STRESS"; do
    gflag=""; [ -n "$g" ] && gflag="--garbage $g"
    echo "== garbage: ${g:-none}" | tee -a "$OUT"
    for k in 2 3 5; do
        "$PY" "$DIR/play.py" --model "$MODEL" --games 16 --depth 1 --policy qs --propose $k $gflag | grep "^AtTenTris" | tee -a "$OUT"
    done
    "$PY" "$DIR/play.py" --model "$MODEL" --games 16 --depth 1 --policy q1 --propose 3 $gflag | grep "^AtTenTris" | tee -a "$OUT"
    "$PY" "$DIR/play.py" --model "$MODEL" --games 16 --depth 3 --beam 8 --policy qs --propose 3 $gflag | grep "^AtTenTris" | tee -a "$OUT"
done
echo "all done" | tee -a "$OUT"
