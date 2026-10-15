#!/bin/sh
# AtTenTris v2 (value heads): move choice by the policy, by argmax of each value head, and a
# depth-3 beam scored like the teacher's (attack + Qs at the leaf). Same games as
# play_tests.sh: 16 x 500 pieces, seeds 5001-5016, normal rules, no garbage and stress.
#   MODEL=attentris/data/model_v2.pt OUT=attentris/results_play_v2.txt sh attentris/play_tests_v2.sh
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR"; . "$ATT/plat.sh"
PY="$CPUPY"
OUT=${OUT:-$DIR/results_play_v2.txt}
MODEL=${MODEL:-$DIR/data/model_v2.pt}
STRESS="10,3-4,cheese0.2,0.2"
: > "$OUT"
for g in "" "$STRESS"; do
    gflag=""; [ -n "$g" ] && gflag="--garbage $g"
    echo "== garbage: ${g:-none}" | tee -a "$OUT"
    for pol in logit qs q1; do
        "$PY" "$DIR/play.py" --model "$MODEL" --games 16 --depth 1 --policy $pol --style normal $gflag --out "$OUT.games" | grep "^AtTenTris" | tee -a "$OUT"
    done
    "$PY" "$DIR/play.py" --model "$MODEL" --games 16 --depth 1 --policy qs --style gate4 $gflag --out "$OUT.games" | grep "^AtTenTris" | tee -a "$OUT"
    "$PY" "$DIR/play.py" --model "$MODEL" --games 16 --depth 3 --beam 8 --policy qs --style normal $gflag --out "$OUT.games" | grep "^AtTenTris" | tee -a "$OUT"
done
echo "all done" | tee -a "$OUT"
