#!/bin/sh
# Intuition + calculation: AtTenTris shortlists its top-k first moves, the teacher's full
# search (handcrafted V, beam 64, depth 6 + chance) picks among them. k = 1 is the policy
# alone; the full search over all moves (search_normal in results_play.txt) is k = all.
# 16 games x 500 pieces, seeds 5001-5016, normal rules, no garbage and stress.
DIR=$(cd "$(dirname "$0")" && pwd)
PY="$DIR/../external/tetrisbattle/.venv/Scripts/python.exe"
OUT=${OUT:-$DIR/results_play_shortlist.txt}
MODEL=${MODEL:-$DIR/data/model_v3.pt}
STRESS="10,3-4,cheese0.2,0.2"
: > "$OUT"
for g in "" "$STRESS"; do
    gflag=""; [ -n "$g" ] && gflag="--garbage $g"
    echo "== garbage: ${g:-none}" | tee -a "$OUT"
    for k in 1 2 3 5 10; do
        "$PY" "$DIR/play.py" --model "$MODEL" --games 16 --shortlist $k $gflag | grep "^AtTenTris" | tee -a "$OUT"
    done
done
echo "all done" | tee -a "$OUT"
