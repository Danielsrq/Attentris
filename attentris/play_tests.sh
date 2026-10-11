#!/bin/sh
# AtTenTris v1 playing real games under normal rules: no garbage, and the stress garbage
# (every 10 pieces 3-4 rows, 80% clean aligned well / 20% cheese). 16 games x 500 pieces,
# seeds 5001-5016 (not in the training data). Baselines on the same seeds: the teacher's
# V greedy (depth 1, no chance ply) and the full search.
DIR=$(cd "$(dirname "$0")" && pwd)
PY="$DIR/../external/tetrisbattle/.venv/Scripts/python.exe"
BOT="$DIR/../bot/target/release/bot.exe"
OUT=${OUT:-$DIR/results_play.txt}
MODEL=${MODEL:-$DIR/data/model.pt}
STRESS="10,3-4,cheese0.2,0.2"
: > "$OUT"
for g in "" "$STRESS"; do
    gflag=""; [ -n "$g" ] && gflag="--garbage $g"
    echo "== garbage: ${g:-none}" | tee -a "$OUT"
    "$PY" "$DIR/play.py" --model "$MODEL" --games 16 --depth 1 --style normal $gflag --out "$OUT.games" | tee -a "$OUT"
    "$PY" "$DIR/play.py" --model "$MODEL" --games 16 --depth 1 --style gate4 $gflag --out "$OUT.games" | tee -a "$OUT"
    "$PY" "$DIR/play.py" --model "$MODEL" --games 16 --depth 3 --beam 8 --style normal $gflag --out "$OUT.games" | tee -a "$OUT"
    [ -n "$SKIP_BASE" ] || for w in normal; do
        "$BOT" play --mode normal $gflag --weights "$DIR/../bot/weights/$w.txt" --games 16 --pieces 500 --seed 5001 \
            --depth 1 --no-chance --label "V_greedy_$w" 2>/dev/null | grep -E "^V_greedy" | tee -a "$OUT"
        "$BOT" play --mode normal $gflag --weights "$DIR/../bot/weights/$w.txt" --games 16 --pieces 500 --seed 5001 \
            --label "search_$w" 2>/dev/null | grep -E "^search_" | tee -a "$OUT"
    done
done
echo "all done" | tee -a "$OUT"
