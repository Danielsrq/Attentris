#!/bin/sh
# AtTenTris v5.2: two cross-attention passes instead of one, on the v5.1 tokens and recipe.
#   v5.2-loop:   the same cross block applied twice (weight-tied; same parameters as v5.1)
#   v5.2-untied: two independent cross blocks, each with its own FLAT-2D bias (+50k parameters)
# Own checkpoints (data/model_v52_loop.pt, data/model_v52_untied.pt); resumable (rerun).
# Then 64-game play tests (depth 1, depth 2 beam 4, depth 3 beam 8; normal, no garbage) for both.
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR"; . "$ATT/plat.sh"
PYG="$DIR/.venv/$VPY"
PYC="$CPUPY"
OUT="$DIR/results_v52.txt"
for v in loop untied; do
    M="$DIR/data/model_v52_$v.pt"
    if [ "$v" = loop ]; then FLAG="--cross-loops 2"; else FLAG="--cross-layers 2"; fi
    R=""; [ -f "$M" ] && R="--resume"
    "$PYG" "$DIR/train.py" --data "$DIR/data/v51" --train train.npz,dagger1.npz,train1m.npz --epochs 8 --batch 512 --lr 1.5e-3 \
        --device cuda --w-dense 0 --w-q1 0 --w-qs 1 --seed 0 $FLAG --out "$M" $R >> "$DIR/train_v52_$v.out" 2>&1
    echo "== v5.2-$v" | tee -a "$OUT"
    grep "^epoch 8:" "$DIR/train_v52_$v.out" | tee -a "$OUT"
done
for v in loop untied; do
    for db in "1 8" "2 4" "3 8"; do
        set -- $db
        echo "v5.2-$v: $("$PYC" "$DIR/play.py" --model "$DIR/data/model_v52_$v.pt" --games 64 --depth $1 --beam $2 | grep "^AtTenTris")" | tee -a "$OUT"
    done
done
echo "v52 done" | tee -a "$OUT"
