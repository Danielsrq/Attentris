#!/bin/sh
# Learning curve near the top: the v4 recipe (8 epochs, batch 512, lr 1.5e-3, policy + Qs)
# on random 25% / 50% subsets of the 1M positions, against v4 on all 1.09M. If top-1 is
# still rising ~2.5-3 points per doubling at this end, more data keeps paying.
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR"; . "$ATT/plat.sh"
PY="$DIR/.venv/$VPY"
OUT="$DIR/results_learning_curve2.txt"
: > "$OUT"
for n in 272000 544000; do
    "$PY" "$DIR/train.py" --data "$DIR/data/v3" --train train1m.npz --limit $n --epochs 8 --batch 512 --lr 1.5e-3 \
        --device cuda --w-dense 0 --w-q1 0 --w-qs 1 --out "$DIR/data/curve2_$n.pt" > "$DIR/data/curve2_$n.out" 2>&1
    echo "$n positions: $(grep '^epoch 8:' "$DIR/data/curve2_$n.out")" | tee -a "$OUT"
done
