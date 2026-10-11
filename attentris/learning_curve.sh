#!/bin/sh
# Learning curve: the v3 recipe (policy + Qs, all-roots labels) on 25 / 50 / 100% of the
# original training positions (no DAgger data, so the subsets are comparable), same val set.
# If validation accuracy is still rising steeply at 100%, more data will help.
DIR=$(cd "$(dirname "$0")" && pwd)
PY="$DIR/.venv/Scripts/python.exe"
OUT="$DIR/results_learning_curve.txt"
: > "$OUT"
for n in 22480 44960 89920; do
    "$PY" "$DIR/train.py" --data "$DIR/data/v3" --train train.npz --limit $n --epochs 16 --batch 256 --device cuda \
        --w-dense 0 --w-q1 0 --w-qs 1 --out "$DIR/data/curve_$n.pt" > "$DIR/data/curve_$n.out" 2>&1
    echo "$n positions: $(grep '^epoch 16:' "$DIR/data/curve_$n.out")" | tee -a "$OUT"
done
