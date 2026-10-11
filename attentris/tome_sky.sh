#!/bin/sh
# AtTenTris-ToMe v2, pre-tokenized sky: the empty cells above each column's top start as one
# token per column, the stack cells as single cells (~77 tokens instead of 200); 25% merged after
# attention in layers 2-4 (~33 blobs on average, as the base run's 34), batches grouped by token
# count (--bucket). Same data and recipe as base ToMe / v5.1. Own checkpoint data/model_tome_sky.pt.
# Resumable: rerun to continue (mid-epoch checkpoints every 300 batches). Then tome_post.sh.
DIR=$(cd "$(dirname "$0")" && pwd)
PYG="$DIR/.venv/Scripts/python.exe"
M="$DIR/data/model_tome_sky.pt"
R=""; [ -f "$M" ] && R="--resume"
"$PYG" "$DIR/train.py" --arch tome --tome-sky --merge 0,0.25,0.25,0.25 --bucket 50 --data "$DIR/data/v51" \
    --train train.npz,dagger1.npz,train1m.npz --epochs 8 --batch 512 --lr 1.5e-3 --device cuda \
    --w-dense 0 --w-q1 0 --w-qs 1 --seed 0 --out "$M" $R >> "$DIR/train_tome_sky.out" 2>&1
grep "^epoch" "$DIR/train_tome_sky.out" | tee "$DIR/results_tome_sky_train.txt"
sh "$DIR/tome_post.sh" model_tome_sky > "$DIR/tome_sky_post.out" 2>&1
echo "tome sky done"
