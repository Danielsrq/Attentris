#!/bin/sh
# AtTenTris-ToMe, base version: 200 cell tokens + 7 piece tokens, no rule tokenizer and no sky
# shortcut; adjacency-constrained token merging after attention in layers 2-4 (45% of the cell
# tokens each: 200 -> ~110 -> ~61 -> ~34). Same data and recipe as v4 / v5.1 (it reads only the
# cells, piece tokens and candidates from the v5.1 packing). Own checkpoint: data/model_tome.pt
# (train.py refuses to overwrite an AtTenTris checkpoint). Resumable: rerun to continue.
DIR=$(cd "$(dirname "$0")" && pwd)
PYG="$DIR/.venv/Scripts/python.exe"
M="$DIR/data/model_tome.pt"
R=""; [ -f "$M" ] && R="--resume"
"$PYG" "$DIR/train.py" --arch tome --merge 0,0.45,0.45,0.45 --data "$DIR/data/v51" --train train.npz,dagger1.npz,train1m.npz \
    --epochs 8 --batch 512 --lr 1.5e-3 --device cuda --w-dense 0 --w-q1 0 --w-qs 1 --seed 0 --out "$M" $R >> "$DIR/train_tome.out" 2>&1
grep "^epoch" "$DIR/train_tome.out" | tee "$DIR/results_tome_train.txt"
echo "tome done"
