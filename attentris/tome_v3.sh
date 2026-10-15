#!/bin/sh
# AtTenTris-ToMe v3: rule pre-tokens (tome_tok.py, precomputed by pretok.py): one sky token (rows
# above the tallest column, + the 10 column heights), one token per solid filled block (>= 2x2),
# single cells for everything else below the sky (~36 cell tokens + 7 pieces); 17% merged after
# attention in layers 2-4 (~22 at the end); FLAT-2D self-attention bias; batches grouped by token
# count. Same data and recipe as v5.1 / ToMe base / v2. Own checkpoint data/model_tome_v3.pt;
# resumable (rerun). Then tome_post.sh.
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR"; . "$ATT/plat.sh"
PYG="$DIR/.venv/$VPY"
M="$DIR/data/model_tome_v3.pt"
"$PYG" "$DIR/pretok.py" --data "$DIR/data/v51" | grep -v pinned
R=""; [ -f "$M" ] && R="--resume"
"$PYG" "$DIR/train.py" --arch tome --tome-pretok --tome-flat --merge 0,0.17,0.17,0.17 --bucket 50 --data "$DIR/data/v51" \
    --train train.npz,dagger1.npz,train1m.npz --epochs 8 --batch 512 --lr 1.5e-3 --device cuda \
    --w-dense 0 --w-q1 0 --w-qs 1 --seed 0 --out "$M" $R >> "$DIR/train_tome_v3.out" 2>&1
grep "^epoch" "$DIR/train_tome_v3.out" | tee "$DIR/results_tome_v3_train.txt"
sh "$DIR/tome_post.sh" model_tome_v3 > "$DIR/tome_v3_post.out" 2>&1
echo "tome v3 done"
