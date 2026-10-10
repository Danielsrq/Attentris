#!/bin/sh
# v4b: warm start from v4 (8 more epochs, fresh one-cycle schedule at a lower peak lr) on the
# same 1.09M positions, then 64-game play tests of the policy alone and the depth-3 beam.
DIR=$(cd "$(dirname "$0")" && pwd)
PYG="$DIR/.venv/Scripts/python.exe"
PYC="$DIR/../external/tetrisbattle/.venv/Scripts/python.exe"
OUT="$DIR/results_v4b.txt"
: > "$OUT"
"$PYG" "$DIR/train.py" --data "$DIR/data/v3" --train train.npz,dagger1.npz,train1m.npz --init "$DIR/data/model_v4.pt" \
    --epochs 8 --batch 512 --lr 5e-4 --device cuda --w-dense 0 --w-q1 0 --w-qs 1 --out "$DIR/data/model_v4b.pt" > "$DIR/train_v4b.out" 2>&1
grep "^epoch" "$DIR/train_v4b.out" | tee -a "$OUT"
STRESS="10,3-4,cheese0.2,0.2"
for g in "" "$STRESS"; do
    gflag=""; [ -n "$g" ] && gflag="--garbage $g"
    echo "== garbage: ${g:-none}" | tee -a "$OUT"
    "$PYC" "$DIR/play.py" --model "$DIR/data/model_v4b.pt" --games 64 --depth 1 $gflag --out "$OUT.games" | grep "^AtTenTris" | tee -a "$OUT"
    "$PYC" "$DIR/play.py" --model "$DIR/data/model_v4b.pt" --games 64 --depth 3 --beam 8 $gflag --out "$OUT.games" | grep "^AtTenTris" | tee -a "$OUT"
done
echo "v4b done" | tee -a "$OUT"
