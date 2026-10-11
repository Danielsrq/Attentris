#!/bin/sh
# AtTenTris v5 (new tokens: covered piece slots + hangs, setups, post-clear open fits, 12
# overlays): re-tokenize the same 1.09M positions into data/v5, train FROM SCRATCH with the v4
# recipe (so the only change vs v4 is the tokenizer), then play tests: policy depth 1, depth 2
# beam 4 (cms-RL's compute), depth 3 beam 8, top-10 shortlist + teacher; no garbage and stress;
# and single-game latency (--jobs 1) for depth 1 / 2 / 3 against v4b at depth 2.
DIR=$(cd "$(dirname "$0")" && pwd)
PYG="$DIR/.venv/Scripts/python.exe"
PYC="$DIR/../external/tetrisbattle/.venv/Scripts/python.exe"
OUT="$DIR/results_v5.txt"
V5="$DIR/data/v5"
M="$DIR/data/model_v5.pt"
mkdir -p "$V5"
: > "$OUT"
[ -f "$V5/val.npz" ] || "$PYC" "$DIR/pack.py" --raw "$DIR/data/raw_v3" --out "$V5" | tail -2 | tee -a "$OUT"
[ -f "$V5/dagger1.npz" ] || "$PYC" "$DIR/pack.py" --only "$DIR/data/raw_dagger1_v3" --name dagger1 --jobs 16 --out "$V5" | tail -1 | tee -a "$OUT"
[ -f "$V5/train1m.npz" ] || "$PYC" "$DIR/pack.py" --only "$DIR/data/raw_1m" --name train1m --jobs 16 --out "$V5" | tail -1 | tee -a "$OUT"
echo "packed" | tee -a "$OUT"
RES=""; [ -f "$M" ] && RES="--resume"
"$PYG" "$DIR/train.py" --data "$V5" --train train.npz,dagger1.npz,train1m.npz --epochs 8 --batch 512 --lr 1.5e-3 \
    --device cuda --w-dense 0 --w-q1 0 --w-qs 1 --out "$M" $RES > "$DIR/train_v5.out" 2>&1
grep "^epoch" "$DIR/train_v5.out" | tee -a "$OUT"
STRESS="10,3-4,cheese0.2,0.2"
for g in "" "$STRESS"; do
    gflag=""; [ -n "$g" ] && gflag="--garbage $g"
    echo "== garbage: ${g:-none}" | tee -a "$OUT"
    "$PYC" "$DIR/play.py" --model "$M" --games 64 --depth 1 $gflag | grep "^AtTenTris" | tee -a "$OUT"
    "$PYC" "$DIR/play.py" --model "$M" --games 64 --depth 2 --beam 4 $gflag | grep "^AtTenTris" | tee -a "$OUT"
    "$PYC" "$DIR/play.py" --model "$M" --games 64 --depth 3 --beam 8 $gflag | grep "^AtTenTris" | tee -a "$OUT"
    "$PYC" "$DIR/play.py" --model "$M" --games 64 --shortlist 10 $gflag | grep "^AtTenTris" | tee -a "$OUT"
done
echo "== latency, one game at a time (--jobs 1, 8 games)" | tee -a "$OUT"
for m in "$M" "$DIR/data/model_v4b.pt"; do
    for db in "1 8" "2 4" "3 8"; do
        set -- $db
        echo "$(basename "$m"): $("$PYC" "$DIR/play.py" --model "$m" --games 8 --jobs 1 --depth $1 --beam $2 | grep "^AtTenTris")" | tee -a "$OUT"
    done
done
echo "v5 done" | tee -a "$OUT"
