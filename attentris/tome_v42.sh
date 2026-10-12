#!/bin/sh
# ToMe v4.2 = ToMe v4.1's recipes with the direction-blind A/B split (--tome-split hash; the token split
# made ~78% of side-by-side neighbour pairs eligible to merge vs ~51% of stacked: tome_orient.py,
# test_tome_split.py). PLAN_2026-10-13.md (not started until the user says go). GPU chain, one job at a time:
#   1. tome_v42_l4: 4 layers, merge 0,1,1,1 + per-position budget (screen C's config), 8 epochs
#   2. tome_v42_l6: 6 layers, merge 0,1,1,1,1,1 + budget (= v4.1), 8 epochs
#   3. DAgger 3 fine-tunes of both (as v4.1-dagger3: from the base, 70/30 with danger.npz, 3 epochs,
#      lr 3e-4, 3 style rows)
# Tests run on the CPU in tome_v42_eval.sh (start it alongside; it waits for each model). Resumable;
# one copy at a time.   sh tome_v42.sh > tome_v42.out 2>&1
DIR=$(cd "$(dirname "$0")" && pwd)
PYG="$DIR/.venv/Scripts/python.exe"
V51="$DIR/data/v51"
OUT="$DIR/results_tome_v42.txt"
mkdir "$DIR/.tome_v42.lock" 2>/dev/null || { echo "tome_v42.sh is already running (remove $DIR/.tome_v42.lock if not)"; exit 1; }
trap 'rmdir "$DIR/.tome_v42.lock"' EXIT
"$PYG" "$DIR/test_tome_split.py" | tail -1 | grep -q "ALL OK" || { echo "test_tome_split.py failed: not training" | tee -a "$OUT"; exit 1; }
COMMON="--arch tome --tome-pretok --tome-flat --tome-streams 2 --tome-split hash --tome-budget 20,36,0.25,0.10 --bucket 50"
L4="$COMMON --layers 4 --merge 0,1,1,1 --micro 4"
L6="$COMMON --layers 6 --merge 0,1,1,1,1,1 --micro 8"
train() {   # name epochs log args...
    name=$1; ep=$2; log=$3; shift 3
    M="$DIR/data/model_$name.pt"
    if ! grep -q "^epoch $ep:" "$log" 2>/dev/null; then
        R=""; [ -f "$M" ] && R="--resume"
        echo "$(date '+%H:%M') $name: training" | tee -a "$OUT"
        "$PYG" "$DIR/train.py" $ARCH --data "$V51" --batch 512 --device cuda --w-dense 0 --w-q1 0 --w-qs 1 --seed 0 --out "$M" $R "$@" >> "$log" 2>&1
    fi
    grep -q "^epoch $ep:" "$log" || { echo "$name: training did not finish (see $log)" | tee -a "$OUT"; exit 1; }
    echo "== $name ($(date '+%H:%M'))" | tee -a "$OUT"
    grep "^epoch $ep:" "$log" | sed 's/argmax Qs.*Qs [0-9.]* |/|/' | tee -a "$OUT"
}
F=0.1851      # 70/30 with the 86,195 danger positions over train + dagger1 + train1m (as tome_v41.sh)
ARCH=$L4; train tome_v42_l4 8 "$DIR/train_tome_v42_l4.out" --train train.npz,dagger1.npz,train1m.npz --epochs 8 --lr 1.5e-3
ARCH=$L6; train tome_v42_l6 8 "$DIR/train_tome_v42_l6.out" --train train.npz,dagger1.npz,train1m.npz --epochs 8 --lr 1.5e-3
for n in l4 l6; do
    [ $n = l4 ] && ARCH=$L4 || ARCH=$L6
    train tome_v42_${n}_dagger3_ft 3 "$DIR/train_tome_v42_${n}_dagger3_ft.out" \
        --train train.npz@$F,dagger1.npz@$F,train1m.npz@$F,danger.npz --epochs 3 --lr 3e-4 --n-teachers 3 \
        --init "$DIR/data/model_tome_v42_$n.pt" --val2 val_danger.npz
done
echo "tome v42 training done"
