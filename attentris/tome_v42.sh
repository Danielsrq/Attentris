#!/bin/sh
# ToMe v4.2 (PLAN_2026-10-13.md): 4-layer ToMe with the merging fixed, both with pure merges (blobs merge
# only with blobs of the same kind: all empty / all filled):
#   tome_v42_l4h  bipartite matching with the direction-blind hash split (--tome-split hash)
#   tome_v42_l4g  greedy matching, no A/B sets (--tome-split greedy)
# Each: screen C's config (4 layers, merge 0,1,1,1, per-position budget 20 + max(0, (n0 - 36) / 4) blobs
# per stream, first merge 10%), 2 streams, pre-tokens, FLAT-2D, train + dagger1 + train1m, lr 1.5e-3,
# seed 0, 8 epochs; then one DAgger 2 fine-tune (70/30 with dagger2.npz, 3 epochs, lr 3e-4, val2 =
# val_dagger2.npz, as v52_dagger.sh; no DAgger 3 for now). RUNS="l4h l4g l6g" adds the 6-layer greedy
# model (v4.1's layout). GPU only, one job at a time; tests in tome_v42_eval.sh (CPU, start alongside).
# Resumable; one copy at a time.   sh tome_v42.sh > tome_v42.out 2>&1
DIR=$(cd "$(dirname "$0")" && pwd)
PYG="$DIR/.venv/Scripts/python.exe"
V51="$DIR/data/v51"
OUT="$DIR/results_tome_v42.txt"
mkdir "$DIR/.tome_v42.lock" 2>/dev/null || { echo "tome_v42.sh is already running (remove $DIR/.tome_v42.lock if not)"; exit 1; }
trap 'rmdir "$DIR/.tome_v42.lock"' EXIT
"$PYG" "$DIR/test_tome_split.py" | tail -1 | grep -q "ALL OK" || { echo "test_tome_split.py failed: not training" | tee -a "$OUT"; exit 1; }
RUNS=${RUNS:-"l4h l4g"}
COMMON="--arch tome --tome-pretok --tome-flat --tome-streams 2 --tome-pure --tome-budget 20,36,0.25,0.10 --bucket 50"
arch_of() {
    case $1 in
        l4h) echo "$COMMON --tome-split hash --layers 4 --merge 0,1,1,1 --micro 4";;
        l4g) echo "$COMMON --tome-split greedy --layers 4 --merge 0,1,1,1 --micro 4";;
        l6g) echo "$COMMON --tome-split greedy --layers 6 --merge 0,1,1,1,1,1 --micro 8 --tome-budget 15,36,0.25,0.10";;   # ~15 blobs per stream
    esac
}
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
for n in $RUNS; do
    ARCH=$(arch_of $n)
    train tome_v42_$n 8 "$DIR/train_tome_v42_$n.out" --train train.npz,dagger1.npz,train1m.npz --epochs 8 --lr 1.5e-3
done
for n in $RUNS; do
    ARCH=$(arch_of $n)
    case $n in l6*) FTM=16;; *) FTM=8;; esac     # DAgger 2 boards have ~25% more tokens: smaller micro-batches
    train tome_v42_${n}_dagger_ft 3 "$DIR/train_tome_v42_${n}_dagger_ft.out" \
        --train train.npz@0.19,dagger1.npz@0.19,train1m.npz@0.19,dagger2.npz --epochs 3 --lr 3e-4 --micro $FTM \
        --init "$DIR/data/model_tome_v42_$n.pt" --val2 val_dagger2.npz
done
echo "tome v42 training done"
