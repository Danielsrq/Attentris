#!/bin/sh
# v5.1 at d=96 through the v5.1 fine-tune lineage (same data, seeds, schedules as fit_ablation.sh /
# setup_ablation.sh): base (capacity_d96.sh) -> DAgger 2 ft (70/30, 3 ep, lr 3e-4) -> DAgger 3 ft (+ danger
# style, 3 ep). Tests of the DAgger 3 stage vs v5.1 dagger3-ft on the same seeds: no garbage 256 (depth 1),
# depth 2 beam 4 (64), 80/20 and 50/50 (64 each) in normal and danger style. Resumable.
#   sh d96_lineage.sh > d96_lineage.out 2>&1
DIR=$(cd "$(dirname "$0")" && pwd)
PYG="$DIR/.venv/Scripts/python.exe"
PYC="$DIR/../external/tetrisbattle/.venv/Scripts/python.exe"
V51="$DIR/data/v51"
D="$DIR/data/dagger3_eval"
WD=$(cygpath -m "$D")
OUT="$DIR/results_capacity.txt"
stage() {   # name epochs log args...
    name=$1; ep=$2; log=$3; shift 3
    if ! grep -q "^epoch $ep:" "$log" 2>/dev/null; then
        R=""; [ -f "$DIR/data/model_$name.pt" ] && R="--resume"
        "$PYG" "$DIR/train.py" --data "$V51" --batch 512 --device cuda --w-dense 0 --w-q1 0 --w-qs 1 --seed 0 --d 96 \
            --out "$DIR/data/model_$name.pt" $R "$@" >> "$log" 2>&1
    fi
    grep -q "^epoch $ep:" "$log" || { echo "$name: training did not finish (see $log)" | tee -a "$OUT"; exit 1; }
    echo "== $name" | tee -a "$OUT"
    grep -E "^epoch $ep:" "$log" | tail -2 | sed 's/argmax Qs.*Qs [0-9.]* |/|/' | tee -a "$OUT"
}
[ -f "$DIR/data/model_v51_d96.pt" ] || { echo "no d=96 base model (capacity_d96.sh)"; exit 1; }
stage v51_d96_dagger_ft 3 "$DIR/train_v51_d96_dagger_ft.out" --train train.npz@0.19,dagger1.npz@0.19,train1m.npz@0.19,dagger2.npz \
    --epochs 3 --lr 3e-4 --init "$DIR/data/model_v51_d96.pt" --val2 val_dagger2.npz
F=0.1712
stage v51_d96_dagger3_ft 3 "$DIR/train_v51_d96_dagger3_ft.out" --train train.npz@$F,dagger1.npz@$F,train1m.npz@$F,dagger2.npz@$F,danger.npz \
    --epochs 3 --lr 3e-4 --n-teachers 3 --init "$DIR/data/model_v51_d96_dagger_ft.pt" --val2 val_danger.npz
play() {   # model style tag games flags...
    f="$D/$1_$2_$3.games"; m=$1; s=$2; t=$3; g=$4; shift 4
    [ -f "$f" ] && grep -q "^AtTenTris" "$f" && return
    rm -f "$f"
    echo "$m $s $t: $("$PYC" "$DIR/play.py" --model "$DIR/data/model_$m.pt" --style $s --games $g "$@" --out "$(cygpath -m "$f")" \
        | grep "^AtTenTris" | sed 's/.*: APP/APP/' | cut -c1-200)" | tee -a "$OUT"
}
n=v51_d96_dagger3_ft
play $n normal none 256 --depth 1
play $n normal d2 64 --depth 2 --beam 4
for s in normal danger; do
    play $n $s g20 64 --depth 1 --seed 33001 --garbage 10,3-4,cheese0.2,0.2
    play $n $s g50 64 --depth 1 --seed 33001 --garbage 10,3-4,cheese0.5,0.2
done
args=""
for m in v51_dagger3_ft v51_d96_dagger3_ft; do
    for x in normal_none normal_d2 normal_g20 normal_g50 danger_g20 danger_g50; do args="$args ${m}_$x=$WD/${m}_$x.games"; done
done
echo "== APP / VS: v5.1 dagger3-ft (d=64) vs d=96 dagger3-ft (same seeds)" | tee -a "$OUT"
"$PYC" "$DIR/vs_score.py" $args | tee -a "$OUT"
echo "d96 lineage done"
