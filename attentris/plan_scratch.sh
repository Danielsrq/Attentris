#!/bin/sh
# Multi-move AtTenTris trained FROM SCRATCH with exactly the recipe of the d128 base (capacity_d128.sh:
# 8 epochs, lr 1.5e-3, batch 512, seed 0, train + dagger1 + train1m), plus the plan loss: compared with
# model_v51_d128.pt, the plan loss is the only difference (same data, epochs, schedule, seed).
#   VARIANT=autoreg N=2 sh attentris/plan_scratch.sh   -> data/model_v51_d128_autoreg2.pt
# Then the standard play block for both the base and the plan model (depth 1: none 256, d2 64,
# normal / danger x 80/20, 50/50 at 64) and vs_score. Resumable.
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR"; . "$ATT/plat.sh"
PYG="$DIR/.venv/$VPY"
PYC="$CPUPY"
V51="$DIR/data/v51"
VARIANT=${VARIANT:-autoreg}
N=${N:-2}
W_PLAN=${W_PLAN:-1.0}
BASE=v51_d128
NAME=v51_d128_${VARIANT}${N}
OUT="$DIR/results_plan.txt"
m() { wpath "$1"; }
echo "== $NAME from scratch (the $BASE recipe + $VARIANT plan loss, N=$N, w_plan $W_PLAN) $(date '+%Y-%m-%d %H:%M')" | tee -a "$OUT"
LOG="$DIR/train_$NAME.out"
if ! grep -q "^epoch 8: train" "$LOG" 2>/dev/null; then
    R=""; [ -f "$DIR/data/model_$NAME.pt" ] && R="--resume"
    "$PYG" "$DIR/train.py" --data "$(m "$V51")" --batch 512 --threads 8 --device cuda --w-dense 0 --w-q1 0 --w-qs 1 --seed 0 \
        --train train.npz,dagger1.npz,train1m.npz --epochs 8 --lr 1.5e-3 --d 128 --micro 2 \
        --plan-variant $VARIANT --plan-steps $N --w-plan $W_PLAN --out "$(m "$DIR/data/model_$NAME.pt")" $R >> "$LOG" 2>&1
fi
grep -q "^epoch 8: train" "$LOG" || { echo "training did not finish (see $LOG)" | tee -a "$OUT"; exit 1; }
grep "^epoch 8:" "$LOG" | sed 's/argmax Qs top1 [0-9.]* \/ [0-9.]*, Q1.*Qs [0-9.]* |/|/' | tee -a "$OUT"
grep "^epoch 8:" "$DIR/capacity_d128.out" "$DIR/train_v51_d128.out" 2>/dev/null | head -1 | sed 's/^/base: /' | tee -a "$OUT"
DE="$DIR/data/dagger3_eval"
play() {   # model style tag games flags...
    f="$DE/$1_$2_$3.games"; md=$1; s=$2; t=$3; g=$4; shift 4
    [ -f "$f" ] && grep -q "^AtTenTris" "$f" && return
    rm -f "$f"
    echo "$md $s $t: $("$PYC" "$DIR/play.py" --model "$DIR/data/model_$md.pt" --style $s --games $g "$@" --out "$(m "$f")" \
        | grep "^AtTenTris" | sed 's/.*: APP/APP/' | cut -c1-200)" | tee -a "$OUT"
}
# the base has two styles (normal, gate4) and no danger style: both models are tested in normal style
for md in $BASE $NAME; do
    play $md normal none 256 --depth 1
    play $md normal d2 64 --depth 2 --beam 4
    play $md normal g20 64 --depth 1 --seed 33001 --garbage 10,3-4,cheese0.2,0.2
    play $md normal g50 64 --depth 1 --seed 33001 --garbage 10,3-4,cheese0.5,0.2
done
args=""
for md in $BASE $NAME; do
    for x in normal_none normal_d2 normal_g20 normal_g50; do args="$args ${md}_$x=$(m "$DE")/${md}_$x.games"; done
done
echo "== APP / VS: $BASE vs $NAME (both from scratch, same recipe)" | tee -a "$OUT"
"$PYC" "$DIR/vs_score.py" $args | tee -a "$OUT"
echo "$NAME done"
