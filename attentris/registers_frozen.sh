#!/bin/sh
# Register tokens, frozen backbone: v51_d128_dagger5b_ft + R=4 register tokens; only the registers train
# (reg, reg_k, reg_q, reg_c: ~530 numbers), 3 epochs on the dagger5b mix, new-parameter lr 1e-3.
# Start = dagger5b-ft exactly (register keys at -8; test_registers.py). Then the standard play block,
# compared with dagger5b-ft (same seeds). Resumable.
#   N=v51_d128_reg4frzb2_ft BIAS=-2 sh attentris/registers_frozen.sh > attentris/registers_frozen_b2.out 2>&1
# BIAS = the register keys' attention bias at the start. -8 (first run, v51_d128_reg4frz_ft): the model starts
# exactly as dagger5b-ft but the registers get ~0.05% attention and stayed shut (bias -8 -> -7.7 in 3 epochs,
# validation identical); -2: a 5% top-1 change at the start, and the bias is free to open or close.
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR"; . "$ATT/plat.sh"
PYG="$DIR/.venv/$VPY"
PYC="$CPUPY"
m() { wpath "$1"; }
V51="$DIR/data/v51"
OUT="$DIR/results_registers.txt"
N=${N:-v51_d128_reg4frz_ft}
BIAS=${BIAS:--8}
F=$("$PYC" -c "
import numpy as np
d = '$(m "$V51")/'
new = sum(len(np.load(d + f)['chosen']) for f in ('danger.npz', 'dagger4.npz', 'dagger5all.npz'))
rest = sum(len(np.load(d + f)['chosen']) for f in ('train.npz', 'dagger1.npz', 'train1m.npz', 'dagger2.npz'))
print('%.4f' % (7 / 3 * new / rest))")
LOG="$DIR/train_$N.out"
if ! grep -q "^epoch 3: val2" "$LOG" 2>/dev/null; then
    R=""; [ -f "$DIR/data/model_$N.pt" ] && R="--resume"
    "$PYG" "$DIR/train.py" --data "$V51" --batch 512 --device cuda --w-dense 0 --w-q1 0 --w-qs 1 --seed 0 --d 128 --micro 2 --val2 val_dagger5all.npz \
        --train train.npz@$F,dagger1.npz@$F,train1m.npz@$F,dagger2.npz@$F,danger.npz,dagger4.npz,dagger5all.npz --epochs 3 --lr 3e-4 --n-teachers 3 \
        --registers 4 --reg-bias-init $BIAS --freeze-base --new-lr 1e-3 \
        --init "$DIR/data/model_v51_d128_dagger5b_ft.pt" --out "$DIR/data/model_$N.pt" $R >> "$LOG" 2>&1
fi
grep -q "^epoch 3: val2" "$LOG" || { echo "training did not finish (see $LOG)" | tee -a "$OUT"; exit 1; }
echo "== $N vs v51_d128_dagger5b_ft (validation, epoch 3)" | tee -a "$OUT"
grep "^epoch [0-9]:" "$LOG" | sed 's/argmax Qs top1 [0-9.]* \/ [0-9.]*, Q1.*Qs [0-9.]* |/|/' | tee -a "$OUT"
grep "^epoch 3:" "$DIR/train_v51_d128_dagger5b_ft.out" | sed 's/^/dagger5b-ft /; s/argmax Qs top1 [0-9.]* \/ [0-9.]*, Q1.*Qs [0-9.]* |/|/' | tee -a "$OUT"
DE="$DIR/data/dagger3_eval"
play() {   # model style tag games flags...
    f="$DE/$1_$2_$3.games"; md=$1; s=$2; t=$3; g=$4; shift 4
    [ -f "$f" ] && grep -q "^AtTenTris" "$f" && return
    rm -f "$f"
    echo "$md $s $t: $("$PYC" "$DIR/play.py" --model "$DIR/data/model_$md.pt" --style $s --games $g "$@" --out "$(m "$f")" \
        | grep "^AtTenTris" | sed 's/.*: APP/APP/' | cut -c1-200)" | tee -a "$OUT"
}
play $N normal none 256 --depth 1
play $N normal d2 64 --depth 2 --beam 4
for s in normal danger; do
    play $N $s g20 64 --depth 1 --seed 33001 --garbage 10,3-4,cheese0.2,0.2
    play $N $s g50 64 --depth 1 --seed 33001 --garbage 10,3-4,cheese0.5,0.2
done
args=""
for md in v51_d128_dagger5b_ft $N; do
    for x in normal_none normal_d2 normal_g20 normal_g50 danger_g20 danger_g50; do args="$args ${md}_$x=$(m "$DE")/${md}_$x.games"; done
done
echo "== APP / VS: dagger5b-ft vs + 4 frozen-trained registers" | tee -a "$OUT"
"$PYC" "$DIR/vs_score.py" $args | tee -a "$OUT"
echo "registers (frozen) done"
