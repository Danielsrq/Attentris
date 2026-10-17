#!/bin/sh
# Register tokens (R = 4), with the backbone training too:
#   step 2  v51_d128_reg4_ft: dagger5b-ft + registers (key bias starts at -2), all weights train, 3 epochs on the dagger5b
#           mix (backbone lr 3e-4, registers 1e-3). Compared with v51_d128_ctrl_ft (dagger5b-ft + the same 3 epochs, no registers).
#   step 3  v51_d128_reg4: from scratch, the v51_d128 recipe (8 epochs, lr 1.5e-3, train + dagger1 + train1m) with
#           registers (key bias starts at 0). Compared with v51_d128.
# Each: register_usage.py (attention share, knock-out and mean ablation) and play tests without 50/50 garbage:
# depth 1 no garbage (256), depth 2 beam 4 (64), 80/20 at depth 1 (64; step 2 in normal and danger style).
# Resumable.   sh attentris/registers_steps.sh > attentris/registers_steps.out 2>&1
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR"; . "$ATT/plat.sh"
PYG="$DIR/.venv/$VPY"
PYC="$CPUPY"
m() { wpath "$1"; }
V51="$DIR/data/v51"
OUT="$DIR/results_registers.txt"
DE="$DIR/data/dagger3_eval"
play() {   # model style tag games flags...
    f="$DE/$1_$2_$3.games"; md=$1; s=$2; t=$3; g=$4; shift 4
    [ -f "$f" ] && grep -q "^AtTenTris" "$f" && return
    rm -f "$f"
    echo "$md $s $t: $("$PYC" "$DIR/play.py" --model "$DIR/data/model_$md.pt" --style $s --games $g "$@" --out "$(m "$f")" \
        | grep "^AtTenTris" | sed 's/.*: APP/APP/' | cut -c1-200)" | tee -a "$OUT"
}
usage() {   # model vals
    U="$DIR/register_usage_$1.txt"
    [ -s "$U" ] || "$PYG" "$DIR/register_usage.py" --model "$DIR/data/model_$1.pt" --val "$2" > "$U" 2>&1
    cat "$U" | tee -a "$OUT"
}

# --- step 2: unfrozen fine-tune from dagger5b-ft ---
N=v51_d128_reg4_ft
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
        --registers 4 --reg-bias-init -2 --new-lr 1e-3 \
        --init "$DIR/data/model_v51_d128_dagger5b_ft.pt" --out "$DIR/data/model_$N.pt" $R >> "$LOG" 2>&1
fi
grep -q "^epoch 3: val2" "$LOG" || { echo "$N: training did not finish (see $LOG)" | tee -a "$OUT"; exit 1; }
echo "== step 2: $N vs v51_d128_ctrl_ft (validation, epoch 3)" | tee -a "$OUT"
grep "^epoch 3:" "$LOG" | sed 's/argmax Qs top1 [0-9.]* \/ [0-9.]*, Q1.*Qs [0-9.]* |/|/' | tee -a "$OUT"
grep "^epoch 3:" "$DIR/train_v51_d128_ctrl_ft.out" | sed 's/^/ctrl_ft /; s/argmax Qs top1 [0-9.]* \/ [0-9.]*, Q1.*Qs [0-9.]* |/|/' | tee -a "$OUT"
usage $N val.npz,val_dagger5all.npz
play $N normal none 256 --depth 1
play $N normal d2 64 --depth 2 --beam 4
play $N normal g20 64 --depth 1 --seed 33001 --garbage 10,3-4,cheese0.2,0.2
play $N danger g20 64 --depth 1 --seed 33001 --garbage 10,3-4,cheese0.2,0.2
args=""
for md in v51_d128_ctrl_ft $N; do
    for x in normal_none normal_d2 normal_g20 danger_g20; do args="$args ${md}_$x=$(m "$DE")/${md}_$x.games"; done
done
echo "== APP / VS: step 2, ctrl_ft vs + 4 registers (all weights trained)" | tee -a "$OUT"
"$PYC" "$DIR/vs_score.py" $args | tee -a "$OUT"

# --- step 3: from scratch ---
N=v51_d128_reg4
LOG="$DIR/train_$N.out"
if ! grep -q "^epoch 8:" "$LOG" 2>/dev/null; then
    R=""; [ -f "$DIR/data/model_$N.pt" ] && R="--resume"
    "$PYG" "$DIR/train.py" --data "$V51" --batch 512 --device cuda --w-dense 0 --w-q1 0 --w-qs 1 --seed 0 \
        --train train.npz,dagger1.npz,train1m.npz --epochs 8 --lr 1.5e-3 --d 128 --micro 2 \
        --registers 4 --reg-bias-init 0 --out "$DIR/data/model_$N.pt" $R >> "$LOG" 2>&1
fi
grep -q "^epoch 8:" "$LOG" || { echo "$N: training did not finish (see $LOG)" | tee -a "$OUT"; exit 1; }
echo "== step 3: $N vs v51_d128 (validation, epoch 8)" | tee -a "$OUT"
grep "^epoch 8:" "$LOG" | sed 's/argmax Qs top1 [0-9.]* \/ [0-9.]*, Q1.*Qs [0-9.]* |/|/' | tee -a "$OUT"
grep "^epoch 8:" "$DIR/train_v51_d128.out" | sed 's/^/v51_d128 /; s/argmax Qs top1 [0-9.]* \/ [0-9.]*, Q1.*Qs [0-9.]* |/|/' | tee -a "$OUT"
usage $N val.npz
play $N normal none 256 --depth 1
play $N normal d2 64 --depth 2 --beam 4
play $N normal g20 64 --depth 1 --seed 33001 --garbage 10,3-4,cheese0.2,0.2
args=""
for md in v51_d128 $N; do
    for x in normal_none normal_d2 normal_g20; do args="$args ${md}_$x=$(m "$DE")/${md}_$x.games"; done
done
echo "== APP / VS: step 3, v51_d128 vs + 4 registers (from scratch)" | tee -a "$OUT"
"$PYC" "$DIR/vs_score.py" $args | tee -a "$OUT"
echo "registers steps 2-3 done"
