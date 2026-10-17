#!/bin/sh
# The from-scratch register model (v51_d128_reg4, R = 4) through the same DAgger lineage as dagger5b-ft
# (v52_d128.sh recipe):
#   dagger_ft: 3 epochs, lr 3e-4, train / dagger1 / train1m at 0.19 + dagger2, val2 val_dagger2
#   dagger3_ft: + danger style (3 styles), the older files at 0.1712 + danger, val2 val_danger
#   dagger5b_ft: the older files at F (as dagger5b.sh) + danger + dagger4 + dagger5all, val2 val_dagger5all
# Then register_usage.py on the final model, and play tests vs v51_d128_dagger5b_ft on the same seeds:
# depth 1 none 256, depth 2 beam 4 (64), normal and danger 80/20 (64). 50/50 only if it looks promising.
# Resumable per stage and per game file.   sh attentris/registers_lineage.sh > attentris/registers_lineage.out 2>&1
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR"; . "$ATT/plat.sh"
PYG="$DIR/.venv/$VPY"
PYC="$CPUPY"
V51="$DIR/data/v51"
OUT="$DIR/results_registers.txt"
m() { wpath "$1"; }
stage() {   # name epochs args...
    name=$1; ep=$2; shift 2
    log="$DIR/train_$name.out"
    if ! grep -q "^epoch $ep: train" "$log" 2>/dev/null; then
        R=""; [ -f "$DIR/data/model_$name.pt" ] && R="--resume"
        echo "$name: training $(date '+%H:%M')" | tee -a "$OUT"
        "$PYG" "$DIR/train.py" --data "$(m "$V51")" --batch 512 --device cuda --w-dense 0 --w-q1 0 --w-qs 1 --seed 0 \
            --d 128 --micro 2 --registers 4 --out "$(m "$DIR/data/model_$name.pt")" $R "$@" >> "$log" 2>&1
    fi
    grep -q "^epoch $ep: train" "$log" || { echo "$name: training did not finish (see $log)" | tee -a "$OUT"; exit 1; }
    echo "== $name" | tee -a "$OUT"
    grep "^epoch $ep:" "$log" | sed 's/argmax Qs top1 [0-9.]* \/ [0-9.]*, Q1.*Qs [0-9.]* |/|/' | tee -a "$OUT"
}
F=$("$PYC" -c "
import numpy as np
d = '$(m "$V51")/'
new = sum(len(np.load(d + f)['chosen']) for f in ('danger.npz', 'dagger4.npz', 'dagger5all.npz'))
rest = sum(len(np.load(d + f)['chosen']) for f in ('train.npz', 'dagger1.npz', 'train1m.npz', 'dagger2.npz'))
print('%.4f' % (7 / 3 * new / rest))")
b=v51_d128_reg4
[ -f "$DIR/data/model_$b.pt" ] || { echo "no $b (registers_steps.sh step 3)"; exit 1; }
echo "== registers (R = 4) through the DAgger lineage $(date '+%Y-%m-%d %H:%M') (dagger5b mix F=$F)" | tee -a "$OUT"
stage ${b}_dagger_ft 3 --train train.npz@0.19,dagger1.npz@0.19,train1m.npz@0.19,dagger2.npz --epochs 3 --lr 3e-4 \
    --init "$(m "$DIR/data/model_$b.pt")" --val2 val_dagger2.npz
stage ${b}_dagger3_ft 3 --train train.npz@0.1712,dagger1.npz@0.1712,train1m.npz@0.1712,dagger2.npz@0.1712,danger.npz \
    --epochs 3 --lr 3e-4 --n-teachers 3 --init "$(m "$DIR/data/model_${b}_dagger_ft.pt")" --val2 val_danger.npz
stage ${b}_dagger5b_ft 3 --train train.npz@$F,dagger1.npz@$F,train1m.npz@$F,dagger2.npz@$F,danger.npz,dagger4.npz,dagger5all.npz \
    --epochs 3 --lr 3e-4 --n-teachers 3 --init "$(m "$DIR/data/model_${b}_dagger3_ft.pt")" --val2 val_dagger5all.npz
grep "^epoch 3:" "$DIR/train_v51_d128_dagger5b_ft.out" | sed 's/^/dagger5b-ft /; s/argmax Qs top1 [0-9.]* \/ [0-9.]*, Q1.*Qs [0-9.]* |/|/' | tee -a "$OUT"
N=${b}_dagger5b_ft
U="$DIR/register_usage_$N.txt"
[ -s "$U" ] || "$PYG" "$DIR/register_usage.py" --model "$DIR/data/model_$N.pt" --val val.npz,val_dagger5all.npz > "$U" 2>&1
cat "$U" | tee -a "$OUT"
DE="$DIR/data/dagger3_eval"
play() {   # model style tag games flags...
    f="$DE/$1_$2_$3.games"; md=$1; s=$2; t=$3; g=$4; shift 4
    [ -f "$f" ] && grep -q "^AtTenTris" "$f" && return
    rm -f "$f"
    echo "$md $s $t: $("$PYC" "$DIR/play.py" --model "$DIR/data/model_$md.pt" --style $s --games $g "$@" --out "$(m "$f")" \
        | grep "^AtTenTris" | sed 's/.*: APP/APP/' | cut -c1-200)" | tee -a "$OUT"
}
G20="--seed 33001 --garbage 10,3-4,cheese0.2,0.2"
play $N normal none 256 --depth 1
play $N normal d2 64 --depth 2 --beam 4
play $N normal g20 64 --depth 1 $G20
play $N danger g20 64 --depth 1 $G20
args=""
for md in v51_d128_dagger5b_ft $N; do
    for x in normal_none normal_d2 normal_g20 danger_g20; do args="$args ${md}_$x=$(m "$DE")/${md}_$x.games"; done
done
echo "== APP / VS: dagger5b-ft vs the register model through the same lineage" | tee -a "$OUT"
"$PYC" "$DIR/vs_score.py" $args | tee -a "$OUT"
echo "registers lineage done"
