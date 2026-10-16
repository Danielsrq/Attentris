#!/bin/sh
# v5.2 (two cross-attention passes) at d=128 through the full v5.1 d128 lineage, both forms:
#   untied = 2 independent cross blocks (--cross-layers 2), loop = the same block applied twice (--cross-loops 2).
# Stages, as v51_d128 -> dagger_ft -> dagger3_ft -> dagger5b_ft (capacity_d128.sh, d128_lineage.sh, dagger5b.sh):
#   base: 8 epochs from scratch, lr 1.5e-3, train + dagger1 + train1m
#   dagger_ft: 3 epochs, lr 3e-4, train / dagger1 / train1m at 0.19 + dagger2, val2 val_dagger2
#   dagger3_ft: + danger style (3 styles), the older files at 0.1712 + danger, val2 val_danger
#   dagger5b_ft: the older files at F (as dagger5b.sh) + danger + dagger4 + dagger5all, val2 val_dagger5all
# Then the standard play block for each final model (depth 1: none 256, d2 64, normal / danger x 80/20, 50/50
# at 64; depth 2 beam 4 under the same garbage) and vs_score against v51_d128_dagger5b_ft on the same seeds.
# GPU: one training at a time, --threads 8. Resumable per stage and per game file.
#   sh attentris/v52_d128.sh > attentris/v52_d128.out 2>&1
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR"; . "$ATT/plat.sh"
PYG="$DIR/.venv/$VPY"
PYC="$CPUPY"
V51="$DIR/data/v51"
OUT="$DIR/results_v52_d128.txt"
m() { wpath "$1"; }
stage() {   # name epochs args...
    name=$1; ep=$2; shift 2
    log="$DIR/train_$name.out"
    if ! grep -q "^epoch $ep: train" "$log" 2>/dev/null; then
        R=""; [ -f "$DIR/data/model_$name.pt" ] && R="--resume"
        echo "$name: training $(date '+%H:%M')" | tee -a "$OUT"
        "$PYG" "$DIR/train.py" --data "$(m "$V51")" --batch 512 --threads 8 --device cuda --w-dense 0 --w-q1 0 --w-qs 1 --seed 0 \
            --d 128 --micro 2 --out "$(m "$DIR/data/model_$name.pt")" $R "$@" >> "$log" 2>&1
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
echo "== v5.2 at d=128, untied and loop, full lineage $(date '+%Y-%m-%d %H:%M') (dagger5b mix F=$F)" | tee -a "$OUT"
for v in untied loop; do
    case $v in untied) FLAG="--cross-layers 2" ;; loop) FLAG="--cross-loops 2" ;; esac
    b=v52_d128_$v
    stage $b 8 --train train.npz,dagger1.npz,train1m.npz --epochs 8 --lr 1.5e-3 $FLAG
    stage ${b}_dagger_ft 3 --train train.npz@0.19,dagger1.npz@0.19,train1m.npz@0.19,dagger2.npz --epochs 3 --lr 3e-4 $FLAG \
        --init "$(m "$DIR/data/model_$b.pt")" --val2 val_dagger2.npz
    stage ${b}_dagger3_ft 3 --train train.npz@0.1712,dagger1.npz@0.1712,train1m.npz@0.1712,dagger2.npz@0.1712,danger.npz \
        --epochs 3 --lr 3e-4 --n-teachers 3 $FLAG --init "$(m "$DIR/data/model_${b}_dagger_ft.pt")" --val2 val_danger.npz
    stage ${b}_dagger5b_ft 3 --train train.npz@$F,dagger1.npz@$F,train1m.npz@$F,dagger2.npz@$F,danger.npz,dagger4.npz,dagger5all.npz \
        --epochs 3 --lr 3e-4 --n-teachers 3 $FLAG --init "$(m "$DIR/data/model_${b}_dagger3_ft.pt")" --val2 val_dagger5all.npz
done
DE="$DIR/data/dagger3_eval"
play() {   # model style tag games flags...
    f="$DE/$1_$2_$3.games"; md=$1; s=$2; t=$3; g=$4; shift 4
    [ -f "$f" ] && grep -q "^AtTenTris" "$f" && return
    rm -f "$f"
    echo "$md $s $t: $("$PYC" "$DIR/play.py" --model "$DIR/data/model_$md.pt" --style $s --games $g "$@" --out "$(m "$f")" \
        | grep "^AtTenTris" | sed 's/.*: APP/APP/' | cut -c1-200)" | tee -a "$OUT"
}
G20="--seed 33001 --garbage 10,3-4,cheese0.2,0.2"
G50="--seed 33001 --garbage 10,3-4,cheese0.5,0.2"
for v in untied loop; do
    n=v52_d128_${v}_dagger5b_ft
    play $n normal none 256 --depth 1
    play $n normal d2 64 --depth 2 --beam 4
    for s in normal danger; do
        play $n $s g20 64 --depth 1 $G20
        play $n $s g50 64 --depth 1 $G50
        play $n $s d2g20 64 --depth 2 --beam 4 $G20
        play $n $s d2g50 64 --depth 2 --beam 4 $G50
    done
done
args=""
for md in v51_d128_dagger5b_ft v52_d128_untied_dagger5b_ft v52_d128_loop_dagger5b_ft; do
    for x in normal_none normal_d2 normal_g20 normal_g50 danger_g20 danger_g50 normal_d2g20 normal_d2g50 danger_d2g20 danger_d2g50; do
        [ -f "$DE/${md}_$x.games" ] && args="$args ${md}_$x=$(m "$DE")/${md}_$x.games"
    done
done
echo "== APP / VS: v5.1 d128 dagger5b-ft vs v5.2 d128 untied / loop dagger5b-ft (same lineage, same seeds)" | tee -a "$OUT"
"$PYC" "$DIR/vs_score.py" $args | tee -a "$OUT"
echo "v52 d128 done"
