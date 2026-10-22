#!/bin/sh
# ToMe at d=128 through the full v5.1 d128 lineage, against v51_d128_dagger5b_ft (the ablation recipes,
# ablation_d128.sh, with the ToMe v4.2 l4g architecture of tome_v42.sh instead of the rule tokens):
#   l4g = 4 layers, merges in layers 2-4 (--merge 0,1,1,1), 2 streams, pre-tokens, FLAT-2D, pure merges,
#         budget 20 + max(0, (n0 - 36) / 4) blobs per stream, first merge 10%, greedy matching
# Stages: base 8 epochs (lr 1.5e-3, train + dagger1 + train1m), dagger_ft / dagger3_ft / dagger5b_ft 3 epochs
# each (lr 3e-4), the same mixes and val2 sets as the v5.1 d128 lineage. --micro 4 (d128: 5.7 GB at the base
# stage); a stage that runs out of GPU memory resumes at twice the micro-batches.
# Then rule-token recovery (tome_iou.py), the standard play block (depth 1: none 256, d2 64, normal / danger x
# 80/20, 50/50 at depth 1 and depth 2 beam 4, 64 each, seeds 33001+), vs_score, and ablation_more.sh (64 more
# games per garbage setting on seeds 53001+, paired McNemar vs the baseline).
# GPU ~2 h, CPU ~1.5 h. Resumable per stage and per game file:
#   sh attentris/tome_d128.sh > attentris/tome_d128.out 2>&1
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR"; . "$ATT/plat.sh"
PYG="$DIR/.venv/$VPY"
PYC="$CPUPY"
V51="$DIR/data/v51"
OUT="$DIR/results_tome_d128.txt"
B=tome_d128_l4g
ARCH="--arch tome --tome-pretok --tome-flat --tome-streams 2 --tome-pure --bucket 50 --tome-split greedy \
--tome-budget 20,36,0.25,0.10 --layers 4 --merge 0,1,1,1 --d 128"
m() { wpath "$1"; }
stage() {   # name epochs args...
    name=$1; ep=$2; shift 2
    log="$DIR/train_$name.out"
    mi=4
    while ! grep -q "^epoch $ep: train" "$log" 2>/dev/null; do
        R=""; [ -f "$DIR/data/model_$name.pt" ] && R="--resume"
        echo "$name: training, --micro $mi $(date '+%H:%M')" | tee -a "$OUT"
        "$PYG" "$DIR/train.py" $ARCH --data "$(m "$V51")" --batch 512 --threads 8 --device cuda --w-dense 0 --w-q1 0 --w-qs 1 \
            --seed 0 --micro $mi --out "$(m "$DIR/data/model_$name.pt")" $R "$@" >> "$log" 2>&1
        grep -q "^epoch $ep: train" "$log" && break
        if tail -50 "$log" | grep -q -i "out of memory" && [ $mi -lt 16 ]; then
            mi=$((mi * 2)); echo "$name: out of GPU memory, again at --micro $mi" | tee -a "$OUT"
        else
            echo "$name: training did not finish (see $log)" | tee -a "$OUT"; exit 1
        fi
    done
    echo "== $name" | tee -a "$OUT"
    grep "^epoch $ep:" "$log" | sed 's/argmax Qs top1 [0-9.]* \/ [0-9.]*, Q1.*Qs [0-9.]* |/|/' | tee -a "$OUT"
}
F=$("$PYC" -c "
import numpy as np
d = '$(m "$V51")/'
new = sum(len(np.load(d + f)['chosen']) for f in ('danger.npz', 'dagger4.npz', 'dagger5all.npz'))
rest = sum(len(np.load(d + f)['chosen']) for f in ('train.npz', 'dagger1.npz', 'train1m.npz', 'dagger2.npz'))
print('%.4f' % (7 / 3 * new / rest))")
echo "== ToMe l4g at d=128, full lineage $(date '+%Y-%m-%d %H:%M') (dagger5b mix F=$F)" | tee -a "$OUT"
# ToMe pre-tokens (pretok.py) for every file the lineage reads; DAgger 4 / 5 had none before this run
for f in train dagger1 train1m dagger2 danger dagger4 dagger5all val val_dagger2 val_danger val_dagger5all; do
    [ -f "$V51/$f.pretok.npy" ] || "$PYC" "$DIR/pretok.py" --data "$(m "$V51")" --files $f.npz | tail -1 | tee -a "$OUT"
done
stage $B 8 --train train.npz,dagger1.npz,train1m.npz --epochs 8 --lr 1.5e-3
stage ${B}_dagger_ft 3 --train train.npz@0.19,dagger1.npz@0.19,train1m.npz@0.19,dagger2.npz --epochs 3 --lr 3e-4 \
    --init "$(m "$DIR/data/model_$B.pt")" --val2 val_dagger2.npz
stage ${B}_dagger3_ft 3 --train train.npz@0.1712,dagger1.npz@0.1712,train1m.npz@0.1712,dagger2.npz@0.1712,danger.npz \
    --epochs 3 --lr 3e-4 --n-teachers 3 --init "$(m "$DIR/data/model_${B}_dagger_ft.pt")" --val2 val_danger.npz
stage ${B}_dagger5b_ft 3 --train train.npz@$F,dagger1.npz@$F,train1m.npz@$F,dagger2.npz@$F,danger.npz,dagger4.npz,dagger5all.npz \
    --epochs 3 --lr 3e-4 --n-teachers 3 --init "$(m "$DIR/data/model_${B}_dagger3_ft.pt")" --val2 val_dagger5all.npz
n=${B}_dagger5b_ft
IOU="$DIR/data/tome_d128/${n}_iou.txt"
mkdir -p "$DIR/data/tome_d128"
[ -s "$IOU" ] || "$PYC" "$DIR/tome_iou.py" --model "$(m "$DIR/data/model_$n.pt")" --n 2000 | grep -v "positions$" > "$IOU"
echo "== $n: rule-token recovery" | tee -a "$OUT"; tee -a "$OUT" < "$IOU"
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
play $n normal none 256 --depth 1
play $n normal d2 64 --depth 2 --beam 4
for s in normal danger; do
    play $n $s g20 64 --depth 1 $G20
    play $n $s g50 64 --depth 1 $G50
    play $n $s d2g20 64 --depth 2 --beam 4 $G20
    play $n $s d2g50 64 --depth 2 --beam 4 $G50
done
args=""
for md in v51_d128_dagger5b_ft $n; do
    for x in normal_none normal_d2 normal_g20 normal_g50 danger_g20 danger_g50 normal_d2g20 normal_d2g50 danger_d2g20 danger_d2g50; do
        [ -f "$DE/${md}_$x.games" ] && args="$args ${md}_$x=$(m "$DE")/${md}_$x.games"
    done
done
echo "== APP / VS: v5.1 d128 dagger5b-ft vs ToMe l4g d128 dagger5b-ft (same lineage, same seeds)" | tee -a "$OUT"
"$PYC" "$DIR/vs_score.py" $args | tee -a "$OUT"
MODELS="v51_d128_dagger5b_ft $n" sh "$DIR/ablation_more.sh" | tee -a "$OUT"
echo "tome d128 done"
