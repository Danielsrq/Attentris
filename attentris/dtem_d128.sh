#!/bin/sh
# DTEM on ToMe at d=128 (Lee & Hong, NeurIPS 2024, arXiv 2412.10569; reference code github.com/movinghoon/dtem):
# the finished ToMe l4g d128 dagger5b-ft (tome_d128.sh) gets a learned merge metric per merging layer (model_tome.py
# --tome-metric: a linear map of the detached normed layer input, initialised from the key rows each stream used,
# so it starts exactly as ToMe). DTEM's modular recipe: the backbone frozen (--freeze-base), only the metric trains,
# through soft merges (soft greedy matching + mass-weighted soft merging, tau 0.1 / 0.1); validation and play with
# hard merges as ToMe, by the learned similarity. Same budget, same backbone: any change is the merge choice.
# 3 epochs of the dagger5b mix at LR (default 1e-4, DTEM's own default; lr 1e-3 (tome_d128_l4g_dtem_lr1e-3) made the
# soft and the hard loss worse: val 0.641 -> 0.624, the soft loss on held-out positions 1.87 -> 2.10, Adam's steps
# far past the range where the sharp soft-merge loss (cosine / 0.01) is near linear).
# Then the tests of tome_d128.sh against v5.1 d128 and ToMe l4g d128 (same seeds).
# GPU ~40 min (--micro 8: soft merges keep every token), CPU ~1.1 h. Resumable per stage and per game file:
#   LR=1e-4 sh attentris/dtem_d128.sh > attentris/dtem_d128_lr1e-4.out 2>&1
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR"; . "$ATT/plat.sh"
PYG="$DIR/.venv/$VPY"
PYC="$CPUPY"
V51="$DIR/data/v51"
OUT="$DIR/results_dtem_d128.txt"
LR=${LR:-1e-4}
B=tome_d128_l4g_dtem_lr$LR
ARCH="--arch tome --tome-pretok --tome-flat --tome-streams 2 --tome-pure --bucket 50 --tome-split greedy \
--tome-budget 20,36,0.25,0.10 --tome-metric --layers 4 --merge 0,1,1,1 --d 128"
m() { wpath "$1"; }
stage() {   # name epochs args...
    name=$1; ep=$2; shift 2
    log="$DIR/train_$name.out"
    mi=8
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
echo "== DTEM (modular, lr $LR) on ToMe l4g d=128 dagger5b-ft $(date '+%Y-%m-%d %H:%M') (dagger5b mix F=$F)" | tee -a "$OUT"
stage $B 3 --train train.npz@$F,dagger1.npz@$F,train1m.npz@$F,dagger2.npz@$F,danger.npz,dagger4.npz,dagger5all.npz \
    --epochs 3 --lr $LR --new-lr $LR --freeze-base --n-teachers 3 --init "$(m "$DIR/data/model_tome_d128_l4g_dagger5b_ft.pt")" --val2 val_dagger5all.npz
n=$B
IOU="$DIR/data/dtem_d128/${n}_iou.txt"
mkdir -p "$DIR/data/dtem_d128"
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
for md in v51_d128_dagger5b_ft tome_d128_l4g_dagger5b_ft $n; do
    for x in normal_none normal_d2 normal_g20 normal_g50 danger_g20 danger_g50 normal_d2g20 normal_d2g50 danger_d2g20 danger_d2g50; do
        [ -f "$DE/${md}_$x.games" ] && args="$args ${md}_$x=$(m "$DE")/${md}_$x.games"
    done
done
echo "== APP / VS: v5.1 d128 vs ToMe l4g d128 vs DTEM on ToMe l4g d128 (same seeds)" | tee -a "$OUT"
"$PYC" "$DIR/vs_score.py" $args | tee -a "$OUT"
MODELS="v51_d128_dagger5b_ft tome_d128_l4g_dagger5b_ft $n" sh "$DIR/ablation_more.sh" | tee -a "$OUT"
echo "dtem d128 done"
