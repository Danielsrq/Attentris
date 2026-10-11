#!/bin/sh
# ToMe v4 + the DAgger round 3 danger style: the same recipe as dagger3-ft (warm start, style table
# grown to 3 rows, 3 epochs of 70/30, lr 3e-4); the 70 is ToMe v4's own data (train, dagger1,
# train1m; it never had dagger2). Then 64-game tests at depth 1 (no depth 3): ToMe v4 vs ToMe-dagger3
# in style normal / danger / auto, no garbage, 80/20 and 50/50 (seeds 5001+ / 33001+).
DIR=$(cd "$(dirname "$0")" && pwd)
PYG="$DIR/.venv/Scripts/python.exe"
PYC="$DIR/../external/tetrisbattle/.venv/Scripts/python.exe"
V51="$DIR/data/v51"
D="$DIR/data/dagger3_eval"
WD=$(cygpath -m "$D")
OUT="$DIR/results_tome_dagger3.txt"
M="$DIR/data/model_tome_v4_dagger3_ft.pt"
mkdir -p "$D"
[ -f "$V51/danger.pretok.npy" ] && [ -f "$V51/val_danger.pretok.npy" ] || \
    "$PYG" "$DIR/pretok.py" --data "$(cygpath -m "$V51")" --files danger.npz,val_danger.npz | grep -v pinned
F=$("$PYC" -c "
import numpy as np
d = '$(cygpath -m "$V51")/'
nd = len(np.load(d + 'danger.npz')['chosen'])
rest = sum(len(np.load(d + f)['chosen']) for f in ('train.npz', 'dagger1.npz', 'train1m.npz'))
print('%.4f' % (7 / 3 * nd / rest))")
echo "== ToMe v4 + danger style, mix: existing files at $F each per epoch, $(date)" | tee -a "$OUT"
LOG="$DIR/train_tome_v4_dagger3_ft.out"
if ! grep -q "^epoch 3: val2" "$LOG" 2>/dev/null; then
    R=""; [ -f "$M" ] && R="--resume"
    "$PYG" "$DIR/train.py" --arch tome --tome-pretok --tome-flat --tome-streams 2 --micro 4 --merge 0,0.17,0.17,0.17 --bucket 50 \
        --data "$V51" --train train.npz@$F,dagger1.npz@$F,train1m.npz@$F,danger.npz --epochs 3 --batch 512 --lr 3e-4 --device cuda \
        --w-dense 0 --w-q1 0 --w-qs 1 --seed 0 --val2 val_danger.npz --n-teachers 3 --init "$DIR/data/model_tome_v4.pt" --out "$M" $R >> "$LOG" 2>&1
fi
grep -q "^epoch 3: val2" "$LOG" || { echo "training did not finish (see $LOG); no play tests" | tee -a "$OUT"; exit 1; }
grep -E "^epoch [0-9]+:" "$LOG" | sed 's/argmax Qs.*Qs [0-9.]* |/|/' | tee -a "$OUT"
for g in "none:-" "g20:10,3-4,cheese0.2,0.2" "g50:10,3-4,cheese0.5,0.2"; do
    tag=${g%%:*}; garb=${g#*:}
    gs="--games 64"; [ "$garb" != "-" ] && gs="--games 64 --seed 33001 --garbage $garb"
    for ms in "tome_v4:normal" "tome_v4_dagger3_ft:normal" "tome_v4_dagger3_ft:danger" "tome_v4_dagger3_ft:auto"; do
        m=${ms%%:*}; s=${ms#*:}
        f="$D/${m}_${s}_$tag.games"
        [ -f "$f" ] && grep -q "^AtTenTris" "$f" && continue
        rm -f "$f"
        echo "$m $s $tag: $("$PYC" "$DIR/play.py" --model "$DIR/data/model_$m.pt" --depth 1 --style $s $gs --out "$(cygpath -m "$f")" \
            | grep "^AtTenTris" | sed 's/.*: APP/APP/' | cut -c1-200)" | tee -a "$OUT"
    done
done
args=""
for tag in none g20 g50; do
    for ms in tome_v4_normal tome_v4_dagger3_ft_normal tome_v4_dagger3_ft_danger tome_v4_dagger3_ft_auto; do args="$args ${ms}_$tag=$WD/${ms}_$tag.games"; done
done
"$PYC" "$DIR/vs_score.py" $args | tee -a "$OUT"
echo "tome dagger3 done"
