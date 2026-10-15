#!/bin/sh
# DAgger round 3, training: dagger-ft + a "danger" style (CEM-survival's labels on the students' trouble
# positions, dagger3_more.sh). Warm start from model_v51_dagger_ft.pt, the style table grown to 3 rows
# (danger starts as a copy of normal), 3 epochs of a 70/30 mix (a fresh draw of the existing data each
# epoch : all of the danger data), lr 3e-4; val = the teacher's positions, val2 = held-out danger
# positions (every 10th seed). Then play tests at depth 1 (no depth 3): dagger-ft (normal) vs the new
# model with style normal / danger / auto, with no garbage (256 games), 80/20 and 50/50 (64 each).
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR"; . "$ATT/plat.sh"
PYG="$DIR/.venv/$VPY"
PYC="$CPUPY"
V51="$DIR/data/v51"
D="$DIR/data/dagger3"
OUT="$DIR/results_dagger3_train.txt"
m() { wpath "$1"; }
[ -f "$D/raw_tr/danger_s0.jsonl" ] || "$PYC" "$DIR/split_by_seed.py" --src "$(m "$D/labelled.jsonl")" --tr "$(m "$D/raw_tr")" --val "$(m "$D/raw_val")"
[ -f "$V51/danger.npz" ] || "$PYC" "$DIR/pack.py" --only "$(m "$D/raw_tr")" --name danger --jobs 16 --out "$(m "$V51")" | tail -1
[ -f "$V51/val_danger.npz" ] || "$PYC" "$DIR/pack.py" --only "$(m "$D/raw_val")" --name val_danger --jobs 8 --out "$(m "$V51")" | tail -1
echo "== dagger3 training $(date)" | tee -a "$OUT"
"$PYG" "$DIR/eval_val.py" --model "$DIR/data/model_v51_dagger_ft.pt" --data "$V51" --val val_danger.npz --as-teacher normal | tee -a "$OUT"
# 70/30: the existing files together take 7/3 x (danger positions) per epoch
F=$("$PYC" -c "
import numpy as np, sys
d = '$(m "$V51")/'
nd = len(np.load(d + 'danger.npz')['chosen'])
rest = sum(len(np.load(d + f)['chosen']) for f in ('train.npz', 'dagger1.npz', 'train1m.npz', 'dagger2.npz'))
print('%.4f' % (7 / 3 * nd / rest))")
echo "danger mix: existing files at $F each per epoch" | tee -a "$OUT"
LOG="$DIR/train_v51_dagger3_ft.out"
if ! grep -q "^epoch 3: val2" "$LOG" 2>/dev/null; then
    R=""; [ -f "$DIR/data/model_v51_dagger3_ft.pt" ] && R="--resume"
    "$PYG" "$DIR/train.py" --data "$V51" --batch 512 --device cuda --w-dense 0 --w-q1 0 --w-qs 1 --seed 0 --val2 val_danger.npz \
        --train train.npz@$F,dagger1.npz@$F,train1m.npz@$F,dagger2.npz@$F,danger.npz --epochs 3 --lr 3e-4 --n-teachers 3 \
        --init "$DIR/data/model_v51_dagger_ft.pt" --out "$DIR/data/model_v51_dagger3_ft.pt" $R >> "$LOG" 2>&1
fi
echo "== v51_dagger3_ft" | tee -a "$OUT"
grep "^epoch 3:" "$LOG" | tee -a "$OUT"

play() {   # label model style garbage(or -) games
    g=""; [ "$4" != "-" ] && g="--seed 33001 --garbage $4"
    echo "$1 style $3: $("$PYC" "$DIR/play.py" --model "$DIR/data/model_$2.pt" --games $5 --depth 1 --style $3 $g | grep "^AtTenTris")" | tee -a "$OUT"
}
for gb in "- 256" "10,3-4,cheese0.2,0.2 64" "10,3-4,cheese0.5,0.2 64"; do
    set -- $gb; garb=$1; n=$2
    play dagger-ft v51_dagger_ft normal $garb $n
    for s in normal danger auto; do
        play dagger3-ft v51_dagger3_ft $s $garb $n
    done
done
echo "dagger3 train done $(date)" | tee -a "$OUT"
