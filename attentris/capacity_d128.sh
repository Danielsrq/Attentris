#!/bin/sh
# Capacity test: the v5.1 recipe (same data, seed, schedule, losses) at width d=128 instead of 64 (--micro 2 for VRAM; 4 heads,
# MLP 4d, ~1.06M params vs 275k). Is v5.1 capacity-limited? Then the fit check (fit_gap.py: own training
# positions vs val) and the v5.1 base play test (256 games, depth 1, no garbage, play.py's default seeds)
# against v5.1's (APP 0.587, 1/256 died). Resumable.   sh capacity_d128.sh > capacity_d128.out 2>&1
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR"; . "$ATT/plat.sh"
PYG="$DIR/.venv/$VPY"
PYC="$CPUPY"
OUT="$DIR/results_capacity.txt"
M="$DIR/data/model_v51_d128.pt"
LOG="$DIR/train_v51_d128.out"
if ! grep -q "^epoch 8:" "$LOG" 2>/dev/null; then
    R=""; [ -f "$M" ] && R="--resume"
    "$PYG" "$DIR/train.py" --data "$DIR/data/v51" --batch 512 --device cuda --w-dense 0 --w-q1 0 --w-qs 1 --seed 0 \
        --train train.npz,dagger1.npz,train1m.npz --epochs 8 --lr 1.5e-3 --d 128 --micro 2 --out "$M" $R >> "$LOG" 2>&1
fi
grep -q "^epoch 8:" "$LOG" || { echo "d128: training did not finish (see $LOG)" | tee -a "$OUT"; exit 1; }
echo "== v5.1 at d=128 ($(grep '^params' "$LOG" | tail -1 | cut -d, -f1))" | tee -a "$OUT"
grep "^epoch [0-9]*:" "$LOG" | sed 's/argmax Qs.*Qs [0-9.]* |/|/' | tee -a "$OUT"
echo "== fit check (own training positions vs val): d=128, then v5.1 (d=64)" | tee -a "$OUT"
"$PYG" "$DIR/fit_gap.py" --model "$(wpath "$M")" --train train.npz,dagger1.npz,train1m.npz --n 20000 | tee -a "$OUT"
"$PYG" "$DIR/fit_gap.py" --model "$(wpath "$DIR/data/model_v51.pt")" --train train1m.npz --n 20000 | tee -a "$OUT"
f="$DIR/data/dagger3_eval/v51_d128_normal_none.games"
if ! grep -q "^AtTenTris" "$f" 2>/dev/null; then
    echo "v51_d128 normal none: $("$PYC" "$DIR/play.py" --model "$(wpath "$M")" --style normal --games 256 --depth 1 --out "$(wpath "$f")" \
        | grep "^AtTenTris" | sed 's/.*: APP/APP/' | cut -c1-200)" | tee -a "$OUT"
fi
"$PYC" "$DIR/vs_score.py" "v51_base_none=$(wpath "$DIR/data/dagger3_eval/v51_normal_none.games")" "v51_d128_none=$(wpath "$f")" | tee -a "$OUT"
echo "capacity d128 done"
