#!/bin/sh
# ToMe merge schedules x depth, 4-epoch screens (from scratch, ToMe v4 recipe: pre-tokens, FLAT-2D, 2
# streams, train + dagger1 + train1m, lr 1.5e-3, seed 0):
#   A  4 layers  fixed 0,0.17,0.17,0.17 (ToMe v4's schedule; a 4-epoch baseline)
#   B  6 layers  per-position budget: 20 + max(0, (n0 - 36) / 4) blobs per stream (n0 = the board's
#                tokens; no merging if n0 <= budget), first merge 10%, then equal-ratio steps
#   C  4 layers  the same budget plan (B vs C: depth; A vs C: the budget)
# Per model: rule-token recovery (tome_iou.py, best IoU per rule token vs random merges), 64-game tests
# at depth 1 and depth 2 beam 4 (no garbage) and depth 1 under 80/20 and 50/50 (VS), a blob viewer page.
# Baseline: ToMe v4 (4 layers, 0,0.17x3, 8 epochs). Resumable; one copy at a time (lock).
#   sh tome_sched.sh > tome_sched.out 2>&1
DIR=$(cd "$(dirname "$0")" && pwd)
PYG="$DIR/.venv/Scripts/python.exe"
PYC="$DIR/../external/tetrisbattle/.venv/Scripts/python.exe"
V51="$DIR/data/v51"
D="$DIR/data/tome_sched"
WD=$(cygpath -m "$D")
OUT="$DIR/results_tome_sched.txt"
mkdir "$DIR/.tome_sched.lock" 2>/dev/null || { echo "tome_sched.sh is already running (remove $DIR/.tome_sched.lock if not)"; exit 1; }
trap 'rmdir "$DIR/.tome_sched.lock"' EXIT
mkdir -p "$D"
for cfg in "A 4 0,0.17,0.17,0.17 -" "B 6 0,1,1,1,1,1 20,36,0.25,0.10" "C 4 0,1,1,1 20,36,0.25,0.10"; do
    set -- $cfg; tag=$1; L=$2; MS=$3; BU=""; [ "$4" != "-" ] && BU="--tome-budget $4"
    name=tome_s$tag; M="$DIR/data/model_$name.pt"; LOG="$DIR/train_$name.out"
    if ! grep -q "^epoch 4:" "$LOG" 2>/dev/null; then
        R=""; [ -f "$M" ] && R="--resume"
        "$PYG" "$DIR/train.py" --arch tome --tome-pretok --tome-flat --tome-streams 2 --micro 4 --layers $L --merge $MS $BU --bucket 50 \
            --data "$V51" --train train.npz,dagger1.npz,train1m.npz --epochs 4 --batch 512 --lr 1.5e-3 --device cuda \
            --w-dense 0 --w-q1 0 --w-qs 1 --seed 0 --out "$M" $R >> "$LOG" 2>&1
    fi
    grep -q "^epoch 4:" "$LOG" || { echo "$name: training did not finish (see $LOG)" | tee -a "$OUT"; exit 1; }
    echo "== $name: $L layers, merge $MS ${BU:-fixed fractions}" | tee -a "$OUT"
    grep "^epoch 4:" "$LOG" | sed 's/argmax Qs.*Qs [0-9.]* |/|/' | tee -a "$OUT"
    [ -f "$D/${name}_iou.txt" ] || "$PYG" "$DIR/tome_iou.py" --model "$M" --n 2000 | grep -v "positions$" > "$D/${name}_iou.txt"
    cat "$D/${name}_iou.txt" | tee -a "$OUT"
    [ -f "$DIR/${name}_viz.html" ] || "$PYG" "$DIR/tome_viz.py" --model "$M" --random 24 --out "$DIR/${name}_viz.html" | tail -1
    for t in "d1:--depth 1" "d2:--depth 2 --beam 4" "g20:--depth 1 --seed 33001 --garbage 10,3-4,cheese0.2,0.2" \
             "g50:--depth 1 --seed 33001 --garbage 10,3-4,cheese0.5,0.2"; do
        tt=${t%%:*}; flags=${t#*:}
        f="$D/${name}_$tt.games"
        [ -f "$f" ] && grep -q "^AtTenTris" "$f" && continue
        rm -f "$f"
        echo "$name $tt: $("$PYC" "$DIR/play.py" --model "$M" --games 64 $flags --out "$(cygpath -m "$f")" | grep "^AtTenTris" | sed 's/.*: APP/APP/' | cut -c1-180)" | tee -a "$OUT"
    done
done
args=""
for n in tome_sA tome_sB tome_sC; do for tt in d1 d2 g20 g50; do args="$args ${n}_$tt=$WD/${n}_$tt.games"; done; done
E=$(cygpath -m "$DIR/data/dagger3_eval")
args="$args tome_v4_d1=$E/tome_v4_normal_none.games tome_v4_g20=$E/tome_v4_normal_g20.games tome_v4_g50=$E/tome_v4_normal_g50.games"
"$PYC" "$DIR/vs_score.py" $args | tee -a "$OUT"
echo "tome sched done"
