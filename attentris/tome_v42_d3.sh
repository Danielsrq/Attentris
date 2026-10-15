#!/bin/sh
# DAgger 3 fine-tune of a ToMe v4.2 dagger-ft (RUN=l4g: 4 layers, RUN=l6g: 6 layers, budget 15; greedy
# matching, pure merges), as the v5.x
# dagger3-ft recipe (v52_dagger.sh): 70/30 with danger.npz over train / dagger1 / train1m / dagger2
# (0.1712 each), 3 epochs, lr 3e-4, a third style row (danger), val2 = val_danger.npz; --micro 8 (danger
# boards have many tokens). Then rule-token recovery and 64 games: normal style at depth 1 / depth 2 beam 4
# / 80/20 / 50/50, danger style at 80/20 / 50/50; one APP / VS table with l4g dagger-ft, v4.1-dagger3 and
# v5.1 dagger3-ft on the same seeds. GPU for training only, tests on the CPU. Resumable; one copy at a time.
# With RUN=l6g it waits for that run's DAgger 2 fine-tune (tome_v42.sh), and its games wait until
# tome_v42_eval.sh is done (no two 16-worker play runs at once).
#   RUN=l4g sh tome_v42_d3.sh > tome_v42_d3.out 2>&1
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR"; . "$ATT/plat.sh"
PYG="$DIR/.venv/$VPY"
PYC="$CPUPY"
V51="$DIR/data/v51"
D="$DIR/data/tome_v42"
WD=$(wpath "$D")
OUT="$DIR/results_tome_v42.txt"
mkdir "$DIR/.tome_v42_d3.lock" 2>/dev/null || { echo "tome_v42_d3.sh is already running (remove $DIR/.tome_v42_d3.lock if not)"; exit 1; }
trap 'rmdir "$DIR/.tome_v42_d3.lock"' EXIT
mkdir -p "$D"
RUN=${RUN:-l4g}
m=tome_v42_${RUN}_dagger3_ft
M="$DIR/data/model_$m.pt"
LOG="$DIR/train_$m.out"
BASE="--arch tome --tome-pretok --tome-flat --tome-streams 2 --tome-pure --bucket 50 --tome-split greedy"
case $RUN in
    l4g) ARCH="$BASE --tome-budget 20,36,0.25,0.10 --layers 4 --merge 0,1,1,1 --micro 8";;
    l6g) ARCH="$BASE --tome-budget 15,36,0.25,0.10 --layers 6 --merge 0,1,1,1,1,1 --micro 16";;
    *) echo "unknown RUN $RUN"; exit 1;;
esac
until grep -q "^epoch 3: val2" "$DIR/train_tome_v42_${RUN}_dagger_ft.out" 2>/dev/null; do
    grep -q "tome_v42_${RUN}.*training did not finish" "$OUT" 2>/dev/null && { echo "no ${RUN} dagger-ft, stopping"; exit 1; }
    sleep 60
done
sleep 30
F=0.1712
if ! grep -q "^epoch 3: val2" "$LOG" 2>/dev/null; then
    R=""; [ -f "$M" ] && R="--resume"
    echo "$(date '+%H:%M') $m: training" | tee -a "$OUT"
    "$PYG" "$DIR/train.py" $ARCH --data "$V51" --batch 512 --device cuda --w-dense 0 --w-q1 0 --w-qs 1 --seed 0 --out "$M" $R \
        --train train.npz@$F,dagger1.npz@$F,train1m.npz@$F,dagger2.npz@$F,danger.npz --epochs 3 --lr 3e-4 --n-teachers 3 \
        --init "$DIR/data/model_tome_v42_${RUN}_dagger_ft.pt" --val2 val_danger.npz >> "$LOG" 2>&1
fi
grep -q "^epoch 3: val2" "$LOG" || { echo "$m: training did not finish (see $LOG)" | tee -a "$OUT"; exit 1; }
echo "== $m ($(date '+%H:%M'))" | tee -a "$OUT"
grep "^epoch 3:" "$LOG" | sed 's/argmax Qs.*Qs [0-9.]* |/|/' | tee -a "$OUT"
[ -f "$D/${m}_iou.txt" ] || "$PYG" "$DIR/tome_iou.py" --model "$M" --n 2000 | grep -v "positions$" > "$D/${m}_iou.txt"
echo "== $m: rule-token recovery" | tee -a "$OUT"; cat "$D/${m}_iou.txt" | tee -a "$OUT"
play() {   # style tag flags...
    s=$1; tag=$2; shift 2
    f="$D/${m}_${s}_$tag.games"
    [ -f "$f" ] && grep -q "^AtTenTris" "$f" && return
    rm -f "$f"
    echo "$m $s $tag: $("$PYC" "$DIR/play.py" --model "$M" --games 64 --style $s "$@" --out "$(wpath "$f")" \
        | grep "^AtTenTris" | sed 's/.*: APP/APP/' | cut -c1-200)" | tee -a "$OUT"
}
while [ -d "$DIR/.tome_v42_eval.lock" ]; do sleep 60; done        # one play run at a time
G20="--depth 1 --seed 33001 --garbage 10,3-4,cheese0.2,0.2"
G50="--depth 1 --seed 33001 --garbage 10,3-4,cheese0.5,0.2"
play normal d1 --depth 1
play normal d2 --depth 2 --beam 4
play normal g20 $G20
play normal g50 $G50
play danger g20 $G20
play danger g50 $G50
args=""
for x in normal_d1 normal_d2 normal_g20 normal_g50 danger_g20 danger_g50; do args="$args ${m}_$x=$WD/${m}_$x.games"; done
for t in d1 d2 g20 g50; do args="$args tome_v42_${RUN}_dagger_ft_normal_$t=$WD/tome_v42_${RUN}_dagger_ft_normal_$t.games"; done
[ $RUN = l4g ] || for x in normal_d1 normal_d2 normal_g20 normal_g50 danger_g20 danger_g50; do
    args="$args tome_v42_l4g_dagger3_ft_$x=$WD/tome_v42_l4g_dagger3_ft_$x.games"; done
V=$(wpath "$DIR/data/tome_v41"); E=$(wpath "$DIR/data/dagger3_eval")
for x in normal_d1 normal_g20 normal_g50 danger_g20 danger_g50; do args="$args tome_v41_dagger3_ft_$x=$V/tome_v41_dagger3_ft_$x.games"; done
args="$args v51_dagger3_ft_normal_d1=$E/v51_dagger3_ft_normal_none.games v51_dagger3_ft_normal_d2=$E/v51_dagger3_ft_normal_d2.games"
for x in normal_g20 normal_g50 danger_g20 danger_g50; do args="$args v51_dagger3_ft_$x=$E/v51_dagger3_ft_$x.games"; done
echo "== APP / VS: $RUN dagger3-ft vs $RUN dagger-ft, l4g dagger3-ft, v4.1-dagger3, v5.1 dagger3-ft (same seeds)" | tee -a "$OUT"
"$PYC" "$DIR/vs_score.py" $args | tee -a "$OUT"
echo "tome v42 d3 $RUN done"
