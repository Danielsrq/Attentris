#!/bin/sh
# ToMe v4.1 (PLAN_2026-10-12.md step 2) = the screen's winner B: 6 layers, per-position merge budget
# (20 + max(0, (n0 - 36) / 4) blobs per stream, first merge 10%, then equal ratios), 2 streams, the
# ToMe v4 recipe for 8 epochs from scratch. Then its DAgger 3 fine-tune (as ToMe-dagger3: from the base,
# 70/30 with danger.npz, 3 epochs, lr 3e-4, 3 style rows). Tests at depth 1 (64 games): base and
# v4.1-dagger3 in style normal / danger / auto x no garbage / 80/20 / 50/50, plus depth 2 beam 4 for the
# base; rule-token recovery and a blob viewer page for the base. Resumable; one copy at a time.
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR"; . "$ATT/plat.sh"
PYG="$DIR/.venv/$VPY"
PYC="$CPUPY"
V51="$DIR/data/v51"
D="$DIR/data/tome_v41"
WD=$(wpath "$D")
OUT="$DIR/results_tome_v41.txt"
mkdir "$DIR/.tome_v41.lock" 2>/dev/null || { echo "tome_v41.sh is already running (remove $DIR/.tome_v41.lock if not)"; exit 1; }
trap 'rmdir "$DIR/.tome_v41.lock"' EXIT
mkdir -p "$D"
ARCH="--arch tome --tome-pretok --tome-flat --tome-streams 2 --micro 8 --layers 6 --merge 0,1,1,1,1,1 --tome-budget 20,36,0.25,0.10 --bucket 50"
train() {   # name epochs log args...
    name=$1; ep=$2; log=$3; shift 3
    M="$DIR/data/model_$name.pt"
    if ! grep -q "^epoch $ep:" "$log" 2>/dev/null; then
        R=""; [ -f "$M" ] && R="--resume"
        "$PYG" "$DIR/train.py" $ARCH --data "$V51" --batch 512 --device cuda --w-dense 0 --w-q1 0 --w-qs 1 --seed 0 --out "$M" $R "$@" >> "$log" 2>&1
    fi
    grep -q "^epoch $ep:" "$log" || { echo "$name: training did not finish (see $log)" | tee -a "$OUT"; exit 1; }
    echo "== $name" | tee -a "$OUT"
    grep "^epoch $ep:" "$log" | sed 's/argmax Qs.*Qs [0-9.]* |/|/' | tee -a "$OUT"
}
train tome_v41 8 "$DIR/train_tome_v41.out" --train train.npz,dagger1.npz,train1m.npz --epochs 8 --lr 1.5e-3
M="$DIR/data/model_tome_v41.pt"
[ -f "$D/tome_v41_iou.txt" ] || "$PYG" "$DIR/tome_iou.py" --model "$M" --n 2000 | grep -v "positions$" > "$D/tome_v41_iou.txt"
cat "$D/tome_v41_iou.txt" | tee -a "$OUT"
[ -f "$DIR/tome_v41_viz.html" ] || "$PYG" "$DIR/tome_viz.py" --model "$M" --random 24 --out "$DIR/tome_v41_viz.html" --label "AtTenTris-ToMe v4.1 · 6 layers · 2 streams · per-position merge budget · 8 epochs" | tail -1
F=0.1851      # 70/30 with the 86,195 danger positions over train + dagger1 + train1m (as tome_dagger3.sh)
train tome_v41_dagger3_ft 3 "$DIR/train_tome_v41_dagger3_ft.out" --train train.npz@$F,dagger1.npz@$F,train1m.npz@$F,danger.npz \
    --epochs 3 --lr 3e-4 --n-teachers 3 --init "$M" --val2 val_danger.npz
play() {   # model style tag flags...
    m=$1; s=$2; tag=$3; shift 3
    f="$D/${m}_${s}_$tag.games"
    [ -f "$f" ] && grep -q "^AtTenTris" "$f" && return
    rm -f "$f"
    echo "$m $s $tag: $("$PYC" "$DIR/play.py" --model "$DIR/data/model_$m.pt" --games 64 --style $s "$@" --out "$(wpath "$f")" \
        | grep "^AtTenTris" | sed 's/.*: APP/APP/' | cut -c1-200)" | tee -a "$OUT"
}
play tome_v41 normal d1 --depth 1
play tome_v41 normal d2 --depth 2 --beam 4
play tome_v41 normal g20 --depth 1 --seed 33001 --garbage 10,3-4,cheese0.2,0.2
play tome_v41 normal g50 --depth 1 --seed 33001 --garbage 10,3-4,cheese0.5,0.2
for s in normal danger auto; do
    play tome_v41_dagger3_ft $s d1 --depth 1
    play tome_v41_dagger3_ft $s g20 --depth 1 --seed 33001 --garbage 10,3-4,cheese0.2,0.2
    play tome_v41_dagger3_ft $s g50 --depth 1 --seed 33001 --garbage 10,3-4,cheese0.5,0.2
done
args=""
for x in tome_v41_normal_d1 tome_v41_normal_d2 tome_v41_normal_g20 tome_v41_normal_g50; do args="$args $x=$WD/$x.games"; done
for s in normal danger auto; do for t in d1 g20 g50; do args="$args tome_v41_dagger3_ft_${s}_$t=$WD/tome_v41_dagger3_ft_${s}_$t.games"; done; done
E=$(wpath "$DIR/data/dagger3_eval")
for s in normal danger auto; do for t in none g20 g50; do args="$args tome_v4_dagger3_ft_${s}_$t=$E/tome_v4_dagger3_ft_${s}_$t.games"; done; done
"$PYC" "$DIR/vs_score.py" $args | tee -a "$OUT"
echo "tome v41 done"
