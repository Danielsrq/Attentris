#!/bin/sh
# ToMe v4.1 with layer 6 merging 30% at inference (play.py / tome_iou.py --merge-override 6:0.30; no
# retraining): rule-token recovery; 64-game tests of the base (depth 1 and depth 2 beam 4, no garbage;
# depth 1 at 80/20 and 50/50) and of v4.1-dagger3 (normal, no garbage; danger at 80/20 and 50/50).
# Waits for the DAgger 4 labelling (dagger4_label.out) to free the CPU; then marks the day's plan done
# (attentris/.plan_done), which starts the queued depth-1 CEM tunes.
DIR=$(cd "$(dirname "$0")" && pwd)
PYG="$DIR/.venv/Scripts/python.exe"
PYC="$DIR/../external/tetrisbattle/.venv/Scripts/python.exe"
D="$DIR/data/tome_v41"
WD=$(cygpath -m "$D")
OUT="$DIR/results_tome_v41_ov.txt"
OV="--merge-override 6:0.30"
until grep -q "dagger4 label done" "$DIR/dagger4_label.out" 2>/dev/null; do sleep 30; done
[ -f "$D/tome_v41_ov_iou.txt" ] || "$PYG" "$DIR/tome_iou.py" --model "$DIR/data/model_tome_v41.pt" --n 2000 $OV | grep -v "positions$" > "$D/tome_v41_ov_iou.txt"
echo "== ToMe v4.1, layer 6 merges 30% (inference): rule-token recovery" | tee -a "$OUT"
cat "$D/tome_v41_ov_iou.txt" | tee -a "$OUT"
play() {   # model style tag flags...
    m=$1; s=$2; tag=$3; shift 3
    f="$D/${m}_ov_${s}_$tag.games"
    [ -f "$f" ] && grep -q "^AtTenTris" "$f" && return
    rm -f "$f"
    echo "$m (layer 6 at 30%) $s $tag: $("$PYC" "$DIR/play.py" --model "$DIR/data/model_$m.pt" --games 64 --style $s $OV "$@" --out "$(cygpath -m "$f")" \
        | grep "^AtTenTris" | sed 's/.*: APP/APP/' | cut -c1-200)" | tee -a "$OUT"
}
play tome_v41 normal d1 --depth 1
play tome_v41 normal d2 --depth 2 --beam 4
play tome_v41 normal g20 --depth 1 --seed 33001 --garbage 10,3-4,cheese0.2,0.2
play tome_v41 normal g50 --depth 1 --seed 33001 --garbage 10,3-4,cheese0.5,0.2
play tome_v41_dagger3_ft normal d1 --depth 1
play tome_v41_dagger3_ft danger g20 --depth 1 --seed 33001 --garbage 10,3-4,cheese0.2,0.2
play tome_v41_dagger3_ft danger g50 --depth 1 --seed 33001 --garbage 10,3-4,cheese0.5,0.2
args=""
for x in tome_v41_normal_d1 tome_v41_normal_d2 tome_v41_normal_g20 tome_v41_normal_g50 tome_v41_dagger3_ft_normal_d1 tome_v41_dagger3_ft_danger_g20 tome_v41_dagger3_ft_danger_g50; do
    m=${x%_*_*}; rest=${x#${m}_}
    args="$args $x=$WD/$x.games ${m}_ov_$rest=$WD/${m}_ov_$rest.games"
done
"$PYC" "$DIR/vs_score.py" $args | tee -a "$OUT"
echo "tome v41 override done"
touch "$DIR/.plan_done"
