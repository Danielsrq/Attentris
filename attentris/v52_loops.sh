#!/bin/sh
# Loop-count test for the weight-tied v5.2 (cross block applied twice in training): play the d128 loop
# dagger5b-ft with 1, 3 and 4 passes (play.py --cross-loops; 2 = as trained, from v52_d128.sh), depth 1,
# no garbage and normal / danger x 80/20, 50/50, 64 games each, same seeds. If 3-4 passes beat 2, the block
# learned an iterative refinement; if not, it learned "exactly two passes". Resumable.
#   sh attentris/v52_loops.sh > attentris/v52_loops.out 2>&1
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR"; . "$ATT/plat.sh"
PYC="$CPUPY"
OUT="$DIR/results_v52_d128.txt"
DE="$DIR/data/dagger3_eval"
N=v52_d128_loop_dagger5b_ft
m() { wpath "$1"; }
[ -f "$DIR/data/model_$N.pt" ] || { echo "no $N (v52_d128.sh)"; exit 1; }
echo "== $N: cross passes at play time (trained with 2) $(date '+%H:%M')" | tee -a "$OUT"
play() {   # tag games loops flags...
    f="$DE/${N}_$1_x$3.games"; t=$1; g=$2; L=$3; shift 3
    [ -f "$f" ] && grep -q "^AtTenTris" "$f" && return
    rm -f "$f"
    echo "$N loops $L $t: $("$PYC" "$DIR/play.py" --model "$DIR/data/model_$N.pt" --games $g --cross-loops $L "$@" --out "$(m "$f")" \
        | grep "^AtTenTris" | sed 's/.*: APP/APP/' | cut -c1-160)" | tee -a "$OUT"
}
for L in 1 3 4; do
    play normal_none 64 $L --style normal --depth 1
    for s in normal danger; do
        play ${s}_g20 64 $L --style $s --depth 1 --seed 33001 --garbage 10,3-4,cheese0.2,0.2
        play ${s}_g50 64 $L --style $s --depth 1 --seed 33001 --garbage 10,3-4,cheese0.5,0.2
    done
done
args=""
for x in normal_none normal_g20 normal_g50 danger_g20 danger_g50; do
    args="$args loops2_$x=$(m "$DE")/${N}_$x.games"
    for L in 1 3 4; do args="$args loops${L}_$x=$(m "$DE")/${N}_${x}_x$L.games"; done
done
echo "== APP / VS: $N with 1 / 2 (trained) / 3 / 4 cross passes" | tee -a "$OUT"
"$PYC" "$DIR/vs_score.py" $args | tee -a "$OUT"
echo "v52 loops done"
