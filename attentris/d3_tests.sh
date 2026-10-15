#!/bin/sh
# Depth 3 beam 8 play tests (the user re-enabled depth 3 for this, 2026-10-15) of one model, 64 games each,
# 8 games at a time (one physical P-core each: clean ms/move): no garbage (normal style), 80/20 and 50/50
# in normal and danger style; same seeds as the depth-1 tests. Resumable.
#   MODEL=v51_d128_dagger5b_ft sh d3_tests.sh > d3_tests.out 2>&1
DIR=$(cd "$(dirname "$0")" && pwd)
PYC="$DIR/../external/tetrisbattle/.venv/Scripts/python.exe"
DE="$DIR/data/dagger3_eval"
OUT="$DIR/results_d3.txt"
M=${MODEL:-v51_d128_dagger5b_ft}
m() { cygpath -m "$1"; }
play() {   # style tag flags...
    f="$DE/${M}_$1_$2.games"; s=$1; t=$2; shift 2
    [ -f "$f" ] && grep -q "^AtTenTris" "$f" && return
    rm -f "$f"
    echo "$M $s $t: $("$PYC" "$DIR/play.py" --model "$DIR/data/model_$M.pt" --style $s --games 64 --jobs 8 --depth 3 --beam 8 "$@" --out "$(m "$f")" \
        | grep "^AtTenTris" | sed 's/.*: APP/APP/' | cut -c1-220)" | tee -a "$OUT"
}
echo "== $M depth 3 beam 8 $(date '+%H:%M')" | tee -a "$OUT"
play normal d3_none
for s in normal danger; do
    play $s d3_g20 --seed 33001 --garbage 10,3-4,cheese0.2,0.2
    play $s d3_g50 --seed 33001 --garbage 10,3-4,cheese0.5,0.2
done
args=""
for x in normal_d3_none normal_d3_g20 normal_d3_g50 danger_d3_g20 danger_d3_g50; do args="$args ${M}_$x=$(m "$DE")/${M}_$x.games"; done
"$PYC" "$DIR/vs_score.py" $args | tee -a "$OUT"
echo "d3 tests done"
