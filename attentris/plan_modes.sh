#!/bin/sh
# Ways to use an attentris-autoreg (N = 2) model's move 2, all with the same weights (MODEL), against
# dagger5b-ft: depth 1 / --commit 2 (move 2 played without a new decision) / depth 2 beam 4 (the usual
# policy beam: children re-encoded) / depth 2 beam 4 --autoreg-beam (move 2 from the step-2 pass over the
# root's encoding, children never encoded) / --plan-check (how often move X+1 = the move 2 predicted at X).
# No garbage (seed 5001) and normal / danger x 80/20, 50/50 (seed 33001), 64 games (commit 2 no garbage:
# 256, as depth 1); then clean timing: 8 games at a time, no garbage. Resumable (finished games kept).
#   MODEL=v51_d128_autoreg2_ft sh attentris/plan_modes.sh
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR"; . "$ATT/plat.sh"
PYC="$CPUPY"
MODEL=${MODEL:-v51_d128_autoreg2_ft}
BASE=v51_d128_dagger5b_ft
OUT="$DIR/results_plan.txt"
DE="$DIR/data/dagger3_eval"
m() { wpath "$1"; }
echo "== plan modes: $MODEL (depth 1 / commit 2 / depth 2 / depth 2 autoreg beam / plan check) $(date '+%Y-%m-%d %H:%M')" | tee -a "$OUT"
play() {   # model style tag games flags...
    f="$DE/$1_$2_$3.games"; md=$1; s=$2; t=$3; g=$4; shift 4
    [ -f "$f" ] && grep -q "^AtTenTris" "$f" && return
    rm -f "$f"
    echo "$md $s $t: $("$PYC" "$DIR/play.py" --model "$DIR/data/model_$md.pt" --style $s --games $g "$@" --out "$(m "$f")" \
        | grep "^AtTenTris" | sed 's/.*: APP/APP/' | cut -c1-260)" | tee -a "$OUT"
}
G20="--seed 33001 --garbage 10,3-4,cheese0.2,0.2"
G50="--seed 33001 --garbage 10,3-4,cheese0.5,0.2"
play $MODEL normal c2none 256 --commit 2
play $MODEL normal arnone 64 --depth 2 --beam 4 --autoreg-beam
play $MODEL normal pcnone 64 --plan-check
for s in normal danger; do
    play $MODEL $s c2g20 64 --commit 2 $G20
    play $MODEL $s c2g50 64 --commit 2 $G50
    play $MODEL $s d2g20 64 --depth 2 --beam 4 $G20
    play $MODEL $s d2g50 64 --depth 2 --beam 4 $G50
    play $MODEL $s arg20 64 --depth 2 --beam 4 --autoreg-beam $G20
    play $MODEL $s arg50 64 --depth 2 --beam 4 --autoreg-beam $G50
    play $MODEL $s pcg20 64 --plan-check $G20
    play $MODEL $s pcg50 64 --plan-check $G50
    play $BASE $s d2g20 64 --depth 2 --beam 4 $G20
    play $BASE $s d2g50 64 --depth 2 --beam 4 $G50
done
args=""
for md in $BASE $MODEL; do
    for x in normal_none normal_d2 normal_g20 normal_g50 danger_g20 danger_g50 normal_d2g20 normal_d2g50 danger_d2g20 danger_d2g50; do
        args="$args ${md}_$x=$(m "$DE")/${md}_$x.games"
    done
done
for x in normal_c2none normal_c2g20 normal_c2g50 danger_c2g20 danger_c2g50 normal_arnone normal_arg20 normal_arg50 danger_arg20 danger_arg50; do
    args="$args ${MODEL}_$x=$(m "$DE")/${MODEL}_$x.games"
done
echo "== APP / VS: $BASE vs $MODEL modes (d2 = depth 2 beam 4, c2 = commit 2, ar = depth 2 beam 4 autoreg)" | tee -a "$OUT"
"$PYC" "$DIR/vs_score.py" $args | tee -a "$OUT"
echo "== clean timing: 8 games at a time, no garbage" | tee -a "$OUT"
for mode in "d1:" "c2:--commit 2" "d2:--depth 2 --beam 4" "ar:--depth 2 --beam 4 --autoreg-beam"; do
    tag=${mode%%:*}; fl=${mode#*:}
    echo "$MODEL $tag timing: $("$PYC" "$DIR/play.py" --model "$DIR/data/model_$MODEL.pt" --games 8 --jobs 8 $fl \
        | grep "^AtTenTris" | sed 's/.*: APP/APP/' | cut -c1-200)" | tee -a "$OUT"
done
echo "plan modes done"
