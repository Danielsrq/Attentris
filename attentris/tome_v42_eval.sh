#!/bin/sh
# Tests for ToMe v4.2 (tome_v42.sh; PLAN_2026-10-13.md), CPU only; waits for each model's last epoch,
# so it can run alongside the GPU chain (the 4-layer model's tests overlap the 6-layer training).
# Per base model (tome_v42_l4, tome_v42_l6):
#   rule-token recovery (tome_iou.py, 2000 val positions; v4.1 and screen C printed alongside),
#   merge direction / split invariance (tome_orient.py --split own,token,col,random: a fixed model's
#   blobs should barely change with the split; v4.1's TSD stem moved 35% -> 49%), a blob viewer page,
#   64 games: depth 1 and depth 2 beam 4 without garbage, depth 1 at 80/20 and 50/50.
# Per DAgger 3 fine-tune: normal / danger / auto x no garbage / 80/20 / 50/50 at depth 1, 64 games.
# Then one APP / VS table with v4.1 and v4.1-dagger3 on the same seeds. Resumable.
#   sh tome_v42_eval.sh > tome_v42_eval.out 2>&1
DIR=$(cd "$(dirname "$0")" && pwd)
PYG="$DIR/.venv/Scripts/python.exe"
PYC="$DIR/../external/tetrisbattle/.venv/Scripts/python.exe"
D="$DIR/data/tome_v42"
WD=$(cygpath -m "$D")
OUT="$DIR/results_tome_v42.txt"
mkdir "$DIR/.tome_v42_eval.lock" 2>/dev/null || { echo "tome_v42_eval.sh is already running (remove $DIR/.tome_v42_eval.lock if not)"; exit 1; }
trap 'rmdir "$DIR/.tome_v42_eval.lock"' EXIT
mkdir -p "$D"
await() {   # log epoch name: wait for the training run, give up if the GPU chain reported a failure
    until grep -q "^epoch $2:" "$1" 2>/dev/null; do
        grep -q "$3: training did not finish\|test_tome_split.py failed" "$OUT" 2>/dev/null && { echo "$3: no model, stopping"; exit 1; }
        sleep 60
    done
    sleep 30      # the checkpoint is written right after the epoch line
}
play() {   # model style tag flags...
    m=$1; s=$2; tag=$3; shift 3
    f="$D/${m}_${s}_$tag.games"
    [ -f "$f" ] && grep -q "^AtTenTris" "$f" && return
    rm -f "$f"
    echo "$m $s $tag: $("$PYC" "$DIR/play.py" --model "$DIR/data/model_$m.pt" --games 64 --style $s "$@" --out "$(cygpath -m "$f")" \
        | grep "^AtTenTris" | sed 's/.*: APP/APP/' | cut -c1-200)" | tee -a "$OUT"
}
G20="--depth 1 --seed 33001 --garbage 10,3-4,cheese0.2,0.2"
G50="--depth 1 --seed 33001 --garbage 10,3-4,cheese0.5,0.2"
for n in l4 l6; do
    m=tome_v42_$n; M="$DIR/data/model_$m.pt"
    await "$DIR/train_$m.out" 8 $m
    [ -f "$D/${m}_iou.txt" ] || "$PYG" "$DIR/tome_iou.py" --model "$M" --n 2000 | grep -v "positions$" > "$D/${m}_iou.txt"
    echo "== $m: rule-token recovery" | tee -a "$OUT"; cat "$D/${m}_iou.txt" | tee -a "$OUT"
    [ $n = l4 ] && ref=tome_sched/tome_sC_iou.txt || ref=tome_v41/tome_v41_iou.txt
    echo "== baseline (token split): $ref" | tee -a "$OUT"; cat "$DIR/data/$ref" | tee -a "$OUT"
    [ -f "$D/${m}_orient.txt" ] || "$PYG" "$DIR/tome_orient.py" --model "$M" --n 2000 --split own,token,col,random --out "$D/${m}_orient.txt" > /dev/null
    echo "== $m: merge direction (own split = hash; token / col / random swapped in at inference)" | tee -a "$OUT"
    grep -E "^=====|picked V|^   (tsd|well|pit1|pit2|fit_t|play|\(all\))" "$D/${m}_orient.txt" | tee -a "$OUT"
    [ -f "$DIR/${m}_viz.html" ] || "$PYG" "$DIR/tome_viz.py" --model "$M" --random 24 --overrides "none;split=token" --val-eval \
        --out "$DIR/${m}_viz.html" --label "AtTenTris-ToMe v4.2 ($n) · hash split · 2 streams · per-position merge budget · 8 epochs" | tail -1
    play $m normal d1 --depth 1
    play $m normal d2 --depth 2 --beam 4
    play $m normal g20 $G20
    play $m normal g50 $G50
done
for n in l4 l6; do
    m=tome_v42_${n}_dagger3_ft
    await "$DIR/train_$m.out" 3 $m
    for s in normal danger auto; do
        play $m $s d1 --depth 1
        play $m $s g20 $G20
        play $m $s g50 $G50
    done
done
args=""
V=$(cygpath -m "$DIR/data/tome_v41")
for m in tome_v42_l4 tome_v42_l6; do for t in d1 d2 g20 g50; do args="$args ${m}_$t=$WD/${m}_normal_$t.games"; done; done
for t in d1 d2 g20 g50; do args="$args tome_v41_$t=$V/tome_v41_normal_$t.games"; done
for m in tome_v42_l4_dagger3_ft tome_v42_l6_dagger3_ft; do for s in normal danger auto; do for t in d1 g20 g50; do
    args="$args ${m}_${s}_$t=$WD/${m}_${s}_$t.games"; done; done; done
for s in normal danger auto; do for t in d1 g20 g50; do args="$args tome_v41_dagger3_ft_${s}_$t=$V/tome_v41_dagger3_ft_${s}_$t.games"; done; done
echo "== APP / VS: v4.2 (hash split) vs v4.1 (token split), same seeds" | tee -a "$OUT"
"$PYC" "$DIR/vs_score.py" $args | tee -a "$OUT"
echo "tome v42 eval done"
