#!/bin/sh
# Tests for ToMe v4.2 (tome_v42.sh; PLAN_2026-10-13.md), CPU only; waits for each model's last epoch, so
# it runs alongside the GPU chain (l4h's tests overlap l4g's training). Per base model:
#   rule-token recovery (tome_iou.py, 2000 val positions; screen C = 4 layers, token split, 4 epochs, and
#   v4.1 printed alongside), merge direction and split invariance (tome_orient.py with the token / hash /
#   greedy merge swapped in at inference, pure kept: a model that fixed the bug should barely change),
#   a blob viewer page (as trained / token split / the other method), 64 games: depth 1 and depth 2 beam 4
#   without garbage, depth 1 at 80/20 and 50/50 (normal style).
# Per DAgger 2 fine-tune: the same 4 settings. Then one APP / VS table with screen C, v4.1 and v5.1
# dagger-ft on the same seeds. RUNS as in tome_v42.sh. Resumable.
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
RUNS=${RUNS:-"l4h l4g"}
await() {   # log epoch name: wait for the training run, give up if the GPU chain reported a failure
    until grep -q "^epoch $2:" "$1" 2>/dev/null; do
        grep -q "$3: training did not finish\|test_tome_split.py failed" "$OUT" 2>/dev/null && { echo "$3: no model, stopping"; exit 1; }
        sleep 60
    done
    sleep 30      # the checkpoint is written right after the epoch line
}
play() {   # model tag flags...
    m=$1; tag=$2; shift 2
    f="$D/${m}_normal_$tag.games"
    [ -f "$f" ] && grep -q "^AtTenTris" "$f" && return
    rm -f "$f"
    echo "$m normal $tag: $("$PYC" "$DIR/play.py" --model "$DIR/data/model_$m.pt" --games 64 --style normal "$@" --out "$(cygpath -m "$f")" \
        | grep "^AtTenTris" | sed 's/.*: APP/APP/' | cut -c1-200)" | tee -a "$OUT"
}
tests() {   # model
    play $1 d1 --depth 1
    play $1 d2 --depth 2 --beam 4
    play $1 g20 --depth 1 --seed 33001 --garbage 10,3-4,cheese0.2,0.2
    play $1 g50 --depth 1 --seed 33001 --garbage 10,3-4,cheese0.5,0.2
}
for n in $RUNS; do
    m=tome_v42_$n; M="$DIR/data/model_$m.pt"
    case $n in *h) own=hash; other=greedy;; *) own=greedy; other=hash;; esac
    await "$DIR/train_$m.out" 8 $m
    [ -f "$D/${m}_iou.txt" ] || "$PYG" "$DIR/tome_iou.py" --model "$M" --n 2000 | grep -v "positions$" > "$D/${m}_iou.txt"
    echo "== $m ($own split, pure): rule-token recovery" | tee -a "$OUT"; cat "$D/${m}_iou.txt" | tee -a "$OUT"
    case $n in l4*) ref=tome_sched/tome_sC_iou.txt;; *) ref=tome_v41/tome_v41_iou.txt;; esac
    echo "== baseline (token split): $ref" | tee -a "$OUT"; cat "$DIR/data/$ref" | tee -a "$OUT"
    [ -f "$D/${m}_orient.txt" ] || "$PYG" "$DIR/tome_orient.py" --model "$M" --n 2000 \
        --variants "none;split=token;split=hash;split=greedy" --out "$D/${m}_orient.txt" > /dev/null
    echo "== $m: merge direction (as trained = $own; token / hash / greedy swapped in at inference, pure kept)" | tee -a "$OUT"
    grep -E "^=====|picked V|^   (tsd|well|pit1|pit2|fit_t|play|\(all\))|^   (no T|no \||\|-)" "$D/${m}_orient.txt" | tee -a "$OUT"
    [ -f "$DIR/${m}_viz.html" ] || "$PYG" "$DIR/tome_viz.py" --model "$M" --random 24 --overrides "none;split=token;split=$other" --val-eval \
        --title "ToMe v4.2 $n" --out "$DIR/${m}_viz.html" \
        --label "AtTenTris-ToMe v4.2 $n · 2 streams · $own merging, pure · per-position merge budget · 8 epochs" | tail -4 | tee -a "$OUT"
    tests $m
done
for n in $RUNS; do
    m=tome_v42_${n}_dagger_ft
    await "$DIR/train_$m.out" 3 $m
    tests $m
done
args=""
for n in $RUNS; do for m in tome_v42_$n tome_v42_${n}_dagger_ft; do for t in d1 d2 g20 g50; do args="$args ${m}_$t=$WD/${m}_normal_$t.games"; done; done; done
S=$(cygpath -m "$DIR/data/tome_sched"); V=$(cygpath -m "$DIR/data/tome_v41"); E=$(cygpath -m "$DIR/data/dagger3_eval")
for t in d1 d2 g20 g50; do args="$args screenC_token_4ep_$t=$S/tome_sC_$t.games tome_v41_$t=$V/tome_v41_normal_$t.games"; done
args="$args v51_dagger_ft_d1=$E/v51_dagger_ft_normal_none64.games v51_dagger_ft_d2=$E/v51_dagger_ft_normal_d264.games"
args="$args v51_dagger_ft_g20=$E/v51_dagger_ft_normal_g2064.games v51_dagger_ft_g50=$E/v51_dagger_ft_normal_g5064.games"
echo "== APP / VS: v4.2 (hash / greedy, pure) vs screen C (token split, 4 epochs), v4.1 and v5.1 dagger-ft, same seeds" | tee -a "$OUT"
"$PYC" "$DIR/vs_score.py" $args | tee -a "$OUT"
echo "tome v42 eval done"
