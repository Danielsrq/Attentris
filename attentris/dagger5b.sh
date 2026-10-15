#!/bin/sh
# DAgger round 5, extension (dagger5b): more DEATHS, same selection (60-move death window): the student
# (v51_d128_dagger3_ft, as round 5) plays 200 games in danger style and 200 in normal style at 50/50 (the
# settings where it dies most: 73% / 85% in round 5), fresh seeds 43001+ / 44001+. Same selection, labels
# and routing as dagger5.sh (CEM-survival -> danger for all; normal-style games: hot -> CEM-survival as
# normal, calm -> normal.txt). Round 5 + extension packed together as dagger5all; a fresh fine-tune from
# v51_d128_dagger3_ft (same recipe as dagger5-ft) -> v51_d128_dagger5b_ft; tests vs dagger3-ft / dagger5-ft.
# Resumable per step.   sh dagger5b.sh > dagger5b.out 2>&1
DIR=$(cd "$(dirname "$0")" && pwd)
PYG="$DIR/.venv/Scripts/python.exe"
PYC="$DIR/../external/tetrisbattle/.venv/Scripts/python.exe"
BOT="$DIR/../bot/target/release/bot.exe"
V51="$DIR/data/v51"
D="$DIR/data/dagger5b"
DA="$DIR/data/dagger5all"
SURV="$DIR/../bot/results/survive/survive.txt"
NORM="$DIR/../bot/weights/normal.txt"
OUT="$DIR/results_dagger5.txt"
STUDENT=v51_d128_dagger3_ft
m() { cygpath -m "$1"; }
mkdir -p "$D/games" "$DA"
echo "== dagger5b (more deaths) $(date '+%H:%M')" | tee -a "$OUT"
for g in "danger:g50:43001:10,3-4,cheese0.5,0.2" "normal:g50:44001:10,3-4,cheese0.5,0.2"; do
    st=$(echo $g | cut -d: -f1); tag=$(echo $g | cut -d: -f2); seed=$(echo $g | cut -d: -f3); garb=$(echo $g | cut -d: -f4)
    f="$D/games/${st}_$tag"
    [ -f "$f.games" ] && grep -q "^AtTenTris" "$f.games" && continue
    rm -f "$f.games" "$f.jsonl"
    echo "collect $st $tag: $("$PYC" "$DIR/play.py" --model "$DIR/data/model_$STUDENT.pt" --games 200 --depth 1 --style $st --seed $seed \
        --garbage $garb --out "$(m "$f.games")" --dump "$(m "$f.jsonl")" | grep "^AtTenTris" | sed 's/.*: APP/APP/' | cut -c1-140)" | tee -a "$OUT"
done
[ -f "$D/selected.jsonl" ] || "$PYC" "$DIR/dagger_select.py" --games "$(m "$D")/games/*.games" --states "$(m "$D")/games/*.jsonl" \
    --out "$(m "$D/selected.jsonl")" --random 0 | tee -a "$OUT"
if [ ! -f "$D/labelled_surv.jsonl" ]; then
    "$BOT" label --mode normal --weights "$(m "$SURV")" --label danger --threads 16 --all-roots 1 --beam 128 \
        --out "$(m "$D/labelled_surv.jsonl.part")" < "$D/selected.jsonl" > "$D/label_surv.log" 2>&1 && mv "$D/labelled_surv.jsonl.part" "$D/labelled_surv.jsonl"
fi
[ -f "$D/labelled_surv.jsonl" ] || { echo "labelling failed (see $D/label_surv.log)" | tee -a "$OUT"; exit 1; }
if [ ! -f "$D/labelled.jsonl" ]; then
    "$PYC" "$DIR/dagger5_route.py" --labelled "$(m "$D/labelled_surv.jsonl")" --selected "$(m "$D/selected.jsonl")" \
        --normal-seeds 44001-44200 --out "$(m "$D/routed.jsonl")" --calm-out "$(m "$D/calm_states.jsonl")" | tee -a "$OUT"
    "$BOT" label --mode normal --weights "$(m "$NORM")" --label normal --threads 16 --all-roots 1 --beam 128 \
        --out "$(m "$D/labelled_calm.jsonl")" < "$D/calm_states.jsonl" > "$D/label_calm.log" 2>&1 || { echo "calm labelling failed" | tee -a "$OUT"; exit 1; }
    cat "$D/routed.jsonl" "$D/labelled_calm.jsonl" > "$D/labelled.jsonl.part" && mv "$D/labelled.jsonl.part" "$D/labelled.jsonl"
fi
echo "dagger5b records: $(wc -l < "$D/labelled.jsonl") (danger $(grep -c '"teacher": *"danger"' "$D/labelled.jsonl"), normal $(grep -c '"teacher": *"normal"' "$D/labelled.jsonl"))" | tee -a "$OUT"
[ -f "$DA/labelled.jsonl" ] || cat "$DIR/data/dagger5/labelled.jsonl" "$D/labelled.jsonl" > "$DA/labelled.jsonl"
[ -f "$DA/raw_tr/dagger5all_s0.jsonl" ] || "$PYC" "$DIR/split_by_seed.py" --src "$(m "$DA/labelled.jsonl")" --tr "$(m "$DA/raw_tr")" --val "$(m "$DA/raw_val")" --name dagger5all
[ -f "$V51/dagger5all.npz" ] || "$PYC" "$DIR/pack.py" --only "$(m "$DA/raw_tr")" --name dagger5all --jobs 16 --out "$(m "$V51")" | tail -1
[ -f "$V51/val_dagger5all.npz" ] || "$PYC" "$DIR/pack.py" --only "$(m "$DA/raw_val")" --name val_dagger5all --jobs 8 --out "$(m "$V51")" | tail -1
F=$("$PYC" -c "
import numpy as np
d = '$(m "$V51")/'
new = sum(len(np.load(d + f)['chosen']) for f in ('danger.npz', 'dagger4.npz', 'dagger5all.npz'))
rest = sum(len(np.load(d + f)['chosen']) for f in ('train.npz', 'dagger1.npz', 'train1m.npz', 'dagger2.npz'))
print('%.4f' % (7 / 3 * new / rest))")
echo "mix: existing files at $F each per epoch, all of danger + dagger4 + dagger5all" | tee -a "$OUT"
LOG="$DIR/train_v51_d128_dagger5b_ft.out"
if ! grep -q "^epoch 3: val2" "$LOG" 2>/dev/null; then
    R=""; [ -f "$DIR/data/model_v51_d128_dagger5b_ft.pt" ] && R="--resume"
    "$PYG" "$DIR/train.py" --data "$V51" --batch 512 --device cuda --w-dense 0 --w-q1 0 --w-qs 1 --seed 0 --d 128 --micro 2 --val2 val_dagger5all.npz \
        --train train.npz@$F,dagger1.npz@$F,train1m.npz@$F,dagger2.npz@$F,danger.npz,dagger4.npz,dagger5all.npz --epochs 3 --lr 3e-4 --n-teachers 3 \
        --init "$DIR/data/model_$STUDENT.pt" --out "$DIR/data/model_v51_d128_dagger5b_ft.pt" $R >> "$LOG" 2>&1
fi
grep -q "^epoch 3: val2" "$LOG" || { echo "training did not finish (see $LOG)" | tee -a "$OUT"; exit 1; }
echo "== v51_d128_dagger5b_ft" | tee -a "$OUT"
grep "^epoch 3:" "$LOG" | sed 's/argmax Qs top1 [0-9.]* \/ [0-9.]*, Q1.*Qs [0-9.]* |/|/' | tee -a "$OUT"
DE="$DIR/data/dagger3_eval"
play() {   # model style tag games flags...
    f="$DE/$1_$2_$3.games"; md=$1; s=$2; t=$3; g=$4; shift 4
    [ -f "$f" ] && grep -q "^AtTenTris" "$f" && return
    rm -f "$f"
    echo "$md $s $t: $("$PYC" "$DIR/play.py" --model "$DIR/data/model_$md.pt" --style $s --games $g "$@" --out "$(m "$f")" \
        | grep "^AtTenTris" | sed 's/.*: APP/APP/' | cut -c1-200)" | tee -a "$OUT"
}
n=v51_d128_dagger5b_ft
play $n normal none 256 --depth 1
play $n normal d2 64 --depth 2 --beam 4
for s in normal danger; do
    play $n $s g20 64 --depth 1 --seed 33001 --garbage 10,3-4,cheese0.2,0.2
    play $n $s g50 64 --depth 1 --seed 33001 --garbage 10,3-4,cheese0.5,0.2
done
args=""
for md in v51_d128_dagger3_ft v51_d128_dagger5_ft v51_d128_dagger5b_ft; do
    for x in normal_none normal_d2 normal_g20 normal_g50 danger_g20 danger_g50; do args="$args ${md}_$x=$(m "$DE")/${md}_$x.games"; done
done
echo "== APP / VS: d=128 dagger3-ft vs dagger5-ft (round 5, 50k) vs dagger5b-ft (round 5 + extension, more deaths)" | tee -a "$OUT"
"$PYC" "$DIR/vs_score.py" $args | tee -a "$OUT"
echo "dagger5b done"
