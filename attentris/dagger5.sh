#!/bin/sh
# DAgger round 5 from the best model (v5.1 d=128 dagger3-ft), with danger mixed into the normal style.
#   1. collect: depth 1, 100 games each: danger style at 80/20 (seeds 39001+) and 50/50 (40001+), normal style
#      at 80/20 (41001+) and 50/50 (42001+); every state dumped (fresh seeds; rounds 2-4 used 30001-37100)
#   2. select the trouble positions (dagger_select.py, as round 4: death / danger / holes rising / unsure)
#   3. label every selected position with CEM-survival (beam 128, all moves) as the danger style
#   4. route (dagger5_route.py): positions of the normal-style games also go to the normal style --
#      hot boards (stack >= 12 or >= 6 holes) with CEM-survival's label, calm ones labelled by normal.txt
#   5. pack DAgger 4 (unused so far) and DAgger 5 (every 10th seed held out as val_*)
#   6. fine-tune v51_d128_dagger3_ft -> v51_d128_dagger5_ft: 3 epochs, 70/30 (existing files at F : all of
#      danger + dagger4 + dagger5), lr 3e-4, val2 val_dagger5
#   7. tests vs v51_d128_dagger3_ft (same seeds): none 256, d2 64, 80/20 + 50/50 in normal and danger style
# Resumable per step.   sh dagger5.sh > dagger5.out 2>&1
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR"; . "$ATT/plat.sh"
PYG="$DIR/.venv/$VPY"
PYC="$CPUPY"
BOT="$DIR/../bot/target/release/bot$EXE"
V51="$DIR/data/v51"
D="$DIR/data/dagger5"
D4="$DIR/data/dagger4"
SURV="$DIR/../bot/results/survive/survive.txt"
NORM="$DIR/../bot/weights/normal.txt"
OUT="$DIR/results_dagger5.txt"
STUDENT=v51_d128_dagger3_ft
m() { wpath "$1"; }
mkdir -p "$D/games"
echo "== dagger5 $(date '+%H:%M')" | tee -a "$OUT"
# 1. collect
for g in "danger:g20:39001:10,3-4,cheese0.2,0.2" "danger:g50:40001:10,3-4,cheese0.5,0.2" \
         "normal:g20:41001:10,3-4,cheese0.2,0.2" "normal:g50:42001:10,3-4,cheese0.5,0.2"; do
    st=$(echo $g | cut -d: -f1); tag=$(echo $g | cut -d: -f2); seed=$(echo $g | cut -d: -f3); garb=$(echo $g | cut -d: -f4)
    f="$D/games/${st}_$tag"
    [ -f "$f.games" ] && grep -q "^AtTenTris" "$f.games" && continue
    rm -f "$f.games" "$f.jsonl"
    echo "collect $st $tag: $("$PYC" "$DIR/play.py" --model "$DIR/data/model_$STUDENT.pt" --games 100 --depth 1 --style $st --seed $seed \
        --garbage $garb --out "$(m "$f.games")" --dump "$(m "$f.jsonl")" | grep "^AtTenTris" | sed 's/.*: APP/APP/' | cut -c1-140)" | tee -a "$OUT"
done
# 2. select
[ -f "$D/selected.jsonl" ] || "$PYC" "$DIR/dagger_select.py" --games "$(m "$D")/games/*.games" --states "$(m "$D")/games/*.jsonl" \
    --out "$(m "$D/selected.jsonl")" --random 0 | tee -a "$OUT"
# 3. CEM-survival labels (danger)
if [ ! -f "$D/labelled_surv.jsonl" ]; then
    "$BOT" label --mode normal --weights "$(m "$SURV")" --label danger --threads 16 --all-roots 1 --beam 128 \
        --out "$(m "$D/labelled_surv.jsonl.part")" < "$D/selected.jsonl" > "$D/label_surv.log" 2>&1 && mv "$D/labelled_surv.jsonl.part" "$D/labelled_surv.jsonl"
fi
[ -f "$D/labelled_surv.jsonl" ] || { echo "labelling failed (see $D/label_surv.log)" | tee -a "$OUT"; exit 1; }
# 4. route + normal.txt labels for the calm normal-style positions
if [ ! -f "$D/labelled.jsonl" ]; then
    "$PYC" "$DIR/dagger5_route.py" --labelled "$(m "$D/labelled_surv.jsonl")" --selected "$(m "$D/selected.jsonl")" \
        --normal-seeds 41001-41100,42001-42100 --out "$(m "$D/routed.jsonl")" --calm-out "$(m "$D/calm_states.jsonl")" | tee -a "$OUT"
    "$BOT" label --mode normal --weights "$(m "$NORM")" --label normal --threads 16 --all-roots 1 --beam 128 \
        --out "$(m "$D/labelled_calm.jsonl")" < "$D/calm_states.jsonl" > "$D/label_calm.log" 2>&1 || { echo "calm labelling failed" | tee -a "$OUT"; exit 1; }
    cat "$D/routed.jsonl" "$D/labelled_calm.jsonl" > "$D/labelled.jsonl.part" && mv "$D/labelled.jsonl.part" "$D/labelled.jsonl"
fi
echo "dagger5 records: $(wc -l < "$D/labelled.jsonl") (danger $(grep -c '"teacher": *"danger"' "$D/labelled.jsonl"), normal $(grep -c '"teacher": *"normal"' "$D/labelled.jsonl"))" | tee -a "$OUT"
# 5. pack DAgger 4 and DAgger 5
for r in dagger4:$D4 dagger5:$D; do
    name=${r%%:*}; src=${r#*:}
    [ -f "$src/raw_tr/${name}_s0.jsonl" ] || "$PYC" "$DIR/split_by_seed.py" --src "$(m "$src/labelled.jsonl")" --tr "$(m "$src/raw_tr")" \
        --val "$(m "$src/raw_val")" --name $name
    [ -f "$V51/$name.npz" ] || "$PYC" "$DIR/pack.py" --only "$(m "$src/raw_tr")" --name $name --jobs 16 --out "$(m "$V51")" | tail -1
    [ -f "$V51/val_$name.npz" ] || "$PYC" "$DIR/pack.py" --only "$(m "$src/raw_val")" --name val_$name --jobs 8 --out "$(m "$V51")" | tail -1
done
# 6. fine-tune
F=$("$PYC" -c "
import numpy as np
d = '$(m "$V51")/'
new = sum(len(np.load(d + f)['chosen']) for f in ('danger.npz', 'dagger4.npz', 'dagger5.npz'))
rest = sum(len(np.load(d + f)['chosen']) for f in ('train.npz', 'dagger1.npz', 'train1m.npz', 'dagger2.npz'))
print('%.4f' % (7 / 3 * new / rest))")
echo "mix: existing files at $F each per epoch, all of danger + dagger4 + dagger5" | tee -a "$OUT"
LOG="$DIR/train_v51_d128_dagger5_ft.out"
if ! grep -q "^epoch 3: val2" "$LOG" 2>/dev/null; then
    R=""; [ -f "$DIR/data/model_v51_d128_dagger5_ft.pt" ] && R="--resume"
    "$PYG" "$DIR/train.py" --data "$V51" --batch 512 --device cuda --w-dense 0 --w-q1 0 --w-qs 1 --seed 0 --d 128 --micro 2 --val2 val_dagger5.npz \
        --train train.npz@$F,dagger1.npz@$F,train1m.npz@$F,dagger2.npz@$F,danger.npz,dagger4.npz,dagger5.npz --epochs 3 --lr 3e-4 --n-teachers 3 \
        --init "$DIR/data/model_$STUDENT.pt" --out "$DIR/data/model_v51_d128_dagger5_ft.pt" $R >> "$LOG" 2>&1
fi
grep -q "^epoch 3: val2" "$LOG" || { echo "training did not finish (see $LOG)" | tee -a "$OUT"; exit 1; }
echo "== v51_d128_dagger5_ft" | tee -a "$OUT"
grep "^epoch 3:" "$LOG" | sed 's/argmax Qs top1 [0-9.]* \/ [0-9.]*, Q1.*Qs [0-9.]* |/|/' | tee -a "$OUT"
"$PYG" "$DIR/eval_val.py" --model "$(m "$DIR/data/model_$STUDENT.pt")" --data "$(m "$V51")" --val val_dagger5.npz 2>&1 | tail -3 | sed "s/^/$STUDENT on val_dagger5: /" | tee -a "$OUT"
# 7. tests
DE="$DIR/data/dagger3_eval"
play() {   # model style tag games flags...
    f="$DE/$1_$2_$3.games"; md=$1; s=$2; t=$3; g=$4; shift 4
    [ -f "$f" ] && grep -q "^AtTenTris" "$f" && return
    rm -f "$f"
    echo "$md $s $t: $("$PYC" "$DIR/play.py" --model "$DIR/data/model_$md.pt" --style $s --games $g "$@" --out "$(m "$f")" \
        | grep "^AtTenTris" | sed 's/.*: APP/APP/' | cut -c1-200)" | tee -a "$OUT"
}
n=v51_d128_dagger5_ft
play $n normal none 256 --depth 1
play $n normal d2 64 --depth 2 --beam 4
for s in normal danger; do
    play $n $s g20 64 --depth 1 --seed 33001 --garbage 10,3-4,cheese0.2,0.2
    play $n $s g50 64 --depth 1 --seed 33001 --garbage 10,3-4,cheese0.5,0.2
done
args=""
for md in v51_d128_dagger3_ft v51_d128_dagger5_ft; do
    for x in normal_none normal_d2 normal_g20 normal_g50 danger_g20 danger_g50; do args="$args ${md}_$x=$(m "$DE")/${md}_$x.games"; done
done
echo "== APP / VS: d=128 dagger3-ft vs dagger5-ft (same seeds)" | tee -a "$OUT"
"$PYC" "$DIR/vs_score.py" $args | tee -a "$OUT"
echo "dagger5 done"
