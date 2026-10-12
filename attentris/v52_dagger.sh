#!/bin/sh
# v5.2 through the dagger3-ft lineage: for v5.2-loop (1 cross block x 2 passes) and v5.2-untied (2
# independent cross blocks): DAgger 2 fine-tune (70/30 with dagger2, 3 epochs, lr 3e-4, as
# dagger2_train.sh) -> DAgger 3 fine-tune (70/30 with danger, 3 epochs, lr 3e-4, 3 style rows, as
# dagger3_train.sh). Tests at depth 1 (normal: no garbage 256 games; normal + danger at 80/20 and 50/50,
# 64 games) and depth 2 beam 4 (normal, no garbage, 64 games), against v5.1-dagger3-ft. Resumable;
# one copy at a time.
DIR=$(cd "$(dirname "$0")" && pwd)
PYG="$DIR/.venv/Scripts/python.exe"
PYC="$DIR/../external/tetrisbattle/.venv/Scripts/python.exe"
V51="$DIR/data/v51"
D="$DIR/data/dagger3_eval"
WD=$(cygpath -m "$D")
OUT="$DIR/results_v52_dagger.txt"
mkdir "$DIR/.v52_dagger.lock" 2>/dev/null || { echo "v52_dagger.sh is already running"; exit 1; }
trap 'rmdir "$DIR/.v52_dagger.lock"' EXIT
C="--data $V51 --batch 512 --device cuda --w-dense 0 --w-q1 0 --w-qs 1 --seed 0"
stage() {   # name epochs log args...
    name=$1; ep=$2; log=$3; shift 3
    if ! grep -q "^epoch $ep: val2" "$log" 2>/dev/null; then
        R=""; [ -f "$DIR/data/model_$name.pt" ] && R="--resume"
        "$PYG" "$DIR/train.py" $C --out "$DIR/data/model_$name.pt" $R "$@" >> "$log" 2>&1
    fi
    grep -q "^epoch $ep: val2" "$log" || { echo "$name: training did not finish (see $log)" | tee -a "$OUT"; exit 1; }
    echo "== $name" | tee -a "$OUT"
    grep "^epoch $ep:" "$log" | sed 's/argmax Qs.*Qs [0-9.]* |/|/' | tee -a "$OUT"
}
F=0.1712
for v in "loop 1 2" "untied 2 1"; do
    set -- $v; kind=$1; X="--cross-layers $2 --cross-loops $3"
    stage v52_${kind}_dagger_ft 3 "$DIR/train_v52_${kind}_dagger_ft.out" $X \
        --train train.npz@0.19,dagger1.npz@0.19,train1m.npz@0.19,dagger2.npz --epochs 3 --lr 3e-4 \
        --init "$DIR/data/model_v52_$kind.pt" --val2 val_dagger2.npz
    stage v52_${kind}_dagger3_ft 3 "$DIR/train_v52_${kind}_dagger3_ft.out" $X \
        --train train.npz@$F,dagger1.npz@$F,train1m.npz@$F,dagger2.npz@$F,danger.npz --epochs 3 --lr 3e-4 --n-teachers 3 \
        --init "$DIR/data/model_v52_${kind}_dagger_ft.pt" --val2 val_danger.npz
done
play() {   # model style tag games flags...
    m=$1; s=$2; tag=$3; n=$4; shift 4
    f="$D/${m}_${s}_$tag.games"
    [ -f "$f" ] && grep -q "^AtTenTris" "$f" && return
    rm -f "$f"
    echo "$m $s $tag: $("$PYC" "$DIR/play.py" --model "$DIR/data/model_$m.pt" --games $n --style $s "$@" --out "$(cygpath -m "$f")" \
        | grep "^AtTenTris" | sed 's/.*: APP/APP/' | cut -c1-190)" | tee -a "$OUT"
}
for m in v52_loop_dagger3_ft v52_untied_dagger3_ft v51_dagger3_ft; do
    play $m normal none 256 --depth 1
    play $m normal d2 64 --depth 2 --beam 4
    for s in normal danger; do
        play $m $s g20 64 --depth 1 --seed 33001 --garbage 10,3-4,cheese0.2,0.2
        play $m $s g50 64 --depth 1 --seed 33001 --garbage 10,3-4,cheese0.5,0.2
    done
done
args=""
for m in v51_dagger3_ft v52_loop_dagger3_ft v52_untied_dagger3_ft; do
    for x in normal_none normal_d2 normal_g20 normal_g50 danger_g20 danger_g50; do args="$args ${m}_$x=$WD/${m}_$x.games"; done
done
"$PYC" "$DIR/vs_score.py" $args | tee -a "$OUT"
echo "v52 dagger done"
