#!/bin/sh
# Setup-token ablation: the v5.1 lineage again with every setup_* token blanked (train.py --drop-kinds
# setup_; play.py then asks bot expand for no setups), same data, seeds and schedules:
#   1. v5.1-nosetup   from scratch, 8 epochs (v51.sh)
#   2. + DAgger 2     fine-tune 3 epochs 70/30 dagger2, lr 3e-4 (dagger2_train.sh)
#   3. + DAgger 3     fine-tune 3 epochs 70/30 danger, 3 style rows (dagger3_train.sh)
# Tests at depth 1 on the seeds of the with-setup models: stage 1 vs v5.1 (256 games, no garbage);
# stage 3 vs dagger3-ft: no garbage 256, 80/20 and 50/50 64 each, normal and danger styles. Resumable.
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR"; . "$ATT/plat.sh"
PYG="$DIR/.venv/$VPY"
PYC="$CPUPY"
V51="$DIR/data/v51"
D="$DIR/data/dagger3_eval"
WD=$(wpath "$D")
OUT="$DIR/results_setup_ablation.txt"
C="--data $V51 --batch 512 --device cuda --w-dense 0 --w-q1 0 --w-qs 1 --seed 0 --drop-kinds setup_"
mkdir -p "$D"
stage() {   # name epochs log args...
    name=$1; ep=$2; log=$3; shift 3
    if ! grep -q "^epoch $ep:" "$log" 2>/dev/null; then
        R=""; [ -f "$DIR/data/model_$name.pt" ] && R="--resume"
        "$PYG" "$DIR/train.py" $C --out "$DIR/data/model_$name.pt" $R "$@" >> "$log" 2>&1
    fi
    grep -q "^epoch $ep:" "$log" || { echo "$name: training did not finish (see $log)" | tee -a "$OUT"; exit 1; }
    echo "== $name" | tee -a "$OUT"
    grep "^epoch $ep:" "$log" | sed 's/argmax Qs.*Qs [0-9.]* |/|/' | tee -a "$OUT"
}
stage v51_nosetup 8 "$DIR/train_v51_nosetup.out" --train train.npz,dagger1.npz,train1m.npz --epochs 8 --lr 1.5e-3
stage v51_nosetup_dagger_ft 3 "$DIR/train_v51_nosetup_dagger_ft.out" --train train.npz@0.19,dagger1.npz@0.19,train1m.npz@0.19,dagger2.npz \
    --epochs 3 --lr 3e-4 --init "$DIR/data/model_v51_nosetup.pt" --val2 val_dagger2.npz
F=0.1712
stage v51_nosetup_dagger3_ft 3 "$DIR/train_v51_nosetup_dagger3_ft.out" --train train.npz@$F,dagger1.npz@$F,train1m.npz@$F,dagger2.npz@$F,danger.npz \
    --epochs 3 --lr 3e-4 --n-teachers 3 --init "$DIR/data/model_v51_nosetup_dagger_ft.pt" --val2 val_danger.npz

play() {   # model style tag garbage games
    f="$D/$1_$2_$3.games"
    [ -f "$f" ] && grep -q "^AtTenTris" "$f" && return
    rm -f "$f"
    g=""; [ "$4" != "-" ] && g="--seed 33001 --garbage $4"
    echo "$1 $2 $3: $("$PYC" "$DIR/play.py" --model "$DIR/data/model_$1.pt" --depth 1 --style $2 --games $5 $g --out "$(wpath "$f")" \
        | grep "^AtTenTris" | sed 's/.*: APP/APP/' | cut -c1-200)" | tee -a "$OUT"
}
play v51_nosetup normal none - 256
play v51 normal none - 256
for m in v51_nosetup_dagger3_ft v51_dagger3_ft; do
    play $m normal none - 256
    for s in normal danger; do
        play $m $s g20 10,3-4,cheese0.2,0.2 64
        play $m $s g50 10,3-4,cheese0.5,0.2 64
    done
done
args=""
for x in v51_normal_none v51_nosetup_normal_none v51_dagger3_ft_normal_none v51_nosetup_dagger3_ft_normal_none; do args="$args $x=$WD/${x}.games"; done
for m in v51_dagger3_ft v51_nosetup_dagger3_ft; do for s in normal danger; do for t in g20 g50; do args="$args ${m}_${s}_$t=$WD/${m}_${s}_$t.games"; done; done; done
"$PYC" "$DIR/vs_score.py" $args | tee -a "$OUT"
echo "setup ablation done"
