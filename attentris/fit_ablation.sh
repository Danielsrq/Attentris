#!/bin/sh
# Fit / T-structure token ablations of the v5.1 lineage (as setup_ablation.sh): every token of a group
# blanked in training, validation and play (train.py --drop-kinds; play.py blanks the same kinds), same
# data, seeds and schedules. Setup tokens stay in every arm.
#   A  v51_nofitx  non-T piece slots: fit_s, fit_z, fit_l, fit_j, fit_i (open recesses and covered slots)
#   B  v51_not     T information: fit_t (open + covered) and every T structure (tsd, stsd, fractal,
#                  imperial, cspin, vertical)
#   C  v51_nofit   both (ARMS="A B C")
# Per arm: scratch 8 epochs -> DAgger 2 fine-tune -> DAgger 3 fine-tune. Tests at depth 1: the base vs
# v5.1 (256 games, no garbage); the DAgger 3 stage vs v5.1 dagger3-ft: no garbage 256 and depth 2 beam 4
# (64), 80/20 and 50/50 (64 each) in normal and danger style. Resumable.
#   sh fit_ablation.sh > fit_ablation.out 2>&1
DIR=$(cd "$(dirname "$0")" && pwd)
PYG="$DIR/.venv/Scripts/python.exe"
PYC="$DIR/../external/tetrisbattle/.venv/Scripts/python.exe"
V51="$DIR/data/v51"
D="$DIR/data/dagger3_eval"
WD=$(cygpath -m "$D")
OUT="$DIR/results_fit_ablation.txt"
mkdir "$DIR/.fit_ablation.lock" 2>/dev/null || { echo "fit_ablation.sh is already running"; exit 1; }
trap 'rmdir "$DIR/.fit_ablation.lock"' EXIT
mkdir -p "$D"
ARMS=${ARMS:-"A B"}
name_of() { case $1 in A) echo v51_nofitx;; B) echo v51_not;; C) echo v51_nofit;; esac; }
drop_of() {
    case $1 in
        A) echo "fit_s,fit_z,fit_l,fit_j,fit_i";;
        B) echo "fit_t,tsd,stsd,fractal,imperial,cspin,vertical";;
        C) echo "fit_s,fit_z,fit_l,fit_j,fit_i,fit_t,tsd,stsd,fractal,imperial,cspin,vertical";;
    esac
}
stage() {   # drop name epochs log args...
    drop=$1; name=$2; ep=$3; log=$4; shift 4
    if ! grep -q "^epoch $ep:" "$log" 2>/dev/null; then
        R=""; [ -f "$DIR/data/model_$name.pt" ] && R="--resume"
        "$PYG" "$DIR/train.py" --data "$V51" --batch 512 --device cuda --w-dense 0 --w-q1 0 --w-qs 1 --seed 0 --drop-kinds "$drop" \
            --out "$DIR/data/model_$name.pt" $R "$@" >> "$log" 2>&1
    fi
    grep -q "^epoch $ep:" "$log" || { echo "$name: training did not finish (see $log)" | tee -a "$OUT"; exit 1; }
    echo "== $name (dropped $drop)" | tee -a "$OUT"
    grep -E "^dropped token kinds|^epoch $ep:" "$log" | tail -3 | sed 's/argmax Qs.*Qs [0-9.]* |/|/' | tee -a "$OUT"
}
F=0.1712
for a in $ARMS; do
    n=$(name_of $a); dr=$(drop_of $a)
    stage "$dr" $n 8 "$DIR/train_$n.out" --train train.npz,dagger1.npz,train1m.npz --epochs 8 --lr 1.5e-3
    stage "$dr" ${n}_dagger_ft 3 "$DIR/train_${n}_dagger_ft.out" --train train.npz@0.19,dagger1.npz@0.19,train1m.npz@0.19,dagger2.npz \
        --epochs 3 --lr 3e-4 --init "$DIR/data/model_$n.pt" --val2 val_dagger2.npz
    stage "$dr" ${n}_dagger3_ft 3 "$DIR/train_${n}_dagger3_ft.out" --train train.npz@$F,dagger1.npz@$F,train1m.npz@$F,dagger2.npz@$F,danger.npz \
        --epochs 3 --lr 3e-4 --n-teachers 3 --init "$DIR/data/model_${n}_dagger_ft.pt" --val2 val_danger.npz
done
play() {   # model style tag games flags...
    f="$D/$1_$2_$3.games"; m=$1; s=$2; t=$3; g=$4; shift 4
    [ -f "$f" ] && grep -q "^AtTenTris" "$f" && return
    rm -f "$f"
    echo "$m $s $t: $("$PYC" "$DIR/play.py" --model "$DIR/data/model_$m.pt" --style $s --games $g "$@" --out "$(cygpath -m "$f")" \
        | grep "^AtTenTris" | sed 's/.*: APP/APP/' | cut -c1-200)" | tee -a "$OUT"
}
for a in $ARMS; do
    n=$(name_of $a)
    play $n normal none 256 --depth 1
    play ${n}_dagger3_ft normal none 256 --depth 1
    play ${n}_dagger3_ft normal d2 64 --depth 2 --beam 4
    for s in normal danger; do
        play ${n}_dagger3_ft $s g20 64 --depth 1 --seed 33001 --garbage 10,3-4,cheese0.2,0.2
        play ${n}_dagger3_ft $s g50 64 --depth 1 --seed 33001 --garbage 10,3-4,cheese0.5,0.2
    done
done
args="v51_base_none=$WD/v51_normal_none.games v51_nosetup_none=$WD/v51_nosetup_normal_none.games"
for x in normal_none normal_d2 normal_g20 normal_g50 danger_g20 danger_g50; do args="$args v51_dagger3_ft_$x=$WD/v51_dagger3_ft_$x.games"; done
for x in normal_none normal_g20 normal_g50 danger_g20 danger_g50; do args="$args v51_nosetup_dagger3_ft_$x=$WD/v51_nosetup_dagger3_ft_$x.games"; done
for a in $ARMS; do
    n=$(name_of $a)
    args="$args ${n}_base_none=$WD/${n}_normal_none.games"
    for x in normal_none normal_d2 normal_g20 normal_g50 danger_g20 danger_g50; do args="$args ${n}_dagger3_ft_$x=$WD/${n}_dagger3_ft_$x.games"; done
done
echo "== APP / VS: fit / T-structure ablations vs v5.1 and the setup ablation (same seeds)" | tee -a "$OUT"
"$PYC" "$DIR/vs_score.py" $args | tee -a "$OUT"
echo "fit ablation done"
