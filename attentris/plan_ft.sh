#!/bin/sh
# Multi-move fine-tunes (model_plan.py) from v51_d128_dagger5b_ft, the dagger5b recipe: 3 epochs, lr 3e-4,
# batch 512, the 70/30 mix (all of danger + dagger4 + dagger5all each epoch, the older files at F each),
# val2 = val_dagger5all. The plan loss (mean over a position's plan steps, weight W_PLAN) is added to the
# move-1 loss; plan targets = the teacher's planned line (plan_label.sh, pack_plan.py).
#   VARIANT=autoreg N=2 sh attentris/plan_ft.sh   -> data/model_v51_d128_autoreg2_ft.pt
#   VARIANT=plan    N=2 sh attentris/plan_ft.sh   -> data/model_v51_d128_plan2_ft.pt
#   VARIANT=pieces  N=6 sh attentris/plan_ft.sh   -> data/model_v51_d128_pieces6_ft.pt (piece queries)
#   PLAN_LAYERS=2: pieces with 2 rounds of (cross, self) -> model_v51_d128_pieces6L2_ft.pt
#   FREEZE=1 EPOCHS=10: train only the plan branch (train.py --freeze-base) for 10 epochs -> ..._pieces6L2fze10_ft
#   PLAY=0: train only (a frozen backbone plays move 1 exactly as dagger5b-ft: its play block would repeat it)
#   NEW_LR=1.5e-3: the new plan branch's peak lr (train.py --new-lr), the pretrained parts keep 3e-4
#   VARIANT=none        sh attentris/plan_ft.sh   -> data/model_v51_d128_ctrl_ft.pt (control: the same
#                                                    3 epochs without a plan loss)
# Then the standard play block (depth 1: none 256 games, d2 64, normal / danger x 80/20, 50/50 at 64),
# depth 2 beam 4 under the same garbage, and vs_score against dagger5b-ft (and COMPARE=<model name>) on
# the same seeds. Resumable: training resumes, finished games are kept.
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR"; . "$ATT/plat.sh"
PYG="$DIR/.venv/$VPY"
PYC="$CPUPY"
V51="$DIR/data/v51"
VARIANT=${VARIANT:-autoreg}
N=${N:-2}
W_PLAN=${W_PLAN:-1.0}
EPOCHS=${EPOCHS:-3}
BASE=v51_d128_dagger5b_ft
case $VARIANT in
    none) NAME=v51_d128_ctrl_ft; PLAN="" ;;
    autoreg|plan|pieces) NAME=v51_d128_${VARIANT}${N}${PLAN_LAYERS:+L$PLAN_LAYERS}${FREEZE:+fz}$( [ "$EPOCHS" = 3 ] || echo "e$EPOCHS" )_ft; PLAN="--plan-variant $VARIANT --plan-steps $N --w-plan $W_PLAN${NEW_LR:+ --new-lr $NEW_LR}${PLAN_LAYERS:+ --plan-layers $PLAN_LAYERS}${FREEZE:+ --freeze-base}" ;;
    *) echo "VARIANT = autoreg | plan | pieces | none"; exit 1 ;;
esac
OUT="$DIR/results_plan.txt"
m() { wpath "$1"; }
echo "== $NAME (from $BASE; $VARIANT, N=$N, w_plan $W_PLAN) $(date '+%Y-%m-%d %H:%M')" | tee -a "$OUT"
F=$("$PYC" -c "
import numpy as np
d = '$(m "$V51")/'
new = sum(len(np.load(d + f)['chosen']) for f in ('danger.npz', 'dagger4.npz', 'dagger5all.npz'))
rest = sum(len(np.load(d + f)['chosen']) for f in ('train.npz', 'dagger1.npz', 'train1m.npz', 'dagger2.npz'))
print('%.4f' % (7 / 3 * new / rest))")
LOG="$DIR/train_$NAME.out"
if ! grep -q "^epoch $EPOCHS: val2 (" "$LOG" 2>/dev/null; then
    R=""; [ -f "$DIR/data/model_$NAME.pt" ] && R="--resume"
    "$PYG" "$DIR/train.py" --data "$(m "$V51")" --batch 512 --threads 8 --device cuda --w-dense 0 --w-q1 0 --w-qs 1 --seed 0 --d 128 --micro 2 \
        --val2 val_dagger5all.npz --train train.npz@$F,dagger1.npz@$F,train1m.npz@$F,dagger2.npz@$F,danger.npz,dagger4.npz,dagger5all.npz \
        --epochs $EPOCHS --lr 3e-4 --n-teachers 3 $PLAN --init "$(m "$DIR/data/model_$BASE.pt")" --out "$(m "$DIR/data/model_$NAME.pt")" $R >> "$LOG" 2>&1
fi
grep -q "^epoch $EPOCHS: val2 (" "$LOG" || { echo "training did not finish (see $LOG)" | tee -a "$OUT"; exit 1; }
grep "^epoch $EPOCHS:" "$LOG" | sed 's/argmax Qs top1 [0-9.]* \/ [0-9.]*, Q1.*Qs [0-9.]* |/|/' | tee -a "$OUT"
if [ "${PLAY:-1}" = 0 ]; then echo "$NAME trained (PLAY=0: no play block)"; exit 0; fi
DE="$DIR/data/dagger3_eval"
play() {   # model style tag games flags...
    f="$DE/$1_$2_$3.games"; md=$1; s=$2; t=$3; g=$4; shift 4
    [ -f "$f" ] && grep -q "^AtTenTris" "$f" && return
    rm -f "$f"
    echo "$md $s $t: $("$PYC" "$DIR/play.py" --model "$DIR/data/model_$md.pt" --style $s --games $g "$@" --out "$(m "$f")" \
        | grep "^AtTenTris" | sed 's/.*: APP/APP/' | cut -c1-200)" | tee -a "$OUT"
}
play $NAME normal none 256 --depth 1
play $NAME normal d2 64 --depth 2 --beam 4
for s in normal danger; do
    play $NAME $s g20 64 --depth 1 --seed 33001 --garbage 10,3-4,cheese0.2,0.2
    play $NAME $s g50 64 --depth 1 --seed 33001 --garbage 10,3-4,cheese0.5,0.2
    # depth 2 under garbage too (where autoreg2-ft searched much better than dagger5b-ft)
    play $NAME $s d2g20 64 --depth 2 --beam 4 --seed 33001 --garbage 10,3-4,cheese0.2,0.2
    play $NAME $s d2g50 64 --depth 2 --beam 4 --seed 33001 --garbage 10,3-4,cheese0.5,0.2
done
args=""
for md in $BASE ${COMPARE:-} $NAME; do
    for x in normal_none normal_d2 normal_g20 normal_g50 danger_g20 danger_g50 normal_d2g20 normal_d2g50 danger_d2g20 danger_d2g50; do
        [ -f "$DE/${md}_$x.games" ] && args="$args ${md}_$x=$(m "$DE")/${md}_$x.games"
    done
done
echo "== APP / VS: $BASE vs ${COMPARE:+$COMPARE vs }$NAME" | tee -a "$OUT"
"$PYC" "$DIR/vs_score.py" $args | tee -a "$OUT"
echo "$NAME done"
