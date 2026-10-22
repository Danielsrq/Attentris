#!/bin/sh
# MergeTris context pilot: the gate's FiLM context from R register tokens (--gate-registers R) instead of the
# mean board blob; R = 2 and 4, 200k positions of the base mix, 1 epoch, lambda 3 (as mtris_d128.sh). The
# mean-context reference, same recipe: val 0.450 / 0.430, 36.8 -> 20.9 blobs per stream (pilot 2026-10-22).
# Waits until mtris_d128.sh has finished its GPU stages (one GPU run at a time). ~2 min per run.
#   sh attentris/mtris_reg_pilot.sh > attentris/mtris_reg_pilot.out 2>&1
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR"; . "$ATT/plat.sh"
PYG="$DIR/.venv/$VPY"
until grep -q "^== mtris_d128_dagger5b_ft$" "$DIR/mtris_d128.out" 2>/dev/null; do sleep 60; done
mkdir -p "$DIR/data/mtris_pilot"
C="--arch tome --tome-pretok --tome-flat --tome-streams 2 --tome-pure --bucket 50 --tome-split greedy --tome-budget 20,36,0.25,0.10
--tome-gate --gate-lambda 3 --layers 4 --merge 0,1,1,1 --d 128 --data $(wpath "$DIR/data/v51") --batch 512 --threads 8 --device cuda
--w-dense 0 --w-q1 0 --w-qs 1 --seed 0 --micro 4 --train train.npz,dagger1.npz,train1m.npz --limit 200000 --epochs 1 --lr 1.5e-3"
for R in 2 4; do
    echo "== registers $R $(date '+%H:%M')"
    "$PYG" "$DIR/train.py" $C --gate-registers $R --out "$(wpath "$DIR/data/mtris_pilot/reg$R.pt")" 2>&1 | grep -E "^params|batch (0|200|300)/|^epoch"
done
echo "pilot done"
