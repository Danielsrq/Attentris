#!/bin/sh
# DAgger round 2, training: v5.1 architecture and tokens, two ways of using the 88k labelled student
# states (data/raw_dagger2, split_dagger2.py: every 10th seed held out as "student positions")
#   5.1-dagger     from scratch, 8 epochs (as v5.1), original 1.09M + dagger2 twice per epoch (~14%)
#   5.1-dagger-ft  warm start from model_v51.pt, 3 epochs of a 70/30 mix (a fresh 19% draw of each
#                  original file + all of dagger2 each epoch), lr 3e-4
# Both report the teacher's validation set and the held-out student positions (val2). Then play
# tests against v5.1: depth 1 x 256 games (survival), depth 2 beam 4 and depth 3 beam 8 x 64 games,
# and depth 1 with garbage x 64 games (fresh seeds 33001+). Resumable: every step skips what is done.
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR"; . "$ATT/plat.sh"
PYG="$DIR/.venv/$VPY"
PYC="$CPUPY"
V51="$DIR/data/v51"
OUT="$DIR/results_dagger2_train.txt"
[ -f "$DIR/data/raw_dagger2_tr/normal_dagger2_s0.jsonl" ] || "$PYC" "$DIR/split_dagger2.py"
[ -f "$V51/dagger2.npz" ] || "$PYC" "$DIR/pack.py" --only "$(wpath "$DIR/data/raw_dagger2_tr")" --name dagger2 --jobs 16 --out "$(wpath "$V51")" | tail -1
[ -f "$V51/val_dagger2.npz" ] || "$PYC" "$DIR/pack.py" --only "$(wpath "$DIR/data/raw_dagger2_val")" --name val_dagger2 --jobs 4 --out "$(wpath "$V51")" | tail -1
echo "== dagger2 training $(date)" | tee -a "$OUT"
"$PYG" "$DIR/eval_val.py" --model "$DIR/data/model_v51.pt" --data "$V51" --val val_dagger2.npz | tee -a "$OUT"

C="--data $V51 --batch 512 --device cuda --w-dense 0 --w-q1 0 --w-qs 1 --seed 0 --val2 val_dagger2.npz"
run() {   # name, log, args...
    name=$1; log=$2; shift 2
    if ! grep -q "^epoch $E: val2" "$log" 2>/dev/null; then
        R=""; [ -f "$DIR/data/model_$name.pt" ] && R="--resume"
        "$PYG" "$DIR/train.py" $C --out "$DIR/data/model_$name.pt" $R "$@" >> "$log" 2>&1
    fi
    echo "== $name" | tee -a "$OUT"
    grep "^epoch $E:" "$log" | tee -a "$OUT"
}
E=8 run v51_dagger "$DIR/train_v51_dagger.out" --train train.npz,dagger1.npz,train1m.npz,dagger2.npz@2 --epochs 8 --lr 1.5e-3
E=3 run v51_dagger_ft "$DIR/train_v51_dagger_ft.out" --train train.npz@0.19,dagger1.npz@0.19,train1m.npz@0.19,dagger2.npz \
    --epochs 3 --lr 3e-4 --init "$DIR/data/model_v51.pt"

for m in model_v51 model_v51_dagger model_v51_dagger_ft; do
    echo "$m: $("$PYC" "$DIR/play.py" --model "$DIR/data/$m.pt" --games 256 --depth 1 | grep "^AtTenTris")" | tee -a "$OUT"
    echo "$m: $("$PYC" "$DIR/play.py" --model "$DIR/data/$m.pt" --games 64 --depth 1 --seed 33001 --garbage 10,3-4,cheese0.2,0.2 | grep "^AtTenTris")" | tee -a "$OUT"
done
for m in model_v51_dagger model_v51_dagger_ft; do
    for db in "2 4"      ; do
        set -- $db
        echo "$m: $("$PYC" "$DIR/play.py" --model "$DIR/data/$m.pt" --games 64 --depth $1 --beam $2 | grep "^AtTenTris")" | tee -a "$OUT"
    done
done
echo "dagger2 train done $(date)" | tee -a "$OUT"
