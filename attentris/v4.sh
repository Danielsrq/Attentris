#!/bin/sh
# AtTenTris v4 after the 1M dataset: wait for data/v3/train1m.npz, train (GPU, v3 recipe:
# policy + Qs on every-move labels), then play tests (policy depth 1 and depth 3, top-10
# shortlist + teacher search; 64 games, no garbage and stress) and the teacher's
# self-agreement (beam 128 vs 64 on the validation games).
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR"; . "$ATT/plat.sh"
PYG="$DIR/.venv/$VPY"
PYC="$CPUPY"
OUT="$DIR/results_v4.txt"
until [ -f "$DIR/data/v3/train1m.npz" ] && grep -q "gen_1m done" "$DIR/gen_1m.out"; do sleep 30; done
: > "$OUT"
"$PYG" "$DIR/train.py" --data "$DIR/data/v3" --train train.npz,dagger1.npz,train1m.npz --epochs 8 --batch 512 --lr 1.5e-3 \
    --device cuda --w-dense 0 --w-q1 0 --w-qs 1 --out "$DIR/data/model_v4.pt" > "$DIR/train_v4.out" 2>&1
grep "^epoch" "$DIR/train_v4.out" | tee -a "$OUT"
STRESS="10,3-4,cheese0.2,0.2"
for g in "" "$STRESS"; do
    gflag=""; [ -n "$g" ] && gflag="--garbage $g"
    echo "== garbage: ${g:-none}" | tee -a "$OUT"
    "$PYC" "$DIR/play.py" --model "$DIR/data/model_v4.pt" --games 64 --depth 1 $gflag | grep "^AtTenTris" | tee -a "$OUT"
    "$PYC" "$DIR/play.py" --model "$DIR/data/model_v4.pt" --games 64 --depth 3 --beam 8 $gflag | grep "^AtTenTris" | tee -a "$OUT"
    "$PYC" "$DIR/play.py" --model "$DIR/data/model_v4.pt" --games 64 --shortlist 10 $gflag | grep "^AtTenTris" | tee -a "$OUT"
done
"$PYC" "$DIR/teacher_agreement.py" | tee -a "$OUT"
echo "v4 done" | tee -a "$OUT"
