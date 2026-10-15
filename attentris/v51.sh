#!/bin/sh
# Is v5's depth-1 dying (5/64; v4 0/64) the tokenizer or the luck of one training run?
#  (a) v4 tokens again with a different seed (init + batch order): data/model_v4_s1.pt
#  (b) v5.1 tokens, seed 0: data/model_v51.pt. v5.1 = slot masks exactly the tetromino, the
#      overhang implied (no hang token), setup masks the setup placement only (slot centre in
#      its extras)
# (a) trains on the GPU (started separately; this script waits for it) while (b) packs on the
# P-cores; then (b) trains; then 64-game play tests (depth 1, depth 2 beam 4, depth 3 beam 8;
# no garbage) for both.
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR"; . "$ATT/plat.sh"
PYG="$DIR/.venv/$VPY"
PYC="$CPUPY"
OUT="$DIR/results_v51.txt"
V51="$DIR/data/v51"
mkdir -p "$V51"
: > "$OUT"
[ -f "$V51/val.npz" ] || "$PYC" "$DIR/pack.py" --raw "$DIR/data/raw_v3" --out "$V51" | tail -2
[ -f "$V51/dagger1.npz" ] || "$PYC" "$DIR/pack.py" --only "$DIR/data/raw_dagger1_v3" --name dagger1 --jobs 16 --out "$V51" | tail -1
[ -f "$V51/train1m.npz" ] || "$PYC" "$DIR/pack.py" --only "$DIR/data/raw_1m" --name train1m --jobs 16 --out "$V51" | tail -1
echo "v5.1 packed"
until grep -q "^epoch 8:" "$DIR/train_v4_s1.out" 2>/dev/null; do sleep 30; done
sleep 20
# (a) was trained on v4 tokens with a 40-row kind table: record that, so play.py tokenizes as v4
"$PYG" -c "import torch,sys; p=sys.argv[1]; ck=torch.load(p,map_location='cpu',weights_only=False); ck['tok_version']=4; torch.save(ck,p)" "$DIR/data/model_v4_s1.pt"
echo "== v4 tokens, seed 1" | tee -a "$OUT"
grep "^epoch 8:" "$DIR/train_v4_s1.out" | tee -a "$OUT"
R=""; [ -f "$DIR/data/model_v51.pt" ] && R="--resume"
"$PYG" "$DIR/train.py" --data "$V51" --train train.npz,dagger1.npz,train1m.npz --epochs 8 --batch 512 --lr 1.5e-3 \
    --device cuda --w-dense 0 --w-q1 0 --w-qs 1 --seed 0 --out "$DIR/data/model_v51.pt" $R > "$DIR/train_v51.out" 2>&1
echo "== v5.1 tokens, seed 0" | tee -a "$OUT"
grep "^epoch 8:" "$DIR/train_v51.out" | tee -a "$OUT"
for m in model_v4_s1 model_v51; do
    for db in "1 8" "2 4" "3 8"; do
        set -- $db
        echo "$m: $("$PYC" "$DIR/play.py" --model "$DIR/data/$m.pt" --games 64 --depth $1 --beam $2 | grep "^AtTenTris")" | tee -a "$OUT"
    done
done
echo "v51 done" | tee -a "$OUT"
