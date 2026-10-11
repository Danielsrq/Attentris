#!/bin/sh
# AtTenTris v4 data: ~1M positions, 1000 games x 500 pieces per teacher (normal.txt, gate4),
# normal rules, every legal move with a search score (--all-roots 1). Two thirds of the games
# under the training garbage (every 15 pieces 4 clean rows, the hole moving to an adjacent
# column with chance 0.2), one third under the stress garbage (every 10 pieces 3-4 rows, 20%
# of batches cheese) so the data includes play under pressure and the teacher's own deaths.
# One file per game (resumable), then packed in parallel into data/v3/train1m.npz.
#   THREADS=16 sh attentris/gen_1m.sh
DIR=$(cd "$(dirname "$0")" && pwd)
BOT="$DIR/../bot/target/release/bot.exe"
PYCPU="$DIR/../external/tetrisbattle/.venv/Scripts/python.exe"
OUT="$DIR/data/raw_1m"
T=${THREADS:-16}
TRAIN_G="15,4,0,0.2"
STRESS_G="10,3-4,cheese0.2,0.2"
W_normal="$DIR/../bot/weights/normal.txt"
W_gate4="$DIR/../bot/results/gate4/gate4.txt"
mkdir -p "$OUT"
gen() {   # teacher weights label garbage games seed
    "$BOT" dataset --mode normal --garbage $4 --weights "$2" --label $1 --all-roots 1 \
        --games $5 --pieces 500 --seed $6 --threads $T --out "$OUT"
}
gen normal "$W_normal" x $TRAIN_G 667 10001
gen gate4 "$W_gate4" x $TRAIN_G 667 20001
gen normal "$W_normal" x $STRESS_G 333 30001
gen gate4 "$W_gate4" x $STRESS_G 333 40001
echo "games: $(ls "$OUT"/*.jsonl | wc -l)"
"$PYCPU" "$DIR/pack.py" --only "$OUT" --name train1m --jobs 16 --out "$DIR/data/v3"
echo "gen_1m done"
