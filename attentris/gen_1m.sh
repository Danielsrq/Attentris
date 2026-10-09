#!/bin/sh
# AtTenTris v4 data: 1M positions, 1000 games x 500 pieces per teacher (normal.txt, gate4),
# normal rules + training garbage, every legal move with a search score (--all-roots 1).
# One file per game (resumable), then packed in parallel into data/v3/train1m.npz.
#   THREADS=16 sh attentris/gen_1m.sh
DIR=$(cd "$(dirname "$0")" && pwd)
BOT="$DIR/../bot/target/release/bot.exe"
PYCPU="$DIR/../external/tetrisbattle/.venv/Scripts/python.exe"
OUT="$DIR/data/raw_1m"
GAMES=${GAMES:-1000}
T=${THREADS:-16}
G="15,4,0,0.2"
mkdir -p "$OUT"
"$BOT" dataset --mode normal --garbage $G --weights "$DIR/../bot/weights/normal.txt" --label normal --all-roots 1 \
    --games $GAMES --pieces 500 --seed 10001 --threads $T --out "$OUT"
"$BOT" dataset --mode normal --garbage $G --weights "$DIR/../bot/results/gate4/gate4.txt" --label gate4 --all-roots 1 \
    --games $GAMES --pieces 500 --seed 20001 --threads $T --out "$OUT"
echo "games: $(ls "$OUT"/*.jsonl | wc -l)"
"$PYCPU" "$DIR/pack.py" --only "$OUT" --name train1m --jobs 16 --out "$DIR/data/v3"
echo "gen_1m done"
