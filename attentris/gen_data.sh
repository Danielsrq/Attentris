#!/bin/sh
# AtTenTris training data: 50/50 normal.txt and gate4 teachers, both under normal rules with
# the gate4 garbage (every 15 pieces 4 clean rows, the hole moving to an adjacent column
# with chance 0.2). 500-piece games, one file per game in attentris/data/raw/; finished
# games are skipped, so rerunning resumes.
#   GAMES=100 THREADS=16 sh attentris/gen_data.sh
DIR=$(cd "$(dirname "$0")" && pwd)
BOT="$DIR/../bot/target/release/bot.exe"
OUT="$DIR/data/raw"
GAMES=${GAMES:-100}
T=${THREADS:-16}
G="15,4,0,0.2"
mkdir -p "$OUT"
# different seed ranges, so the two teachers see different piece sequences
"$BOT" dataset --mode normal --garbage $G --weights "$DIR/../bot/weights/normal.txt" --label normal \
    --games $GAMES --pieces 500 --seed 1001 --threads $T --out "$OUT"
"$BOT" dataset --mode normal --garbage $G --weights "$DIR/../bot/results/gate4/gate4.txt" --label gate4 \
    --games $GAMES --pieces 500 --seed 2001 --threads $T --out "$OUT"
echo "all done: $(ls "$OUT"/*.jsonl | wc -l) games"
