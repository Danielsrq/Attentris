#!/bin/sh
# Cheese hazard test: is it cheese batches that kill the bots? Same garbage seeds (the garbage
# schedule depends on the seed only) for the teacher (CEM-depth6: normal.txt, beam 64, depth 6) and
# 5.1-dagger-ft at depth 1; every game's garbage batches logged; then cheese_hazard.py.
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR"; . "$ATT/plat.sh"
PYC="$CPUPY"
BOT="$DIR/../bot/target/release/bot$EXE"
G=10,3-4,cheese0.2,0.2
N=128
D="$DIR/data/cheese"
mkdir -p "$D"
WD=$(wpath "$D")
[ -f "$D/teacher.jsonl" ] || "$BOT" play --mode normal --weights "$DIR/../bot/weights/normal.txt" --garbage $G --games $N --seed 33001 \
    --threads 16 --game-log "$WD/teacher.jsonl" | grep -E "^config|^beam" > "$D/teacher.summary"
cat "$D/teacher.summary"
[ -f "$D/dagger_ft.games" ] || "$PYC" "$DIR/play.py" --model "$DIR/data/model_v51_dagger_ft.pt" --games $N --depth 1 --seed 33001 \
    --garbage $G --out "$WD/dagger_ft.games" | grep "^AtTenTris"
"$PYC" "$DIR/cheese_hazard.py" --runs "CEM-depth6=$WD/teacher.jsonl" "5.1-dagger-ft (depth 1)=$WD/dagger_ft.games" | tee "$DIR/results_cheese.txt"
