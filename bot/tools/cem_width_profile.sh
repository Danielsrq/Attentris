#!/bin/sh
# CEM beam-width profile at depth 6, the NN profile's conditions (attentris/nn_search_profile.sh): coldclear
# rules, no 180, 80/20 cheese garbage, 16 games x 500 pieces, seeds 33001+, 8 games at a time.
#   sh bot/tools/cem_width_profile.sh > bot/results/cem_width/profile.out 2>&1
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR/../../attentris"; . "$ATT/plat.sh"
ROOT="$DIR/../.."
BOT="$ROOT/bot/target/release/bot$EXE"
PY="$CPUPY"
R="$ROOT/bot/results/cem_width"
WR=$(wpath "$R")
mkdir -p "$R"
args=""
for w in normal survive; do
    case $w in normal) WF=$(wpath "$ROOT/bot/weights/normal.txt");; survive) WF=$(wpath "$ROOT/bot/results/survive/survive.txt");; esac
    for b in 8 16 20 64; do
        f="$WR/${w}_d6b$b.jsonl"
        [ -f "$f" ] || "$BOT" play --mode coldclear --no-180 --weights "$WF" --depth 6 --beam $b --games 16 --pieces 500 --seed 33001 \
            --threads 8 --label ${w}_d6b$b --garbage 10,3-4,cheese0.2,0.2 --game-log "$f.part" > "$R/${w}_d6b$b.log" 2>&1 && mv "$f.part" "$f"
        echo "${w}_d6b$b ms/pc $(grep -A1 '^config' "$R/${w}_d6b$b.log" | tail -1 | awk '{print $(NF-1)}')"
        args="$args ${w}_d6b$b=$f"
    done
done
"$PY" "$ROOT/attentris/vs_score.py" $args
