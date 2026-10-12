#!/bin/sh
# The CEM weight sets at depth 1 (one placement, no lookahead -- the classic heuristic-bot setting):
# normal.txt (APP-tuned) and survive.txt (VS-tuned under 50/50), no garbage / 80/20 / 50/50, 16 games.
DIR=$(cd "$(dirname "$0")" && pwd)
BOT="$DIR/../../target/release/bot.exe"
for w in "normal $DIR/../../weights/normal.txt" "survive $DIR/survive.txt"; do
    set -- $w; name=$1; wf=$2
    for g in none 10,3-4,cheese0.2,0.2 10,3-4,cheese0.5,0.2; do
        ga=""; [ "$g" != none ] && ga="--garbage $g"
        line=$("$BOT" play --mode normal --weights "$wf" $ga --depth 1 --beam 1 --games 16 --pieces 500 --seed 33001 --threads 4 --label "$name" 2>/dev/null | grep "^$name ")
        echo "$name depth 1, garbage $g: $line"
    done
done
