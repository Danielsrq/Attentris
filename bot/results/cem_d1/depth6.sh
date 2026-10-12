#!/bin/sh
# The depth-1-tuned weights (d1_app.txt, d1_vs.txt) played WITH the teacher's search (beam 64, depth 6),
# 64 games each with no garbage / 80/20 / 50/50 on seeds 33001+ -- the same games as
# bot/results/survive/eval.txt (normal.txt and survive.txt at depth 6).
DIR=$(cd "$(dirname "$0")" && pwd)
BOT="$DIR/../../target/release/bot.exe"
PYC="$DIR/../../../external/tetrisbattle/.venv/Scripts/python.exe"
VS="$DIR/../../../attentris/vs_score.py"
WD=$(cygpath -m "$DIR")
SV=$(cygpath -m "$DIR/../survive")
args=""
for w in d1_app d1_vs; do
    for g in none:- cheese20:10,3-4,cheese0.2,0.2 cheese50:10,3-4,cheese0.5,0.2; do
        tag=${g%%:*}; garb=${g#*:}; ga=""; [ "$garb" != "-" ] && ga="--garbage $garb"
        out="$WD/eval_d6_${w}_$tag.jsonl"
        [ -f "$out" ] || "$BOT" play --mode normal --weights "$DIR/$w.txt" $ga --games 64 --pieces 500 --seed 33001 --threads 16 \
            --label "$w" --game-log "$out" > /dev/null 2>&1
        args="$args ${w}_d6_$tag=$out"
    done
done
for w in normal survive; do for tag in none cheese20 cheese50; do args="$args ${w}_d6_$tag=$SV/eval_${w}_$tag.jsonl"; done; done
"$PYC" "$VS" $args | tee "$DIR/eval_depth6.txt"
