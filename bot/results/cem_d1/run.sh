#!/bin/sh
# CEM tuned AT DEPTH 1 (one placement, no lookahead -- the classic heuristic-bot setting), from
# normal.txt, 30 gens x 32 candidates x 24 games x 500 pieces (depth-1 games are ~250x cheaper):
#   d1_app.txt  no garbage, fitness APP over the full budget
#   d1_vs.txt   50/50 clean/cheese (10,3-4,cheese0.5,0.2), fitness VS = (attack + garbage cleared) / full budget
# Then all four weight sets (+ normal.txt, survive.txt) at depth 1, 64 games, no garbage / 80/20 / 50/50
# (seeds 33001+; the tune plays 1000-3950). Waits for attentris/.plan_done (today's plan uses the CPU).
DIR=$(cd "$(dirname "$0")" && pwd)
BOT="$DIR/../../target/release/bot.exe"
PYC="$DIR/../../../external/tetrisbattle/.venv/Scripts/python.exe"
VS="$DIR/../../../attentris/vs_score.py"
T=${THREADS:-16}
WD=$(cygpath -m "$DIR")
until [ -f "$DIR/../../../attentris/.plan_done" ]; do sleep 60; done
TUNE="--algo cem --mode normal --weights $DIR/../../weights/normal.txt --freeze default --gens 30 --pop 32 --elite 8 --games 24 --pieces 500 --beam 1 --depth 1 --threads $T"
echo "== CEM at depth 1, APP, no garbage $(date)"
"$BOT" tune $TUNE --save "$DIR/d1_app.txt"
echo "== CEM at depth 1, VS, 50/50 garbage $(date)"
"$BOT" tune $TUNE --garbage 10,3-4,cheese0.5,0.2 --fitness vs --save "$DIR/d1_vs.txt"
args=""
for w in "normal $DIR/../../weights/normal.txt" "survive $DIR/../survive/survive.txt" "d1_app $DIR/d1_app.txt" "d1_vs $DIR/d1_vs.txt"; do
    set -- $w; name=$1; wf=$2
    for g in none:- g20:10,3-4,cheese0.2,0.2 g50:10,3-4,cheese0.5,0.2; do
        tag=${g%%:*}; garb=${g#*:}; ga=""; [ "$garb" != "-" ] && ga="--garbage $garb"
        out="$WD/eval_${name}_$tag.jsonl"
        [ -f "$out" ] || "$BOT" play --mode normal --weights "$wf" $ga --depth 1 --beam 1 --games 64 --pieces 500 --seed 33001 --threads $T \
            --label "$name" --game-log "$out" > /dev/null 2>&1
        args="$args ${name}_d1_$tag=$out"
    done
done
"$PYC" "$VS" $args | tee "$DIR/eval.txt"
echo "cem d1 done $(date)"
