#!/bin/sh
# CEM-classic-d1: CEM tuned at depth 1 for LINES CLEARED over the full piece budget (the classic
# Tetris objective: survive and clear lines, attack ignored), no garbage, from normal.txt, 30 gens x 32
# candidates x 24 games x 1000 pieces (1000 so survivors still differ in lines). Then the strict test
# at depth 1: 64 games x 500 pieces and 64 x 2000 pieces without garbage (survival must be 0/64), and
# 80/20 / 50/50 for completeness. Waits for depth6.sh (it uses the bot binary) to finish.
DIR=$(cd "$(dirname "$0")" && pwd)
BOT="$DIR/../../target/release/bot.exe"
PYC="$DIR/../../../external/tetrisbattle/.venv/Scripts/python.exe"
VS="$DIR/../../../attentris/vs_score.py"
WD=$(cygpath -m "$DIR")
until [ -f "$DIR/eval_depth6.txt" ]; do sleep 20; done
(cd "$DIR/../.." && cargo build --release 2>&1 | tail -1)
[ -f "$DIR/d1_classic.txt" ] || "$BOT" tune --algo cem --mode normal --weights "$DIR/../../weights/normal.txt" --freeze default \
    --gens 30 --pop 32 --elite 8 --games 24 --pieces 1000 --beam 1 --depth 1 --threads 16 --fitness lines --save "$DIR/d1_classic.txt"
args=""
for g in none500:-:500 none2000:-:2000 cheese20:10,3-4,cheese0.2,0.2:500 cheese50:10,3-4,cheese0.5,0.2:500; do
    tag=$(echo $g | cut -d: -f1); garb=$(echo $g | cut -d: -f2); pcs=$(echo $g | cut -d: -f3)
    ga=""; [ "$garb" != "-" ] && ga="--garbage $garb"
    out="$WD/eval_d1_classic_$tag.jsonl"
    [ -f "$out" ] || "$BOT" play --mode normal --weights "$DIR/d1_classic.txt" $ga --depth 1 --beam 1 --games 64 --pieces $pcs --seed 33001 \
        --threads 16 --label classic --game-log "$out" > /dev/null 2>&1
done
for tag in none500 cheese20 cheese50; do args="$args d1_classic_$tag=$WD/eval_d1_classic_$tag.jsonl"; done
"$PYC" "$VS" $args | tee "$DIR/eval_classic.txt"
"$PYC" "$VS" --pieces 2000 "d1_classic_none2000=$WD/eval_d1_classic_none2000.jsonl" | tee -a "$DIR/eval_classic.txt"
echo "classic done"
