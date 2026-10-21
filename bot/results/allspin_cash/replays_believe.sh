#!/bin/sh
# 1) Replays to watch: allspin_cash_e15.txt under its own rules (--mode allspin_cash), 2 games per setting
#    (no garbage, 80/20 and 50/50 at E=15, seeds 33001-2, the first games of the evals), written to
#    data/bot/allspin_cash/ for the Godot viewer:
#      godot --path . res://replay.tscn -- --replay=res://data/bot/allspin_cash/cash_g20_e15.jsonl
# 2) Belief test: the same weights searching as if the rules were allspin_cash (--believe allspin_cash)
#    while the game scores normal rules (--mode normal), 16 games per setting, garbage every 10 pieces
#    (the standard protocol, as in ../allspin_survive/believe/) and every 15; replays to
#    data/bot/allspin_cash/believe_*.jsonl, B2B kept etc. by spin_stats.py into believe/eval_spin.txt.
#   sh bot/results/allspin_cash/replays_believe.sh > bot/results/allspin_cash/replays_believe.out 2>&1
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR/../../../attentris"; . "$ATT/plat.sh"
BOT="$DIR/../../target/release/bot$EXE"
WF="$(wpath "$DIR/allspin_cash_e15.txt")"
REP="$DIR/../../../data/bot/allspin_cash"
OUT="$DIR/believe"
mkdir -p "$REP" "$OUT"

for tag in none g20_e15 g50_e15; do
    case $tag in none) gflag="" ;; g20*) gflag="--garbage 15,3-4,cheese0.2,0.2" ;; g50*) gflag="--garbage 15,3-4,cheese0.5,0.2" ;; esac
    echo "== replay cash_$tag $(date '+%H:%M')"
    "$BOT" play --mode allspin_cash $gflag --weights "$WF" --games 2 --pieces 500 --seed 33001 \
        --label "allspin_cash_e15" --out "$(wpath "$REP/cash_$tag.jsonl")" | sed -n '1,3p;/wrote/p'
done

for tag in none g20 g50 g20_e15 g50_e15; do
    case $tag in
        none) gflag="" ;;
        g20) gflag="--garbage 10,3-4,cheese0.2,0.2" ;; g50) gflag="--garbage 10,3-4,cheese0.5,0.2" ;;
        g20_e15) gflag="--garbage 15,3-4,cheese0.2,0.2" ;; g50_e15) gflag="--garbage 15,3-4,cheese0.5,0.2" ;;
    esac
    echo "== believe allspin_cash, scored normal: $tag $(date '+%H:%M')"
    "$BOT" play --mode normal --believe allspin_cash $gflag --weights "$WF" --games 16 --pieces 500 --seed 33001 \
        --label "allspin_cash_e15" --game-log "$(wpath "$OUT/allspin_cash_e15_$tag.jsonl")" \
        --out "$(wpath "$REP/believe_$tag.jsonl")" | sed -n "1,3p;/wrote/p"
done
args=""
for tag in none g20 g50 g20_e15 g50_e15; do
    args="$args believe_$tag=$(wpath "$REP/believe_$tag.jsonl"),$(wpath "$OUT/allspin_cash_e15_$tag.jsonl")"
done
"$CPUPY" "$DIR/../../tools/spin_stats.py" $args | tee "$OUT/eval_spin.txt"
echo "all done"
