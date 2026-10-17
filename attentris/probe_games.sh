#!/bin/sh
# On-policy games for the register probes: v51_d128_reg4_dagger5b_ft, normal style, depth 1, 128 games each at
# 80/20 and 50/50 garbage, every state dumped (play.py --dump) with each game's result (--out: died, pieces).
#   sh attentris/probe_games.sh > attentris/data/probe/play.log 2>&1
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR"; . "$ATT/plat.sh"
P="$DIR/data/probe"
mkdir -p "$P"
M="$(wpath "$DIR/data/model_v51_d128_reg4_dagger5b_ft.pt")"
[ -s "$P/play_g20.jsonl" ] || "$CPUPY" "$DIR/play.py" --model "$M" --style normal --games 128 --depth 1 --seed 71001 \
    --garbage 10,3-4,cheese0.2,0.2 --dump "$(wpath "$P/play_g20.jsonl")" --out "$(wpath "$P/play_g20.games")" | tail -1
[ -s "$P/play_g50.jsonl" ] || "$CPUPY" "$DIR/play.py" --model "$M" --style normal --games 128 --depth 1 --seed 72001 \
    --garbage 10,3-4,cheese0.5,0.2 --dump "$(wpath "$P/play_g50.jsonl")" --out "$(wpath "$P/play_g50.games")" | tail -1
echo "probe games done"
