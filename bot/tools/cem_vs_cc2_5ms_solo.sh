#!/bin/sh
# Solo quality at a ~5 ms budget, CC2's rules (--mode standard --no-180), 16 games x 500 pieces, no garbage and
# 50/50 garbage: CEM-survival at d4 b16 / d5 b12 (~5 ms per move), d6 b64 (reference) vs Cold Clear 2 at 5 ms.
#   sh bot/tools/cem_vs_cc2_5ms_solo.sh        results in bot/results/survive/budget/std_*
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR/../../attentris"; . "$ATT/plat.sh"
ROOT=$(cd "$DIR/../.." && pwd)
R="$ROOT/bot/results/survive/budget"
mkdir -p "$R"
BOT="$ROOT/bot/target/release/bot$EXE"
W="$(wpath "$ROOT/bot/results/survive/survive.txt")"
G50="--garbage 10,3-4,cheese0.5,0.2"
for cfg in 4x16 5x12 6x64; do
    d=${cfg%x*}; b=${cfg#*x}
    [ -s "$R/std_d${d}b${b}_none.jsonl" ] || "$BOT" play --mode standard --no-180 --weights "$W" --depth "$d" --beam "$b" --games 16 \
        --pieces 500 --seed 1 --threads 8 --game-log "$(wpath "$R/std_d${d}b${b}_none.jsonl")" > "$R/std_d${d}b${b}_none.txt" 2>&1
    [ -s "$R/std_d${d}b${b}_g50.jsonl" ] || "$BOT" play --mode standard --no-180 --weights "$W" --depth "$d" --beam "$b" --games 16 \
        --pieces 500 --seed 33001 --threads 8 $G50 --game-log "$(wpath "$R/std_d${d}b${b}_g50.jsonl")" > "$R/std_d${d}b${b}_g50.txt" 2>&1
done
RW="$(wpath "$R")"
cd "$ROOT"
[ -s "$R/std_cc2_5_none.jsonl" ] || "$CPUPY" bot/tools/cc2_play.py --games 16 --pieces 500 --seed 1 --think-ms 5 --jobs 8 \
    --out "$(wpath "$R/std_cc2_5_none.jsonl")" > "$R/std_cc2_5_none.txt" 2>&1
[ -s "$R/std_cc2_5_g50.jsonl" ] || "$CPUPY" bot/tools/cc2_play.py --games 16 --pieces 500 --seed 33001 --think-ms 5 --jobs 8 $G50 \
    --out "$(wpath "$R/std_cc2_5_g50.jsonl")" > "$R/std_cc2_5_g50.txt" 2>&1
"$CPUPY" attentris/vs_score.py --pieces 500 \
    "cem d4b16 none=$RW/std_d4b16_none.jsonl" "cem d5b12 none=$RW/std_d5b12_none.jsonl" "cem d6b64 none=$RW/std_d6b64_none.jsonl" \
    "cc2 5ms none=$RW/std_cc2_5_none.jsonl" \
    "cem d4b16 50/50=$RW/std_d4b16_g50.jsonl" "cem d5b12 50/50=$RW/std_d5b12_g50.jsonl" "cem d6b64 50/50=$RW/std_d6b64_g50.jsonl" \
    "cc2 5ms 50/50=$RW/std_cc2_5_g50.jsonl"
grep -h -o 'mean [0-9.]* ms' "$R"/std_d*_none.txt
