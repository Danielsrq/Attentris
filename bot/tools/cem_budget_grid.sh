#!/bin/sh
# Think time vs quality of CEM-survival over depth x beam (no garbage and 50/50 garbage), 16 games x 500 pieces,
# 8 games at a time (one per physical P-core, single-threaded search), so ms/move is clean.
#   sh bot/tools/cem_budget_grid.sh            results in bot/results/survive/budget/
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR/../../attentris"; . "$ATT/plat.sh"
ROOT=$(cd "$DIR/../.." && pwd)
R="$ROOT/bot/results/survive/budget"
mkdir -p "$R"
BOT="$ROOT/bot/target/release/bot$EXE"
W="$(wpath "$ROOT/bot/results/survive/survive.txt")"
for cfg in ${CFGS:-1x1 2x4 2x8 3x4 3x8 3x16 4x4 4x8 4x16 6x4 6x8 6x16}; do
    d=${cfg%x*}; b=${cfg#*x}
    for g in none g50; do
        out="$R/d${d}b${b}_$g.txt"
        [ -s "$out" ] && continue
        if [ "$g" = none ]; then
            "$BOT" play --weights "$W" --depth "$d" --beam "$b" --games 16 --pieces 500 --seed 1 --threads 8 > "$out" 2>&1
        else
            "$BOT" play --weights "$W" --depth "$d" --beam "$b" --games 16 --pieces 500 --seed 33001 --threads 8 \
                --garbage 10,3-4,cheese0.5,0.2 > "$out" 2>&1
        fi
    done
    echo "d$d b$b  none: $(grep -E '^beam' "$R/d${d}b${b}_none.txt" | awk '{print "APP", $2, "died", $3}')  $(grep -o 'mean [0-9.]* ms' "$R/d${d}b${b}_none.txt")" \
         " | 50/50: $(grep -E '^beam' "$R/d${d}b${b}_g50.txt" | awk '{print "APP", $2, "died", $3}')"
done
