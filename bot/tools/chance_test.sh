#!/bin/sh
# What does the chance ply bring? Our CEM search at depth 6 has 6 max plies over the known pieces (current +
# 5 previews), then one expectimax ply over the piece after the preview on its 12 best leaves; --no-chance
# drops that ply. Same weights, depth 6 beam 64, Cold Clear's rules (coldclear table, no 180).
# Solo games only: 64 games x 500 pieces, seeds 33001+, no garbage / 80-20 / 50-50 cheese, normal.txt and
# CEM-survival; 8 games at a time (one physical P-core each) so the ms per move are comparable.
# Resumable.   sh bot/tools/chance_test.sh > bot/results/chance_solo/chance.out 2>&1
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR/../../attentris"; . "$ATT/plat.sh"
ROOT="$DIR/../.."
BOT="$ROOT/bot/target/release/bot$EXE"
PY="$CPUPY"
VS="$ROOT/attentris/vs_score.py"
R="$ROOT/bot/results/chance_solo"
WR=$(wpath "$R")
OUT="$R/chance.txt"
mkdir -p "$R"
echo "== start $(date '+%H:%M')"
args=""
for w in normal survive; do
    case $w in normal) WF=$(wpath "$ROOT/bot/weights/normal.txt");; survive) WF=$(wpath "$ROOT/bot/results/survive/survive.txt");; esac
    for g in none g20 g50; do
        case $g in none) ga="";; g20) ga="--garbage 10,3-4,cheese0.2,0.2";; g50) ga="--garbage 10,3-4,cheese0.5,0.2";; esac
        for c in chance nochance; do
            nc=""; [ $c = nochance ] && nc="--no-chance"
            f="$WR/${w}_${c}_$g.jsonl"
            [ -f "$f" ] || "$BOT" play --mode coldclear --no-180 $nc --weights "$WF" --depth 6 --beam 64 --games 64 --pieces 500 --seed 33001 \
                --threads 8 --label ${w}_$c $ga --game-log "$f.part" > "$R/${w}_${c}_$g.log" 2>&1 && mv "$f.part" "$f"
            args="$args ${w}_${c}_$g=$f"
        done
    done
done
echo "== solo: chance ply on / off (coldclear rules, no 180, d6 b64; ms/pc in the logs)" | tee -a "$OUT"
"$PY" "$VS" $args | tee -a "$OUT"
for f in "$R"/*_none.log "$R"/*_g20.log "$R"/*_g50.log; do echo "$(basename "$f" .log): $(grep -A1 '^config' "$f" | tail -1 | awk '{print "ms/pc", $(NF-1)}')"; done | tee -a "$OUT"
echo "chance test done $(date '+%H:%M')"
